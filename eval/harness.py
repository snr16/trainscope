"""Evaluation harness: run conditions A–E over benchmark runs and grade the diagnoses.

    python -m eval.harness --split dev --conditions E                       # no API key needed
    python -m eval.harness --split dev --conditions A B C D --models claude-opus-5-5 claude-sonnet-5-5
    python -m eval.harness --split dev --conditions D --hide-signals gradients      # signal ablation
    python -m eval.harness --split test --conditions A B C D E --repeats 2          # final numbers

Needs ANTHROPIC_API_KEY (environment or a .env file in the repo root) for the LLM conditions.
Results: results/eval/<tag>.jsonl (one line per diagnosis, with every tool call) + a summary table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from trainscope_server.db import open_store
from trainscope_server.detectors import RunData, run_detectors

from .conditions import CONDITIONS, csv_context, logs_context
from .grading import grade

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = ROOT / "benchmark" / "ground_truth"
OUT_DIR = ROOT / "results" / "eval"

QUESTION = "Diagnose this training run (run_id: {run_id})."

# canonical fix the detectors-only baseline proposes for a cause
CANONICAL_FIX: dict[str, dict[str, Any]] = {
    "lr_too_high": {"type": "config_change", "field": "lr", "value": "default"},
    "exploding_no_clip": {"type": "config_change", "field": "grad_clip", "value": 1.0},
    "fp16_no_loss_scaling": {"type": "config_change", "field": "precision", "value": "fp32"},
    "overfitting": {"type": "config_change", "field": "weight_decay", "value": 0.05},
    "healthy": {"type": "none"},
}


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_truth(split: str) -> list[dict[str, Any]]:
    p = GT_DIR / f"{split}.jsonl"
    if not p.exists():
        raise SystemExit(f"{p} not found; run python -m benchmark.run_scenarios --split {split} first")
    rows = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok"):
                rows[r["run_id"]] = r  # last write wins
    return list(rows.values())


def detectors_only(store: Any, run_id: str) -> dict[str, Any]:
    fs = run_detectors(RunData.from_store(store, run_id))
    if not fs:
        return {
            "cause": "healthy",
            "location": "",
            "evidence": "no findings",
            "fix": {"type": "none"},
            "confidence": "medium",
        }
    top = fs[0]
    cause = (top.suspected_causes or ["healthy"])[0]
    loc = json.dumps(top.evidence)[:300]
    return {
        "cause": cause,
        "location": f"{top.detector}: {loc}",
        "evidence": top.summary,
        "fix": CANONICAL_FIX.get(cause, {"type": "code_fix", "name": cause}),
        "confidence": "medium",
    }


async def run_llm(
    cond_key: str, model: str, truth: dict[str, Any], store: Any, client: Any, args: Any
) -> dict:
    from mcp.client.client import Client

    from trainscope_mcp.server import build_server

    from .llm_clients import diagnose

    cond = CONDITIONS[cond_key]
    rid = truth["run_id"]
    if cond.key == "A":
        prompt = (
            QUESTION.format(run_id=rid) + "\n\nHere is the training script's stdout:\n\n" + logs_context(rid)
        )
        return (await diagnose(client, model, prompt, None, args.max_tool_calls, args.effort)).to_dict()
    if cond.key == "B":
        prompt = QUESTION.format(run_id=rid) + "\n\n" + csv_context(store, rid)
        return (await diagnose(client, model, prompt, None, args.max_tool_calls, args.effort)).to_dict()
    server = build_server(
        store,
        include_findings=cond.include_findings,
        include_actions=False,
        hidden_signals=set(args.hide_signals or []),
    )
    prompt = QUESTION.format(run_id=rid) + " Use the TrainScope tools to investigate."
    async with Client(server) as session:
        tr = await diagnose(client, model, prompt, session, args.max_tool_calls, args.effort)
    return tr.to_dict()


async def main_async(args: Any) -> None:
    load_env()
    truths = load_truth(args.split)
    if args.scenarios:
        truths = [t for t in truths if t["scenario"] in args.scenarios]
    truths = sorted(truths, key=lambda t: t["run_id"])[: args.limit]
    store = open_store(args.db)
    client = None
    if any(CONDITIONS[c].uses_llm for c in args.conditions):
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise SystemExit("LLM conditions need ANTHROPIC_API_KEY (environment or .env)")
        import anthropic

        client = anthropic.AsyncAnthropic(max_retries=4)
    tag = args.tag or f"{args.split}-{'-'.join(args.conditions)}-{int(time.time())}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{tag}.jsonl"
    sem = asyncio.Semaphore(args.concurrency)
    jobs = []
    for cond in args.conditions:
        models = ["detectors"] if not CONDITIONS[cond].uses_llm else args.models
        for model in models:
            for rep in range(args.repeats if CONDITIONS[cond].uses_llm else 1):
                for t in truths:
                    jobs.append((cond, model, rep, t))
    print(f"{len(jobs)} diagnoses over {len(truths)} runs -> {out_path}")

    async def one(cond: str, model: str, rep: int, t: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            if cond == "E":
                tr = {
                    "diagnosis": detectors_only(store, t["run_id"]),
                    "tool_calls": [],
                    "cost_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "wall_s": 0.0,
                }
            else:
                tr = await run_llm(cond, model, t, store, client, args)
        g = grade(tr.get("diagnosis"), t)
        return {
            "condition": cond,
            "model": model,
            "repeat": rep,
            "run_id": t["run_id"],
            "scenario": t["scenario"],
            "task": t["task"],
            "cause": t["cause"],
            "category": t["category"],
            "hidden_signals": args.hide_signals or [],
            **g,
            "transcript": tr,
        }

    results = []
    with open(out_path, "w") as f:
        for coro in asyncio.as_completed([one(*j) for j in jobs]):
            r = await coro
            results.append(r)
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
    print_summary(results)


def print_summary(results: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[(r["condition"], r["model"])].append(r)
    print(
        f"\n{'cond':4} {'model':20} {'n':>4} {'cause':>6} {'categ':>6} {'loc':>6} {'fix':>6} {'calls':>6} {'$/diag':>8}"
    )
    for (c, m), rs in sorted(groups.items()):
        n = len(rs)

        def pct(key: str, rs: list[dict[str, Any]] = rs, n: int = n) -> str:
            return f"{100 * sum(r[key] for r in rs) / n:5.1f}%"

        calls = sum(len(r["transcript"].get("tool_calls", [])) for r in rs) / n
        cost = sum(r["transcript"].get("cost_usd", 0) for r in rs) / n
        print(
            f"{c:4} {m:20} {n:4d} {pct('cause_ok')} {pct('category_ok')} {pct('loc_ok')} "
            f"{pct('fix_static_ok')} {calls:6.1f} {cost:8.4f}"
        )
    by_cause: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_cause[(r["condition"], r["cause"])].append(r["cause_ok"])
    print("\nper-cause accuracy:")
    for (c, cause), oks in sorted(by_cause.items()):
        print(f"  {c} {cause:24} {sum(oks)}/{len(oks)}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--conditions", nargs="+", default=["E"], choices=list(CONDITIONS))
    p.add_argument("--models", nargs="+", default=["claude-opus-5-5"])
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-tool-calls", type=int, default=12)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--scenarios", nargs="*")
    p.add_argument("--hide-signals", nargs="*", choices=["gradients", "activations", "system", "data"])
    p.add_argument("--db", default=os.environ.get("TRAINSCOPE_DB", str(ROOT / "results" / "bench.duckdb")))
    p.add_argument("--tag")
    asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    main()
