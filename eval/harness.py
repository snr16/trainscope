"""Evaluation harness: run conditions A–E over benchmark runs and grade the diagnoses.

    python -m eval.harness --split dev --conditions E                       # no API key needed
    python -m eval.harness --split dev --conditions A B C D            # model(s) from eval/config.yaml
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

from . import llm_clients
from .conditions import CONDITIONS, csv_context, logs_context
from .grading import grade
from .settings import CONFIG_PATH, load_config

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


# where each cause is usually fixed; detectors-only (E) uses this as its "location"
CANONICAL_LOCATION: dict[str, str] = {
    "lr_too_high": "config: lr (learning rate)",
    "missing_warmup": "lr scheduler warmup",
    "scheduler_misstep": "lr scheduler stepping (T_max / step frequency)",
    "fp16_no_loss_scaling": "precision / loss scaling",
    "nan_from_op": "model forward: division / normalization without eps",
    "exploding_no_clip": "config: grad_clip",
    "frozen_layers": "requires_grad on backbone layers",
    "double_softmax": "model head: softmax before loss",
    "bad_init": "weight init",
    "shuffled_labels": "data labels",
    "val_leakage": "train/validation split",
    "missing_normalization": "input preprocessing / normalization",
    "bad_padding_mask": "attention / padding mask",
    "missing_zero_grad": "training loop: optimizer.zero_grad()",
    "grad_accum_unscaled": "training loop: accumulation",
    "eval_mode_training": "training loop: model.train() / eval mode",
    "overfitting": "train_size / regularization",
    "dataloader_bottleneck": "data loader workers / transforms",
    "memory_leak": "training loop: history keeps graph (.item())",
    "healthy": "",
}


def sample_per_cause(truths: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Deterministic stratified sample: the first k runs (by run id) of every cause."""
    by_cause: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in sorted(truths, key=lambda t: t["run_id"]):
        by_cause[t["cause"]].append(t)
    return [t for runs in by_cause.values() for t in runs[:k]]


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
    return {
        "cause": cause,
        "location": CANONICAL_LOCATION.get(cause, top.detector),
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
        unpriced = [m for m in args.models if m not in llm_clients.PRICES]
        if unpriced:
            print(f"note: no price for {', '.join(unpriced)} in the config; their cost is reported as $0")
    tag = args.tag or f"{args.split}-{'-'.join(args.conditions)}-{int(time.time())}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{tag}.jsonl"
    sem = asyncio.Semaphore(args.concurrency)
    # LLM conditions cost money: on the dev split they only need a small sample to sanity-check the prompt and
    # tools (dev is for tuning detectors, which is free). The test split always runs in full.
    llm_truths = truths
    if args.sample_per_cause:  # explicit, any split (e.g. a repeat subset for variance)
        llm_truths = sample_per_cause(truths, args.sample_per_cause)
    elif args.split == "dev" and args.dev_sample_per_cause > 0 and not args.limit:
        llm_truths = sample_per_cause(truths, args.dev_sample_per_cause)
    jobs = []
    for cond in args.conditions:
        uses_llm = CONDITIONS[cond].uses_llm
        models = args.models if uses_llm else ["detectors"]
        for model in models:
            for rep in range(args.repeats if uses_llm else 1):
                for t in llm_truths if uses_llm else truths:
                    jobs.append((cond, model, rep, t))
    n_llm = sum(CONDITIONS[c].uses_llm for c, *_ in jobs)
    print(f"{len(jobs)} diagnoses ({n_llm} LLM) -> {out_path}")
    if n_llm and llm_truths is not truths:
        print(
            f"{args.split} split: LLM conditions use {len(llm_truths)} of {len(truths)} runs "
            f"({args.sample_per_cause or args.dev_sample_per_cause} per cause)"
        )

    async def one(cond: str, model: str, rep: int, t: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            tr: dict[str, Any]
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
    print(f"API spend: ${sum(r['transcript'].get('cost_usd', 0.0) for r in results):.4f}")
    errors = [r for r in results if r["transcript"].get("error")]
    if errors:
        # API failures are not wrong answers: report them and leave them out of the accuracy table
        print(
            f"\n{len(errors)} diagnosis/es failed with an API error and are excluded from the scores, e.g.:"
        )
        for msg in sorted({r["transcript"]["error"][:200] for r in errors})[:3]:
            print("  -", msg)
        results = [r for r in results if not r["transcript"].get("error")]
    if results:
        print_summary(results)


def print_summary(results: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[(r["condition"], r["model"])].append(r)
    print(
        f"\n{'cond':4} {'model':20} {'n':>4} {'cause':>6} {'categ':>6} {'loc':>6} {'fix':>6} {'calls':>6} "
        f"{'$/diag':>8} {'cached':>7}"
    )
    for (c, m), rs in sorted(groups.items()):
        n = len(rs)

        def pct(key: str, rs: list[dict[str, Any]] = rs, n: int = n) -> str:
            return f"{100 * sum(r[key] for r in rs) / n:5.1f}%"

        calls = sum(len(r["transcript"].get("tool_calls", [])) for r in rs) / n
        cost = sum(r["transcript"].get("cost_usd", 0) for r in rs) / n
        read = sum(r["transcript"].get("cache_read_tokens", 0) for r in rs)
        total_in = read + sum(
            r["transcript"].get("input_tokens", 0) + r["transcript"].get("cache_write_tokens", 0) for r in rs
        )
        cached = f"{100 * read / total_in:6.1f}%" if total_in else "     —"
        print(
            f"{c:4} {m:20} {n:4d} {pct('cause_ok')} {pct('category_ok')} {pct('loc_ok')} "
            f"{pct('fix_static_ok')} {calls:6.1f} {cost:8.4f} {cached}"
        )
    by_cause: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        by_cause[(r["condition"], r["cause"])].append(r["cause_ok"])
    print("\nper-cause accuracy:")
    for (c, cause), oks in sorted(by_cause.items()):
        print(f"  {c} {cause:24} {sum(oks)}/{len(oks)}")


def main(argv: list[str] | None = None) -> None:
    # read --config first so the chosen file supplies the defaults for model / effort / limits
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(CONFIG_PATH))
    known, _ = pre.parse_known_args(argv)
    cfg = load_config(Path(known.config))
    llm_clients.configure(cfg)  # prices and fallback models from the chosen config

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--conditions", nargs="+", default=["E"], choices=list(CONDITIONS))
    p.add_argument("--models", nargs="+", default=cfg["models"], help="override `models` from the config")
    p.add_argument("--effort", default=cfg["effort"], choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-tool-calls", type=int, default=cfg["max_tool_calls"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=cfg["concurrency"])
    p.add_argument("--dev-sample-per-cause", type=int, default=cfg["dev_sample_per_cause"])
    p.add_argument("--sample-per-cause", type=int, help="LLM conditions use N runs per cause on any split")
    p.add_argument("--limit", type=int)
    p.add_argument("--scenarios", nargs="*")
    p.add_argument("--hide-signals", nargs="*", choices=["gradients", "activations", "system", "data"])
    p.add_argument("--db", default=os.environ.get("TRAINSCOPE_DB", str(ROOT / "results" / "bench.duckdb")))
    p.add_argument("--tag")
    args = p.parse_args(argv)
    if any(CONDITIONS[c].uses_llm for c in args.conditions) and not args.models:
        raise SystemExit(f"no model configured: set `models` in {args.config}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
