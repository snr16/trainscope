"""Fix verification: apply each proposed fix, rerun the scenario on a short budget, check recovery.

    python -m eval.verify_fix results/eval/<tag>.jsonl --workers 4

Reports the verified fix rate per condition/model (the share of diagnoses whose fix actually recovers the run).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _verify_one(job: dict[str, Any], db: str, max_steps: int | None) -> dict[str, Any]:
    import warnings

    warnings.filterwarnings("ignore")
    from benchmark.fixes.apply import FixError, apply_fix, recovered
    from benchmark.tasks.trainer import train

    truth, fix = job["truth"], job["fix"]
    if truth["cause"] == "healthy":
        ok = (fix or {}).get("type") in (None, "none")
        return {
            **job["key"],
            "verified": ok,
            "reason": "healthy run: no change expected" if ok else "changed a healthy run",
        }
    if not fix or fix.get("type") == "none":
        return {**job["key"], "verified": False, "reason": "no fix proposed"}
    try:
        cfg, bugs = apply_fix(truth["task"], truth["cfg"], truth["bugs"], fix)
    except FixError as e:
        return {**job["key"], "verified": False, "reason": f"invalid fix: {e}"}
    res = train(
        truth["task"], cfg, set(bugs), seed=truth["seed"] + 100, db=db, project="verify", max_steps=max_steps
    )
    ok, why = recovered(truth["task"], truth["cause"], res)
    return {**job["key"], "verified": ok, "reason": why, "rerun": asdict(res)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--conditions", nargs="*", help="only verify these conditions (default: all in the file)")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--db", default=str(ROOT / "results" / "verify.duckdb"))
    a = p.parse_args(argv)

    from eval.harness import load_truth

    rows = [json.loads(line) for line in a.results.read_text().splitlines() if line.strip()]
    rows = [
        r for r in rows if not (r.get("transcript") or {}).get("error")
    ]  # API failures have no fix to test
    if a.conditions:
        rows = [r for r in rows if r["condition"] in a.conditions]
    truths = {}
    for split in ("dev", "test"):
        try:
            truths.update({t["run_id"]: t for t in load_truth(split)})
        except SystemExit:
            pass
    jobs = []
    cache: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = (r.get("transcript") or {}).get("diagnosis") or {}
        t = truths[r["run_id"]]
        key = {
            "condition": r["condition"],
            "model": r["model"],
            "repeat": r["repeat"],
            "run_id": r["run_id"],
            "cause": r["cause"],
        }
        jobs.append({"key": key, "truth": t, "fix": d.get("fix")})
    out_path = a.results.with_suffix(".verify.jsonl")
    results = []
    with ProcessPoolExecutor(a.workers) as ex, open(out_path, "w") as f:
        # identical (run, fix) pairs are rerun once
        futs = {}
        for j in jobs:
            sig = json.dumps([j["truth"]["run_id"], j["fix"]], sort_keys=True)
            if sig not in cache:
                cache[sig] = {}
                futs[ex.submit(_verify_one, j, a.db, a.max_steps)] = sig
        done: dict[str, dict[str, Any]] = {}
        for fut in as_completed(futs):
            done[futs[fut]] = fut.result()
        for j in jobs:
            sig = json.dumps([j["truth"]["run_id"], j["fix"]], sort_keys=True)
            r = {**done[sig], **j["key"]}
            results.append(r)
            f.write(json.dumps(r, default=str) + "\n")
    agg: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        agg[(r["condition"], r["model"])].append(r["verified"])
    print(f"{'cond':4} {'model':20} {'n':>4} {'verified fix rate':>18}")
    for (c, m), v in sorted(agg.items()):
        print(f"{c:4} {m:20} {len(v):4d} {100 * sum(v) / len(v):17.1f}%")
    print(f"details: {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
