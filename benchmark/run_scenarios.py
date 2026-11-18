"""Generate benchmark runs with ground-truth labels.

    python -m benchmark.run_scenarios --split dev --workers 4
    python -m benchmark.run_scenarios --split test --seeds 0 1 2       # held-out set: run once, at the end
    python -m benchmark.run_scenarios --scenarios healthy lr_too_high --tasks vision --limit 4

Writes telemetry to the DuckDB store, stdout logs to results/runs/<run_id>/stdout.log (condition A), and
ground truth to benchmark/ground_truth/<split>.jsonl (never visible to assistants).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = ROOT / "benchmark" / "ground_truth"
LOG_DIR = ROOT / "results" / "runs"


def _run_one(v: dict, db: str, max_steps: int | None) -> dict:
    import warnings

    warnings.filterwarnings("ignore")
    from benchmark.tasks.trainer import train

    log_dir = LOG_DIR / v["run_id"]
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stdout.log").unlink(missing_ok=True)
    # each run writes its own DuckDB file (no cross-process lock contention); the parent merges it
    run_db = log_dir / "telemetry.duckdb"
    run_db.unlink(missing_ok=True)
    t = time.time()
    try:
        res = train(
            v["task"],
            v["cfg"],
            set(v["bugs"]),
            seed=v["seed"],
            run_id=v["run_id"],
            name=f"exp-{v['run_id'][:6]}",
            db=str(run_db),
            spool_dir=str(log_dir / "spool"),
            stdout_log=str(log_dir / "stdout.log"),
            max_steps=max_steps,
        )
        return {**v, "result": asdict(res), "ok": True, "run_db": str(run_db)}
    except Exception:
        return {**v, "ok": False, "error": traceback.format_exc()[-2000:], "wall_s": time.time() - t}


def main(argv: list[str] | None = None) -> None:
    from benchmark.scenarios import expand
    from trainscope_server.db.duckdb_store import DuckDBStore

    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--seeds", type=int, nargs="+", help="default: all (0-2 hash-split, 3 test-only)")
    p.add_argument("--scenarios", nargs="*")
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--db", default=os.environ.get("TRAINSCOPE_DB", str(ROOT / "results" / "bench.duckdb")))
    p.add_argument("--max-steps", type=int)
    p.add_argument("--skip-existing", action="store_true")
    a = p.parse_args(argv)

    variants = expand(a.seeds, a.scenarios, a.tasks, a.split)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    gt_path = GT_DIR / f"{a.split}.jsonl"
    done = set()
    if a.skip_existing and gt_path.exists():
        done = {json.loads(line)["run_id"] for line in gt_path.read_text().splitlines() if line.strip()}
    todo = [v.to_dict() for v in variants if v.run_id not in done][: a.limit]
    print(f"{len(todo)} run(s) to generate for split={a.split} -> {a.db}")
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(gt_path, "a") as gt:
        futs = {ex.submit(_run_one, v, a.db, a.max_steps): v for v in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r["ok"]:
                DuckDBStore(a.db).merge_from(r.pop("run_db"))
            gt.write(json.dumps(r) + "\n")
            gt.flush()
            status = "ok" if r["ok"] else "ERROR"
            wall = r.get("result", {}).get("wall_s", r.get("wall_s", 0))
            print(f"[{i}/{len(todo)}] {status} {r['run_id']} ({wall:.0f}s)")
            if not r["ok"]:
                print(r["error"])
    print(f"done in {time.time() - t0:.0f}s; ground truth in {gt_path}")


if __name__ == "__main__":
    main()
