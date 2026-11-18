"""Steps-to-detection: how early do the detectors flag the right cause?

Replays each failing run's telemetry in growing prefixes (every --every steps) and records the first step at
which the top detector finding names the injected cause *and keeps naming it* until the end of the run (so a
lucky early guess that later flips doesn't count).

    python -m eval.detection_latency --split dev
    python -m eval.detection_latency --split test
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from trainscope_server.db import open_store
from trainscope_server.detectors import RunData, run_detectors

from .harness import load_truth

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "eval"


def top_cause(store: Any, run_id: str, upto: int, cache: dict[str, Any]) -> str:
    if run_id not in cache:
        run = store.get_run(run_id) or {}
        cache[run_id] = (store.metrics_df(run_id), run.get("config"), store.events(run_id))
    df, cfg, events = cache[run_id]
    fs = run_detectors(RunData(df, cfg, events, upto_step=upto))
    return (fs[0].suspected_causes or ["healthy"])[0] if fs else "healthy"


def latency(store: Any, truth: dict[str, Any], every: int) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    last = store.max_step(truth["run_id"])
    steps = list(range(every - 1, last, every)) + [last]
    tops = [top_cause(store, truth["run_id"], s, cache) for s in steps]
    hit = None
    for i in range(len(steps)):
        if all(t == truth["cause"] for t in tops[i:]):
            hit = steps[i]
            break
    return {
        "run_id": truth["run_id"],
        "cause": truth["cause"],
        "task": truth["task"],
        "last_step": last,
        "detected_at": hit,
        "fraction": (hit + 1) / (last + 1) if hit is not None else None,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--every", type=int, default=10, help="replay granularity in steps")
    p.add_argument("--db", default=os.environ.get("TRAINSCOPE_DB", str(ROOT / "results" / "bench.duckdb")))
    a = p.parse_args(argv)
    store = open_store(a.db)
    truths = [t for t in load_truth(a.split) if t["cause"] != "healthy"]
    rows = [latency(store, t, a.every) for t in sorted(truths, key=lambda t: t["run_id"])]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"latency-{a.split}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["cause"]].append(r)
    print(f"{'cause':24} {'n':>3} {'detected':>9} {'median step':>12} {'median % of run':>16}")
    for cause, rs in sorted(by.items()):
        hits = [r for r in rs if r["detected_at"] is not None]
        med_step = statistics.median(r["detected_at"] for r in hits) if hits else None
        med_frac = statistics.median(r["fraction"] for r in hits) if hits else None
        print(
            f"{cause:24} {len(rs):3d} {len(hits):4d}/{len(rs):<4d} "
            f"{med_step if med_step is not None else '-':>12} "
            f"{f'{100 * med_frac:.0f}%' if med_frac is not None else '-':>16}"
        )
    hits = [r for r in rows if r["detected_at"] is not None]
    if hits:
        print(
            f"\noverall: {len(hits)}/{len(rows)} failing runs detected; median at "
            f"{100 * statistics.median(r['fraction'] for r in hits):.0f}% of the run "
            f"(step {statistics.median(r['detected_at'] for r in hits):.0f}). Details: {out}"
        )


if __name__ == "__main__":
    main()
