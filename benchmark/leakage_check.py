"""Check that no scenario information leaks into run names, configs, events or stdout logs.

python -m benchmark.leakage_check --db results/bench.duckdb
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from trainscope_server.db import open_store

from .scenarios import load_catalog

ROOT = Path(__file__).resolve().parent.parent


def forbidden_terms() -> set[str]:
    cat = load_catalog()
    terms = set()
    for sid, sc in cat.items():
        if sid == "healthy":
            continue
        terms |= {sid, sc["cause"]}
        for spec in sc["severities"].values():
            terms |= set(spec.get("bugs") or [])
    terms |= {"inject", "scenario", "ground_truth", "bug"}
    return {t.lower() for t in terms}


def scan(db: str) -> list[str]:
    store = open_store(db)
    terms = forbidden_terms()
    pat = re.compile("|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)))
    problems = []
    for r in store.list_runs():
        blob = json.dumps({"name": r["name"], "config": r["config"], "env": r["env"]}).lower()
        for m in set(pat.findall(blob)):
            problems.append(f"run {r['id']}: '{m}' in name/config/env")
        for e in store.events(r["id"]):
            for m in set(pat.findall(json.dumps(e).lower())):
                problems.append(f"run {r['id']}: '{m}' in event {e['type']}")
        log = ROOT / "results" / "runs" / r["id"] / "stdout.log"
        if log.exists():
            for m in set(pat.findall(log.read_text().lower())):
                problems.append(f"run {r['id']}: '{m}' in stdout log")
    return problems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(ROOT / "results" / "bench.duckdb"))
    a = p.parse_args()
    problems = scan(a.db)
    for pr in problems:
        print(pr)
    print(f"{len(problems)} leakage problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
