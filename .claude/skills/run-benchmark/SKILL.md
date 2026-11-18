---
name: run-benchmark
description: Generate TrainScope benchmark runs, measure detector quality and SDK overhead, and save results. Use when asked to run the benchmark, regenerate runs, or measure detectors/overhead.
---

# Running the benchmark

All commands run from the repo root with `.venv/bin/python`.

1. **Dev set** (iterate on this):
   `python -m benchmark.run_scenarios --split dev --seeds 0 1 2 --workers 4 --skip-existing`
   It writes telemetry to `results/bench.duckdb`, stdout logs to `results/runs/<id>/`, and ground truth to `benchmark/ground_truth/dev.jsonl`. On CPU each run takes ~1–3 min; tell the user the expected wall time before starting a big batch.
2. **Detector quality:** `python -m eval.harness --split dev --conditions E --tag dev-detectors`. This needs no API key. Report per-cause accuracy and false alarms on `healthy` runs.
3. **Leakage:** `python -m benchmark.leakage_check`.
4. **SDK overhead:** `python -m benchmark.overhead --steps 300 --repeats 3`. Run it with nothing else using the CPU; it writes `results/overhead.json`.
5. **Test set:** generate it only once tuning is finished:
   `python -m benchmark.run_scenarios --split test --workers 4`
   Never edit `benchmark/ground_truth/test.jsonl` by hand; a hook blocks it.
6. Put the headline numbers in the local `REPORT.md`. It is deliberately untracked, so never commit it, and never commit the DuckDB files either.
