---
name: add-scenario
description: Add a failure scenario to the TrainScope benchmark (catalog entry, injection, ground truth, leakage check, fix). Use when asked to add a new training bug or failure to the benchmark.
---

# Adding a benchmark scenario

Scenarios live in `benchmark/scenarios/catalog.yaml`. Bugs are injected in `benchmark/tasks/trainer.py`.

1. **Cause label.** Reuse an id from `src/trainscope_server/taxonomy.py`, or add one there with its category.
2. **Injection.**
   - If the failure is a pure config value (e.g. a bad `lr`), put it in `cfg:` or use `lr_mult` / `lr_abs`. Configs are visible to assistants, so they must look like a plausible real config.
   - If the failure is code-level, add a bug name to `BUGS` in `trainer.py` and implement it at the right injection point (`_build`, `_forward`, `_loss`, `_lr_lambda`, or the loop). Bugs are **never** stored with the run.
3. **Catalog entry.** Fill in `cause`, `tasks`, `severities` (`cfg` / `bugs` / `lr_mult`), `component` (add keywords for it in `eval/grading.py: COMPONENT_KEYWORDS`), and `fixes`.
4. **Fix.**
   - Code fixes map to bugs in `benchmark/fixes/apply.py: CODE_FIXES`.
   - Config fixes must name a field that is in `FIELD_BOUNDS` or `CATEGORICAL`.
   - If the generic criteria in `recovered()` don't capture this cause, add a recovery rule there.
5. **No leakage.** Run names are opaque hashes; keep them that way. Never print scenario hints in stdout. After generating runs, run `python -m benchmark.leakage_check --db results/bench.duckdb`.
6. **Smoke-test it.** Run `python -m benchmark.run_scenarios --split dev --seeds 0 --scenarios <id> --workers 1`, then check that the run actually fails as intended. Then check that the canonical fix recovers it with `python -m eval.verify_fix`.
7. **Variants and the split.** These are automatic: the split is a stable hash of scenario/task/severity/seed. Don't hand-pick the split.
