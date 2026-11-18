---
name: add-detector
description: Add a new TrainScope failure detector (module, registration, unit test, docs). Use when asked to detect a new training failure pattern or tune an existing detector.
---

# Adding a detector

Detectors live in `src/trainscope_server/detectors/`. Each one reads a `RunData` view and returns `Finding`s.

1. **Pick the signals.** Check what the SDK records (`src/trainscope/run.py`, `hooks.py`): scalars such as `loss`, `lr`, `step_time`, `data_wait_frac`, `eval/*`, `train/accuracy`, `sys/*`, `amp/*`, `data/*`, `model/train_mode`, and per-layer metrics such as `grad/norm`, `grad/zero_frac`, `weight/norm`, `weight/update_ratio`, `weight/trainable`, `act/mean|std|dead_frac`. If you need a new signal, add it to the SDK first and keep it cheap. It must not add GPU syncs on non-sampled steps.
2. **Write the module** `detectors/<family>.py`:
   - Subclass `Detector`, set `name`, and keep thresholds as `__init__` args.
   - Use `data.scalar(name)`, `data.layers(name)`, `data.events_of(type)`, `data.cfg(...)`, and the helpers `smooth`, `robust_z`, `trend`, `finite`.
   - Return `Finding(detector, severity, step_start, step_end, summary, evidence, suspected_causes, confidence)`.
   - `summary` is one line with the key numbers. `evidence` holds the raw numbers. `suspected_causes` are ids from `trainscope_server/taxonomy.py`.
   - Never raise on short or empty runs; return `[]`.
3. **Register it** in `DETECTORS` in `detectors/engine.py`.
4. **Test it** in `tests/test_detectors.py` with synthetic series from `tests/helpers.py`:
   - one positive case;
   - a check that `test_healthy_run_has_no_findings` still passes, which guards against false alarms.
5. **Tune it on the dev set only.** Run `python -m eval.harness --split dev --conditions E` and look at the per-cause accuracy. Never look at test-split results while tuning.
6. Add a line to the detector table in `README.md`.
