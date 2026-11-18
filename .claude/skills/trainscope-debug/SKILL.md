---
name: trainscope-debug
description: Investigate a training run with the TrainScope MCP tools like an ML engineer (summary, findings, curves, layers, system, data). Use when asked why a training run is failing, diverging, slow, or not learning.
---

# Debugging a run with TrainScope

The MCP server is `trainscope-mcp`. Point it at the store with `--db` or `TRAINSCOPE_DB`. If its tools aren't connected, the same functions are available in Python through `trainscope_mcp.tools.TrainScopeTools(open_store(db))`.

1. `list_runs`, then `get_run_summary(run_id)`. Note the config, progress, current metrics and open findings.
2. `get_findings(run_id)`. Treat these as strong leads, not verdicts, and verify each one against the raw curves.
3. `get_metric` on `loss`, `eval/loss`, `eval/accuracy`, `lr` and `grad/global_norm`. Use `step_start` / `step_end` to zoom in on change points.
4. `get_layer_stats(run_id, step)` at the problem step. Look for:
   - outlier gradient norms;
   - layers with `weight/update_ratio` = 0 (frozen);
   - a high `act/dead_frac`.
5. `get_system_metrics`: a high `data_wait_frac` means the input pipeline is the bottleneck; steadily growing memory means a leak.
6. `inspect_data`: label distribution, padding ratio, sample ids.
7. If a healthy reference run exists, use `compare_runs` and `diff_configs`.

**Answer** with the root cause, the evidence (specific steps and numbers), the location (layer, data component or config field), and a concrete fix.

**Live runs:**
- `request_action(run_id, "checkpoint")` runs immediately.
- `pause`, `resume` and `set_lr` need a human to run `trainscope approve <id>`. Say so; never approve on the user's behalf.
