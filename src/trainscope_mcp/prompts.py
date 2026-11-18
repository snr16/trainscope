"""MCP prompts guiding a structured investigation."""

from __future__ import annotations

from typing import Any

from trainscope_server.taxonomy import CAUSES

DIAGNOSE = """You are debugging the training run `{run_id}` with the TrainScope tools.

Investigate like an experienced ML engineer:
1. `get_run_summary` — status, config, current metrics, open findings.
2. `get_findings` — detector findings are strong evidence, but verify them.
3. `get_metric` on loss / eval metrics / lr / grad/global_norm to see *when* things changed.
4. `get_layer_stats` around the problem step to localize it to a layer.
5. `get_system_metrics` and `inspect_data` to rule in/out pipeline and data problems.
6. Use `compare_runs` / `diff_configs` if a healthy reference run exists.

Then answer with:
- **Diagnosis**: the single most likely root cause, chosen from: {causes}
- **Evidence**: the specific numbers and steps that support it.
- **Location**: the layer, data component, or config field responsible.
- **Fix**: a concrete fix, plus a JSON object `{{"type": "config_change", "field": ..., "value": ...}}` or
  `{{"type": "code_fix", "name": <cause_id>}}`.
- **Confidence** (low/medium/high) and what would change your mind.
"""

COMPARE = """Compare TrainScope runs `{run_a}` (reference) and `{run_b}`.

1. `diff_configs` to list hyperparameter and environment differences.
2. `compare_runs` to find where the curves diverge.
3. For the divergence step, inspect `get_layer_stats` and `get_findings` on both runs.

Explain which difference most plausibly causes the change in behaviour, with evidence.
"""


def register(mcp: Any) -> None:
    @mcp.prompt(name="diagnose_run", description="Structured investigation of one training run")
    def diagnose_run(run_id: str) -> str:
        return DIAGNOSE.format(run_id=run_id, causes=", ".join(CAUSES))

    @mcp.prompt(name="compare_runs", description="Explain behavioural differences between two runs")
    def compare_runs(run_a: str, run_b: str) -> str:
        return COMPARE.format(run_a=run_a, run_b=run_b)
