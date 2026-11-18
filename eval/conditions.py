"""Evaluation conditions: what each assistant gets to see.

A  logs only        the training script's printed stdout
B  raw metrics      CSV export of logged scalars + run config
C  tools            TrainScope MCP tools, without detector findings
D  full TrainScope  all MCP tools including get_findings
E  detectors only   top detector finding as the diagnosis (no LLM)
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Condition:
    key: str
    name: str
    uses_llm: bool
    uses_tools: bool
    include_findings: bool = False


CONDITIONS = {
    "A": Condition("A", "logs_only", True, False),
    "B": Condition("B", "raw_metrics", True, False),
    "C": Condition("C", "tools_no_detectors", True, True, include_findings=False),
    "D": Condition("D", "full_trainscope", True, True, include_findings=True),
    "E": Condition("E", "detectors_only", False, False),
}


def logs_context(run_id: str, max_lines: int = 400) -> str:
    p = ROOT / "results" / "runs" / run_id / "stdout.log"
    if not p.exists():
        return "(no stdout captured)"
    lines = p.read_text().splitlines()
    if len(lines) > max_lines:
        head, tail = max_lines // 4, 3 * max_lines // 4
        lines = lines[:head] + [f"... ({len(lines) - max_lines} lines omitted) ..."] + lines[-tail:]
    return "\n".join(lines)


def csv_context(store: Any, run_id: str, max_rows: int = 400) -> str:
    df = store.metrics_df(run_id)
    df = df[df["layer"] == ""]
    wide = df.pivot_table(index="step", columns="name", values="value", aggfunc="last").sort_index()
    if len(wide) > max_rows:
        stride = max(1, len(wide) // max_rows)
        keep = wide.index[::stride]
        eval_cols = [c for c in wide.columns if c.startswith("eval/")]
        if eval_cols:  # never drop the (sparse) eval rows
            keep = keep.union(wide.dropna(subset=eval_cols, how="all").index)
        wide = wide.loc[keep]
    buf = io.StringIO()
    wide.to_csv(buf, float_format="%.5g")
    run = store.get_run(run_id) or {}
    cfg = run.get("config") or {}
    return f"Run config:\n{json.dumps(cfg, indent=1, default=str)}\n\nLogged scalars (CSV):\n{buf.getvalue()}"
