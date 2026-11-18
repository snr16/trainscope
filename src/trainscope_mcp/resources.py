"""MCP resources: run summaries and configs."""

from __future__ import annotations

import json
from typing import Any

from .tools import TrainScopeTools


def register(mcp: Any, tools: TrainScopeTools) -> None:
    @mcp.resource(
        "trainscope://runs",
        name="runs",
        mime_type="application/json",
        description="All runs with status and key metrics",
    )
    def runs() -> str:
        return json.dumps(tools.list_runs(limit=100), default=str)

    @mcp.resource(
        "trainscope://runs/{run_id}/summary",
        name="run_summary",
        mime_type="application/json",
        description="Compact summary of one run",
    )
    def run_summary(run_id: str) -> str:
        return json.dumps(tools.get_run_summary(run_id), default=str)

    @mcp.resource(
        "trainscope://runs/{run_id}/config",
        name="run_config",
        mime_type="application/json",
        description="Hyperparameters and environment of one run",
    )
    def run_config(run_id: str) -> str:
        r = tools._run(run_id)
        return json.dumps(
            {"config": tools._public_config(r.get("config") or {}), "env": r.get("env")}, default=str
        )
