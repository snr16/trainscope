"""TrainScope MCP server (MCP Python SDK v2, ``MCPServer``).

trainscope-mcp                       # stdio (Claude Desktop / Cursor / Claude Code)
trainscope-mcp --http --port 8766    # streamable HTTP; set TRAINSCOPE_MCP_TOKEN to require a bearer token
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from trainscope_server.db import Storage, open_store

from . import prompts, resources
from .tools import TrainScopeTools

INSTRUCTIONS = (
    "TrainScope exposes the internals of ML training runs: scalar curves, per-layer gradient/weight/activation "
    "statistics, system and data-pipeline metrics, automated detector findings, and safe control actions. "
    "Start with list_runs, then get_run_summary and get_findings. Tool outputs are already summarized; prefer "
    "step ranges over requesting whole series."
)


def build_server(
    store: Storage | None = None,
    include_findings: bool = True,
    include_actions: bool = True,
    hidden_signals: set[str] | None = None,
    name: str = "trainscope",
) -> MCPServer:
    store = store or open_store(os.environ.get("TRAINSCOPE_DB"))
    t = TrainScopeTools(store, hidden_signals=hidden_signals)
    mcp = MCPServer(name, instructions=INSTRUCTIONS, version="0.1.0")

    @mcp.tool(description="List training runs with status, last step and loss.")
    def list_runs(
        project: str | None = None, status: str | None = None, limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        return t.list_runs(project, status, limit, offset)

    @mcp.tool(
        description="Config highlights, progress, current metrics, available metric names"
        + (" and open detector findings" if include_findings else "")
        + " for one run."
    )
    def get_run_summary(run_id: str) -> dict[str, Any]:
        return t.get_run_summary(run_id, include_findings=include_findings)

    @mcp.tool(
        description="Downsampled series plus statistics (trend, min/max, change points, NaN count) for a "
        "metric. Per-layer metrics (grad/norm, weight/norm, weight/update_ratio, act/*) need layer=."
    )
    def get_metric(
        run_id: str,
        name: str,
        layer: str = "",
        step_start: int | None = None,
        step_end: int | None = None,
        max_points: int = 60,
    ) -> dict[str, Any]:
        return t.get_metric(run_id, name, layer, step_start, step_end, max_points)

    @mcp.tool(
        description="Per-layer gradient/weight/activation stats at a step (default: latest sampled), "
        "highlighting outlier layers."
    )
    def get_layer_stats(run_id: str, step: int | None = None) -> dict[str, Any]:
        return t.get_layer_stats(run_id, step)

    if include_findings:

        @mcp.tool(
            description="Automated detector findings (severity, step range, summary, evidence, "
            "suspected causes)."
        )
        def get_findings(run_id: str) -> dict[str, Any]:
            return t.get_findings(run_id)

    @mcp.tool(description="Step time, data-loader wait fraction, GPU utilization and memory trends.")
    def get_system_metrics(run_id: str, last_n_steps: int | None = None) -> dict[str, Any]:
        return t.get_system_metrics(run_id, last_n_steps)

    @mcp.tool(description="Label distribution, padding ratio and sample ids over time, plus dataset config.")
    def inspect_data(run_id: str) -> dict[str, Any]:
        return t.inspect_data(run_id)

    @mcp.tool(description="Side-by-side metrics for two runs and the step where curves diverge.")
    def compare_runs(run_a: str, run_b: str, metrics: list[str] | None = None) -> dict[str, Any]:
        return t.compare_runs(run_a, run_b, metrics)

    @mcp.tool(description="Hyperparameter and environment differences between two runs.")
    def diff_configs(run_a: str, run_b: str) -> dict[str, Any]:
        return t.diff_configs(run_a, run_b)

    if include_actions:

        @mcp.tool(
            description="Queue an action on a live run: checkpoint (runs immediately), pause / resume / "
            "set_lr (args={'lr': x}) which require human approval."
        )
        def request_action(run_id: str, action: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
            return t.request_action(run_id, action, args, requested_by="assistant:mcp")

    resources.register(mcp, t)
    prompts.register(mcp)
    return mcp


def main() -> None:
    p = argparse.ArgumentParser("trainscope-mcp")
    p.add_argument("--db", default=os.environ.get("TRAINSCOPE_DB", "trainscope.duckdb"))
    p.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    a = p.parse_args()
    mcp = build_server(open_store(a.db))
    if not a.http:
        mcp.run("stdio")
        return
    import uvicorn

    app = mcp.streamable_http_app()
    token = os.environ.get("TRAINSCOPE_MCP_TOKEN")
    if token:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse

        async def check(request, call_next):
            if request.headers.get("authorization") != f"Bearer {token}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=check)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
