"""Read endpoints for the web dashboard, plus the dashboard's static files.

The dashboard is a single page served at ``/ui/``. Its data comes from these JSON routes, which reuse the same
summarizing tools as the MCP server, so a person and an assistant see the same numbers.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from trainscope_mcp.summarize import lttb
from trainscope_mcp.tools import TrainScopeTools

from .db import Storage
from .detectors.memory import memory_floor
from .ingest.auth import check_project

UI_DIR = Path(__file__).with_name("ui")
MAX_POINTS = 400


def _clean(v: float) -> float | str:
    """JSON has no NaN/Inf: send them as strings so the chart can mark them instead of dropping them."""
    f = float(v)
    return f if math.isfinite(f) else str(f)


def series_points(
    store: Storage, run_id: str, name: str, layer: str = "", max_points: int = MAX_POINTS
) -> dict:
    df = store.metrics_df(run_id, names=[name], layer=layer)
    if df.empty:
        return {"name": name, "layer": layer, "points": [], "nonfinite": []}
    s = df.groupby("step")["value"].last().sort_index()
    vals = s.to_numpy(dtype=float)
    finite = np.isfinite(vals)
    fs = s[finite]
    nonfinite = [int(x) for x in s.index[~finite][:50]]
    if len(fs) > max_points:
        idx = lttb(fs.index.to_numpy(dtype=float), fs.to_numpy(dtype=float), max_points)
        fs = fs.iloc[idx]
    return {
        "name": name,
        "layer": layer,
        "points": [[int(k), float(v)] for k, v in fs.items()],
        "nonfinite": nonfinite,
    }


class FindingsCache:
    """Detector results per run, computed by one background thread and refreshed when a run gets new data.
    Run lists read from here so they never wait on detectors."""

    def __init__(self, tools: TrainScopeTools):
        self.tools = tools
        self.data: dict[str, tuple[int, dict[str, Any]]] = {}  # run_id -> (last_step seen, badge)
        self.wanted: dict[str, int] = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        threading.Thread(target=self._loop, name="trainscope-findings", daemon=True).start()

    def get(self, run_id: str, last_step: int) -> dict[str, Any] | None:
        with self.lock:
            hit = self.data.get(run_id)
            if hit is None or hit[0] != last_step:
                self.wanted[run_id] = last_step
                self.wake.set()
            return hit[1] if hit else None

    def _loop(self) -> None:
        while True:
            self.wake.wait(5)
            self.wake.clear()
            while True:
                with self.lock:
                    if not self.wanted:
                        break
                    run_id, step = self.wanted.popitem()
                try:
                    fs = self.tools._findings(run_id)
                    badge = {
                        "critical": sum(f["severity"] == "critical" for f in fs),
                        "warning": sum(f["severity"] == "warning" for f in fs),
                        "top": fs[0]["summary"] if fs else None,
                        "cause": (fs[0]["evidence"].get("suspected_causes") or [None])[0] if fs else None,
                    }
                except Exception:
                    badge = {"critical": 0, "warning": 0, "top": None, "cause": None, "error": True}
                with self.lock:
                    self.data[run_id] = (step, badge)
                time.sleep(0)  # yield to request threads


def register(app: FastAPI, store: Storage, authz: Any, project_of: Callable[[str], str]) -> None:
    tools = TrainScopeTools(store, max_chars=200_000)
    cache = FindingsCache(tools)

    @app.get("/v1/overview")
    def overview(granted=authz) -> dict[str, Any]:
        progress = store.run_progress()
        runs, pending = [], 0
        for r in store.list_runs():
            if granted not in ("*", r["project"]):
                continue
            pr = progress.get(r["id"], {"last_step": -1, "last_loss": None})
            badge = cache.get(r["id"], pr["last_step"]) if pr["last_step"] >= 0 else None
            pending += badge is None and pr["last_step"] >= 0
            ll = pr["last_loss"]
            runs.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "project": r["project"],
                    "status": r["status"],
                    "started_at": r["started_at"],
                    "ended_at": r["ended_at"],
                    "last_step": pr["last_step"],
                    "total_steps": (r.get("config") or {}).get("total_steps"),
                    "last_loss": _clean(ll) if ll is not None else None,
                    "findings": badge,
                }
            )
        return {"runs": runs, "pending_findings": pending, "generated_at": time.time()}

    def guard(run_id: str, granted: str | None) -> None:
        check_project(granted, project_of(run_id))

    def call(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return fn()
        except ValueError as e:  # unknown run etc.
            raise HTTPException(404, str(e)) from e

    @app.get("/v1/runs/{run_id}/summary")
    def run_summary(run_id: str, granted=authz) -> dict[str, Any]:
        guard(run_id, granted)
        return call(lambda: tools.get_run_summary(run_id))

    @app.get("/v1/runs/{run_id}/series")
    def run_series(
        run_id: str,
        names: str = Query(..., description="comma-separated metric names"),
        layer: str = "",
        granted=authz,
    ) -> dict[str, Any]:
        guard(run_id, granted)
        wanted = [n for n in names.split(",") if n][:12]
        return {"run_id": run_id, "series": [series_points(store, run_id, n, layer) for n in wanted]}

    @app.get("/v1/runs/{run_id}/layers")
    def run_layers(run_id: str, step: int | None = None, granted=authz) -> dict[str, Any]:
        guard(run_id, granted)
        return call(lambda: tools.get_layer_stats(run_id, step, top=200))

    @app.get("/v1/runs/{run_id}/system")
    def run_system(run_id: str, granted=authz) -> dict[str, Any]:
        guard(run_id, granted)
        out = call(lambda: tools.get_system_metrics(run_id))
        names = set(store.metric_names(run_id))
        # chart the same memory series the summary describes (the one it computed a floor for); allocator series
        # such as sys/mps_alloc_mb are all zeros on CPU runs and would show nothing
        mem_name = next((k for k, v in out.items() if isinstance(v, dict) and "floor_start" in v), None)
        series = [series_points(store, run_id, n) for n in ("data_wait_frac", "step_time") if n in names]
        if mem_name:
            mem = series_points(store, run_id, mem_name)
            s = store.metrics_df(run_id, names=[mem_name]).groupby("step")["value"].last().sort_index()
            fl = memory_floor(s[np.isfinite(s.to_numpy(dtype=float))])
            mem["floor"] = (
                [[int(k), float(v)] for k, v in fl[0].items()][:: max(1, len(fl[0]) // MAX_POINTS)]
                if fl is not None
                else []
            )
            series.append(mem)
        out["series"] = series
        return out

    @app.get("/v1/runs/{run_id}/data")
    def run_data(run_id: str, granted=authz) -> dict[str, Any]:
        guard(run_id, granted)
        return call(lambda: tools.inspect_data(run_id))

    @app.get("/v1/compare")
    def compare(a: str, b: str, granted=authz) -> dict[str, Any]:
        guard(a, granted)
        guard(b, granted)
        res = call(lambda: tools.compare_runs(a, b))
        res["config_diff"] = call(lambda: tools.diff_configs(a, b))["differences"]
        return res

    if UI_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def root() -> RedirectResponse:
            return RedirectResponse("/ui/")
