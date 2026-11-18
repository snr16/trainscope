"""FastAPI ingestion service.

POST /v1/runs                      register / update a run
POST /v1/ingest                    batched metrics + events (idempotent)
POST /v1/runs/{id}/status          mark finished / crashed / paused
GET  /v1/runs, /v1/runs/{id}, /v1/runs/{id}/findings
GET  /v1/runs/{id}/commands        SDK poller (status=approved)
POST /v1/runs/{id}/commands        request a command (policy applies)
POST /v1/commands/{id}/approve     human approval
POST /v1/commands/{id}/ack         SDK reports execution
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException

from .commands import policy
from .db import Storage, open_store
from .detectors.engine import refresh_findings
from .ingest.auth import Auth, check_project
from .ingest.schemas import AckIn, ApproveIn, CommandIn, EventIn, MetricIn, RunIn, StatusIn, validate_records


class DetectorScheduler:
    """Re-runs detectors for a run at most every `min_interval` seconds after new data arrives."""

    def __init__(self, store: Storage, min_interval: float = 5.0):
        self.store = store
        self.min_interval = min_interval
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def maybe_run(self, run_id: str, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if not force and now - self._last.get(run_id, 0) < self.min_interval:
                return
            self._last[run_id] = now
        try:
            refresh_findings(self.store, run_id)
        except Exception:
            pass


def create_app(
    store: Storage | None = None, auth: Auth | None = None, detector_interval: float = 5.0
) -> FastAPI:
    store = store or open_store(os.environ.get("TRAINSCOPE_DB"))
    auth = auth or Auth()
    sched = DetectorScheduler(store, detector_interval)
    app = FastAPI(title="TrainScope ingestion", version="0.1.0")
    app.state.store = store
    authz = Depends(auth.dependency())
    run_projects: dict[str, str] = {}

    def project_of(run_id: str) -> str:
        if run_id not in run_projects:
            r = store.get_run(run_id)
            if r is None:
                raise HTTPException(404, f"unknown run {run_id}")
            run_projects[run_id] = r["project"]
        return run_projects[run_id]

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/runs")
    def register_run(run: RunIn, granted=authz) -> dict[str, str]:
        check_project(granted, run.project)
        store.upsert_run(run.model_dump())
        run_projects[run.id] = run.project
        return {"id": run.id}

    @app.post("/v1/ingest")
    def ingest(batch: dict[str, Any], bg: BackgroundTasks, granted=authz) -> dict[str, Any]:
        metrics, bad_m = validate_records(MetricIn, batch.get("metrics") or [])
        events, bad_e = validate_records(EventIn, batch.get("events") or [])
        run_ids = {m["run_id"] for m in metrics} | {e["run_id"] for e in events}
        unknown = set()
        for rid in run_ids:
            try:
                check_project(granted, project_of(rid))
            except HTTPException as e:
                if e.status_code == 404:
                    unknown.add(rid)
                else:
                    raise
        if unknown:
            metrics = [m for m in metrics if m["run_id"] not in unknown]
            events = [e for e in events if e["run_id"] not in unknown]
        n_m = store.write_metrics(metrics)
        n_e = store.write_events(events)
        for rid in run_ids - unknown:
            bg.add_task(sched.maybe_run, rid, any(e["type"].startswith("nonfinite") for e in events))
        return {
            "accepted_metrics": n_m,
            "accepted_events": n_e,
            "rejected": bad_m + bad_e,
            "unknown_runs": sorted(unknown),
        }

    @app.post("/v1/runs/{run_id}/status")
    def set_status(run_id: str, body: StatusIn, bg: BackgroundTasks, granted=authz) -> dict[str, str]:
        check_project(granted, project_of(run_id))
        store.set_run_status(
            run_id, body.status, None if body.status in ("running", "paused") else time.time()
        )
        bg.add_task(sched.maybe_run, run_id, True)
        return {"status": body.status}

    @app.get("/v1/runs")
    def list_runs(
        project: str | None = None, status: str | None = None, granted=authz
    ) -> list[dict[str, Any]]:
        runs = store.list_runs(project, status)
        return [r for r in runs if granted in ("*", r["project"])]

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str, granted=authz) -> dict[str, Any]:
        check_project(granted, project_of(run_id))
        return store.get_run(run_id) or {}

    @app.get("/v1/runs/{run_id}/findings")
    def findings(run_id: str, granted=authz) -> list[dict[str, Any]]:
        check_project(granted, project_of(run_id))
        sched.maybe_run(run_id, force=True)
        return store.findings(run_id)

    @app.get("/v1/runs/{run_id}/commands")
    def run_commands(run_id: str, status: str | None = None, granted=authz) -> list[dict[str, Any]]:
        check_project(granted, project_of(run_id))
        return store.list_commands(run_id, status)

    @app.post("/v1/runs/{run_id}/commands")
    def request_command(run_id: str, body: CommandIn, granted=authz) -> dict[str, Any]:
        check_project(granted, project_of(run_id))
        try:
            return policy.request(store, run_id, body.action, body.args, body.requested_by)
        except policy.CommandError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/v1/commands")
    def list_commands(
        run_id: str | None = None, status: str | None = None, granted=authz
    ) -> list[dict[str, Any]]:
        return [c for c in store.list_commands(run_id, status) if granted in ("*", project_of(c["run_id"]))]

    @app.post("/v1/commands/{cmd_id}/approve")
    def approve(cmd_id: str, body: ApproveIn, granted=authz) -> dict[str, Any]:
        cmd = store.get_command(cmd_id)
        if cmd is None:
            raise HTTPException(404, "unknown command")
        check_project(granted, project_of(cmd["run_id"]))
        try:
            return policy.approve(store, cmd_id, body.approved_by, body.approve)
        except policy.CommandError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/v1/commands/{cmd_id}/ack")
    def ack(cmd_id: str, body: AckIn, granted=authz) -> dict[str, Any]:
        cmd = store.get_command(cmd_id)
        if cmd is None:
            raise HTTPException(404, "unknown command")
        check_project(granted, project_of(cmd["run_id"]))
        return (
            store.update_command(cmd_id, status=body.status, result=body.result, executed_at=time.time())
            or {}
        )

    return app


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser("trainscope-server")
    p.add_argument("--host", default=os.environ.get("TRAINSCOPE_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("TRAINSCOPE_PORT", 8765)))
    p.add_argument("--db", default=os.environ.get("TRAINSCOPE_DB", "trainscope.duckdb"))
    a = p.parse_args()
    uvicorn.run(create_app(open_store(a.db)), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
