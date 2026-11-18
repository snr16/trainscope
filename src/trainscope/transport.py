"""Sinks: where telemetry goes (remote ingestion service, local DuckDB file, or nowhere)."""

from __future__ import annotations

from typing import Any


class HttpSink:
    """Sends batches to the ingestion service. Short timeouts: the flush thread must not hang."""

    def __init__(self, url: str, token: str | None = None, timeout: float = 5.0):
        import httpx

        self.url = url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = httpx.Client(base_url=self.url, headers=headers, timeout=timeout)

    def register_run(self, run: dict[str, Any]) -> None:
        self.client.post("/v1/runs", json=run).raise_for_status()

    def send(self, batch: dict[str, list[dict[str, Any]]]) -> None:
        r = self.client.post("/v1/ingest", json=batch)
        r.raise_for_status()

    def set_status(self, run_id: str, status: str) -> None:
        self.client.post(f"/v1/runs/{run_id}/status", json={"status": status}).raise_for_status()

    def fetch_commands(self, run_id: str) -> list[dict[str, Any]]:
        r = self.client.get(f"/v1/runs/{run_id}/commands", params={"status": "approved"})
        r.raise_for_status()
        return r.json()

    def ack_command(self, cmd_id: str, status: str, result: dict[str, Any] | None = None) -> None:
        self.client.post(f"/v1/commands/{cmd_id}/ack", json={"status": status, "result": result or {}})


class LocalSink:
    """Writes straight into a local store (DuckDB). No server needed."""

    def __init__(self, db_path: str):
        from trainscope_server.db import open_store

        self.store = open_store(db_path)

    def register_run(self, run: dict[str, Any]) -> None:
        self.store.upsert_run(run)

    def send(self, batch: dict[str, list[dict[str, Any]]]) -> None:
        self.store.write_metrics(batch.get("metrics", []))
        self.store.write_events(batch.get("events", []))

    def set_status(self, run_id: str, status: str) -> None:
        import time

        self.store.set_run_status(run_id, status, time.time() if status != "running" else None)

    def fetch_commands(self, run_id: str) -> list[dict[str, Any]]:
        return self.store.list_commands(run_id, status="approved")

    def ack_command(self, cmd_id: str, status: str, result: dict[str, Any] | None = None) -> None:
        import time

        self.store.update_command(cmd_id, status=status, result=result or {}, executed_at=time.time())


class NullSink:
    def register_run(self, run):
        pass

    def send(self, batch):
        pass

    def set_status(self, run_id, status):
        pass

    def fetch_commands(self, run_id):
        return []

    def ack_command(self, cmd_id, status, result=None):
        pass
