"""DuckDB backend (local mode, default).

Connections are opened per operation so several processes (SDK in local mode, MCP server, CLI) can share
the same file; DuckDB allows only one writer at a time, so lock conflicts are retried with backoff.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .rows import (
    COMMAND_COLS,
    FINDING_COLS,
    RUN_COLS,
    command_out,
    event_key,
    finding_out,
    new_command,
    run_out,
)

_SCHEMA = (Path(__file__).with_name("schema.sql")).read_text()

# DuckDB allows one open handle per file per process, so every store on the same file shares one lock
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.RLock:
    key = path if path == ":memory:" else str(Path(path).resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


class DuckDBStore:
    def __init__(self, path: str = "trainscope.duckdb", retries: int = 40):
        self.path = path
        self.retries = retries
        self._lock = threading.RLock() if path == ":memory:" else _lock_for(path)
        self._mem: duckdb.DuckDBPyConnection | None = None
        if path == ":memory:":
            self._mem = duckdb.connect(":memory:")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[duckdb.DuckDBPyConnection]:
        with self._lock:
            if self._mem is not None:
                yield self._mem
                return
            delay = 0.01
            for attempt in range(self.retries):
                try:
                    con = duckdb.connect(self.path)
                    break
                except (duckdb.IOException, duckdb.ResourceInUseException):
                    if attempt == self.retries - 1:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 1.5, 0.5)
            try:
                yield con
            finally:
                con.close()

    # ---------------------------------------------------------------- runs
    def upsert_run(self, run: dict[str, Any]) -> None:
        with self._conn() as c:
            existing = c.execute("SELECT id FROM runs WHERE id = ?", [run["id"]]).fetchone()
            if existing:
                c.execute(
                    "UPDATE runs SET project=?, name=COALESCE(?, name), status=COALESCE(?, status),"
                    " config_json=COALESCE(?, config_json), env_json=COALESCE(?, env_json) WHERE id=?",
                    [
                        run["project"],
                        run.get("name"),
                        run.get("status"),
                        json.dumps(run["config"]) if run.get("config") is not None else None,
                        json.dumps(run["env"]) if run.get("env") is not None else None,
                        run["id"],
                    ],
                )
            else:
                c.execute(
                    "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        run["id"],
                        run["project"],
                        run.get("name"),
                        run.get("status", "running"),
                        json.dumps(run.get("config") or {}),
                        json.dumps(run.get("env") or {}),
                        run.get("started_at") or time.time(),
                        None,
                    ],
                )

    def set_run_status(self, run_id: str, status: str, ended_at: float | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET status=?, ended_at=COALESCE(?, ended_at) WHERE id=?",
                [status, ended_at, run_id],
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(f"SELECT {', '.join(RUN_COLS)} FROM runs WHERE id=?", [run_id]).fetchone()
        return run_out(row) if row else None

    def list_runs(self, project: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        q = f"SELECT {', '.join(RUN_COLS)} FROM runs WHERE 1=1"
        args: list[Any] = []
        if project:
            q, args = q + " AND project=?", args + [project]
        if status:
            q, args = q + " AND status=?", args + [status]
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY started_at DESC", args).fetchall()
        return [run_out(r) for r in rows]

    # ------------------------------------------------------------- metrics
    def write_metrics(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        df = pd.DataFrame(rows, columns=["run_id", "step", "ts", "name", "layer", "value"])
        df["layer"] = df["layer"].fillna("")
        # idempotency: last write wins within the batch as well as across batches
        df = df.drop_duplicates(subset=["run_id", "step", "name", "layer"], keep="last")
        with self._conn() as c:
            c.register("incoming", df)
            c.execute(
                "INSERT OR REPLACE INTO metrics SELECT run_id, step, ts, name, layer, value FROM incoming"
            )
            c.unregister("incoming")
        return len(df)

    def write_events(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        vals = [
            (
                e["run_id"],
                int(e["step"]),
                float(e.get("ts") or time.time()),
                e["type"],
                json.dumps(e.get("payload") or {}),
                event_key(e),
            )
            for e in rows
        ]
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?, ?)", vals)
        return len(vals)

    def metrics_df(self, run_id, names=None, layer=None, step_min=None, step_max=None, name_prefix=None):
        q, args = "SELECT step, ts, name, layer, value FROM metrics WHERE run_id=?", [run_id]
        if names:
            q += f" AND name IN ({', '.join('?' * len(names))})"
            args += list(names)
        if name_prefix:
            q, args = q + " AND name LIKE ?", args + [name_prefix + "%"]
        if layer is not None:
            q, args = q + " AND layer=?", args + [layer]
        if step_min is not None:
            q, args = q + " AND step>=?", args + [int(step_min)]
        if step_max is not None:
            q, args = q + " AND step<=?", args + [int(step_max)]
        with self._conn() as c:
            return c.execute(q + " ORDER BY step", args).df()

    def metric_names(self, run_id: str) -> list[str]:
        with self._conn() as c:
            return [
                r[0]
                for r in c.execute(
                    "SELECT DISTINCT name FROM metrics WHERE run_id=? ORDER BY name", [run_id]
                ).fetchall()
            ]

    def run_progress(self) -> dict[str, dict[str, Any]]:
        """Last step and last loss of every run, in one query (for run lists)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT run_id, max(step), arg_max(value, step) FILTER (WHERE name = 'loss' AND layer = '') "
                "FROM metrics GROUP BY run_id"
            ).fetchall()
        return {r[0]: {"last_step": int(r[1]), "last_loss": r[2]} for r in rows}

    def max_step(self, run_id: str) -> int:
        with self._conn() as c:
            r = c.execute("SELECT max(step) FROM metrics WHERE run_id=?", [run_id]).fetchone()
        return int(r[0]) if r and r[0] is not None else -1

    def events(self, run_id: str, type: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT step, ts, type, payload_json FROM events WHERE run_id=?", [run_id]
        if type:
            q, args = q + " AND type=?", args + [type]
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY step", args).fetchall()
        return [{"step": r[0], "ts": r[1], "type": r[2], "payload": json.loads(r[3] or "{}")} for r in rows]

    def merge_from(self, other_path: str) -> None:
        """Copy every row of another TrainScope DuckDB file into this store (idempotent)."""
        with self._conn() as c:
            c.execute(f"ATTACH '{other_path}' AS src (READ_ONLY)")
            try:
                for table in ("runs", "metrics", "events", "findings", "commands"):
                    c.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM src.{table}")
            finally:
                c.execute("DETACH src")

    # ------------------------------------------------------------ findings
    def replace_findings(self, run_id: str, findings: list[dict[str, Any]]) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM findings WHERE run_id=?", [run_id])
            if findings:
                c.executemany(
                    f"INSERT INTO findings VALUES ({', '.join('?' * len(FINDING_COLS))})",
                    [
                        (
                            f["id"],
                            run_id,
                            f["detector"],
                            f["severity"],
                            f["step_start"],
                            f["step_end"],
                            f["summary"],
                            json.dumps(f.get("evidence") or {}),
                            f.get("created_at") or time.time(),
                        )
                        for f in findings
                    ],
                )

    def findings(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                f"SELECT {', '.join(FINDING_COLS)} FROM findings WHERE run_id=? ORDER BY step_start", [run_id]
            ).fetchall()
        return [finding_out(r) for r in rows]

    # ------------------------------------------------------------ commands
    def create_command(self, cmd: dict[str, Any]) -> dict[str, Any]:
        row = new_command(cmd)
        with self._conn() as c:
            c.execute(f"INSERT INTO commands VALUES ({', '.join('?' * len(COMMAND_COLS))})", list(row))
        return command_out(row)

    def update_command(self, cmd_id: str, **fields: Any) -> dict[str, Any] | None:
        if "result" in fields:
            fields["result_json"] = json.dumps(fields.pop("result"))
        if fields:
            sets = ", ".join(f"{k}=?" for k in fields)
            with self._conn() as c:
                c.execute(f"UPDATE commands SET {sets} WHERE id=?", [*fields.values(), cmd_id])
        return self.get_command(cmd_id)

    def get_command(self, cmd_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(f"SELECT {', '.join(COMMAND_COLS)} FROM commands WHERE id=?", [cmd_id]).fetchone()
        return command_out(row) if row else None

    def list_commands(self, run_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        q = f"SELECT {', '.join(COMMAND_COLS)} FROM commands WHERE 1=1"
        args: list[Any] = []
        if run_id:
            q, args = q + " AND run_id=?", args + [run_id]
        if status:
            q, args = q + " AND status=?", args + [status]
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY created_at", args).fetchall()
        return [command_out(r) for r in rows]
