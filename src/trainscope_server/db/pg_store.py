"""PostgreSQL / TimescaleDB backend (team / server mode).

Same semantics as DuckDBStore. Requires ``pip install trainscope[postgres]``. If the timescaledb extension is
available, metrics becomes a hypertable with a 100-step continuous aggregate (see timescale.sql).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

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

_HERE = Path(__file__).parent
log = logging.getLogger("trainscope.db")


class PostgresStore:
    def __init__(self, url: str):
        import psycopg
        from psycopg_pool import ConnectionPool  # type: ignore[import-not-found]

        self._psycopg = psycopg
        self.pool = ConnectionPool(url, min_size=1, max_size=8, kwargs={"autocommit": True})
        with self.pool.connection() as c:
            c.execute((_HERE / "schema.sql").read_text())
            # TimescaleDB extras, one statement at a time; on plain Postgres they fail and the store still works,
            # just without hypertables / downsampled aggregates
            for stmt in (_HERE / "timescale.sql").read_text().split("-- @@"):
                if not stmt.strip() or all(
                    line.strip().startswith("--") for line in stmt.strip().splitlines()
                ):
                    continue
                try:
                    c.execute(stmt)
                except psycopg.Error as e:
                    log.warning("timescale setup step skipped: %s", str(e).splitlines()[0])

    def _q(self, sql: str, args: Any = (), fetch: str | None = None):
        with self.pool.connection() as c:
            cur = c.execute(sql.replace("?", "%s"), args)
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return None

    def upsert_run(self, run):
        self._q(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, NULL) ON CONFLICT (id) DO UPDATE SET"
            " project=EXCLUDED.project, name=COALESCE(EXCLUDED.name, runs.name),"
            " status=COALESCE(EXCLUDED.status, runs.status), config_json=EXCLUDED.config_json,"
            " env_json=EXCLUDED.env_json",
            (
                run["id"],
                run["project"],
                run.get("name"),
                run.get("status", "running"),
                json.dumps(run.get("config") or {}),
                json.dumps(run.get("env") or {}),
                run.get("started_at") or time.time(),
            ),
        )

    def set_run_status(self, run_id, status, ended_at=None):
        self._q(
            "UPDATE runs SET status=?, ended_at=COALESCE(?, ended_at) WHERE id=?", (status, ended_at, run_id)
        )

    def get_run(self, run_id):
        r = self._q(f"SELECT {', '.join(RUN_COLS)} FROM runs WHERE id=?", (run_id,), "one")
        return run_out(r) if r else None

    def list_runs(self, project=None, status=None):
        q, a = f"SELECT {', '.join(RUN_COLS)} FROM runs WHERE TRUE", []
        if project:
            q, a = q + " AND project=?", a + [project]
        if status:
            q, a = q + " AND status=?", a + [status]
        return [run_out(r) for r in self._q(q + " ORDER BY started_at DESC", a, "all")]

    def write_metrics(self, rows):
        if not rows:
            return 0
        df = pd.DataFrame(rows, columns=["run_id", "step", "ts", "name", "layer", "value"])
        df["layer"] = df["layer"].fillna("")
        df = df.drop_duplicates(subset=["run_id", "step", "name", "layer"], keep="last")
        with self.pool.connection() as c, c.cursor() as cur:
            cur.executemany(
                "INSERT INTO metrics VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (run_id, step, name, layer)"
                " DO UPDATE SET value=EXCLUDED.value, ts=EXCLUDED.ts",
                list(df.itertuples(index=False, name=None)),
            )
        return len(df)

    def write_events(self, rows):
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
        with self.pool.connection() as c, c.cursor() as cur:
            cur.executemany("INSERT INTO events VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING", vals)
        return len(vals)

    def metrics_df(self, run_id, names=None, layer=None, step_min=None, step_max=None, name_prefix=None):
        q, a = "SELECT step, ts, name, layer, value FROM metrics WHERE run_id=?", [run_id]
        if names:
            q, a = q + " AND name = ANY(?)", a + [list(names)]
        if name_prefix:
            q, a = q + " AND name LIKE ?", a + [name_prefix + "%"]
        if layer is not None:
            q, a = q + " AND layer=?", a + [layer]
        if step_min is not None:
            q, a = q + " AND step>=?", a + [step_min]
        if step_max is not None:
            q, a = q + " AND step<=?", a + [step_max]
        rows = self._q(q + " ORDER BY step", a, "all")
        return pd.DataFrame(rows, columns=["step", "ts", "name", "layer", "value"])

    def metric_names(self, run_id):
        return [
            r[0]
            for r in self._q("SELECT DISTINCT name FROM metrics WHERE run_id=? ORDER BY 1", (run_id,), "all")
        ]

    def run_progress(self):
        rows = self._q(
            "SELECT m.run_id, m.last_step, l.value FROM "
            "(SELECT run_id, max(step) AS last_step FROM metrics GROUP BY run_id) m LEFT JOIN "
            "(SELECT DISTINCT ON (run_id) run_id, value FROM metrics WHERE name = 'loss' AND layer = '' "
            " ORDER BY run_id, step DESC) l ON l.run_id = m.run_id", (), "all")
        return {r[0]: {"last_step": int(r[1]), "last_loss": r[2]} for r in rows}

    def max_step(self, run_id):
        r = self._q("SELECT max(step) FROM metrics WHERE run_id=?", (run_id,), "one")
        return int(r[0]) if r and r[0] is not None else -1

    def events(self, run_id, type=None):
        q, a = "SELECT step, ts, type, payload_json FROM events WHERE run_id=?", [run_id]
        if type:
            q, a = q + " AND type=?", a + [type]
        return [
            {"step": r[0], "ts": r[1], "type": r[2], "payload": json.loads(r[3] or "{}")}
            for r in self._q(q + " ORDER BY step", a, "all")
        ]

    def replace_findings(self, run_id, findings):
        self._q("DELETE FROM findings WHERE run_id=?", (run_id,))
        for f in findings:
            self._q(
                f"INSERT INTO findings VALUES ({', '.join('?' * len(FINDING_COLS))})",
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
                ),
            )

    def findings(self, run_id):
        return [
            finding_out(r)
            for r in self._q(
                f"SELECT {', '.join(FINDING_COLS)} FROM findings WHERE run_id=? ORDER BY step_start",
                (run_id,),
                "all",
            )
        ]

    def create_command(self, cmd):
        row = new_command(cmd)
        self._q(f"INSERT INTO commands VALUES ({', '.join('?' * len(COMMAND_COLS))})", row)
        return command_out(row)

    def update_command(self, cmd_id, **fields):
        if "result" in fields:
            fields["result_json"] = json.dumps(fields.pop("result"))
        if fields:
            self._q(
                f"UPDATE commands SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                (*fields.values(), cmd_id),
            )
        return self.get_command(cmd_id)

    def get_command(self, cmd_id):
        r = self._q(f"SELECT {', '.join(COMMAND_COLS)} FROM commands WHERE id=?", (cmd_id,), "one")
        return command_out(r) if r else None

    def list_commands(self, run_id=None, status=None):
        q, a = f"SELECT {', '.join(COMMAND_COLS)} FROM commands WHERE TRUE", []
        if run_id:
            q, a = q + " AND run_id=?", a + [run_id]
        if status:
            q, a = q + " AND status=?", a + [status]
        return [command_out(r) for r in self._q(q + " ORDER BY created_at", a, "all")]
