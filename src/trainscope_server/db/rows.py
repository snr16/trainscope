"""Row (de)serialisation helpers shared by backends."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

RUN_COLS = ["id", "project", "name", "status", "config_json", "env_json", "started_at", "ended_at"]
FINDING_COLS = [
    "id",
    "run_id",
    "detector",
    "severity",
    "step_start",
    "step_end",
    "summary",
    "evidence_json",
    "created_at",
]
COMMAND_COLS = [
    "id",
    "run_id",
    "action",
    "args_json",
    "status",
    "requested_by",
    "approved_by",
    "created_at",
    "executed_at",
    "result_json",
]


def _loads(s: Any) -> Any:
    if s is None or s == "":
        return {}
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return {}


def run_out(row: tuple) -> dict[str, Any]:
    d = dict(zip(RUN_COLS, row, strict=False))
    d["config"] = _loads(d.pop("config_json"))
    d["env"] = _loads(d.pop("env_json"))
    return d


def finding_out(row: tuple) -> dict[str, Any]:
    d = dict(zip(FINDING_COLS, row, strict=False))
    d["evidence"] = _loads(d.pop("evidence_json"))
    return d


def command_out(row: tuple) -> dict[str, Any]:
    d = dict(zip(COMMAND_COLS, row, strict=False))
    d["args"] = _loads(d.pop("args_json"))
    d["result"] = _loads(d.pop("result_json"))
    return d


def event_key(e: dict[str, Any]) -> str:
    if e.get("dedup_key"):
        return e["dedup_key"]
    # hash the payload: some events (e.g. dataset_ids) carry thousands of ids, and Postgres caps index rows at 8 KB
    digest = hashlib.sha1(json.dumps(e.get("payload", {}), sort_keys=True).encode()).hexdigest()[:16]
    return f"{e['type']}:{e['step']}:{digest}"


def new_command(cmd: dict[str, Any]) -> tuple:
    return (
        cmd.get("id") or uuid.uuid4().hex[:12],
        cmd["run_id"],
        cmd["action"],
        json.dumps(cmd.get("args") or {}),
        cmd.get("status", "pending"),
        cmd.get("requested_by"),
        cmd.get("approved_by"),
        cmd.get("created_at") or time.time(),
        None,
        None,
    )
