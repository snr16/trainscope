"""Command safety policy: checkpoint is always allowed; disruptive actions need human approval."""

from __future__ import annotations

from typing import Any

ALLOWED = {"checkpoint", "pause", "resume", "set_lr"}
AUTO_APPROVED = {"checkpoint"}


class CommandError(ValueError):
    pass


def validate(action: str, args: dict[str, Any]) -> None:
    if action not in ALLOWED:
        raise CommandError(f"action must be one of {sorted(ALLOWED)}")
    if action == "set_lr":
        lr = args.get("lr")
        if not isinstance(lr, int | float) or not (0 < float(lr) < 10):
            raise CommandError("set_lr requires args.lr in (0, 10)")


def request(
    store: Any, run_id: str, action: str, args: dict[str, Any] | None, requested_by: str
) -> dict[str, Any]:
    """Queue a command. Returns the command record (status 'approved' or 'pending_approval')."""
    args = args or {}
    validate(action, args)
    run = store.get_run(run_id)
    if run is None:
        raise CommandError(f"unknown run {run_id}")
    if run["status"] not in ("running", "paused"):
        raise CommandError(f"run is {run['status']}; commands only apply to live runs")
    auto = action in AUTO_APPROVED
    return store.create_command(
        {
            "run_id": run_id,
            "action": action,
            "args": args,
            "requested_by": requested_by,
            "status": "approved" if auto else "pending_approval",
            "approved_by": "policy:auto" if auto else None,
        }
    )


def approve(store: Any, cmd_id: str, approved_by: str, approve: bool = True) -> dict[str, Any]:
    cmd = store.get_command(cmd_id)
    if cmd is None:
        raise CommandError(f"unknown command {cmd_id}")
    if cmd["status"] != "pending_approval":
        raise CommandError(f"command is {cmd['status']}, not pending approval")
    if not approved_by or approved_by.startswith(("assistant", "mcp", "policy")):
        raise CommandError("approval must come from a human")
    return store.update_command(cmd_id, status="approved" if approve else "rejected", approved_by=approved_by)
