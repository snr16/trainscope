"""``trainscope`` CLI: list runs/commands, approve or reject pending commands, show findings."""

from __future__ import annotations

import argparse
import getpass
import json
import os

import httpx


def _local_store(db: str | None):
    from trainscope_server.db import open_store

    return open_store(db)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser("trainscope")
    p.add_argument("--db", help="local store (default $TRAINSCOPE_DB or trainscope.duckdb)")
    p.add_argument("--url", default=os.environ.get("TRAINSCOPE_URL"), help="server URL (remote mode)")
    p.add_argument("--token", default=os.environ.get("TRAINSCOPE_TOKEN"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("runs")
    pc = sub.add_parser("commands")
    pc.add_argument("--run")
    pc.add_argument("--status")
    for name in ("approve", "reject"):
        pa = sub.add_parser(name)
        pa.add_argument("command_id")
        pa.add_argument("--by", default=getpass.getuser())
    pf = sub.add_parser("findings")
    pf.add_argument("run_id")
    a = p.parse_args(argv)

    if a.url:
        c = httpx.Client(base_url=a.url, headers={"Authorization": f"Bearer {a.token}"} if a.token else {})
        if a.cmd == "runs":
            out = c.get("/v1/runs").json()
        elif a.cmd == "commands":
            out = c.get(
                "/v1/commands", params={k: v for k, v in {"run_id": a.run, "status": a.status}.items() if v}
            ).json()
        elif a.cmd in ("approve", "reject"):
            out = c.post(
                f"/v1/commands/{a.command_id}/approve",
                json={"approved_by": f"human:{a.by}", "approve": a.cmd == "approve"},
            ).json()
        else:
            out = c.get(f"/v1/runs/{a.run_id}/findings").json()
    else:
        from trainscope_server.commands import policy

        store = _local_store(a.db)
        if a.cmd == "runs":
            out = [{k: r[k] for k in ("id", "project", "name", "status")} for r in store.list_runs()]
        elif a.cmd == "commands":
            out = store.list_commands(a.run, a.status)
        elif a.cmd in ("approve", "reject"):
            out = policy.approve(store, a.command_id, f"human:{a.by}", a.cmd == "approve")
        else:
            from trainscope_server.detectors.engine import refresh_findings

            out = refresh_findings(store, a.run_id)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
