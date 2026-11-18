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
    pr = sub.add_parser("replay", help="send telemetry left in the SDK spool to the store/server")
    pr.add_argument(
        "--spool-dir", default=os.environ.get("TRAINSCOPE_SPOOL", os.path.expanduser("~/.trainscope/spool"))
    )
    pr.add_argument("--keep", action="store_true", help="keep spool files after a successful replay")
    pd = sub.add_parser("detectors", help="print detector thresholds as a starting detectors.yaml")
    pd.add_argument(
        "--check", metavar="FILE", help="validate a detectors.yaml and show the thresholds it results in"
    )
    pd.add_argument("--project", help="with --check: show the thresholds for this project")
    a = p.parse_args(argv)

    if a.cmd == "detectors":
        print(detectors_yaml(a.check, a.project))
        return
    if a.cmd == "replay":
        print(json.dumps(replay(a), indent=2))
        return

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


def _yaml_num(v: float | int) -> str:
    """A number YAML reads back as a number: '1e-06' is a string in YAML 1.1, '1.0e-06' is a float."""
    s = repr(v)
    if "e" in s and "." not in s.split("e")[0]:
        mant, exp = s.split("e")
        s = f"{mant}.0e{exp}"
    return s


def detectors_yaml(check: str | None = None, project: str | None = None) -> str:
    """Every detector's thresholds as YAML, each with its description from the detector source."""
    import inspect
    import re

    from trainscope_server.detectors import DETECTOR_CLASSES
    from trainscope_server.detectors.config import build, load_file

    cfg = load_file(check, DETECTOR_CLASSES) if check else None
    active = {d.name: d.params for d in build(DETECTOR_CLASSES, cfg, project)}
    head = (
        f"# effective thresholds from {check}" + (f" for project {project!r}" if project else "")
        if check
        else "# TrainScope detector thresholds (built-in defaults). Copy what you want to change into\n"
        "# a detectors.yaml under `defaults:` or `projects: <name>:`, then set TRAINSCOPE_DETECTORS to its path."
    )
    lines = [head, "defaults:"]
    for name, cls in DETECTOR_CLASSES.items():
        if name not in active:
            lines.append(f"  # {name}: disabled")
            continue
        if not cls.defaults:
            continue
        src = inspect.getsource(cls)
        lines.append(f"  {name}:")
        for key, value in active[name].items():
            m = re.search(rf'"{re.escape(key)}":\s*[^,]+,\s*#\s*(.+)', src)
            note = f"  # {m.group(1).strip()}" if m else ""
            changed = "" if value == cls.defaults[key] else f"  (default {cls.defaults[key]})"
            lines.append(f"    {key}: {_yaml_num(value)}{note}{changed}")
    return "\n".join(lines)


def replay(a: argparse.Namespace) -> dict:
    from pathlib import Path

    from trainscope.transport import HttpSink, LocalSink

    sink = (
        HttpSink(a.url, a.token)
        if a.url
        else LocalSink(a.db or os.environ.get("TRAINSCOPE_DB", "trainscope.duckdb"))
    )
    done, failed, batches = [], [], 0
    for f in sorted(Path(a.spool_dir).glob("*.jsonl")):
        try:
            for line in f.read_text().splitlines():
                if line.strip():
                    sink.send(json.loads(line))
                    batches += 1
            done.append(f.name)
            if not a.keep:
                f.unlink()
        except Exception as e:  # leave the file for the next attempt
            failed.append({"file": f.name, "error": str(e)})
    return {"replayed_files": done, "batches": batches, "failed": failed}


if __name__ == "__main__":
    main()
