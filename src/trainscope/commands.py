"""Command poller: fetches approved commands for this run and executes them at step boundaries."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

log = logging.getLogger("trainscope")

SUPPORTED = {"checkpoint", "pause", "resume", "set_lr"}


class CommandPoller(threading.Thread):
    def __init__(self, sink: Any, run_id: str, interval: float = 3.0):
        super().__init__(name="trainscope-commands", daemon=True)
        self.sink = sink
        self.run_id = run_id
        self.interval = interval
        self.pending: queue.Queue[dict[str, Any]] = queue.Queue()
        self._seen: set[str] = set()
        self._halt = threading.Event()

    def poll_once(self) -> None:
        try:
            for cmd in self.sink.fetch_commands(self.run_id):
                if cmd["id"] not in self._seen:
                    self._seen.add(cmd["id"])
                    self.pending.put(cmd)
        except Exception as e:
            log.debug("command poll failed: %s", e)

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            self.poll_once()

    def stop(self) -> None:
        self._halt.set()

    def drain(self) -> list[dict[str, Any]]:
        out = []
        while True:
            try:
                out.append(self.pending.get_nowait())
            except queue.Empty:
                return out


def execute(run: Any, cmd: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Execute one command at a step boundary. Returns (status, result)."""
    action, args = cmd["action"], cmd.get("args") or {}
    if action not in SUPPORTED:
        return "failed", {"error": f"unsupported action {action}"}
    try:
        if action == "checkpoint":
            path = run.save_checkpoint(args.get("path"))
            return "executed", {"path": path, "step": run.step}
        if action == "set_lr":
            if run.optimizer is None:
                return "failed", {"error": "no optimizer registered; pass optimizer= to ts.watch()"}
            lr = float(args["lr"])
            for g in run.optimizer.param_groups:
                g["lr"] = lr
            return "executed", {"lr": lr, "step": run.step}
        if action == "pause":
            run._pause_requested = True
            return "executed", {"paused_at": run.step}
        if action == "resume":
            run._pause_requested = False
            return "executed", {"resumed_at": run.step}
    except Exception as e:  # never crash training
        return "failed", {"error": repr(e)}
    return "failed", {}


def wait_while_paused(
    run: Any, poller: CommandPoller, poll: float = 1.0, timeout: float | None = None
) -> None:
    """Block at a step boundary until a resume command arrives."""
    t0 = time.time()
    run.event("paused", {})
    run._set_status("paused")
    while run._pause_requested:
        poller.poll_once()
        for cmd in poller.drain():
            status, result = execute(run, cmd)
            run.sink.ack_command(cmd["id"], status, result)
        if timeout is not None and time.time() - t0 > timeout:
            run._pause_requested = False
        if run._pause_requested:
            time.sleep(poll)
    run.event("resumed", {})
    run._set_status("running")
