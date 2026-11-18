"""In-memory ring buffer + background flush thread with on-disk spooling.

Training threads only append to a deque (O(1), no I/O). A daemon thread drains it in batches and hands
batches to a sink. If the sink fails, the batch is appended to a JSONL spool file and retried later, so
training never blocks or crashes because telemetry is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("trainscope")


class Sink(Protocol):
    def send(self, batch: dict[str, list[dict[str, Any]]]) -> None: ...


class RingBuffer:
    """Bounded, thread-safe buffer of telemetry records. Oldest records are dropped when full."""

    def __init__(self, maxlen: int = 200_000):
        self.metrics: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.events: deque[dict[str, Any]] = deque(maxlen=maxlen // 10 or 1)
        self.dropped = 0
        self._maxlen = maxlen

    def add_metric(self, rec: dict[str, Any]) -> None:
        if len(self.metrics) == self._maxlen:
            self.dropped += 1
        self.metrics.append(rec)

    def add_event(self, rec: dict[str, Any]) -> None:
        self.events.append(rec)

    def drain(self, max_items: int) -> dict[str, list[dict[str, Any]]]:
        out_m: list[dict[str, Any]] = []
        out_e: list[dict[str, Any]] = []
        try:
            while len(out_m) < max_items:
                out_m.append(self.metrics.popleft())
        except IndexError:
            pass
        try:
            while True:
                out_e.append(self.events.popleft())
        except IndexError:
            pass
        return {"metrics": out_m, "events": out_e}

    def __len__(self) -> int:
        return len(self.metrics) + len(self.events)


class Flusher(threading.Thread):
    def __init__(
        self,
        buffer: RingBuffer,
        sink: Sink,
        spool_path: str | os.PathLike,
        interval: float = 1.0,
        batch_size: int = 5000,
        retry_interval: float = 5.0,
    ):
        super().__init__(name="trainscope-flush", daemon=True)
        self.buffer = buffer
        self.sink = sink
        self.spool_path = Path(spool_path)
        self.interval = interval
        self.batch_size = batch_size
        self.retry_interval = retry_interval
        self._halt = threading.Event()
        self._wake = threading.Event()
        self._last_retry = 0.0
        self.sent_batches = 0
        self.failed_batches = 0
        self._send_lock = threading.Lock()

    # -- public API
    def wake(self) -> None:
        self._wake.set()

    def stop(self, timeout: float = 10.0, final_retry_s: float = 20.0) -> None:
        self._halt.set()
        self._wake.set()
        self.join(timeout)
        self.flush_now(final=True)
        # training is over, so it's fine to wait a little for a busy/locked sink before leaving data on disk
        delay, waited = 0.25, 0.0
        while self.spooled and waited < final_retry_s:
            time.sleep(delay)
            waited += delay
            delay = min(delay * 2, 4.0)
            self._replay_spool(force=True)
        if self.spooled:
            log.warning(
                "trainscope: telemetry left in %s; replay it with `trainscope replay`", self.spool_path
            )

    def flush_now(self, final: bool = False) -> None:
        with self._send_lock:
            while len(self.buffer):
                batch = self.buffer.drain(self.batch_size)
                if not batch["metrics"] and not batch["events"]:
                    break
                self._send_or_spool(batch)
            self._replay_spool(force=final)

    # -- internals
    def run(self) -> None:
        while not self._halt.is_set():
            self._wake.wait(self.interval)
            self._wake.clear()
            try:
                self.flush_now()
            except Exception:  # never let the thread die
                log.exception("trainscope flush failed")

    def _send_or_spool(self, batch: dict[str, list[dict[str, Any]]]) -> bool:
        try:
            self.sink.send(batch)
            self.sent_batches += 1
            return True
        except Exception as e:
            self.failed_batches += 1
            log.debug("trainscope sink unavailable (%s); spooling %d records", e, len(batch["metrics"]))
            self._spool(batch)
            return False

    def _spool(self, batch: dict[str, list[dict[str, Any]]]) -> None:
        try:
            self.spool_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.spool_path, "a") as f:
                f.write(json.dumps(batch) + "\n")
        except OSError:
            log.warning("trainscope could not spool telemetry to %s; dropping batch", self.spool_path)

    def _replay_spool(self, force: bool = False) -> None:
        if not self.spool_path.exists():
            return
        now = time.time()
        if not force and now - self._last_retry < self.retry_interval:
            return
        self._last_retry = now
        tmp = self.spool_path.with_suffix(".replay")
        try:
            os.replace(self.spool_path, tmp)
        except OSError:
            return
        remaining: list[str] = []
        failed = False
        with open(tmp) as f:
            for line in f:
                if failed:
                    remaining.append(line)
                    continue
                try:
                    self.sink.send(json.loads(line))
                    self.sent_batches += 1
                except json.JSONDecodeError:
                    continue
                except Exception:
                    failed = True
                    remaining.append(line)
        if remaining:
            with open(self.spool_path, "a") as f:
                f.writelines(remaining)
        tmp.unlink(missing_ok=True)

    @property
    def spooled(self) -> bool:
        return self.spool_path.exists() and self.spool_path.stat().st_size > 0
