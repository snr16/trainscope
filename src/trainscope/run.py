"""The Run object: owns the buffer, flush thread, system sampler, command poller and model watcher."""

from __future__ import annotations

import atexit
import logging
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .buffer import Flusher, RingBuffer
from .commands import CommandPoller, execute, wait_while_paused
from .dataloader import TimedLoader
from .system import SystemSampler
from .transport import HttpSink, LocalSink, NullSink

log = logging.getLogger("trainscope")


def _env_info() -> dict[str, Any]:
    env: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "numpy"):
        m = sys.modules.get(mod)
        if m is not None:
            env[mod] = getattr(m, "__version__", "?")
    try:
        env["git_commit"] = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2
            ).stdout.strip()
            or None
        )
    except Exception:
        env["git_commit"] = None
    try:
        import torch

        env["device"] = (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    except Exception:
        pass
    return env


class Run:
    def __init__(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        name: str | None = None,
        mode: str | None = None,
        url: str | None = None,
        token: str | None = None,
        db: str | None = None,
        run_id: str | None = None,
        system_interval: float | None = 1.0,
        poll_commands: bool = True,
        command_interval: float = 3.0,
        flush_interval: float = 1.0,
        data_every: int = 50,
        spool_dir: str | None = None,
        checkpoint_dir: str | None = None,
        stdout_log: str | None = None,
    ):
        self.project = project
        self.id = run_id or uuid.uuid4().hex[:12]
        self.name = name or f"run-{self.id[:6]}"
        self.config = dict(config or {})
        self.step = -1
        self.data_every = max(1, data_every)
        self.optimizer: Any = None
        self.model: Any = None
        self.scaler: Any = None
        self.watcher: Any = None
        self._checkpoint_fn: Callable[[str], None] | None = None
        self.checkpoint_dir = Path(checkpoint_dir or os.environ.get("TRAINSCOPE_CKPT_DIR", "checkpoints"))
        self._pause_requested = False
        self._data_wait = 0.0
        self._last_t: float | None = None
        self._nan_reported = False
        self._amp_skipped = 0
        self._last_scale: float | None = None
        self._finished = False
        self._stdout = open(stdout_log, "a") if stdout_log else None

        mode = (
            mode
            or os.environ.get("TRAINSCOPE_MODE")
            or ("remote" if (url or os.environ.get("TRAINSCOPE_URL")) else "local")
        )
        self.mode = mode
        try:
            if mode == "remote":
                self.sink: Any = HttpSink(
                    url or os.environ["TRAINSCOPE_URL"], token or os.environ.get("TRAINSCOPE_TOKEN")
                )
            elif mode == "local":
                self.sink = LocalSink(db or os.environ.get("TRAINSCOPE_DB", "trainscope.duckdb"))
            else:
                self.sink = NullSink()
        except Exception as e:
            log.warning("trainscope: could not create %s sink (%s); telemetry disabled", mode, e)
            self.sink = NullSink()

        self._run_record: dict[str, Any] = {
            "id": self.id,
            "project": project,
            "name": self.name,
            "status": "running",
            "config": self.config,
            "env": _env_info(),
            "started_at": time.time(),
        }
        self._registered = False
        self._register()

        self.buffer = RingBuffer()
        spool = Path(spool_dir or os.environ.get("TRAINSCOPE_SPOOL") or Path.home() / ".trainscope" / "spool")
        self.flusher = Flusher(self.buffer, self, spool / f"{self.id}.jsonl", interval=flush_interval)
        self.flusher.start()
        self.sampler = None
        if system_interval:
            self.sampler = SystemSampler(lambda k, v: self._metric(k, "", v, self.step), system_interval)
            self.sampler.start()
        self.poller = None
        if poll_commands and not isinstance(self.sink, NullSink):
            self.poller = CommandPoller(self.sink, self.id, command_interval)
            self.poller.start()
        atexit.register(self.finish)

    # ---------------------------------------------------- sink proxy (retries run registration)
    def _register(self) -> None:
        try:
            self.sink.register_run(self._run_record)
            self._registered = True
        except Exception as e:
            log.debug("trainscope: run registration deferred (%s)", e)

    def send(self, batch: dict[str, list[dict[str, Any]]]) -> None:
        if not self._registered:
            self._register()
            if not self._registered:
                raise ConnectionError("run not registered yet")
        self.sink.send(batch)

    def _set_status(self, status: str) -> None:
        try:
            self.sink.set_status(self.id, status)
        except Exception:
            pass

    # ----------------------------------------------------------------- recording
    def _metric(self, name: str, layer: str, value: float, step: int | None = None) -> None:
        self.buffer.add_metric(
            {
                "run_id": self.id,
                "step": max(0, self.step if step is None else step),
                "ts": time.time(),
                "name": name,
                "layer": layer,
                "value": float(value),
            }
        )

    def event(self, type: str, payload: dict[str, Any] | None = None, step: int | None = None) -> None:
        self.buffer.add_event(
            {
                "run_id": self.id,
                "step": max(0, self.step if step is None else step),
                "ts": time.time(),
                "type": type,
                "payload": payload or {},
            }
        )
        self.flusher.wake()

    def _add_data_wait(self, seconds: float) -> None:
        self._data_wait += seconds

    def watch(
        self,
        model: Any,
        optimizer: Any = None,
        grad_every: int = 50,
        act_every: int = 200,
        scaler: Any = None,
        max_act_layers: int = 12,
    ) -> None:
        from .hooks import ModelWatcher

        self.model, self.optimizer, self.scaler = model, optimizer, scaler
        self.watcher = ModelWatcher(
            model, grad_every=grad_every, act_every=act_every, max_act_layers=max_act_layers
        )
        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._run_record["config"] = {
            **self.config,
            "_model": {
                "class": type(model).__name__,
                "params": n_params,
                "trainable_params": n_train,
                "layers": len(self.watcher.layer_names),
            },
        }
        if optimizer is not None:
            self._run_record["config"]["_optimizer"] = {
                "class": type(optimizer).__name__,
                **{k: v for k, v in optimizer.defaults.items() if isinstance(v, int | float | bool)},
            }
        self._register()

    def wrap_loader(self, loader: Iterable) -> TimedLoader:
        return TimedLoader(loader, self)

    def print(self, *args: Any) -> None:
        """print() that also tees to the run's stdout log (used for the logs-only eval condition)."""
        msg = " ".join(str(a) for a in args)
        print(msg)
        if self._stdout:
            self._stdout.write(msg + "\n")
            self._stdout.flush()

    def log(self, step: int, **scalars: Any) -> None:
        """Record per-step scalars. Call once per optimizer step, after optimizer.step()."""
        try:
            self._log(step, scalars)
        except Exception:  # telemetry must never break training
            log.exception("trainscope.log failed")

    def _log(self, step: int, scalars: dict[str, Any]) -> None:
        now = time.perf_counter()
        self.step = step
        for k, v in scalars.items():
            if v is None:
                continue
            v = float(v)
            self._metric(k, "", v)
            if k == "loss" and not math.isfinite(v) and not self._nan_reported:
                self._nan_reported = True
                where: dict[str, Any] = self.watcher.locate_nonfinite() if self.watcher else {}
                self.event("nonfinite_loss", {"value": str(v), **where})
        if self._last_t is not None:
            dt = now - self._last_t
            self._metric("step_time", "", dt)
            self._metric("data_wait", "", self._data_wait)
            self._metric("data_wait_frac", "", min(1.0, self._data_wait / dt) if dt > 0 else 0.0)
            bs = self.config.get("batch_size")
            if bs:
                self._metric("samples_per_sec", "", bs / dt if dt > 0 else 0.0)
        self._data_wait = 0.0
        if self.scaler is not None:
            self._log_amp()
        if self.watcher is not None:
            metrics, events = self.watcher.collect(step)
            for name, layer, val in metrics:
                self._metric(name, layer, val)
            for e in events:
                self.event(e["type"], e["payload"])
            self.watcher.before_step(step)
        if self.poller is not None:
            for cmd in self.poller.drain():
                status, result = execute(self, cmd)
                self.sink.ack_command(cmd["id"], status, result)
                self.event(f"command_{cmd['action']}", {"id": cmd["id"], "status": status, **result})
            if self._pause_requested:
                wait_while_paused(self, self.poller)
        self._last_t = time.perf_counter()

    def _log_amp(self) -> None:
        try:
            scale = float(self.scaler.get_scale())
        except Exception:
            return
        if self._last_scale is not None and scale < self._last_scale:
            self._amp_skipped += 1
        self._last_scale = scale
        self._metric("amp/loss_scale", "", scale)
        self._metric("amp/skipped_steps", "", self._amp_skipped)

    def log_batch(
        self,
        step: int,
        labels: Any = None,
        attention_mask: Any = None,
        sample_ids: Iterable[Any] | None = None,
        num_classes: int | None = None,
    ) -> None:
        """Data statistics: label distribution, padding ratio, a few sample ids. Sampled every data_every steps."""
        if step % self.data_every != 0:
            return
        try:
            import numpy as np

            if labels is not None:
                lab = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
                lab = lab.reshape(-1)
                lab = lab[lab >= 0]
                if lab.size:
                    k = num_classes or int(lab.max()) + 1
                    counts = np.bincount(lab.astype(int), minlength=k)[: max(k, 1)]
                    p = counts / counts.sum()
                    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / math.log(max(k, 2)))
                    self._metric("data/label_entropy", "", ent, step)
                    self._metric("data/majority_frac", "", float(p.max()), step)
                    if k <= 50:
                        for i, pi in enumerate(p):
                            self._metric("data/label_frac", str(i), float(pi), step)
            if attention_mask is not None:
                m = (
                    attention_mask.detach().float()
                    if hasattr(attention_mask, "detach")
                    else np.asarray(attention_mask)
                )
                pad = 1.0 - float(m.mean())
                self._metric("data/pad_ratio", "", pad, step)
            if sample_ids is not None:
                ids = [str(s) for s in list(sample_ids)[:8]]
                self.event("data_samples", {"ids": ids}, step)
        except Exception:
            log.debug("log_batch failed", exc_info=True)

    def log_eval(self, step: int, **metrics: Any) -> None:
        self.log_scalars(step, {f"eval/{k}": v for k, v in metrics.items()})

    def log_scalars(self, step: int, scalars: dict[str, Any]) -> None:
        """Record scalars at a step without affecting step-time measurement."""
        for k, v in scalars.items():
            if v is not None:
                self._metric(k, "", float(v), step)

    # ----------------------------------------------------------------- control
    def set_checkpoint_fn(self, fn: Callable[[str], None]) -> None:
        self._checkpoint_fn = fn

    def save_checkpoint(self, path: str | None = None) -> str:
        path = path or str(self.checkpoint_dir / self.id / f"step_{self.step}.pt")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self._checkpoint_fn is not None:
            self._checkpoint_fn(path)
        else:
            import torch

            state: dict[str, Any] = {"step": self.step}
            if self.model is not None:
                state["model"] = self.model.state_dict()
            if self.optimizer is not None:
                state["optimizer"] = self.optimizer.state_dict()
            torch.save(state, path)
        return path

    def flush(self) -> None:
        self.flusher.flush_now()

    def finish(self, status: str = "finished") -> None:
        if self._finished:
            return
        self._finished = True
        try:
            if self.sampler:
                self.sampler.stop()
            if self.poller:
                self.poller.stop()
            if self.watcher:
                self.watcher.remove()
            self.event("run_end", {"status": status})
            self.flusher.stop()
            self._set_status(status)
            if self._stdout:
                self._stdout.close()
        except Exception:
            log.exception("trainscope.finish failed")
