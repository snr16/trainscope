"""TrainScope SDK.

import trainscope as ts
run = ts.init(project="cifar", config=cfg)
ts.watch(model, optimizer=opt, grad_every=50, act_every=200)
loader = ts.wrap_loader(train_loader)
for step, batch in enumerate(loader):
    loss = train_step(batch)
    ts.log(step, loss=loss.item(), lr=sched.get_last_lr()[0])
ts.finish()
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .run import Run

__version__ = "0.1.0"
__all__ = [
    "Run",
    "init",
    "watch",
    "wrap_loader",
    "log",
    "log_batch",
    "log_eval",
    "log_scalars",
    "event",
    "finish",
    "get_run",
]

_run: Run | None = None


def init(project: str, config: dict[str, Any] | None = None, **kwargs: Any) -> Run:
    global _run
    if _run is not None and not _run._finished:
        _run.finish()
    _run = Run(project, config=config, **kwargs)
    return _run


def get_run() -> Run:
    if _run is None:
        raise RuntimeError("call trainscope.init() first")
    return _run


def watch(model: Any, **kwargs: Any) -> None:
    get_run().watch(model, **kwargs)


def wrap_loader(loader: Iterable) -> Iterable:
    return get_run().wrap_loader(loader)


def log(step: int, **scalars: Any) -> None:
    if _run is not None:
        _run.log(step, **scalars)


def log_batch(step: int, **kwargs: Any) -> None:
    if _run is not None:
        _run.log_batch(step, **kwargs)


def log_eval(step: int, **metrics: Any) -> None:
    if _run is not None:
        _run.log_eval(step, **metrics)


def log_scalars(step: int, scalars: dict[str, Any]) -> None:
    if _run is not None:
        _run.log_scalars(step, scalars)


def event(type: str, payload: dict[str, Any] | None = None) -> None:
    if _run is not None:
        _run.event(type, payload)


def finish(status: str = "finished") -> None:
    if _run is not None:
        _run.finish(status)
