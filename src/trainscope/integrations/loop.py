"""Helpers for plain PyTorch training loops."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from typing import Any

import trainscope as ts


@contextlib.contextmanager
def tracked_run(project: str, config: dict[str, Any] | None = None, **kwargs: Any) -> Iterator[ts.Run]:
    """``with tracked_run("proj", cfg) as run:`` — marks the run failed if the body raises."""
    run = ts.init(project, config=config, **kwargs)
    try:
        yield run
    except BaseException as e:
        run.event("exception", {"type": type(e).__name__, "message": str(e)[:500]})
        run.finish("crashed")
        raise
    else:
        run.finish("finished")


class StepTimer:
    """Measures pure compute time per step (complements the data-loader wait time)."""

    def __init__(self) -> None:
        self.t0 = 0.0
        self.last = 0.0

    def __enter__(self) -> StepTimer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.last = time.perf_counter() - self.t0
