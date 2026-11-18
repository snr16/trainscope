"""Wrap an iterable data loader to measure how long training waits for each batch."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import Any


class TimedLoader:
    def __init__(self, loader: Iterable, run: Any):
        self.loader = loader
        self.run = run

    def __iter__(self) -> Iterator:
        it = iter(self.loader)
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                return
            self.run._add_data_wait(time.perf_counter() - t0)
            yield batch

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def __getattr__(self, item: str) -> Any:
        return getattr(self.loader, item)
