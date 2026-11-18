"""Synthetic telemetry builders for detector / summarizer tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainscope_server.detectors import RunData


def frame(
    series: dict[str, list[float] | np.ndarray],
    steps=None,
    layers: dict[str, dict[str, list[float]]] | None = None,
    every: int = 1,
) -> pd.DataFrame:
    rows = []
    for name, vals in series.items():
        st = steps if steps is not None else range(0, len(vals) * every, every)
        rows += [
            {"step": s, "ts": float(s), "name": name, "layer": "", "value": float(v)}
            for s, v in zip(st, vals, strict=False)
        ]
    for name, per_layer in (layers or {}).items():
        for layer, vals in per_layer.items():
            rows += [
                {"step": i * every, "ts": float(i), "name": name, "layer": layer, "value": float(v)}
                for i, v in enumerate(vals)
            ]
    return pd.DataFrame(rows, columns=["step", "ts", "name", "layer", "value"])


def data(series=None, layers=None, config=None, events=None, every=1) -> RunData:
    return RunData(frame(series or {}, layers=layers, every=every), config or {}, events or [])


def healthy_loss(n: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 2.3 * np.exp(-np.arange(n) / 80) + 0.3 + 0.03 * rng.normal(size=n)
