"""Detector base types and the RunData view detectors read from."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

SEVERITY_RANK = {"critical": 3, "warning": 2, "info": 1}


@dataclass
class Finding:
    detector: str
    severity: str
    step_start: int
    step_end: int
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    suspected_causes: list[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self, run_id: str) -> dict[str, Any]:
        key = f"{run_id}:{self.detector}:{self.summary.split(':')[0]}"
        return {
            "id": hashlib.sha1(key.encode()).hexdigest()[:12],
            "run_id": run_id,
            "detector": self.detector,
            "severity": self.severity,
            "step_start": int(self.step_start),
            "step_end": int(self.step_end),
            "summary": self.summary,
            "evidence": {
                **_jsonable(self.evidence),
                "suspected_causes": self.suspected_causes,
                "confidence": round(self.confidence, 2),
            },
        }


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating | float):
        v = float(x)
        return None if not np.isfinite(v) else round(v, 6)
    return x


class RunData:
    """Lazy, cached access to one run's telemetry as pandas objects."""

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
        upto_step: int | None = None,
    ):
        if upto_step is not None and len(df):
            df = df[df["step"] <= upto_step]
            events = [e for e in (events or []) if e["step"] <= upto_step]
        self.df = df
        self.config = config or {}
        self.events = events or []

    @classmethod
    def from_store(cls, store: Any, run_id: str, upto_step: int | None = None) -> RunData:
        run = store.get_run(run_id) or {}
        return cls(store.metrics_df(run_id), run.get("config"), store.events(run_id), upto_step)

    @cached_property
    def _by_name(self) -> dict[str, pd.DataFrame]:
        return {n: g for n, g in self.df.groupby("name")} if len(self.df) else {}

    def has(self, name: str) -> bool:
        return name in self._by_name

    def scalar(self, name: str) -> pd.Series:
        g = self._by_name.get(name)
        if g is None:
            return pd.Series(dtype=float)
        g = g[g["layer"] == ""]
        return g.groupby("step")["value"].last().sort_index()

    def layers(self, name: str) -> pd.DataFrame:
        """steps x layers table."""
        g = self._by_name.get(name)
        if g is None:
            return pd.DataFrame()
        return g.pivot_table(index="step", columns="layer", values="value", aggfunc="last").sort_index()

    def events_of(self, type_: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == type_]

    @property
    def max_step(self) -> int:
        return int(self.df["step"].max()) if len(self.df) else -1

    def cfg(self, *keys: str, default: Any = None) -> Any:
        for k in keys:
            if k in self.config and self.config[k] is not None:
                return self.config[k]
        return default


class Detector:
    name = "base"
    min_steps = 20

    def run(self, data: RunData) -> list[Finding]:
        raise NotImplementedError


# ------------------------------------------------------------------ numeric helpers
def smooth(s: pd.Series, frac: float = 0.05, min_w: int = 5) -> pd.Series:
    w = max(min_w, int(len(s) * frac))
    return s.rolling(w, min_periods=1, center=False).median()


def robust_z(s: pd.Series, window: int = 50) -> pd.Series:
    med = s.rolling(window, min_periods=10).median().shift(1)
    mad = (s - med).abs().rolling(window, min_periods=10).median().shift(1)
    return (s - med) / (1.4826 * mad + 1e-12)


def trend(s: pd.Series) -> tuple[float, float]:
    """(slope per step, pearson r) of a series vs its step index."""
    s = s.dropna()
    if len(s) < 3:
        return 0.0, 0.0
    x = s.index.to_numpy(dtype=float)
    y = s.to_numpy(dtype=float)
    if np.std(y) == 0 or np.std(x) == 0:
        return 0.0, 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    r = float(np.corrcoef(x, y)[0, 1])
    return slope, r


def finite(s: pd.Series) -> pd.Series:
    return s[np.isfinite(s.to_numpy(dtype=float))]
