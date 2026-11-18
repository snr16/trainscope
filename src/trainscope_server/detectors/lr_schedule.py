"""LR schedule anomaly: logged learning rate doesn't match the configured schedule."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .base import Detector, Finding, RunData, finite


def expected_lr(
    steps: np.ndarray,
    base_lr: float,
    schedule: str,
    total: int,
    warmup: int,
    min_lr: float = 0.0,
    step_size: int | None = None,
    gamma: float = 0.1,
) -> np.ndarray:
    s = steps.astype(float)
    lr = np.full_like(s, base_lr)
    if schedule in ("linear", "linear_decay"):
        prog = np.clip((s - warmup) / max(total - warmup, 1), 0, 1)
        lr = min_lr + (base_lr - min_lr) * (1 - prog)
    elif schedule == "cosine":
        prog = np.clip((s - warmup) / max(total - warmup, 1), 0, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(math.pi * prog))
    elif schedule == "step" and step_size:
        lr = base_lr * gamma ** np.floor(s / step_size)
    if warmup > 0:
        lr = np.where(s < warmup, base_lr * (s + 1) / warmup, lr)
    return lr


class LRScheduleDetector(Detector):
    name = "lr_schedule"

    def __init__(self, tol: float = 0.25):
        self.tol = tol

    def run(self, data: RunData) -> list[Finding]:
        lr = finite(data.scalar("lr"))
        if len(lr) < 10:
            return []
        base = data.cfg("lr", "learning_rate")
        sched = str(data.cfg("scheduler", "lr_schedule", default="constant")).lower()
        total = int(data.cfg("total_steps", "max_steps", default=0) or 0)
        warmup = int(data.cfg("warmup_steps", default=0) or 0)
        out: list[Finding] = []
        if base is None:
            return out
        base = float(base)
        if not total:
            total = int(lr.index.max()) + 1
        exp = pd.Series(
            expected_lr(
                lr.index.to_numpy(),
                base,
                sched,
                total,
                warmup,
                float(data.cfg("min_lr", default=0.0) or 0.0),
                data.cfg("step_size"),
                float(data.cfg("gamma", default=0.1) or 0.1),
            ),
            index=lr.index,
        )
        rel = (lr - exp).abs() / (exp.abs() + base * 1e-3)
        bad = rel[rel > self.tol]
        if len(bad) / len(rel) < 0.2:
            return out
        n_unique = lr.round(12).nunique()
        observed = {
            "lr_first": float(lr.iloc[0]),
            "lr_max": float(lr.max()),
            "lr_last": float(lr.iloc[-1]),
            "distinct_values": int(n_unique),
        }
        expected = {
            "schedule": sched,
            "base_lr": base,
            "warmup_steps": warmup,
            "total_steps": total,
            "lr_first": float(exp.iloc[0]),
            "lr_last": float(exp.iloc[-1]),
        }
        if warmup and lr.iloc[0] > 0.5 * base:
            reason, causes = (
                "no warmup: LR starts at its peak although warmup is configured",
                ["missing_warmup"],
            )
        elif sched != "constant" and n_unique <= max(3, len(lr) // 50):
            reason = f"LR changes in only {n_unique} discrete step(s); scheduler likely stepped per epoch, not per step"
            causes = ["scheduler_misstep"]
        elif (
            sched != "constant"
            and float(lr.iloc[-1]) < 0.01 * float(exp.iloc[len(exp) // 2])
            and lr.index[-1] < total * 0.6
        ):
            reason, causes = "LR decayed far faster than configured", ["scheduler_misstep"]
        elif float(lr.max()) > 1.5 * base:
            reason, causes = (
                f"LR peaks at {float(lr.max()):.3g}, above configured {base:.3g}",
                ["lr_too_high"],
            )
        else:
            reason, causes = (
                "logged LR deviates from configured schedule",
                ["scheduler_misstep", "missing_warmup"],
            )
        out.append(
            Finding(
                self.name,
                "warning",
                int(bad.index[0]),
                int(bad.index[-1]),
                f"LR schedule anomaly: {reason} ({len(bad) / len(rel):.0%} of steps off by >{self.tol:.0%})",
                {"observed": observed, "expected": expected, "mismatch_fraction": len(bad) / len(rel)},
                causes,
                0.7,
            )
        )
        return out
