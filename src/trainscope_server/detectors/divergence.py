"""Divergence: NaN/Inf loss, or loss rising well above its best value.

The likely cause is picked from context: fp16 precision, gradient growth before the NaN, and whether gradient
clipping is configured.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import Detector, Finding, RunData, finite, smooth


def _clip_off(data: RunData) -> bool:
    return not data.cfg("grad_clip", "max_grad_norm", default=None)


def _nan_causes(data: RunData, grew: bool) -> list[str]:
    precision = str(data.cfg("precision", "dtype", default="")).lower()
    if precision in ("fp16", "float16", "half", "16"):
        return ["fp16_no_loss_scaling", "nan_from_op", "lr_too_high"]
    if grew:
        return (
            ["exploding_no_clip", "lr_too_high"] if _clip_off(data) else ["lr_too_high", "exploding_no_clip"]
        ) + ["missing_zero_grad"]
    return ["nan_from_op", "lr_too_high", "fp16_no_loss_scaling"]


class DivergenceDetector(Detector):
    name = "divergence"

    def __init__(self, rise_factor: float = 1.5):
        self.rise_factor = rise_factor

    def run(self, data: RunData) -> list[Finding]:
        loss = data.scalar("loss")
        out: list[Finding] = []
        if loss.empty:
            return out
        bad = loss[~np.isfinite(loss.to_numpy(dtype=float))]
        nan_events = data.events_of("nonfinite_loss")
        if len(bad) or nan_events:
            first = int(bad.index[0]) if len(bad) else int(nan_events[0]["step"])
            payload = nan_events[0]["payload"] if nan_events else {}
            fl = finite(loss)
            pre = fl[fl.index < first]
            grad = finite(data.scalar("grad/global_norm"))
            gpre = grad[grad.index < first]
            g_before = float(gpre.iloc[-1]) if len(gpre) else None
            g_med = float(gpre.iloc[: max(1, len(gpre) // 2)].median()) if len(gpre) else None
            grew = bool(g_before and g_med and g_before > 10 * g_med)
            ev: dict[str, Any] = {
                "first_nonfinite_step": first,
                "last_finite_loss": float(pre.iloc[-1]) if len(pre) else None,
                "loss_at_start": float(fl.iloc[0]) if len(fl) else None,
                "nonfinite_location": payload or None,
                "grad_norm_before": g_before,
                "grad_norm_early_median": g_med,
                "grad_grew_before_nan": grew,
                "grad_clip": data.cfg("grad_clip"),
                "precision": data.cfg("precision"),
            }
            nf_grad = data.events_of("nonfinite_grad")
            if nf_grad:
                ev["first_nonfinite_grad_step"] = int(nf_grad[0]["step"])
                ev["nonfinite_grad_layer"] = nf_grad[0]["payload"].get("first_layer")
            where = f" (first non-finite in {payload.get('layer')})" if payload.get("layer") else ""
            how = "after gradients grew sharply" if grew else "without prior gradient growth"
            out.append(
                Finding(
                    self.name,
                    "critical",
                    first,
                    data.max_step,
                    f"loss became NaN/Inf at step {first}{where}, {how}",
                    ev,
                    _nan_causes(data, grew),
                    0.9,
                )
            )
            return out
        fl = finite(loss)
        if len(fl) < self.min_steps:
            return out
        sm = smooth(fl)
        best_step = int(sm.idxmin())
        best = float(sm.min())
        last = float(sm.iloc[-1])
        start = float(sm.iloc[: max(3, len(sm) // 20)].mean())
        causes = (
            ["exploding_no_clip", "lr_too_high"] if _clip_off(data) else ["lr_too_high", "missing_warmup"]
        )
        if best > 0 and last > best * self.rise_factor and last > start * 0.9 and best_step < sm.index[-1]:
            rose = (last / best - 1) * 100
            out.append(
                Finding(
                    self.name,
                    "critical",
                    best_step,
                    data.max_step,
                    f"loss rose {rose:.0f}% from its best ({best:.3g} at step {best_step}) to {last:.3g}",
                    {"best_loss": best, "best_step": best_step, "final_loss": last, "initial_loss": start},
                    causes,
                    0.7,
                )
            )
        elif start > 0 and math.isfinite(last) and last > start * 1.2 and len(fl) > 50:
            out.append(
                Finding(
                    self.name,
                    "warning",
                    int(fl.index[0]),
                    data.max_step,
                    f"loss ended higher than it started ({start:.3g} -> {last:.3g})",
                    {"initial_loss": start, "final_loss": last},
                    causes,
                    0.5,
                )
            )
        return out
