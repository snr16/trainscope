"""Divergence: NaN/Inf loss, or loss rising well above its best value."""

from __future__ import annotations

import math

import numpy as np

from .base import Detector, Finding, RunData, finite, smooth


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
            grad = data.scalar("grad/global_norm")
            gpre = finite(grad[grad.index < first])
            ev = {
                "first_nonfinite_step": first,
                "last_finite_loss": float(pre.iloc[-1]) if len(pre) else None,
                "loss_at_start": float(fl.iloc[0]) if len(fl) else None,
                "nonfinite_location": payload or None,
                "grad_norm_before": float(gpre.iloc[-1]) if len(gpre) else None,
                "grad_norm_median": float(gpre.median()) if len(gpre) else None,
            }
            nf_grad = data.events_of("nonfinite_grad")
            if nf_grad:
                ev["first_nonfinite_grad_step"] = int(nf_grad[0]["step"])
                ev["nonfinite_grad_layer"] = nf_grad[0]["payload"].get("first_layer")
            causes = ["lr_too_high", "nan_from_op", "fp16_no_loss_scaling", "exploding_no_clip"]
            if (
                ev["grad_norm_before"]
                and ev["grad_norm_median"]
                and ev["grad_norm_before"] < 10 * ev["grad_norm_median"]
                and first < 0.1 * max(data.max_step, 1)
            ):
                causes = ["nan_from_op", "fp16_no_loss_scaling", "lr_too_high"]
            where = f" (first non-finite in {payload.get('layer')})" if payload.get("layer") else ""
            out.append(
                Finding(
                    self.name,
                    "critical",
                    first,
                    data.max_step,
                    f"loss became NaN/Inf at step {first}{where}",
                    ev,
                    causes,
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
                    ["lr_too_high", "missing_warmup", "exploding_no_clip"],
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
                    ["lr_too_high"],
                    0.5,
                )
            )
        return out
