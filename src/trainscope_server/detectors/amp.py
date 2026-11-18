"""Mixed-precision overflow: repeated loss-scale reductions / skipped steps, or fp16 without a scaler."""

from __future__ import annotations

from .base import Detector, Finding, RunData, finite


class AmpOverflowDetector(Detector):
    name = "amp_overflow"
    # mixed-precision thresholds
    defaults = {
        "skipped_frac": 0.1,  # share of steps skipped for overflow
        "min_scale": 1.0,  # loss scale collapsing below this
        "underflow_zero_frac": 0.5,  # fp16 without a scaler: median share of exactly-zero gradient entries
    }

    def run(self, data: RunData) -> list[Finding]:
        out: list[Finding] = []
        sc = finite(data.scalar("amp/loss_scale"))
        sk = finite(data.scalar("amp/skipped_steps"))
        if len(sc) >= 5 and len(sk):
            skipped = float(sk.iloc[-1])
            frac = skipped / max(len(sc), 1)
            if frac > self.skipped_frac or float(sc.min()) < self.min_scale:
                out.append(
                    Finding(
                        self.name,
                        "critical",
                        int(sc.index[0]),
                        int(sc.index[-1]),
                        f"loss scale collapsed to {float(sc.min()):.3g}; {int(skipped)} overflow-skipped steps "
                        f"({frac:.0%})",
                        {
                            "min_loss_scale": float(sc.min()),
                            "skipped_steps": skipped,
                            "skipped_fraction": frac,
                        },
                        ["fp16_no_loss_scaling", "lr_too_high"],
                        0.7,
                    )
                )
        precision = str(data.cfg("precision", "dtype", default="")).lower()
        if precision in ("fp16", "float16", "16", "half") and not len(sc):
            loss = data.scalar("loss")
            nonfinite = (
                int((~loss.apply(lambda v: v == v and abs(v) != float("inf"))).sum()) if len(loss) else 0
            )
            zero_g = (
                finite(data.layers("grad/zero_frac").stack())
                if not data.layers("grad/zero_frac").empty
                else None
            )
            zf = float(zero_g.median()) if zero_g is not None and len(zero_g) else None
            if nonfinite or (zf is not None and zf > self.underflow_zero_frac):
                out.append(
                    Finding(
                        "fp16_unscaled",
                        "critical",
                        int(loss.index[0]) if len(loss) else 0,
                        data.max_step,
                        "fp16 training with no loss scaler: "
                        + (
                            f"{nonfinite} non-finite losses"
                            if nonfinite
                            else f"{zf:.0%} of gradient entries are zero (underflow)"
                        ),
                        {"precision": precision, "nonfinite_losses": nonfinite, "median_grad_zero_frac": zf},
                        ["fp16_no_loss_scaling"],
                        0.8,
                    )
                )
        return out
