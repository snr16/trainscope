"""Loss spikes: rolling robust z-score."""

from __future__ import annotations

from .base import Detector, Finding, RunData, finite, robust_z


class LossSpikeDetector(Detector):
    name = "loss_spikes"

    # spike and early-blow-up thresholds
    defaults = {
        "z": 8.0,  # robust z-score of a spike against the rolling median
        "window": 50,  # rolling window in logged steps
        "min_jump": 1.5,  # a spike must also be this many x the rolling median
        "critical_count": 5,  # this many spikes or more = critical
        "early_blowup": 2.0,  # loss peak in the first 20% of the run this many x the step-0 loss
        "warmup_lr_frac": 0.5,  # LR at step 0 above this fraction of the peak = warmup missing
        "conf_early_instability": 0.75,  # ranking weight: loss blew up early
        "conf_spikes": 0.5,  # ranking weight: loss spikes
    }

    def run(self, data: RunData) -> list[Finding]:
        loss = finite(data.scalar("loss"))
        out = self._early_blowup(data, loss)
        if len(loss) < int(self.window):
            return out
        z = robust_z(loss, int(self.window))
        spikes = z[z > self.z]
        # require a meaningful relative jump too (avoid flagging tiny noise on very flat curves)
        med = loss.rolling(int(self.window), min_periods=10).median().shift(1)
        spikes = spikes[(loss[spikes.index] > self.min_jump * med[spikes.index])]
        if spikes.empty:
            return out
        top = spikes.sort_values(ascending=False).head(5)
        sev = "warning" if len(spikes) < self.critical_count else "critical"
        return out + [
            Finding(
                self.name,
                sev,
                int(spikes.index[0]),
                int(spikes.index[-1]),
                f"{len(spikes)} loss spike(s); largest at step {int(top.index[0])} "
                f"({loss[top.index[0]]:.3g} vs rolling median {med[top.index[0]]:.3g})",
                {
                    "n_spikes": len(spikes),
                    "spike_steps": [int(s) for s in top.index],
                    "spike_values": [float(loss[s]) for s in top.index],
                },
                ["lr_too_high", "missing_warmup", "exploding_no_clip"],
                self.conf_spikes,
            )
        ]

    def _early_blowup(self, data: RunData, loss) -> list[Finding]:
        """Loss jumps far above its starting value early in training: classic too-high LR / missing warmup."""
        if len(loss) < 20:
            return []
        start = float(loss.iloc[:3].median())
        early = loss.iloc[: max(10, len(loss) // 5)]
        peak = float(early.max())
        if start <= 0 or peak < self.early_blowup * start:
            return []
        lr = finite(data.scalar("lr"))
        base = data.cfg("lr", "learning_rate")
        warm = int(data.cfg("warmup_steps", default=0) or 0)
        no_warmup = bool(warm and len(lr) and base and float(lr.iloc[0]) > self.warmup_lr_frac * float(base))
        causes = ["missing_warmup", "lr_too_high"] if no_warmup else ["lr_too_high", "missing_warmup"]
        return [
            Finding(
                "early_instability",
                "critical",
                int(early.index[0]),
                int(early.idxmax()),
                f"loss blew up early: {start:.3g} at the start -> {peak:.3g} at step {int(early.idxmax())}",
                {
                    "start_loss": start,
                    "early_peak_loss": peak,
                    "peak_step": int(early.idxmax()),
                    "warmup_configured": warm,
                    "lr_at_step0": float(lr.iloc[0]) if len(lr) else None,
                    "base_lr": base,
                },
                causes,
                self.conf_early_instability,
            )
        ]
