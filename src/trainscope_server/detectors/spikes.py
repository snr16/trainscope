"""Loss spikes: rolling robust z-score."""

from __future__ import annotations

from .base import Detector, Finding, RunData, finite, robust_z


class LossSpikeDetector(Detector):
    name = "loss_spikes"

    def __init__(self, z: float = 8.0, window: int = 50):
        self.z, self.window = z, window

    def run(self, data: RunData) -> list[Finding]:
        loss = finite(data.scalar("loss"))
        out = self._early_blowup(data, loss)
        if len(loss) < self.window:
            return out
        z = robust_z(loss, self.window)
        spikes = z[z > self.z]
        # require a meaningful relative jump too (avoid flagging tiny noise on very flat curves)
        med = loss.rolling(self.window, min_periods=10).median().shift(1)
        spikes = spikes[(loss[spikes.index] > 1.5 * med[spikes.index])]
        if spikes.empty:
            return out
        top = spikes.sort_values(ascending=False).head(5)
        sev = "warning" if len(spikes) < 5 else "critical"
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
                0.5,
            )
        ]

    @staticmethod
    def _early_blowup(data: RunData, loss) -> list[Finding]:
        """Loss jumps far above its starting value early in training: classic too-high LR / missing warmup."""
        if len(loss) < 20:
            return []
        start = float(loss.iloc[:3].median())
        early = loss.iloc[: max(10, len(loss) // 5)]
        peak = float(early.max())
        if start <= 0 or peak < 2 * start:
            return []
        lr = finite(data.scalar("lr"))
        base = data.cfg("lr", "learning_rate")
        warm = int(data.cfg("warmup_steps", default=0) or 0)
        no_warmup = bool(warm and len(lr) and base and float(lr.iloc[0]) > 0.5 * float(base))
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
                0.75,
            )
        ]
