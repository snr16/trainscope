"""Loss spikes: rolling robust z-score."""

from __future__ import annotations

from .base import Detector, Finding, RunData, finite, robust_z


class LossSpikeDetector(Detector):
    name = "loss_spikes"

    def __init__(self, z: float = 8.0, window: int = 50):
        self.z, self.window = z, window

    def run(self, data: RunData) -> list[Finding]:
        loss = finite(data.scalar("loss"))
        if len(loss) < self.window:
            return []
        z = robust_z(loss, self.window)
        spikes = z[z > self.z]
        # require a meaningful relative jump too (avoid flagging tiny noise on very flat curves)
        med = loss.rolling(self.window, min_periods=10).median().shift(1)
        spikes = spikes[(loss[spikes.index] > 1.5 * med[spikes.index])]
        if spikes.empty:
            return []
        top = spikes.sort_values(ascending=False).head(5)
        sev = "warning" if len(spikes) < 5 else "critical"
        return [
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
