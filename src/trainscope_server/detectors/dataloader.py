"""Data-loader bottleneck: training spends much of each step waiting for data."""

from __future__ import annotations

from .base import Detector, Finding, RunData, finite


class DataLoaderDetector(Detector):
    name = "dataloader_bottleneck"

    # data-loader thresholds
    defaults = {
        "wait_frac": 0.3,  # median share of each step spent waiting for data
        "critical_wait": 0.5,  # above this share = critical
        "conf_bottleneck": 0.85,  # ranking weight: training waits on the data loader
    }

    def run(self, data: RunData) -> list[Finding]:
        w = finite(data.scalar("data_wait_frac"))
        if len(w) < 10:
            return []
        w = w.iloc[min(5, len(w) // 10) :]  # skip worker start-up
        med = float(w.median())
        if med < self.wait_frac:
            return []
        gpu = finite(data.scalar("sys/gpu_util"))
        st = finite(data.scalar("step_time"))
        ev = {
            "median_data_wait_frac": med,
            "median_step_time_s": float(st.median()) if len(st) else None,
            "median_gpu_util": float(gpu.median()) if len(gpu) else None,
            "num_workers": data.cfg("num_workers"),
        }
        msg = f"training waits on the data loader for {med:.0%} of each step"
        if ev["median_gpu_util"] is not None:
            msg += f" (GPU util median {ev['median_gpu_util']:.0f}%)"
        return [
            Finding(
                self.name,
                "critical" if med > self.critical_wait else "warning",
                int(w.index[0]),
                int(w.index[-1]),
                msg,
                ev,
                ["dataloader_bottleneck"],
                self.conf_bottleneck,
            )
        ]
