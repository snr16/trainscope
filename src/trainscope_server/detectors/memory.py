"""Memory leak: steady growth of process / device memory across steps."""

from __future__ import annotations

from .base import Detector, Finding, RunData, finite, trend

_SERIES = ["sys/cuda_alloc_mb", "sys/mps_alloc_mb", "sys/gpu_mem_mb", "sys/proc_rss_mb", "mem/tracked_mb"]


class MemoryLeakDetector(Detector):
    name = "memory_leak"

    def __init__(self, min_growth_mb: float = 30.0, min_r: float = 0.9):
        self.min_growth_mb, self.min_r = min_growth_mb, min_r

    def run(self, data: RunData) -> list[Finding]:
        for name in _SERIES:
            s = finite(data.scalar(name))
            if len(s) < 8:
                continue
            s = s.iloc[len(s) // 10 :]  # ignore warm-up allocations
            growth = float(s.iloc[-1] - s.iloc[0])
            slope, r = trend(s)
            if growth > self.min_growth_mb and r > self.min_r and growth > 0.15 * max(float(s.iloc[0]), 1.0):
                return [
                    Finding(
                        self.name,
                        "critical" if growth > float(s.iloc[0]) else "warning",
                        int(s.index[0]),
                        int(s.index[-1]),
                        f"{name} grew steadily by {growth:.0f} MB ({float(s.iloc[0]):.0f} -> {float(s.iloc[-1]):.0f}, "
                        f"r={r:.2f}, {slope * 1000:.1f} MB per 1k steps)",
                        {
                            "series": name,
                            "start_mb": float(s.iloc[0]),
                            "end_mb": float(s.iloc[-1]),
                            "slope_mb_per_step": slope,
                            "trend_r": r,
                        },
                        ["memory_leak"],
                        0.8,
                    )
                ]
        return []
