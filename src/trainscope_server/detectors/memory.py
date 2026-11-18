"""Memory leak: steady growth of process / device memory across steps."""

from __future__ import annotations

import numpy as np

from .base import Detector, Finding, RunData, finite, trend

_SERIES = ["sys/cuda_alloc_mb", "sys/mps_alloc_mb", "sys/gpu_mem_mb", "sys/proc_rss_mb", "mem/tracked_mb"]


def memory_floor(s):
    """Rolling-minimum floor of a memory series after warm-up: (floor series, start, end, trend r)."""
    s = s.iloc[len(s) // 10 :]
    w = max(5, len(s) // 8)
    floor = s.rolling(w).min().dropna()
    if len(floor) < 4:
        return None
    return floor, float(floor.iloc[0]), float(floor.iloc[-1]), trend(floor)[1]


class MemoryLeakDetector(Detector):
    """Steady growth of memory. Allocator stats (CUDA/MPS) are exact; process RSS is noisy (other processes,
    page compression), so for RSS we follow its *floor* (rolling minimum), which only rises if memory leaks."""

    name = "memory_leak"

    # Thresholds are relative to the run itself, so they hold on a laptop and on an 80 GB GPU alike.
    # Dev set: leaks grew their floor 33-58% of the starting memory, non-leaking runs at most 9%. Growth relative to
    # the run's own memory noise does NOT separate them on its own (leaks make memory noisier: 2.9-7x noise, while
    # a very stable healthy run reached 69x), so noise is only a guard against jitter faking a trend.
    defaults = {
        "min_growth_frac": 0.2,  # memory floor must grow at least this fraction of its starting value
        "noise_x": 2.0,  # ... and more than this many times the run's early memory noise (robust std)
        "min_growth_mb": 16.0,  # ... and at least this many MB (ignores trivially small changes)
        "min_r": 0.5,  # ... with at least this upward trend (correlation with step)
        "critical_frac": 0.4,  # growth above this fraction of the start = critical
        "min_samples": 30,  # memory samples needed (~30 s at 1 Hz); shorter runs are dominated by start-up allocation
    }

    def run(self, data: RunData) -> list[Finding]:
        for name in _SERIES:
            s = finite(data.scalar(name))
            if len(s) < self.min_samples:
                continue
            s = s.iloc[len(s) // 10 :]  # ignore warm-up allocations
            w = max(5, len(s) // 8)
            floor = s.rolling(w).min().dropna()
            if len(floor) < 4:
                continue
            start, end = float(floor.iloc[0]), float(floor.iloc[-1])
            growth = end - start
            slope, r = trend(floor)
            early = s.iloc[: max(8, len(s) // 4)]
            noise = float(np.median(np.abs(early - early.median()))) * 1.4826  # robust std, early in the run
            if (
                growth > self.min_growth_frac * max(start, 1.0)
                and growth > self.noise_x * noise
                and growth > self.min_growth_mb
                and r > self.min_r
            ):
                return [
                    Finding(
                        self.name,
                        "critical" if growth > self.critical_frac * start else "warning",
                        int(floor.index[0]),
                        int(floor.index[-1]),
                        f"{name} floor grew steadily by {growth:.0f} MB ({start:.0f} -> {end:.0f}, "
                        f"{growth / max(start, 1.0):.0%} of its start, r={r:.2f}, "
                        f"{slope * 1000:.0f} MB per 1k steps)",
                        {
                            "series": name,
                            "floor_start_mb": start,
                            "floor_end_mb": end,
                            "growth_frac": growth / max(start, 1.0),
                            "early_noise_mb": noise,
                            "growth_over_noise": growth / max(noise, 1e-9),
                            "peak_mb": float(s.max()),
                            "slope_mb_per_step": slope,
                            "trend_r": r,
                        },
                        ["memory_leak"],
                        0.8,
                    )
                ]
        return []
