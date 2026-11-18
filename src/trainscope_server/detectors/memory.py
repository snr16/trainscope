"""Memory leak: steady growth of process / device memory across steps."""

from __future__ import annotations

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

    def __init__(self, min_growth_mb: float = 100.0, min_r: float = 0.5):
        # dev set: non-leaking runs grew their RSS floor by <=29 MB (9%), leaks by >=138 MB (33%), often
        # in steps (r ~0.6), so the size of the growth does the work and the trend bar stays loose
        self.min_growth_mb, self.min_r = min_growth_mb, min_r

    def run(self, data: RunData) -> list[Finding]:
        for name in _SERIES:
            s = finite(data.scalar(name))
            if len(s) < 16:
                continue
            s = s.iloc[len(s) // 10 :]  # ignore warm-up allocations
            w = max(5, len(s) // 8)
            floor = s.rolling(w).min().dropna()
            if len(floor) < 4:
                continue
            start, end = float(floor.iloc[0]), float(floor.iloc[-1])
            growth = end - start
            slope, r = trend(floor)
            if growth > self.min_growth_mb and growth > 0.2 * max(start, 1.0) and r > self.min_r:
                return [
                    Finding(
                        self.name,
                        "critical" if growth > 0.4 * start else "warning",
                        int(floor.index[0]),
                        int(floor.index[-1]),
                        f"{name} floor grew steadily by {growth:.0f} MB ({start:.0f} -> {end:.0f}, r={r:.2f}, "
                        f"{slope * 1000:.0f} MB per 1k steps)",
                        {
                            "series": name,
                            "floor_start_mb": start,
                            "floor_end_mb": end,
                            "peak_mb": float(s.max()),
                            "slope_mb_per_step": slope,
                            "trend_r": r,
                        },
                        ["memory_leak"],
                        0.8,
                    )
                ]
        return []
