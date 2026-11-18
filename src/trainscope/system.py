"""Background sampler for GPU / CPU / memory metrics (~1 Hz)."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

import psutil


def _load_libproc() -> Any:
    if sys.platform != "darwin":
        return None
    try:
        return ctypes.CDLL("/usr/lib/libproc.dylib")
    except OSError:
        return None


_LIBPROC = _load_libproc()


def _footprint_mb() -> float | None:
    """macOS physical footprint (what Activity Monitor shows). Unlike RSS it includes compressed and swapped
    pages, so a leak stays visible when the system is under memory pressure and squeezes the process's RSS."""
    if _LIBPROC is None:
        return None
    buf = (ctypes.c_uint64 * 64)()
    if _LIBPROC.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf)) != 0:  # RUSAGE_INFO_V2
        return None
    return buf[2 + 7] / 2**20  # after the 16-byte uuid: ri_phys_footprint is the 8th uint64 field


class SystemSampler(threading.Thread):
    def __init__(self, emit: Callable[[str, float], None], interval: float = 1.0):
        super().__init__(name="trainscope-system", daemon=True)
        self.emit = emit
        self.interval = interval
        self._halt = threading.Event()
        self.proc = psutil.Process(os.getpid())
        self._nvml = None
        self._handle = None
        try:
            import pynvml  # provided by nvidia-ml-py

            pynvml.nvmlInit()
            idx = int(os.environ.get("LOCAL_RANK", 0))
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._nvml = pynvml
        except Exception:
            pass
        self.proc.cpu_percent(None)

    def sample(self) -> dict[str, float]:
        out = {
            "sys/cpu": psutil.cpu_percent(None),
            "sys/ram_pct": psutil.virtual_memory().percent,
            "sys/proc_rss_mb": self.proc.memory_info().rss / 2**20,
        }
        fp = _footprint_mb()
        if fp is not None:
            out["sys/proc_footprint_mb"] = fp
        if self._nvml is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                out["sys/gpu_util"] = float(util.gpu)
                out["sys/gpu_mem_mb"] = mem.used / 2**20
            except Exception:
                pass
        try:
            import torch

            if torch.cuda.is_available():
                out["sys/cuda_alloc_mb"] = torch.cuda.memory_allocated() / 2**20
            elif torch.backends.mps.is_available():
                out["sys/mps_alloc_mb"] = torch.mps.current_allocated_memory() / 2**20
        except Exception:
            pass
        return out

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            try:
                for k, v in self.sample().items():
                    self.emit(k, v)
            except Exception:
                pass

    def stop(self) -> None:
        self._halt.set()
