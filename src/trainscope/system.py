"""Background sampler for GPU / CPU / memory metrics (~1 Hz)."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

import psutil


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
