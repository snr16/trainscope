"""Runs all detectors over a run and ranks findings. Can be run on a prefix (upto_step) for
steps-to-detection measurements."""

from __future__ import annotations

import logging
from typing import Any

from .amp import AmpOverflowDetector
from .base import SEVERITY_RANK, Detector, Finding, RunData
from .data import SuspiciousDataDetector
from .dataloader import DataLoaderDetector
from .divergence import DivergenceDetector
from .frozen import FrozenLayerDetector, TrainModeDetector
from .gradients import ExplodingGradientDetector, GradCarryoverDetector, VanishingGradientDetector
from .lr_schedule import LRScheduleDetector
from .memory import MemoryLeakDetector
from .overfit import OverfittingDetector
from .plateau import PlateauDetector
from .spikes import LossSpikeDetector

log = logging.getLogger("trainscope.detectors")

DETECTORS: list[Detector] = [
    DivergenceDetector(),
    LossSpikeDetector(),
    PlateauDetector(),
    ExplodingGradientDetector(),
    GradCarryoverDetector(),
    VanishingGradientDetector(),
    FrozenLayerDetector(),
    TrainModeDetector(),
    OverfittingDetector(),
    DataLoaderDetector(),
    MemoryLeakDetector(),
    AmpOverflowDetector(),
    LRScheduleDetector(),
    SuspiciousDataDetector(),
]


def rank_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (-SEVERITY_RANK.get(f.severity, 0), -f.confidence, f.step_start))


def run_detectors(data: RunData, detectors: list[Detector] | None = None) -> list[Finding]:
    out: list[Finding] = []
    for d in detectors or DETECTORS:
        try:
            out.extend(d.run(data))
        except Exception:
            log.exception("detector %s failed", d.name)
    return rank_findings(out)


def refresh_findings(store: Any, run_id: str) -> list[dict[str, Any]]:
    """Recompute findings for a run from stored telemetry and persist them."""
    data = RunData.from_store(store, run_id)
    findings = [f.to_dict(run_id) for f in run_detectors(data)]
    store.replace_findings(run_id, findings)
    return findings
