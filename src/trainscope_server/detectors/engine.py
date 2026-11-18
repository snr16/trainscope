"""Runs all detectors over a run and ranks findings. Can be run on a prefix (upto_step) for
steps-to-detection measurements."""

from __future__ import annotations

import logging
from typing import Any

from .amp import AmpOverflowDetector
from .base import SEVERITY_RANK, Detector, Finding, RunData
from .config import ConfigCache
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

# every detector, keyed by the name used in findings and in detectors.yaml (order = evaluation order)
DETECTOR_CLASSES: dict[str, type[Detector]] = {
    cls.name: cls
    for cls in (
        DivergenceDetector,
        LossSpikeDetector,
        PlateauDetector,
        ExplodingGradientDetector,
        GradCarryoverDetector,
        VanishingGradientDetector,
        FrozenLayerDetector,
        TrainModeDetector,
        OverfittingDetector,
        DataLoaderDetector,
        MemoryLeakDetector,
        AmpOverflowDetector,
        LRScheduleDetector,
        SuspiciousDataDetector,
    )
}
# detectors with their built-in thresholds (no config file)
DETECTORS: list[Detector] = [cls() for cls in DETECTOR_CLASSES.values()]
_config = ConfigCache(DETECTOR_CLASSES)


def detectors_for(project: str | None = None) -> list[Detector]:
    """Detectors with thresholds from $TRAINSCOPE_DETECTORS (if set) for this project."""
    return _config.detectors_for(project)


def rank_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (-SEVERITY_RANK.get(f.severity, 0), -f.confidence, f.step_start))


def run_detectors(data: RunData, detectors: list[Detector] | None = None) -> list[Finding]:
    out: list[Finding] = []
    for d in detectors if detectors is not None else detectors_for(data.project):
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
