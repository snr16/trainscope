"""Scenario catalog -> concrete benchmark variants (public config + hidden bugs + ground truth)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from trainscope_server.taxonomy import CAUSES

from ..tasks.trainer import DEFAULTS

CATALOG = Path(__file__).with_name("catalog.yaml")
SALT = "trainscope-bench-v1"
TEST_FRACTION = 0.4


@dataclass
class Variant:
    scenario: str
    cause: str
    category: str
    task: str
    severity: str
    seed: int
    cfg: dict[str, Any]
    bugs: list[str]
    component: str
    fixes: list[dict[str, Any]]
    run_id: str = ""
    split: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.scenario}/{self.task}/{self.severity}/{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_catalog(path: Path = CATALOG) -> dict[str, Any]:
    cat = yaml.safe_load(path.read_text())
    for sid, sc in cat.items():
        if sc["cause"] not in CAUSES:
            raise ValueError(f"scenario {sid}: unknown cause {sc['cause']}")
        bad = set(sc["tasks"]) - set(DEFAULTS)
        if bad:
            raise ValueError(f"scenario {sid}: unknown tasks {bad}")
    return cat


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _split_of(key: str) -> str:
    h = int(hashlib.sha256(f"{SALT}:split:{key}".encode()).hexdigest(), 16)
    return "test" if (h % 1000) / 1000 < TEST_FRACTION else "dev"


def _run_id(key: str) -> str:
    return hashlib.sha256(f"{SALT}:run:{key}".encode()).hexdigest()[:12]


def resolve_cfg(task: str, spec: dict[str, Any]) -> dict[str, Any]:
    spec = {**spec, **((spec.get("per_task") or {}).get(task) or {})}  # task-specific overrides
    cfg = dict(spec.get("cfg") or {})
    base_lr = cfg.get("lr", DEFAULTS[task]["lr"])
    if "lr_mult" in spec:
        cfg["lr"] = float(base_lr) * float(spec["lr_mult"])
    if "lr_abs" in spec:
        cfg["lr"] = float(spec["lr_abs"])
    return cfg


def expand(
    seeds: list[int] | None = None,
    scenarios: list[str] | None = None,
    tasks: list[str] | None = None,
    split: str | None = None,
) -> list[Variant]:
    cat = load_catalog()
    out = []
    has_cuda = _has_cuda()
    for sid, sc in cat.items():
        if scenarios and sid not in scenarios:
            continue
        if sc.get("requires") == "cuda" and not has_cuda:
            continue
        for task in sc["tasks"]:
            if tasks and task not in tasks:
                continue
            for sev, spec in sc["severities"].items():
                for seed in seeds or [0, 1, 2]:
                    v = Variant(
                        sid,
                        sc["cause"],
                        CAUSES[sc["cause"]][0],
                        task,
                        sev,
                        seed,
                        resolve_cfg(task, spec),
                        list({**spec, **((spec.get("per_task") or {}).get(task) or {})}.get("bugs") or []),
                        sc["component"],
                        sc["fixes"],
                    )
                    v.run_id = _run_id(v.key)
                    v.split = _split_of(v.key)
                    if split is None or v.split == split:
                        out.append(v)
    return out
