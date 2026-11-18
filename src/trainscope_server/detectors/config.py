"""Detector thresholds from a ``detectors.yaml`` file.

Every detector ships with default thresholds (its ``defaults`` table). A config file overrides them globally and
per project, and can switch detectors off::

    defaults:                       # applies to every project
      memory_leak: {min_growth_mb: 500}
    disabled: [amp_overflow]
    projects:
      bert-finetune:                # runs whose ts.init(project=...) is "bert-finetune"
        plateau: {high_update: 0.005}
        disabled: [suspicious_data]

The file is found through ``$TRAINSCOPE_DETECTORS`` and re-read when it changes, so a running server picks up
new thresholds without a restart. ``trainscope detectors`` prints every detector's thresholds and defaults.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .base import Detector

ENV_VAR = "TRAINSCOPE_DETECTORS"


class ConfigError(ValueError):
    pass


def _validate_overrides(
    where: str, section: Any, classes: dict[str, type[Detector]]
) -> dict[str, dict[str, Any]]:
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ConfigError(f"{where}: expected a mapping of detector name -> thresholds")
    out: dict[str, dict[str, Any]] = {}
    for name, values in section.items():
        if name == "disabled":
            continue
        if name not in classes:
            raise ConfigError(f"{where}: unknown detector {name!r}; known: {sorted(classes)}")
        if not isinstance(values, dict):
            raise ConfigError(f"{where}.{name}: expected a mapping of threshold -> value")
        values = {k: _number(f"{where}.{name}.{k}", v) for k, v in values.items()}
        try:
            classes[name](**values)  # raises on unknown threshold names
        except ValueError as e:
            raise ConfigError(f"{where}.{name}: {e}") from e
        out[name] = values
    return out


def _number(where: str, v: Any) -> float | int:
    # YAML 1.1 reads "1e-6" (no decimal point) as a string, so accept numeric strings too
    if isinstance(v, bool):
        raise ConfigError(f"{where}: expected a number, got {v!r}")
    if isinstance(v, int | float):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: expected a number, got {v!r}") from None


def _validate_disabled(where: str, value: Any, classes: dict[str, type[Detector]]) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where}.disabled: expected a list of detector names")
    unknown = set(value) - set(classes)
    if unknown:
        raise ConfigError(f"{where}.disabled: unknown detector(s) {sorted(unknown)}")
    return set(value)


def parse_config(raw: Any, classes: dict[str, type[Detector]]) -> dict[str, Any]:
    """Validate a loaded YAML document into {defaults, disabled, projects: {name: {overrides, disabled}}}."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError("detectors config: top level must be a mapping")
    extra = set(raw) - {"defaults", "disabled", "projects"}
    if extra:
        raise ConfigError(f"detectors config: unknown top-level key(s) {sorted(extra)}")
    projects = {}
    for pname, section in (raw.get("projects") or {}).items():
        projects[str(pname)] = {
            "overrides": _validate_overrides(f"projects.{pname}", section, classes),
            "disabled": _validate_disabled(f"projects.{pname}", (section or {}).get("disabled"), classes),
        }
    return {
        "defaults": _validate_overrides("defaults", raw.get("defaults"), classes),
        "disabled": _validate_disabled("top level", raw.get("disabled"), classes),
        "projects": projects,
    }


def load_file(path: str | os.PathLike, classes: dict[str, type[Detector]]) -> dict[str, Any]:
    import yaml

    try:
        raw = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML ({e})") from e
    return parse_config(raw, classes)


def build(
    classes: dict[str, type[Detector]], cfg: dict[str, Any] | None, project: str | None
) -> list[Detector]:
    """Detector instances for one project: defaults, then global overrides, then the project's overrides."""
    cfg = cfg or {"defaults": {}, "disabled": set(), "projects": {}}
    proj = cfg["projects"].get(project or "", {"overrides": {}, "disabled": set()})
    disabled = cfg["disabled"] | proj["disabled"]
    out = []
    for name, cls in classes.items():
        if name in disabled:
            continue
        out.append(cls(**{**cfg["defaults"].get(name, {}), **proj["overrides"].get(name, {})}))
    return out


class ConfigCache:
    """Re-reads the config file when its path or modification time changes; caches detectors per project."""

    def __init__(self, classes: dict[str, type[Detector]]):
        self.classes = classes
        self.lock = threading.Lock()
        self.key: tuple[str, float] | None = None
        self.cfg: dict[str, Any] | None = None
        self.per_project: dict[str | None, list[Detector]] = {}

    def detectors_for(self, project: str | None) -> list[Detector]:
        path = os.environ.get(ENV_VAR, "")
        try:
            key = (path, Path(path).stat().st_mtime) if path else ("", 0.0)
        except OSError as e:
            raise ConfigError(f"${ENV_VAR}={path}: {e.strerror}") from e
        with self.lock:
            if key != self.key:
                self.cfg = load_file(path, self.classes) if path else None
                self.key = key
                self.per_project = {}
            if project not in self.per_project:
                self.per_project[project] = build(self.classes, self.cfg, project)
            return self.per_project[project]


def thresholds_table(classes: dict[str, type[Detector]]) -> dict[str, dict[str, Any]]:
    return {name: dict(cls.defaults) for name, cls in classes.items() if cls.defaults}
