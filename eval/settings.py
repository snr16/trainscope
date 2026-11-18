"""Evaluation settings from eval/config.yaml (models, limits, prices). CLI flags in harness.py override them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("config.yaml")
CONFIG_DEFAULTS: dict[str, Any] = {
    "models": [],
    "effort": "medium",
    "max_tool_calls": 12,
    "concurrency": 4,
    "dev_sample_per_cause": 1,
    "prices": {},
    "fallback_models": [],
}


class SettingsError(ValueError):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    cfg = dict(CONFIG_DEFAULTS)
    if path.exists():
        cfg.update(yaml.safe_load(path.read_text()) or {})
    if isinstance(cfg["models"], str):
        cfg["models"] = [cfg["models"]]
    cfg["prices"] = parse_prices(cfg.get("prices") or {}, path)
    cfg["fallback_models"] = set(cfg.get("fallback_models") or [])
    return cfg


def parse_prices(raw: dict[str, Any], where: Path | str = "config") -> dict[str, tuple[float, float, float]]:
    """{model: {input, output, cache_read}} in $ per 1M tokens -> {model: (input, output, cache_read)}."""
    out = {}
    for model, p in raw.items():
        try:
            out[str(model)] = (
                float(p["input"]),
                float(p["output"]),
                float(p.get("cache_read", 0.1 * p["input"])),
            )
        except (TypeError, KeyError, ValueError) as e:
            raise SettingsError(
                f"{where}: prices.{model} needs numeric input and output ($ per 1M tokens)"
            ) from e
    return out
