"""detectors.yaml: overrides, per-project settings, validation and live reload."""

import os
import time

import numpy as np
import pytest
import yaml

from trainscope_server.commands.cli import detectors_yaml
from trainscope_server.detectors import DETECTOR_CLASSES, ConfigError, RunData, detectors_for, run_detectors
from trainscope_server.detectors.config import build, parse_config
from trainscope_server.detectors.dataloader import DataLoaderDetector

from .helpers import frame


def _loader_run(project=None, wait=0.25):
    return RunData(frame({"data_wait_frac": np.full(40, wait)}), {}, [], project=project)


def test_every_threshold_is_overridable():
    for cls in DETECTOR_CLASSES.values():
        d = cls(**{k: v for k, v in cls.defaults.items()})
        assert d.params == cls.defaults
    assert DataLoaderDetector(wait_frac=0.1).wait_frac == 0.1
    with pytest.raises(ValueError, match="unknown threshold"):
        DataLoaderDetector(wiat_frac=0.1)


def test_global_and_project_overrides_and_disabled():
    cfg = parse_config(
        {
            "defaults": {"dataloader_bottleneck": {"wait_frac": 0.2}},
            "disabled": ["amp_overflow"],
            "projects": {"tight": {"dataloader_bottleneck": {"wait_frac": 0.1}, "disabled": ["loss_spikes"]}},
        },
        DETECTOR_CLASSES,
    )
    default_set = {d.name: d for d in build(DETECTOR_CLASSES, cfg, None)}
    tight_set = {d.name: d for d in build(DETECTOR_CLASSES, cfg, "tight")}
    assert default_set["dataloader_bottleneck"].wait_frac == 0.2
    assert tight_set["dataloader_bottleneck"].wait_frac == 0.1
    assert "amp_overflow" not in default_set and "amp_overflow" not in tight_set
    assert "loss_spikes" in default_set and "loss_spikes" not in tight_set
    # untouched thresholds keep their defaults
    assert tight_set["dataloader_bottleneck"].critical_wait == DataLoaderDetector.defaults["critical_wait"]


@pytest.mark.parametrize(
    "raw, msg",
    [
        ({"defaults": {"no_such_detector": {}}}, "unknown detector"),
        ({"defaults": {"plateau": {"high_updte": 1}}}, "unknown threshold"),
        ({"disabled": ["nope"]}, "unknown detector"),
        ({"thresholds": {}}, "unknown top-level"),
    ],
)
def test_config_errors_are_explicit(raw, msg):
    with pytest.raises(ConfigError, match=msg):
        parse_config(raw, DETECTOR_CLASSES)


def test_env_file_applies_per_project_and_reloads(tmp_path, monkeypatch):
    path = tmp_path / "detectors.yaml"
    path.write_text(yaml.safe_dump({"projects": {"strict": {"dataloader_bottleneck": {"wait_frac": 0.2}}}}))
    monkeypatch.setenv("TRAINSCOPE_DETECTORS", str(path))
    # 25% data wait: below the default 30% bar, above the strict project's 20%
    assert not any(f.detector == "dataloader_bottleneck" for f in run_detectors(_loader_run("other")))
    assert any(f.detector == "dataloader_bottleneck" for f in run_detectors(_loader_run("strict")))
    # edit the file: the next call sees the new value without a restart
    path.write_text(yaml.safe_dump({"projects": {"strict": {"dataloader_bottleneck": {"wait_frac": 0.5}}}}))
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert not any(f.detector == "dataloader_bottleneck" for f in run_detectors(_loader_run("strict")))
    monkeypatch.delenv("TRAINSCOPE_DETECTORS")
    assert {d.name for d in detectors_for(None)} == set(DETECTOR_CLASSES)


def test_cli_output_is_a_valid_config_with_the_defaults():
    doc = yaml.safe_load(detectors_yaml())
    cfg = parse_config(doc, DETECTOR_CLASSES)
    for d in build(DETECTOR_CLASSES, cfg, None):
        assert d.params == type(d).defaults


def test_yaml_style_numbers():
    # YAML 1.1 loads "1e-6" as a string; it must still work, and non-numbers must fail clearly
    cfg = parse_config(yaml.safe_load("defaults:\n  plateau: {frozen_update: 1e-6}\n"), DETECTOR_CLASSES)
    assert cfg["defaults"]["plateau"]["frozen_update"] == 1e-6
    with pytest.raises(ConfigError, match="expected a number"):
        parse_config({"defaults": {"plateau": {"high_update": "high"}}}, DETECTOR_CLASSES)
