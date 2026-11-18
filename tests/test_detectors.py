import math

import numpy as np

from trainscope_server.detectors import run_detectors
from trainscope_server.detectors.amp import AmpOverflowDetector
from trainscope_server.detectors.dataloader import DataLoaderDetector
from trainscope_server.detectors.divergence import DivergenceDetector
from trainscope_server.detectors.frozen import FrozenLayerDetector, TrainModeDetector
from trainscope_server.detectors.gradients import ExplodingGradientDetector, VanishingGradientDetector
from trainscope_server.detectors.lr_schedule import LRScheduleDetector
from trainscope_server.detectors.memory import MemoryLeakDetector
from trainscope_server.detectors.overfit import OverfittingDetector
from trainscope_server.detectors.plateau import PlateauDetector
from trainscope_server.detectors.spikes import LossSpikeDetector

from .helpers import data, healthy_loss


def test_healthy_run_has_no_findings():
    n = 300
    lr = 0.1 * 0.5 * (1 + np.cos(np.pi * np.arange(n) / n))
    d = data(
        {
            "loss": healthy_loss(n),
            "lr": lr,
            "data_wait_frac": np.full(n, 0.02),
            "grad/global_norm": 1 + 0.1 * np.random.default_rng(0).normal(size=n),
        },
        config={"lr": 0.1, "scheduler": "cosine", "total_steps": n, "num_classes": 10},
    )
    assert run_detectors(d) == []


def test_divergence_nan():
    loss = list(healthy_loss(100)) + [float("nan")] * 5
    f = DivergenceDetector().run(data({"loss": loss}))
    assert f and f[0].severity == "critical" and f[0].step_start == 100


def test_divergence_rising():
    loss = np.concatenate([healthy_loss(100), np.linspace(0.6, 5, 100)])
    f = DivergenceDetector().run(data({"loss": loss}))
    assert f and "rose" in f[0].summary


def test_loss_spikes():
    loss = healthy_loss(300)
    loss[150] = 8
    f = LossSpikeDetector().run(data({"loss": loss}))
    assert f and 150 in f[0].evidence["spike_steps"]


def test_plateau_at_chance():
    rng = np.random.default_rng(0)
    loss = math.log(10) + 0.02 * rng.normal(size=200)
    f = PlateauDetector().run(data({"loss": loss}, config={"num_classes": 10}))
    assert any(x.detector == "plateau" for x in f)
    assert "shuffled_labels" in f[0].suspected_causes


def test_bad_initial_loss():
    loss = np.concatenate([np.full(20, 40.0), healthy_loss(200)])
    f = PlateauDetector().run(data({"loss": loss}, config={"num_classes": 10}))
    assert any(x.detector == "bad_initial_loss" for x in f)


def test_exploding_and_growth():
    g = np.concatenate([np.ones(20), np.linspace(1, 200, 20)])
    f = ExplodingGradientDetector().run(
        data({"grad/global_norm": g}, layers={"grad/norm": {"a": g, "b": np.ones(40)}}, every=10)
    )
    names = {x.detector for x in f}
    assert "exploding_gradients" in names
    assert f[0].evidence["top_layers"][0]["layer"] == "a"


def test_linear_grad_growth_flags_zero_grad():
    g = np.arange(1, 31, dtype=float)
    f = ExplodingGradientDetector().run(data({"grad/global_norm": g}, every=10))
    assert any(x.detector == "grad_norm_growth" and "missing_zero_grad" in x.suspected_causes for x in f)


def test_vanishing_and_dead_units():
    f = VanishingGradientDetector().run(
        data(
            layers={
                "grad/norm": {"a": np.full(10, 1.0), "b": np.full(10, 1e-9)},
                "act/dead_frac": {"relu1": np.full(10, 0.97)},
            }
        )
    )
    assert {x.detector for x in f} == {"vanishing_gradients", "dead_units"}


def test_frozen_layers_and_eval_mode():
    f = FrozenLayerDetector().run(
        data(
            layers={
                "weight/update_ratio": {
                    "stem": np.zeros(10),
                    "block": np.zeros(10),
                    "head": np.full(10, 1e-3),
                },
                "weight/trainable": {"stem": np.zeros(10), "block": np.zeros(10), "head": np.ones(10)},
            }
        )
    )
    assert f and f[0].severity == "critical" and "stem" in f[0].evidence["frozen_layers"]
    f = TrainModeDetector().run(data({"model/train_mode": np.zeros(10)}))
    assert f and f[0].suspected_causes == ["eval_mode_training"]


def test_overfitting():
    steps = 20
    ev = np.concatenate([np.linspace(2, 1, 10), np.linspace(1, 2, 10)])
    d = data({"loss": np.linspace(2, 0.05, 200), "eval/loss": ev})
    d.df.loc[d.df.name == "eval/loss", "step"] = np.arange(steps) * 10
    f = OverfittingDetector().run(d)
    assert f and f[0].detector == "overfitting"


def test_dataloader_bottleneck():
    f = DataLoaderDetector().run(data({"data_wait_frac": np.full(50, 0.7)}))
    assert f and f[0].severity == "critical"


def test_memory_leak_thresholds_scale_with_the_run():
    """Thresholds are relative to the run's own memory, so the same detector works on a laptop and a big GPU."""
    rng = np.random.default_rng(0)
    for start in (500.0, 50_000.0):  # same 80% leak at two very different scales
        s = np.linspace(start, start * 1.8, 60) + rng.normal(0, start * 0.002, 60)
        assert MemoryLeakDetector().run(data({"sys/cuda_alloc_mb": s})), (
            f"missed a leak starting at {start} MB"
        )
    # +200 MB on a 70 GB process is 0.3%: not a leak (a fixed 100 MB bar would have flagged it)
    big = np.linspace(70_000, 70_200, 60)
    assert MemoryLeakDetector().run(data({"sys/cuda_alloc_mb": big})) == []
    # too few samples to judge (start-up allocation dominates short runs)
    assert MemoryLeakDetector().run(data({"sys/proc_rss_mb": np.linspace(500, 900, 20)})) == []


def test_memory_leak():
    f = MemoryLeakDetector().run(data({"sys/proc_rss_mb": np.linspace(500, 900, 50)}))
    assert f and f[0].evidence["series"] == "sys/proc_rss_mb"
    assert MemoryLeakDetector().run(data({"sys/proc_rss_mb": np.full(50, 500.0)})) == []


def test_amp_overflow():
    f = AmpOverflowDetector().run(
        data({"amp/loss_scale": [2.0 ** (16 - i) for i in range(20)], "amp/skipped_steps": np.arange(20)})
    )
    assert f


def test_lr_schedule_no_warmup_and_misstep():
    n = 200
    f = LRScheduleDetector().run(
        data(
            {"lr": np.full(n, 1e-3)},
            config={"lr": 1e-3, "scheduler": "linear", "warmup_steps": 50, "total_steps": n},
        )
    )
    assert f and "warmup" in f[0].summary
    lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * np.arange(n) / 3))  # cosine with T_max=3 stepped per batch
    f = LRScheduleDetector().run(
        data({"lr": lr}, config={"lr": 1e-3, "scheduler": "cosine", "total_steps": n})
    )
    assert f and f[0].detector == "lr_schedule"


def test_detectors_tolerate_empty_run():
    assert run_detectors(data()) == []


def test_split_overlap_flags_leakage():
    from trainscope_server.detectors.data import SuspiciousDataDetector

    ev = [
        {"step": 0, "type": "dataset_ids", "payload": {"split": "train", "ids": ["a", "b", "c", "d"]}},
        {"step": 0, "type": "dataset_ids", "payload": {"split": "eval", "ids": ["c", "d", "e"]}},
    ]
    f = SuspiciousDataDetector().run(data(events=ev))
    assert f and f[0].detector == "split_overlap" and f[0].evidence["overlap"] == 2
    clean = [ev[0], {"step": 0, "type": "dataset_ids", "payload": {"split": "eval", "ids": ["x"]}}]
    assert SuspiciousDataDetector().run(data(events=clean)) == []


def test_plateau_double_softmax_vs_shuffled():
    rng = np.random.default_rng(0)
    loss = math.log(10) - 0.05 + 0.02 * rng.normal(size=200)
    ds = PlateauDetector().run(
        data({"loss": loss, "train/accuracy": np.full(200, 0.25)}, config={"num_classes": 10})
    )
    sh = PlateauDetector().run(
        data({"loss": loss, "train/accuracy": np.full(200, 0.10)}, config={"num_classes": 10})
    )
    assert ds[0].suspected_causes[0] == "double_softmax"
    assert sh[0].suspected_causes[0] == "shuffled_labels"


def test_carryover_detector():
    from trainscope_server.detectors.gradients import GradCarryoverDetector

    assert GradCarryoverDetector().run(data({"grad/carryover_ratio": np.ones(10)})) == []
    f = GradCarryoverDetector().run(data({"grad/carryover_ratio": np.full(10, 1.4)}))
    assert f and f[0].suspected_causes[0] == "missing_zero_grad"
    assert (
        GradCarryoverDetector().run(
            data({"grad/carryover_ratio": np.full(10, 3.0)}, config={"grad_accum": 2})
        )
        == []
    )
