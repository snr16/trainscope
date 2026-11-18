"""Frozen layers (weights never change) and model left in eval mode during training."""

from __future__ import annotations

import numpy as np

from .base import Detector, Finding, RunData


class FrozenLayerDetector(Detector):
    name = "frozen_layers"

    # frozen-layer thresholds
    defaults = {
        "eps": 1e-09,  # max update ratio at or below this = layer never changes
        "critical_frac": 0.5,  # this fraction of layers frozen or more = critical
        "conf_frozen_critical": 0.8,  # ranking weight: most layers frozen
        "conf_frozen_warning": 0.6,  # ranking weight: some layers frozen
    }

    def run(self, data: RunData) -> list[Finding]:
        upd = data.layers("weight/update_ratio")
        trainable = data.layers("weight/trainable")
        out: list[Finding] = []
        frozen: dict[str, str] = {}
        if not trainable.empty:
            for layer, v in trainable.min().items():
                if v == 0:
                    frozen[layer] = "requires_grad=False"
        if not upd.empty and len(upd) >= 2:
            mx = upd.max()
            moving = mx[mx > self.eps]
            for layer, v in mx.items():
                if v <= self.eps and layer not in frozen:
                    frozen[layer] = "weights never change"
            if not len(moving):
                frozen = {k: v for k, v in frozen.items()}
        if not frozen:
            return out
        n_layers = max(
            len(upd.columns) if not upd.empty else 0, len(trainable.columns) if not trainable.empty else 0
        )
        frac = len(frozen) / max(n_layers, 1)
        sev = "critical" if frac >= self.critical_frac else "warning"
        names = list(frozen)
        return [
            Finding(
                self.name,
                sev,
                data.df["step"].min() if len(data.df) else 0,
                data.max_step,
                f"{len(frozen)}/{n_layers} layer(s) frozen (e.g. {', '.join(names[:3])})",
                {
                    "frozen_layers": dict(list(frozen.items())[:20]),
                    "frozen_fraction": frac,
                    "note": "intentional if fine-tuning with a frozen backbone",
                },
                ["frozen_layers"],
                self.conf_frozen_critical if sev == "critical" else self.conf_frozen_warning,
            )
        ]


class TrainModeDetector(Detector):
    name = "eval_mode_training"
    # eval-mode threshold
    defaults = {
        "eval_frac": 0.8,  # model in eval mode on at least this fraction of sampled steps
        "conf_eval_mode": 0.9,  # ranking weight: model left in eval mode
    }

    def run(self, data: RunData) -> list[Finding]:
        tm = data.scalar("model/train_mode")
        if len(tm) < 3:
            return []
        frac = float(np.mean(tm.to_numpy() == 0))
        if frac < self.eval_frac:
            return []
        return [
            Finding(
                self.name,
                "critical",
                int(tm.index[0]),
                int(tm.index[-1]),
                f"model was in eval mode for {frac:.0%} of sampled training steps "
                "(dropout/batch-norm behave as in inference)",
                {"eval_mode_fraction": frac, "samples": len(tm)},
                ["eval_mode_training"],
                self.conf_eval_mode,
            )
        ]
