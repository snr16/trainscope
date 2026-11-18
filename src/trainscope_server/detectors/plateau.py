"""Plateau / no learning, plus bad initial loss."""

from __future__ import annotations

import math

import numpy as np

from .base import Detector, Finding, RunData, finite


class PlateauDetector(Detector):
    name = "plateau"
    min_steps = 60

    def __init__(self, min_rel_drop: float = 0.05):
        self.min_rel_drop = min_rel_drop

    def run(self, data: RunData) -> list[Finding]:
        loss = finite(data.scalar("loss"))
        out: list[Finding] = []
        if len(loss) < self.min_steps:
            return out
        n = len(loss)
        head = float(loss.iloc[: max(5, n // 10)].mean())
        tail = float(loss.iloc[-max(5, n // 5) :].mean())
        k = data.cfg("num_classes", "n_classes")
        chance = math.log(k) if k else None
        upd = data.layers("weight/update_ratio")
        med_upd = float(np.nanmedian(upd.to_numpy())) if not upd.empty else None
        grad = finite(data.scalar("grad/global_norm"))
        ev = {
            "initial_loss": head,
            "final_loss": tail,
            "rel_drop": (head - tail) / abs(head) if head else 0.0,
            "chance_loss": chance,
            "median_update_ratio": med_upd,
            "median_grad_norm": float(grad.median()) if len(grad) else None,
        }
        if chance and head > 3 * chance:
            out.append(
                Finding(
                    "bad_initial_loss",
                    "warning",
                    int(loss.index[0]),
                    int(loss.index[min(n - 1, 10)]),
                    f"initial loss {head:.3g} is far above chance level ln({k})={chance:.3g}",
                    {"initial_loss": head, "chance_loss": chance},
                    ["bad_init", "missing_normalization"],
                    0.6,
                )
            )
        if head <= 0 or (head - tail) / abs(head) >= self.min_rel_drop:
            # learning, but check "stuck at chance" for classification
            if chance and tail > 0.95 * chance and head < 1.5 * chance:
                out.append(
                    Finding(
                        self.name,
                        "critical",
                        int(loss.index[0]),
                        data.max_step,
                        f"loss stays near chance level ({tail:.3g} vs ln({k})={chance:.3g})",
                        ev,
                        ["shuffled_labels", "double_softmax", "frozen_layers"],
                        0.6,
                    )
                )
            return out
        causes = ["shuffled_labels", "double_softmax", "frozen_layers", "fp16_no_loss_scaling"]
        detail = ""
        if med_upd is not None and med_upd < 1e-6:
            causes = ["frozen_layers", "fp16_no_loss_scaling", "lr_too_high"]
            detail = f"; weights barely move (median update ratio {med_upd:.1e})"
        elif chance and abs(tail - chance) < 0.1 * chance:
            detail = f"; loss sits at chance level ln({k})={chance:.3g}"
        out.append(
            Finding(
                self.name,
                "critical",
                int(loss.index[0]),
                data.max_step,
                f"no learning: loss {head:.3g} -> {tail:.3g} ({ev['rel_drop'] * 100:.1f}% drop){detail}",
                ev,
                causes,
                0.7,
            )
        )
        return out
