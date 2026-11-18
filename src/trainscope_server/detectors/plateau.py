"""Plateau / no learning, plus bad initial loss.

Which cause a plateau points to depends on how the weights move:
- update ratio ~0: frozen weights / underflow
- update ratio large (>5e-3) or many dead units: LR too high (weights thrash, ReLUs die)
- normal updates but loss at chance: the inputs carry no usable signal (shuffled labels, double softmax)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import Detector, Finding, RunData, finite


def weight_motion(data: RunData) -> tuple[float | None, float | None]:
    upd = data.layers("weight/update_ratio")
    dead = data.layers("act/dead_frac")
    med_upd = float(np.nanmedian(upd.to_numpy())) if not upd.empty else None
    max_dead = (
        float(np.nanmax(dead.iloc[-max(1, len(dead) // 3) :].mean().to_numpy())) if not dead.empty else None
    )
    return med_upd, max_dead


class PlateauDetector(Detector):
    name = "plateau"
    min_steps = 60

    # plateau, chance-level and bad-init thresholds
    defaults = {
        "min_rel_drop": 0.05,  # loss must fall at least this fraction to count as learning
        "thrash_min_drop": 0.2,  # ... and this much when the weights are thrashing
        "high_update": 0.002,  # median update ratio above this inside a plateau = LR too high
        "frozen_update": 1e-06,  # median update ratio below this = weights not moving
        "dead_frac": 0.25,  # a layer with more dead units than this = LR too high / bad init
        "bad_init_x_chance": 2.0,  # step-0 loss this many x ln(num_classes) = bad init
        "bad_init_critical_x": 3.0,  # ... this many x = critical
        "rising_x": 1.5,  # loss ending this many x its start is divergence, not a plateau
        "double_softmax_acc_x": 1.6,  # train accuracy above this x chance while loss is stuck = double softmax
        "chance_band": 0.1,  # within this fraction of ln(num_classes) = sitting at chance
        "near_chance": 0.95,  # final loss above this fraction of chance = stuck near chance
        "near_chance_head_x": 1.5,  # ... when the run also started below this x chance
    }

    def run(self, data: RunData) -> list[Finding]:
        raw = data.scalar("loss")
        loss = finite(raw)
        out: list[Finding] = []
        if len(loss) < self.min_steps:
            return out
        n = len(loss)
        first = float(loss.iloc[:3].median())  # loss before any real learning
        head = float(loss.iloc[: max(5, n // 10)].mean())
        tail = float(loss.iloc[-max(5, n // 5) :].mean())
        k = data.cfg("num_classes", "n_classes")
        chance = math.log(k) if k else None
        med_upd, max_dead = weight_motion(data)
        grad = finite(data.scalar("grad/global_norm"))
        ev: dict[str, Any] = {
            "initial_loss": head,
            "final_loss": tail,
            "rel_drop": (head - tail) / abs(head) if head else 0.0,
            "chance_loss": chance,
            "median_update_ratio": med_upd,
            "max_dead_unit_frac": max_dead,
            "median_grad_norm": float(grad.median()) if len(grad) else None,
        }
        if chance and first > self.bad_init_x_chance * chance:
            out.append(
                Finding(
                    "bad_initial_loss",
                    "critical" if first > self.bad_init_critical_x * chance else "warning",
                    int(loss.index[0]),
                    int(loss.index[min(n - 1, 10)]),
                    f"loss at step 0 is {first:.3g}, far above chance level ln({k})={chance:.3g}",
                    {"loss_at_start": first, "chance_loss": chance},
                    ["bad_init", "missing_normalization"],
                    0.8,
                )
            )
        if len(loss) < len(raw) or tail > self.rising_x * head:
            return out  # NaN or rising loss: the divergence detector owns this
        if med_upd is not None and med_upd < self.frozen_update:
            causes = ["frozen_layers", "fp16_no_loss_scaling", "lr_too_high"]
            detail = f"; weights barely move (median update ratio {med_upd:.1e})"
        elif (med_upd is not None and med_upd > self.high_update) or (
            max_dead is not None and max_dead > self.dead_frac
        ):
            causes = ["lr_too_high", "missing_warmup"]
            detail = (f"; weights move a lot (median update ratio {med_upd:.1e})" if med_upd else "") + (
                f"; up to {max_dead:.0%} dead units" if max_dead and max_dead > self.dead_frac else ""
            )
        else:
            tr_acc = finite(data.scalar("train/accuracy"))
            acc = float(tr_acc.iloc[-3:].mean()) if len(tr_acc) >= 3 else None
            ev["train_accuracy"] = acc
            if acc is not None and k and acc > self.double_softmax_acc_x / k:
                # loss can't move but the model still ranks the right class first: probabilities fed to the loss
                causes = ["double_softmax", "shuffled_labels", "frozen_layers"]
                detail = f"; yet train accuracy is {acc:.0%} (chance {1 / k:.0%}), so the model is learning"
            else:
                causes = ["shuffled_labels", "double_softmax", "frozen_layers"]
                detail = f"; train accuracy at chance ({acc:.0%})" if acc is not None else ""
        thrashing = causes[0] == "lr_too_high"
        # with weights thrashing, a small drop is not learning: healthy runs here fall 60-80%
        bar = self.thrash_min_drop if thrashing else self.min_rel_drop
        learning = head > 0 and (head - tail) / abs(head) >= bar
        if not learning:
            if chance and abs(tail - chance) < self.chance_band * chance:
                detail += f"; loss sits at chance level ln({k})={chance:.3g}"
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
        elif chance and tail > self.near_chance * chance and head < self.near_chance_head_x * chance:
            out.append(
                Finding(
                    self.name,
                    "critical",
                    int(loss.index[0]),
                    data.max_step,
                    f"loss stays near chance level ({tail:.3g} vs ln({k})={chance:.3g}){detail}",
                    ev,
                    causes,
                    0.65,
                )
            )
        return out
