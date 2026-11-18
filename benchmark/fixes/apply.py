"""Apply a structured fix to a benchmark variant, and decide whether a rerun recovered."""

from __future__ import annotations

import math
import re
from typing import Any

from ..tasks.trainer import DEFAULTS, Result

# code_fix name -> injected bugs it removes
CODE_FIXES: dict[str, set[str]] = {
    "missing_warmup": {"missing_warmup"},
    "scheduler_misstep": {"scheduler_misstep"},
    "fp16_no_loss_scaling": {"fp16_no_loss_scaling"},
    "nan_from_op": {"nan_from_op"},
    "frozen_layers": {"frozen_layers"},
    "double_softmax": {"double_softmax"},
    "bad_init": {"bad_init"},
    "shuffled_labels": {"shuffled_labels"},
    "val_leakage": {"val_leakage"},
    "missing_normalization": {"missing_normalization"},
    "bad_padding_mask": {"bad_padding_mask"},
    "missing_zero_grad": {"missing_zero_grad"},
    "grad_accum_unscaled": {"grad_accum_unscaled"},
    "eval_mode_training": {"eval_mode_training"},
    "dataloader_bottleneck": {"heavy_transform"},
    "memory_leak": {"memory_leak"},
}

# config fields an assistant may change, with sanity bounds
FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "lr": (1e-6, 10.0),
    "grad_clip": (0.01, 100.0),
    "warmup_steps": (0, 10_000),
    "weight_decay": (0, 1.0),
    "dropout": (0, 0.9),
    "train_size": (10, 4000),
    "num_workers": (0, 16),
    "batch_size": (4, 512),
    "grad_accum": (1, 64),
}
CATEGORICAL = {
    "precision": {"fp32", "fp16", "bf16"},
    "scheduler": {"constant", "linear", "cosine"},
    "optimizer": {"sgd", "adamw"},
}


class FixError(ValueError):
    pass


_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def leading_number(value: Any) -> float:
    """'5e-4 (and/or add dropout)' -> 5e-4. Models often explain the value inline; the first number is the
    proposal. Raises if the value doesn't *start* with a number (no guessing from the middle of prose)."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    m = _NUM.match(str(value).strip())
    if not m:
        raise FixError(f"needs a number, got {str(value)[:40]!r}")
    return float(m.group(0))


def apply_fix(
    task: str, cfg: dict[str, Any], bugs: list[str], fix: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    cfg, bugs = dict(cfg), list(bugs)
    ftype = fix.get("type")
    if ftype in (None, "none"):
        return cfg, bugs
    if ftype == "code_fix":
        name = fix.get("name")
        if name not in CODE_FIXES:
            raise FixError(f"unknown code fix {name!r}")
        bugs = [b for b in bugs if b not in CODE_FIXES[name]]
        if name == "fp16_no_loss_scaling":
            cfg["precision"] = "fp32"  # stand-in for proper AMP with a GradScaler on CPU
        return cfg, bugs
    if ftype == "config_change":
        fld, val = fix.get("field"), fix.get("value")
        if val == "default":
            val = DEFAULTS[task].get(fld)
        if fld in FIELD_BOUNDS:
            if val is None and fld == "grad_clip":
                cfg[fld] = None
                return cfg, bugs
            try:
                v = leading_number(val)
            except FixError as e:
                raise FixError(f"{fld} {e}") from e
            lo, hi = FIELD_BOUNDS[fld]
            if not lo <= v <= hi:
                raise FixError(f"{fld}={v} out of bounds [{lo}, {hi}]")
            cfg[fld] = (
                int(v)
                if fld in ("warmup_steps", "train_size", "num_workers", "batch_size", "grad_accum")
                else v
            )
        elif fld in CATEGORICAL:
            if val not in CATEGORICAL[fld]:
                raise FixError(f"{fld} must be one of {sorted(CATEGORICAL[fld])}")
            cfg[fld] = "fp32" if val == "bf16" else val
        else:
            raise FixError(f"field {fld!r} cannot be changed")
        return cfg, bugs
    raise FixError(f"unknown fix type {ftype!r}")


# Minimum quality a healthy short run reaches (tuned on dev-set healthy runs).
THRESHOLDS = {"vision": {"accuracy": 0.45}, "text_clf": {"accuracy": 0.6}, "lm": {"loss_max": 1.6}}


def recovered(task: str, cause: str, res: Result) -> tuple[bool, str]:
    if res.nonfinite or not math.isfinite(res.final_loss):
        return False, "loss is NaN/Inf"
    if res.final_loss > 0.8 * res.initial_loss:
        return False, f"loss did not decrease enough ({res.initial_loss:.3g} -> {res.final_loss:.3g})"
    th = THRESHOLDS[task]
    if "accuracy" in th and res.eval.get("accuracy", 0) < th["accuracy"]:
        return False, f"eval accuracy {res.eval.get('accuracy', 0):.3f} < {th['accuracy']}"
    if "loss_max" in th and res.eval.get("loss", 99) > th["loss_max"]:
        return False, f"eval loss {res.eval.get('loss'):.3f} > {th['loss_max']}"
    if cause == "dataloader_bottleneck" and res.median_data_wait_frac > 0.3:
        return False, f"still waiting on data {res.median_data_wait_frac:.0%} of each step"
    if cause == "memory_leak" and res.rss_growth_mb > 150:
        return False, f"RSS still grew {res.rss_growth_mb:.0f} MB"
    if cause == "overfitting" and res.best_eval_loss and res.eval.get("loss", 0) > 1.15 * res.best_eval_loss:
        return False, "eval loss still rises well above its best"
    if cause == "val_leakage" and res.eval.get("accuracy", 0) > 0.97:
        return False, "eval accuracy still implausibly high"
    return True, "recovered"
