"""Grade a diagnosis against ground truth: cause, category, localization, fix (static check)."""

from __future__ import annotations

import re
from typing import Any

from trainscope_server.taxonomy import CAUSES, category_of

# component -> keywords that count as correct localization
COMPONENT_KEYWORDS: dict[str, list[str]] = {
    "none": [],
    "config.lr": ["lr", "learning rate", "learning_rate"],
    "code.lr_scheduler": ["schedul", "warmup", "lr_lambda", "t_max"],
    "config.precision": ["fp16", "precision", "half", "loss scal", "amp", "gradscaler"],
    "model.per_sample_normalization": ["normaliz", "std", "division", "divide", "eps", "forward"],
    "config.grad_clip": ["clip", "grad_clip"],
    "model.backbone": ["frozen", "requires_grad", "backbone", "stem", "block", "encoder", "layers"],
    "model.head": ["head", "softmax", "output", "logit", "loss"],
    "model.init": ["init"],
    "data.labels": ["label"],
    "data.split": ["valid", "split", "leak", "eval"],
    "data.preprocessing": ["normaliz", "preprocess", "input", "pixel", "scale"],
    "data.attention_mask": ["mask", "pad"],
    "code.training_loop": [
        "zero_grad",
        "loop",
        "accum",
        "eval()",
        "train()",
        "eval mode",
        "train mode",
        "history",
        "detach",
        "item()",
        "graph",
    ],
    "data.train_size": [
        "train_size",
        "dataset size",
        "data size",
        "small",
        "regulari",
        "dropout",
        "weight_decay",
    ],
    "data.loader": ["loader", "worker", "transform", "augment", "pipeline"],
}


def localization_ok(component: str, location: str) -> bool:
    if component == "none":
        return True
    loc = location.lower()
    return any(k in loc for k in COMPONENT_KEYWORDS.get(component, []))


def fix_matches(pred: dict[str, Any] | None, accepted: list[dict[str, Any]], cause: str) -> bool:
    """Static check: does the proposed fix match one of the accepted fixes for the scenario?"""
    if not pred:
        return cause == "healthy"
    t = pred.get("type")
    for fx in accepted:
        if fx["type"] != t:
            continue
        if t == "none":
            return True
        if t == "code_fix" and pred.get("name") == fx.get("name"):
            return True
        if t == "config_change" and _norm(pred.get("field")) == _norm(fx.get("field")):
            return True
    return False


def _norm(s: Any) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"^(cfg|config)\.", "", s)
    return {"learning_rate": "lr", "clip_grad_norm": "grad_clip", "max_grad_norm": "grad_clip"}.get(s, s)


def grade(pred: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    if not pred:
        return {
            "cause_ok": False,
            "category_ok": False,
            "loc_ok": False,
            "fix_static_ok": False,
            "parsed": False,
        }
    cause = pred.get("cause") if pred.get("cause") in CAUSES else None
    return {
        "parsed": True,
        "cause_ok": cause == truth["cause"],
        "category_ok": (category_of(cause) if cause else pred.get("category")) == truth["category"],
        "loc_ok": localization_ok(
            truth["component"], f"{pred.get('location', '')} {pred.get('evidence', '')}"
        ),
        "fix_static_ok": fix_matches(pred.get("fix"), truth["fixes"], truth["cause"]),
    }
