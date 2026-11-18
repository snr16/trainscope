"""Fixed label set shared by detectors, benchmark ground truth, and grading."""

from __future__ import annotations

CAUSES: dict[str, tuple[str, str]] = {
    # cause_id: (category, description)
    "lr_too_high": ("optimization", "Learning rate far too high"),
    "missing_warmup": ("optimization", "No learning-rate warmup where one is needed"),
    "scheduler_misstep": ("optimization", "LR scheduler stepped at the wrong frequency"),
    "fp16_no_loss_scaling": ("numerical", "FP16 training without loss scaling"),
    "nan_from_op": ("numerical", "An operation produces NaN/Inf (e.g. division by zero, log(0))"),
    "exploding_no_clip": ("numerical", "Exploding gradients without clipping"),
    "frozen_layers": ("model", "Layers accidentally frozen"),
    "double_softmax": ("model", "Softmax applied before cross-entropy"),
    "bad_init": ("model", "Wrong weight initialization"),
    "shuffled_labels": ("data", "Labels shuffled relative to inputs"),
    "val_leakage": ("data", "Validation data leaking into training"),
    "missing_normalization": ("data", "Inputs not normalized"),
    "bad_padding_mask": ("data", "Wrong attention / padding mask"),
    "missing_zero_grad": ("loop", "optimizer.zero_grad() never called"),
    "grad_accum_unscaled": ("loop", "Gradient accumulation without dividing the loss"),
    "eval_mode_training": ("loop", "Model left in eval mode during training"),
    "overfitting": ("generalization", "Severe overfitting"),
    "dataloader_bottleneck": ("systems", "Data loader is the bottleneck"),
    "memory_leak": ("systems", "Memory leak (e.g. storing tensors that hold the graph)"),
    "healthy": ("healthy", "No problem"),
}

CATEGORIES = sorted({c for c, _ in CAUSES.values()})

FIX_TYPES = {
    "config_change": "Change a config field: {'type':'config_change','field':..., 'value':...}",
    "code_fix": "Apply a named code fix: {'type':'code_fix','name':<cause_id>}",
    "none": "No fix needed",
}


def category_of(cause: str) -> str:
    return CAUSES.get(cause, ("unknown", ""))[0]
