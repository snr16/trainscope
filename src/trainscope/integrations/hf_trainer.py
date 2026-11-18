"""Hugging Face Trainer integration: ``Trainer(..., callbacks=[TrainScopeCallback(project="bert")])``.

Per optimizer step (``on_step_end``) the callback runs TrainScope's step bookkeeping: step time, sampled
gradient/weight/activation stats (gradients are captured by hooks during backward, i.e. before clipping and
zero_grad), and approved control commands (checkpoint / pause / LR change) at the step boundary. The
Trainer's own logs (loss, learning rate, eval metrics) are recorded in ``on_log``.
"""

from __future__ import annotations

from typing import Any

import trainscope as ts

try:
    from transformers import TrainerCallback
except ImportError:  # keep import-time cheap / optional
    TrainerCallback = object  # type: ignore[misc,assignment]

# Trainer log keys that are bookkeeping, not training signals
_SKIP = {"epoch", "total_flos", "train_runtime", "train_samples_per_second", "train_steps_per_second"}


class TrainScopeCallback(TrainerCallback):  # type: ignore[misc]
    def __init__(self, project: str = "hf", grad_every: int = 50, act_every: int = 200, **init_kwargs: Any):
        self.project = project
        self.grad_every = grad_every
        self.act_every = act_every
        self.init_kwargs = init_kwargs
        self.run: ts.Run | None = None

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kw):
        if not state.is_world_process_zero:
            return
        cfg = {k: v for k, v in args.to_dict().items() if isinstance(v, int | float | str | bool)}
        cfg["batch_size"] = args.per_device_train_batch_size * max(1, args.gradient_accumulation_steps)
        cfg["lr"] = args.learning_rate
        cfg["grad_accum"] = args.gradient_accumulation_steps
        cfg["grad_clip"] = args.max_grad_norm or None
        cfg["total_steps"] = state.max_steps
        cfg["warmup_steps"] = args.get_warmup_steps(state.max_steps) if state.max_steps else args.warmup_steps
        cfg["scheduler"] = str(getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type))
        self.run = ts.init(self.project, config=cfg, **self.init_kwargs)
        if model is not None:
            self.run.watch(model, optimizer=optimizer, grad_every=self.grad_every, act_every=self.act_every)

    def on_step_end(self, args, state, control, **kw):
        # one call per optimizer step: step time, sampled layer stats, commands at the step boundary
        if self.run is not None:
            self.run.log(state.global_step)

    def on_log(self, args, state, control, logs=None, **kw):
        if self.run is None or not logs:
            return
        scalars: dict[str, float] = {}
        for k, v in logs.items():
            if k in _SKIP or not isinstance(v, int | float):
                continue
            if k.startswith("eval_"):
                scalars["eval/" + k.removeprefix("eval_")] = v
            elif k == "learning_rate":
                scalars["lr"] = v
            elif k == "grad_norm":
                scalars["hf/grad_norm"] = v  # Trainer's own (post-clip) norm; TrainScope's is pre-clip
            else:
                scalars[k] = v
        self.run.log_scalars(state.global_step, scalars)

    def on_train_end(self, args, state, control, **kw):
        if self.run is not None:
            self.run.finish()
