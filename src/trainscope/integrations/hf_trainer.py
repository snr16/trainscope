"""Hugging Face Trainer integration: ``Trainer(..., callbacks=[TrainScopeCallback()])``."""

from __future__ import annotations

from typing import Any

import trainscope as ts

try:
    from transformers import TrainerCallback
except ImportError:  # keep import-time cheap / optional
    TrainerCallback = object  # type: ignore[misc,assignment]


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
        cfg["batch_size"] = args.per_device_train_batch_size
        self.run = ts.init(self.project, config=cfg, **self.init_kwargs)
        if model is not None:
            self.run.watch(model, optimizer=optimizer, grad_every=self.grad_every, act_every=self.act_every)

    def on_optimizer_step(self, args, state, control, **kw):
        # grads are still populated here (before zero_grad)
        if self.run and self.run.watcher and state.global_step % self.grad_every == 0:
            metrics, events = self.run.watcher.collect(state.global_step)
            for n, layer, v in metrics:
                self.run._metric(n, layer, v, state.global_step)

    def on_log(self, args, state, control, logs=None, **kw):
        if not self.run or not logs:
            return
        train = {k: v for k, v in logs.items() if not k.startswith("eval_") and isinstance(v, int | float)}
        evals = {
            k.removeprefix("eval_"): v
            for k, v in logs.items()
            if k.startswith("eval_") and isinstance(v, int | float)
        }
        if "learning_rate" in train:
            train["lr"] = train.pop("learning_rate")
        if train:
            self.run.log(state.global_step, **train)
        if evals:
            self.run.log_eval(state.global_step, **evals)

    def on_train_end(self, args, state, control, **kw):
        if self.run:
            self.run.finish()
