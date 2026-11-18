"""One training loop for all benchmark tasks, with failure injection points.

``cfg`` is the *public* config (stored with the run, visible to assistants). ``bugs`` are code-level defects
injected by the scenario and are never stored with the run.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import trainscope as ts

from . import data as D
from .models import SmallCNN, TextClassifier, TinyGPT, bad_init_

DEFAULTS: dict[str, dict[str, Any]] = {
    "vision": dict(
        task="image_classification",
        model="small_resnet",
        dataset="synthetic-vision-10",
        num_classes=10,
        width=12,
        train_size=4000,
        val_size=500,
        batch_size=32,
        optimizer="sgd",
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4,
        scheduler="cosine",
        warmup_steps=0,
        total_steps=300,
        grad_clip=None,
        dropout=0.1,
        precision="fp32",
        num_workers=0,
        normalize=True,
        eval_every=30,
        grad_accum=1,
    ),
    "text_clf": dict(
        task="text_classification",
        model="transformer_encoder_2l",
        dataset="synthetic-keywords-4",
        task_type="padded_seq",
        num_classes=4,
        train_size=3000,
        val_size=600,
        seq_len=64,
        batch_size=32,
        optimizer="adamw",
        lr=1e-3,
        weight_decay=0.01,
        scheduler="linear",
        warmup_steps=30,
        total_steps=300,
        grad_clip=1.0,
        dropout=0.1,
        precision="fp32",
        num_workers=0,
        eval_every=30,
        grad_accum=1,
    ),
    "lm": dict(
        task="language_modeling",
        model="tiny_gpt_2l",
        dataset="synthetic-grammar-chars",
        seq_len=64,
        train_sentences=3000,
        val_sentences=300,
        batch_size=32,
        optimizer="adamw",
        lr=2e-3,
        weight_decay=0.01,
        scheduler="cosine",
        warmup_steps=30,
        total_steps=300,
        grad_clip=1.0,
        dropout=0.1,
        precision="fp32",
        num_workers=0,
        eval_every=30,
        grad_accum=1,
    ),
}

BUGS = {
    "double_softmax",
    "frozen_layers",
    "shuffled_labels",
    "val_leakage",
    "missing_normalization",
    "bad_padding_mask",
    "missing_zero_grad",
    "grad_accum_unscaled",
    "eval_mode_training",
    "memory_leak",
    "nan_from_op",
    "scheduler_misstep",
    "missing_warmup",
    "bad_init",
    "heavy_transform",
    "fp16_no_loss_scaling",
}


@dataclass
class Result:
    run_id: str
    steps: int
    initial_loss: float
    final_loss: float
    nonfinite: bool
    eval: dict[str, float] = field(default_factory=dict)
    best_eval_loss: float | None = None
    median_data_wait_frac: float = 0.0
    rss_growth_mb: float = 0.0
    wall_s: float = 0.0


def _build(task: str, cfg: dict[str, Any], bugs: set[str], seed: int):
    g = torch.Generator().manual_seed(seed)
    cost = 0.0015 if "heavy_transform" in bugs else 0.0
    if task == "vision":
        n_const = 4 if "nan_from_op" in bugs or cfg.get("_blank_images") else 0
        xtr, ytr = D.make_vision(cfg["train_size"], seed, cfg["num_classes"], noise=1.6, n_constant=n_const)
        xva, yva = D.make_vision(cfg["val_size"], seed + 10_000, cfg["num_classes"], noise=1.6)
        if cfg.get("normalize", True) and "missing_normalization" not in bugs:
            xtr, xva = (xtr - D.VISION_MEAN) / D.VISION_STD, (xva - D.VISION_MEAN) / D.VISION_STD
        if "val_leakage" in bugs:
            xtr, ytr = torch.cat([xtr] + [xva] * 3), torch.cat([ytr] + [yva] * 3)
        if "shuffled_labels" in bugs:
            ytr = ytr[torch.randperm(len(ytr), generator=g)]
        train = D.TensorDS(xtr, ytr, transform_cost_s=cost)
        val = D.TensorDS(xva, yva)
        eps = 0.0 if "nan_from_op" in bugs else 1e-5
        model = SmallCNN(
            cfg["num_classes"],
            width=cfg.get("width", 12),
            dropout=cfg["dropout"],
            per_sample_norm_eps=eps,
            input_norm="missing_normalization" not in bugs,
        )
        return model, train, val
    if task == "text_clf":
        xtr, mtr, ytr = D.make_text(cfg["train_size"], seed)
        xva, mva, yva = D.make_text(cfg["val_size"], seed + 10_000)
        if "val_leakage" in bugs:
            xtr, mtr, ytr = (
                torch.cat([xtr] + [xva] * 3),
                torch.cat([mtr] + [mva] * 3),
                torch.cat([ytr] + [yva] * 3),
            )
        if "shuffled_labels" in bugs:
            ytr = ytr[torch.randperm(len(ytr), generator=g)]
        model = TextClassifier(D.TEXT_VOCAB, cfg["num_classes"], dropout=cfg["dropout"])
        return model, D.TensorDS(xtr, mtr, ytr, transform_cost_s=cost), D.TensorDS(xva, mva, yva)
    if task == "lm":
        text_tr = D.make_corpus(cfg["train_sentences"], seed)
        text_va = D.make_corpus(cfg["val_sentences"], seed + 10_000)
        tr = D.CharLM(text_tr, cfg["seq_len"])
        va = D.CharLM(text_va, cfg["seq_len"], vocab=tr.vocab)
        cfg["num_classes"] = len(tr.vocab)  # vocabulary size (chance loss = ln V)
        xs = torch.stack([tr[i][0] for i in range(len(tr))])
        ys = torch.stack([tr[i][1] for i in range(len(tr))])
        if "val_leakage" in bugs:
            xv = torch.stack([va[i][0] for i in range(len(va))])
            yv = torch.stack([va[i][1] for i in range(len(va))])
            xs, ys = torch.cat([xs] + [xv] * 3), torch.cat([ys] + [yv] * 3)
        if "shuffled_labels" in bugs:
            ys = ys[torch.randperm(len(ys), generator=g)]
        xv = torch.stack([va[i][0] for i in range(len(va))])
        yv = torch.stack([va[i][1] for i in range(len(va))])
        model = TinyGPT(len(tr.vocab), seq_len=cfg["seq_len"], dropout=cfg["dropout"])
        return model, D.TensorDS(xs, ys, transform_cost_s=cost), D.TensorDS(xv, yv)
    raise ValueError(task)


def _forward(task: str, model, batch, bugs: set[str], dtype=torch.float32):
    if task == "vision":
        x, y, ids = batch
        return model(x.to(dtype)), y, ids, None
    if task == "text_clf":
        x, m, y, ids = batch
        if "bad_padding_mask" in bugs:
            m = (x != 1).long()  # wrong pad id -> nothing is masked
        return model(x, m), y, ids, m
    x, y, ids = batch
    return model(x), y, ids, None


def _loss(task: str, logits, y, bugs: set[str]):
    if "double_softmax" in bugs:
        logits = torch.softmax(logits, dim=-1)
    if task == "lm":
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
    return F.cross_entropy(logits.float(), y)


def _lr_lambda(cfg: dict[str, Any], bugs: set[str], steps_per_epoch: int):
    total, warm = cfg["total_steps"], cfg.get("warmup_steps", 0) or 0
    if "missing_warmup" in bugs:
        warm = 0
    sched = cfg.get("scheduler", "constant")
    if "scheduler_misstep" in bugs:
        # T_max given in epochs, but the scheduler is stepped every batch
        total = max(2, math.ceil(cfg["total_steps"] / steps_per_epoch))
        warm = 0

    def f(step: int) -> float:
        if warm and step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(total - warm, 1)
        if "scheduler_misstep" in bugs:
            return 0.5 * (1 + math.cos(math.pi * prog))  # periodic: wraps around
        prog = min(max(prog, 0.0), 1.0)
        if sched == "cosine":
            return 0.5 * (1 + math.cos(math.pi * prog))
        if sched == "linear":
            return 1 - prog
        return 1.0

    return f


@torch.no_grad()
def _evaluate(task, model, loader, bugs, dtype) -> dict[str, float]:
    was = model.training
    model.eval()
    tot, correct, n = 0.0, 0, 0
    for batch in loader:
        logits, y, _, _ = _forward(task, model, batch, bugs, dtype)
        loss = _loss(task, logits, y, bugs)
        bs = y.numel()
        tot += float(loss) * bs
        correct += int((logits.argmax(-1) == y).sum())
        n += bs
    if was and "eval_mode_training" not in bugs:
        model.train()
    out = {"loss": tot / max(n, 1)}
    if task != "lm":
        out["accuracy"] = correct / max(n, 1)
    return out


def train(
    task: str,
    cfg: dict[str, Any] | None = None,
    bugs: set[str] | None = None,
    seed: int = 0,
    run_id: str | None = None,
    name: str | None = None,
    db: str | None = None,
    stdout_log: str | None = None,
    project: str = "bench",
    max_steps: int | None = None,
    live: bool = False,
    mode: str | None = None,
    url: str | None = None,
) -> Result:
    bugs = set(bugs or ())
    unknown = bugs - BUGS
    if unknown:
        raise ValueError(f"unknown bugs {unknown}")
    cfg = {**DEFAULTS[task], **(cfg or {})}
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(cfg.get("_threads", 1)))  # 1 thread is fastest for these tiny models
    model, train_ds, val_ds = _build(task, cfg, bugs, seed)
    total = int(max_steps or cfg["total_steps"])
    cfg["total_steps"] = total
    if "bad_init" in bugs:
        bad_init_(model)
    if "frozen_layers" in bugs:
        params = list(model.named_parameters())
        for n, p in params:
            if not n.startswith("head"):
                p.requires_grad_(False)
    dtype = torch.float32
    if cfg.get("precision") == "fp16":
        model = model.half()
        dtype = torch.float16
    nw = int(cfg.get("num_workers", 0) or 0)
    loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        persistent_workers=nw > 0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=256)
    params = [p for p in model.parameters() if p.requires_grad] or list(model.parameters())
    if cfg["optimizer"] == "sgd":
        opt = torch.optim.SGD(
            params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.9), weight_decay=cfg.get("weight_decay", 0)
        )
    else:
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0))
    steps_per_epoch = max(1, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg, bugs, steps_per_epoch))

    public_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    public_cfg["seed"] = seed
    run = ts.init(
        project,
        config=public_cfg,
        run_id=run_id,
        name=name,
        db=db,
        stdout_log=stdout_log,
        system_interval=0.5,
        poll_commands=live,
        command_interval=1.0,
        mode=mode,
        url=url,
    )
    ts.watch(model, optimizer=opt, grad_every=10, act_every=30)
    run.print(
        f"task={cfg['task']} model={cfg['model']} params={sum(p.numel() for p in model.parameters())} "
        f"steps={total} batch={cfg['batch_size']} lr={cfg['lr']}"
    )

    if "eval_mode_training" not in bugs:
        model.train()
    else:
        model.eval()
    history: list[Any] = []
    accum = int(cfg.get("grad_accum", 1) or 1)
    step, micro = 0, 0
    losses: list[float] = []
    waits: list[float] = []
    nonfinite_streak = 0
    t_start = time.time()
    rss0 = run.sampler.sample().get("sys/proc_rss_mb", 0.0) if run.sampler else 0.0
    tr_correct = tr_n = 0
    eval_hist: list[dict[str, float]] = []
    timed = ts.wrap_loader(loader)
    done = False
    if "missing_zero_grad" not in bugs:
        opt.zero_grad(set_to_none=True)
    while not done:
        for batch in timed:
            logits, y, ids, m = _forward(task, model, batch, bugs, dtype)
            loss = _loss(task, logits, y, bugs)
            back = loss if (accum == 1 or "grad_accum_unscaled" in bugs) else loss / accum
            back.backward()
            micro += 1
            if "memory_leak" in bugs:
                history.append(loss)  # keeps the whole autograd graph alive
            else:
                history.append(float(loss))
            tr_correct += int((logits.detach().argmax(-1) == y).sum())
            tr_n += y.numel()
            if micro % accum != 0:
                continue
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()
            sched.step()
            if "missing_zero_grad" not in bugs:
                opt.zero_grad(set_to_none=True)
            lv = float(loss)
            losses.append(lv)
            ts.log(step, loss=lv, lr=sched.get_last_lr()[0])
            labels = y if task != "lm" else None
            ts.log_batch(
                step,
                labels=labels,
                attention_mask=m,
                sample_ids=ids.tolist(),
                num_classes=cfg.get("num_classes") if task != "lm" else None,
            )
            if step % 10 == 0:
                run.print(f"step {step:5d} | loss {lv:.4f} | lr {sched.get_last_lr()[0]:.3g}")
            if (step + 1) % cfg["eval_every"] == 0 or step + 1 == total:
                ev = _evaluate(task, model, val_loader, bugs, dtype)
                eval_hist.append(ev)
                ts.log_eval(step, **ev)
                if tr_n and task != "lm":
                    ts.log_scalars(step, {"train/accuracy": tr_correct / tr_n})
                tr_correct = tr_n = 0
                run.print(f"eval  step {step:5d} | " + " | ".join(f"{k} {v:.4f}" for k, v in ev.items()))
            nonfinite_streak = nonfinite_streak + 1 if not math.isfinite(lv) else 0
            step += 1
            if nonfinite_streak >= 20:
                run.print(f"step {step} | loss is {lv}; aborting")
                done = True
                break
            if step >= total:
                done = True
                break
    rss1 = run.sampler.sample().get("sys/proc_rss_mb", 0.0) if run.sampler else 0.0
    run.flush()
    df_wait = []
    try:
        from trainscope_server.db import open_store

        if run.mode == "local":
            st = open_store(db)
            df_wait = st.metrics_df(run.id, names=["data_wait_frac"])["value"].tolist()
    except Exception:
        pass
    waits = df_wait
    status = "crashed" if nonfinite_streak >= 20 else "finished"
    ts.finish(status)
    finite = [v for v in losses if math.isfinite(v)]
    k = max(1, len(finite) // 10)
    evl = [e["loss"] for e in eval_hist if math.isfinite(e["loss"])]
    return Result(
        run_id=run.id,
        steps=step,
        initial_loss=float(np.mean(finite[:k])) if finite else float("nan"),
        final_loss=float(np.mean(finite[-k:])) if finite and len(finite) == len(losses) else float("nan"),
        nonfinite=len(finite) != len(losses),
        eval=eval_hist[-1] if eval_hist else {},
        best_eval_loss=min(evl) if evl else None,
        median_data_wait_frac=float(np.median(waits)) if waits else 0.0,
        rss_growth_mb=rss1 - rss0,
        wall_s=time.time() - t_start,
    )
