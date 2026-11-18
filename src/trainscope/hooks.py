"""Gradient / weight / activation statistics via PyTorch hooks.

All statistics are computed on the training device and gathered with a single device->host transfer per
sampling step (``torch.stack(...).cpu()``), which keeps GPU syncs to a minimum.

Layers are modules that own parameters directly (``conv1``, ``layer1.0.bn1``, ...); stats of a module's
weight and bias are combined.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import nn


def _layer_of(param_name: str) -> str:
    return param_name.rsplit(".", 1)[0] if "." in param_name else param_name


class ModelWatcher:
    def __init__(
        self,
        model: nn.Module,
        grad_every: int = 50,
        act_every: int = 200,
        max_act_layers: int = 12,
        track_update_ratio: bool = True,
    ):
        self.model = model
        self.grad_every = max(1, grad_every)
        self.act_every = max(1, act_every)
        self.track_update_ratio = track_update_ratio
        self.groups: dict[str, list[tuple[str, nn.Parameter]]] = defaultdict(list)
        for n, p in model.named_parameters():
            self.groups[_layer_of(n)].append((n, p))
        self.layer_names = list(self.groups)
        self._snapshot: dict[str, torch.Tensor] | None = None
        self._act_on = False
        self._act_stats: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._register_activation_hooks(max_act_layers)
        # gradient stats are captured right after backward (before clipping / unscaling / zero_grad)
        self._grad_on = True
        self._grad_stats: dict[str, tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]] = {}
        # squared norm of the gradient *arriving* in the latest backward (before it is added to .grad)
        self._incoming: dict[str, torch.Tensor] = {}
        self._register_grad_hooks()

    # --------------------------------------------------------------- gradients
    def _register_grad_hooks(self) -> None:
        if not hasattr(torch.Tensor, "register_post_accumulate_grad_hook"):
            return  # torch < 2.1: fall back to reading .grad at log time
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._handles.append(p.register_hook(self._make_incoming_hook(n)))
                self._handles.append(p.register_post_accumulate_grad_hook(self._make_grad_hook(n)))

    def _make_incoming_hook(self, name: str):
        def hook(grad: torch.Tensor) -> None:
            if self._grad_on:
                self._incoming[name] = grad.detach().float().pow(2).sum()

        return hook

    def _make_grad_hook(self, name: str):
        def hook(p: torch.Tensor) -> None:
            if not self._grad_on or p.grad is None:
                return
            with torch.no_grad():
                g = p.grad.detach().float()
                # overwrite on every micro-batch: the last call sees the fully accumulated gradient
                self._grad_stats[name] = (
                    g.pow(2).sum(),
                    (g == 0).sum(),
                    g.numel(),
                    (~torch.isfinite(g)).any(),
                )

        return hook

    # ------------------------------------------------------------ activations
    def _register_activation_hooks(self, max_layers: int) -> None:
        leaves = [
            (n, m)
            for n, m in self.model.named_modules()
            if n and not list(m.children()) and not isinstance(m, nn.Dropout | nn.Identity)
        ]
        if not leaves:
            return
        stride = max(1, len(leaves) // max_layers)
        chosen = leaves[::stride][:max_layers]
        for name, mod in chosen:
            self._handles.append(mod.register_forward_hook(self._make_act_hook(name)))

    def _make_act_hook(self, name: str):
        def hook(_mod, _inp, out):
            if not self._act_on:
                return
            t = out[0] if isinstance(out, tuple | list) else out
            if not isinstance(t, torch.Tensor) or not t.is_floating_point():
                return
            with torch.no_grad():
                t = t.detach().float()
                # dead units: features that are <= 0 for every element in the batch
                if t.dim() >= 2:
                    feat = t.transpose(0, 1).reshape(t.shape[1], -1) if t.dim() > 2 else t.T
                    dead = (feat.max(dim=1).values <= 0).float().mean()
                else:
                    dead = (t <= 0).float().mean()
                self._act_stats[name] = torch.stack([t.mean(), t.std(unbiased=False), dead])

        return hook

    def before_step(self, step: int) -> None:
        """Called by Run.log for the *next* step, so the following forward pass can be sampled."""
        nxt = step + 1
        self._act_on = nxt % self.act_every == 0
        self._grad_on = nxt % self.grad_every == 0
        self._grad_stats = {}
        self._incoming = {}
        if self.track_update_ratio and nxt % self.grad_every == 0:
            with torch.no_grad():
                self._snapshot = {n: p.detach().clone() for n, p in self.model.named_parameters()}

    # ---------------------------------------------------------------- collect
    def collect(self, step: int) -> tuple[list[tuple[str, str, float]], list[dict[str, Any]]]:
        """Return (metrics [(name, layer, value)], events) for this step."""
        metrics: list[tuple[str, str, float]] = []
        events: list[dict[str, Any]] = []
        if self._act_stats:
            names = list(self._act_stats)
            vals = torch.stack([self._act_stats[n] for n in names]).cpu().tolist()
            for n, (mean, std, dead) in zip(names, vals, strict=True):
                metrics += [("act/mean", n, mean), ("act/std", n, std), ("act/dead_frac", n, dead)]
            self._act_stats = {}
        if step % self.grad_every != 0:
            return metrics, events
        rows: list[torch.Tensor] = []
        layers: list[str] = []
        with torch.no_grad():
            for layer, params in self.groups.items():
                gsq: Any = None
                wsq: Any = None
                usq: Any = None
                zeros: Any = None
                numel: Any = None
                nonfinite = None
                for n, p in params:
                    w = p.detach()
                    wsq = w.float().pow(2).sum() if wsq is None else wsq + w.float().pow(2).sum()
                    if self._snapshot is not None and n in self._snapshot:
                        d = (w - self._snapshot[n]).float().pow(2).sum()
                        usq = d if usq is None else usq + d
                    stats = self._grad_stats.get(n)
                    if stats is None and p.grad is not None:
                        g = p.grad.detach().float()
                        stats = (g.pow(2).sum(), (g == 0).sum(), g.numel(), (~torch.isfinite(g)).any())
                    if stats is not None:
                        s, z, ne, nf = stats
                        gsq = s if gsq is None else gsq + s
                        zeros = z if zeros is None else zeros + z
                        numel = ne if numel is None else numel + ne
                        nonfinite = nf if nonfinite is None else nonfinite | nf
                dev = wsq.device
                nan = torch.tensor(float("nan"), device=dev)
                rows.append(
                    torch.stack(
                        [
                            gsq.sqrt() if gsq is not None else nan,
                            wsq.sqrt(),
                            usq.sqrt() if usq is not None else nan,
                            (zeros.float() / numel) if zeros is not None else nan,
                            nonfinite.float() if nonfinite is not None else torch.zeros((), device=dev),
                            torch.tensor(1.0 if any(p.requires_grad for _, p in params) else 0.0, device=dev),
                        ]
                    )
                )
                layers.append(layer)
            self._snapshot = None
            carry = None
            shared = [n for n in self._incoming if n in self._grad_stats]
            if shared:
                acc = torch.stack([self._grad_stats[n][0] for n in shared]).sum()
                inc = torch.stack([self._incoming[n] for n in shared]).sum()
                carry = (acc / (inc + 1e-30)).sqrt()
            self._grad_stats = {}
            self._incoming = {}
            if not rows:
                return metrics, events
            if carry is not None:
                rows.append(torch.stack([carry.to(rows[0].device)] * 6))  # piggy-back on the same transfer
            table = torch.stack([r.to(rows[0].device) for r in rows]).cpu().tolist()  # one sync
            if carry is not None:
                metrics.append(("grad/carryover_ratio", "", table.pop()[0]))
        gsum = 0.0
        has_grad = False
        bad_layers = []
        for layer, (gn, wn, un, zf, nf, trainable) in zip(layers, table, strict=True):
            if gn == gn:  # not NaN-placeholder
                metrics.append(("grad/norm", layer, gn))
                metrics.append(("grad/zero_frac", layer, zf))
                gsum += gn * gn
                has_grad = True
            metrics.append(("weight/norm", layer, wn))
            if un == un:
                metrics.append(("weight/update_ratio", layer, un / (wn + 1e-12)))
            metrics.append(("weight/trainable", layer, trainable))
            if nf > 0:
                bad_layers.append(layer)
        if has_grad:
            metrics.append(("grad/global_norm", "", gsum**0.5))
        metrics.append(("model/train_mode", "", 1.0 if self.model.training else 0.0))
        if bad_layers:
            events.append(
                {
                    "type": "nonfinite_grad",
                    "step": step,
                    "payload": {"layers": bad_layers[:20], "first_layer": bad_layers[0]},
                }
            )
        return metrics, events

    def locate_nonfinite(self) -> dict[str, Any]:
        """Full scan (used once when loss goes NaN/Inf): first layer with non-finite weights or grads."""
        with torch.no_grad():
            for layer, params in self.groups.items():
                for n, p in params:
                    if not torch.isfinite(p).all():
                        return {"layer": layer, "param": n, "where": "weight"}
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        return {"layer": layer, "param": n, "where": "grad"}
        return {}

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
