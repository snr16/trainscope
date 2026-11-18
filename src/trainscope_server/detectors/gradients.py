"""Exploding gradients (global / per-layer, incl. steady accumulation) and vanishing gradients / dead units."""

from __future__ import annotations

import numpy as np

from .base import Detector, Finding, RunData, finite, trend


class ExplodingGradientDetector(Detector):
    name = "exploding_gradients"

    def __init__(self, ratio: float = 20.0):
        self.ratio = ratio

    def run(self, data: RunData) -> list[Finding]:
        g = finite(data.scalar("grad/global_norm"))
        out: list[Finding] = []
        if len(g) < 5:
            return out
        win = max(3, len(g) // 5)
        base = float(g.iloc[:win].median())
        later = g.iloc[
            win:
        ]  # a blow-up is growth *after* the baseline window (large initial grads = bad init)
        peak_step = int(later.idxmax()) if len(later) else int(g.idxmax())
        peak = float(later.max()) if len(later) else float(g.max())
        clip_off = not data.cfg("grad_clip", "max_grad_norm", default=None)
        layers = data.layers("grad/norm")
        top_layers = []
        if not layers.empty:
            lmed = layers.iloc[: max(3, len(layers) // 5)].median()
            growth = (
                (layers.max() / (lmed + 1e-12))
                .replace([np.inf], np.nan)
                .dropna()
                .sort_values(ascending=False)
            )
            top_layers = [{"layer": k, "growth": float(v)} for k, v in growth.head(3).items()]
        slope, r = trend(g)
        if base > 0 and peak / base > self.ratio:
            out.append(
                Finding(
                    self.name,
                    "critical",
                    int(g[g > base * 5].index[0]) if (g > base * 5).any() else peak_step,
                    peak_step,
                    f"global grad norm rose {peak / base:.0f}x (median {base:.3g} -> {peak:.3g} at step {peak_step})"
                    + (f"; largest growth in {top_layers[0]['layer']}" if top_layers else ""),
                    {
                        "baseline_grad_norm": base,
                        "peak_grad_norm": peak,
                        "peak_step": peak_step,
                        "top_layers": top_layers,
                        "trend_r": r,
                    },
                    ["exploding_no_clip", "lr_too_high", "missing_zero_grad"]
                    if clip_off
                    else ["lr_too_high", "missing_zero_grad", "exploding_no_clip"],
                    0.7,
                )
            )
        # steady, near-linear growth of the gradient norm: accumulation without zero_grad / unscaled accumulation
        if r > 0.9 and len(g) >= 8 and float(g.iloc[-1]) > 4 * max(base, 1e-12):
            out.append(
                Finding(
                    "grad_norm_growth",
                    "critical",
                    int(g.index[0]),
                    int(g.index[-1]),
                    f"grad norm grows steadily with step (r={r:.2f}, {float(g.iloc[0]):.3g} -> {float(g.iloc[-1]):.3g})",
                    {
                        "trend_r": r,
                        "slope_per_step": slope,
                        "first": float(g.iloc[0]),
                        "last": float(g.iloc[-1]),
                    },
                    ["missing_zero_grad", "grad_accum_unscaled"],
                    0.85,
                )
            )
        return out


class VanishingGradientDetector(Detector):
    name = "vanishing_gradients"

    def __init__(self, rel: float = 1e-4, dead_frac: float = 0.25):
        self.rel, self.dead_frac = rel, dead_frac

    def run(self, data: RunData) -> list[Finding]:
        out: list[Finding] = []
        layers = data.layers("grad/norm")
        if not layers.empty:
            med = layers.median()
            ref = float(med.max())
            tiny = med[(med < ref * self.rel) & (med > 0)]
            if len(tiny):
                out.append(
                    Finding(
                        self.name,
                        "warning",
                        int(layers.index[0]),
                        int(layers.index[-1]),
                        f"{len(tiny)} layer(s) with near-zero gradients (e.g. {tiny.index[0]}: {tiny.iloc[0]:.1e} "
                        f"vs max {ref:.1e})",
                        {"layers": {k: float(v) for k, v in tiny.head(8).items()}, "max_layer_grad": ref},
                        ["bad_init", "double_softmax"],
                        0.5,
                    )
                )
        dead = data.layers("act/dead_frac")
        if not dead.empty:
            last = dead.iloc[-max(1, len(dead) // 3) :].mean()
            bad = last[last > self.dead_frac]
            if len(bad):
                out.append(
                    Finding(
                        "dead_units",
                        "warning",
                        int(dead.index[0]),
                        int(dead.index[-1]),
                        f"{len(bad)} layer(s) with >{self.dead_frac:.0%} dead units (e.g. {bad.index[0]}: {bad.iloc[0]:.0%})",
                        {"layers": {k: float(v) for k, v in bad.head(8).items()}},
                        ["lr_too_high", "bad_init", "missing_normalization"],
                        0.5,
                    )
                )
        return out


class GradCarryoverDetector(Detector):
    """`.grad` holds much more than the latest backward pass contributed: gradients are not being zeroed."""

    name = "grad_carryover"

    def run(self, data: RunData) -> list[Finding]:
        c = finite(data.scalar("grad/carryover_ratio"))
        if len(c) < 4:
            return []
        accum = int(data.cfg("grad_accum", "gradient_accumulation_steps", default=1) or 1)
        late = c.iloc[len(c) // 2 :]
        med = float(late.median())
        # without accumulation .grad must equal the latest backward exactly (ratio 1.00 in healthy runs)
        limit = 1.15 if accum == 1 else 2.0 * accum + 1.0
        if med <= limit:
            return []
        return [
            Finding(
                self.name,
                "critical",
                int(c.index[0]),
                int(c.index[-1]),
                f".grad is {med:.1f}x the gradient of the latest backward pass (expected <= {accum} with "
                f"grad_accum={accum}); gradients carry over between optimizer steps",
                {
                    "median_carryover_ratio": med,
                    "grad_accum": accum,
                    "first": float(c.iloc[0]),
                    "last": float(c.iloc[-1]),
                },
                ["missing_zero_grad", "grad_accum_unscaled"],
                0.9,
            )
        ]
