"""Turn long series into compact, LLM-friendly facts: downsampled points + computed statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _r(x: Any, sig: int = 4) -> Any:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    if not math.isfinite(f):
        return str(f)
    if f == 0:
        return 0.0
    return float(f"{f:.{sig}g}")


def lttb(x: np.ndarray, y: np.ndarray, n_out: int) -> np.ndarray:
    """Largest-Triangle-Three-Buckets downsampling. Returns indices into x/y."""
    n = len(x)
    if n_out >= n or n_out < 3:
        return np.arange(n)
    idx = [0]
    every = (n - 2) / (n_out - 2)
    a = 0
    for i in range(n_out - 2):
        start = int(math.floor((i + 1) * every)) + 1
        end = min(int(math.floor((i + 2) * every)) + 1, n)
        avg_x, avg_y = x[start:end].mean(), y[start:end].mean()
        rs, re = int(math.floor(i * every)) + 1, int(math.floor((i + 1) * every)) + 1
        area = np.abs((x[a] - avg_x) * (y[rs:re] - y[a]) - (x[a] - x[rs:re]) * (avg_y - y[a]))
        a = rs + int(np.argmax(area)) if len(area) else rs
        idx.append(a)
    idx.append(n - 1)
    return np.array(idx)


def change_points(s: pd.Series, max_points: int = 3, min_seg: int = 10) -> list[dict[str, Any]]:
    """Greedy mean-shift change points (binary segmentation on log-scaled values where positive)."""
    if len(s) < 2 * min_seg:
        return []
    y = s.to_numpy(dtype=float)
    yt = np.log(np.clip(y, 1e-12, None)) if (y > 0).all() else y
    out: list[tuple[int, float]] = []
    segments = [(0, len(yt))]
    for _ in range(max_points):
        best = None
        for a, b in segments:
            if b - a < 2 * min_seg:
                continue
            seg = yt[a:b]
            csum = np.cumsum(seg)
            total = csum[-1]
            ks = np.arange(min_seg, b - a - min_seg)
            if not len(ks):
                continue
            left = csum[ks - 1] / ks
            right = (total - csum[ks - 1]) / (b - a - ks)
            gain = ks * (b - a - ks) / (b - a) * (left - right) ** 2
            k = int(np.argmax(gain))
            if best is None or gain[k] > best[0]:
                best = (float(gain[k]), a, b, a + int(ks[k]))
        if best is None or best[0] < 1e-6 * len(yt) or best[0] < 0.5 * np.var(yt) * 4:
            break
        _, a, b, k = best
        out.append((k, best[0]))
        segments.remove((a, b))
        segments += [(a, k), (k, b)]
    res = []
    for k, _ in sorted(out):
        before = float(np.mean(y[max(0, k - min_seg) : k]))
        after = float(np.mean(y[k : k + min_seg]))
        res.append(
            {
                "step": int(s.index[k]),
                "before": _r(before),
                "after": _r(after),
                "change_pct": _r((after / before - 1) * 100 if before else None, 3),
            }
        )
    return res


def series_summary(s: pd.Series, max_points: int = 60, name: str = "") -> dict[str, Any]:
    """Downsampled series + facts. NaN/Inf are reported, not silently dropped."""
    if s.empty:
        return {"name": name, "n": 0, "summary": f"{name}: no data"}
    vals = s.to_numpy(dtype=float)
    finite_mask = np.isfinite(vals)
    fs = s[finite_mask]
    out: dict[str, Any] = {"name": name, "n": int(len(s)), "step_range": [int(s.index[0]), int(s.index[-1])]}
    if (~finite_mask).any():
        out["nonfinite_count"] = int((~finite_mask).sum())
        out["first_nonfinite_step"] = int(s.index[~finite_mask][0])
    if fs.empty:
        out["summary"] = f"{name}: all {len(s)} values are NaN/Inf"
        return out
    head = fs.iloc[: max(1, len(fs) // 10)].mean()
    tail = fs.iloc[-max(1, len(fs) // 10) :].mean()
    out.update(
        {
            "first": _r(fs.iloc[0]),
            "last": _r(fs.iloc[-1]),
            "min": _r(fs.min()),
            "min_step": int(fs.idxmin()),
            "max": _r(fs.max()),
            "max_step": int(fs.idxmax()),
            "mean": _r(fs.mean()),
            "start_avg": _r(head),
            "end_avg": _r(tail),
            "change_pct": _r((tail / head - 1) * 100 if head else None, 3),
        }
    )
    if len(fs) >= 3:
        x = fs.index.to_numpy(dtype=float)
        r = float(np.corrcoef(x, fs.to_numpy())[0, 1]) if fs.std() > 0 else 0.0
        out["trend"] = "rising" if r > 0.5 else "falling" if r < -0.5 else "flat/noisy"
        out["trend_r"] = _r(r, 3)
    cps = change_points(fs)
    if cps:
        out["change_points"] = cps
    idx = lttb(fs.index.to_numpy(dtype=float), fs.to_numpy(dtype=float), max_points)
    out["points"] = [[int(fs.index[i]), _r(fs.iloc[i])] for i in idx]
    desc = f"{name}: {out['first']} -> {out['last']} over steps {out['step_range'][0]}-{out['step_range'][1]}"
    desc += f" (min {out['min']} @ {out['min_step']}, max {out['max']} @ {out['max_step']}, {out.get('trend', '')})"
    if cps:
        c = max(cps, key=lambda c: abs(c["change_pct"] or 0))
        desc += f"; biggest shift at step {c['step']}: {c['before']} -> {c['after']} ({c['change_pct']}%)"
    if "nonfinite_count" in out:
        desc += f"; {out['nonfinite_count']} NaN/Inf values from step {out['first_nonfinite_step']}"
    out["summary"] = desc
    return out


def layer_table(df: pd.DataFrame, top: int = 15) -> dict[str, Any]:
    """Per-layer stats at one step; flags outliers by robust z-score of log values per metric."""
    if df.empty:
        return {"layers": [], "outliers": []}
    wide = df.pivot_table(index="layer", columns="name", values="value", aggfunc="last")
    outliers = []
    for col in wide.columns:
        v = wide[col].astype(float)
        pos = v[(v > 0) & np.isfinite(v)]
        if len(pos) >= 4:
            lv = np.log10(pos)
            med = lv.median()
            mad = (lv - med).abs().median() + 1e-9
            z = (lv - med) / (1.4826 * mad)
            for layer in z[abs(z) > 3.5].index:
                outliers.append(
                    {
                        "layer": layer,
                        "metric": col,
                        "value": _r(v[layer]),
                        "median": _r(10**med),
                        "x_median": _r(v[layer] / 10**med, 3),
                    }
                )
        if col == "weight/update_ratio":
            for layer in v[v == 0].index:
                outliers.append({"layer": layer, "metric": col, "value": 0.0, "note": "not updating"})
    vals = df["value"].astype(float)
    for _, r in df[~np.isfinite(vals) & vals.notna()].iterrows():
        outliers.append(
            {"layer": r["layer"], "metric": r["name"], "value": str(r["value"]), "note": "non-finite"}
        )
    nan_rows = df[vals.isna()]
    for _, r in nan_rows.iterrows():
        outliers.append({"layer": r["layer"], "metric": r["name"], "value": "nan", "note": "non-finite"})
    outliers.sort(
        key=lambda o: (
            -abs(math.log10(o["x_median"]))
            if isinstance(o.get("x_median"), float) and o["x_median"] > 0
            else -99
        )
    )
    rows = [
        {"layer": layer, **{k: _r(v) for k, v in r.items() if not (isinstance(v, float) and math.isnan(v))}}
        for layer, r in wide.iterrows()
    ]
    out_layers = {o["layer"] for o in outliers}
    rows.sort(key=lambda r: (r["layer"] not in out_layers,))
    return {
        "n_layers": len(rows),
        "layers": rows[:top],
        "truncated": len(rows) > top,
        "outliers": outliers[:20],
    }


def cap(obj: Any, max_chars: int = 12000) -> Any:
    """Hard cap on response size: trims lists until the JSON fits."""
    import json

    s = json.dumps(obj, default=str)
    if len(s) <= max_chars:
        return obj
    if isinstance(obj, dict):
        obj = dict(obj)
        for k in sorted(obj, key=lambda k: -len(json.dumps(obj[k], default=str))):
            if isinstance(obj[k], list) and len(obj[k]) > 4:
                obj[k] = obj[k][: max(4, len(obj[k]) // 2)]
                obj["_truncated"] = True
                return cap(obj, max_chars)
    return obj
