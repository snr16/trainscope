"""Tool implementations. Plain Python over a Storage so they're usable from the MCP server, the evaluation
harness, and tests alike. Every tool returns JSON-able dicts with a short human-readable ``summary``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from trainscope_server.commands import policy
from trainscope_server.db import Storage
from trainscope_server.detectors import RunData, run_detectors
from trainscope_server.detectors.memory import memory_floor

from .summarize import _r, cap, layer_table, series_summary

SIGNAL_GROUPS: dict[str, tuple[str, ...]] = {
    "gradients": ("grad/", "weight/"),
    "activations": ("act/",),
    "system": ("sys/", "data_wait", "step_time", "samples_per_sec", "mem/"),
    "data": ("data/",),
}

SIGNAL_EVENTS: dict[str, tuple[str, ...]] = {
    "gradients": ("nonfinite_grad",),
    "data": ("data_samples", "dataset_ids"),
}

KEY_SCALARS = ["loss", "eval/loss", "eval/accuracy", "train/accuracy", "lr", "grad/global_norm"]


class TrainScopeTools:
    def __init__(self, store: Storage, hidden_signals: set[str] | None = None, max_chars: int = 12000):
        self.store = store
        self.hidden = tuple(p for g in (hidden_signals or set()) for p in SIGNAL_GROUPS.get(g, ()))
        # event types that belong to a signal family (hidden together with its metrics in ablations)
        self.hidden_events = {e for g in (hidden_signals or set()) for e in SIGNAL_EVENTS.get(g, ())}
        self.max_chars = max_chars

    # ------------------------------------------------------------------ helpers
    def _visible(self, name: str) -> bool:
        return not self.hidden or not name.startswith(self.hidden)

    def _df(self, run_id: str, **kw: Any) -> pd.DataFrame:
        df = self.store.metrics_df(run_id, **kw)
        if self.hidden and len(df):
            df = df[~df["name"].str.startswith(self.hidden)]
        return df

    def _events(self, run_id: str, type: str | None = None) -> list[dict[str, Any]]:
        if type in self.hidden_events:
            return []
        return [e for e in self.store.events(run_id, type) if e["type"] not in self.hidden_events]

    def _run(self, run_id: str) -> dict[str, Any]:
        r = self.store.get_run(run_id)
        if r is None:
            raise ValueError(f"unknown run_id {run_id!r}; call list_runs first")
        return r

    def _series(self, run_id: str, name: str, layer: str = "", **kw: Any) -> pd.Series:
        df = self._df(run_id, names=[name], layer=layer, **kw)
        return df.groupby("step")["value"].last().sort_index() if len(df) else pd.Series(dtype=float)

    def _findings(self, run_id: str) -> list[dict[str, Any]]:
        run = self._run(run_id)
        data = RunData(self._df(run_id), run.get("config"), self._events(run_id))
        return [f.to_dict(run_id) for f in run_detectors(data)]

    def _out(self, d: dict[str, Any]) -> dict[str, Any]:
        return cap(d, self.max_chars)

    @staticmethod
    def _public_config(cfg: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in cfg.items() if not k.startswith("_bench")}

    # -------------------------------------------------------------------- tools
    def list_runs(
        self, project: str | None = None, status: str | None = None, limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        runs = self.store.list_runs(project, status)
        page = runs[offset : offset + limit]
        rows = []
        for r in page:
            loss = self._series(r["id"], "loss")
            fl = loss[np.isfinite(loss.to_numpy(dtype=float))] if len(loss) else loss
            rows.append(
                {
                    "run_id": r["id"],
                    "name": r["name"],
                    "project": r["project"],
                    "status": r["status"],
                    "last_step": int(loss.index[-1]) if len(loss) else None,
                    "last_loss": _r(loss.iloc[-1]) if len(loss) else None,
                    "best_loss": _r(fl.min()) if len(fl) else None,
                }
            )
        return self._out(
            {
                "total": len(runs),
                "offset": offset,
                "runs": rows,
                "summary": f"{len(runs)} run(s); showing {len(rows)} from offset {offset}",
            }
        )

    def get_run_summary(self, run_id: str, include_findings: bool = True) -> dict[str, Any]:
        run = self._run(run_id)
        names = [n for n in self.store.metric_names(run_id) if self._visible(n)]
        current: dict[str, Any] = {}
        for n in KEY_SCALARS:
            if n in names:
                s = self._series(run_id, n)
                if len(s):
                    current[n] = {
                        "last": _r(s.iloc[-1]),
                        "first": _r(s.iloc[0]),
                        "min": _r(s[np.isfinite(s.to_numpy(dtype=float))].min())
                        if np.isfinite(s.to_numpy(dtype=float)).any()
                        else None,
                    }
        cfg = self._public_config(run.get("config") or {})
        events = self._events(run_id)
        ev_counts: dict[str, int] = {}
        for e in events:
            ev_counts[e["type"]] = ev_counts.get(e["type"], 0) + 1
        out: dict[str, Any] = {
            "run_id": run_id,
            "name": run["name"],
            "project": run["project"],
            "status": run["status"],
            "progress": {"last_step": self.store.max_step(run_id), "total_steps": cfg.get("total_steps")},
            "config": cfg,
            "environment": run.get("env"),
            "current_metrics": current,
            "available_metrics": names,
            "events": ev_counts,
        }
        fsum = ""
        if include_findings:
            fs = self._findings(run_id)
            out["open_findings"] = [
                {"detector": f["detector"], "severity": f["severity"], "summary": f["summary"]}
                for f in fs[:5]
            ]
            fsum = f"; {len(fs)} finding(s)" + (f", top: {fs[0]['summary']}" if fs else "")
        loss = current.get("loss", {})
        out["summary"] = (
            f"run {run['name']} [{run['status']}] at step {out['progress']['last_step']}; "
            f"loss {loss.get('first')} -> {loss.get('last')}{fsum}"
        )
        return self._out(out)

    def get_metric(
        self,
        run_id: str,
        name: str,
        layer: str = "",
        step_start: int | None = None,
        step_end: int | None = None,
        max_points: int = 60,
    ) -> dict[str, Any]:
        self._run(run_id)
        if not self._visible(name):
            return {"summary": f"metric {name} is not available in this configuration", "n": 0}
        s = self._series(run_id, name, layer, step_min=step_start, step_max=step_end)
        if s.empty:
            avail = [n for n in self.store.metric_names(run_id) if self._visible(n)]
            hint = ""
            if name in avail and not layer:
                layers = self._df(run_id, names=[name])["layer"].unique().tolist()
                hint = f"; {name} is per-layer, pass layer= one of {layers[:10]}"
            return {"summary": f"no data for {name!r}{hint}", "n": 0, "available_metrics": avail}
        out = series_summary(s, max(5, min(max_points, 200)), name + (f"[{layer}]" if layer else ""))
        return self._out(out)

    def get_layer_stats(self, run_id: str, step: int | None = None, top: int = 15) -> dict[str, Any]:
        self._run(run_id)
        df = self._df(run_id)
        df = df[df["layer"] != ""]
        df = df[df["name"].str.startswith(("grad/", "weight/", "act/"))]
        if df.empty:
            return {"summary": "no per-layer statistics recorded (was ts.watch(model) called?)"}
        steps = np.sort(df["step"].unique())
        chosen = (
            int(steps[-1])
            if step is None
            else int(steps[steps <= step][-1])
            if (steps <= step).any()
            else int(steps[0])
        )
        at = df[df["step"] == chosen]
        # activations are sampled less often: use the nearest activation sample at/before the step
        act = df[df["name"].str.startswith("act/") & (df["step"] <= chosen)]
        if len(act) and not at["name"].str.startswith("act/").any():
            at = pd.concat([at, act[act["step"] == act["step"].max()]])
        table = layer_table(at[["layer", "name", "value"]], top)
        # how did the top-gradient layer evolve?
        g = df[df["name"] == "grad/norm"]
        if len(g):
            wide = g.pivot_table(index="step", columns="layer", values="value", aggfunc="last")
            first, last = (
                wide.iloc[0],
                wide.iloc[-1] if chosen >= wide.index[-1] else wide.loc[:chosen].iloc[-1],
            )
            growth = (
                (last / (first + 1e-12))
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .sort_values(ascending=False)
            )
            table["grad_growth_since_start"] = [
                {"layer": k, "x": _r(v, 3)} for k, v in growth.head(5).items()
            ]
        table["step"] = chosen
        table["sampled_steps"] = [int(steps[0]), int(steps[-1]), int(len(steps))]
        o = table["outliers"]
        table["summary"] = f"{table['n_layers']} layers at step {chosen}; {len(o)} outlier(s)" + (
            f", e.g. {o[0]['layer']} {o[0]['metric']}={o[0]['value']}"
            + (f" ({o[0].get('x_median')}x median)" if o[0].get("x_median") else f" ({o[0].get('note', '')})")
            if o
            else ""
        )
        return self._out(table)

    def get_findings(self, run_id: str) -> dict[str, Any]:
        fs = self._findings(run_id)
        try:
            self.store.replace_findings(run_id, fs)
        except Exception:
            pass
        return self._out(
            {
                "findings": fs,
                "summary": "; ".join(f"[{f['severity']}] {f['summary']}" for f in fs[:5])
                or "no detector findings",
            }
        )

    def get_system_metrics(self, run_id: str, last_n_steps: int | None = None) -> dict[str, Any]:
        self._run(run_id)
        if "system" in {g for g, p in SIGNAL_GROUPS.items() if p[0] in self.hidden}:
            return {"summary": "system metrics are not available in this configuration"}
        mx = self.store.max_step(run_id)
        lo = mx - last_n_steps if last_n_steps else None
        out: dict[str, Any] = {}
        parts = []
        for name in [
            "data_wait_frac",
            "step_time",
            "samples_per_sec",
            "sys/gpu_util",
            "sys/gpu_mem_mb",
            "sys/cuda_alloc_mb",
            "sys/mps_alloc_mb",
            "sys/proc_rss_mb",
            "sys/cpu",
            "mem/tracked_mb",
        ]:
            s = self._series(run_id, name, step_min=lo)
            s = s[np.isfinite(s.to_numpy(dtype=float))] if len(s) else s
            if len(s) < 2:
                continue
            ss = series_summary(s, 12, name)
            out[name] = {k: ss.get(k) for k in ("first", "last", "min", "max", "mean", "trend", "change_pct")}
            out[name]["median"] = _r(s.median())
        if "data_wait_frac" in out:
            parts.append(f"data wait {out['data_wait_frac']['median']:.0%} of step time (median)")
        if "step_time" in out:
            parts.append(f"step time median {out['step_time']['median']}s")
        for m in ("sys/cuda_alloc_mb", "sys/gpu_mem_mb", "sys/proc_rss_mb", "mem/tracked_mb"):
            if m in out:
                # raw RSS is spiky; its floor (rolling minimum) only rises when memory is really retained
                s = self._series(run_id, m, step_min=lo)
                fl = memory_floor(s[np.isfinite(s.to_numpy(dtype=float))])
                if fl is not None:
                    _, f0, f1, r = fl
                    rising = f1 - f0 > 0.2 * max(f0, 1.0) and r > 0.5
                    out[m].update(
                        {
                            "floor_start": _r(f0),
                            "floor_end": _r(f1),
                            "floor_trend_r": _r(r, 2),
                            "floor_trend": "rising" if rising else "flat",
                        }
                    )
                    parts.append(
                        f"{m} floor {f0:.0f} -> {f1:.0f} MB ({'RISING' if rising else 'flat'}), "
                        f"raw {out[m]['first']} -> {out[m]['last']} MB"
                    )
                else:
                    parts.append(f"{m} {out[m]['first']} -> {out[m]['last']} MB ({out[m]['trend']})")
                break
        if "sys/gpu_util" in out:
            parts.append(f"GPU util median {out['sys/gpu_util']['median']}%")
        out["summary"] = "; ".join(parts) or "no system metrics recorded"
        return self._out(out)

    def inspect_data(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        if any(p.startswith("data/") for p in self.hidden):
            return {"summary": "data statistics are not available in this configuration"}
        out: dict[str, Any] = {}
        parts = []
        for name in ["data/label_entropy", "data/majority_frac", "data/pad_ratio"]:
            s = self._series(run_id, name)
            if len(s):
                out[name] = {
                    "first": _r(s.iloc[0]),
                    "last": _r(s.iloc[-1]),
                    "median": _r(s.median()),
                    "min": _r(s.min()),
                    "max": _r(s.max()),
                }
                parts.append(f"{name.split('/')[1]} median {_r(s.median(), 3)}")
        lf = self._df(run_id, names=["data/label_frac"])
        if len(lf):
            wide = lf.pivot_table(index="step", columns="layer", values="value", aggfunc="last")
            h = max(1, len(wide) // 2)
            out["label_distribution_first_half"] = {k: _r(v, 3) for k, v in wide.iloc[:h].mean().items()}
            out["label_distribution_second_half"] = {k: _r(v, 3) for k, v in wide.iloc[h:].mean().items()}
        samples = self._events(run_id, "data_samples")
        if samples:
            out["sample_ids"] = [{"step": e["step"], "ids": e["payload"].get("ids", [])} for e in samples[:3]]
            seen: dict[str, int] = {}
            for e in samples:
                for i in e["payload"].get("ids", []):
                    seen[i] = seen.get(i, 0) + 1
            out["sample_ids_seen"] = len(seen)
        cfg = self._public_config(run.get("config") or {})
        out["dataset_config"] = {
            k: v
            for k, v in cfg.items()
            if any(
                t in k
                for t in (
                    "data",
                    "train_size",
                    "val_size",
                    "num_classes",
                    "seq_len",
                    "normal",
                    "augment",
                    "batch",
                    "workers",
                )
            )
        }
        out["summary"] = "; ".join(parts) or "no data statistics recorded (use ts.log_batch)"
        return self._out(out)

    def compare_runs(self, run_a: str, run_b: str, metrics: list[str] | None = None) -> dict[str, Any]:
        self._run(run_a)
        self._run(run_b)
        metrics = metrics or ["loss", "eval/loss", "eval/accuracy", "grad/global_norm", "lr"]
        out: dict[str, Any] = {"run_a": run_a, "run_b": run_b, "metrics": {}}
        parts = []
        for m in metrics:
            if not self._visible(m):
                continue
            a, b = self._series(run_a, m), self._series(run_b, m)
            if a.empty or b.empty:
                continue
            j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            entry: dict[str, Any] = {
                "a_last": _r(a.iloc[-1]),
                "b_last": _r(b.iloc[-1]),
                "a_min": _r(a[np.isfinite(a)].min()) if np.isfinite(a).any() else None,
                "b_min": _r(b[np.isfinite(b)].min()) if np.isfinite(b).any() else None,
            }
            if len(j) >= 5:
                rel = (j["a"] - j["b"]).abs() / (j[["a", "b"]].abs().max(axis=1) + 1e-12)
                div = rel[rel > 0.2]
                entry["diverge_step"] = int(div.index[0]) if len(div) else None
                entry["max_rel_diff"] = _r(rel.max(), 3)
                if entry["diverge_step"] is not None:
                    parts.append(f"{m} diverges at step {entry['diverge_step']}")
            out["metrics"][m] = entry
        out["summary"] = "; ".join(parts) or "curves track each other closely on the compared metrics"
        return self._out(out)

    def diff_configs(self, run_a: str, run_b: str) -> dict[str, Any]:
        ra, rb = self._run(run_a), self._run(run_b)

        def flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
            out = {}
            for k, v in (d or {}).items():
                if isinstance(v, dict):
                    out.update(flat(v, f"{prefix}{k}."))
                else:
                    out[f"{prefix}{k}"] = v
            return out

        diffs = {}
        for section in ("config", "env"):
            fa = flat(self._public_config(ra.get(section) or {}))
            fb = flat(self._public_config(rb.get(section) or {}))
            for k in sorted(set(fa) | set(fb)):
                if fa.get(k) != fb.get(k):
                    diffs[f"{section}.{k}"] = {"a": fa.get(k, "<missing>"), "b": fb.get(k, "<missing>")}
        summary = f"{len(diffs)} difference(s)" + (": " + ", ".join(list(diffs)[:6]) if diffs else "")
        return self._out({"run_a": run_a, "run_b": run_b, "differences": diffs, "summary": summary})

    def request_action(
        self, run_id: str, action: str, args: dict[str, Any] | None = None, requested_by: str = "assistant"
    ) -> dict[str, Any]:
        try:
            cmd = policy.request(self.store, run_id, action, args or {}, requested_by)
        except policy.CommandError as e:
            return {"ok": False, "summary": f"rejected: {e}"}
        msg = (
            "queued; will execute at the next step boundary"
            if cmd["status"] == "approved"
            else f"awaiting human approval: run `trainscope approve {cmd['id']}`"
        )
        return {"ok": True, "command": cmd, "summary": f"{action} {msg}"}


def jsonable(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x
