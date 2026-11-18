"""Overfitting (eval worsening while train improves) and a suspicious 'eval better than train' gap."""

from __future__ import annotations

from .base import Detector, Finding, RunData, finite


class OverfittingDetector(Detector):
    name = "overfitting"

    def run(self, data: RunData) -> list[Finding]:
        ev = finite(data.scalar("eval/loss"))
        tr = finite(data.scalar("loss"))
        out: list[Finding] = []
        if len(ev) < 4 or len(tr) < 10:
            return out
        best_step = int(ev.idxmin())
        best = float(ev.min())
        last = float(ev.iloc[-1])
        tr_at_best = (
            float(tr[tr.index <= best_step].iloc[-max(1, min(10, len(tr) // 20)) :].mean())
            if (tr.index <= best_step).any()
            else float(tr.iloc[0])
        )
        tr_last = float(tr.iloc[-max(5, len(tr) // 20) :].mean())
        after = ev[ev.index > best_step]
        sustained = len(after) >= 2 and float(after.iloc[-2:].min()) > best * 1.1
        if sustained and last > best * 1.15 and tr_last < tr_at_best * 0.9:
            gap = last - tr_last
            ev_acc = finite(data.scalar("eval/accuracy"))
            out.append(
                Finding(
                    self.name,
                    "warning" if last < best * 1.5 else "critical",
                    best_step,
                    int(ev.index[-1]),
                    f"eval loss rose {((last / best) - 1) * 100:.0f}% after step {best_step} while train loss kept "
                    f"falling ({tr_at_best:.3g} -> {tr_last:.3g}); train/eval gap {gap:.3g}",
                    {
                        "best_eval_loss": best,
                        "best_eval_step": best_step,
                        "final_eval_loss": last,
                        "final_train_loss": tr_last,
                        "gap": gap,
                        "final_eval_accuracy": float(ev_acc.iloc[-1]) if len(ev_acc) else None,
                        "train_set_size": data.cfg("train_size", "n_train"),
                    },
                    ["overfitting"],
                    0.7,
                )
            )
        # eval consistently *better* than train by a wide margin -> possible leakage
        tr_acc = finite(data.scalar("train/accuracy"))
        ev_acc = finite(data.scalar("eval/accuracy"))
        if len(ev_acc) >= 3 and len(tr_acc) >= 3:
            ea, ta = float(ev_acc.iloc[-3:].mean()), float(tr_acc.iloc[-3:].mean())
            if ea > ta + 0.1 and ea > 0.9:
                out.append(
                    Finding(
                        "eval_better_than_train",
                        "warning",
                        int(ev_acc.index[0]),
                        int(ev_acc.index[-1]),
                        f"eval accuracy ({ea:.1%}) is well above train accuracy ({ta:.1%}); check for validation "
                        "leakage",
                        {"eval_accuracy": ea, "train_accuracy": ta},
                        ["val_leakage"],
                        0.6,
                    )
                )
        return out
