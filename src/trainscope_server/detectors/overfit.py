"""Overfitting (eval worsening while train improves) and a suspicious 'eval better than train' gap."""

from __future__ import annotations

from .base import Detector, Finding, RunData, finite


class OverfittingDetector(Detector):
    name = "overfitting"
    # overfitting and leakage thresholds
    defaults = {
        "rise_x": 1.15,  # final eval loss this many x its best = overfitting
        "sustained_x": 1.1,  # the last two evals must both be above this x the best
        "train_drop": 0.9,  # ... while train loss fell below this fraction of its value at the best eval
        "critical_x": 1.5,  # eval loss this many x its best = critical
        "leak_loss_ratio": 0.9,  # eval loss below this fraction of train loss (with higher eval accuracy) = leakage
    }

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
        sustained = len(after) >= 2 and float(after.iloc[-2:].min()) > best * self.sustained_x
        if sustained and last > best * self.rise_x and tr_last < tr_at_best * self.train_drop:
            gap = last - tr_last
            ev_acc = finite(data.scalar("eval/accuracy"))
            out.append(
                Finding(
                    self.name,
                    "warning" if last < best * self.critical_x else "critical",
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
        # eval consistently *better* than train -> possible leakage of eval data into training
        tr_acc = finite(data.scalar("train/accuracy"))
        ev_acc = finite(data.scalar("eval/accuracy"))
        k = max(5, len(tr) // 20)
        tr_tail = float(tr.iloc[-k:].mean())
        ev_tail = float(ev.iloc[-2:].mean())
        acc_gap = (
            (float(ev_acc.iloc[-3:].mean()) - float(tr_acc.iloc[-3:].mean()))
            if len(ev_acc) >= 3 and len(tr_acc) >= 3
            else None
        )
        if (
            ev_tail < self.leak_loss_ratio * tr_tail and acc_gap is not None and acc_gap > 0
        ):  # LM-style runs: dropout alone does this
            out.append(
                Finding(
                    "eval_better_than_train",
                    "warning",
                    int(ev.index[0]),
                    int(ev.index[-1]),
                    f"eval loss ({ev_tail:.3g}) is clearly below train loss ({tr_tail:.3g})"
                    + (f" and eval accuracy is {acc_gap:+.1%} vs train" if acc_gap is not None else "")
                    + "; the eval set may overlap the training data",
                    {"eval_loss": ev_tail, "train_loss": tr_tail, "eval_minus_train_accuracy": acc_gap},
                    ["val_leakage"],
                    0.6,
                )
            )
        return out
