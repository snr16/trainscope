"""Suspicious data: label imbalance/shift, extreme padding."""

from __future__ import annotations

from .base import Detector, Finding, RunData, finite


class SuspiciousDataDetector(Detector):
    name = "suspicious_data"

    def run(self, data: RunData) -> list[Finding]:
        out: list[Finding] = []
        pad = finite(data.scalar("data/pad_ratio"))
        if len(pad) >= 3 and float(pad.median()) > 0.8:
            out.append(
                Finding(
                    "extreme_padding",
                    "warning",
                    int(pad.index[0]),
                    int(pad.index[-1]),
                    f"batches are {float(pad.median()):.0%} padding on median",
                    {"median_pad_ratio": float(pad.median())},
                    ["bad_padding_mask"],
                    0.5,
                )
            )
        if len(pad) >= 3 and float(pad.median()) < 1e-6 and data.cfg("task_type") == "lm_padded":
            out.append(
                Finding(
                    "no_padding_masked",
                    "warning",
                    int(pad.index[0]),
                    int(pad.index[-1]),
                    "attention mask marks no tokens as padding although inputs are padded",
                    {"median_pad_ratio": float(pad.median())},
                    ["bad_padding_mask"],
                    0.6,
                )
            )
        maj = finite(data.scalar("data/majority_frac"))
        k = data.cfg("num_classes")
        if len(maj) >= 3 and k:
            m = float(maj.median())
            if m > max(0.6, 3.0 / k):
                out.append(
                    Finding(
                        "label_imbalance",
                        "warning",
                        int(maj.index[0]),
                        int(maj.index[-1]),
                        f"one class makes up {m:.0%} of labels per batch (uniform would be {1 / k:.0%})",
                        {"median_majority_frac": m, "num_classes": k},
                        ["shuffled_labels"],
                        0.4,
                    )
                )
        lf = data.layers("data/label_frac")
        if len(lf) >= 6:
            h = len(lf) // 2
            shift = float((lf.iloc[:h].mean() - lf.iloc[h:].mean()).abs().sum() / 2)
            if shift > 0.25:
                out.append(
                    Finding(
                        "label_shift",
                        "warning",
                        int(lf.index[h]),
                        int(lf.index[-1]),
                        f"label distribution shifted by {shift:.0%} (total variation) mid-run",
                        {"total_variation": shift},
                        ["shuffled_labels"],
                        0.4,
                    )
                )
        return out
