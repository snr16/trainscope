"""Suspicious data: label imbalance/shift, extreme padding."""

from __future__ import annotations

from .base import Detector, Finding, RunData, finite


class SuspiciousDataDetector(Detector):
    name = "suspicious_data"
    # data thresholds
    defaults = {
        "max_pad": 0.8,  # median share of padding tokens per batch
        "majority_frac": 0.6,  # one class above this share of labels per batch
        "majority_x_uniform": 3.0,  # ... or above this x its uniform share, whichever is larger
        "label_shift": 0.25,  # label distribution shift (total variation) between run halves
        "min_overlap": 0.01,  # share of eval samples also in training that counts as leakage
        "no_pad_ratio": 1e-06,  # padding share below this on padded inputs = the mask marks nothing as padding
        "conf_extreme_padding": 0.5,  # ranking weight: batches mostly padding
        "conf_no_padding_masked": 0.6,  # ranking weight: padded inputs with nothing masked
        "conf_split_overlap": 0.95,  # ranking weight: eval samples also in training
        "conf_label_imbalance": 0.4,  # ranking weight: one class dominates the labels
        "conf_label_shift": 0.4,  # ranking weight: label mix shifts mid-run
    }

    def run(self, data: RunData) -> list[Finding]:
        out: list[Finding] = self._split_overlap(data)
        pad = finite(data.scalar("data/pad_ratio"))
        if len(pad) >= 3 and float(pad.median()) > self.max_pad:
            out.append(
                Finding(
                    "extreme_padding",
                    "warning",
                    int(pad.index[0]),
                    int(pad.index[-1]),
                    f"batches are {float(pad.median()):.0%} padding on median",
                    {"median_pad_ratio": float(pad.median())},
                    ["bad_padding_mask"],
                    self.conf_extreme_padding,
                )
            )
        tok = finite(data.scalar("data/pad_token_ratio"))
        tok_med = float(tok.median()) if len(tok) >= 3 else None
        # padded inputs: measured from the input ids when the run logs them, else declared by the config
        padded = tok_med > self.no_pad_ratio if tok_med is not None else data.cfg("task_type") == "padded_seq"
        if len(pad) >= 3 and float(pad.median()) < self.no_pad_ratio and padded:
            out.append(
                Finding(
                    "no_padding_masked",
                    "critical" if tok_med is not None else "warning",
                    int(pad.index[0]),
                    int(pad.index[-1]),
                    "attention mask marks no tokens as padding although inputs are padded"
                    + (f" ({tok_med:.0%} of input tokens are the pad token)" if tok_med is not None else ""),
                    {"median_pad_ratio": float(pad.median()), "median_pad_token_ratio": tok_med},
                    ["bad_padding_mask"],
                    self.conf_no_padding_masked,
                )
            )
        maj = finite(data.scalar("data/majority_frac"))
        k = data.cfg("num_classes")
        if len(maj) >= 3 and k:
            m = float(maj.median())
            if m > max(self.majority_frac, self.majority_x_uniform / k):
                out.append(
                    Finding(
                        "label_imbalance",
                        "warning",
                        int(maj.index[0]),
                        int(maj.index[-1]),
                        f"one class makes up {m:.0%} of labels per batch (uniform would be {1 / k:.0%})",
                        {"median_majority_frac": m, "num_classes": k},
                        ["shuffled_labels"],
                        self.conf_label_imbalance,
                    )
                )
        lf = data.layers("data/label_frac")
        if len(lf) >= 6:
            h = len(lf) // 2
            shift = float((lf.iloc[:h].mean() - lf.iloc[h:].mean()).abs().sum() / 2)
            if shift > self.label_shift:
                out.append(
                    Finding(
                        "label_shift",
                        "warning",
                        int(lf.index[h]),
                        int(lf.index[-1]),
                        f"label distribution shifted by {shift:.0%} (total variation) mid-run",
                        {"total_variation": shift},
                        ["shuffled_labels"],
                        self.conf_label_shift,
                    )
                )
        return out

    def _split_overlap(self, data: RunData) -> list[Finding]:
        splits: dict[str, set[str]] = {}
        for e in data.events_of("dataset_ids"):
            splits.setdefault(e["payload"].get("split", ""), set()).update(e["payload"].get("ids", []))
        train = splits.get("train", set())
        evals = {k: v for k, v in splits.items() if k not in ("train", "") and v}
        out = []
        for name, ids in evals.items():
            shared = train & ids
            if not train or not shared:
                continue
            frac = len(shared) / len(ids)
            if frac < self.min_overlap:
                continue
            out.append(
                Finding(
                    "split_overlap",
                    "critical",
                    0,
                    data.max_step,
                    f"{len(shared)} of {len(ids)} logged {name} samples ({frac:.0%}) also appear in the training set",
                    {
                        "split": name,
                        "overlap": len(shared),
                        "eval_ids_logged": len(ids),
                        "train_ids_logged": len(train),
                        "examples": sorted(shared)[:5],
                    },
                    ["val_leakage"],
                    self.conf_split_overlap,
                )
            )
        return out
