"""Offline, deterministic synthetic datasets sized for CPU (no downloads needed).

- vision:   10-class 3x16x16 images built from class prototypes + noise + shifts, raw pixel scale (0..255)
- text_clf: 4-class token sequences with variable length (padded) — label depends on keyword counts
- lm:       character-level text from a small stochastic grammar
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------- vision
def make_vision(
    n: int, seed: int, num_classes: int = 10, noise: float = 0.9, proto_seed: int = 1234, n_constant: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    prng = np.random.default_rng(proto_seed)
    base = prng.normal(size=(num_classes, 3, 8, 8))
    protos = np.kron(base, np.ones((1, 1, 2, 2)))  # smooth 16x16 patterns
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, n)
    x = protos[y] * rng.uniform(0.6, 1.4, (n, 1, 1, 1)) + noise * rng.normal(size=(n, 3, 16, 16))
    shifts = rng.integers(-2, 3, (n, 2))
    for i in range(n):
        x[i] = np.roll(x[i], tuple(shifts[i]), axis=(1, 2))
    x = np.clip(120 + 45 * x, 0, 255)  # raw pixel scale
    if n_constant:
        idx = rng.choice(n, n_constant, replace=False)
        x[idx] = 128.0  # blank images (e.g. corrupt files decoded as gray)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


VISION_MEAN, VISION_STD = 120.0, 50.0


# -------------------------------------------------------------------- text clf
TEXT_VOCAB, TEXT_LEN, TEXT_PAD = 60, 64, 0
KEYWORDS = [2, 3, 4, 5]


def make_text(n: int, seed: int, min_len: int = 44) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = np.zeros((n, TEXT_LEN), dtype=np.int64)
    mask = np.zeros((n, TEXT_LEN), dtype=np.int64)
    y = rng.integers(0, len(KEYWORDS), n)
    for i in range(n):
        L = rng.integers(min_len, TEXT_LEN + 1)
        seq = rng.integers(6, TEXT_VOCAB, L)
        # the label keyword appears 3-5 times, distractor keywords 0-2 times
        for k, kw in enumerate(KEYWORDS):
            cnt = rng.integers(3, 6) if k == y[i] else rng.integers(0, 3)
            seq[rng.choice(L, cnt, replace=False)] = kw
        x[i, :L] = seq
        mask[i, :L] = 1
    return torch.tensor(x), torch.tensor(mask), torch.tensor(y)


# -------------------------------------------------------------------------- lm
_WORDS = {
    "N": ["cat", "dog", "bird", "fox", "child", "robot", "tree", "river"],
    "V": ["sees", "likes", "finds", "chases", "draws", "hears"],
    "A": ["small", "red", "quick", "lazy", "old", "bright"],
    "P": ["near", "under", "over", "behind"],
}


def make_corpus(n_sentences: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_sentences):

        def np_():
            w = ["the"]
            if rng.random() < 0.5:
                w.append(rng.choice(_WORDS["A"]))
            w.append(rng.choice(_WORDS["N"]))
            return w

        s = np_() + [rng.choice(_WORDS["V"])] + np_()
        if rng.random() < 0.4:
            s += [rng.choice(_WORDS["P"])] + np_()
        out.append(" ".join(s) + ".")
    return " ".join(out)


class CharLM(Dataset):
    def __init__(self, text: str, seq_len: int, vocab: list[str] | None = None):
        self.vocab = vocab or sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.data = torch.tensor([self.stoi.get(c, 0) for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, i: int):
        s = i * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# --------------------------------------------------------------- wrappers
class TensorDS(Dataset):
    """Dataset over tensors with an optional (deliberately) expensive per-sample transform."""

    def __init__(self, *tensors: torch.Tensor, transform_cost_s: float = 0.0, ids_offset: int = 0):
        self.tensors = tensors
        self.cost = transform_cost_s
        self.ids_offset = ids_offset

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, i: int):
        if self.cost:
            # stands in for heavy CPU augmentation (decode + resize + color jitter)
            t_end = time.perf_counter() + self.cost
            while time.perf_counter() < t_end:
                pass
        return tuple(t[i] for t in self.tensors) + (i + self.ids_offset,)
