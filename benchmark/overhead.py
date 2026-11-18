"""SDK overhead: median step time with TrainScope off vs. on at different sampling rates.

python -m benchmark.overhead --steps 300 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import trainscope as ts
from benchmark.tasks.models import SmallCNN

ROOT = Path(__file__).resolve().parent.parent


def run(steps: int, setting: dict | None, device: str, width: int, batch: int, tmp: str) -> float:
    torch.manual_seed(0)
    model = SmallCNN(10, width=width).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    xs = torch.randn(16, batch, 3, 16, 16, device=device)
    ys = torch.randint(0, 10, (16, batch), device=device)
    if setting is not None:
        ts.init(
            "overhead",
            config={"batch_size": batch},
            db=f"{tmp}/o.duckdb",
            spool_dir=f"{tmp}/sp",
            poll_commands=False,
            system_interval=1.0,
        )
        ts.watch(model, optimizer=opt, grad_every=setting["grad_every"], act_every=setting["act_every"])
    times = []
    for step in range(steps):
        t0 = time.perf_counter()
        x, y = xs[step % 16], ys[step % 16]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        lv = loss.item()  # a training loop syncs for logging anyway
        if setting is not None:
            ts.log(step, loss=lv, lr=0.05)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    if setting is not None:
        ts.finish()
    return statistics.median(times[10:])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    a = p.parse_args()
    settings = {
        "off": None,
        "dense (grad 1, act 10)": {"grad_every": 1, "act_every": 10},
        "default (grad 50, act 200)": {"grad_every": 50, "act_every": 200},
        "sparse (grad 200, act 1000)": {"grad_every": 200, "act_every": 1000},
    }
    res: dict[str, list[float]] = {k: [] for k in settings}
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(a.repeats):
            for name, s in settings.items():  # interleave to cancel drift
                res[name].append(run(a.steps, s, a.device, a.width, a.batch, tmp))
    base = statistics.median(res["off"])
    out = {"device": a.device, "width": a.width, "batch": a.batch, "steps": a.steps, "results": {}}
    print(f"device={a.device}  model=SmallCNN(width={a.width}) batch={a.batch}")
    print(f"{'setting':30} {'median step ms':>15} {'overhead':>9}")
    for name, v in res.items():
        m = statistics.median(v)
        out["results"][name] = {"median_step_ms": m * 1000, "overhead_pct": (m / base - 1) * 100}
        print(f"{name:30} {m * 1000:15.2f} {(m / base - 1) * 100:8.1f}%")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "overhead.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
