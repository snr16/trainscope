import torch
from torch import nn

import trainscope as ts
from trainscope.hooks import ModelWatcher
from trainscope_server.db import open_store


def _model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))


def test_watcher_collects_grad_weight_act_stats():
    m = _model()
    w = ModelWatcher(m, grad_every=1, act_every=1)
    w.before_step(-1)  # sample next forward (step 0)
    loss = nn.functional.cross_entropy(m(torch.randn(4, 8)), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    metrics, events = w.collect(0)
    names = {n for n, _, _ in metrics}
    assert {"grad/norm", "weight/norm", "grad/global_norm", "act/mean", "act/dead_frac"} <= names
    assert not events


def test_nonfinite_grad_event_names_layer():
    m = _model()
    w = ModelWatcher(m, grad_every=1)
    x = torch.randn(2, 8)
    x[0, 0] = float("nan")  # a bad input poisons the gradients during backward
    m(x).sum().backward()
    _, events = w.collect(0)
    assert events and events[0]["payload"]["first_layer"] == "0"


def test_end_to_end_local_mode(tmp_path):
    db = str(tmp_path / "x.duckdb")
    m = _model()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    run = ts.init(
        "t",
        config={"lr": 0.1, "batch_size": 4},
        db=db,
        spool_dir=str(tmp_path / "sp"),
        system_interval=None,
        poll_commands=False,
    )
    ts.watch(m, optimizer=opt, grad_every=5, act_every=5)
    data = [(torch.randn(4, 8), torch.randint(0, 3, (4,))) for _ in range(20)]
    for step, (x, y) in enumerate(ts.wrap_loader(data)):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        ts.log(step, loss=loss.item(), lr=0.1)
        ts.log_batch(step, labels=y, num_classes=3)
    ts.log(20, loss=float("nan"))
    ts.finish()
    st = open_store(db)
    assert st.get_run(run.id)["status"] == "finished"
    names = st.metric_names(run.id)
    assert {"loss", "step_time", "data_wait_frac", "grad/norm", "model/train_mode"} <= set(names)
    assert st.events(run.id, "nonfinite_loss")


def test_log_never_raises_without_init():
    ts.log(0, loss=1.0)  # no active run -> no-op


def test_checkpoint_command_executes_at_step_boundary(tmp_path):
    from trainscope_server.commands import policy

    db = str(tmp_path / "c.duckdb")
    m = _model()
    run = ts.init(
        "t",
        db=db,
        spool_dir=str(tmp_path / "sp"),
        system_interval=None,
        poll_commands=True,
        command_interval=0.05,
        checkpoint_dir=str(tmp_path / "ck"),
    )
    ts.watch(m)
    store = open_store(db)
    cmd = policy.request(store, run.id, "checkpoint", {}, "test")
    run.poller.poll_once()
    ts.log(7, loss=1.0)
    ts.finish()
    done = store.get_command(cmd["id"])
    assert done["status"] == "executed" and done["result"]["step"] == 7
    assert (tmp_path / "ck" / run.id / "step_7.pt").exists()


def test_grad_norm_is_measured_before_clipping():
    m = _model()
    w = ModelWatcher(m, grad_every=1)
    (m(torch.randn(4, 8)) * 1000).sum().backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 0.01)
    metrics, _ = w.collect(0)
    g = dict(((n, layer), v) for n, layer, v in metrics)[("grad/global_norm", "")]
    assert g > 1.0, "should report the pre-clip norm, not the clipped 0.01"
