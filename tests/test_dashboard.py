"""Dashboard API and static files."""

import time

import torch
from fastapi.testclient import TestClient
from torch import nn

import trainscope as ts
from trainscope_server.app import create_app
from trainscope_server.dashboard import MAX_POINTS
from trainscope_server.db import open_store


def _train_run(db: str, tmp_path, steps: int = 60, nan_at: int | None = None) -> str:
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    run = ts.init(
        "dash",
        config={"lr": 0.1, "num_classes": 3, "total_steps": steps},
        db=db,
        spool_dir=str(tmp_path / "sp"),
        system_interval=None,
        poll_commands=False,
    )
    ts.watch(m, optimizer=opt, grad_every=5, act_every=10)
    for step in range(steps):
        x, y = torch.randn(8, 8), torch.randint(0, 3, (8,))
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        ts.log(step, loss=float("nan") if step == nan_at else loss.item(), lr=0.1)
        ts.log_batch(step, labels=y, num_classes=3)
    ts.finish()
    return run.id


def test_dashboard_endpoints(tmp_path):
    db = str(tmp_path / "d.duckdb")
    rid = _train_run(db, tmp_path, nan_at=40)
    c = TestClient(create_app(open_store(db)))

    # static UI is served and the root redirects to it
    page = c.get("/")
    assert page.status_code == 200 and "<title>TrainScope</title>" in page.text
    assert c.get("/ui/app.js").status_code == 200 and c.get("/ui/app.css").status_code == 200

    # overview answers immediately; findings arrive from the background worker
    o = c.get("/v1/overview").json()
    run = next(r for r in o["runs"] if r["id"] == rid)
    assert run["last_step"] == 59 and run["total_steps"] == 60
    for _ in range(50):
        run = next(r for r in c.get("/v1/overview").json()["runs"] if r["id"] == rid)
        if run["findings"] is not None:
            break
        time.sleep(0.1)
    assert run["findings"]["critical"] >= 1  # the NaN loss
    # the home page's sparkline (recently ended runs) skips the NaN step
    assert 2 <= len(run["spark"]) <= 40 and all(p[0] != 40 for p in run["spark"])

    # requests waiting for a person show up across runs, with the run's name
    store = c.app.state.store
    cmd = store.create_command(
        {"run_id": rid, "action": "pause", "requested_by": "assistant:mcp", "status": "pending_approval"}
    )
    approvals = c.get("/v1/overview").json()["pending_approvals"]
    assert [(a["id"], a["run_name"]) for a in approvals] == [(cmd["id"], run["name"])]

    s = c.get(f"/v1/runs/{rid}/series", params={"names": "loss,lr,missing"}).json()["series"]
    loss = s[0]
    assert loss["nonfinite"] == [40], "NaN steps are reported, not silently dropped"
    assert all(p[0] != 40 for p in loss["points"])
    assert s[2]["points"] == []

    layers = c.get(f"/v1/runs/{rid}/layers").json()
    assert layers["n_layers"] >= 2 and "grad/norm" in layers["layers"][0]
    assert c.get(f"/v1/runs/{rid}/summary").json()["status"] == "finished"
    assert "label_distribution_first_half" in c.get(f"/v1/runs/{rid}/data").json()
    assert "series" in c.get(f"/v1/runs/{rid}/system").json()
    assert c.get("/v1/runs/nope/summary").status_code == 404


def test_series_are_downsampled(tmp_path):
    st = open_store(str(tmp_path / "big.duckdb"))
    st.upsert_run({"id": "r", "project": "p"})
    st.write_metrics(
        [
            {"run_id": "r", "step": i, "ts": 0, "name": "loss", "layer": "", "value": 1 / (i + 1)}
            for i in range(5000)
        ]
    )
    c = TestClient(create_app(st))
    pts = c.get("/v1/runs/r/series", params={"names": "loss"}).json()["series"][0]["points"]
    assert len(pts) <= MAX_POINTS and pts[0][0] == 0 and pts[-1][0] == 4999


def test_taxonomy_is_served_from_one_place(tmp_path):
    from trainscope_server import taxonomy

    assert set(taxonomy.LABELS) == set(taxonomy.CAUSES), "every cause needs a display label"
    c = TestClient(create_app(open_store(str(tmp_path / "t.duckdb"))))
    causes = c.get("/v1/taxonomy").json()["causes"]
    assert causes["missing_zero_grad"] == {
        "label": "missing zero_grad",
        "category": "loop",
        "description": "optimizer.zero_grad() never called",
    }
    assert set(causes) == set(taxonomy.CAUSES)
