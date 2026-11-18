"""End to end: SDK -> HTTP ingestion -> store -> detectors -> MCP tools."""

import asyncio
import json
import socket
import threading
import time

import pytest
import torch
import uvicorn
from torch import nn

import trainscope as ts
from trainscope_server.app import create_app
from trainscope_server.db import open_store
from trainscope_server.ingest.auth import Auth


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    def __init__(self, store, port, tokens=None):
        cfg = uvicorn.Config(
            create_app(store, Auth(tokens or {}), detector_interval=0),
            host="127.0.0.1",
            port=port,
            log_level="error",
        )
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        return self

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(5)


def _train(steps, url, tmp_path, token=None, lr=0.1, start=0, run=None):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    if run is None:
        run = ts.init(
            "it",
            config={"lr": lr, "num_classes": 3, "batch_size": 8},
            url=url,
            token=token,
            spool_dir=str(tmp_path / "spool"),
            system_interval=None,
            poll_commands=False,
            flush_interval=0.05,
        )
        ts.watch(m, optimizer=opt, grad_every=5, act_every=10)
    for step in range(start, start + steps):
        x, y = torch.randn(8, 8), torch.randint(0, 3, (8,))
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        ts.log(step, loss=loss.item(), lr=lr)
    return run


@pytest.mark.slow
def test_http_ingest_detectors_and_mcp(tmp_path):
    store = open_store(str(tmp_path / "srv.duckdb"))
    port = _free_port()
    with Server(store, port, tokens={"secret": "it"}):
        run = _train(60, f"http://127.0.0.1:{port}", tmp_path, token="secret")
        ts.finish()
    assert store.get_run(run.id)["status"] == "finished"
    assert store.max_step(run.id) == 59

    from mcp.client.client import Client

    from trainscope_mcp.server import build_server

    async def go():
        async with Client(build_server(store)) as c:
            r = await c.call_tool("get_run_summary", {"run_id": run.id})
            summary = json.loads(r.content[0].text)
            r = await c.call_tool("get_metric", {"run_id": run.id, "name": "loss", "max_points": 10})
            metric = json.loads(r.content[0].text)
            r = await c.call_tool("get_layer_stats", {"run_id": run.id})
            layers = json.loads(r.content[0].text)
            return summary, metric, layers

    summary, metric, layers = asyncio.run(go())
    assert summary["progress"]["last_step"] == 59
    assert len(metric["points"]) <= 10
    assert layers["n_layers"] >= 2


@pytest.mark.slow
def test_auth_rejects_bad_token(tmp_path):
    store = open_store(str(tmp_path / "srv.duckdb"))
    port = _free_port()
    import httpx

    with Server(store, port, tokens={"secret": "it"}):
        r = httpx.post(
            f"http://127.0.0.1:{port}/v1/runs",
            json={"id": "x", "project": "it"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 401
        r = httpx.post(
            f"http://127.0.0.1:{port}/v1/runs",
            json={"id": "x", "project": "other"},
            headers={"Authorization": "Bearer secret"},
        )
        assert r.status_code == 403


@pytest.mark.slow
def test_server_outage_training_continues_and_backfills(tmp_path):
    """Reliability: kill the ingestion server mid-run; training must not block, data is backfilled."""
    store = open_store(str(tmp_path / "srv.duckdb"))
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    with Server(store, port):
        run = _train(30, url, tmp_path)
        run.flush()
    # server is down now
    t0 = time.time()
    _train(40, url, tmp_path, start=30, run=run)
    run.flush()
    elapsed = time.time() - t0
    assert elapsed < 10, f"training blocked on telemetry for {elapsed:.1f}s"
    assert run.flusher.spooled, "telemetry should be spooled to disk while the server is down"
    assert store.max_step(run.id) == 29
    with Server(store, port):
        run.flusher.retry_interval = 0
        ts.finish()
    assert store.max_step(run.id) == 69
    steps = store.metrics_df(run.id, names=["loss"])["step"].tolist()
    assert steps == list(range(70)), "every step must arrive exactly once"
