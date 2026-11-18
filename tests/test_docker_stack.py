"""End to end against the docker compose stack (TimescaleDB + ingest + MCP over HTTP).

Runs only when TRAINSCOPE_STACK=1 (CI's docker job starts the stack first):
    docker compose up -d --build && TRAINSCOPE_STACK=1 pytest tests/test_docker_stack.py
"""

import asyncio
import json
import os
import time
import uuid

import httpx
import pytest
import torch
from torch import nn

import trainscope as ts

pytestmark = pytest.mark.skipif(os.environ.get("TRAINSCOPE_STACK") != "1", reason="needs the docker stack")
INGEST = os.environ.get("TRAINSCOPE_URL", "http://127.0.0.1:8765")
MCP = os.environ.get("TRAINSCOPE_MCP_URL", "http://127.0.0.1:8766/mcp")


def _wait_healthy(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(f"{INGEST}/healthz", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(1)
    pytest.fail("ingest service not healthy")


def test_train_ingest_timescale_mcp(tmp_path):
    _wait_healthy()
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    run = ts.init("stack-test", config={"lr": 0.1, "num_classes": 3}, url=INGEST, name=f"stack-{uuid.uuid4().hex[:6]}",
                  spool_dir=str(tmp_path / "spool"), system_interval=None, poll_commands=False, flush_interval=0.2)
    ts.watch(m, optimizer=opt, grad_every=5, act_every=10)
    run.log_dataset_ids("train", [f"s{i}" for i in range(3000)])  # big event payload (Postgres index limits)
    run.log_dataset_ids("eval", [f"s{i}" for i in range(2900, 3100)])  # overlaps train -> leakage finding
    for step in range(60):
        x, y = torch.randn(16, 8), torch.randint(0, 3, (16,))
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        ts.log(step, loss=loss.item(), lr=0.1)
    ts.finish()
    assert not run.flusher.spooled, "every batch must be accepted by the server"

    findings = httpx.get(f"{INGEST}/v1/runs/{run.id}/findings").json()
    assert any(f["detector"] == "split_overlap" for f in findings)

    from mcp.client.client import Client
    from mcp.client.streamable_http import streamable_http_client

    async def go():
        async with Client(streamable_http_client(MCP)) as c:
            r = await c.call_tool("get_run_summary", {"run_id": run.id})
            return json.loads(r.content[0].text)

    summary = asyncio.run(go())
    assert summary["progress"]["last_step"] == 59 and summary["status"] == "finished"
