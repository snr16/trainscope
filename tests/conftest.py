from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def store(tmp_path):
    from trainscope_server.db import open_store

    return open_store(str(tmp_path / "t.duckdb"))


@pytest.fixture
def spool(tmp_path):
    return str(tmp_path / "spool")
