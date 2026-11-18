import numpy as np
import pandas as pd

from trainscope_mcp.summarize import cap, change_points, layer_table, lttb, series_summary


def test_lttb_keeps_endpoints_and_size():
    x = np.arange(1000.0)
    y = np.sin(x / 50)
    idx = lttb(x, y, 50)
    assert len(idx) == 50 and idx[0] == 0 and idx[-1] == 999


def test_series_summary_facts_and_nan():
    s = pd.Series(np.concatenate([np.linspace(2, 1, 100), [np.nan] * 3]), index=range(103))
    out = series_summary(s, 20, "loss")
    assert out["nonfinite_count"] == 3 and out["first_nonfinite_step"] == 100
    assert out["trend"] == "falling" and len(out["points"]) <= 20
    assert "NaN" in out["summary"]


def test_change_point_found():
    s = pd.Series(
        np.concatenate([np.full(100, 1.0), np.full(100, 5.0)])
        + 0.01 * np.random.default_rng(0).normal(size=200)
    )
    cps = change_points(s)
    assert cps and abs(cps[0]["step"] - 100) <= 3


def test_layer_table_flags_outlier():
    rows = [{"layer": f"l{i}", "name": "grad/norm", "value": 1.0} for i in range(8)]
    rows.append({"layer": "l8", "name": "grad/norm", "value": 500.0})
    out = layer_table(pd.DataFrame(rows))
    assert out["outliers"][0]["layer"] == "l8"


def test_cap_truncates_lists():
    obj = {"points": list(range(10000)), "summary": "x"}
    out = cap(obj, 2000)
    assert out.get("_truncated") and len(out["points"]) < 10000


def test_hidden_data_signals_also_hide_sample_id_events(tmp_path):
    from trainscope_mcp.tools import TrainScopeTools
    from trainscope_server.db import open_store

    st = open_store(str(tmp_path / "x.duckdb"))
    st.upsert_run({"id": "r", "project": "p"})
    st.write_metrics(
        [{"run_id": "r", "step": s, "ts": 0, "name": "loss", "layer": "", "value": 1.0} for s in range(5)]
    )
    st.write_events(
        [
            {
                "run_id": "r",
                "step": 0,
                "ts": 0,
                "type": "dataset_ids",
                "payload": {"split": "train", "ids": ["a"]},
            },
            {
                "run_id": "r",
                "step": 0,
                "ts": 0,
                "type": "dataset_ids",
                "payload": {"split": "eval", "ids": ["a"]},
            },
        ]
    )
    assert any(f["detector"] == "split_overlap" for f in TrainScopeTools(st).get_findings("r")["findings"])
    assert not TrainScopeTools(st, hidden_signals={"data"}).get_findings("r")["findings"]
