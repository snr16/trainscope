def _m(step, name="loss", v=1.0, run="r1", layer=""):
    return {"run_id": run, "step": step, "ts": 0.0, "name": name, "layer": layer, "value": v}


def test_idempotent_metric_writes(store):
    store.upsert_run({"id": "r1", "project": "p"})
    store.write_metrics([_m(0, v=1.0), _m(1, v=0.9)])
    store.write_metrics([_m(0, v=1.0), _m(1, v=0.8)])  # retried batch, updated value
    df = store.metrics_df("r1")
    assert len(df) == 2
    assert df[df.step == 1].value.iloc[0] == 0.8


def test_nan_values_round_trip(store):
    store.upsert_run({"id": "r1", "project": "p"})
    store.write_metrics([_m(0, v=float("nan")), _m(1, v=float("inf"))])
    df = store.metrics_df("r1")
    assert df.value.isna().iloc[0] and df.value.iloc[1] == float("inf")


def test_events_dedup_and_filters(store):
    store.upsert_run({"id": "r1", "project": "p"})
    e = {"run_id": "r1", "step": 3, "ts": 0, "type": "nonfinite_loss", "payload": {"x": 1}}
    store.write_events([e, e])
    assert len(store.events("r1")) == 1
    assert store.events("r1", "other") == []


def test_runs_and_commands(store):
    store.upsert_run({"id": "r1", "project": "p", "config": {"lr": 0.1}})
    store.upsert_run({"id": "r1", "project": "p", "name": "renamed"})
    r = store.get_run("r1")
    assert r["name"] == "renamed" and r["config"] == {"lr": 0.1}
    c = store.create_command({"run_id": "r1", "action": "checkpoint", "status": "approved"})
    assert store.list_commands("r1", "approved")[0]["id"] == c["id"]
    store.update_command(c["id"], status="executed", result={"path": "x"})
    assert store.get_command(c["id"])["result"] == {"path": "x"}
