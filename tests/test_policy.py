import pytest

from trainscope_server.commands import policy


def test_checkpoint_auto_approved_pause_needs_human(store):
    store.upsert_run({"id": "r1", "project": "p"})
    c = policy.request(store, "r1", "checkpoint", {}, "assistant")
    assert c["status"] == "approved"
    p = policy.request(store, "r1", "pause", {}, "assistant")
    assert p["status"] == "pending_approval"
    with pytest.raises(policy.CommandError):
        policy.approve(store, p["id"], "assistant:mcp")  # assistants can't approve
    assert policy.approve(store, p["id"], "human:alice")["status"] == "approved"


def test_invalid_commands_rejected(store):
    store.upsert_run({"id": "r1", "project": "p"})
    with pytest.raises(policy.CommandError):
        policy.request(store, "r1", "rm_rf", {}, "x")
    with pytest.raises(policy.CommandError):
        policy.request(store, "r1", "set_lr", {"lr": 1e9}, "x")
    store.set_run_status("r1", "finished")
    with pytest.raises(policy.CommandError):
        policy.request(store, "r1", "checkpoint", {}, "x")
