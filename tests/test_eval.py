"""Offline tests for the evaluation harness: request shape, tool loop and cost accounting with a fake client."""

import asyncio
from types import SimpleNamespace as NS

from eval.grading import grade
from eval.harness import sample_per_cause
from eval.llm_clients import diagnose


class FakeMessages:
    """Turn 1 calls a tool, turn 2 submits the diagnosis."""

    def __init__(self, calls):
        self.calls = calls

    async def create(self, **params):
        self.calls.append(params)
        usage = NS(input_tokens=100, output_tokens=50, cache_read_input_tokens=900, cache_creation_input_tokens=0)
        if len(self.calls) == 1:
            block = NS(type="tool_use", id="t1", name="get_run_summary", input={"run_id": "r1"})
        else:
            block = NS(type="tool_use", id="t2", name="submit_diagnosis", input={
                "cause": "lr_too_high", "location": "config lr", "evidence": "update ratio 1e-2",
                "fix": {"type": "config_change", "field": "lr", "value": "0.001", "name": ""}, "confidence": "high"})
        return NS(content=[block], stop_reason="tool_use", usage=usage)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = FakeMessages(self.calls)
        self.beta = NS(messages=FakeMessages(self.calls))


class FakeSession:
    async def list_tools(self):
        return NS(tools=[NS(name="get_run_summary", description="summary", input_schema={"type": "object"})])

    async def call_tool(self, name, args):
        return NS(content=[NS(text='{"summary": "ok"}')], structured_content=None, is_error=False)


def test_diagnose_request_shape_and_caching():
    client = FakeClient()
    tr = asyncio.run(diagnose(client, "claude-haiku-5-5", "Diagnose run r1.", FakeSession()))
    first = client.calls[0]
    # stable prefix is cached across diagnoses, and the conversation is cached across turns
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["cache_control"] == {"type": "ephemeral"}
    # Haiku has no server-side refusal fallback: plain messages endpoint, no betas
    assert "fallbacks" not in first and "betas" not in first
    assert "temperature" not in first
    assert tr.diagnosis["cause"] == "lr_too_high" and tr.diagnosis["fix"]["value"] == 0.001
    assert len(tr.tool_calls) == 1 and tr.turns == 2
    assert tr.cache_read_tokens == 1800 and tr.cost_usd > 0


def test_fallback_only_for_supported_models():
    client = FakeClient()
    asyncio.run(diagnose(client, "claude-opus-5-5", "x", FakeSession()))
    assert client.calls[0]["fallbacks"] == "default"


def test_sample_per_cause_is_stratified_and_deterministic():
    truths = [{"run_id": f"{c}{i}", "cause": c} for c in "abc" for i in range(5)]
    s1, s2 = sample_per_cause(truths, 1), sample_per_cause(list(reversed(truths)), 1)
    assert [t["run_id"] for t in s1] == [t["run_id"] for t in s2] == ["a0", "b0", "c0"]


def test_grading():
    truth = {"cause": "lr_too_high", "category": "optimization", "component": "config.lr",
             "fixes": [{"type": "config_change", "field": "lr", "value": "default"}]}
    g = grade({"cause": "lr_too_high", "location": "the learning rate", "evidence": "",
               "fix": {"type": "config_change", "field": "learning_rate", "value": 1e-3}}, truth)
    assert g == {"parsed": True, "cause_ok": True, "category_ok": True, "loc_ok": True, "fix_static_ok": True}
    assert not grade(None, truth)["cause_ok"]
