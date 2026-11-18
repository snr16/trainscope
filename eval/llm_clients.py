"""Claude client for the evaluation harness: a manual tool-use loop over MCP tools with a capped number of
tool calls, a strict `submit_diagnosis` tool for the structured answer, and per-call logging.

Note: current Claude models don't accept `temperature`, so "temperature 0" from the plan isn't available;
we fix the model, effort level, prompt and tool budget instead and report variance over repeats.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from trainscope_server.taxonomy import CAUSES

# $ per 1M tokens: (input, output, cache read). Cache writes cost 1.25x input. Haiku 5.5 rates apply to prompts
# up to 100K tokens (ours are far below); its cache-read rate is assumed at the usual 10% of input.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-fable-5-2": (8.0, 40.0, 0.20),
}
# models that accept the server-side refusal fallback (Haiku 5.5 has none)
FALLBACK_MODELS = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-2",
    "claude-fable-5-1",
}

SYSTEM = (
    "You are an experienced ML engineer debugging a training run. Find the single most likely root cause. "
    "Base your answer on evidence. When you're done, call submit_diagnosis exactly once.\n\n"
    "Allowed causes:\n" + "\n".join(f"- {k}: {v[1]} ({v[0]})" for k, v in CAUSES.items()) + "\n\n"
    "Fix format: type=config_change with field/value (e.g. field='lr', value='0.001'), or type=code_fix with "
    "name=<cause id>, or type=none for a healthy run. Leave unused fix fields as empty strings."
)

SUBMIT_TOOL = {
    "name": "submit_diagnosis",
    "description": "Submit the final diagnosis. Call exactly once, at the end.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["cause", "location", "evidence", "fix", "confidence"],
        "properties": {
            "cause": {"type": "string", "enum": list(CAUSES)},
            "location": {
                "type": "string",
                "description": "Layer, data component, or config field responsible",
            },
            "evidence": {"type": "string", "description": "Key numbers and steps supporting the diagnosis"},
            "fix": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "field", "value", "name"],
                "properties": {
                    "type": {"type": "string", "enum": ["config_change", "code_fix", "none"]},
                    "field": {"type": "string"},
                    "value": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
    },
}


@dataclass
class Transcript:
    model: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    turns: int = 0
    wall_s: float = 0.0
    diagnosis: dict[str, Any] | None = None
    stop_reason: str | None = None
    error: str | None = None

    @property
    def cost_usd(self) -> float:
        pin, pout, pread = PRICES.get(self.model, (0.0, 0.0, 0.0))
        return (
            self.input_tokens * pin
            + self.cache_write_tokens * pin * 1.25
            + self.cache_read_tokens * pread
            + self.output_tokens * pout
        ) / 1e6

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["cost_usd"] = round(self.cost_usd, 5)
        return d


def _clean_fix(fix: dict[str, Any] | None) -> dict[str, Any] | None:
    if not fix:
        return None
    out = {"type": fix.get("type", "none")}
    if out["type"] == "config_change":
        out["field"] = fix.get("field", "")
        v = fix.get("value", "")
        try:
            out["value"] = float(v) if v not in ("", None) and v.lower() not in ("null", "none") else None
        except (ValueError, AttributeError):
            out["value"] = v
    elif out["type"] == "code_fix":
        out["name"] = fix.get("name", "")
    return out


async def diagnose(
    client: Any,
    model: str,
    user_prompt: str,
    mcp_session: Any | None = None,
    max_tool_calls: int = 12,
    effort: str = "medium",
    max_tokens: int = 16000,
) -> Transcript:
    """Run one diagnosis. `mcp_session` is an MCP client (mcp.client.client.Client) or None for no tools."""
    import anthropic

    tr = Transcript(model)
    t0 = time.time()
    tools: list[dict[str, Any]] = []
    if mcp_session is not None:
        listed = await mcp_session.list_tools()
        for t in listed.tools:
            tools.append({"name": t.name, "description": t.description or "", "input_schema": t.inputSchema})
    tools.append(SUBMIT_TOOL)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    budget_exhausted = False
    try:
        for _ in range(max_tool_calls + 4):
            tr.turns += 1
            params: dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM,
                tools=[SUBMIT_TOOL] if budget_exhausted else tools,
                messages=messages,
                output_config={"effort": effort},
                # every turn resends the whole conversation; caching makes the repeated prefix ~10x cheaper
                cache_control={"type": "ephemeral"},
            )
            if model in FALLBACK_MODELS:
                resp = await client.beta.messages.create(
                    **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                resp = await client.messages.create(**params)
            tr.input_tokens += resp.usage.input_tokens
            tr.output_tokens += resp.usage.output_tokens
            tr.cache_read_tokens += getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            tr.cache_write_tokens += getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
            tr.stop_reason = resp.stop_reason
            if resp.stop_reason == "refusal":
                tr.error = "refusal"
                break
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                if tr.diagnosis is None:
                    messages.append({"role": "user", "content": "Please call submit_diagnosis now."})
                    budget_exhausted = True
                    continue
                break
            results = []
            for b in uses:
                if b.name == "submit_diagnosis":
                    d = dict(b.input)
                    d["fix"] = _clean_fix(d.get("fix"))
                    tr.diagnosis = d
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": "received"})
                    continue
                if mcp_session is None or len(tr.tool_calls) >= max_tool_calls:
                    budget_exhausted = True
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": b.id,
                            "is_error": True,
                            "content": "Tool budget exhausted; call submit_diagnosis now.",
                        }
                    )
                    continue
                ts = time.time()
                try:
                    r = await mcp_session.call_tool(b.name, dict(b.input))
                    text = r.content[0].text if r.content else json.dumps(r.structured_content)
                    is_err = bool(getattr(r, "is_error", False))
                except Exception as e:  # tool errors go back to the model
                    text, is_err = f"error: {e}", True
                tr.tool_calls.append(
                    {
                        "name": b.name,
                        "args": dict(b.input),
                        "chars": len(text),
                        "error": is_err,
                        "s": round(time.time() - ts, 3),
                    }
                )
                results.append(
                    {"type": "tool_result", "tool_use_id": b.id, "content": text, "is_error": is_err}
                )
            messages.append({"role": "user", "content": results})
            if tr.diagnosis is not None:
                break
            if len(tr.tool_calls) >= max_tool_calls:
                budget_exhausted = True
    except anthropic.APIStatusError as e:
        tr.error = f"api {e.status_code}: {e.message}"
    except anthropic.APIConnectionError as e:
        tr.error = f"connection: {e}"
    tr.wall_s = time.time() - t0
    return tr
