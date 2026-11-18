"""Wire schemas for the ingestion API."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class MetricIn(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    step: int = Field(ge=0)
    ts: float
    name: str = Field(min_length=1, max_length=128)
    layer: str = Field(default="", max_length=256)
    value: float | None

    @field_validator("value", mode="before")
    @classmethod
    def _nan_ok(cls, v: Any) -> Any:
        # NaN/Inf are meaningful telemetry; JSON may carry them as strings
        if isinstance(v, str) and v.lower() in ("nan", "inf", "-inf", "infinity", "-infinity"):
            return float(v)
        return v


class EventIn(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    step: int = Field(ge=0)
    ts: float
    type: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


class RunIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    project: str = Field(min_length=1, max_length=128)
    name: str | None = None
    status: str = "running"
    config: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, Any] = Field(default_factory=dict)
    started_at: float | None = None


class StatusIn(BaseModel):
    status: str


class CommandIn(BaseModel):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    requested_by: str = "api"


class AckIn(BaseModel):
    status: str
    result: dict[str, Any] = Field(default_factory=dict)


class ApproveIn(BaseModel):
    approved_by: str
    approve: bool = True


def validate_records(
    model: type[BaseModel], raw: list[Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate records one by one: bad records are rejected without failing the batch."""
    ok, bad = [], []
    for i, r in enumerate(raw):
        try:
            rec = model.model_validate(r).model_dump()
            if "value" in rec and rec["value"] is not None and not math.isfinite(rec["value"]):
                rec["value"] = float(rec["value"])
            ok.append(rec)
        except ValidationError as e:
            bad.append({"index": i, "error": e.errors(include_url=False)[0]["msg"]})
    return ok, bad
