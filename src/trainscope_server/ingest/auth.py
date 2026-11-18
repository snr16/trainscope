"""Per-project API tokens.

Tokens come from ``$TRAINSCOPE_TOKENS`` as ``project:token,project2:token2`` (``*:token`` = all projects).
If no tokens are configured, auth is disabled (local/dev use).
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException


def load_tokens() -> dict[str, str]:
    raw = os.environ.get("TRAINSCOPE_TOKENS", "")
    out: dict[str, str] = {}
    for part in raw.split(","):
        if ":" in part:
            proj, tok = part.split(":", 1)
            out[tok.strip()] = proj.strip()
    return out


class Auth:
    def __init__(self, tokens: dict[str, str] | None = None):
        self.tokens = load_tokens() if tokens is None else tokens

    @property
    def enabled(self) -> bool:
        return bool(self.tokens)

    def project_for(self, authorization: str | None) -> str | None:
        """Returns the project the token grants ('*' = any), or raises 401."""
        if not self.enabled:
            return "*"
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        tok = authorization.split(" ", 1)[1].strip()
        for known, proj in self.tokens.items():
            if hmac.compare_digest(known, tok):
                return proj
        raise HTTPException(401, "invalid token")

    def dependency(self):
        def dep(authorization: str | None = Header(default=None)) -> str | None:
            return self.project_for(authorization)

        return dep


def check_project(granted: str | None, project: str) -> None:
    if granted not in ("*", project):
        raise HTTPException(403, f"token not valid for project {project}")
