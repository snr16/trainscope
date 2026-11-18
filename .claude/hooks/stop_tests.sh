#!/usr/bin/env bash
# Stop: if Python files changed since the last green run, run the fast test suite and report failures
# back to Claude (once — stop_hook_active prevents loops).
input=$(cat)
[ "$(echo "$input" | jq -r '.stop_hook_active // false')" = "true" ] && exit 0
root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root" || exit 0
stamp=".claude/.last-green-tests"
if [ -f "$stamp" ] && [ -z "$(find src tests eval benchmark -name '*.py' -newer "$stamp" -print -quit 2>/dev/null)" ]; then
  exit 0
fi
out=$(.venv/bin/python -m pytest -q -m "not slow" -x 2>&1)
if [ $? -eq 0 ]; then
  touch "$stamp"
  exit 0
fi
jq -n --arg r "Fast tests are failing:\n$(echo "$out" | tail -30)" '{decision: "block", reason: $r}'
