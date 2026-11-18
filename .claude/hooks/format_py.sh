#!/usr/bin/env bash
# PostToolUse(Edit|Write): format + autofix Python files with the project's ruff. Never blocks.
f=$(jq -r '.tool_response.filePath // .tool_input.file_path // empty')
case "$f" in *.py) ;; *) exit 0 ;; esac
[ -f "$f" ] || exit 0
root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
ruff="$root/.venv/bin/ruff"
[ -x "$ruff" ] || exit 0
"$ruff" format --quiet "$f"
"$ruff" check --fix --quiet "$f" >/dev/null
exit 0
