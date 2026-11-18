#!/usr/bin/env bash
# PreToolUse(Edit|Write): the held-out test set and its results must never be hand-edited (plan.md §9).
f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  */benchmark/ground_truth/test.jsonl|*/results/eval/test-*)
    jq -n --arg f "$f" '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny",
      permissionDecisionReason: ("\($f) is part of the held-out test set. Regenerate it with benchmark/run_scenarios.py or eval/harness.py instead of editing it.")}}'
    ;;
esac
exit 0
