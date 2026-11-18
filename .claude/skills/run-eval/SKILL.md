---
name: run-eval
description: Run the TrainScope assistant evaluation (conditions A–E, model comparison, signal ablations, fix verification). Use when asked to evaluate assistants, compare conditions/models, or verify fixes.
---

# Running the evaluation

LLM conditions call the Claude API and **cost money**. Before running, estimate:
- runs × conditions × models × repeats;
- the model set in `eval/config.yaml` (default `claude-haiku-5-5`, roughly $0.002–0.01 per diagnosis; Opus is ~40x more).

Confirm with the user before any run over ~$5.

- **Credentials:** `ANTHROPIC_API_KEY` in the environment or in `.env` at the repo root (gitignored). `eval/harness.py` loads it.
- **Smoke test:** `python -m eval.harness --split dev --conditions A D --limit 3`.
- **Main comparison (dev):** `python -m eval.harness --split dev --conditions A B C D E --tag dev-main`.
- **Model:** set `models` in `eval/config.yaml` (list several to compare); `--models` overrides it for one run.
- **Variance:** use `--repeats 3` on a subset (`--limit 20`). Current models don't accept `temperature`, so variance comes from repeats.
- **Signal ablation:** run condition D with `--hide-signals gradients`, then `activations`, `system`, `data`, with one `--tag` per run.
- **Fix verification:** `python -m eval.verify_fix results/eval/<tag>.jsonl --workers 4`. This reruns training for every proposed fix, so it uses CPU time, not API cost.
- **Final numbers:** use `--split test` exactly once, with the settings frozen on dev.

Results go to `results/eval/<tag>.jsonl` with every tool call logged. Summarize accuracy, localization, fix validity, verified-fix rate, tool calls, tokens and $/diagnosis per condition and model in the local `REPORT.md` (untracked; don't commit it).
