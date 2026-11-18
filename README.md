# TrainScope

**An AI training-run debugger.** TrainScope records what happens inside a PyTorch training run: gradients, weights, activations, the data pipeline, the GPU/CPU and the loss curves. It flags common failures automatically. It also gives AI assistants (Claude, Cursor, any MCP client) structured tools to investigate a run the way an experienced ML engineer would.

```
training process ──(SDK: buffered, background flush, disk spool)──▶ ingest API ──▶ DuckDB / TimescaleDB
                     ▲                                                         │
                     └── command poller (checkpoint / pause / set_lr) ◀── approvals ◀── MCP server ◀── assistant
                                                                               │
                                                                    detector engine → findings
```

| Part | Where |
|---|---|
| Telemetry SDK (`import trainscope as ts`) | `src/trainscope/` |
| Ingestion API, storage, detectors, command queue | `src/trainscope_server/` |
| MCP server (tools, resources, prompts) | `src/trainscope_mcp/` |
| Failure-injection benchmark | `benchmark/` |
| Assistant evaluation (conditions A–E, fix verification) | `eval/` |

## Quickstart

```bash
pip install -e ".[server,mcp,torch]"
```

```python
import trainscope as ts

run = ts.init(project="cifar", config=cfg)        # local mode: writes ./trainscope.duckdb
ts.watch(model, optimizer=opt, grad_every=50, act_every=200)
loader = ts.wrap_loader(train_loader)              # measures data-loader wait time

for step, (x, y) in enumerate(loader):
    loss = train_step(x, y)
    ts.log(step, loss=loss.item(), lr=sched.get_last_lr()[0])
    ts.log_batch(step, labels=y)                   # optional: label / padding stats
ts.finish()
```

For the Hugging Face Trainer, add `Trainer(..., callbacks=[TrainScopeCallback(project="bert")])` from `trainscope.integrations.hf_trainer`.

**Remote / team mode.** Run `docker compose up` to start TimescaleDB, the ingestion API on port 8765 and the MCP server on port 8766. Then point the SDK at the API with `TRAINSCOPE_URL=http://host:8765` and `TRAINSCOPE_TOKEN=...`. Per-project tokens come from `TRAINSCOPE_TOKENS="proj:token,..."` on the server.

The SDK never blocks or crashes training:
- statistics are computed on the device and gathered with one host transfer per sampled step;
- a daemon thread sends the batches;
- if the server is unreachable, batches are spooled to `~/.trainscope/spool` and replayed later. Writes are idempotent, so retries never duplicate data.

## Dashboard

The ingest server also serves a web dashboard at **http://127.0.0.1:8765/ui/**. With a local DuckDB store:

```bash
trainscope-server --db trainscope.duckdb       # then open http://127.0.0.1:8765/ui/
```

With `docker compose up`, the same dashboard is served from the ingest container on port 8765.

- **Run list:** every run, with live status, progress, last loss and a badge for detector findings. Search by name, project or likely cause, and filter by runs with issues, live runs or crashed runs.
- **Findings:** each finding with its severity, step range, likely causes and the evidence behind it.
- **Curves:** loss (train and eval), learning rate, gradient norm (measured before clipping), gradient carry-over and accuracy. Findings tied to a window of the run are shaded, and NaN steps are marked.
- **Layers:** per-layer gradient norm, update ratio, weight norm, dead units and zero gradients. Outliers and frozen layers are highlighted.
- **System and data:** time spent waiting on the data loader, step time, the memory floor, label balance across the run, and train/eval overlap.
- **Control:** buttons to checkpoint, pause, resume or change the learning rate on a live run, and **Approve / Reject** for anything an assistant has requested.

Set your name in Settings so approvals record who made them. If the server sets `TRAINSCOPE_TOKENS`, add the token there too.

## Using it from an assistant (MCP)

The server implements MCP protocol `2026-07-28` (official Python SDK v2, `MCPServer`), and its handshake also accepts `2025-11-25` and older. It supports two transports:
- **stdio:** `trainscope-mcp --db trainscope.duckdb`
- **streamable HTTP:** `trainscope-mcp --http --port 8766`. Set `TRAINSCOPE_MCP_TOKEN` to require a bearer token.

**Claude Code:**
```bash
claude mcp add trainscope -- /path/to/.venv/bin/trainscope-mcp --db /path/to/trainscope.duckdb
```

**Claude Desktop / Cursor** (`claude_desktop_config.json` / `.cursor/mcp.json`):
```json
{ "mcpServers": { "trainscope": {
    "command": "/path/to/.venv/bin/trainscope-mcp",
    "args": ["--db", "/path/to/trainscope.duckdb"] } } }
```

| Tool | Returns |
|---|---|
| `list_runs` | runs with status, last step and loss |
| `get_run_summary` | config, progress, current metrics, available metrics, open findings |
| `get_metric` | downsampled series (LTTB) plus trend, min/max, change points and NaN count |
| `get_layer_stats` | per-layer grad/weight/activation stats at a step, with outlier layers flagged |
| `get_findings` | detector findings with evidence and suspected causes |
| `get_system_metrics` | step time, data-loader wait, GPU utilization and memory trends |
| `inspect_data` | label distribution, padding ratio, sample ids |
| `compare_runs` / `diff_configs` | where two runs diverge / how their configs differ |
| `request_action` | checkpoint (immediate); pause / resume / set_lr (need human approval) |

Resources: `trainscope://runs`, `trainscope://runs/{id}/summary`, `trainscope://runs/{id}/config`. Prompts: `diagnose_run`, `compare_runs`. Tool outputs are summarized and size-capped, so a 50K-step curve becomes about 60 points plus computed facts.

**Control channel.** The SDK polls for approved commands and executes them only at step boundaries.
- Checkpoints are auto-approved.
- Anything disruptive waits for a human: run `trainscope commands --status pending_approval`, then `trainscope approve <id>`.
- Every command records who requested it and who approved it.

## Detectors

| Detector | Signals | Flags |
|---|---|---|
| `divergence` | loss, NaN events, grad norm | NaN/Inf loss (with the first bad layer), loss rising far above its best |
| `loss_spikes` | loss | rolling robust z-score spikes |
| `plateau` / `bad_initial_loss` | loss, update ratio, `num_classes` | no learning, stuck at chance, initial loss far above ln(K) |
| `exploding_gradients` / `grad_norm_growth` | global and per-layer grad norm | sudden blow-up (top layers named), steady linear growth (missing `zero_grad`) |
| `vanishing_gradients` / `dead_units` | grad norm per layer, activations | near-zero gradients, mostly-dead ReLU layers |
| `frozen_layers` | `weight/trainable`, update ratio | layers that never change |
| `eval_mode_training` | `model/train_mode` | model left in `eval()` while training |
| `overfitting` / `eval_better_than_train` | train vs. eval loss and accuracy | widening gap; suspicious eval > train (leakage) |
| `dataloader_bottleneck` | data wait fraction, GPU util | input pipeline starving compute |
| `memory_leak` | RSS / CUDA / MPS memory | steady growth across steps |
| `amp_overflow` / `fp16_unscaled` | loss scale, skipped steps, grad zeros | scale collapse; fp16 without a scaler |
| `lr_schedule` | logged LR vs. the configured schedule | missing warmup, scheduler stepped at the wrong rate, LR above config |
| `suspicious_data` | label stats, padding ratio | label imbalance or shift, extreme or unmasked padding |

To add a detector, see `.claude/skills/add-detector/SKILL.md`. In short:
1. Create a module in `src/trainscope_server/detectors/`.
2. Register it in `engine.py`.
3. Add a test in `tests/test_detectors.py`.

## Benchmark and evaluation

```bash
python -m benchmark.run_scenarios --split dev --workers 4          # 20 failure types x 3 tasks x seeds
python -m eval.harness --split dev --conditions E                  # detectors only (no API key)
python -m eval.harness --split dev --conditions A B C D            # needs ANTHROPIC_API_KEY (env or .env)
python -m eval.verify_fix results/eval/<tag>.jsonl                 # apply fixes, rerun, check recovery
python -m benchmark.overhead                                       # SDK overhead vs. sampling rate
python -m benchmark.leakage_check                                  # no scenario hints in names/configs/logs
```

There are three CPU-sized tasks with offline synthetic data:
- 10-class vision with a small ResNet-style CNN;
- keyword text classification with a padded 2-layer transformer;
- a char-level tiny GPT.

Failures are injected either as plausible config values or as hidden code bugs. Ground truth never reaches run names, configs or logs. Variants are split into dev and test by a stable hash.

To add a scenario, see `.claude/skills/add-scenario/SKILL.md`.

## Development

```bash
pip install -e ".[server,mcp,dev,bench,eval]"
pytest -q                 # unit + integration + reliability (kill the server mid-run)
pytest -q -m "not slow"   # fast subset
ruff check . && mypy src/trainscope src/trainscope_server src/trainscope_mcp
```

Claude Code project hooks (`.claude/settings.json`):
- ruff formats Python files after each edit;
- hand edits to the held-out test split are blocked;
- the fast tests run when a session ends with Python changes.
