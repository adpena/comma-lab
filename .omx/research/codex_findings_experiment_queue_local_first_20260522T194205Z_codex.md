# Codex Findings: Local-First Experiment Queue Automation

Date: 2026-05-22T19:42:05Z

## Verdict

The local-first DQS1 pairset loop now has a reusable declarative queue substrate
instead of ad hoc command sequencing.

This is an orchestration/control landing, not score authority. The queue can
select, run, pause, freeze, rewind, and observe experiment steps, but any local
CPU/MLX output it produces remains advisory unless a separate contest-axis auth
eval artifact passes the existing strict gates.

## Landed Surfaces

- Queue library: `src/comma_lab/scheduler/experiment_queue.py`
- Operator CLI: `tools/experiment_queue.py`
- DQS1 queue definition:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`
- Tests: `src/tac/tests/test_experiment_queue.py`
- Generated queue state/log ignore rules in `.gitignore`
- Registered lane:
  `lane_dqs1_pairset_drop_one_rank020_pair0430_local_first_20260522`
- Registered current queue lane:
  `lane_dqs1_pairset_drop_one_rank022_pair0167_local_first_20260522`
- Registered current queue lane after rank022 exact observation:
  `lane_dqs1_pairset_drop_one_rank019_pair0151_local_first_20260522`
- Registered current queue lane after rank019 exact observation:
  `lane_dqs1_pairset_drop_one_rank023_pair0440_local_first_20260522`

## Contract

- Queue definitions are JSON-compatible `.yaml` / `.json` files.
- SQLite stores mutable execution state, append-only queue events, attempts,
  controls, and last-event telemetry.
- Commands must be argv lists. Shell strings are rejected.
- Cloud resources (`cloud_*`, `modal_*`, `cuda_auth`) are hidden unless the CLI
  is run with `--allow-cloud`.
- `control paused|frozen|running` changes queue-wide scheduling without editing
  the definition file.
- `rewind EXPERIMENT STEP` resets the target and, by default, dependent steps so
  stale downstream success state cannot survive a dependency reset.
- Timeout, execution, and postcondition failures mark steps `failed` with
  telemetry instead of leaving durable state stuck at `running`.
- `run-worker --execute --max-steps N` now runs bounded local loops with
  SIGINT/SIGTERM stop requests, queue-definition reload between steps, and
  append-only worker telemetry.
- Queue summaries count only steps present in the current queue definition and
  report rerouted/stale SQLite rows separately as `orphaned_steps`, so a
  rank019-to-rank023 reroute cannot inflate the active pending count.

## Current DQS1 Use

The checked-in queue now encodes the current next rank023/pair0440 local-first
flow after rank019/pair0151 exact CPU observation:

1. Plan selective packet.
2. Materialize submission directory.
3. Run locality controls.
4. Run macOS CPU advisory eval.

The queue defaults to local resources only: `local_cpu=1`, `local_mlx=1`,
`modal_cpu=0`, `modal_gpu=0`. Exact auth anchoring remains a separate explicit
dispatch action until the operator intentionally raises cloud concurrency and
passes `--allow-cloud`.

## Empirical Context

Rank020/pair0430 exact Modal `[contest-CPU]` recovery is already recorded in
the current frontier report and canonical pairset component marginal surfaces:
`0.19202928295713673`, a regression versus rank021 and the same
SegNet-penalized one-byte-drop response class as ranks 013/020/026/027/031.

The local macOS CPU advisory run on the same archive produced
`0.19204028295713674` with `score_claim=false`, confirming the batch/substrate
drift class is still present and that local queue outputs must stay diagnostic
unless anchored.

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml status`
- `.venv/bin/ruff check src/comma_lab/scheduler/experiment_queue.py tools/experiment_queue.py src/comma_lab/scheduler/__init__.py src/tac/tests/test_experiment_queue.py src/tac/canonical_equations/pairset_component_marginal.py src/tac/canonical_equations/tests/test_pairset_component_marginal.py`
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py -q`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml --state .omx/tmp/dqs1_pairset_queue_smoke.sqlite status`

## Next Engineering Tranche

- Add a queue step generator that consumes the latest component-marginal action
  summary and writes the next candidate queue definition without hand editing.
- Add concurrent worker mode with bounded resource pools.
- Add first-class postcondition types for SHA-256 equality, score-axis schema
  validation, and required false-authority payloads.
- Add telemetry export for dashboards: pending/running/failed/succeeded counts,
  elapsed seconds, attempts, and current log paths.
- Add exact-auth queue steps only behind explicit cloud controls and existing
  dispatch-claim gates.
