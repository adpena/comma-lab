# Codex Findings: DQS1 MLX Queue Wiring

Timestamp UTC: 2026-05-23T09:34:22Z

## Scope

Wired the DQS1 local-first experiment queue to use local MLX as an explicit
candidate-generation/debug substrate after a local CPU advisory run. The prior
queue declared `local_mlx: 1` capacity but had no MLX-consuming step, so the
resource could not be saturated by normal worker execution.

## Fix

- Added `tools/build_mlx_scorer_input_cache_from_local_advisory.py`.
- Added `tools/run_mlx_scorer_response_from_local_advisory.py`.
- Extended `tools/build_dqs1_local_first_queue.py` and
  `tac.comma_lab.scheduler.dqs1_local_first_queue` with opt-in MLX local
  advisory debug steps.
- Regenerated `configs/experiment_queues/dqs1_pairset_local_first.yaml` with
  MLX enabled for the current queued candidate.
- Reconciled `.omx/state/experiment_queue_dqs1_pairset_local_first.sqlite` so
  the checked-in queue now has seven queued current steps and zero blocking
  stale orphans.

## Safety Contract

- The MLX path is explicitly gated by `--include-mlx-local-advisory-debug` plus
  `--allow-large-mlx-cache`; the builder fails closed without the large-cache
  acknowledgement.
- MLX scorer execution remains `[macOS-MLX research-signal]`, not score
  authority.
- Candidate caches stamped from local CPU advisory identity are eligible only
  for local debug and speed/quality delta measurement, not auth-axis transfer
  calibration or exact-eval spend triage.
- MLX response batching remains singleton-only until the batch-shape invariance
  gate passes.

## Current DQS1 Queue Observation

`tools/experiment_queue.py validate` now derives:

- `local_only.max_parallel = 2`
- `resource_limits = {"local_cpu": 1, "local_mlx": 1}`
- `step_count = 7`

The worker dry-run still starts no process and plans the first CPU step:
`pairset_drop_two_r029_020_p0259_0430.plan_packet`.

State reconciliation artifact:
`.omx/research/dqs1_queue_state_reconciliation_20260523T093324Z_codex.json`

- `blocking_orphan_count_before = 0`
- `blocking_orphan_count_after = 0`
- `retired_step_count = 0`
- `after.status_counts = {"queued": 7}`

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_scorer_response.py`
- `.venv/bin/ruff check src/comma_lab/scheduler/dqs1_local_first_queue.py tools/build_dqs1_local_first_queue.py tools/build_mlx_scorer_input_cache_from_local_advisory.py tools/run_mlx_scorer_response_from_local_advisory.py tools/experiment_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py`
- `git diff --check`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --max-steps 1 --max-idle-cycles 0 --idle-sleep-seconds 0 --no-reload-definition`

## Next Hooks

- Execute the DQS1 queue under the local worker and harvest the first real
  MLX speed/quality delta for the currently queued candidate.
- Feed MLX response rows back into the byte-shaving signal surface, Pareto
  planner, and learned sweep acquisition rules once local CPU advisory and MLX
  response are both present.
- Add retention/compaction policy for per-candidate `mlx_delta_cache/` outputs
  so full caches remain reproducible but do not recreate the prior raw-inflation
  disk-pressure failure mode.
