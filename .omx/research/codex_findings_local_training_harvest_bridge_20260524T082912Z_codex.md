# Codex Findings: Queue-Owned Local Training Harvest Bridge

UTC: 2026-05-24T08:29:12Z

## Landing

- Added `src/comma_lab/scheduler/local_training_harvest.py`.
- Added `tools/harvest_local_training_optimizer_candidates.py`.
- Exported the harvest bridge from `comma_lab.scheduler`.
- Added focused tests in
  `src/tac/tests/test_local_training_optimizer_candidate_harvest.py`.

## Behavior

The bridge reads an `experiment_queue.v1` definition plus its SQLite state and
harvests only succeeded local-training steps with completed
`representation_training_manifest.json` sidecars. It refuses plan sidecars such
as `representation_training_plan.json`, refuses failed/queued rows, validates
false-authority fields, and then delegates candidate construction to
`tac.optimizer.candidate_queue.build_candidate_queue`.

## Live PR95/HNeRV MLX Harvest

Applied to:
`experiments/results/pr95_hnerv_mlx_queue_smoke_20260524T082140Z`

Output:
`experiments/results/pr95_hnerv_mlx_queue_smoke_20260524T082140Z/optimizer_candidate_queue.harvested.json`

Observed:

- `harvested_representation_manifest_count=3`
- `n_candidates=3`
- `dispatch_ready_count=0`
- rank fields are `seconds_per_step_cost_signal_not_score`

## Authority Boundary

Harvest metadata carries:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The harvested optimizer queue remains planning-only; it ranks local MLX cost
signals for throughput scheduling, not score or promotion.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/local_training_harvest.py tools/harvest_local_training_optimizer_candidates.py src/comma_lab/scheduler/__init__.py src/tac/tests/test_local_training_optimizer_candidate_harvest.py`
- `.venv/bin/python -m pytest src/tac/tests/test_local_training_optimizer_candidate_harvest.py -q`

Observed result: 5 tests passed.

## Remaining Gap

This closes the manual/stale-sidecar harvest gap. The next frontier-enabling
gap is source-checkpoint PyTorch-to-MLX forward parity for PR95/HNeRV, so
queue-owned local MLX timing can move from synthetic throughput signal to
source-faithful representation-training signal.
