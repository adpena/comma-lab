# Codex Session Summary - 2026-05-24T11:59:16Z

Evidence grade: `[repo-local/live-queue/MLX-research-signal]`

## Scope

Continued the two-week optimization tranche by removing the pass-level
coarseness from MLX learned-sweep auto batch roots. The queue/autopilot stack
now supports exact `queue_candidate_id` filters end to end, so the planner can
water-fill individual high-utility learned-sweep rows instead of launching a
whole optimization pass and hoping the right row is first.

This remains local MLX research signal only. It creates no score authority,
promotion eligibility, rank/kill authority, or exact-eval dispatch authority.

## Code Changes

- `src/tac/optimization/mlx_dynamic_learned_sweep_local_actuator.py`
  accepts `candidate_ids` and `queue_candidate_ids`, validates exact-row
  filters, refuses missing exact rows, and emits filter proof fields:
  `executed_filter_match`, `executed_filter_violation_count`,
  `executed_unique_candidate_id`, and
  `executed_unique_queue_candidate_id`.
- `src/tac/optimization/mlx_dynamic_learned_sweep_local_autopilot.py`
  threads those filters through actuation, rejects exact-row filters in
  chained/ambiguous modes, and rolls exact-row proof up to the top-level
  autopilot summary.
- `tools/run_mlx_dynamic_learned_sweep_autopilot.py` exposes
  `--candidate-id` and `--queue-candidate-id`.
- `src/comma_lab/scheduler/mlx_learned_sweep_autopilot_queue.py`
  threads filters into command specs, validates ready rows against those
  filters, and adds queue postconditions requiring
  `executed_filter_match=true` plus the expected unique queue-candidate id.
- `tools/build_mlx_learned_sweep_autopilot_queue.py` exposes the filter args
  for manual/single-root queue builds.
- `src/tac/optimization/mlx_learned_sweep_batch_roots.py` now emits
  `queue_candidate_waterfill` roots with `row_specific_filter_supported=true`,
  `queue_candidate_filter_supported=true`, and disjoint exact row filters.

## Live Queue Proof

Live source root:

`experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/`

Auto-row batch artifacts:

- Root plan:
  `mlx_autopilot_auto_row_batch_roots_20260524T115743Z.json`
- Queue:
  `mlx_autopilot_auto_row_batch_queue_20260524T115743Z.json`
- Queue state:
  `.omx/state/experiment_queue_mlx_learned_sweep_auto_row_batch_live_20260524T115743Z.sqlite`

Selected exact-row roots:

- `mlx_scorer_response:window:544:545::mlx_local_response::micro`
- `mlx_scorer_response:window:496:497::mlx_local_response::intermediate`

Queue validation:

- schema `experiment_queue.v1`
- `experiment_count=2`
- `step_count=2`
- `local_mlx` parallelism cap `2`
- `valid=true`

Worker execution:

- `success_count=2`
- `failure_count=0`
- `claim_refused_count=0`
- `resource_limits={"local_mlx": 2}`
- elapsed `15.0219s` and `15.0235s`
- no failed postconditions
- no postcondition errors
- `timed_out=false`

Both summaries reported:

- schema `mlx_dynamic_learned_sweep_local_autopilot.v1`
- `executed_filter_match=true`
- `executed_row_count=1`
- `new_observation_row_count=1`
- `local_mlx_device_used=true`
- `gpu_launched=false`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The first live row-filter attempt at `20260524T115525Z` intentionally caught a
real integration gap: nested actuation proved the exact row ran, but the
top-level autopilot summary did not expose `executed_filter_match`, so the
queue postcondition failed. The follow-up patch rolled the filter proof up and
the `20260524T115743Z` queue passed.

## Verification

Focused:

```text
.venv/bin/python -m pytest -q \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py \
  src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py
```

Result: `30 passed in 1.00s`.

Broader MLX/scheduler tranche:

```text
.venv/bin/python -m pytest -q \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py \
  src/tac/tests/test_mlx_effective_spend_triage_selection.py \
  src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep.py \
  src/tac/tests/test_mlx_dynamic_sweep_observations.py \
  src/tac/tests/test_cross_family_candidate_portfolio.py \
  src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py \
  src/tac/tests/test_mlx_execution_queue.py \
  src/tac/tests/test_staircase_dag.py
```

Result: `139 passed in 4.28s`.

Additional checks:

- `ruff check` on touched MLX/autopilot/queue/tool/test files passed.
- `git diff --check` passed.

## Remaining Gaps

- The live exact-row batch runs one row per root. Next step is larger
  row-specific saturation with root groups sized to the available local MLX
  slots and learned observation value.
- Candidate-level filters are wired but auto roots currently prefer exact
  queue-candidate filters. Candidate-root chained walks should be a separate
  mode with explicit duplicate guards.
- This still consumes MLX response rows only. Archive materialization,
  byte-closed export, exact CPU/CUDA auth-eval dispatch, and promotion gates
  remain downstream authority surfaces.
