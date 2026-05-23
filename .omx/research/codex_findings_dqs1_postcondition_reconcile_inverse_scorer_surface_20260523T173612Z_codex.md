# Codex Findings: DQS1 Postcondition Reconcile And Inverse Scorer Surface

**UTC**: 2026-05-23T17:36:12Z
**Lanes**:
- `lane_codex_materializer_work_queue_executor_bridge_20260523`
- `lane_inverse_steganalysis_acquisition_surface_20260523`

## Findings

The DQS1 queue was bottlenecked by state/hash churn rather than compute. After
retention timeout/resource edits, already-satisfied steps had stale definition
hashes and would have been requeued by a blanket worker run. This is wasted
wall-clock on a powerful local machine: the artifact postconditions already
proved the work.

The stronger byte-shaving formulation is inverse scorer learning, not raw byte
mutation enumeration. Existing MLX scorer-response rows can now be projected
into planning-only receiver-coordinate cells: scorer null-space,
receiver-sufficient-statistic, and fragile-boundary cells. Native MLX window
rows are allowed only behind an explicit planning blocker and false-authority
boundary.

## Landed

- Added `reconcile_satisfied_queued_steps(...)` and
  `tools/experiment_queue.py reconcile-satisfied`.
- Hardened reconciliation so queued steps with satisfied postconditions cannot
  skip unsatisfied dependencies.
- Hardened queue initialization so definition drift with a running downstream
  fails before mutating state.
- Made `tools/experiment_queue.py performance` read-only so telemetry reads do
  not requeue drifted definitions.
- Propagated queue-level paused/frozen mode into staircase DAG plans.
- De-duplicated recursive artifact footprint telemetry while preserving raw
  record bytes separately.
- Blocked ambiguous multi-context materializer backlog rows instead of
  executing one context while claiming multiple source units.
- Reconciled 15 already-satisfied DQS1 steps, refreshed definition hashes, and
  ran the single missing rank024 eureka step.
- Final DQS1 observation: 16 succeeded, 0 queued, 0 failed, 0 running, 0
  definition drift.
- Hardened DQS1 retention/cleanup policy to `local_io_heavy` with 1200s
  timeout in code, checked-in queue definition, and tests.
- Added `tac.optimization.scorer_inverse_decision_surface`.
- Extended byte-shaving signal surfaces and campaign planning with
  `scorer_inverse_surface_cell` units and
  `probe_inverse_scorer_surface_cell` operations.
- Built planning artifacts:
  - `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_byte_shaving_signal_surface_20260523_codex.json`
  - `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_byte_shaving_campaign_plan_20260523_codex.json`

## Evidence

- DQS1 rank023 raw retention previously timed out at 120s; after policy
  hardening it succeeded in 411.0969s.
- DQS1 rank024 local CPU advisory succeeded in 916.2876s.
- DQS1 rank024 eureka succeeded in 0.2613s after postcondition reconciliation.
- Rank023 and rank024 eureka rows remain observe-only:
  - rank023 conservative projected contest score `0.19203178295713672`,
    eureka margin `-0.000003499999999961867`.
  - rank024 conservative projected contest score `0.19203078295713674`,
    eureka margin `-0.0000024999999999886224`.
- Inverse scorer surface consumed 1200 MLX native-window scorer-response rows,
  emitted 32 planning cells, and kept `score_claim=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_signal_surface_builder.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_byte_shaving_signal_surface_builder.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_staircase_dag.py`
- `.venv/bin/python -m ruff check ...` on touched scheduler, inverse-scorer,
  signal-surface, staircase, and test files.
- `.venv/bin/python -m py_compile ...` on touched scheduler, inverse-scorer,
  signal-surface, staircase, and test files.
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml observe --tail-lines 8`

## Next Integration

- Teach the materializer registry how to turn
  `probe_inverse_scorer_surface_cell` rows into byte-closed candidate archives.
- Feed `experiment_queue_performance_summary.v1` into the inverse acquisition
  model so expected score gain is normalized by seconds, artifact GB, and
  resource kind.
- Promote only after same-runtime locality/inflate controls and exact
  contest CPU/CUDA auth eval. MLX/native-window inverse-surface rows remain
  planning signal only.
