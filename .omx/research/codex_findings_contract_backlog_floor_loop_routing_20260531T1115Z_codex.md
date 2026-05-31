# Contract Backlog Floor-Loop Routing

Date: 2026-05-31T11:15Z
Author: Codex

## Finding

The bounded repair floor loop now consumes the shared
`tac_archive_bound_candidate_contract_migration_backlog_queue.v1` artifact as
executable routing input. Migration-required archive/entropy rows are no longer
only audit output: the runner emits
`repair_campaign_contract_migration_backlog_work_selection.v1`, selects the
highest-pressure smallest byte-closed migration/blocker tasks, and carries the
selected row blockers into the loop summary before any budget spend can occur.

## Code Changes

- `tools/run_repair_campaign_autonomous_floor_loop.py`
  - Added `--contract-migration-backlog`.
  - Emits `repair_contract_migration_backlog_work_selection.json`.
  - Adds selected migration family/stage/scope/entropy-position summaries.
  - Keeps all budget, score, promotion, and exact-dispatch authority false.
- `src/tac/optimization/repair_family_stack_search.py`
  - Fixed archive-bound runtime proof custody: exact handoff now reads
    `runtime_consumption_proof_path` from the shared contract and contract
    identity, not only from the candidate-archive subobject.
  - Fixed composed entropy-stage chain custody: exact handoff binds to the
    final chain archive/proof instead of the first stage sharing the same
    archive SHA.
- `src/tac/tests/test_repair_campaign_materialization_queue.py`
  - Added backlog-consumption coverage.
  - Updated the frontier no-op test to assert contract-first refusal rather
    than allowing non-contract rows through.

## Live Artifact

Ran the bounded loop against:

- Queue:
  `.omx/research/repair_multi_archive_autonomous_stage_disciplined_psv3_fec6_20260528T0648Z/repair_materialization_queue.json`
- Backlog:
  `.omx/research/archive_bound_contract_migration_backlog_queue_20260531T1038Z.json`
- Output:
  `.omx/research/repair_floor_loop_contract_backlog_routing_20260531T1115Z/floor_loop_summary.json`

Key results:

- `contract_migration_backlog_consumed=true`
- `contract_migration_backlog_row_count=22`
- `contract_migration_selected_work_order_count=12`
- Selected families:
  `archive_candidate`, `dqs1`, `zip_ordering`, `ans_coder`, `fec`, `header`,
  `pr95`
- Selected entropy positions:
  `archive_entropy_position_unknown`, `before_entropy_coder`,
  `after_entropy_coder`, `at_entropy_coder`
- `archive_bound_exact_handoff_candidate_count=10`
- `exact_ready_bridge_candidate_count=12`
- `measured_mlx_posterior_budget_routing_update_count=52`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- stop reason:
  `strictly_better_archive_bound_candidate_exact_axis_blocked`

## Verification

- `.venv/bin/ruff check tools/run_repair_campaign_autonomous_floor_loop.py src/tac/optimization/repair_family_stack_search.py src/tac/tests/test_repair_campaign_materialization_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`
  - `10 passed`
- `.venv/bin/python tools/lane_maturity.py validate`
  - `1567 lane(s) validated cleanly`

## Next Blocker

The runner now routes migration-required rows automatically, but exact promotion
still needs contest CPU/CUDA authority plus submission runtime content-tree
custody for the selected archive-bound candidates. The next unblocked work is
to materialize contract rows for the selected DQS1/ZIP-ordering/ANS/FEC/header/
PR95 backlog families or write precise blockers where a byte-closed adapter is
still missing.
