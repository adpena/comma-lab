# Codex Findings: Repair Real-Archive Five-Family Runtime Custody

UTC: 2026-05-28T05:44:33Z

## Scope

- `src/tac/optimization/repair_archive_candidate_intake.py`
- `src/tac/optimization/repair_family_exact_ready_bridge.py`
- `tools/build_repair_campaign_work_order_from_archives.py`
- `src/tac/tests/test_repair_campaign_materialization_queue.py`
- `.omx/research/repair_real_archive_psv3_all5_floor_loop_20260528T053526Z/`

## Findings

No blocking findings after three recursive adversarial review passes.

The five repair families are now executable from a real byte-closed archive candidate:

- `posenet_null_bottom_decile`
- `segnet_class_region_waterfill`
- `per_region_selector_codec`
- `frame0_k16_palette_asymmetry`
- `entropy_boundary_probe`

The live PSV3 archive was compiled into a repair work order, scored, queued,
executed locally, lifted into exact-ready bridge inputs, closed into
contest-shaped submission/runtime packets, and re-bridged with runtime content
tree custody proven for all five rows.

## Live Artifact Evidence

- Work order: `.omx/research/repair_real_archive_psv3_all5_floor_loop_20260528T053526Z/work_order.json`
- Score report: `.omx/research/repair_real_archive_psv3_all5_floor_loop_20260528T053526Z/score_report.json`
- Materialization queue: `.omx/research/repair_real_archive_psv3_all5_floor_loop_20260528T053526Z/repair_materialization_queue.json`
- Runtime closure report: `.omx/research/repair_real_archive_psv3_all5_floor_loop_20260528T053526Z/submission_runtime_closure_report.json`
- Runtime-custody floor-loop summary: `.omx/research/repair_real_archive_psv3_all5_floor_loop_20260528T053526Z/floor_loop_runtime_custody_summary.json`

Key live counts:

- `archive_bound_exact_handoff_candidate_count`: 5
- `exact_eval_handoff_candidate_count`: 5
- `exact_ready_bridge_candidate_count`: 5
- `exact_ready_bridge_runtime_content_tree_custody_proven_count`: 5
- `posterior_learning_signal_count`: 5
- `ready_for_exact_eval_dispatch`: false
- terminal outcome: `strictly_better_archive_bound_candidate_exact_axis_blocked`

## Bug Class Extinguished

The exact-ready bridge previously preserved proof custody paths but dropped
`receiver_contract_satisfied` and `receiver_contract_kind` when writing the
source optimizer queue. That made the downstream materializer submission
closure reject otherwise valid proof-backed rows with
`receiver_contract_not_satisfied`.

Fix:

- propagate receiver proof schema, pass status, contract id, contract kind, and
  `receiver_contract_satisfied` into bridge source rows;
- bind submission runtime custody to candidate-specific archive manifests when
  same-sha archive rows would otherwise be ambiguous;
- add a regression that executes all five families from a real archive, checks
  receiver contract preservation, and proves the submission closure can close
  all five rows.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/repair_archive_candidate_intake.py src/tac/optimization/repair_family_exact_ready_bridge.py tools/build_repair_campaign_work_order_from_archives.py src/tac/tests/test_repair_campaign_materialization_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_materialization_queue.py::test_archive_candidate_intake_builds_real_archive_five_family_work_order src/tac/tests/test_repair_campaign_materialization_queue.py::test_real_archive_intake_runs_all_families_through_floor_loop -q`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_materialization_queue.py src/tac/tests/test_repair_family_materializers.py::test_repair_exact_ready_bridge_emits_blocked_source_queue -q`
- `.venv/bin/python tools/review_tracker.py policy-check src/tac/optimization/repair_archive_candidate_intake.py src/tac/optimization/repair_family_exact_ready_bridge.py tools/build_repair_campaign_work_order_from_archives.py src/tac/tests/test_repair_campaign_materialization_queue.py`
- Recursive adversarial review bundle `c7d26eab3e318067`: clean rounds `f004adc9f779`, `8b0d3fccfaf5`, `fba5a08af538`; counter sealed at 3.

## Remaining Exact-Axis Blocker

This is not a score claim. The rows remain fail-closed until contest CPU/CUDA
auth eval and lane-dispatch claim authority are present.
