# Codex Findings: Repair Child Component Replay Manifests

UTC: 2026-05-27T02:59:31Z

## Verdict

Repair-budget child component replay is now queue-owned. The repair waterfill
queue emits a child replay manifest collection before materializer binding, and
the binding report flattens that collection into per-child evidence rows.

## What Changed

- Added `frontier_rate_attack_repair_budget_child_component_replay_manifests.v1`.
- Added `tools/build_frontier_repair_budget_child_component_replay_manifests.py`.
- Wired the repair waterfill queue step order:
  `work_order -> materialization_plan -> child_component_replay_manifests -> binding -> execution_audit`.
- Taught materializer-manifest loading to flatten replay manifest collections.
- Preserved replay evidence for child rows even when the child archive is not
  byte-closed yet.
- Cleared the stale `component_response_replay_required_before_budget_spend`
  execution blocker when binding supplies replay evidence.

## Live Artifact

Live MLX repair plan:
`experiments/results/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/frontier_repair_budget_waterfill/frontier_mlx_repair_dynamics_paired_reference_20260526t155611z_repair_budget_waterfill/global_many_op_rate_distortion_receiver_campaign/repair_budget_materialization_plan.json`

Emitted replay manifest:
`experiments/results/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/frontier_repair_budget_waterfill/frontier_mlx_repair_dynamics_paired_reference_20260526t155611z_repair_budget_waterfill/global_many_op_rate_distortion_receiver_campaign/repair_budget_child_component_replay_manifests.json`

Observed live result:
- `manifest_count=4`
- `component_response_replayed_count=4`
- `byte_closed_candidate_emitted_count=0`
- binding report now records `component_response_replayed=true` for all four
  spent-budget repair children
- execution report now records `component_response_replayed_count=5` including
  the rate-only parent and four repair children

## Remaining Blocker

The repair children are still correctly blocked on byte-closed child archive
materialization and receiver-runtime consumption. MLX replay evidence is local
materialization signal only; it does not grant budget spend, exact dispatch,
promotion, rank/kill, or score authority.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_repair_budget_materialization_execution.py src/tac/tests/test_frontier_rate_attack_feedback.py tools/build_frontier_repair_budget_child_component_replay_manifests.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_budget_materialization_execution.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `.venv/bin/python tools/lane_maturity.py validate`
- review policy checks on touched scheduler/test files: 0 violations
