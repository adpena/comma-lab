# Codex Findings: NeRV Rate Allocator Queue And Fresh-Eyes Hardening

UTC: 2026-06-02T12:39:58Z

## Landed

- Added `nerv_rate_allocator_work_queue.v1`, a planner-only queue compiled from
  `nerv_rate_allocator_bridge.v1`.
- Wired the queue builder into NeRV Gate #35 so the normal preflight chain now
  checks modelsize curve, control inventory, implementation sweep, top-priority
  seam, master bridge, rate bridge, and rate queue together.
- Hardened `pareto_carrier_fit_consumer` so bytes-only carrier rows cannot
  default missing `d_seg`/`d_pose` to zero and become perfect-distortion
  frontier candidates.
- Hardened the PR95 baseline seam so missing PR metadata, missing head SHA/ref,
  wrong PR URL/state, non-PR95 upstream remotes, and upstream HEAD mismatches
  block baseline authority.

## Durable Artifacts

- `.omx/research/nerv_modelsize_archive_curve_20260602T124222Z.json`
- `.omx/research/nerv_control_inventory_20260602T124222Z_queue_chain.json`
- `.omx/research/nerv_implementation_design_sweep_20260602T124222Z_queue_chain.json`
- `.omx/research/nerv_top_priority_stack_seam_20260602T124222Z_queue_chain.json`
- `.omx/research/nerv_master_consumer_bridge_20260602T124222Z_queue_chain.json`
- `.omx/research/nerv_rate_allocator_bridge_20260602T124222Z_queue_chain.json`
- `.omx/research/nerv_rate_allocator_queue_20260602T124222Z.json`

## Verification

- `pytest src/tac/tests/test_nerv_control_surfaces.py src/tac/tests/test_nerv_top_priority_stack_seam.py src/tac/cathedral_consumers/pareto_carrier_fit_consumer/tests/test_pareto_carrier_fit_consumer.py -q`
  passed: 49 tests.
- Ruff passed on the edited code and tests.
- Gate #35 direct run passed with `bridge_units=47`, `rate_work_orders=27`,
  `queue_rows=27`, `memo_refs=399`, `blockers=102`, `blocked_dispatch=True`.

## Remaining Fresh-Eyes Signal

- `src/tac/substrates/_shared/mlx_score_aware/modelsize_budget_plan.py` still
  labels projected/advisory inputs as `measured_modelsize_budget_selected`.
  Next hardening should split measured vs projected schemas and reserve
  measured status for receiver-closed byte fields.
- `src/tac/analysis/nerv_source_parity_contract.py` was flagged by the
  fresh-eyes subagent in the dirty shared checkout, but that file is absent
  from the current clean `origin/main` worktree. Re-evaluate only if that WIP is
  reconciled to main.

## Verdict

GO for local planner/allocator ingestion. NO-GO for score claims, promotion,
rank/kill, exact/full-video/CUDA dispatch, or real bit assignment.
