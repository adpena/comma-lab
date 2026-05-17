# L5 v2 Lightning Plan Run-Identity Hardening

Date: 2026-05-17
Author: Codex
Scope: TT5L L5 v2 side-info effect-curve Lightning paired-axis plan

## Finding

The TT5L side-info effect-curve validator now requires per-variant CPU/CUDA
cells to agree on both `pair_group_id` and `run_id`. The Lightning paired-axis
dry-run plan already carried `pair_group_id`, but it did not carry `run_id` in
the cell or queue metadata. That would force a future harvester or operator to
repair run identity out-of-band before the aggregate effect-curve artifact could
pass validation.

## Fix

- Added `run_id` to every Lightning paired-axis plan cell.
- Added `run_id` to each exact-eval job's queue metadata.
- Added invariant checks that queue metadata agrees with cell-level
  `pair_group_id` and `run_id`.
- Added readiness validation blockers for missing cell `pair_group_id`,
  missing cell `run_id`, and spec queue-metadata mismatches.
- Regenerated the Lightning paired-axis plan and architecture-lock packet.

## Verification

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan.py \
  src/tac/tests/test_l5_staircase_v2.py

.venv/bin/python -m pytest \
  src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan.py -q

.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_readiness_surfaces_current_lightning_paired_axis_plan \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_rejects_missing_run_id \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_allows_head_only_drift \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_blocks_relevant_source_drift \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_rejects_missing_axis_cell -q

.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan.py --repo-root .
.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py --repo-root .
```

The plan remains dry-run only and does not launch provider work. The
architecture lock remains false until the paired exact CPU/CUDA effect-curve
cells are harvested and validated.
