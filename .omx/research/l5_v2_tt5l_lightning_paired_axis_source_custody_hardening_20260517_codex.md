# L5 v2 TT5L Lightning paired-axis source custody hardening

Date: 2026-05-17

Scope: Time-Traveler L5 v2 staircase readiness, Lightning paired-axis dry-run
plan, architecture-lock packet, and no-signal-loss control-plane reporting.

## Finding

The TT5L side-info effect-curve Lightning paired-axis plan is a valid
historical dry-run artifact, but it records `source_commit`
`649f44f7179216cc372fd18ad64f8a55b3d0aad0`, while the current `main` HEAD
after the readiness bridge is `c7aa814161dcb3c5ba2cf53f7029e3861cdd0e7d`.

Before this hardening pass, the L5 v2 architecture-lock packet surfaced the
paired CPU/CUDA dry-run plan but did not expose whether the dry-run source
commit matched current HEAD. That was an execution-custody gap: a later
operator could incorrectly treat an old dry-run plan as current source custody.

## Change

- `src/tac/optimization/l5_staircase_v2.py` now records
  `source_commit`, `current_head_commit`, and `source_commit_matches_head` in
  `sideinfo_effect_curve_lightning_paired_axis_plan_status`.
- A source mismatch is an execution blocker:
  `l5_v2_tt5l_lightning_paired_axis_plan_source_commit_not_current_head`.
- The historical dry-run artifact remains `artifact_valid=true` when its cells,
  schemas, axis roles, and non-authority flags are internally valid. The source
  mismatch blocks execution readiness only.
- The architecture-lock Markdown packet now prints the source-custody fields in
  the `Lightning Paired-Axis Dry-Run Plan` section.

## Current Status

- Paired-axis plan artifact: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
- Plan cells: 10/10 (`zero`, `random_lsb`, `shuffled`, `trained`, `ablated`
  times `[contest-CPU]` and `[contest-CUDA]`)
- Plan artifact validity: true
- Source commit matches current HEAD: false
- Execution readiness: false
- Score claim: false
- Promotion eligible: false

## Reactivation Criteria

Before any non-dry-run Lightning submit for this effect curve:

1. Regenerate the paired-axis dry-run plan from current `main`.
2. Confirm `source_commit_matches_head=true` in the architecture-lock packet.
3. Configure Lightning identity/workspace values.
4. Stage a fresh source manifest to the Lightning workspace.
5. Claim each per-axis lane before non-dry-run submit.
6. Harvest all ten `[contest-CPU]` and `[contest-CUDA]` cells before building
   the side-info effect-curve claim artifact.

## Verification

- `python -m py_compile src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
- `python -m pytest src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_readiness_surfaces_current_lightning_paired_axis_plan src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_surfaces_stale_source_commit -q`
- `python -m pytest src/tac/tests/test_l5_staircase_v2.py -q` (`115 passed`)
- `python -m ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
- `python tools/build_l5_v2_architecture_lock_packet.py`
