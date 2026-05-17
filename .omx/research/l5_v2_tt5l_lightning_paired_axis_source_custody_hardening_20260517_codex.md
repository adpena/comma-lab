# L5 v2 TT5L paired-axis source and provider custody hardening

Date: 2026-05-17

Scope: Time-Traveler L5 v2 staircase readiness, Lightning paired-axis dry-run
plan, Modal provider-blocker state, paired runtime content custody,
architecture-lock packet, and no-signal-loss control-plane reporting.

## Findings

The TT5L side-info effect-curve Lightning paired-axis plan is a valid
historical dry-run artifact, but it records `source_commit`
`649f44f7179216cc372fd18ad64f8a55b3d0aad0`, while the code paths that build
and execute the paired-axis plan had changed on `main`.

Before this hardening pass, the L5 v2 architecture-lock packet surfaced the
paired CPU/CUDA dry-run plan but did not expose whether source-relevant paths
changed since the dry-run source commit. That was an execution-custody gap: a
later operator could incorrectly treat an old dry-run plan as current source
custody.

Read-only adversarial review also found that a stale Modal provider-blocker
artifact could fall through to
`review_and_execute_l5_v2_tt5l_materialized_paired_measurement`. The blocker
artifact existed but was invalid because its archive SHA did not match the
current materialized work unit. The old branch predicate only handled
`provider_blocker_status.active=true`; invalid blocker artifacts were neither
refreshed nor retired before the execute command was surfaced.

A second paired-custody gap was present in the materialized work-unit layer:
CPU and CUDA runtime content SHA-256 fields were individually validated, but
the layer did not require both axes to share the same runtime content tree.

## Change

- `src/tac/optimization/l5_staircase_v2.py` now records source-custody fields
  in `sideinfo_effect_curve_lightning_paired_axis_plan_status`:
  `source_commit`, `current_head_commit`, `source_commit_matches_head`,
  `source_commit_is_ancestor`, `source_relevant_paths`,
  `source_relevant_diff_paths`, `source_relevant_paths_match`, and
  `source_custody_current_for_execution`.
- Exact HEAD drift alone is not a blocker. Committing a regenerated plan
  necessarily advances HEAD, so execution custody is based on source-relevant
  path drift, not raw equality with HEAD.
- Relevant source drift is an execution blocker:
  `l5_v2_tt5l_lightning_paired_axis_plan_source_relevant_paths_changed`.
- The paired-axis dry-run plan was regenerated from current source, so the
  architecture-lock packet now reports `source_relevant_paths_match=true` and
  `source_custody_current_for_execution=true`.
- Invalid/stale Modal provider-blocker artifacts now force next action
  `refresh_or_retire_l5_v2_tt5l_modal_provider_blocker` and refuse to surface an
  execute command.
- Materialized paired work units now fail closed on
  `l5_v2_tt5l_materialized_paired_work_unit_runtime_content_axis_mismatch`.
- The historical dry-run artifact remains `artifact_valid=true` when its cells,
  schemas, axis roles, and non-authority flags are internally valid. Provider
  execution remains blocked by explicit non-dry-run prerequisites.
- The architecture-lock Markdown packet now prints the source-custody fields in
  the `Lightning Paired-Axis Dry-Run Plan` section.

## Current Status

- Paired-axis plan artifact: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
- Plan cells: 10/10 (`zero`, `random_lsb`, `shuffled`, `trained`, `ablated`
  times `[contest-CPU]` and `[contest-CUDA]`)
- Plan artifact validity: true
- Source-relevant paths match: true
- Source custody current for execution: true
- Next action: `refresh_or_retire_l5_v2_tt5l_modal_provider_blocker`
- Execution readiness: false
- Score claim: false
- Promotion eligible: false

## Reactivation Criteria

Before any non-dry-run Lightning submit for this effect curve:

1. Refresh or retire the stale Modal provider-blocker artifact for the current
   materialized archive SHA.
2. Confirm `source_custody_current_for_execution=true` in the
   architecture-lock packet.
3. Configure Lightning identity/workspace values.
4. Stage a fresh source manifest to the Lightning workspace.
5. Claim each per-axis lane before non-dry-run submit.
6. Harvest all ten `[contest-CPU]` and `[contest-CUDA]` cells before building
   the side-info effect-curve claim artifact.

## Verification

- `python -m py_compile src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
- `python -m pytest src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_readiness_surfaces_current_lightning_paired_axis_plan src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_allows_head_only_drift src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_blocks_relevant_source_drift src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_stale_modal_blocker_blocks_execute_command src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_materialized_work_unit_rejects_runtime_content_axis_mismatch -q`
- `python -m pytest src/tac/tests/test_l5_staircase_v2.py -q` (`118 passed`)
- `python -m ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
- `python tools/build_l5_v2_architecture_lock_packet.py`
