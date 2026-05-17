# L5 v2 probe intake pair identity guard - 2026-05-17

## Summary

This landing closes the remaining L5 v2 probe-intake identity gap surfaced by
the read-only adversarial review of the TT5L readiness path. The previous
archive-SHA grouping prevented obvious archive mixing, but it did not carry
`pair_group_id` or `run_id` through the observation model. That left a false
authority path where a CPU row and CUDA row with the same archive hash could be
accepted as a paired exact observation even when they came from different
materialized pair groups or run identities.

No score claim, promotion claim, provider dispatch, or architecture-lock claim
is made here.

## Fix

- `L5V2ProbeObservation` now records `pair_group_id` and `run_id`.
- Probe intake extracts pair identity from materialized TT5L work-unit plans,
  direct payload fields, custody/provenance fields, or explicit
  `--pair-group-id` / `--run-id` command flags.
- TT5L axis evidence is grouped by archive SHA plus `pair_group_id`/`run_id`
  when either identity field is available.
- The disambiguator now rejects mixed CPU/CUDA pair identities with
  `l5_v2_probe_axis_pair_group_mismatch` or
  `l5_v2_probe_axis_run_id_mismatch`.
- The regenerated probe gate and architecture-lock packet include the new
  identity fields and remain fail-closed.

## Validation

- `.venv/bin/python -m ruff check src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/optimization/l5_v2_probe_intake.py src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_intake.py`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_probe_intake.py src/tac/tests/test_l5_v2_probe_disambiguator.py -q` -> 45 passed
- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_readiness_surfaces_current_lightning_paired_axis_plan src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_allows_head_only_drift src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_blocks_relevant_source_drift -q` -> 3 passed
- `.venv/bin/python tools/audit_l5_v2_probe_observations.py --output-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json --output-md .omx/research/l5_v2_probe_observation_intake_20260516_codex.md --probe-gate-out .omx/research/l5_v2_probe_gate_artifact_20260516_codex.json` -> expected fail-closed `architecture_lock_allowed=false`
- `.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py --repo-root . --output-json .omx/research/l5_v2_architecture_lock_packet_20260516_codex.json --output-md .omx/research/l5_v2_architecture_lock_packet_20260516_codex.md` -> `architecture_lock_allowed=false`
- `git diff --check`

## Current State

L5 v2 still correctly blocks architecture lock on missing valid gate evidence
and missing paired CPU/CUDA side-info effect curve. This patch hardens the
custody path; it does not change frontier score status.
