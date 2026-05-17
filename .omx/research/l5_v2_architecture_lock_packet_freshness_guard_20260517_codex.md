# L5 v2 architecture-lock packet freshness guard - 2026-05-17

## Finding

The committed architecture-lock packet had drifted behind the live
`l5_v2_tt5l_campaign_readiness()` state. The stale packet still routed the
next action to `populate_and_evaluate_c1_z5_tt5l_probe_observations`, even
though the current probe-observation intake artifact is present and valid for
measurement planning.

The live state now routes to:

`materialize_l5_v2_paired_probe_measurements`

This is the non-retread L5-v2 action: build the lattice measurement schedule,
build the paired CPU/CUDA measurement dispatch plan, and fill each work unit
with byte-closed archive/runtime custody before any operator-executed dispatch.

## Changes

- Regenerated `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.{json,md}`.
- Regenerated `.omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.{json,md}` from the current schedule artifact.
- Added `test_l5_v2_architecture_lock_packet_artifact_tracks_live_payload` so the committed architecture-lock packet must match the live readiness payload for the TT5L next action, side-info effect-curve status, materialized paired work-unit status, and blockers.

## Current L5-v2 state

- `architecture_lock_allowed=false`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- Next action: `materialize_l5_v2_paired_probe_measurements`
- Dispatch plan: planning-only, `work_unit_count=3`, `ready_work_unit_count=0`
- Materialized TT5L paired work unit remains blocked by `l5_v2_tt5l_materialized_paired_work_unit_tt5l_sideinfo_all_zero`

## Verification

```bash
.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py
.venv/bin/python tools/build_l5_v2_lattice_measurement_schedule.py \
  --probe-intake-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json \
  --sideinfo-effect-curve-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json
.venv/bin/python tools/build_l5_v2_paired_measurement_dispatch_plan.py
.venv/bin/python -m pytest -q \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_architecture_lock_packet_artifact_tracks_live_payload \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_architecture_lock_packet_cli_writes_no_lock_packet \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_asymptotic_candidate_surface_artifact_tracks_live_payload
```

Result: `3 passed`.

`git diff --check` clean.
