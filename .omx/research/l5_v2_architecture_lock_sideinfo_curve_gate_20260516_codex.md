# L5 v2 architecture lock side-info curve gate

Date: 2026-05-16

## Context

Read-only adversarial review flagged a false-authority path in the TT5L-first
L5-v2 staircase: `architecture_lock_allowed` could become true after valid probe
and paired-axis plan evidence even when the paired CPU/CUDA side-info causal
effect curve was still missing. That would let a consumption proof stand in for
usefulness evidence.

## Change

- Promoted `L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH` and
  `validate_l5_v2_sideinfo_effect_curve()` from the measurement-schedule module
  so readiness uses the same curve contract as the lattice scheduler.
- Added `tt5l_sideinfo_effect_curve_status()` to the L5-v2 staircase surface.
- Changed TT5L readiness so `architecture_lock_allowed` and
  `first_anchor_timing_smoke_allowed` require a valid paired CPU/CUDA side-info
  effect-curve artifact, not just permission to measure one.
- Routed the next TT5L action to `measure_tt5l_sideinfo_effect_curve` before
  timing-smoke custody when probe and paired-axis gates are present but the
  curve is missing.
- Surfaced both `tt5l_sideinfo_effect_curve_artifact_valid` and
  `tt5l_architecture_lock_allowed` in `tools/operator_briefing.py`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_staircase_v2.py -q`
  - `102 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/l5_v2_measurement_schedule.py src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
  - clean
- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_remote_lane_time_traveler_l5_script.py -q`
  - `104 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py src/tac/tests/test_all_lanes_operator_briefing_gate.py -q`
  - `43 passed`
- `.venv/bin/python tools/lane_maturity.py validate`
  - `767 lane(s) validated cleanly`

## Authority

This is a planning and false-authority hardening patch only.

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The current live TT5L state remains non-promotional until paired probe
observations, paired CPU/CUDA plan custody, side-info effect curve, first-anchor
timing smoke, and exact/diagnostic anchor packet all land with custody.
