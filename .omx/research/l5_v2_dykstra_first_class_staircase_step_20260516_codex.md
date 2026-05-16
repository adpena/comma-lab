# L5 v2 Dykstra First-Class Staircase Step

Date: 2026-05-16
Author: Codex
Axis: TT5L / L5 v2 typed staircase ordering
Evidence grade: source-and-test hardening; no score claim

## Finding

The L5 v2 readiness surface correctly requires TT5L Dykstra feasibility before
side-info proof and timing smoke, but the typed staircase still listed temporal
side-info as step 1. That made the machine-readable staircase disagree with the
actual fail-closed control plane.

## Change

`l5_v2_staircase_steps()` now inserts:

`l5v2_01_dykstra_feasibility_polytope`

before temporal side-info. The step points at:

- `tools/check_substrate_dykstra_feasibility.py`
- `.omx/state/dykstra_feasibility_time_traveler_l5.json`

The remaining steps were renumbered:

- `l5v2_02_sideinfo_consumption_proof`
- `l5v2_03_probe_disambiguator`
- `l5v2_04_paired_axis_anchor`
- `l5v2_05_stack_of_stacks_candidate`

This keeps Cathedral/operator consumers aligned with the Dykstra-first TT5L
L5-v2 protocol instead of treating Dykstra as an invisible side condition.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q`
  - `66 passed`
- `ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
  - `All checks passed`

## No Score Claim

This is control-plane hardening only. It does not dispatch, evaluate, promote,
or rank TT5L.
