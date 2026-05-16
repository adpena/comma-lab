# L5 v2 Dykstra Move-Level Feasibility Split

Date: 2026-05-16
Author: Codex
Scope: TT5L / L5 v2 staircase readiness

## Verdict

`score_claim=false`; `promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`.

The TT5L Dykstra artifact is now treated as scalar score-axis sanity only. It
does not unlock side-info effect curves or first-anchor timing smoke unless a
separate move-level feasibility artifact is also valid.

## Failure Class

`score_axis_sanity_misclassified_as_move_level_feasibility`

The Dykstra checker explicitly emits a score-axis projection with
`move_level_constraint_proof=false`. Previous L5 readiness nevertheless used
`dykstra_valid` directly as one of the conditions for
`sideinfo_effect_curve_allowed`. That made a scalar planning sanity check act
like a proof that the TT5L five-move constraint intersection was feasible.

## Landed Fix

Added a separate move-level feasibility status:

- artifact: `.omx/state/tt5l_move_level_feasibility.json`
- schema: `tt5l_move_level_feasibility_v1`
- predicate: `tt5l_move_level_constraint_feasibility_v1`
- requires `predicate_passed=true`
- requires `move_level_constraint_proof=true`
- requires exact TT5L constraint-set ids and count
- requires finite `residual_max <= residual_tolerance`
- requires `score_claim=false`, `promotion_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`

`sideinfo_effect_curve_allowed` now requires:

1. Dykstra score-axis sanity artifact valid;
2. move-level feasibility artifact valid;
3. side-info consumption gate evidence valid.

`first_anchor_timing_smoke_allowed` inherits the same stricter condition plus
probe and paired-axis plan validity.

## Verification

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_l5_staircase_v2.py

.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q

.venv/bin/python -m pytest \
  src/tac/tests/test_l5_v2_measurement_schedule.py \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py \
  src/tac/tests/test_cathedral_autopilot.py -q
```

Observed:

- `ruff`: all checks passed
- `test_l5_staircase_v2.py`: `88 passed in 0.56s`
- related schedule/operator/Cathedral tests: `67 passed in 0.52s`

## Next Work

Build the producer for the move-level feasibility artifact. Until that lands,
score-axis Dykstra sanity remains useful only as a cargo-cult unwind guard, not
as architecture-lock or timing-smoke authority.
