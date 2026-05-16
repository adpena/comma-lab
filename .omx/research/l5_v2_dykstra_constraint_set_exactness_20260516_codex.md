# L5 v2 Dykstra constraint-set exactness

- date: 2026-05-16
- scope: TT5L Dykstra feasibility artifact validation
- code: `src/tac/optimization/l5_staircase_v2.py`
- tests: `src/tac/tests/test_l5_staircase_v2.py`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

The TT5L Dykstra feasibility validator required the canonical five move
constraint ids to be present, but allowed extra ids and did not enforce the
declared `constraint_set_count`. That let a stale or expanded constraint set
masquerade as the exact five-move feasibility projection.

## Change

The validator now requires:

- `constraint_set_ids == TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS`
- `constraint_set_count == len(TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS)`

Missing constraints still emit the existing missing-constraints blocker, while
extra ids or wrong counts emit exactness/count blockers.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q`
