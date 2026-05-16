# L5 v2 Dykstra input-band authority cleanup

- date: 2026-05-16
- scope: TT5L Dykstra feasibility artifact schema and readiness consumer
- tool: `tools/check_substrate_dykstra_feasibility.py`
- consumer: `src/tac/optimization/l5_staircase_v2.py`
- artifact: `.omx/state/dykstra_feasibility_time_traveler_l5.json`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

The TT5L Dykstra artifact correctly carried score-axis-only limitations, but
still serialized the retired input interval under the authority-shaped key
`predicted_band`. That made it too easy for an operator surface or future
automation to cargo-cult the retired additive band back into rank or dispatch
logic.

## Change

The Dykstra tool now emits `tested_score_axis_band` plus
`input_band_role=planning_band_not_score_or_rank_authority` instead of
`predicted_band`. The TT5L L5 v2 readiness consumer rejects Dykstra artifacts
that still contain `predicted_band` or omit the non-authority input-band role.
The committed TT5L Dykstra state artifact was regenerated through the tool with
the new field names.

## Authority

This remains a feasibility-control artifact only. It proves the tested interval
is not instantly inconsistent with the scalar score-axis budgets; it is not a
move-level proof, score claim, promotion signal, rank reward, or exact-eval
dispatch authorization.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_check_substrate_dykstra_feasibility.py src/tac/tests/test_l5_staircase_v2.py -q`
- `.venv/bin/python -m ruff check tools/check_substrate_dykstra_feasibility.py src/tac/tests/test_check_substrate_dykstra_feasibility.py src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
