# L5 v2 composition campaign-id canonicalization

- date: 2026-05-16
- scope: TT5L substrate-composition/autopilot campaign wiring
- code: `src/tac/optimization/substrate_composition_matrix.py`
- tests: `src/tac/tests/test_autopilot_dispatch_ranking.py`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

The TT5L substrate-composition row still emitted
`campaign_time_traveler_l5_autonomy_20260516`, while the canonical L5 v2
readiness and Cathedral/autopilot surfaces use
`campaign_time_traveler_l5_v2_staircase_20260516`. That split can fragment
operator state, notes, and downstream claim grouping even when the lane id is
correct.

## Change

The TT5L composition row now imports and uses `tac.optimization.l5_staircase_v2.CAMPAIGN_ID`.
The autopilot dispatch-ranking regression asserts the canonical campaign id
flows through `campaign_metadata` and `composition_notes`, which are the fields
consumed by the Cathedral autonomous loop.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_autopilot_dispatch_ranking.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py src/tac/tests/test_build_composition_ranking_json.py -q`
- `.venv/bin/python -m ruff check src/tac/optimization/substrate_composition_matrix.py src/tac/tests/test_autopilot_dispatch_ranking.py`
