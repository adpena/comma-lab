# L5 v2 first-anchor timing smoke semantics

- date: 2026-05-16
- scope: TT5L-first L5 v2 readiness / operator and Cathedral surfaces
- code:
  - `src/tac/optimization/l5_staircase_v2.py`
  - `tools/all_lanes_preflight.py`
  - `tools/operator_briefing.py`
  - `tools/cathedral_autopilot.py`
- tests:
  - `src/tac/tests/test_l5_staircase_v2.py`
  - `src/tac/tests/test_all_lanes_operator_briefing_gate.py`
  - `src/tac/tests/test_operator_briefing.py`
  - `src/tac/tests/test_cathedral_autopilot.py`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

`first_anchor_timing_smoke_allowed` was true after only the Dykstra feasibility
artifact and side-info consumption proof were valid. That wording implied
first-anchor readiness and could shortcut the C1/Z5/TT5L probe lattice plus
paired CPU/CUDA axis plan.

## Change

The TT5L campaign readiness now splits the concepts:

- `sideinfo_effect_curve_allowed`: Dykstra artifact plus side-info proof are
  valid, so a non-promotional side-info effect curve can proceed.
- `first_anchor_timing_smoke_allowed`: Dykstra, side-info, probe gate, and
  paired-axis plan are all valid.

The operator briefing and Cathedral surfaces expose the distinction, and
`tools/all_lanes_preflight.py` rejects either flag when its prerequisite set is
not satisfied.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/l5_staircase_v2.py tools/all_lanes_preflight.py tools/operator_briefing.py tools/cathedral_autopilot.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_all_lanes_operator_briefing_gate.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_cathedral_autopilot.py`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_all_lanes_operator_briefing_gate.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_cathedral_autopilot.py -q`
