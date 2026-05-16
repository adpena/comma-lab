# L5 v2 Cathedral completed-action hardening

- schema: `l5_v2_cathedral_completed_action_hardening_v1`
- created_at_utc: `2026-05-16T22:21:45Z`
- surface: `tools/cathedral_autopilot.py`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

The L5 v2 asymptotic candidate payload already distinguishes the original
recommended action from the effective current action. When an L1 scaffold is
present, `l5_v2_asymptotic_pursuit_candidates()` marks:

- `recommended_next_action_status=completed_or_superseded`
- `effective_recommended_next_action_id=completed_or_superseded:<old_id>`
- `recommended_next_action_completed_or_superseded=true`
- `ready_for_recommended_next_action=false`

Cathedral was still reading `recommended_next_action_id` directly for the
validation queue. That could revive already-landed Z6/Rudin/Tishby scaffold
work as if it were still active work, creating duplicate action and dispatch
pressure.

## Fix

`tools/cathedral_autopilot.py` now consumes the effective/status fields for
L5 v2 asymptotic candidate rows:

- `validation_status` uses `effective_recommended_next_action_id` when present;
- `recommended_next_action` uses `effective_recommended_next_action` when
  present;
- completed/superseded actions force
  `ready_for_recommended_next_action=false` and `ready_for_l1_build=false`;
- completed/superseded actions add explicit dispatch blockers instead of
  appearing pending.

## Regression

`src/tac/tests/test_cathedral_autopilot.py::test_l5_v2_validation_queue_uses_effective_completed_actions`
builds a mocked completed Z6 L1 candidate and verifies Cathedral surfaces the
completed effective action, not the stale original action, and refuses both
recommended-next-action and L1-build readiness.
