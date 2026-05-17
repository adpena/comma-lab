# Venn / Predicted Dispatch Risk Composition Guard

Date: 2026-05-17
Owner: codex
research_only: false
score_claim: false
promotion_eligible: false

## Finding

The 2026-05-17 Venn reweighting wire-in adds a per-archive rank adjustment from
`master_gradient_consumers` sidecars. During review of the active WIP, the main
false-authority risk was that this new rank signal could bypass or replace the
older OP-3 `predicted_dispatch_risk` structural refusal hook.

That would be unsafe: a candidate with `predicted_dispatch_risk >= 50` must floor
its effective score delta at `0.0` even if a Venn sidecar reports HIGH
PAIR_INVARIANT byte mass. Venn classification is planning evidence, not a
dispatch-safety override.

## Fix

`tools/cathedral_autopilot_autonomous_loop.py` now documents the intended
composition explicitly:

1. Apply score-axis rank adjustments.
2. Apply `adjust_predicted_delta_for_predicted_dispatch_risk`.
3. Apply `adjust_predicted_delta_for_venn_classification` to the
   already-risk-adjusted delta.

The regression
`test_venn_reweight_does_not_replace_predicted_dispatch_risk_refusal` constructs
a synthetic HIGH PAIR_INVARIANT sidecar and a candidate with
`predicted_dispatch_risk=75.0`. The rank key must remain `0.0`.

## Verification

Commands:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_session_20260517_cli_flag_additions.py \
  src/tac/tests/test_cathedral_autopilot_tier_c_and_composition.py -q
```

Expected result: all tests pass. No score claim, archive promotion, GPU dispatch,
or leaderboard-axis claim is made by this guard.
