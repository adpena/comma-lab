# Frontier Feedback Operator/Preflight/Eureka Wiring

- timestamp_utc: 2026-05-25T14:16:29Z
- agent: codex
- scope: frontier-rate feedback cycle, operator briefing, preflight no-orphan guard, local CPU eureka acquisition hints
- authority: planning/local advisory only

## What Changed

The frontier feedback cycle is now surfaced as a first-class operator briefing section and dispatch-readiness phase. The briefing scans `.omx/research` and `experiments/results` for `frontier_rate_attack_feedback_cycle.json` and `feedback_refresh_report.json`, reports the next bounded local cycle command, and fails closed on any truthy score/promotion/rank/dispatch authority.

`tools/all_lanes_preflight.py` now requires the operator briefing JSON to carry `frontier_feedback_cycle`, mirrors it against `dispatch_readiness.phase_6d_frontier_feedback_cycle`, rejects missing cycle tooling, rejects authority leaks, and rejects cycle error rows. This turns the cycle from a manually remembered tool into a protected no-orphan surface.

Local macOS-CPU eureka drift JSONs are now canonicalized by `frontier_rate_attack_feedback.py` as planning-only acquisition signal. Near-frontier observe-only drop-two rows become an explicit `dqs1_expand_beyond_drop_two_near_boundary` planner hint that recommends expanding beyond drop two into learned multi-drop, drop-many beam search, within-set masked/feather probes, master-gradient-constrained low-sensitivity drops, and inverse-scorer null-direction variants. The hint is injected into follow-up queue experiment metadata while preserving false authority.

The preflight xray allowlist now includes `tools/master_gradient_xray.py`, matching the operator briefing xray toolkit and closing that orphan-diagnostic mismatch.

## False-Authority Boundary

- score_claim: false
- score_claim_valid: false
- promotion_eligible: false
- rank_or_kill_eligible: false
- ready_for_exact_eval_dispatch: false
- dispatch_attempted: false
- gpu_launched: false

The eureka planner hint is not a win claim. It records that drop-two near-misses are useful evidence that the acquisition policy should widen beyond drop-two, not that drop-two itself is optimal or dispatch-ready.

## Verification

- `ruff` on touched scheduler/tool/test files: passed
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_all_lanes_operator_briefing_gate.py -q`: 36 passed
- fresh combined `pytest src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_all_lanes_operator_briefing_gate.py -q`: passed, 79 tests
- `tools/operator_briefing.py --json --skip-pareto --skip-dashboard --skip-reconciler --skip-provider-readiness`: passed; `frontier_feedback_cycle.status=READY_LOCAL_EXECUTION`, `phase_6d=READY_LOCAL_EXECUTION`, authority fields explicitly false including `gpu_launched=false`
- live `_run_operator_briefing_dispatch_gate()`: frontier feedback and xray checks pass; remaining blocker is unrelated live L5 state `l5_v2_packetir_matrix_artifact_sha_mismatch`
- plan-only proof cycle: `.omx/research/codex_frontier_feedback_eureka_cycle_20260525T141629Z/frontier_rate_attack_feedback_cycle.json`
  - eureka signal count: 2
  - planner hint count: 1
  - hint id: `dqs1_expand_beyond_drop_two_near_boundary`
  - selected follow-up candidates: 4 drop-two candidates from the existing action surface
  - follow-up execution: false
  - all score/promotion/rank/dispatch authority fields: false

## Residual Gap

The L5 PacketIR matrix SHA mismatch remains a live repository blocker outside this scoped landing. It should be repaired in a separate L5 PacketIR custody turn rather than absorbed into frontier feedback/eureka wiring.
