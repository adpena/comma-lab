# L5-v2 Operator Briefing Active-Claim Suppression - 2026-05-16

## Context

`tools/operator_briefing.py --json` already exposed live dispatch-claim state
and L5-v2 PacketIR exact-eval target rows, but those two surfaces were
independent. A briefing could show `dispatch_claim_summary.active_count > 0`
while still listing PR106 PacketIR next exact-eval targets, which created an
operator-facing ambiguity during active or stale dispatch custody.

## Fix

- L5-v2 frontier readiness now receives the same live dispatch-claim summary
  used by the composite operator briefing.
- If active dispatch claims exist, PR106 PacketIR target rows are suppressed
  fail-closed.
- The L5-v2 payload records:
  - `active_dispatch_claim_count`
  - `dispatch_claim_gate_blocked`
  - blocker `blocked_active_dispatch_claims_present:<count>`
- Matrix SHA mismatch suppression remains independent and still fail-closed.

## Evidence

- Focused unit coverage added for active-claim suppression against a fresh
  PacketIR matrix fixture.
- Composite JSON coverage now accepts both live states:
  - no active claims: target rows remain visible as fail-fast, non-promotional
    rows;
  - active claims: target rows are suppressed and the active-claim blocker is
    present.

## Scope

This is a visibility/custody hardening change only. It does not change the
PacketIR candidate matrix, score claims, promotion eligibility, or any provider
actuator. The briefing remains read-only and every target remains
`score_claim=false`, `promotion_eligible=false`, and
`ready_for_exact_eval_dispatch=false`.
