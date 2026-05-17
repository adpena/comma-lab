# Exact Dispatch Claim Policy Follow-Up - 2026-05-17

## Scope

Preserves and validates a partner WIP change in the exact-dispatch authority
path. This is dispatch-readiness hardening only. It does not launch provider
work, claim score movement, promote a candidate, or change archive bytes.

## Finding

The exact-ready promotion path has two distinct claim-ledger phases:

- preclaim audit: an active same-lane claim is a conflict before a new dispatch
  is created;
- postclaim submit audit: a matching active claim is required after the
  launcher has claimed the lane and before non-dry-run provider submit.

Using one "active claim is always a conflict" rule for both phases blocks the
postclaim submit audit from proving that the required claim exists.

## Landed Behavior

`src/tac/optimizer/exact_dispatch_authority.py` now accepts an explicit
`claim_policy`:

- `preclaim_conflict_check` keeps the existing fail-closed same-lane active
  claim conflict behavior.
- `require_active_claim` suppresses the conflict interpretation and instead
  requires a matching active claim by lane, optional platform, and optional
  job id.

`src/tac/optimizer/exact_readiness.py` exposes
`ignore_active_claim_conflicts` only for this authority-controlled
postclaim path.

## Verified Invariants

Focused tests verify:

- preclaim policy still blocks active same-lane claims;
- `require_active_claim` blocks when no matching active claim exists;
- `require_active_claim` accepts a matching active Lightning claim;
- terminal closeout rows do not satisfy the active-claim requirement;
- the existing exact-ready active-claim conflict test still passes.

## Authority

- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false
- ready_for_provider_dispatch: false
- dispatch_attempted: false
- provider_spend_attempted: false
