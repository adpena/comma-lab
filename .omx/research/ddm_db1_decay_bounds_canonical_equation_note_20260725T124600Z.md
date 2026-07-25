---
schema: ddm_db1_canonical_equation_disposition.v1
date_utc: 2026-07-25T12:46:00Z
research_only: true
main_landing_review_required: true
---

# DB1 canonical-equation disposition

## Registered

`ddm_sn1_margin_mass_duplicate_budget_bounds_v1`

For a fixed predicted-boundary atlas with ordered-incidence CDF `I(delta)`, measured unique
boundary total `B`, and complete duplicate budget `D=I(infinity)-B`:

`max(0, I(delta)-D) <= N(delta) <= min(I(delta), B)`.

The callable is
`tac.analysis.ddm_db1_decay_bounds:unique_count_bounds`. The n600 empirical anchor is
`I(infinity)=2,569,387`, `B=2,551,382`, and `D=18,005`, where both bounds collapse to `B` at
infinity.

Included domain: hash-bound SN1 ordered predicted-boundary margins, AT1 exact linear-head
distance, and fixed-atlas threshold queries.

Excluded domain: target-error-conditioned correctability, non-boundary pixels, live boundary
replenishment, steps-to-target, terminal descent, contest scores, and promotion.

Two append-only `registered` events exist. The first preserves custody of the preflight receipt
SHA `42f165fc...`; the latest points to the path-normalized final receipt SHA `f36fccd5...`.
Latest-event reduction is the current canonical state. MAIN must merge-review both historical
rows and preserve append-only ordering.

## Not registered

Neither `c+a*n^-p` nor `c+a*exp(-k*n)` is registered as a canonical descent law. They are
conditional fits to one finite V19C accepted proposal ordering. The scorer-recursive live
transport kernel, selection process, boundary replenishment, and E7 residual entropy are absent,
so a transferable law would be false authority.

## Consumers

- DB1 analyzer and FEED-603-db1 consume the exact fixed-atlas inequality.
- E3 may consume `N(delta)` only as advisory collateral-sentinel mass after MAIN review.
- G2/E7 consume the explicit exclusions and stay blocked pending the same-parent trace.
- No bit-allocation price, dispatch, score, promotion, or pointer mutation is emitted.
