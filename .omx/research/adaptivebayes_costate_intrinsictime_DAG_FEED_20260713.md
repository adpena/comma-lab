# Standalone DAG FEED — Adaptive Bayes costate commit / curriculum shadow

**Date:** 2026-07-13 UTC  
**Status:** `DESIGN`, `research_only=true`, `CANONICAL_DAG_APPEND_DEFERRED_MAIN`  
**Reason for deferral:** canonical DAG/equation surfaces are modified by live siblings  
**Pointer delta:** `UNMOVED`

## `FEED-ADAPTIVEBAYES-COSTATE-COMMIT`

```text
fixed costate decision epoch + sacred common checkpoint
  -> fixed eligible expert registry
  -> every expert emits non-mutating advice
  -> common-horizon receiver-realized full loss vector
       OR explicit propensity/bias/martingale estimator path
  -> typed fixed eta + exact exponential update
  -> exact Q_t / V_t / comparator-KL ledger
  -> one-step identity residual gate
  -> adaptivebayes_fixed_eta_directional_commit_v1
  -> {commit advisory | keep hedging | insufficient custody}
  -> exact n600 walk-forward full-facet comparator
  -> {FEED-costate-controller default-off arm | current #436 fallback}
```

### Blocking edges

- Current #436 is a deterministic regime router, not multiplicative weights.
- The prospective expert set does not yet expose a complete common-state loss vector.
- Selected-expert realized `DeltaS` is bandit feedback unless every alternative is evaluated.
- Loss scale/range and comparator semantics are unregistered.
- No `n600` receiver-realized adoption receipt exists.

### #463 composition edge

```text
#463 TOFU-POV ranks which missing outcome to buy
  -> custody-complete measurement row
  -> Adaptive-Bayes arm accounts for mass/KL/intrinsic-time movement.
```

#463 is the acquisition layer; this feed is the conditional arbitration layer. Do not merge their guarantees.

### Triality

- DAG: this standalone FEED; shared append deferred.
- Equation: `adaptivebayes_fixed_eta_directional_commit_v1`, isolated equation feed.
- DSL: owed typed default-off controller epoch/loss/eta/alpha/estimator schema; no live flags or code edits.

## `FEED-ADAPTIVEBAYES-CURRICULUM-SHADOW`

```text
sacred per-stage checkpoint
  -> fork fixed stay-H and advance-H continuation policies
  -> identical seed/data-order protocol + all checkpoints preserved
  -> #315 topology/nucleus/pose/rate eligibility guards
  -> common receiver-closed n600 loss pair
  -> two-arm fixed-eta exact update
  -> Q_n / V_n / terminal KL receipt
  -> directional mass threshold for advance
  -> #344 forecast as separate advisory sensor
  -> transactional {advance | stay | rollback | insufficient information}
```

### Blocking edges

- Live `{stay,advance}` exposes only one continuation and changes future dynamics.
- Rolling slope and NCDE predictions are sensors, not the missing counterfactual loss.
- Intrinsic time alone has no monotone “stage exhausted” semantics.
- Under unified tau, the old CE -> tau event boundary is dissolved; route only to the live transactional tau-rung/event surface.
- No common-checkpoint `n600` stay/advance branch receipt exists.

### Verdict scope

`NO-GO` for an exact live per-step curriculum clock. `FEED-#315-#344` only for the common-checkpoint shadow branch-selection formulation. This negative does not reject event-triggered curriculum, NCDE forecasting, optimal stopping, or transactional tau-rung controllers as families.

### Triality

- DAG: this standalone FEED; shared append deferred.
- Equation: same directional commit law with `K=2` and comparator `advance`.
- DSL: owed typed branch horizon, checkpoint hashes, common loss semantics, fixed `eta`, `alpha`, and rollback contract; no launch authorized.

## Main-review actions

1. Keep shared canonical registries untouched until live sibling ownership clears.
2. Audit whether any existing receipts provide a literal full expert loss vector from one common checkpoint.
3. If not, route measurement acquisition to #463 and keep this arm blocked.
4. If yes, validate the exact one-step identity on a read-only replay before any controller integration.
5. Preserve `DESIGN (MEANS)` and pointer `UNMOVED` until an exact-through-R `n600` win exists.
