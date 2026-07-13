# Standalone DAG FEED — heavy-tail interpolator reliability

**Date:** 2026-07-13  
**Node:** `FEED-heavy-tail-interpolator-reliability-20260713`  
**Status:** `THEORY-FOLD-COMPLETE`; `research_only=true`; shared-DAG append `DEFERRED_MAIN`  
**Verdict scope:** `MEANS x INTERPOLATOR-RELIABILITY`; no archive/score/pointer authority  
**Pointer delta:** `NONE`

## Source theorem node

```text
Zhu-Lu random linear model
  -> iid normalized continuous light-tailed design
  -> p/n -> gamma != 1
  -> ridgeless inverse trace has one-hard-edge-eigenvalue right tail
  -> fixed-dimension polynomial tail
  -> proportional right tail: n log n speed
  -> fixed-positive ridge removes hard-edge singularity
  -> Gaussian ridge LDP / general-entry upper suppression: n^2 speed
```

## Mandatory regime-audit edge

```text
paper theorem
  -> check random object (fitted-design tail vs within-fit example tail)
  -> check iid / covariance / noise assumptions
  -> check proportional p/n regime and gamma != 1
  -> check fixed lambda > 0
  -> check untruncated inverse vs spectral cutoff
  -> check loss map into retained mass or control regret
  -> result for audited Pact surfaces: DIRECT-APPLICATION = N
  -> literal rate claim: NO-GO-regime-mismatch
  -> reliability mechanism: INFERRED-SUGGESTIVE
```

## `pre_se` edge

```text
sealed n600 campaign
  -> 480 fit states / 120 heldout states
  -> d in {188, 332}; 20 unequal class-pair heads
  -> 40/40 rank-truncated MP certificates; no landed ridge receipt
  -> heldout aggregate retained mass {0.202330, 0.093147}
  -> cached deterministic NumPy-fp32 heldout lower-q10 {0.164945, 0.046867}
  -> fixed-replay failure remains
  -> MP retained as optimization/capacity certificate
  -> recommendation: add preregistered fixed-positive ridge reliability rung
  -> report within-fit lower tail AND across-refit lower tail
  -> terminal verdict: REPORT-tail-quantile + ADOPT-ridge-default
```

The cached tail rows are `n=120` heldout diagnostics inside `n600` custody, not an `n600` tail
sample and not a Zhu-Lu rate estimate.

## Costate-organ edge

```text
A_ridge_solve
  -> p=17; n_train=2..8 sequential intervals; fixed ridge=1e-2
  -> current recommendation basis: no passing record, default solve
  -> one-vehicle mean WF loses to persistence
  -> positive ridge remains safer than a ridgeless inverse inside solve family
  -> derive downstream control regret Z_lambda
  -> choose lambda by tail regret subject to mean/custody/action gates
  -> operational inference/decision authority: deterministic NumPy-fp32 reference
  -> persistence remains fail-closed when meta-lambda refuses model
  -> terminal verdict: ADOPT-ridge-default + TUNE-lambda-for-tail
```

## `#433` edge

```text
cached seven-fold walk-forward errors
  -> P prior-mean aniso mean vs A ridge: -18.4595%
  -> Q prior-mean iso mean vs A ridge: -21.4183%
  -> P/Q maximum and worst-two errors do not beat A
  -> shrinkage mechanism tail-consistent
  -> empirical tail suppression NOT confirmed
  -> cached-only lambda/tail Pareto child analysis available
  -> no adoption until transient-rich multi-record accrual
  -> terminal verdict: TUNE-lambda-for-tail
```

Req-R for the scoped tail negative is satisfied by two prior-mean formulations (`P`, `Q`) and two
tail summaries (maximum and worst-two mean) on the same fixed instance. The scope remains
`INSTANCE x ONE-VEHICLE x NINE-INTERVAL-TRAJECTORY`.

## Confound edge

```text
rare estimator error
  -> possible disproportionate verdict/control corruption
  -> L1 alarm / median-freeze / spike guard / satisficing / L3 clearance
  -> require tail + spectral-stability fields for any interpolator consumer
  -> no claim that existing confounds satisfy Zhu-Lu random-design assumptions
  -> terminal verdict: REINFORCE-confound-story
```

## Triality and routing

- Memo: `.omx/research/heavy_tail_interp_fold_20260713.md`.
- Equation: `.omx/research/heavy_tail_interp_fold_equation_feed_20260713.md`.
- DSL: `DEFERRED_MAIN`; no live setting or invented flag in this theory fold.
- Shared DAG append: `DEFERRED_MAIN` because the canonical DAG is sibling-held/dirty.
- Cathedral/autopilot: `REFUSE` training, paid dispatch, live-run mutation, or pointer action.
- Bit allocator/archive: non-binding; no bytes emitted.

Only a future typed consumer with provenance-owned tail thresholds may turn this FEED into an
actuator. This standalone file is the durable handoff; no hot sibling surface was edited.
