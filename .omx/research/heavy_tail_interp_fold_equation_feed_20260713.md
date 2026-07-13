# Canonical equation FEED — control-driving interpolator tail regularization

**Date:** 2026-07-13  
**Proposed equation id:** `control_driving_interpolator_tail_regularization_v1`  
**Status:** `PROPOSED`; shared registry append `DEFERRED_MAIN`; `research_only=true`  
**Authority:** MEANS only; no score or pointer authority  
**Source:** Zhu and Lu, arXiv:2607.09547, plus the Pact control-regret transfer derived in
`.omx/research/heavy_tail_interp_fold_20260713.md`.

## Source theorem edge

For Zhu-Lu ridgeless risk under their proportional random-design assumptions,

```text
lim [1/(n log n)] log P(R_n > r_star + delta) = -|1-gamma|/2.
```

For every fixed `lambda>0`, Gaussian ridge has an `n^2`-speed LDP; general normalized/log-Sobolev
entries have

```text
P(R_n,lambda > r_lambda,star + delta) <= exp(-c_lambda,delta n^2).
```

These rates are **SOURCE** and are not asserted for Pact's finite clustered designs.

## Derived Pact law

Let:

- `D` be the fit dataset;
- `g_hat_lambda(D)` be an estimated marginal-Delta-S field;
- `pi(g_hat_lambda, xi)` be the induced control action for heldout state/regime `xi`;
- `xi` index a heldout costate/regime;
- `S_next(a; xi)` be the receiver-relevant next-state objective under action `a`; and
- `a_star(xi)` be the oracle/reference action for the diagnostic.

Define nonnegative downstream control regret

```text
Z_lambda(D, xi)
  := [S_next(pi(g_hat_lambda(D), xi); xi) - S_next(a_star(xi); xi)]_+.
```

Then the tail-aware fixed-positive regularization law is

```text
lambda_ctrl(alpha, epsilon_mean)
  in argmin_(lambda in Lambda_fixed_positive) CVaR_alpha[Z_lambda]

subject to
  E[Z_lambda] <= min_(lambda' in Lambda_fixed_positive) E[Z_lambda'] + epsilon_mean,
  custody(lambda) = PASS,
  walk_forward(lambda) = PASS,
  action_stability(lambda) = PASS.
```

For a higher-is-better reliability statistic such as retained mass `M_lambda`, use the shortfall

```text
L_lambda := [m_bar - M_lambda]_+
```

inside the same law. Equivalently, report lower quantiles and lower-tail conditional means of
`M_lambda`.

## Labels and provenance

- Zhu-Lu probability rates: **SOURCE**.
- Mapping from estimator error to downstream control regret: **DERIVED**.
- Preference for fixed-positive ridge over a ridgeless control solve: **DERIVED**, conditional on
  the mean/custody/action gates.
- `alpha` and `epsilon_mean`: **UNSET**. They must be derived from an accepted control-error budget;
  this FEED assigns no guessed defaults.
- Literal `n^2` or `n log n` rates for Pact: **NO-GO-regime-mismatch**, `verdict_scope =
  FORMULATION x LITERAL-RATE-TRANSFER`.

## Required empirical anchors before registry adoption

1. A preregistered ridge/MP comparison on identical cached data and split custody.
2. Both average and tail summaries; no mean-only adoption.
3. Across-refit splits/seeds for estimator-level reliability, distinct from within-fit example tails.
4. Spectral rank/cutoff/minimum-eigenvalue context for every inverse-based head.
5. For costate control, past-only downstream action-regret replay with persistence retained as the
   fail-closed baseline.
6. Deterministic NumPy-fp32 inference/decision authority, with parity checks for any accelerated
   implementation.
7. Every negative carries `verdict_scope`; no family kill from one fixed replay.

## Consumer routing

- `pre_se`: MP remains an optimization certificate; ridge becomes the reliability candidate.
- `A_ridge_solve`: positive ridge remains the solve-family default; lambda selection moves from
  mean-only fit toward downstream tail regret.
- `#433`: prior-mean shrinkage is a candidate mechanism, not tail-confirmed by the current record.
- confound gates: consume tail fields conceptually; no live mutation from this FEED.

**Pointer delta:** `NONE`.  
**Shared equation-registry append:** `DEFERRED_MAIN`.
