# Conditional equation feed — Adaptive Bayes directional commit

**Date:** 2026-07-13 UTC  
**Status:** `DESIGN`, `research_only=true`, `REGISTRATION_DEFERRED_MAIN_REVIEW`  
**Canonical registry claim:** **NONE**; isolated equation handoff only  
**Pointer delta:** `UNMOVED`

## Equation candidate — `adaptivebayes_fixed_eta_directional_commit_v1`

### Domain and hypotheses

- A fixed indexed set of `K` eligible expert or continuation-policy labels.
- At every round, one finite common-scale composite-loss vector `c_t in R^K` exists.
- All loss entries refer to the same pre-round state, horizon, receiver, and authority surface.
- Fixed predictable `eta>0`; exact update
  `p_{t+1,i} proportional to p_{t,i} exp(-eta c_t(i))`.
- Comparator `j` means the realized fixed column `c_1(j),...,c_T(j)`, not an unobserved counterfactual training trajectory.
- If an expert is sampled rather than the mixture loss incurred, the sampling martingale is separately persisted and accounted.

### Law

```text
Q_t
  = eta^-2 log sum_i p_t(i)
      exp(-eta[c_t(i)-<p_t,c_t>]),

V_T = sum_t Q_t,

R_T(j)
  = sum_t (<p_t,c_t>-c_t(j)),

I_t(j)
  = KL(e_j || p_t) = -log p_t(j).
```

The fixed-temperature mixed-coincidence identity gives

```text
R_T(j) - eta V_T
  = [I_1(j)-I_{T+1}(j)]/eta.
```

For preregistered residual mixture-mass tolerance `alpha in (0,1)`, the exact directional commit criterion is

```text
p_{T+1}(j) >= 1-alpha

iff

R_T(j) - eta V_T
  >= [I_1(j)+log(1-alpha)]/eta.
```

For uniform prior `p_1(j)=1/K`:

```text
R_T(j) - eta V_T >= log(K(1-alpha))/eta.
```

### Interpretation

The comparator advantage must pay the exact intrinsic-time bill `eta V_T` and enough initial information debt to concentrate at least `1-alpha` update mass on `j`. This mass is an operational controller quantity, not a calibrated probability that `j` is truly optimal. Intrinsic time alone is not directional and is never a sufficient commit statistic.

### Fail-closed conditions

```text
non-fixed or silently relabeled experts
or selected-arm-only feedback with no valid estimator/propensity
or selection-dependent counterfactual comparator semantics
or missing common-state/common-horizon custody
or non-finite exponential normalizer
or guessed/post-round loss scaling
or adaptive eta without the named drift/transport decomposition
=> NO EXACT COMMIT CLAIM; KEEP HEDGING OR FALL BACK TO EXISTING CONTROLLER.
```

### Authority and scope

This is **DERIVED** from Corollary 2.3's fixed-temperature identity in Balsubramani, arXiv:2607.08789v1. It is algebraically equivalent to the posterior-mass threshold and is proposed as an auditable controller gate. It is not empirical evidence, not a standalone intrinsic-time stopping theorem, and not launch/pointer authority.

### Value provenance

- `c_t`: exact receiver-realized, common-state/full-vector record or explicitly corrected estimate.
- `eta`: fixed typed configuration for the first arm.
- `alpha`: preregistered controller risk tolerance.
- `p_t`, `Q_t`, `V_t`, `I_t`: deterministic derived ledger fields with identity-residual check.
- adoption: exact-through-R `n600`, NumPy-fp32 authority only; contest axes separated.
