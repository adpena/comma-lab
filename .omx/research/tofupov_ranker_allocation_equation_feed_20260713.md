# Conditional equation feed — TOFU-POV ranker and VR-GHAL exact-oracle allocation

**Date:** 2026-07-13 UTC
**Status:** `DESIGN`, `research_only=true`, `REGISTRATION_DEFERRED_MAIN_REVIEW`
**Canonical registry claim:** **NONE**; this file is an isolated equation handoff, not a registry row
**Pointer delta:** `UNMOVED`

## Equation candidate 1 — `tofupov_costate_measurement_index_v1`

### Domain and hypotheses

Within a frozen decision epoch `e`:

- all lever measurements fork from the same reference checkpoint/regime;
- full pre-pull action descriptors lie in a rank-`r` linear subspace and their masks satisfy the
  future theorem audit;
- expected raw improvement is locally linear in latent coordinates;
- measurement costs are known and strictly positive;
- P8 headroom is either measured/authoritative or absent.

### Law

```text
U_ei = <theta_hat_e, z_hat_ei>
       + beta_e ||z_hat_ei||_(V_e^-1)
       + epsilon_imp_ei

I_ei = min(H_ei, max(0, U_ei)) / C_i.
```

If `H_ei` is absent, omit the cap. `epsilon_imp_ei` must be a derived confidence radius, never a
point reconstruction error relabeled as an upper bound. Rank by `I_ei` descending.

### Fail-closed conditions

```text
missing cost or C_i <= 0
or no stable/eigengap-bearing subspace receipt
or mask propensity missing / zero
or action outside frozen regime
or stale representation/checkpoint hash
=> FALLBACK_CURRENT_P8_OR_COST_ONLY_RANKER.
```

### Authority

The law ranks measurements. It does not predict an evaluator score with authority, authorize a
launch, or inherit the TOFU-POV regret theorem after the nonlinear P8 cap without a new proof.

## Equation candidate 2 — `vrghal_inverse_probability_exact_oracle_v1`

### Domain and hypotheses

At a VR-GHAL-requested iterate `x_t`, let `G_t=G(x_t;xi_t)` be an exact stochastic-oracle sample
with `E[G_t | F_t,x_t]=T(x_t)`, where `T` is the mean fixed-point operator. Let `G_hat_e,t` be a
frozen cheap proxy computed from pre-query information `C_t`, and let `p_t` be a `C_t`-measurable
exact-query probability with `0 < p_min <= p_t <= 1`. Draw `A_t | C_t ~ Bernoulli(p_t)` using fresh
randomness conditionally independent of `G_t`.

### Law

```text
G_tilde_t
  = G_hat_e,t
    + (A_t/p_t) * (G_t - G_hat_e,t).

E_A[G_tilde_t | C_t,G_t] = G_t.

E[G_tilde_t | F_t,x_t] = T(x_t)             (iterated expectation).
```

For a realized exact sample and residual `e_t=G_t-G_hat_e,t`, the allocation contribution to
conditional variance is

```text
Var_A(G_tilde_t | C_t,G_t) = ((1-p_t)/p_t) ||e_t||^2
```

componentwise/in the corresponding second-moment norm. Stochastic-oracle variance remains in the
total second moment. Hence exact-call saving `1-mean(p_t)` trades directly against a `1/p_t`
allocation-variance penalty.

### Fail-closed conditions

```text
p_t <= 0
or propensity not persisted
or surrogate changes within epoch
or exact oracle is biased / wrong runtime
or fixed-point nonexpansiveness-contraction and second-moment gates absent
=> EXACT_ORACLE_FALLBACK; NO VR-GHAL COMPLEXITY CLAIM.
```

### Authority

This is a derived unbiased-oracle wrapper. It is not claimed to be a theorem from either cited
paper, and it has no current empirical anchor. For a deterministic frozen-teacher label on a fixed
replay state, exact-label caching is the mandatory simpler comparator; this wrapper is relevant only
if valid fixed-oracle samples still require fresh labels and it wins complete call accounting beyond
that cache. Canonical registration waits for main review and a shared-surface-safe serializer
landing.

## Value provenance ladder

- `C_i`: measured/config-derived DSL epoch or dollar/runtime cost, with source row.
- `H_ei`: measured P8 floor/headroom only; otherwise absent.
- `r`, `U_e`, conditioning: measured from real action/effect rows; never inherited from Whitney.
- `beta_e`, `epsilon_imp_ei`: theorem/derivation rung with paper-version and assumption receipt.
- `p_t`: typed policy output with RNG seed, cheap-feature inputs, epoch hash, and persisted draw.
- empirical anchors: exact `n600`, receiver-realized, NumPy-fp32 authority only.
