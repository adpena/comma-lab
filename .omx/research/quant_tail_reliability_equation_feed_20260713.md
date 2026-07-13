# Canonical-equation FEED — quantitative tail reliability — 2026-07-13

The executable equations live in
`src/tac/canonical_equations/control_interpolator_tail_reliability_20260713.py` and are
covered by dedicated regression tests. The append-only canonical equation registry was
already modified by live siblings, so this lane does not absorb or commit their rows.
Main review may populate these records after the shared surface is clean.

## `control_interpolator_tail_cvar_mean_gate_v1`

```text
lambda* = argmin_{lambda in Lambda, lambda>0} CVaR_alpha(L_lambda)
          subject to empirical_mean(L_lambda)
                     <= empirical_mean(L_reference) + epsilon_mean
tie_break = (p99, mean, smaller_lambda)
```

- Domain: control-driving interpolators with block-held-out losses, explicit reference,
  positive regularization, closed search bracket, and declared numerical authority.
- Excluded: score claims, selecting on an official final holdout, open-boundary adoption,
  and relabeling forecast error as counterfactual control regret.
- Producers: `tools.measure_quant_tail_reliability_20260713`.
- Consumers: costate support selector; witness-control organ gate; NCDE/prototype router;
  rate-law and scorer-response planners; cathedral/autopilot admission.
- Empirical anchors: PRE-SE block2/block3 cached dev curves and organ A/P/Q seven-fold
  curves in the receipt.
- Recalibration: three or more new block-independent in-domain anchors.

## `fixed_design_correlated_gaussian_ridge_tail_v1`

For `Q-EQ=u'Au-tr(A)+2c'u`, `u~N(0,I)`, `A` positive semidefinite, and
`t=log(1/delta)`:

```text
P[Q-EQ >= 2 sqrt((||A||F^2 + 2||c||2^2)t) + 2||A||op t] <= delta.
```

- Domain: finite fixed design, known correlated Gaussian covariance whitening, quadratic
  prediction/displacement loss.
- Retained-mass transfer additionally requires a measured top-k boundary-margin envelope.
- Excluded: guessed covariance, arbitrary correlated non-Gaussian coordinates, a numeric
  retained-mass guarantee without boundary margins, and any asymptotic-rate transfer.
- Producer/consumer: canonical equation module and tail measurement/admission tool.
- Empirical status: DERIVED law; numeric closure on current cache is false.

## Registry disposition

`registry_append=DEFERRED_HOT_SHARED_FILE`, not silently registered. The executable law,
typed DSL, DAG FEED, equation FEED, measurement receipt, and tests are complete. Applying
the two registry rows must use `tac.canonical_equations.registry.register_canonical_equation`
under its lock and a serializer patch that excludes all sibling rows.
