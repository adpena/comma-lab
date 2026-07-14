# Codex premise falsification — raw dual Euclidean is not the ordinary Hessian metric — 2026-07-14

**POINTER STATUS: UNCHANGED.** `0.1910828242 [contest-CPU Linux x86_64]`;
the local PR128 defensive bank `0.1880443979880752` remains non-submission.
This is a `$0` mathematical grounding and local NumPy receipt, not a score row.

## Premise tested

The mission proposed the identity

```text
rho = ||eta1-eta2||_2
    = sqrt((theta1-theta2)^T Hess(F) (theta1-theta2))
```

for `eta=grad(F)(theta)`, with no inverse or solve. That identity is false for
a general positive-definite Hessian metric. Silently implementing it would
change `argmax_native_vjp_fidelity_v1` while preserving its name.

## Re-derivation

In a fixed local chart let `H = Hess(F)` and
`delta_eta = H delta_theta`. Then

```text
ordinary Hessian metric:
  delta_theta^T H delta_theta
    = delta_eta^T H^{-1} delta_eta

raw dual Euclidean metric:
  ||delta_eta||_2^2
    = delta_theta^T H^2 delta_theta
```

The Crouzeix relation `H_F H_F* = I` establishes the inverse-Hessian dual
metric. It does not identify the Euclidean metric in an arbitrary dual chart
with the primal Hessian metric. Raw dual Euclidean is exactly a no-solve
**squared-Hessian** geometry; the Fisher-natural cotangent geometry still
requires its typed inverse/solve.

## Local measurement

The deterministic 600-state, five-dimensional SPD fixture in
`.omx/research/bregman_v9_all_surfaces_measurement_20260714.json` measured:

- maximum `primal-H` versus exact `dual-H^{-1}` error:
  `5.684341886080802e-14`;
- maximum raw-dual-Euclidean versus `H^2` error:
  `9.094947017729282e-13`;
- raw dual Euclidean differed from the ordinary Hessian metric on `600/600`
  states.

Axis: `MEASURED_LOCAL_CPU_SYNTHETIC_MATH_FIXTURE_NOT_SCORE`. The retained real
n600 metric-selection status remains `NO_VERDICT_DATA_CUSTODY`.

## Applied correction

- `tac.information_geometry.bregman_v9_surfaces.local_hessian_dual_geometry_summary`
  computes and labels all four objects separately.
- The V9 policy consumes the single existing metric ID
  `argmax_native_vjp_fidelity_v1`; it does not mint or alias a second metric.
- `dual_euclidean_no_solve_scope=squared_hessian_H_squared_only` and
  `fisher_natural_cotangent_solve_elided=false` are sealed in the binding.
- Cosine remains a diagnostic curved restriction and cannot license a
  surrogate replacement.

Verdict scope: **PROMPT IDENTITY FALSIFIED FOR GENERAL SPD H; THE
SQUARED-HESSIAN FORM IS VALID; THE CANONICAL METRIC FAMILY IS NOT REJECTED.**

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; V9 metric helper and policy; canonical equation
registry; Nielsen public DOxML deck and primary gauge/curved-Bregman papers;
watched inbox through per-arm `2026-07-14T14:23:10Z` and fleet
`2026-07-14T14:18:00Z`.
