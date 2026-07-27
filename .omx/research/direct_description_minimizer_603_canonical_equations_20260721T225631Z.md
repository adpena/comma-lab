---
title: Direct-description minimizer canonical-equation notes
utc: 2026-07-21T22:56:31Z
task: 603
status: BUILDER_EQUATIONS_BOUND_NO_LAUNCH
---

# Canonical equations

## Exact objective

`S(z) = 100 d_seg(R(inflate(A(z)))) + sqrt(10 d_pose(R(inflate(A(z))))) + 25 len(A(z)) / 37,545,489`.

`len(A(z))` is the exact final ZIP length after canonical parse/re-encode. Seed bytes, component
bytes, payload sums, and parsed-description bytes are not substitutes.

## Integer cap law

For a full-precision solved target `(d_seg*, d_pose*)` and score ceiling `T`:

`B_cont(T) = (T - 100 d_seg* - sqrt(10 d_pose*)) * 37,545,489 / 25`

`B_strict(T) = ceil(B_cont(T)) - 1`.

The compiler evaluates this with Decimal precision from SHA-bound decimal strings. JSON floats and
planning-only receipts fail. Current displayed planning results, excluded from launch config, are
216,223 B at `T=0.1910828242` and 154,524 B at `T=0.15`.

## Common marginal law

`lambda_B = dS/dB = 25 / 37,545,489 S/byte` exactly.

Every recursive allocation node stores its marginal as a rational. The verifier recomputes
`|marginal_node - lambda_B|`; no rounded `6.6586e-7` value enters the compiler.

## Cells-then-tube constraint

At tolerance rung `tau`, first find `z` satisfying the Seg cell inequalities through integer `R`;
then solve Pose6/`dxi` only within that Seg-feasible polytope. Pose negatives are scoped by `tau`.
The hard integer/uint8 secant oracle decides admission; Euclidean/Fisher/Hilbert-like metrics steer.

## Description/receiver mapping

The v1 carrier mapping is:

- `entropy_state -> manifest.json`
- `xi_curve_knots -> seed.ppcs`
- `static_ground_coefficients -> base.pbase3`
- `pose6_dxi_residuals -> causal.pcr3`
- `sparse_events -> events.pce3`
- `exceptions -> components.pcomp3`

This mapping is a byte-exact re-expression of six legacy opaque z0 sections. It does not establish
that the aliases have the PRIMARY semantics named on the left. In particular, the current
`causal.pcr3` section is empty and the S4 receiver rejects a nonempty Pose6/`dxi` section.

## Completion residual law

For each preregistered pool and interpretation, persist exact signed rational gradient terms
`g_(c,i) = n_(c,i)/d_(c,i)` by governed coordinate `c`. The audited residual is
`r_KKT = max_c |sum_i g_(c,i)| <= epsilon_preregistered`; terms from different coordinates cannot
cancel. The verifier also binds terminal evidence and an independent audit receipt to external
expected hashes. Missing, all-gate-excluded, or unhealthy rows are `OPTIMIZER_NO_ADMISSION`, never
a reachability token.
