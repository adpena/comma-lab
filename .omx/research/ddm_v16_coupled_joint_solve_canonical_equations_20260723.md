---
title: DDM V16 coupled joint solve canonical equations
utc: 2026-07-23T02:01:00Z
tasks: [603, 613, 366]
research_only: true
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
score_claim: false
main_landing_review_required: true
---

# DDM V16 coupled margin level-set system

Let `u` concatenate the counted shared 2x2 template bytes and signed RGB compensation on the 23
measured collateral supports. Let `m(u)` contain 32 target crossings, 23 protected non-crossings,
and 96 two-sided PoseNet trust margins. Backprop through the counted receiver and exact `R` gives
the local coupling operator `M = dm/du`, measured here as a `151 x 141` matrix.

Inside one frozen local model, the minimum-description step is the conditional convex QP

```text
minimize     1/2 d^T (G + M^T W M + lambda I) d
subject to   M d >= epsilon - m(u)
             -rho <= d <= rho.
```

For a fixed active set `A`, the equations are

```text
[H  -A^T] [d ] = [0]
[A    0 ] [mu]   [b].
```

This closed system is exact only conditional on the local affine model and a correct active set.
The measured run did **not** solve it cleanly: all four first-order/Gauss-Newton attempts ended
`MAX_ITERATIONS_RESIDUAL_NOT_CLEAN`. It is therefore incorrect to describe v16 as a converged
closed-form SQP solve.

The VJP matrix passed eight sampled central finite-difference entries under the preregistered
absolute-or-relative rule. Maximum absolute error was `1.5648212865926325e-5`; maximum relative
error was `1.3461337280273438` on a near-zero entry. This validates those samples only, not every
entry or a global linearization.

After Hessian-metric Babai projection and the real counted receiver, realized candidate
correlations were negative: round 1 GN/first-order `-0.135574/-0.159551`; round 2 both
`-0.101633`. The largest rejected candidate used 12 of 23 compensation supports (120 payload bytes
beyond the placement-only program). Both rounds selected the zero-compensation hold control.

The corrected full-n600 receiver is `135,328 B`, `d_seg=0.027470296224`,
`d_pose=163.061327281443`, Movable `0.291615222639`, Lane `0.435195521828`. Its camera pixels are
byte-identical to v15. Fork C fires at `INSTANCE x canonical Lane=1 eight-island operating point x
configured trust boxes`; M is a reviewable #366 costate/preconditioner input, not a promoted score
or launch authorization.

Executable anchor: `src/tac/canonical_equations/ddm_v16_coupled_margin_law_20260723.py`.

CONSUMED-BY: MAIN landing review; #366 only after a fresh lane/dispatch claim. Pointer
`0.1910828242 [contest-CPU]` unchanged.
