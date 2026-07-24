---
schema: ddm_ic1_incumbent_compose_and_buy_row_canonical_equations.v1
date_utc: 2026-07-24
axis: "[macOS-CPU frozen-scorer advisory]"
research_only: true
score_claim: false
---

# IC1 canonical equations note

## Exact objective

For exact packed archive bytes \(B\),

\[
S(B,D_{\rm seg},D_{\rm pose})
=100D_{\rm seg}+\sqrt{10D_{\rm pose}}
+\frac{25B}{37{,}545{,}489}.
\]

For the freshly decoded IC1 packet,

\[
100D_{\rm seg}=7.051923116048177,
\quad \sqrt{10D_{\rm pose}}=16.522253967415644,
\quad \frac{25B}{N}=0.08761505276972155,
\]

so

\[
S_{\rm advisory}=23.66179213623354.
\]

This is `[macOS-CPU frozen-scorer advisory]`, not a contest-axis score.

## Compose then measure

For a receiver composition \(R(T(x))\), the authoritative delta is

\[
\Delta S =
S\!\left(B_T,D_{\rm seg}(R(T(x))),D_{\rm pose}(R(T(x)))\right)
-S\!\left(B_0,D_{\rm seg}(R(x)),D_{\rm pose}(R(x))\right),
\]

not a sum of historical per-piece deltas. The measured W_joint→PA1 transition is

\[
\Delta B=288,\quad \Delta D_{\rm seg}=0,\quad
\Delta D_{\rm pose}=-9.319697135033131,
\]

\[
\Delta S_{\rm pose}=-2.613624573014981,\quad
\Delta S_{\rm rate}=+0.0001917673784991858,
\quad \Delta S=-2.6134328056364815.
\]

## Local secant is not a KKT price

The observed two-point secant is

\[
\frac{\Delta S}{\Delta B}=-0.00907441946401556.
\]

It measures this exact typed runtime transition. It is not a transferable
per-stream dual because the parent W_joint construction is a
`[naive-menu upper bound]` and the omitted scorer-recursive
paint/support/exception controls have not been jointly replayed. Therefore

\[
\lambda_{\rm W\_joint}=\lambda_{\rm paint}
=\lambda_{\rm correction}=\texttt{NULL}.
\]

The c1 byte split remains a reservation identity.

## Scorer-recursive construction typing

PA1 satisfies a named evaluator recursion:

\[
A_{\rm seg}(x_0,x_1)=A_{\rm seg}(x_1),\qquad
A_{\rm pose}(x_0,x_1)\ne A_{\rm pose}(x_1),
\]

so the transform is restricted to frame 0 and derived from frozen PoseNet
stem/BN statistics. W_joint does not satisfy the 2026-07-24 provenance bar;
its replacement must source colors and support from corrected inner-Jacobian
rows, the exact resize adjoint footprint, stride-2 stem lattice, ERF scale,
Fisher-margin ranking, and Pose-null/priced composition.
