# DDM J12 receiver-coordinate custody equations

`research_only=true`; `[macOS-CPU frozen-scorer advisory]`; no score or promotion claim.

## Sealed receiver coordinate

For proposal \(p\), J12 uses its one-dimensional, boundary-feasible ray

\[
W_p(\alpha)=W_0+\alpha\Delta_p,\qquad \alpha\in[0,1].
\]

RG1 made the negative reflection infeasible for three of the four rays. The lawful realized
secant is therefore forward:

\[
J^{(p)}_{\rm pose}
  = P(R(\operatorname{recv}(W_p(1))))
    -P(R(\operatorname{recv}(W_0))),
\]

\[
J^{(p)}_{\rm seg}
  = H_4(R(\operatorname{recv}(W_p(1))))
    -H_4(R(\operatorname{recv}(W_0))).
\]

\(P\) is the frozen PoseNet six-output map and \(H_4\) is the exact SegNet rank-4
winner-rival inner surface. Every array is preserved per n600 pair at batch32 after receiver
parseback, uint8 realization, and the actual scorer resize \(R\).

## Exact null projection

For either measured Jacobian \(J\), the coordinate-space Gram and null projector are

\[
G=J^\top J,\qquad
\Pi_{\ker J}=V\,\operatorname{diag}(\mathbf 1_{\lambda_i\le\tau})\,V^\top,
\quad G=V\operatorname{diag}(\lambda_i)V^\top.
\]

Each sealed proposal has one coordinate. All eight measured Jacobians have rank one and
nullity zero, hence

\[
\Pi_{\ker J^{(p)}_{\rm pose}}
=\Pi_{\ker J^{(p)}_{\rm seg}}=[0].
\]

The integer-realized pose-null Seg and Seg-null Pose singles are therefore active-zero, not
estimated nonzero directions.

## Source-preserving PC1 rehome

Let \(C(W,q)\) be the original PC1 receiver and \(q_0\) its active-zero packet. The J12
adapter is

\[
\widetilde C(W,q)
=\operatorname{uint8clip}\left(
  W + C(W,q)-C(W,q_0)
\right).
\]

At active zero,

\[
\widetilde C(W,q_0)=W
\]

byte-for-byte. For the sealed source this is 138,813 bytes and SHA-256
`2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241`.

## S-primary exact admission

For an exact realized endpoint \(x\),

\[
S(x)=100d_{\rm seg}(x)+\sqrt{10d_{\rm pose}(x)}
     +25\,B(x)/37{,}545{,}489.
\]

Admission is exclusively

\[
\Delta S=S(x)-S(x_0)<0.
\]

The operating-point pose marginal is

\[
\frac{\partial S}{\partial d_{\rm pose}}
=\frac{5}{\sqrt{10d_{\rm pose}}},
\]

with break-even term ratio 1.0 and no fixed \(R^\*\). Auxiliary quantities cannot reject a
negative realized \(\Delta S\).

## Measured consequence

Because every exact null projector is zero, all 16 singles are active-zero with
\(\Delta S=0\). The eight named composites collapse to two unique rehomed-PC1 endpoints:

\[
\Delta S_{\rm Wjoint}=-2.761204260556886,
\qquad
\Delta S_{\rm Wseg}=-3.5711431248357903.
\]

The four-step live J10 smoke changes the W-joint composite by
\(+0.12759259096760986\); EMA remains byte-identical with change \(0\). These are local
advisory measurements, not contest scores. Merged-main reseal and FIRE authority remain with
MAIN after landing review.
