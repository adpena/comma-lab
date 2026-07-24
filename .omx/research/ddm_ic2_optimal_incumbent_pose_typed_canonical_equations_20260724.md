---
schema: ddm_ic2_optimal_incumbent_pose_typed.equations.v1
date_utc: 2026-07-24
research_only: true
score_claim: false
main_review_required: true
---

# IC2 canonical equations

## Evaluator action

\[
S(x,B)=100D_{\rm seg}(R(x_1))
       +\sqrt{10D_{\rm pose}(YUV6(R(x_0),R(x_1)))}
       +{25B\over 37\,545\,489}.
\]

`R`, uint8, frozen scorers, and archive bytes are realized operators; no proxy
substitutes for them.

## PA1

For each frozen PoseNet stem/BN coordinate \(c\), IC2 derives its zero-byte
frame-0 affine from exact parent moments:

\[
\mu_c={1\over N}\sum_i z_{ic},\qquad
\sigma_c^2={1\over N}\sum_i z_{ic}^2-\mu_c^2,\qquad
x'_{0,c}=g_c x_{0,c}+b_c .
\]

The sealed \(g_c,b_c\) live in the export receipt. They are derived from the
frozen Pose stem/BN and exact W_seg moments, then realized through uint8 and R.
This is not a fixed one-quantum correction, so the later dynamic-quantum
directive does not authorize or require an invented per-coordinate Q8 ladder.

## Non-telescoping admission

\[
\Delta S(C\mid W_{\rm seg})
=S(R(C(W_{\rm seg})))-S(R(W_{\rm seg})).
\]

Admission requires the measured conditional value, not
\(\Delta S(C)+\Delta S(W_{\rm seg})\). The incumbent comparator is

\[
S_{\rm IC2}-S_{\rm v0}
=28.00173925293584-23.66179213623354
=4.339947116702302>0,
\]

so promotion is refused.

## Pose-carrier ownership

A counted pose code \(q_\xi\) is admissible only if a deterministic receiver
\(\Phi\) establishes both directions:

\[
q_\xi \xrightarrow{\Phi} (x_0,x_1),\qquad
\Delta q_\xi\ne0 \Rightarrow \Delta R(x_0,x_1)\ne0,
\]

and every output effect has one serialized owner. E2's pre-export
`nested_pose6` lacks this \(\Phi\), so its packet edge is absent rather than
zero-effect.

