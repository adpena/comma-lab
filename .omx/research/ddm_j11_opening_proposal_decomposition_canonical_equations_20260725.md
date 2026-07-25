---
schema: ddm_j11_opening_proposal_decomposition_equations.v1
lane_id: lane_ddm_j11_366_opening_proposal_decomposition_20260725
research_only: true
score_claim: false
pointer_moved: false
main_review_required: true
---

# DDM J11 opening-proposal decomposition — canonical equation contract

Let `theta_p` be the receiver coordinates of one sealed J10 proposal, `R` the exact
resize/uint8/parse-back chain, `G` the receiver, and `W0` the exact J10 source.

The pose-null Seg component is defined only when the source-bound Jacobian exists:

```text
J_pose,p = d(Pose6 o R o G)(W0 + theta_p) / d theta_p | theta_p=0
P_pose-null,p = I - pinv(J_pose,p) J_pose,p
delta_pose-null-seg,p = integer_realize(P_pose-null,p delta_p)
```

The Seg-null Pose component needs an actuator-inner Jacobian in the sealed rank-4 head
quotient, not merely the quotient metric:

```text
J_seg-r4,p = d(SegRank4 o R o G)(W0 + theta_p) / d theta_p | theta_p=0
P_seg-null,p = I - pinv(J_seg-r4,p) J_seg-r4,p
delta_seg-null-pose,p = integer_realize(P_seg-null,p delta_p)
```

Both projectors require proposal foreign keys, source SHA custody, integer realization,
receiver parse-back, and exact n600 scoring. A metric tensor without its receiver-coordinate
inner Jacobian does not define either projector.

The #580 range(A) projection instead guarantees:

```text
A(P_range(A) X) = A(X)
```

It removes resize-invisible `ker(A)` energy. It does not imply either
`J_pose,p P_range(A)=0` or `J_seg-r4,p P_range(A)=0`, so substituting it for a scorer-null
projector is unauthorized.

PC1 composition additionally requires the identity law:

```text
G_PC1(active=0, W0) == G_J5(W0)   byte-for-byte
```

The named PC2 receipt falsifies that law at this source. Therefore the measured PC2 ratio is
not compositional authority.

Once a candidate is lawfully materialized, pricing remains exactly:

```text
delta_S =
  100 * (d_seg_candidate - d_seg_source)
  + sqrt(10 * d_pose_candidate) - sqrt(10 * d_pose_source)
  + 25 * (candidate_bytes - source_bytes) / 37_545_489

admit iff delta_S < 0
```

No component proxy, decomposition norm, or predicted delta may replace this realized-through-R
rule.
