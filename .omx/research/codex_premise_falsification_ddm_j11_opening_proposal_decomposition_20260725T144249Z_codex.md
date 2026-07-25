---
schema: codex_premise_falsification_v1
lane_id: lane_ddm_j11_366_opening_proposal_decomposition_20260725
authority_sha256: 36ec3ede75ac1bf2d9f6565d4e05ba7e4d2202efdc1174895e644586b88d7ec5
research_only: true
score_claim: false
pointer_moved: false
main_review_required: true
verdict: BLOCKED_J11_PROPOSAL_DECOMPOSITION_CUSTODY_PRECONDITION
verdict_scope: PRECONDITION_APPARATUS_NOT_FORMULATION
---

# DDM J11 opening-proposal decomposition — premise falsification

The delegated premise that the named MS4 metric bundle, range(A) projector, and PC2 pose
coordinate are sufficient to materialize the eight requested null-space singles and four
composites is **falsified**.

This is a custody/precondition result, not a negative result for a correctly joined
null-space decomposition:

1. The sealed `BUNDLE-COMPLETE` Pose metric is exact in six-dimensional PoseNet output
   coordinates, but supplies neither a receiver-coordinate PoseNet Jacobian nor a foreign key
   joining any J10 proposal to that metric.
2. The sealed Seg metric declares
   `DIRECT_SCORER_INTRINSIC_NO_ACTUATOR_INPUT`. Its rank-4 head quotient is not joined to the
   J5/J10 proposal coordinates and therefore cannot define a receiver-space Seg-null
   projection.
3. `tac.boundary_math.range_a_projection.apply_projection` removes
   `ker(A)` resize-invisible energy while preserving `A(PX)=A(X)`. It is a resize-gauge
   canonicalizer, not a SegNet-null projector.
4. PC2's active-zero receiver home is not identity-preserving at the exact J10 source. Against
   the 138,813-byte J10 source it is 139,547 bytes, changes d_seg by
   `-0.04482749938964844`, and changes d_pose by `+127.54549145969125`. Its measured
   `14.023295441931698` ratio cannot be transferred or composed as though zero meant the J10
   source.

Consequently no receiver-realizable component archive exists to price. All eight requested
single-component rows and all four composed rows are explicit `null`, not zero; none has n600
evidence. No scorer, smoke, campaign, or READY/FIRE reseal was run.

The resolving apparatus is a SHA-bound, per-proposal receiver-coordinate `J_pose`, the
corresponding rank-4-inner Seg Jacobian, and a PC1 adapter whose active zero provably emits the
exact J10 source bytes. Only after integer realization and receiver parse-back may the
unchanged pure-priced n600 rule arbitrate the formulation.

Evidence receipt:
`.omx/research/ddm_j11_366_opening_proposal_decomposition_refusal_20260725.json`
(SHA-256 `25f092d3499283a77dfcda274015af6826d1f3ce38ac91c26d4a01afa12ad7f4`).

Pointer `0.1910828242 [contest-CPU]` is **UNMOVED**. MAIN landing review is mandatory.
