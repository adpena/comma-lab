---
task: 603
feeds_task: 613
research_only: true
triality_leg: equations
formalization_status: FORMALIZATION_PENDING
main_landing_review_required: true
---

# DDM v5 structured-carrier law draft

`FORMALIZATION_PENDING`: this is a draft registry entry with two composed measurement anchors.
MAIN must review and register it; this branch does not mutate the canonical equation registry.

Let `r` be a self-detected semantic role, `k(r)` its measured frozen-cell class, `G_r(f)` its
structured geometry on frame-pair `f`, `V` the finite counted value set (five C1-role medians plus
the inherited palette row), `C` frozen local SegNet argmax, `R` the exact receiver, and `A` the
deterministic archive compiler.

## Route selection

```text
k(r) = spatial_static_self_detect(r)
v*(r) = argmax_{v in V} sum_f sum_{p in G_r(f) intersect {C(T_f,p)=k(r)}}
                           1[C(R(A(z; r->v))_f,p)=k(r)]
```

The tie-break is candidate order and therefore deterministic.  `k(r)` is measured and persisted;
it is not a fixed class index in the implementation.  The selector includes inherited paint, which
is necessary for Lane and Undrivable in the measured anchor.

## Palette-versus-geometry adjudication

```text
precision_r = |G_r intersect T_{k(r)}| / |G_r|
recall_r    = |G_r intersect T_{k(r)}| / |T_{k(r)}|
paint_r(v)  = |{p in G_r intersect T_{k(r)} : C(R_v,p)=k(r)}|
              / |G_r intersect T_{k(r)}|

geometry_right_values_wrong_palette(r) :=
    precision_r is positive and recall_r is high
    and paint_r(v_inherited) = 0
    and max_{v in V} paint_r(v) > 0
```

The measured anchors satisfy this predicate for Road and MyCar on `[448,464)`.  The threshold
language remains descriptive; a canonical registry formalization must preregister exact admission
thresholds before generalizing beyond this instance.

## Composed receiver and rate

```text
A_5(z) = ZIP(chart+Pose6, U, Road, Lane, MyCar, Movable)
B_5(z) = |A_5(z)|
A_5(parse(A_5(z))) = A_5(z)
```

The last equality is enforced byte-for-byte.  Measured anchors are
`B_5(n64)=119335` and `B_5(n256)=324260`.

## Per-stratum membership and Pose feasibility

```text
M_k(F) = sum_{f in F} |1_k(C(T_f)) intersect 1_k(C(R(A_5(z))_f))|
         / sum_{f in F} |1_k(C(T_f)) union 1_k(C(R(A_5(z))_f))|

P(F) = exact Pose6 coordinates emitted / required Pose6 coordinates
local_admit(F) := B_5 <= B_guard and P(F)=1
```

`M_k` is a local membership instrument, not d_seg.  n64 satisfies `local_admit` at 119,335 bytes
and 384/384 Pose6 coordinates; n256 fails the byte predicate at 324,260 despite 1536/1536 Pose6.
Neither predicate authorizes evaluator promotion.

## Registry draft

```text
law_id: structured_role_self_detected_value_max_composed_membership_v1
status: FORMALIZATION_PENDING
evaluator:
  n64_receipt_sha256: ab5332f2cdda595eb91deef1498b0460291b77e4c96773967b0e641f6a75424a
  n256_receipt_sha256: 3b2ea4c963568ba9cacf909ad507877c63b19a8dc8fba853fe098c4526838e6f
authority: local frozen-SegNet batch16 membership only
score_claim: false
d_seg_claim: false
```

CONSUMED-BY: `ddm_describe_line_rate_distortion_bracket_v1` composed-role lineage; registration landing `.omx/research/ddm_structured_carriers_law_registration_20260722T142000Z.md`; MAIN review required.
