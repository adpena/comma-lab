---
schema: ddm_a2_strategy_verdict_provenance_canonical_equations.v1
date_utc: 2026-07-24
lane_id: lane_ddm_a2_strategy_verdict_provenance_20260724
research_only: true
execution_allowed: false
main_review_required: true
---

# DDM A2 provenance and price-authority equations

## No arithmetic invalidation

The source equations remain arithmetically valid on their named finite domains.
This note changes provenance and permissible consumption, not their numeric
evaluation.

## `ddm_a2_construction_authority_firewall_v1`

For candidate or stream `x`, define:

```text
constructed_from_recursive_scorer(x) :=
    rank4_margin_target_bound
  and corrected_inner_J_bound
  and exact_resize_footprint_bound
  and stride2_stem_support_bound
  and ERF_scale_bound
  and pose_null_or_pose_price_bound

exact_stage_measured(x | parent) :=
    parent_archive_sha_bound
  and serialized_member_and_archive_bytes_measured
  and exact_R_seg_pose_replay_measured
  and receiver_parseback_closed
  and sequential_parentage_preserved
```

An achieved positive with `exact_stage_measured=true` remains a real achieved
point regardless of construction provenance. It cannot become a family ceiling,
negative, or transferable marginal price unless construction coverage is also
adequate for that claim. When `constructed_from_recursive_scorer=false`, its
only wider-family authority is:

```text
achievable_improvement_family >= achieved_improvement_x
best_distortion_family <= measured_distortion_x
```

That is `STANDS-AS-BOUND`, not closure.

## `ddm_a2_negative_scope_law_v1`

For a finite proposal set `M`,

```text
min_{x in M} DeltaS_exact(x) >= 0
```

proves only:

```text
FORMULATION:M has no admitted member under the exact tested parent/order/gates.
```

It does not imply the same inequality for the scorer-recursive family `F` when
`M` is not exhaustive in `F`. Thus v18’s no-negative-column finding and the
finite PT1/Menu1 ordering cannot close or price `F`.

## c1 reservation quotient is not a dual

The exact c1 accounting is:

```text
R_base = 133941 + 270 = 134211 bytes
R_free = 200000 - R_base = 65789 bytes
R_reserved = 25789 + 16384 + 16384 + 7232 = 65789 bytes
lambda_rate = 25 / 37545489
```

The identity `R_reserved=R_free` is exact. It is not a solution of

```text
min_{r_i >= 0, sum r_i <= R_free}
  100 D_seg(r) + sqrt(10 D_pose(r)) + lambda_rate sum r_i
```

because current c1 lacks measured joint `D_seg(r), D_pose(r)` stage curves.
Values obtained by dividing required break-even error reductions by reserved
bytes are feasibility thresholds, not empirical KKT multipliers.

An actionable stream price requires an exact lower-envelope segment:

```text
p_i =
  - Delta[100 D_seg + sqrt(10 D_pose)] / Delta archive_bytes
```

with same-parent sequential replay and nonadditivity measured. Otherwise:

```text
p_i = NULL
```

## RD1 restricted hull

RD1 exactly solves the lower hull of its 110 observations:

```text
H_110 = lower_convex_hull({(R_j, D_j)}_{j=1}^{110})
```

Its seven Pareto points, four lambda-supported points, and observed knee are
exact properties of `H_110`. Since candidate construction is not exhaustive,
`H_110` is a diagnostic upper bound on the unknown global distortion curve, not
the global curve itself. The 162 typed dimension prices correctly remain
`NULL`.

## Conditional 216→264 KB widening

Solving the objective budget at a held Seg distortion once with measured Pose
and once with `D_pose=0` is valid counterfactual arithmetic. It is not evidence
that any legal equal-byte preimage selector realizes the second endpoint. The
widening is therefore a derived conditional interval until the same-archive
preimage-policy A/B measures both scorers through exact R.

## Training-evidence empty set

`training_necessary_residual=EMPTY_ON_CURRENT_EVIDENCE` denotes:

```text
{training claims satisfying all preregistered exhaustion,
 real-fit, receiver, and exact-byte evidence gates} = empty set
```

It does not denote an empty trainable representation family.

## Triality, scope, and landing

- DSL: these laws specify the fail-closed predicates behind the findings enums.
- DAG: the companion DAG feed routes measurements to consumers.
- Equations: this file is the provenance/price firewall.

These are proposed append-only law definitions, not registry mutations.
`main_review_required=true`; canonical registration, if desired, is a separate
MAIN-reviewed action. Pointer `0.1910828242 [contest-CPU] UNMOVED`.

## HISTORICAL_PROVENANCE

Append-only companion to
`codex_findings_ddm_a2_strategy_verdict_provenance_20260724T151509Z_codex.md`.
It supersedes no source equation or receipt.
