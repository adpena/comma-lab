# DDM MS4D direct metric completion: triality and feed

Captured: 2026-07-24T16:58:20Z  
`research_only=true` · `score_claim=false` · `main_landing_review_required=true`

## DSL / code leg

The typed config
`.omx/research/configs/ddm_ms4d_direct_metric_completion_20260724.json`
selects `DIRECT_SCORER_INTRINSIC_NO_ACTUATOR_INPUT`, `n=600`, deterministic
seed `1234`, Torch threads `4`, 12-pair immutable checkpoints, exact sealed
source hashes, false-authority guards, and SSD-first storage.

Producers:

- `tac.optimization.ddm_metric_producers.direct_scorer_intrinsic_bucket_statistics`
- `tac.optimization.ddm_metric_producers.direct_scorer_intrinsic_pair_block`
- `tac.optimization.ddm_ms4d_direct_completion.materialize`

Admission/refusal:

- `tac.optimization.ddm_metric_custody_bundle.load_metric_custody_bundle`
  with `require_complete=True`;
- `tac.optimization.ddm_ms4d_waterfill_admission.build_post_admission_refusal`;
- registered row law
  `ddm_tolerance_capped_min_score_waterfill_v1`.

## DAG leg

```text
sealed PF2 atlas + exact event index
              |
frozen SegNet head + exact solved-plane R roundtrip
              |
rank-4 margin Fisher rows + exact-empty authentication
              |
exact 25 RG3 residual blocks + signed-probe custody
              |
Seg / reused Pose / composite-R / dual components
              |
MS3 BUNDLE-COMPLETE --require_complete=True--> ADMITTED
              |
candidate-materialization gate
              |
REFUSE: receiver builder + uint8 quantum + candidate delta
        + dimension rate home + coder owner all absent
              |
162 cells have complete metric context;
0 same-object rung deltas measured; 162 byte prices remain NULL
```

The refusal edge is part of the executable DAG. It prevents metric curvature
from being silently promoted into a receiver actuator or byte price.

## Equation leg

For class pair `a,b`, centered SegNet-head quotient normal
`n_ab ∈ R^4`, and exact pairwise margin `m_i` at each SHA-bound PF2 support:

```text
f_i = 1/2 sech^2(m_i / 2)
G_b = (1/600) Σ_i f_i n_ab n_ab^T
g_b = (1/600) Σ_i f_i m_i n_ab
```

The exact solved plane is realized through `R` before the post-R quotient is
read. Therefore `G_b` is the direct composite model Hessian and `g_b` its
adjoint in `POST_R_PENULTIMATE_HEAD_QUOTIENT`; no receiver-coordinate secant is
claimed. For each exact RG3 terminal residual block `(pair,bucket)`, the same
formula is evaluated without pooling and the actuation type is
`UNREACHABLE_BY_COUNTED_COORDINATES`.

An admissible materialized rung would use:

```text
S = 100 e/N + sqrt(10 D_pose) + 25 B_best/37,545,489
```

subject to `e ≤ 136,839`, exact uint8 revalidation, one-object parse-back, and a
real coder race. No such rung exists in the current bundle, so all waterfill
and price outputs remain NULL.

## Canonical feed delta

- MS3 consumer gate changes from PARTIAL refusal to genuine COMPLETE admission.
- RD1 learns `162/162` cells have complete direct metric bundle context.
- RD1 does not learn same-object rung deltas or byte prices:
  `0/162` and `0/162` measured respectively; `162/162` prices remain NULL.
- The next planner edge is now precise:
  `PF3_RECEIVER_OBJECT_AND_TYPED_RATE_HOME_ABSENT`.
- The contest pointer is unchanged:
  `0.1910828242 [contest-CPU]`.
