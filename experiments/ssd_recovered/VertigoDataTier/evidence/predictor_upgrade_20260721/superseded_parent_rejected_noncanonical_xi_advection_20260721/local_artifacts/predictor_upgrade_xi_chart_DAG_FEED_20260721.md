# DAG FEED — Task #578 predictor upgrade

`FEED-578-XI-CHART` · `research_only=true` · `[macOS-CPU advisory]` ·
`score_claim=false` · `promotion_eligible=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` · `MAIN_REVIEW_REQUIRED=true`

```text
Task #595 coherent_slot_none_dash.lbnd2
  | raw=159386, brotli11=41303
  | sha256=d2b2a62e...650d810
  v
hash/raw/brotli validation -> canonical LBND2 decode
  -> analytic_lane_render_band per-pair dash raster -> threshold 0.5 Lane mask
  -> executed_in_task578_measurement=true
  -> receiver_closed=false

frozen cache labels n64 development prefix
  |-- self-detected IDs [Road0,Lane1,Undrivable2,Movable3,MyCar4]
  |-- measured adjacency edges
  |-- counted Road/Undrivable static slices
  `-- counted MyCar static hood slice (#139)
                 |
                 v
      PXCH1 parse-backed counted chart payload
                 |
caller-supplied prior decoded field (t>0)       chart-only initialization (t=0)
measurement uses lstars[t-1] oracle stand-in            |
                 +------------------+-------------------+
                                    |
PoseNet targets -> PoseTargetEgoEstimator -> registered s_t/s_r LawRefs
                                    |
                         tac.lie log(T[t-1]^-1 T[t])
                                    |
                         adjacent relative xi advection
                                    |
       Road/Undrivable -> executed Lane -> MyCar -> Movable
                                    |
                   measured-adjacency-only reconciliation
                   deterministic priority [0,2,1,4,3]
                                    |
                         measured predictor P_t
                                    |
                   only P_t violations become PPCS exceptions
                                    |
                    PROJECT sibling-owned receiver boundary
                                    X
            no receiver closure; no realization/projection edits
```

## Triality and measured terminal state

- DSL/schema: additive strict `receiver.generic_predictor_policy`; the external
  custody binds the exact Lane packet SHA, Task #578 measurement execution, and
  `receiver_closed=false`. Legacy absence remains byte-identical.
- DAG: this FEED records the actually executed Lane decode/render/reconcile path.
- Equation: `xi_advected_prior_per_class_chart_reconciliation_v1`, empirically
  scoped to Task #578 cell-description measurement.

At n600, Road=0.922695718466, Lane=0.699462381939,
Undrivable=0.986157668493, Movable=0.996465855203, and
MyCar=0.999132512305. The exact formulation verdict is
`PREDICTOR_TARGET_MISSED`.

D4 remains `FORMALIZATION_PENDING_THROUGH_R_REALIZED_SCORE_RECOVERY`: the R0
inventory is mask-only and no equal-fidelity through-R R1 result exists.

## Authority boundary

No camera-RGB realization, receiver closure, frozen scorer execution, compliant
archive, contest-CPU/CUDA score, promotion, or pointer movement is claimed.
PROJECT remains owned by sibling `realization_g2_lattice`; its receiver file is
untouched at
`6f3704726c57f9e02e628a792127d0ce16f1979e27097bbede003542942dbd1f`.

## STORES CONSULTED

Task #578 spec/current receipt; Task #595 finite packet/canonical renderer;
predecessor B2, G1, #139, and #234 artifacts; frozen n600 cache; current and
superseded SSD evidence trees; CLAUDE/AGENTS vehicle contracts.
