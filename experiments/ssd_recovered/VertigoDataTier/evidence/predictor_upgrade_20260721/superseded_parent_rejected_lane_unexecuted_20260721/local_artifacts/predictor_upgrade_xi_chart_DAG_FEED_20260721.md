# DAG FEED — Task #578 predictor upgrade

`FEED-578-XI-CHART` · `research_only=true` · `[macOS-CPU advisory]` ·
`score_claim=false` · `promotion_eligible=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` · `MAIN_REVIEW_REQUIRED=true`

```text
frozen cache labels n64 development prefix
  |-- self-detect class IDs [Road0,Lane1,Undrivable2,Movable3,MyCar4]
  |-- measured adjacency edges
  |-- counted Road/Undrivable static slices
  `-- counted MyCar static hood slice (#139)
                 |
                 v
      PXCH1 parse-backed counted chart payload
                 |
caller-supplied prior decoded field (t>0)       chart-only initialization (t=0)
                 |                                      |
                 +------------------+-------------------+
                                    |
PoseNet targets -> PoseTargetEgoEstimator -> registered s_t/s_r LawRefs
                                    |
                         tac.lie log(T[t-1]^-1 T[t])
                                    |
                         adjacent relative xi advection
                                    |
     +------------------------------+-----------------------------+
     | Road/Undrivable counted slice | Lane #595 external custody |
     | MyCar #139 static clamp       | Movable #234 prior bulk    |
     +------------------------------+-----------------------------+
                                    |
                   measured-adjacency reconciliation
                   priority [Road,Undriv,Lane,MyCar,Movable]
                                    |
                         doctrine predictor P_t
                                    |
                   only P_t violations become PPCS exceptions
                                    |
                    PROJECT sibling-owned receiver boundary
                                    X
                  no Task #578 realization/projection edits
```

## Triality

- DSL/schema leg: additive strict `receiver.generic_predictor_policy`; legacy absence remains canonical and byte-identical.
- DAG leg: this FEED. The causal prior custody is explicit; oracle `lstars[t-1]` is measurement-only and never a free serialized field.
- Equation leg: `xi_advected_prior_per_class_chart_reconciliation_v1`, with empirical status limited to Task #578 cell-description measurements.

## Measured terminal state

`PREDICTOR_TARGET_MISSED` at n600: Road 0.927235573, Lane 0.376984213, Undrivable 0.986160836, Movable 0.996644583, MyCar 0.999097471. This negative is formulation-scoped. D4 remains `FORMALIZATION_PENDING_THROUGH_R_REALIZED_SCORE_RECOVERY` because equal-fidelity through-R R1 custody does not exist.

## Authority boundary

No camera-RGB realization, frozen scorer execution, compliant archive, contest-CPU/CUDA replay, promotion, or pointer movement is claimed. PROJECT remains owned by sibling `realization_g2_lattice`; its receiver file is untouched.

## STORES CONSULTED

Task #578 spec and receipt; predecessor B2/S2/G1 artifacts; Task #597 schema/receiver; #595 Lane custody; #139 hood and #234 movable components; frozen n600 cache; SSD stages; CLAUDE/AGENTS vehicle contracts.
