# FEED-603-pc2 — DDM PC2 PC1 Pose-Descent Smoke

`research_only=true` · `score_claim=false` · `promotion_eligible=false` ·
`MAIN_REVIEW_REQUIRED`

## Executable DAG

```text
ws4 W_joint_step50_live bytes
        │ exact parse-back
        v
PC1 active zero home + 32-knot smooth qξ
        │ bit-reversal knot order; 4-pair exact proposals
        │ accept iff d_pose descends AND score-domain ΔS < 0
        v
immutable checkpoint after every accepted step
        │
        ├── accepted 0  ─ exact n600 batch32 ─┐
        ├── accepted 8  ─ exact n600 batch32 ─┼─ paired slopes vs R*
        └── accepted 16 ─ exact n600 batch32 ─┘
                                               │
                                               v
               PC1_DESCENT_STAGE / FORMULATION-scoped ξ-advection PREDICT fork
```

## Triality and consumers

- DSL:
  `.omx/research/configs/ddm_pc2_pose_descent_smoke_20260725T121448Z.json`.
- Equations and typed laws:
  `src/tac/optimization/ddm_pc2_pose_descent.py`.
- Resumable runner:
  `tools/run_ddm_pc2_pose_descent_smoke.py`.
- Compact evidence:
  `.omx/research/ddm_pc2_pose_descent_smoke_result_20260725.json`.
- Full evidence:
  `/Volumes/VertigoDataTier/pact/experiments/results/ddm_pc2_pose_descent_smoke_20260725T121448Z`.

## Registered signal

- `MEASURED`: aggregate exact n600 `ΔS=-0.24750113405601581` over 16
  accepted steps; `d_pose` decreases in both 8-step windows.
- `MEASURED`: aggregate pose/Seg ratio `14.023295441931698` clears
  `R*=4.1215446777965665`.
- `MEASURED`: composite `d_seg` regression is class-structured—Movable and
  Undrivable improve while Lane, Road, and MyCar regress.
- `DERIVED`: constant-slope target horizon is about 1,216 total steps and is
  not a convergence or launch claim.
- Disposition: retain a bounded PC1 descent stage. If its exact slope dies,
  route depth-stratified object-local ξ advection to `PREDICT`; do not infer
  family closure from #601/#605 controls.

## Unified-solver wire-in

1. Sensitivity: each accepted row records knot/axis/direction, exact receiver
   visibility, and realized local pose/Seg/rate deltas.
2. Pareto: strict local joint-negative admission and exact n600 paired slope
   rows enforce the Seg/Pose/rate action.
3. Bit allocation: the 16 accepted coordinates added 23 exact archive bytes;
   later stages must stop at marginal score break-even.
4. Autopilot: `execution_allowed=false`; any continuation requires a new
   bounded authority and MAIN-reviewed horizon.
5. Continual learning: canonical probe row
   `ddm_pc2_pc1_solved_plane_pose_descent_n600_20260725` records `PARTIAL`
   advisory evidence, the first measured PC1 solved-plane descent sign, and
   the classwise collateral future planners must consume.
6. Disambiguation: score-domain loss and the raw-pose
   `PoseMarginalWeightLaw` are XOR. This run selected the former with static
   `w_pose=1`.

## Pointer delta

Before and after: `0.1910828242 [contest-CPU]`. No mutation.
