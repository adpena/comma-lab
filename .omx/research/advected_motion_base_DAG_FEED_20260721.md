# DAG FEED — counted-xi advected motion base

`FEED-ADVECTED-BASE-20260721` · `research_only=true` ·
`[macOS-CPU advisory]` · `score_claim=false` · `promotion_eligible=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED`

```text
counted PPCS.trajectory (raw 11,768 B already charged)
  |-- decode canonical (tx, ty, yaw)
  `-- translation-first embed xi=(tx, 0, ty, 0, yaw, 0)
                         |
existing LawRefs --------+------- hash-pinned G1 pitch geometry only
                         |        (external proxy motion NOT consumed)
                         v
              tac.lie / GroundHomographyGeom
                         |
      +------------------+------------------+
      |                                     |
solved #549 frame0 RGB             frame0 five-class chart
      |                                     |
ground homography candidate           one-hot channel warp
      |                                     |
      |                            transported ground mask
      |                                     |
      +--- ground ? warped RGB : source RGB -+
                         |
              ground-stratified frame1 base
                         |
             RGB/chart custody hashes
                         |
deterministic #549/C1 frame1 target (reconstructed from frozen source planes)
                         |
 exact codec race: modular delta | xor delta | target replacement
                         |
                 byte-exact parseback guard
                         |
native CPU-Torch d_seg/d_pose + exact payload bytes
                         |
lambda*=6.658589531221714e-7 S/B admission by |xi| bucket
                         |
            n64 rate gate: advected bytes < static bytes?
                         |
                        NO
                         X
             n600 refused; pointer unchanged
```

## Triality

- DSL/schema: `predict_project_counted_planar_xi.v1` carries the existing
  counted motion; `predict_project_advected_base_measurement.v1` binds inputs,
  deterministic scorer custody, stages, and the n64/n600 gate.
- DAG: the executed path above. Each pair is an atomic stage; each eight-pair
  chunk is a preserved resume checkpoint.
- Equation: `frame_1_base=W_xi(frame_0_base)` and
  `chart_1=first_argmax(W_xi(onehot(chart_0)))`, resolved through existing
  `ego_motion_cumulative_se3_bspline_v1`,
  `lane_band_ego_factorization_source_reparam_v1`, and
  `lane_band_source_reparam_measured_resolution_v1` LawRefs.

## Terminal edge

The n64 prefix measured 19,739,340 B for the advected exact corrections versus
19,559,060 B for static, a +180,280 B loss. This terminal applies only to the
tested planar ground-stratified formulation. Its measured failure mode is warp
fidelity: d_pose improves modestly, but solved-target RGB disagreement and
d_seg both worsen. Depth-stratified, per-class, object, and lossy scorer-aware
successors remain open and must register separate lanes.

Canonical receipt:
`/Volumes/VertigoDataTier/pact/evidence/advected_base_20260721/advected_base/receipt.json`
(`4a6650f2b1c8f8290a0a099fecf8fa7dd7a1d8826b5bfa72441cf553f1e676d0`).

## STORES CONSULTED

PPCS B2 seed/decoder; #549/C1 solved-target memo and frozen source cache;
predict-project receiver/schema; canonical G1 receipt and warp helper; equation
registry; native upstream CPU-Torch scorer; lane and subagent registries; SSD
stage/checkpoint tree. MAIN review is required before landing.
