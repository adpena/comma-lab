# FEED — DDM PC1 Pose Stream Admission

`research_only=true` · `score_claim=false` · `promotion_eligible=false`

## Executable DAG

```text
exact W_seg bytes ─┐
                   ├─ parent receiver ─ decoded frame0 + Movable mask ─┐
exact W_joint bytes┘                                                   │
                                                                       ├─ Φ(qξ)
32-knot qξ + luma controls ─ parse/re-emit ─ smooth pair controls ─────┤
camera intrinsics + height ─ ground D + Movable contact D ─────────────┘
                                                                       │
                                          frame0 ─ W_{ξ,D} ─ frame1 ──┤
                                                                       v
                                  uint8 camera pair ─ evaluator R ─ Seg/Pose
                                                                       │
MS4d active-tube metric ───────────────────────────────────────────────┤
exact W parent solved-plane YUV6 target ─────────────────── #366 descent
                                                                       │
exact archive bytes ─ rate term ─ direct conditional ΔS ──────────────┘
```

## Edges and consumers

- DSL: `.omx/research/configs/ddm_pc1_pose_stream_admission_20260724.json`.
- Equations:
  `src/tac/canonical_equations/ddm_pc1_pose_stream_20260724.py` and
  `.omx/research/ddm_pc1_pose_stream_admission_canonical_equations_20260724.md`.
- Receiver/owner:
  `src/tac/optimization/ddm_pc1_pose_stream.py`.
- Builder/resume/measurement:
  `tools/build_ddm_pc1_pose_stream.py`.
- #366 consumes the 320 stable knot coordinates and the parent-derived exact
  solved-plane YUV6 target. The zero home is not a claimed solution.
- #417 consumes the typed owner ledger: both PC1 output effects have one owner,
  `pose/pc1.ddp`.
- MS6 consumes the nonzero-q causal composite-R support receipt.
- MS4d supplies the exact PoseNet-output quadratic. The measured zero-home row
  makes no tube claim because no descent was run.

## Unified-solver wire-in

1. Sensitivity: the landed MS6 mechanism measured 410,468 composite-R cells for
   the nonzero probe against the active zero-q home for each exact parent.
2. Pareto: direct conditional deltas are +2.1700709084033565 for W_seg and
   +16.652345570764727 for W_joint.
3. Bit allocation: the counted home is 40 bytes; deterministic nesting adds
   734 bytes over either exact parent. Future coordinates must stop at marginal
   score break-even.
4. Autopilot: `research_only=true`, `promotion_eligible=false`, and positive
   conditional deltas make this admission non-dispatchable.
5. Continual learning: the advisory PARTIAL probe-outcome row points to the
   sealed receipt and routes the next action to #366 solved-plane descent.
6. Disambiguation: the measured per-pair/RGB initialization is retained only as
   a coordinate-artifact upper bound. The admitted optimal form is the smooth
   constraint stream targeting the already-solved scorer plane.

## Pointer delta

Pointer before and after: `0.1910828242 [contest-CPU]`. No mutation.
