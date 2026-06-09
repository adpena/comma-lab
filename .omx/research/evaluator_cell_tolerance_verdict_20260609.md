# Evaluator-cell tolerance on the contest video — verdict (RGB is NOT the objective)

UTC 2026-06-09 · claude · `tools/hi_nerv_renderer_sanity_ladder.py evaluator-cell-tolerance`
Artifact: `.omx/research/evaluator_cell_tolerance_20260609.json`. [macOS-CPU advisory] (exact
upstream DistortionNet on 0.mkv pair 0). Operator redirect 2026-06-09: "we don't care about RGB or
human fidelity; all we care about is contest video + upstream evaluate.py + lowest score possible."

## What this measures
NOT "how good must the renderer reconstruct RGB" (wrong question). Instead: start from the SOURCE
(d_seg=0, d_pose=0) and CHEAPEN it (downsample / blur / quantize), measuring how much d_seg/d_pose
stay near zero. The headroom = the SIZE of the evaluator-equivalence class = the rate budget for a
low-score witness. This is the "compress what the evaluator can see" measurement.

## Result — the evaluator-equivalence class is LARGE (lots of cheapening tolerated)
| axis | level | d_seg | d_pose | seg_term(100x) | pose_term(√10x) |
|---|---|---|---|---|---|
| downsample | 2× (1/4 px) | 0.0006 | 0.001 | 0.063 | 0.089 |
| downsample | 4× (1/16 px) | 0.0019 | 0.000 | 0.189 | 0.026 |
| downsample | 8× (1/64 px) | 0.0093 | 0.004 | 0.926 | 0.204 |
| downsample | 16× (1/256 px) | 0.0186 | **19.06** | 1.86 | **13.8** ← pose breaks |
| blur | r=2 | 0.0014 | 0.001 | 0.141 | 0.097 |
| blur | r=8 | 0.0193 | 1.085 | 1.93 | 3.29 |
| blur | r=16 | 0.1008 | 0.804 | 10.08 | 2.84 ← seg breaks |
| quantize | 6-bit | 0.0004 | 0.000 | 0.038 | 0.013 |
| quantize | 4-bit | 0.0027 | 0.001 | 0.269 | 0.081 |
| quantize | 1-bit | 0.0311 | **4.48** | 3.11 | **6.69** ← pose breaks |

Cell boundary (d_seg<0.02): downsample ≤8-12×, blur ≤8, quantize ≤3 bits. The source's evaluator
view survives ~1/64-spatial-DOF + 3-4 bits/value + mild blur. HUGE byte headroom vs the 178 KB frontier.

## THE decisive comparison (answers the burning question's second half)
The HiNeRV renderer at 21 dB PSNR gives **d_seg=0.507**. A 16×-downsampled, far-"lower-fidelity"
source gives **d_seg=0.0186**. A 1-bit-quantized source gives **0.031**. So the renderer is **~27×
WORSE on d_seg than a 1-bit source** despite "higher PSNR". CONCLUSION: the renderer is NOT a
low-fidelity version of the source — it is on the WRONG MANIFOLD entirely (it MISSES the class-boundary
geometry, exactly as hypothesized). d_seg is not bottlenecked by fidelity; it's bottlenecked by the
renderer producing frames outside the evaluator-equivalence class.

## PoseNet × SegNet interaction (operator directive)
The two terms have DIFFERENT tolerances: d_seg degrades gracefully (blur breaks it first, ~r=16);
d_pose has CLIFFS at extreme cheapening (16× downsample → 19.06; 1-bit → 4.48) — PoseNet's 2-frame
temporal/YUV6 structure collapses when the spatial/precision is gutted. The JOINT cell boundary =
MIN of the two axes' tolerances. A low-score witness must respect BOTH; pose is the tighter constraint
at the aggressive (low-byte) end.

## Strategic verdict (retargets task #35)
1. STOP trying to "fix the renderer's RGB fidelity ceiling" — that was the wrong target (RGB is not the
   objective; the renderer is on the wrong manifold, not merely low-fidelity).
2. The lowest-score path is to produce a witness that is a CHEAPENED-BUT-CELL-CORRECT version of the
   source — the evaluator-inverse / direct grammar (V3). The tolerance curve PROVES the headroom exists.
3. The renderer (V1 HiNeRV) needs scorer-aware training that targets the SegNet argmax + pose DIRECTLY
   (the cells), not RGB — OR yield to V3. A one-pair SCORER-objective probe (not RGB) is the test of
   whether the NeRV can be driven into the cells at all.
4. NEXT: the full evaluator response surface (per-pixel SegNet margin field + per-pose-dim PoseNet
   Jacobian, in score units, across the contest video) → the score-domain Lagrangian / waterfilling
   solution. This is the operator's "curves across all dimensions, solve mathematically" — see
   `.omx/research/evaluator_response_surface_solve_plan_20260609.md`.
