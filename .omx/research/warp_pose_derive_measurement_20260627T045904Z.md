# DERIVE-by-warp pose mechanism probe — MEASURED (n96, realized-through-PoseNet CPU-torch authority)

- **Date:** 2026-06-27T04:59:04Z
- **Axis tag:** `[macOS-CPU advisory]` / mechanism-probe — **NON-PROMOTABLE**, NOT a byte-closed row, NOT a score claim.
- **Authority:** frozen CPU-torch PoseNet via the EXISTING harness
  (`experiments/train_witness_realized_through_R_mlx.py::load_gt_from_cache` +
  `cpu_verdict_d_pose_batch`). NEVER hand-rolled, NEVER MPS. `gt_n96.npz`.
- **Probe script (scratch, not persisted as evidence):** `scratchpad/warp_pose_probe.py`;
  raw JSON `scratchpad/warp_pose_results.json`.
- **Libs:** cv2 4.11.0 (phaseCorrelate / ORB+RANSAC estimateAffine2D / findHomography / Farneback),
  scipy.cluster.vq.kmeans2 (posterize-5). Frames 874×1164 uint8 RGB.

## What was tested

Hypothesis "DERIVE-by-warp": frame_1 = a geometric WARP of a textured frame_0 by the ego-motion,
so PoseNet(frame_0, warped_frame_1) READS that ego-motion → low d_pose, making pose "the warp
parameter" instead of stored texture. The warp transform is **fit on the real `gt_f0 → gt_f1`
(full-texture ORACLE params)**, then applied to the texture source to synthesize `f1_hat`.
`d_pose = mean over the 6 PoseNet scalars of (pose(f0,f1_hat)[:6] − gt_pose[:6])²`; `pose_term = √(10·d_pose)`
(the actual contribution to S). Target context: usable pose_term < 0.1 (d_pose < 1e-3); "solved"
stored-sidecar ≈ 0.018 (d_pose ≈ 3.4e-5).

**Pose authority facts (upstream/modules.py, used in interpretation):** PoseNet bilinear-resizes BOTH
frames to ~512×384, converts to YUV6, STACKS into a 12-channel input, and a single fastvit_t12 reads
6 pose scalars (the first 6 of the 12-dim hydra head). It is NOT siamese/flow — the 512×384 resize
averages away camera-res detail, so the warp need only be faithful at the 512×384 YUV6 scale, and only
needs to land the 12-ch stack in the 6-scalar pre-image (huge freedom). The 512×384-scale warp variant
the coordinator suggested was NOT added as a separate measured cell because translation/affine/homography
params are already resolution-independent and the dense-flow `ds8` byte-cost row already estimates the
coarse-scale flow cost; adding a 512×384-fed cell would mismatch the PoseNet resize path against the
full-res `gt_pose` authority and break apples-to-apples.

## ANCHORS (calibration)

| anchor | mean d_pose | pose_term | note |
|---|---:|---:|---|
| UPPER BOUND / harness check `d_pose(gt_f0, gt_f1)` | **4.08e-12** | 6.4e-6 | ≈ 0 ✓ harness VALID (it IS the GT pair) |
| COLLAPSE FLOOR `identity × posterized` (f0=f1=poster5) | 181.9 | 42.66 | static/no-motion floor (this segment gt_pose[0]≈34 forward speed) |

## 2-AXIS TABLE — mean d_pose / pose_term=√(10·d_pose) (n=96, 0 fit-failures across all models)

| WARP MODEL (params/pair → bytes/600) | (a) full GT texture | (b) blurred (σ=8) | (c) posterized 5-color (SDF proxy) |
|---|---:|---:|---:|
| **(0) identity** (0 / 0) | 182.0 / 42.66 | 140.4 / 37.46 | 181.9 / 42.66 |
| **(1) translation** (2 / 2.3 KB) | 182.0 / 42.67 | 129.3 / 35.96 | 182.5 / 42.72 |
| **(2) affine** (6 / 7.0 KB) | 167.1 / 40.88 | 142.4 / 37.74 | 164.5 / 40.56 |
| **(3) homography** (8 / 9.4 KB) | 91.9 / 30.32 | 71.1 / 26.66 | 84.4 / 29.05 |
| **(4) dense optical flow** (ds8 → **3.4 MB**; fullres → **67 MB**) | **1.42 / 3.77** | 1.88 / 4.34 | 2.21 / 4.70 |

Byte cost detail: translation 4 B/pair, affine 12 B/pair, homography 16 B/pair (fp16, the existing
sidecar convention); dense flow quant-0.25px+zlib = 5,810 B/pair downsampled-8× grid (3.4 MB/600) or
114,472 B/pair full-res (67 MB/600). **Reference (NOT measured here):** the already-built
stored-target sidecar = 6 scalars × fp16 × 600 ≈ 7.2 KB raw (~1–5 KB compressed) → pose_term ≈ 0.018.

## THE 4 VERDICTS

**1. Does DERIVE-by-warp work, and how cheap?**
**NO — not at any usable level, and the only model that even partially works is the most expensive.**
The cheap parametric warps are catastrophic: translation (2p) does NOTHING (pose_term 42.7, identical
to the no-motion floor), affine (6p) barely moves it (40.9), homography (8p) is partial but still
catastrophic (30.3). Reason is geometric: forward ego-motion produces a **depth-dependent radial
parallax flow** that NO single 2D global planar warp (translation/affine/homography) can represent.
Only **dense optical flow** reaches single-digit d_pose (1.42 full-texture, pose_term **3.77**) — but
that is still **~38× above** the usable threshold (pose_term < 0.1) and **~210× WORSE** than the stored
sidecar (0.018), while costing **3.4–67 MB/600** (vs ~7 KB). Crucially this is with the **ORACLE flow**
(fit on the true `gt_f0→gt_f1`): pose_term ≈ 3.8 is the **upper bound** of derive-by-warp; any real
implementation can only be worse. The residual warp artifacts (occlusion/disocclusion/interpolation)
perturb the 6 scalars no matter how good the flow.

**2. Does it need texture in frame_0? (a vs b vs c)**
For the cheap warps, texture is irrelevant — they fail regardless. For the winning model (flow), texture
gives only a SMALL gain and does NOT change the verdict: full 1.42 < blur 1.88 < poster5 2.21 — all the
SAME order of magnitude (pose_term 3.8–4.7). So **the bottleneck is the warp residual, NOT texture in
frame_0.** The posterized-5-color (SDF-vehicle proxy) flow cell (2.21) is within 1.6× of full texture —
the SDF vehicle would NOT be rescued by carrying texture in the SegNet-invisible frame_0. (Note: carrying
texture in frame_0 IS free since frame_0 is 100% SegNet-invisible — but it doesn't buy a usable pose.)

**3. Per-vehicle recommendation:**
**DERIVE-by-warp is NOT the optimal pose path for ANY vehicle — it is dominated on BOTH axes by the
already-built stored-target sidecar.** STORE-direct (6 scalars/pair) is ~200× better in d_pose AND
~500–10,000× cheaper in bytes than the flow ceiling, and the 6-scalar pre-image freedom means it
composes trivially with any render.
- **SDF / non-RGB level-set vehicle → STORE-direct sidecar** (the 6 scalars; do NOT attempt warp-derive — the SDF render's flat palette would, if anything, be worse than the posterized cell). Pose stays decoupled from the SDF d_seg render.
- **RGB witness → STORE-in-FiLM or STORE-direct sidecar** — both reach the 6 scalars cheaply; warp-derive offers no advantage.

**4. HONEST caveats (binding):**
- Mechanism probe on **GT frames** with **ORACLE warp params** (fit on the real `gt_f0→gt_f1`). The
  oracle is an UPPER BOUND: a real vehicle must STORE the warp params (counted bytes) or derive them
  imperfectly → strictly worse than these numbers. Even the oracle ceiling fails.
- NOT byte-closed, NOT an exact-eval row, NO score claim. `[macOS-CPU advisory]` / NON-PROMOTABLE.
- The realized witness frame_0 will be the WITNESS RENDER (degraded), not GT — bracketed by (b)/(c),
  which do not change the verdict.
- **Pose is NOT "solved" by warp-derive.** Pose IS solved (separately, already) by the stored-target
  sidecar (pose_term ≈ 0.018); this probe CONFIRMS warp-derive is the wrong path and the stored sidecar
  remains the recommendation.

## Wire-in (per "Results must become system intelligence")
Result is a typed negative that prunes the pose-design search: **derive-by-warp DOMINATED by
STORE-direct sidecar at the PoseNet upper bound.** Feeds the witness-capstone DAG (FEED-cx) as the
closing answer to the pose-design question. 6 unified-Lagrangian hooks: sensitivity-map N/A (no new
byte axis admitted); Pareto N/A (dominated arm, no constraint added); bit-allocator N/A; cathedral
autopilot N/A (NON-PROMOTABLE mechanism probe); continual-learning = this memo + DAG feed; probe-
disambiguator = THIS probe IS the arbitration between derive-by-warp vs store-direct (store-direct wins).
