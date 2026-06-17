# Yousfi road↔lane geometric-solve probe — the dominant-d_seg eureka test

**Date:** 2026-06-17 · **Lane:** `lane_yousfi_road_lane_geometric_solve` (L1)
**Authority:** `[macOS-CPU advisory]` — exact frozen CPU SegNet/PoseNet (NEVER MPS),
GT-decoded via `upstream/frame_utils.yuv420_to_rgb`; NOT the 600-sample contest
harness → NON-PROMOTABLE per Catalog #192/#341/#127/#323.
**Frontier pointer: UNMOVED** (this is a $0 diagnostic, not a score row).
**Artifacts:** `experiments/probe_yousfi_road_lane_geometric_solve.py`,
`.omx/research/yousfi_road_lane_geometric_solve_20260617.json`.

This probe CONSOLIDATES (does not rebuild) the existing real-scorer primitives:
`seg_core.load_real_segnet`/`decode_gt_frame1_pairs`,
`frame1_joint_safe_cone.measure_segnet_frame1_margin`/`measure_pair_distortion`,
`partition`/`contour_codec` (contour bytes), `resize_null_preimage.ResizeProjector`
(the exact bilinear preimage), `margin_conditional_residual` (the KKT waterfill).

## What was measured (4 real GT pairs; survival/KKT on 2)

| measurement | result |
|---|---|
| dominant contour class pair | **class 0 ↔ class 1** (chosen by longest shared boundary, not area) |
| road↔lane share of total boundary | **47.5% mean** (44.9–49.2%) |
| road↔lane share of low-margin (flip-prone) band | **42.7% mean** (41.9–43.5%) |
| L* lossless contour bytes (whole partition) | **~830–915 B / frame** |
| pose-init warp residual (median px, SEG grid) | 0.53 mean |
| **identity (no-warp) residual** | **0.33 mean — contour is nearly STATIC** |
| LSQ homography residual | 0.99 mean (ICP over-fits a near-static curve) |
| resize-deconvolution preimage realization error | **0.19 luma on support (≈exact)** — survives the 874→384 downsample |
| KKT residual coder cost | **0.78–0.83 B/flip < 1.273 waterline → BEATS** |

## The four verdicts (concrete + honest)

### (a) Pose-warp eureka — DOES NOT HOLD AS STATED; the real finding is QUASI-STATIONARITY
The road↔lane contour is **nearly static across the whole window**: the
identity (no-warp) median residual stays 0–1 px even at 22-frame spacing (t=11).
There is almost nothing for the pose to *warp* — the ego car drives straight down
its lane, so the lane edges sit in nearly the same image location frame-to-frame.
The pose-derived homography does NOT measurably beat identity (it can't — the
0/1 px residual floor is the pixel grid). **So the contour IS cheaply
reproducible across many frames, but the mechanism is reuse-of-a-near-static-curve,
NOT pose-warping.** This is *better* for bytes than a per-frame homography (store
γ₀ once + tiny per-frame deltas), but it falsifies the specific "markings = pose-
warped geometry" framing. Reactivation: a turning/lane-change clip where the
contour actually moves would let the pose-warp beat identity — untested here.

### (b) Resize-deconvolution — SOLVED, NOT A WALL (the wall was my probe's proxy)
The bilinear preimage (Landweber on the exact separable `ResizeProjector`)
realizes its SEG-grid target to **0.19 luma on the marking support** and the
signal survives the 874→384 downsample (59.95 of a 60-luma bump delivered). So a
crisp lane marking absolutely CAN be made to survive the resize. The apparent
"naive ≈ deconv, both ~35% support flip" result is because my naive/deconv proxy
edit (a +60 luma bump along the boundary) is the WRONG signal: a blind luma bump
pushes pixels *across* the argmax wall (more bump → MORE flips: 0.35→0.49), it does
not restore the GT class. The deconvolution faithfully delivers whatever target it
is given; **the binding constraint is the target CONTENT** (the GT-class-restoring
color), which is the margin-KKT lever below, not the carrier.

### (c) Margin-KKT residual — NET-POSITIVE on the tail
The residual road↔lane flips code at **0.78–0.83 B/flip**, below the
1.273 B/flip break-even, via the decoder-free margin field (`measure_code_cost` +
`waterfill_select`). So once the bulk contour is carried by the (near-static)
curve, the tail flips are cheap to fix-by-sidecar — the lever D economics hold on
this real contour.

### (d) Total bytes for the road↔lane d_seg contribution
The whole-partition lossless contour is ~900 B/frame; the road↔lane sub-contour
is ~47% of that boundary, and ~43% of the flip-prone band — i.e. the dominant
contour is genuinely O(boundary)-cheap, consistent with the directive's "cheap-by-
geometry" hypothesis on the BYTE axis. The d_seg-CLOSING question (does a carrier
that reproduces this contour actually drive the road↔lane d_seg → 0 through the
real eval chain) is NOT yet answered: the deconv carrier is exact, but choosing the
GT-class-restoring target color at each marking pixel is the open margin-KKT step.

## VERDICT
The lane-marking-geometric-solve is a **math-optimal CHEAP CARRIER (bytes) for the
dominant contour, but NOT yet a closed d_seg-solve**, and the "pose-warp" mechanism
is **falsified at adjacent-frame spacing in favor of quasi-stationarity**. The two
real walls are (1) the contour barely moves so pose-warp adds nothing here, and (2)
the binding work is choosing the GT-class-restoring target (margin-KKT), not the
carrier (deconvolution is exact). Per Catalog #307 this is an IMPLEMENTATION-LEVEL
result; the PARADIGM (geometric/closed-form attack on the dominant d_seg contour)
is INTACT and re-pointed at: **store the quasi-static γ₀ contour + a small per-frame
delta + a margin-KKT GT-class target sidecar**, and measure the realized road↔lane
d_seg of THAT carrier through the exact chain (the next byte-closed step).

## 6-hook wire-in
1 sensitivity-map: ACTIVE (contour = the dominant low-margin band, 43% of flips).
2 Pareto: ACTIVE (contour bytes ~900B vs L* lossless; KKT tail < waterline).
3 bit-allocator: ACTIVE (KKT B/flip < 1.273 admission).
4 cathedral autopilot: N/A (advisory probe, non-promotable).
5 continual-learning: this memo + JSON (the empirical anchor).
6 probe-disambiguator: this IS the disambiguator (pose-warp vs quasi-stationarity).
