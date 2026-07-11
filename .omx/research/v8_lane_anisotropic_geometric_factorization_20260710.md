# v8 LANE rate crux — ground-frame anisotropic factorization, MEASURED on the real cached argmax — 2026-07-10

**Question (deep-geometry dispatch, follow-up to FEED-lag):** the v8 feasibility probe isolated the Lane
class as the expensive carrier (97 B/frame image-space at only 0.835 recall / 0.601 IoU; Road–Lane = 60%
of all boundary pixels) and named "ground-frame factorization + xi transport" as the path from 97 to a
few B/frame. Does exploiting the lane's ANISOTROPY — geometry (IPM to the ground chart) + factorization
(static world paint x ego transport, tracked coefficient matrices) + SPD-cone water-filled coding — make
the lane cheap at >= the image-space fidelity?

**Authority: `[macOS-MLX advisory]` geometric feasibility.** All fidelity is measured against the CACHED
SegNet argmax (`experiments/results/mlx_fleet_gt_cache/gt_n600.npz` `lstars`, real n600, numpy load — NO
scorer forward, NO model inference), same recall/precision/IoU definitions as the FEED-lag baseline. NOT
through R + the frozen SegNet, NOT byte-closed. A rate ESTIMATE / bound, not a score.
**Pointer 0.19108282 UNMOVED.**

**Apparatus:** `src/tac/boundary_math/lane_ground_factorization.py` (knot-chart re-parametrization +
lateral tracker + SPD-cone track coding + lossless world-aligned XOR occupancy codec + skeleton arm) +
CLI `tools/probe_v8_lane_anisotropic_factorization.py`; 11 tests incl. a synthetic anisotropic-vs-
isotropic SPD control, a known-ego-travel world-static dash recovery control, and a wrong-shifts
losslessness control. Artifact: `experiments/results/v8_lane_factorization_probe_20260710/probe.json`
(+ `run.log`). Class order verified by spatial/static SIGNATURE (lane area 0.59%, MyCar temporal IoU
0.993 bottom-static), never luma-sorted. Reused, not rebuilt: `lane_sdf_component` IPM/fit (v_h=174,
#327-optimal), `partition_collapse` baseline, `pose_spd_codec` (the −27% pose winner, commit 348ac229f).

## The construction (three measured factors)

1. **GEOMETRY:** IPM to the ground frame; each per-frame lane line re-parametrized as LATERAL KNOTS —
   lateral (m) at fixed forward distances {8,14,24,40} m — bijective with the deg-3 centerline poly,
   temporally smooth, units-consistent (+ 2 half-width knots + forward range = 8 dims).
2. **FACTORIZATION:** lines tracked ACROSS frames by ground lateral (29 tracks over n600, obs coverage
   0.994); per-track (n_obs, dims) coefficient matrices. Dash/visibility factored as world-static paint
   x shared per-frame forward travel S(t): the occupancy stream is coded as world-aligned XOR deltas
   (shift by S(t), XOR vs the previous obs, brotli) — LOSSLESS, so a wrong shift only costs bytes.
   S(t) is estimated from the lane pixels alone (pattern alignment): total travel 1022.4 m over 600
   pair-frames (1.71 m/frame ~ 17 m/s, plausible); standalone cost 142 B, amortized ~0 as the stored
   pose xi's integral (dual-use).
3. **SPD-CONE CODING:** per-track matrices coded with the reverse-water-filling codec
   (`encode_pose_section_spd` on dim-normalized knots). Measured pooled anisotropy d_H = **6.69** (n600;
   10.79 on n96) — strongly anisotropic, the codec's regime.

## Measured ladder (n600, ALL 600 frames fitted + evaluated; bytes = REAL encode, fidelity = REAL round-trip -> raster)

| stage | B/frame | total /600 | recall | precision | IoU |
|---|---|---|---|---|---|
| S0 image-space baseline (poly runs) | **98.0** | 58.8 KB | 0.824 | 0.666 | **0.590** |
| S1 ground per-frame independent (12-bit, per-obs periodic dash) | 77.6 | 46.6 KB | 0.635 | 0.557 | 0.421 |
| S2 factorized solid (theta=1e-6) | 54.4 | 32.6 KB | 0.871 | 0.404 | 0.381 |
| **S3 factorized + lossless occ gate (theta=1e-6, 105 bins)** | **73.7** | **44.2 KB** | **0.871** | 0.634 | **0.580** |
| S3 cheaper point (theta=1e-6, 60 bins) | 61.5 | 36.9 KB | 0.873 | 0.558 | 0.516 |
| S3 cheaper point (theta=1e-5, 60 bins) | 54.9 | 32.9 KB | 0.828 | 0.529 | 0.476 |
| S4 skeleton: smoothed knots + static paint x visibility (theta=1e-5) | 22.9 | 13.8 KB | 0.625 | 0.327 | 0.274 |
| S4 skeleton (theta=1e-4) | 17.3 | 10.4 KB | 0.571 | 0.297 | 0.244 |

S3 byte split at the headline point: coeff (6 shape dims, SPD) 23.0 KB + occupancy (world-XOR brotli)
21.2 KB + S(t) 0.14 KB standalone / 0 amortized.

## Attribution matrix (n96 diagnostic, theta=1e-6 — which factor carries the fidelity?)

| variant | B/frame | recall | precision | IoU |
|---|---|---|---|---|
| A raw knots + lossless occ | 78.3 | 0.884 | 0.665 | 0.612 |
| B SMOOTHED knots + lossless occ | 67.3 | 0.760 | 0.578 | 0.496 |
| C raw knots + STATIC occ | 44.2 | 0.825 | 0.427 | 0.390 |
| D smoothed + static (the skeleton) | 33.2 | 0.713 | 0.378 | 0.329 |

Removing the knot-trajectory jitter costs 0.12 IoU for ~11 B/f; removing the occupancy flicker costs
0.22 IoU for ~34 B/f. **Both per-frame jitter channels are load-bearing signal for the argmax metric.**

## Verdict (NO-FAKE, both halves)

**WIN — vs the 97 B/frame image-space baseline:** the anisotropy-aligned factorization DOMINATES image
space at matched fidelity: **73.7 B/frame (44.2 KB/600) at recall 0.871 / IoU 0.580** vs 98.0 B/frame at
recall 0.824 / IoU 0.590 — **25% fewer bytes with +0.047 recall and −0.010 IoU** (precision −0.032). Every
designed lever measurably pays: the ground chart + knot parametrization (solid recall 0.871 > 0.824), the
tracking + SPD-cone water-fill (coeff 7.7 B/obs at near-lossless; d_H 6.69), and the world-aligned XOR
occupancy codec (35→24% below raw with losslessness guaranteed). The per-frame PERIODIC dash model is a
measured NEGATIVE (S1: recall collapses to 0.64 — ego speed is not constant; verdict_scope: formulation).

**NEGATIVE — vs the SPEC's 1-2 KB lane target: NOT REACHED, and the residual factor is now MEASURED.**
The bound is ~44 KB at parity fidelity (22–44x the target). The cost is NOT geometry: the smooth-geometry
skeleton (S4) costs only 10–14 KB and the truly static parts (paint 0.8 KB, S(t) 0–0.14 KB) are near-free.
The cost is reproducing the SCORER'S per-frame jitter: knot-trajectory jitter (~8 B/obs survives
near-lossless SPD+delta+brotli — the trajectories are jitter-dominated, not smooth) + occupancy flicker
(~7 B/obs survives world-aligned XOR — the dash pattern is NOT static at the argmax level). The
attribution matrix shows dropping either jitter channel costs 0.12–0.22 IoU. This is the parametric-domain
twin of FEED-lag §4's pixel-delta negative (scattered annulus jitter, high entropy) and matches L66 (~97%
of d_seg lives in the annulus) + L67 (44% of CE-residual spikes = LANE): **the lane carrier's irreducible
cost is the argmax flicker, which no static-world/ego-transport factorization can absorb because it is not
transported — it is scorer noise-shaped signal.** The v8 SPEC's 1-2 KB lane line item is therefore the
SKELETON price (S4-class, measured 10–14 KB at 0.24–0.27 IoU here; tighter smoothing/spline-knot coding
could push toward 2–4 KB), with the annulus jitter owned by the INR/trainer budget — exactly FEED-lag's
"hybrid = rate skeleton + prior, not a standalone carrier."

**xi correlation (advisory):** estimated dS(t) vs cached PoseNet dims peaks at |corr| 0.44 (dim 4) — the
shared temporal mode is pose-coupled but the bin-quantized lane-only estimate is noisy; in a real archive
S(t) comes from the stored pose section directly (dual-use, free), so this does not gate the construction.

## Effect on the v8 per-class hybrid (candidate arithmetic, not byte-closed)

Swapping the FEED-lag hybrid's lane term 97 -> 73.7 B/frame moves the union hybrid 229 -> ~206 B/frame
(rate 0.092 -> ~0.082) at slightly better lane recall. Path to a real row: compose the S3 lane carrier
into the per-class hybrid, byte-close the archive, measure through `upstream/evaluate.py` per v8 SPEC §6
(v7.5-first gating unchanged). A lane-rate estimate is a CANDIDATE until byte-closed — the pointer moves
only through the exact eval.

**Triality:** DAG FEED-lanefact appended (2026-07-10). Equations leg: NO new canonical equation — the
jitter-dominance finding is a measured anchor CANDIDATE for the SPEC §5 per-class carrier-allocation law
(still council-flagged; this is a pre-anchor, not an n600-through-R anchor). DSL leg N/A (analysis/codec
probe, not a trainer lever).
