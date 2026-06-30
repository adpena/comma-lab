# DAG FEED — v2 deterministic witness corner, PHASE 1 byte-close smoke (REAL eval, CPU)

Date: 2026-06-30. Agent: `v2_byteclose_smoke` ($0 CPU-only). Lane:
`lane_v2_deterministic_witness_byteclose_smoke`.

## TL;DR (the win for this phase = a REAL measurement + a validated apparatus)

Three **REAL byte-closed exact-eval rows** through the actual
`archive.zip -> inflate.sh -> inflate.py -> 0.raw -> upstream/evaluate.py --device cpu`
path (NO scorer weights in archive; self-contained). Apparatus VALIDATED: a
lossless witness scores EXACTLY `d_seg=0, d_pose=0`. Parity holds on every mode
(inflate output byte-equals the build render oracle).

| row (n24) | d_seg | d_pose | rate (n24) | S (n24) | n600 rate proj | n600 S proj | blocker |
|---|---|---|---|---|---|---|---|
| store_raw (lossless) | 0.0 | 0.0 | 2.156 | 53.91 | 53.91 | ~1348 | rate |
| v2_warp (crude warp, no resid) | 0.0265 | **190.04** | 1.080 | 73.23 | 26.99 | ~721 | rate + d_pose |
| store_jpeg q40 (lossy codec) | 0.00209 | 0.0134 | 0.032 | 1.38 | 0.801 | ~20.6 | rate (167x) |

Frontier pointer **UNMOVED at contest-CPU 0.19110** (no move claimed; these are
[macOS-CPU advisory] component-real rows, small-n).

## What this DECISIVELY measured (not asserted)

1. **The byte-close pipeline + small-n CPU eval apparatus work and are trustworthy.**
   `store_raw` (store both frames losslessly) -> `d_seg=0, d_pose=0` exactly =>
   the GT cache pair-ordering, the truncation harness (`--batch-size n` so the
   n-pair comp is one full batch; `zip(dl_gt,dl_comp)` stops there -> scores
   first n pairs), and the integer-deterministic decode are all correct.

2. **Per-pair pixel storage is RATE-DEAD in EVERY form — by 150x-450x.** sub-0.19
   needs the whole 600-pair archive < ~180KB (~300 bytes/pair; rate term ~0.118).
   Measured per-pair bytes: store_raw 3.4MB, v2_warp 1.7MB, JPEG-q40 50KB ->
   167x-11000x over budget. The deterministic corner CANNOT store per-pair content.

3. **The scorer is ROBUST to lossy pixel content.** JPEG q40 of both frames:
   `d_seg=0.0021` (~3.5x frontier 6e-4) and `d_pose=0.0134` (contributes 0.366).
   => the corner does NOT need pixel fidelity. Distortion is NOT the blocker for
   stored content; RATE is the entire game.

4. **The crude per-class warp is POSE-CATASTROPHIC.** `v2_warp` (road-band integer
   roll <=6px from pose, sky/hood identity) -> near-static frames -> PoseNet reads
   ~no ego-motion -> `d_pose=190` (frontier ~3.4e-5). A deterministic warp MUST be
   the REAL ground-homography (genuinely move the scene by the 6-dof pose) for
   PoseNet to read the right motion. (d_seg only 0.027 — last-frame warp(f0) is
   close to gt_f1 segmentation-wise; pose is the wall.)

## Honest verdict vs the deliverable's question

- byte-closes? **YES** (3 real exact rows).
- parity? **YES** (deterministic decode == oracle, all modes).
- clears 0.19 at n600? **NO, and not close.** The simplest per-pair-storage
  deterministic corner projects to S 20.6-1347.8 at n600 — RATE-dominated by
  150x-450x. It is NOT PR95-adjacent (~0.19); it is **much worse** as built.

## Why (the structural law this surfaces)

The corner's viability hinges ENTIRELY on the RATE term. Per-pair storage is
structurally impossible. The ONLY form that can fit the ~180KB budget is the
AMBITIOUS one (per FEED-ja / grok / FEED-ko ~0.103):
- ONE/few SHARED canonical keyframe(s) across many frames (NOT per-pair),
- the 600x6 pose trajectory (~14KB),
- a REAL ground-homography per-class warp that GENERATES frames from the
  canonical (fixes the d_pose=190 crude-warp wall),
- a TINY task-survival residual (lane-edge survival + small movables) via a real
  range/entropy coder.

Whether that ambitious form clears 0.19 is **UNMEASURED (build-gated)**. PHASE 1
de-risked the eval apparatus so the ambitious build is measurable identically.

## Scope: what's built vs build-gated (in-tree audit)

[FOLD-IN from parallel Explore agents — pending] The boundary_math modules
(amortized_luma_carrier, lane_sdf_component, hood_static_component,
road_horizon_component, contour_codec, context_partition_codec) + witness_dsl +
the L13 format + task #107/#163 byte-close tooling are the candidate substrate
for the ambitious build. The v2 6-section codec FLAGS are build-gated DESIGN
(FEED-kk). This PHASE-1 smoke used a self-contained throwaway codec
(`experiments/v2_witness_byteclose_smoke.py`) precisely because the full v2
codec is not yet flag-exposed.

## Next unit (aimed at a real exact row that could move the pointer)

Build the AMBITIOUS corner increment that attacks RATE first (the only blocker):
1. Real ground-homography per-class warp from comma2k19 intrinsics + 6-dof pose
   (kills d_pose=190). Probe: v2_warp with the REAL homography, n24 eval — does
   d_pose drop from 190 toward the JPEG-level 0.013?
2. SHARED-canonical generation: how few keyframes can generate the 1200 frames
   with task-survival? (this is the rate unlock — the per-pair wall.)
3. Compose: shared canonical (coded) + trajectory + homography warp + tiny
   lane/movables residual -> byte-close at n600 -> real exact eval.

## Reproduce

```
.venv/bin/python experiments/v2_witness_byteclose_smoke.py --mode {store_raw|v2_warp|store_jpeg|v2_det} --n 24 [--jpeg-q 40] --out <submission_dir>
# unzip + inflate (sh-driven), then:
.venv/bin/python upstream/evaluate.py --submission-dir <abs> --uncompressed-dir upstream/videos \
  --report <abs>/report.txt --video-names-file upstream/public_test_video_names.txt --device cpu --batch-size 24
```
Result JSON: `experiments/results/v2_deterministic_byteclose_smoke_RESULT.json`.
