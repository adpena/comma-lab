# v8 rate-half feasibility — Laguerre generators vs per-class hybrid, MEASURED on the real cached argmax — 2026-07-10

**Question (deep-insight dispatch, #284 / v8 SPEC §2):** does the real SegNet argmax partition compress
to a FEW Laguerre/Bregman power-diagram GENERATORS, and at what byte-cost vs boundary fidelity? Broadened
per coordinator nuance: what is the MINIMAL PER-CLASS representation (curve vs cells vs mask vs periodic),
does the UNION of geometry-matched carriers compress, and where is the irreducible residual?

**Authority: `[macOS-MLX advisory]` geometric feasibility.** All fidelity is measured against the CACHED
SegNet argmax labels (`experiments/results/mlx_fleet_gt_cache/gt_n600.npz` `lstars`, real n600 cache,
numpy load — NO scorer forward, NO model inference). NOT through R + the frozen SegNet, NOT byte-closed.
A candidate/bound, not a score. **Pointer 0.19108282 UNMOVED.**

**Apparatus:** `src/tac/boundary_math/partition_collapse.py` + CLI
`tools/probe_partition_collapse_feasibility.py`; 18 tests incl. a synthetic KNOWN-power-diagram recovery
control (greedy + CE refine recovers a random K=12 diagram at ≥0.98; weights measurably beat weight-blind
Voronoi). Artifact: `experiments/results/partition_collapse_probe_20260710/probe.json` (+ `run.log`).
Global sweep: 12 frames (stride 50); per-class + hybrid: 24 frames (stride 25); class order verified by
spatial/static SIGNATURE (MyCar temporal IoU 0.992 bottom-static, Undriv top, Lane 0.59% area — canonical
`0=Road 1=Lane 2=Undriv 3=Movable 4=MyCar`, never luma-sorted).

## 1. Global Laguerre K sweep (one power diagram for everything) — BOUNDED

Greedy error-driven insertion + anchored Lloyd, then tau-relaxed softmax-CE refinement of sites+weights
(the tau-anneal tropical structure). Mean over 12 frames; bytes = raw 29 bits/generator (9+9 site, 8
weight, 3 class), per-frame independent:

| K | B/frame | refined bulk | refined band | note |
|---|---|---|---|---|
| 8 | 29 | 0.8370 | 0.4729 | |
| 19 | 69 | 0.9377 | 0.6486 | |
| 35 | 128 | 0.9548 | 0.7121 | |
| 70 | 253 | 0.9813 | 0.7942 | knee |
| 138 | 498 | 0.9857 | 0.8276 | |
| 279 | 1010 | 0.9884 | 0.8479 | flattening |
| 542 | 1965 | 0.9890 | 0.8584 | |
| 1024 | 3712 | 0.9927 | 0.8859 | residual 7.3e-3 |

- **The Laguerre WEIGHT is decisively real:** unrefined (weight-blind Voronoi) K=1024 bulk is 0.9048;
  CE-refined (weights+positions) 0.9927. On the synthetic control: 0.917 -> 0.961 (weights only) ->
  0.996 (weights+positions). The argmax IS power-diagram-shaped — #284's structural claim holds.
- **But the curve FLATTENS at ~1e-2 residual:** 7.4x more bytes (K 138->1024) buys only 0.9857->0.9927.
  The tail the cells cannot chase is the boundary-band jitter (band agreement stalls at 0.89). A pure
  few-generator diagram never reaches d_seg-relevant fidelity (~1e-3-5e-3 disagreement) at sane K.

## 2. Per-class geometry-matched carriers (the v8 hybrid) — the measured carrier table

| Class | carrier | B/frame (raw) | fidelity (mean, 24f) | verdict |
|---|---|---|---|---|
| MyCar | static temporal-majority mask (zlib packbits) | **0.21** (125 B TOTAL /600) | IoU 0.9934 (min 0.9890) | NEAR-FREE, confirms #139 + SPEC 0.1-0.5KB |
| Movable | moment ellipses, ~3.0 islands/frame | **10.5** | recall 0.928, IoU 0.817 | cheap; SPEC 2-6KB/600 = measured 6.3KB |
| Lane | poly curve (deg<=3) + width band, ~7.5 curves/frame | **97** (runs) / 68 (solid) | recall 0.835, precision 0.675, IoU 0.601 | the WEAK class — see §4 |
| Road+Undriv | ONE small-K power diagram on the inpainted 2-class base (ru_k=32 + CE refine) | **116** | Road recall 0.981 / Undriv 0.998 | ru_k knee is shallow: hybrid bulk 0.9925/0.9934/0.9940 at ru_k 8/16/64 |
| **UNION hybrid** | compose (base -> MyCar -> Movable -> Lane) | **229** | **bulk 0.9921, band 0.8766, residual 7.9e-3** | = the K=1024 global diagram's fidelity at **16x fewer bytes** |

Comparators (same frames): contour-codec floor (~1.25 b/crack-px, #307) = **419 B/frame** (2681 edge
px/frame); zlib label map = **994 B/frame**. The per-class hybrid is **1.8x under the store-the-boundaries
floor** and 4.3x under zlib — parsimony favors generators/curves over boundaries, PROVIDED each class gets
its geometry-matched form. Largest contour pair: **Road-Lane 1610 px/frame** (60% of all crack edges) —
the lane boundary IS the boundary budget.

## 3. Dashes (measured, 49 dashed-curve instances)

Ego-distance periodic model (phase in u = 1/(row-horizon), #215 dash-phase=ego-distance) agreement
**0.937** vs image-row periodic **0.911** — the ego-distance chart measurably wins. But per-run endpoint
coding costs only ~8.3 B/curve anyway, so the periodic model's byte advantage is marginal at this scale;
its real value is TEMPORAL (phase advances with the screw xi -> near-free across frames, unmeasured here).

## 4. Where the negative lives (honest bounds)

1. **Pure global Laguerre is BOUNDED:** never reaches d_seg-relevant fidelity; flattens at ~7e-3
   residual by K~300 (1KB/frame). "Store a few generators for the whole partition" is REFUTED as a
   sufficient carrier; it IS a good coarse skeleton (0.98 at 70 generators/253 B).
2. **Thin Lane is the irreducible tail in ANY of the tested forms:** as cells it consumes generators
   without converging (global-fit Lane recall 0.912 even at K=1024); as a per-frame image-space curve it
   costs 97 B/frame at only 0.835 recall / 0.601 IoU. Road-Lane = 60% of all boundary pixels. Matches
   L17/L65 (lane long-tail is THE binding residual). The SPEC's 1-2KB lane band is reachable only via the
   GROUND-FRAME factorization (few ground-plane curves + xi transport, L-v8 horizon-poly+xi), not
   per-frame image-space fits (58KB/600 raw).
3. **The hybrid's 7.9e-3 residual is ABOVE the current realized witness d_seg (~4.4e-3):** the hybrid is
   a RATE SKELETON + prior, NOT a sufficient standalone carrier. The residual is boundary-annulus jitter
   (band agreement 0.877; L66: ~97% of d_seg lives in the ~4.7%-area annulus) — the SPEC's "spend bytes
   on SEPARATRIX/ANNULUS PRECISION" is exactly the remaining cost, owned by the INR-annulus/trainer.
4. **Naive temporal coding of the partition does NOT pay:** consecutive frames differ in only 1.07% of
   pixels, but pixel-level delta zlib is 1553 B/frame vs 1071 solo (**0.69x — worse**): the changed
   pixels are scattered annulus jitter (high spatial entropy). Temporal compression must be PARAMETRIC
   (generator/curve tracking, ground-frame + xi), not pixel-delta.

## 5. Verdict (NO-FAKE, both halves)

**FEASIBLE — for the per-class HYBRID skeleton, and it dominates:** bulk 0.9921 / band 0.877 at
**229 B/frame raw-independent (137 KB/600, rate 0.092)** — 16x cheaper than the single power diagram at
matched fidelity, 1.8x under the contour floor. Cheap classes MEASURED: MyCar 125 B total, Movable
10.5 B/frame, Undriv near-perfect inside the shared RU field. This directly supports v8 SPEC §2's
heterogeneous carrier table (MyCar + Movable measured INSIDE the SPEC's ranges; Road/Undriv 70KB/600 raw
lands just above the 20-50KB unknown, temporal-parametric coding owed).

**BOUNDED — for the literal "few Laguerre generators" reading of #284:** the partition is
power-diagram-STRUCTURED (weights decisively beat Voronoi) but a few-generator diagram alone saturates at
~7e-3 residual, above d_seg-relevant fidelity; the thin Lane + annulus jitter are the irreducible tail no
tested collapse technique reaches — that tail is the annulus-precision budget (trainer/INR), not a
generator/curve/mask/periodic carrier.

**Path to a real row (not run here):** ground-frame lane factorization + xi transport for the 97->~3
B/frame lane gap; parametric temporal tracking for RU generators; then byte-close the composed carrier
and measure through `upstream/evaluate.py` per v8 SPEC §6 gates (v7.5-first gating unchanged).

**Triality:** DAG FEED appended (`FEED-lag` 2026-07-10). Equations leg: NO new canonical equation — the
measured curve is an anchor CANDIDATE for the SPEC §5 council-flagged "per-class carrier allocation" law,
which stays council-flagged until P-B/P-C/increment-1 per the SPEC; this probe is a pre-anchor, not the
n600-through-R anchor those require. DSL leg N/A (analysis probe, not a trainer lever).
