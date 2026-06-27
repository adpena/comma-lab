# Morse-Smale dynamical-system partition codec — $0 feasibility (DAG FEED-cg)

Date: 2026-06-26. Author arm: ms-codec-feasibility (sandbox, $0 CPU, NO GPU).
Pointer UNMOVED 0.19110. Means != ends. NO-FAKE: all d_seg below are MEASURED
reconstruction d_seg (rasterized-skeleton partition vs frozen-SegNet GT argmax),
partition-vs-partition. This is the RATE-half best-case (LOWER bound on realized
d_seg; realized-through-R needs the witness/renderer + SegNet forward — NOT run).

GT partition: `experiments/results/indep_dseg_bets_20260623_inflated/seg_argmaps.npz`
`gt (600,384,512) uint8` = exact frozen CPU-torch SegNet argmax over all 600
last-frames (the same authority the witness trainer uses). Classes 0..4; class 2
(road) 49.5%, 0 25.4%, 4 23.3%, rare 1 0.58% + 3 1.26% (lane-marking long-tail).

## Method
Boundary complex per frame via cv2: connected-components per class (skimage.label
connectivity-1) -> outer contour (findContours) -> Douglas-Peucker (approxPolyDP,
eps) = arcs as piecewise-linear (poly order 1 at these eps). Raster by area-desc
fillPoly (nested-partition correct). Byte est = per-region(label 2.32b + start 18b)
+ delta-coded vertex stream at H0 entropy. (= watershed/MS-complex on the argmax;
arXiv 2406.09423 MSz uses MS topology for lossy compression; MPEG-4 polygon+chain
+arithmetic shape coding is the rate model.)

## Per-frame INDEPENDENT RD (measured, sample 40)
S = 100*d_seg + 25*(B*600/37545489) + 0.0184(pose)
| eps | d_seg | B/fr | KB tot | S_seg | S_rate | S |
|----|-------|------|--------|-------|--------|---|
|0.5 | 5.57e-4 | 740 | 444 | 0.056 | 0.296 | **0.370** |
|0.6 | 1.02e-3(≈goal)| 647 | 388 | 0.102 | 0.259 | 0.379 |
|1.0 | 3.37e-3 | 262 | 157 | 0.337 | 0.105 | 0.460 |
|2.0 | 7.50e-3 | 129 | 78 | 0.750 | 0.052 | 0.820 |
RD-optimal indep point = eps0.5, S~0.37. Even with 1.5-2x context-arithmetic
coding -> ~0.22-0.27. DOMINATED by pointer 0.19110. **Rate is the wall.**
Note: eps0.5 hits d_seg 5.57e-4 (BELOW capstone 7.2e-4) — partition IS geometrically
specifiable at capstone fidelity; the cost is pure RATE.

## Intrinsic complexity / bifurcations (measured)
~15-20 regions/frame; ~840 vertices/frame @ capstone d_seg; ~4.5 region
births/deaths per consecutive-frame step (~2700 topological events / 600 frames
— bifurcations are MODERATE, not rare; driven by rare-class flicker).

## Temporal coding — DOES NOT COLLAPSE THE RATE (the decisive negative)
1. DP-vertex delta (consecutive, IoU+nearest): TEMPORAL est 1026 B/fr > 740 indep.
   DP vertices are TEMPORALLY INCOHERENT (slide along arcs as shape drifts) ->
   only 80.8% match, 170 "new" verts/fr. Naive skeleton-delta FAILS.
2. Global motion-comp ceiling (free pose sidecar = best integer shift of gt[i-1]):
   inter-frame d_seg 1.254e-2 -> 1.212e-2 = **only 3.3% reduction**. Partition
   change is HIGH-FREQUENCY boundary jitter, NOT coherent ego-drift.
3. Per-class: rare classes 1&3 drive temporal jitter (class-0/1 boundaries = 74%
   of changed pixels); big-regions-only {0,2,4} inter d_seg 4.67e-3 (better, still
   few-e-3); dropping 1&3 costs d_seg 1.82e-2 (un-droppable long-tail).
=> The "junctions drift smoothly, AR-codeable, rate collapses" premise is
   CONTRADICTED. Drift is NOT subset of pose (3.3%). Junction-stable tracking
   would help big regions only; the binding long-tail (lane markings) flickers.

## Verdict: REVISE (standalone NOT competitive; salvage as instrument + residual)
Decisive question "does skeleton+residual beat the level-set witness rate at equal
d_seg?" = MEASURED NO standalone (444KB indep @ 5.57e-4 vs PR95/witness ~177KB @
6e-4; witness amortizes smooth structure into shared FREE generator weights, MS
re-specs every frame). Genuine value: (a) MEASUREMENT INSTRUMENT (partition
intrinsic complexity + bifurcation rate + existence proof d_seg 5.57e-4 is a pure
geometric spec = a witness TARGET/lower bound); (b) HARD-TAIL RESIDUAL layer on the
witness (witness carries smooth road/sky boundaries via INR; MS codes only the
sparse rare-class lane-marking arcs = the doctrine's "sparse hard-pixel sidecar");
(c) bifurcation events as a sparse-event sidecar. NOT the primary rate carrier.
Composition correction: clamp(sky+hood) HELPS (removes 2 largest stable regions
from per-frame budget); pose does NOT subsume drift (3.3%).
Next: if pursued, build TRUE junction-tracker (stable critical-point parametrization,
not DP) + measure junction-only temporal rate on big regions; else fold the
hard-tail-residual idea into the witness sidecar and drop the standalone codec.
