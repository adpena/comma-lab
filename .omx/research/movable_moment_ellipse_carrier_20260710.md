# Movable-class d_seg carrier — moment-ellipse coding: DESIGN + cached-argmax measurement + byte budget (2026-07-10)

**Task:** the third leg of the v8 per-class routing. The Morse-Smale anisotropy map
(`partition_anisotropy_map_20260710.md`, commit `814fb1aac`) routed **Movable (class 3 = cars)** to
"moment-ellipse" because it measured Road↔Movable at LOW d_H (2.55) and Undrivable↔Movable at 2.86 =
isotropic/compact blobs, where the all-class directional-Fourier basis has LOW leverage. This unit
DESIGNS + MEASURES that carrier on cached argmax and answers: **does moment-ellipse beat direct
boundary-contour coding for Movable?**

**Authority: `[macOS-CPU advisory · NON-PROMOTABLE]`.** $0, MPS-NEVER, no SegNet/PoseNet forward — CACHED
argmax only (`gt_n600.npz` `lstars`, all 600 pairs, + `gt_n96.npz` cross-check). This designs a rate
carrier; it moves NO score. **Pointer 0.19108282 UNMOVED.** Provenance: git `fcfc02309`; class order
SELF-DETECTED (`3=Movable`, MEASURED area 1.24% / mid-band / IoU-per-anisotropy-map 0.90), never
luma-sorted. Apparatus (scratch, foreground): `movable_ellipse_measure.py` + `movable_residual_measure.py`
reusing `tac.boundary_math.movable_site_coder` (`scipy.ndimage.label`, the existing box site-coder).
**Verdict scope: FORMULATION** — this specific moment-ellipse carrier formulation at cached-argmax
fidelity; NOT a byte-closed exact row, NOT a family kill.

---

## 1. Measured Movable blob statistics (n600; n96 agrees)

| statistic | n600 | n96 |
|---|---|---|
| blobs / frame (mean) | 3.58 | 3.06 |
| blobs / frame (max) | 9 | 5 |
| frames with ≥1 Movable | 600/600 | 96/96 |
| total blobs | 2145 | 294 |
| blob area px (median) | 107 | 62 |
| blob area px (mean) | 681 | 999 |
| blob area px (p90) | 2241 | 3104 |
| aspect √(λmax/λmin) (median) | 1.82 | 1.87 |
| aspect (p90) | 3.80 | 3.88 |
| Movable area frac of frame | 1.24% | 1.56% |

**Reads:** Movable is SPARSE (3–4 compact blobs/frame) and MILDLY elongated (aspect median 1.82 — a car
seen from behind/side is a squat rectangle, not a line). The heavy-tailed area (median 107, mean 681)
= a few near/large vehicles among many distant small ones. This confirms the anisotropy map's
"compact/isotropic" routing: the blobs are NOT directional curves (unlike lane/horizon), so the
factorization/SPD-cone treatment that pays on the lane has little to grip here.

## 2. The moment-ellipse carrier design

Per Movable blob, fit the **second-moment equal-area ellipse**: centroid μ=(cx,cy), 2×2 covariance Σ from
the blob's pixel coordinates, eigendecompose → orientation θ (major-axis angle) + semi-axes (a,b) scaled
so π·a·b = blob area (area-preserving). Store **5 params/blob** `(cx, cy, a, b, θ)` — one more than the
existing axis-aligned box `(cx,cy,bw,bh)`, buying orientation + aspect. Coder: quantize (½-px position/axis,
θ to 128 levels over π), per-column zigzag-delta, zlib-9 (the v8 byte-close coder family). Presence =
blobs-per-frame count stream. The ellipse GENERATOR (rasterize a rotated ellipse) is rule-118 FREE; only
the fitted coords are counted. REUSES the `movable_site_coder` extraction + LAP-tracking scaffold.

**Shape-fit quality (per-blob IoU vs true component):**

| | ellipse | axis-aligned box |
|---|---|---|
| IoU mean (n600) | **0.852** | 0.742 |
| IoU median (n600) | **0.862** | 0.758 |

The moment-ellipse is a materially better shape descriptor than the box — orientation + aspect capture the
squat-rotated-rectangle silhouette the axis-aligned box cannot.

## 3. Flip structure (which Movable pixels are hard)

Using the cached `margins` field: of the flip-prone (bottom-10%-margin) Movable pixels, **72.3% sit on the
blob EDGE** (1-px erosion boundary) vs 27.7% interior (n600; n96 = 72.1%). The hard d_seg mass is at the
silhouette boundary — codeable in principle in an ellipse-relative frame (the residual thesis). This is the
same edge-concentration that motivates a boundary residual coder.

## 4. Byte budget vs intrinsic distortion — the load-bearing tradeoff (n600)

Intrinsic distortion = symmetric-difference(reconstructed Movable mask, true Movable mask) / total pixels =
the argmax-disagreement (d_seg) the carrier leaves if it is the sole Movable renderer. Score arithmetic:
seg-contribution = 100·d_seg, rate-contribution = 25·bytes/37,545,489.

| carrier | bytes | B/frame | rate | residual d_seg | 100·d_seg | **TOTAL movable cost** |
|---|---:|---:|---:|---:|---:|---:|
| Axis-aligned box (existing `site_coder`) | 11 798 | 19.7 | 0.00786 | 0.004229 | 0.4229 | 0.4308 |
| **Moment-ellipse ONLY** | 14 558 | 24.3 | 0.00969 | 0.002193 | 0.2193 | 0.2290 |
| Moment-ellipse + 24-angle radial residual | 37 684 | 62.8 | 0.02509 | 0.001395 | 0.1395 | 0.1646 |
| **Lossless boundary contour** (chain-code 1.5 b/px) | 28 227 | 47.0 | 0.01880 | 0.000000 | 0.0000 | **0.0188** |

**Reads:**
1. **Ellipse Pareto-dominates the box** on the lossy frontier: it HALVES the intrinsic d_seg (0.00219 vs
   0.00423) for +23% bytes (+4.6 B/frame). If a lossy Movable operating point were ever wanted, the ellipse
   is the strictly better parametric prior than the axis-aligned box.
2. **BUT the score charges 100× per unit d_seg**, so any lossy Movable carrier is CRUSHED by its distortion
   term. The ellipse-only "total movable cost" 0.229 is 12× the lossless contour's 0.019 — because 0.00219
   residual d_seg = 0.219 seg-score, larger than the entire pointer gap we are chasing. Movable is only
   ~1.2% of the frame but carries ~12% of the d_seg boundary mass; a lossy movable carrier would REGRESS the
   ~0.0047 baseline d_seg by ~half. **Movable must be coded near-losslessly.**
3. **The ellipse+residual does NOT reach lossless and is DOMINATED by direct contour.** The 24-angle radial
   residual (r(θ) − r_ellipse(θ)) only closes d_seg to 0.00140 at 37 684 bytes — MORE bytes than lossless
   contour (28 227) AND still lossy. Root cause: car silhouettes are compact but **NOT star-convex**
   (occlusion, overlapping vehicles, wheels/mirrors, blobs broken by the horizon) — a single-valued radial
   profile around the centroid cannot represent them, so the residual coder wastes bytes fighting
   non-star-convexity. The ellipse prior does not make the contour cheaper because the contour's own
   chain-code entropy (~1.5 b/px) is already near-floor.

## 5. VERDICT — moment-ellipse does NOT beat boundary-contour for Movable

**For the v8 Movable rate carrier, LOSSLESS boundary-contour coding (~47 B/frame, rate 0.0188, d_seg 0)
DOMINATES the moment-ellipse at every operating point.** This REFINES (does not contradict) the anisotropy
map's routing: the map correctly identified Movable as the low-d_H/isotropic class where directional
FACTORIZATION has no leverage — and it named "moment-ellipse / site-coder" as the family. The measurement
sharpens it: the moment-ellipse is the right *shape/orientation descriptor* (and a strict Pareto-improvement
over the existing axis-aligned box site-coder), but it is the WRONG *carrier*, because (a) Movable's
distortion sensitivity (100×) forbids any lossy approximation, and (b) compact-but-non-star-convex car
silhouettes are not smooth-parametric, so neither the ellipse nor an ellipse-relative residual reaches
lossless at competitive rate. **Recommendation: code Movable as a lossless per-frame boundary contour**
(the `contour_codec` chain-code family), NOT a moment-ellipse.

**Where the moment-ellipse DOES earn its keep (secondary, keep in the toolbox):**
- **Better low-rate fallback** than the axis-aligned box IF an ultra-low-rate degraded mode is ever forced
  (halves the box's distortion for +4.6 B/frame).
- **Shape/orientation descriptor for TEMPORAL prediction** — the real Movable headroom is not per-frame
  shape but per-frame REDUNDANCY: cars move smoothly, so the 47 B/frame contour should temporally-delta
  against the previous frame's contour warped by the site's tracked (μ,Σ,θ) ellipse motion + ego-screw ξ.
  The ellipse is the natural low-dim state for that predictor (μ̇ velocity, Σ̇ scale-change as the car
  approaches). This is the next Movable unit; the ellipse feeds it as the tracking state, not as the mask.
- **De-share sibling:** `movable_deshare.py` already handles the horizon/lane double-count of car pixels;
  a lossless Movable contour is the clean owner-of-record for those straddle pixels.

## 6. Candidate equation (MEASURED, NOT registered)

`movable_moment_ellipse_bytes_vs_dseg_v1` — on cached n600 argmax, the Movable moment-ellipse carrier
Pareto-improves the axis-aligned box (IoU 0.85 vs 0.74; intrinsic d_seg 0.00219 vs 0.00423 at +23% bytes)
but is DOMINATED by lossless boundary contour (28 227 B, d_seg 0) at every score operating point, because
100·d_seg forbids lossy Movable and car silhouettes are non-star-convex. **MEASURED but NOT registered** —
needs a byte-closed carrier + through-R n600 to confirm the intrinsic-vs-realized distortion (same gate as
the SPD pose codec: cached-argmax intrinsic distortion is an UPPER bound on the through-R realized d_seg
because the witness co-renders). Registration owed pending the byte-closed contour vs ellipse row.

Cross-refs: `partition_anisotropy_map_20260710.md` (the routing input) · `movable_site_coder.py` (the box
baseline this beats on shape, loses to contour) · `movable_deshare.py` (de-share sibling) · `contour_codec.py`
(the recommended lossless carrier) · #139 hood static-region sibling · v8 SPEC per-class decomposition.
