# MSDF lane carrier — does multi-channel SDF recover the sharp-corner lane holdout? ($0 falsifier)

**UTC:** 20260629T193309Z · **git:** e829df3af · **evidence grade:** `[macOS research-signal]`
(SegNet-FREE resample isolation; advisory) · **score_claim:** false · **promotable:** false ·
**pointer:** contest-CPU **0.19110 UNMOVED**. **Tool:** `tools/r_survival_probe.py` (extended, `--msdf`) ·
**data:** `.omx/research/msdf_lane_carrier_probe_n96.json` (n=96) · **GT:**
`experiments/results/mlx_fleet_gt_cache/gt_n96.npz` (`lstars`).

> means→ends: this is the **second v2 $0 falsifier** from agent a1da84c's Task-2 draw-from
> (`.omx/research/graphics_aa_astronomy_inverse_codec_crosscheck_20260629T184900Z.md`), attacking F1's
> named **last holdout** (`.omx/research/r_survival_physics_20260629T182659Z.md` §4.4): the single-SDF lane
> carrier leaves **3.19% lane-flip @ render-192** while every other class is <0.2%. The hypothesis: MSDF
> (Chlumský 2018, median-of-3 edge-colored distance channels) recovers the dash-end **corners** a single
> SDF rounds. **Verdict: FALSIFIED.** MSDF does NOT help — it HURTS the lane ~3.6× — because the holdout is
> a **thinness/Nyquist** problem, not a corner-rounding problem. Nothing here is a contest score.

---

## 0. TL;DR (the verdict)

| render | hard lane% | **single-SDF lane%** (F1) | **MSDF lane%** | hard tot d_seg | single-SDF tot | MSDF tot |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 25.91 | **3.19** | **11.61** | 0.00630 | **0.00059** | 0.00481 |
| 320 | 25.64 | **0.04** | **3.05** | 0.00616 | **0.00001** | 0.00429 |

(n=96, ramp slope 48/px ≈ 2.6px half-width = F1's best @192; MSDF band 12px.)

- **MSDF LOSES to single-SDF** by ~3.6× @192 (11.61 vs 3.19) and ~76× @320 (3.05 vs 0.04). It beats a
  hard bitmap (25.9%) but is dominated by the plain 1-Lipschitz SDF.
- **single-SDF total d_seg already clears the named ≤1.23e-3 threshold** (5.9e-4 @192, 1e-5 @320); **MSDF
  does NOT** (4.81e-3 @192, 4.29e-3 @320 — ABOVE the threshold). MSDF widens the gap, it does not close it.
- **Root cause (measured decomposition):** **0.0% of single-SDF's lane flips are "corner-only"** (a sharp
  corner in an adequately-resolved region) — at BOTH render-192 and render-320. **100% are thin** (the lane
  is ~2px = sub-Nyquist at render ≤ 320). There is **no isolated corner-rounding residual for MSDF to fix**;
  the dash-end corners that flip are *also* thin, and MSDF cannot add resolution to a sub-Nyquist feature.
- **Mechanism (deep-math):** MSDF's 3 pseudo-distance channels are **NOT 1-Lipschitz** (medial-axis
  discontinuities) and are **poorly conditioned across a 2px width**, so they (i) survive the bicubic R
  round-trip WORSE than the single smooth 1-Lipschitz ramp (defeating F1 §1.3, the very reason the SDF
  carrier wins R) and (ii) cannot place 3 distinct edge-colored channels meaningfully inside a 2px ribbon.
  MSDF trades corner-sharpness (which the lane does NOT need) for boundary fidelity (which the thin lane
  critically needs).
- **The lever for the holdout is RENDER-RESOLUTION**, not corner-coloring — exactly F1 §4: single-SDF
  @render-320 → 0.04% lane (1e-5 total). MSDF is the wrong tool for a thinness-bound residual.

---

## 1. The MSDF implementation is REAL (NO-FAKE synthetic-corner validation — PASS)

Before trusting any lane number, the from-scratch MSDF port is validated on synthetic shapes with **known
sharp convex corners** (rotated square = 90°, thin triangle ≈ 25°): coarsen geometry → magnify back →
threshold → compare corner-region error of single-SDF vs MSDF.

| shape | corner-err single-SDF | corner-err MSDF | global IoU single→MSDF | verdict |
|---|---:|---:|---:|---|
| square (90°) | 14.29% | **2.04%** | 0.990 → 0.988 | **PASS** |
| triangle (≈25°) | 14.29% | **6.80%** | 0.943 → 0.973 | **PASS** |

`ALL_PASS=True`. MSDF genuinely recovers sharp corners single-SDF rounds (2–7× lower corner error) **with
no global shape degradation** (IoU preserved/improved). So the lane negative below is a property of the
LANE, not a broken implementation.

**Implementation provenance (faithful msdfgen port; Chlumský, CGF 37(1) 2018; `github.com/Chlumsky/msdfgen`):**
- `_edge_coloring_simple` — port of `edgeColoringSimple` (`switchColor` multi-corner branch): 2-colors the
  contour so the two edges meeting at a sharp corner share exactly ONE channel.
- `_segment_signed_distances` — port of `LinearSegment::signedDistance` + `distanceToPseudoDistance`
  (clamped true distance for nearest-edge SELECTION; infinite-line pseudo-distance for the VALUE only where
  the foot falls beyond the segment, i.e. at corners) + the **`dot` (orthogonality) tie-breaker**.
- `median(R,G,B)` decode. Lane→other classes composed via K-class argmax (Task-2 option (a)).

**Bug found + fixed during the build (recorded for NO-FAKE):** the first port set each channel's value to
the raw infinite-line ortho ALWAYS → a convex-corner **false-inside extension cone** (triangle field IoU
collapsed to 0.476 even with NO resampling). Root cause: at a sharp vertex the two edges have *exactly*
equal endpoint distance (the shared corner point), so `|distance|`-only selection picked the wrong-signed
line extension. The fix is msdfgen's lexicographic `(|distance|, dot)` ordering — at a tie the
more-perpendicular edge wins, which is the geometrically-correct sharp edge. Post-fix: `outside_pos=0`,
field IoU 0.95–0.99 (residual = the <1px raster-vs-continuous boundary band, identical for single-SDF).

---

## 2. The lane A/B/C is robust (adversarial audit — MSDF loses at every operating point)

Per the not-pessimistic rule, the MSDF-hurts negative was adversarially audited (n=8): MSDF at its OWN
optimal ramp slope, and median-BEFORE-R vs AFTER-R to attribute the loss.

| render | slope | single-SDF lane% | MSDF (median after R) | MSDF (median before R) |
|---:|---:|---:|---:|---:|
| 192 | 48 | 3.68 | 11.49 | 13.50 |
| 192 | 24 | 3.46 | 10.80 | 12.67 |
| 192 | 12 | 3.43 | 10.65 | 12.57 |
| 320 | 48 | 0.09 | 2.88 | 2.90 |
| 320 | 24 | 0.10 | 2.84 | 2.90 |
| 320 | 12 | 0.09 | 2.84 | 2.93 |

- MSDF loses ~3× across **every** slope and both renders → **not a tunable artifact**.
- median-before-R is **no better** (slightly worse) → the loss is **not only** the 3-channel R-survival; the
  median-of-pseudo-distance field is itself a worse THIN-lane carrier than the single Euclidean SDF, even
  before R. Both mechanisms (non-1-Lipschitz R-survival + 2px-width conditioning) are real and compound.

---

## 3. Residual decomposition — corner vs thin (the implementation-robust core finding)

Of the **single-SDF** lane flips through R (the residual MSDF would have to fix):

| render | # lane flips | corner-only | thin-only | both (corner∩thin) | neither | thin-any | corner-any |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 192 | 3431 | **0.0%** | 46.3% | 53.7% | 0.0% | 100% | 53.7% |
| 320 | 39 | **0.0%** | 15.4% | 84.6% | 0.0% | 100% | 84.6% |

(corner = within 3px of a sharp lane-contour vertex at 384, angle>30°; thin = local width·(render/384) ≤ 2px.)

**The decisive number: 0.0% corner-only flips at both render resolutions.** Every lane flip is in a thin
(sub-Nyquist) region; the dash-end corners that flip are *also* thin (the lane is uniformly ~2px). MSDF's
entire value proposition — sharpening corners in adequately-resolved regions — has **no addressable target**
in this residual. This is implementation-independent (it characterizes the single-SDF residual, not MSDF).
It is the mechanistic explanation of §0/§2: the holdout is a **Nyquist/thinness** wall, and the lever is
render-resolution (or a dedicated higher-res lane sub-channel), exactly F1 §4.4.

Lane geometry (n=96, grounds the thinness): width median 2.0px, mean 2.38px, p90 4.0px, 75.9% ≤2px, area
0.59%.

---

## 4. Byte cost of MSDF (rule-118 accounting)

Lane contour measured: **~64 vertices/frame** (median 64, max 82), ~63 sharp corners/frame.

- **`[FREE]`** the MSDF GENERATION (edge 2-coloring + per-channel pseudo-distance + median) is a
  deterministic geometric algorithm, legal to expand inside `inflate.py` (≪100 LOC, numpy-expressible).
- **`[COUNTED]`** the stored lane **contour** (~64 verts × 2 coords/frame; ~hundreds of bytes/frame raw,
  less after delta + AR coding). **This is the SAME contour single-SDF stores** — the 3-channel
  decomposition is a deterministic function of the contour computed at decode.
  - **contour-stored model:** MSDF adds **0 extra COUNTED bytes** vs single-SDF (same descriptor).
  - **field-stored model:** MSDF = 3 lane carrier channels vs 1 → **3× the lane channel** (other 4 classes
    unchanged).
- **`NOT FORBIDDEN`** as long as the descriptor is an honest compact contour, not a per-frame argmax table
  smuggled as "code."

**Net:** the byte cost is moot — MSDF does not help d_seg at any byte model, so there is no rate↔distortion
arm to spend on. The contour-stored insight (MSDF = byte-free richer decode) is banked for any FUTURE
carrier where corner-rounding IS the binding residual (not the lane).

---

## 5. Verdict + what it changes for v2

**MSDF is NOT a carrier win for the lane dash-end holdout. FALSIFIED at $0 (advisory).** Reasons, all
measured: (1) 0% corner-only residual — no corner-rounding for MSDF to fix; (2) MSDF's non-1-Lipschitz
multi-channel field survives the bicubic R worse than the single 1-Lipschitz SDF (F1 §1.3 mechanism); (3)
3 edge-colored channels are ill-conditioned across a 2px ribbon. The holdout is **render-resolution-bound**.

- **KEEP** the single 1-Lipschitz SDF lane carrier (F1's spec is confirmed, not replaced).
- **The lever is render-res / a dedicated higher-res lane sub-channel**, not corner-coloring: single-SDF
  @render-320 already drives the lane to 0.04% (total 1e-5, below the ≤1.23e-3 threshold). The v2 survival
  term should spend capacity on the lane's effective render-resolution, per F1 §4.4 item 4 — but as a
  **higher-res single SDF**, NOT an MSDF.
- **Do NOT** route v2 lane bytes into a 3-channel MSDF.
- **Bank** the validated MSDF generator + the contour-stored "MSDF is byte-free" insight for any FUTURE
  residual where corner-rounding (not thinness) is the binding wall.

This narrows the v2 design space cleanly: of a1da84c's Task-2 draw-from, the corner half is **closed
(negative)**; the render-resolution half of F1 §4.4 is the live lever.

---

## 6. Honest caveats / NO-FAKE

- **SegNet-FREE** isolation (sub-walls A+B of F1): the partition is modeled as `argmax` of the carrier; the
  (C) SegNet-RGB-reading wall is separate (the trainer's realized-through-R `cpu_verdict_d_seg`). **No
  contest score; pointer 0.19110 unmoved.** All numbers are advisory `[macOS research-signal]`.
- **The MSDF impl is validated** (synthetic corner test PASS, field IoU restored), but it is a from-scratch
  numpy port (linear segments only; cv2-extracted raster contours via `approxPolyDP`, not a true vector
  outline). A production msdfgen on a vector lane outline could differ in absolute corner-err magnitude —
  but **the lane verdict does not depend on MSDF fidelity**: the decomposition (0% corner-only) and the
  direct A/B/C (MSDF measured 3× worse, validated impl) are independent confirmations.
- **The ≤1.23e-3 threshold** is taken from the dispatch brief as the lane-survival target; I report the
  measured totals relative to it (single-SDF clears it, MSDF does not) and flag the exact provenance.
- **"thin" threshold** (width·scale ≤ 2px) is generous because the lane is uniformly ~2px → thin-any=100%;
  the load-bearing statistic is **corner-only = 0%** (no corner residual separable from thinness) plus the
  direct A/B/C measurement, not the thin fraction.
- Reproduce: `tools/r_survival_probe.py --n 96 --msdf --msdf-render-res 192,320 --render-res 384 --slopes 48
  --out .omx/research/msdf_lane_carrier_probe_n96.json` (synthetic validation + A/B/C + decomposition). The
  slope-sweep / before-after-R audit (§2) is the documented n=8 snippet over the same `_argmax_through_R_*`
  helpers.

## 7. Wire-in hooks (Catalog #125)
1. sensitivity-map: ACTIVE — "corner-rounding" is now a MEASURED-zero per-axis sensitivity row for the lane
   (the residual is thinness, not corners). 2. Pareto: ACTIVE (negative arm) — MSDF lane bytes ↔ lane-flip
   is a DOMINATED arm (MSDF worse at equal/3× bytes); render-res ↔ lane-flip remains the live arm. 3.
   bit-allocator: ACTIVE — "spend lane capacity on render-res of a single SDF, NOT on a 3-channel MSDF."
   4. cathedral autopilot: N/A (research probe). 5. continual-learning: this memo + JSON + the DAG FEED.
   6. probe-disambiguator: `tools/r_survival_probe.py --msdf` IS the disambiguator between the corner
   hypothesis (falsified) and the thinness/render-res hypothesis (confirmed).

## Primary citations
- Chlumský, V. 2018. *Improved Corners with Multi-Channel Signed Distance Fields.* Computer Graphics Forum
  37(1). DOI 10.1111/cgf.13265. OSS: github.com/Chlumsky/msdfgen.
- Green, C. 2007. *Improved Alpha-Tested Magnification for Vector Textures and Special Effects.* ACM
  SIGGRAPH 2007 Courses 9–18 (the founding SDF-text result; F1's existence proof).
- Prior in-tree: `.omx/research/r_survival_physics_20260629T182659Z.md` (F1; single-SDF 3.19% @192 anchor)
  + `.omx/research/graphics_aa_astronomy_inverse_codec_crosscheck_20260629T184900Z.md` (a1da84c Task-2
  draw-from this probe falsifies).
