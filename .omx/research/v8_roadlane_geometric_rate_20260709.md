# v8 Road↔Lane geometric rate — MEASURED (real machinery, $0, read-only)

**Date:** 2026-07-09 · **Scope:** the biggest owed v8 rate number — Road↔Lane, **59% of the whole-scene
bitmap budget** (0.204 S of 0.346; FEED-v8-ratebudget). `[no-triality]` · pointer 0.19110 UNMOVED ·
#205 untouched · MPS/GPU untouched (pure numpy on the label cache).

## STORES CONSULTED (recall, not re-derive)
- DAG **FEED-v8-realmachinery** (the horizon precedent: deg-3 poly 1.46px over 425/512 cols → 4 coeffs +
  ξ-delta → zlib **0.0032 S = 14.6×**; HONEST dominant-arc scope + `residual_sidecar_owed`). Method mirrored.
- DAG **FEED-v8-ratebudget** (0.346 de-shared bitmap; Road/Lane 0.204 row = the target; the ~0.014 S
  projection assumed horizon-class 14.6× transfer).
- DAG **FEED-v8-voronoi** (store PARSIMONIOUS GENERATORS not boundary; dense-medial≈bitmap was the measured
  negative; lane = centerline poly + width = its own medial axis).
- **Real machinery used (NOT a proxy, NOT a b/px estimate, NOT a generic chain coder):**
  - `tac.boundary_math.lane_sdf_component.{cluster_lane_lines, fit_lane_line, rasterize_lane_band}` —
    Wave-F #234 lane fit: BEV-cluster class-1 px → per-instance deg-3 centerline poly + deg-1 halfwidth +
    matched-filter dash model.
  - `tac.boundary_math.analytic_lane_render_band.{serialize_lane_band_rd_tracked, serialize_lane_band_rd,
    roundtrip_lines_through_rd_tracked}` — the **actual byte-close lane coder** (LBND2 grammar): fp16-class
    quantize → zigzag **temporal-delta** (row0 seed, rows>0 = Δ) → Hungarian coherent-slot correspondence
    (ξ tracking, kills slot-swap jitter losslessly) → **brotli q11**. Roundtrips bit-exact through the
    UNCHANGED inflate decode. This is the real BASIS + real coder the FEED-v8-realmachinery discipline demands.

## MEASUREMENT (all 600 frames, real gt_n600.npz['lstars'], comma10k order Road0/Lane1)

### Fit + coverage (MEASURED)
- **Lines/frame:** median 5.0, mean 4.95 (range 4–6). Real `cluster_lane_lines` output — no smooth-curve
  assumption; the actual per-frame instance count.
- **Fit residual:** median lane-px distance to reconstructed band edge = **1.00 px** (comparable to the
  horizon's 1.46px).
- **Road↔Lane boundary coverage (px within ≤2px of the reconstructed band boundary):** **72.5%**
  (lossless/no-smooth roundtrip). Lane-region recall (class-1 px inside band) = **63.3%**.
- **HONEST scope (NO-FAKE):** ~27.5% of the Road↔Lane boundary is NOT carried by the dominant lane-band
  generators → **`residual_sidecar_owed`** (fit misses + faint/occluded lane fragments). Same class of
  caveat as the horizon (which covered 83% and flagged the same). The 0.0275 S below is a
  **DOMINANT-STRUCTURE** number, not a complete Road↔Lane number.

### ξ temporal coherence (MEASURED — the honest lane finding)
Per-coeff quantized temporal-delta activity (nonzero-fraction; frozen≈0% ↔ ego-moving):

| coeff | nonzero-frac | median &#124;Δq&#124; |
|---|---|---|
| c3 (cubic) | 79.6% | 26 |
| c2 (quad)  | 81.5% | 64 |
| c1 (lin)   | 81.4% | 49 |
| c0 (offset)| 78.2% | 12 |
| hw1/hw0    | ~81% | ~28 |
| dp/dph/dd (dash) | 66–73% | 2–28 |
| f_lo/f_hi (range)| 55–62% | 4–16 |

**Contrast with the horizon:** the horizon's cubic/quadratic were FROZEN (|Δ|≈1e-7/6e-5, only the intercept
moved = ego pitch → 599/600 near-free). **Lanes are NOT frozen** — every coeff moves (55–82% nonzero),
curvature coeffs c2/c1 move the MOST. Lanes are genuinely dynamic (curvature changes as the car drives) +
multi-instance, so the ξ ego-warp lever is WEAKER for lanes than for the rigid horizon. **The 7.4× reduction
is PRIMARILY PARSIMONY** (≈5 lines × 11 coeffs vs a full boundary bitmap), only modestly helped by temporal
delta (159 KB raw → 41 KB brotli = 2.6×).

### Real-coder store (MEASURED)
- **Primary (lossless correspondence, `coherent_slot`, no smooth):** LBND2 raw 159,386 B → **brotli q11
  40,331 B = 40.33 KB @ n600 → S = 25·40331/37,545,489 = `0.02750`**.
- Sort-packed variant: 41,526 B (tracked wins).

### Temporal-denoise headroom (MEASURED — a LOSSY trade, NOT a free win)
Batch denoise in clean persistent-track space (`smooth=median/rts`) reaches **0.0176 S (11.6×)** —
near the ~0.014 projection — BUT **degrades coverage**: boundary-cover 72.5%→66.8%, lane-recall
63.3%→**48.0%**. Smoothing removes real per-frame lane variation, so it is a rate/coverage tradeoff,
NOT the horizon's near-free ego-freezing. **The honest measured geometric rate is the lossless 0.0275 S.**

## THE S-NUMBER TABLE (Road↔Lane)

| representation | S | vs bitmap |
|---|---|---|
| **bitmap** (de-shared, brotli, FEED-v8-ratebudget) | **0.2040** | 1.0× |
| **geometric — lossless** (centerline poly + ξ track + brotli; **MEASURED**) | **0.0275** | **7.4×** |
| geometric — median-smooth (lossy, coverage 72.5→66.8%) | 0.0176 | 11.6× |
| prior projection (horizon 14.6× transfer) | ~0.014 | 14.6× |

**The projection was optimistic.** Lanes don't inherit horizon-class ego-rigidity; the honest lossless
measured number is **7.4× (0.0275 S)**, still decisively inside the band and ≪ the 0.118 pointer rate term.
Road/Lane moves from "~owed" to **MEASURED** — 3 of the 5 whole-scene edges are now measured-geometric.

## Adversarial self-review (before commit)
1. **Coder real?** YES. LBND2 quantize + zigzag temporal-delta + Hungarian slot-track + brotli q11 — the
   actual submission byte-close coder, roundtrips bit-exact. No b/px proxy, no generic chain coder.
2. **Scope honest?** YES. 72.5% boundary coverage stated up front; 27.5% `residual_sidecar_owed`; the
   median-smooth coverage cost MEASURED and reported (not hidden to inflate the reduction).
3. **Coverage justify the claim?** YES. 7.4× is the cost to carry the dominant lane structure covering
   72.5% of the boundary — comparable-basis to the bitmap (both reproduce the lane region). Residual labeled owed.
4. **Numbers MEASURED not guessed?** All from real n600 argmax through the real fit + real coder. Verifier:
   re-run `serialize_lane_band_rd_tracked(pairs, LaneBandRenderConfig())` on `cluster_lane_lines`+`fit_lane_line`
   over `gt_n600.npz['lstars']`.

## Consequence for v8
Rate thesis HOLDS on its biggest owed edge: geometry crushes the bitmap 7.4× on the 59%-of-budget row.
Whole-scene geometric budget now: 3 edges measured (Road/Lane 0.0275 + horizon 0.0032 + hood 0.0202),
2 owed (Movable pairs → sparse object sites). Build the Road/Lane carrier on the PARAMETRIC centerline
generator (Wave-F #234), NOT the boundary bitmap. `[no-triality]` · pointer 0.19110 UNMOVED.
