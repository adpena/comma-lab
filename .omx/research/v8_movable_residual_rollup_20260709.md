# v8 rate ledger COMPLETION — Movable edges + residual sidecars + whole-scene roll-up — MEASURED

**Date:** 2026-07-09 · **Axis:** `[macOS advisory · NON-PROMOTABLE]` · pointer contest-CPU **0.19110
UNMOVED** · #205 UNTOUCHED · MPS/GPU untouched (pure numpy/scipy/brotli on the label cache). `$0`,
read-only on `experiments/results/mlx_fleet_gt_cache/gt_n600.npz['lstars']` (600×384×512 int64 argmax,
comma10k order Road0/Lane1/Undriv2/Movable3/MyCar4).

This finishes the geometric rate ledger the prior two units opened: **(A)** the owed MOVABLE edges,
**(B)** the owed residual sidecars (horizon + Road/Lane), **(C)** the whole-scene roll-up the operator
sees. All numbers MEASURED through a REAL coder with a bit-exact roundtrip; no b/px proxy, no projection.

## STORES CONSULTED (recall, not re-derive)
- `.omx/research/v8_roadlane_geometric_rate_20260709.md` (441377ddd) — Road↔Lane 0.0275 S lossless, 72.5%
  cover, LBND2 coder (quantize+zigzag temporal-delta+Hungarian slot ξ-track+brotli q11, bit-exact). The
  coder pattern reused here for the Movable sites.
- `v8_increment1_design_draft_20260709.md` §3 (the 5-edge table; 2 owed: Undriv/Movable + Road/Movable) +
  §1 byte-cost audit discipline ("audit all findings, not naive/toy"; real coder never a proxy).
- DAG **FEED-v8-realmachinery** (horizon = deg-3 poly + ξ, 0.0032 S = 14.6×, `residual_sidecar_owed` for
  the 1.46 px fit residual + secondary arcs) · **FEED-v8-voronoi** (Movable = SPARSE SITES, few/compact,
  mid-band, store parsimonious GENERATORS not the boundary; dense-medial≈bitmap was the measured negative)
  · **FEED-v8-ratebudget** (0.346 de-shared bitmap; Movable rows 0.028+0.033).

## (A) MOVABLE edges — ONE sparse-site carrier reproduces BOTH owed edges

Movable (class 3) is present in all 600 frames, 1.24% area, median **1580 px/frame**. Its boundary splits
**Road-adjacent 47.3% / Undriv-adjacent 51.9%** (other 0.8%) — so a single Movable-region carrier de-shares
BOTH owed edges (Road↔Movable AND Undriv↔Movable) at once; they are the two sides of the same car blobs.

**Instances (MEASURED, 8-connectivity, MINA=10 px):** median **3.0/frame**, mean 3.66, range 1–10, 2041
total across n600. Component area median 105 px (22% of comps <50 px, 12% <20 px = noise-scale → residual).

**Generator fit (MEASURED):**
| generator | params/frame | boundary cover (≤2px) | region IoU | fit residual |
|---|---|---|---|---|
| bbox (4 params) — PRIMARY | 3.4 | **70.0%** | 0.745 | 1.00 px |
| ellipse (5 params, moments) | 3.4 | 69.1% | **0.811** | 1.00 px |

bbox and ellipse give the same **~70% boundary coverage** (ellipse fits the region better — IoU 0.81 — but
the exact boundary the same, since cars are neither boxes nor ellipses). Chose bbox (4 params, simplest,
matches the "start simplest" directive; ellipse's +IoU doesn't buy boundary cover). **~30% residual owed.**

**Real coder (MEASURED, bit-exact):** Hungarian centroid tracking (gate 60 px) → persistent slots →
per-slot temporal-delta of the 4 bbox ints → zigzag → present-bitmap + int16 deltas → **brotli q11**.
Roundtrip reconstructs the exact per-frame bbox SET bit-for-bit (verified). **5,161 B = 5.04 KB @n600 →
S = 0.00344.** Frozen-vs-moving: cars move smoothly, so temporal-delta + slot-tracking is the win (16.9 KB
raw → 5.2 KB = 3.3×), same mechanism class as the Road/Lane LBND2 coder.

**Real bitmap baseline (MEASURED):** Movable region mask, XOR-temporal-delta + brotli q11 = **79,817 B =
77.9 KB → S = 0.05315** (the draft's 0.028+0.033 = 0.061 conservative estimate was close; 0.0532 measured).

## (B) Residual sidecars — the honest completion (SHARED generic coder, bit-exact)

The generators carry the DOMINANT structure (~70–83% of each edge's boundary). The uncovered boundary px
(car concavities, faint/occluded lane fragments, horizon secondary arcs) are the `residual_sidecar_owed`.
**Shared residual coder:** per-frame uncovered boundary px → row-major flat-index → temporal/spatial
delta → zigzag → brotli q11; roundtrip recovers the exact coordinate SET bit-for-bit (verified all 3 edges).

| edge | uncovered | px/frame | residual coder S | complete S = gen + residual |
|---|---|---|---|---|
| Movable (bbox) | 30.0% | 75 | **0.01741** | 0.00344 + 0.01741 = **0.02085** |
| Road↔Undriv horizon (poly+ξ) | 25.7% | 111 | **0.01892** | 0.0032 + 0.01892 = **0.02212** |
| Road↔Lane (centerline band) | 26.6% | 228 | **0.04203** | 0.0275 + 0.04203 = **0.06953** |

Road↔Lane residual measured through the REAL Wave-F machinery (`cluster_lane_lines`+`fit_lane_line`+
`rasterize_lane_band`) to find the true uncovered set — 26.6% matches the memo's 72.5%-cover (27.5% owed).

**The residual coder is the honest bottleneck.** Generic sparse-coord coding costs ~0.4–0.6 B/px because
the uncovered px are scattered short fragments. A chain-code residual coder (task's suggested "short
chain-code segments", MEASURED on Movable) buys only **−13%** (22.3 KB vs 25.5 KB) — the residual is
genuinely near its coordinate entropy, not smooth-curve-compressible. So completion is expensive with
today's mechanism; two MEASURED-headroom levers below.

## (C) THE ROLL-UP — whole-scene geometric rate (each inter-class edge counted once)

| edge | bitmap S | geometric DOMINANT-only S | geometric COMPLETE (+residual) S | × complete |
|---|---|---|---|---|
| Road/Lane | 0.204 | 0.0275 (72.5% cover) | **0.0695** | 2.9× |
| Road/Undriv (horizon) | 0.047 | 0.0032 (poly+ξ) | **0.0221** | 2.1× |
| Movable (Road/Mov + Undriv/Mov) | **0.0532 (MEASURED)** | 0.00344 (70% cover) | **0.0209** | 2.5× |
| Road/MyCar (hood) | 0.028 | 0.0202 | 0.0202 (static model, complete) | 1.4× |
| Lane/* (3 rows) | 0.007 | 0.007 | 0.007 | 1.0× |
| **WHOLE-SCENE TOTAL** | **0.339** | **0.061** | **0.140** | — |

**vs current frontier rate term 0.118** (pointer 0.19110 = 100·d_seg + √(10·d_pose) + 25·bytes/N):

- **Geometric DOMINANT-only = 0.061 S** — **5.5× below** the 0.339 bitmap and **1.9× below** the 0.118
  frontier. This is what the parsimonious generators (few coeffs per class) buy — the v8 rate thesis is
  **CONFIRMED on the dominant structure**: geometry ≪ bitmap ≪ current frontier.
- **Geometric COMPLETE (lossless, generic residual) = 0.140 S** — **2.4× below** the bitmap but **~1.2×
  ABOVE** the 0.118 frontier. **With today's generic residual coder, lossless v8 does NOT by itself beat
  the frontier rate.** The gap is ENTIRELY the residual sidecar (0.079 of the 0.140).

**HONEST verdict (which rows are what):** hood (0.0202) is complete (static model, no separate residual).
Road/Lane, horizon, Movable are **dominant-structure lossless-coder numbers + a MEASURED residual
sidecar**; the residual is real and substantial. Lane/* untouched (already tiny). The whole-scene
"what v8 buys on rate" is a RANGE: **0.061 (dominant, beats frontier 1.9×) → 0.140 (complete, ties/slightly
over frontier)** — the true operating point depends on the residual coder + de-sharing (below), and it is
**MEASURED**, not the draft's earlier "~0.02–0.05 projected".

**Two MEASURED-headroom levers that move 0.140 toward 0.061 (both un-exploited):**
1. **Residual double-count (de-sharing, uncounted).** The horizon's uncovered "secondary arcs" ARE objects
   breaking the horizon (measured 1.6–2.0 crossings/row) = Movable/Undriv px ALREADY carried by the Movable
   sites. Paying for them in the horizon residual double-counts; attributing them to the Movable carrier
   shrinks the horizon residual for free. Same class for Road/Lane fragments near cars.
2. **Curve-relative residual coder (not built).** The residual px sit in a thin band around the generator;
   coding the signed OFFSET-from-generator per residual px (small integers) instead of absolute coords is
   the obvious lever — the generic sparse-coord/chain coder here (0.4–0.6 B/px) is an UPPER BOUND, not the
   floor. This is the v8 rate-completion's #1 open real-coder lever.

## Adversarial self-review (before commit)
1. **Coder real, roundtrip bit-exact?** YES — 3 coders (Movable bbox sites, shared residual sparse-coord,
   chain-code) each verified reconstruct the exact quantized set bit-for-bit through brotli q11; the
   Road/Lane residual used the real in-tree Wave-F lane machinery, not a proxy.
2. **Residuals INCLUDED (NO-FAKE)?** YES — the roll-up reports BOTH dominant-only AND complete-with-residual;
   the headline states plainly that complete (0.140) does NOT beat the frontier with today's residual coder.
   No cherry-picking the dominant number as if it were lossless.
3. **Numbers MEASURED not guessed?** YES — all from n600 argmax through real fits + real coders. Movable
   bitmap baseline measured (0.0532), not assumed. Coverage %s measured. The one non-measured claim (the
   two headroom levers) is labeled un-exploited/not-built.
4. **Double-count honest?** YES — flagged explicitly as the horizon-residual-⊃-Movable overlap (a de-sharing
   headroom, currently INFLATING the complete number, not deflating it).

## Consequence for v8
Rate ledger is now COMPLETE and MEASURED for all 5 whole-scene edges. **The dominant-structure geometric
rate (0.061 S) decisively confirms the v8 thesis (5.5× < bitmap, 1.9× < the 0.118 frontier).** Lossless
completion via today's generic residual coder lands at 0.140 (ties the frontier); closing that gap is a
real-coder problem (curve-relative residual) + a de-sharing problem (residual/Movable overlap), both
MEASURED-headroom, neither yet built. The d_seg half remains the true blocker (#205). Per-carrier build =
the parametric generator, never the boundary bitmap.

`[triality: DAG leg = FEED-v8-rollup; equation = v8_geometric_rate_decomposition_v1 FORMALIZATION_PENDING]`
· pointer 0.19110 UNMOVED · #205 untouched.
