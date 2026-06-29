# Eikonal-SDF / control-point FULL-5-class partition d_seg recovery from a TINY descriptor — rate-half sizing

- **UTC:** 20260629T164449Z
- **Authority:** `[macOS advisory / research-signal]` — `score_claim=false`, `promotable=false`,
  `ready_for_exact_eval_dispatch=false`. **Pointer UNMOVED** (this is NOT a contest row; it sizes the
  witness *rate-half* generator before GPU spend).
- **Compute:** $0, CPU/numpy, 23.6s, n96 frozen-SegNet argmax (`gt_n96.npz['lstars']`, 384×512, canonical
  comma10k order 0 Road / 1 Lane / 2 Undrivable / 3 Movable / 4 MyCar).
- **Tool:** `tools/measure_eikonal_sdf_dseg_recovery.py` (reuses `tac.boundary_math.bitmask_dseg.d_seg_reference`
  [canonical d_seg AUTHORITY], `contour_codec.partition_description_bytes` [lossless LZMA ceiling],
  `lever_b_levelset_generator.signed_distance_fields` [eikonal SDF form], `lane_sdf_component` [FEED-dm],
  `contest_score`).
- **Data:** `experiments/results/eikonal_sdf_recovery_20260629T164449Z/results.json`.

## Question (lit-hunt H3)
How much d_seg does a **zero-learned-byte deterministic generator** recover from a **tiny boundary
descriptor**, and what residual is irreducible? This sizes the COUNTED video-derived payload (rate-half)
the witness must store; the generator (eikonal/SDF/rasterizer) is FREE in inflate.py (rule 118).

## Prior art (existence proofs — cited, not re-derived; per terminal-conclusion-crosscheck)
The FULL-partition **lossless** store and the **margin-simplify** lossy curve were already measured. This
test ADDS two free generators not previously curve'd (eikonal-seed Voronoi; control-point polygon-SDF) +
the per-class irreducible residual.

| prior measurement | scope | d_seg | bytes | note |
|---|---|---|---|---|
| FEED-af (n600) | FULL 5-class | 0 (exact) | 255,288 B = **425 B/frame** (SOTA arith) → rate **0.17** | lossy region-drop DOMINATED |
| FEED-dj (n24) | FULL 5-class | 0 | 456 B/frame (temporal) | margin-simplify lossy DOMINATED |
| Task #52 (n16) | FULL 5-class | 0 | 895.7 B/frame (LZMA) | matches our ARM A |
| **FEED-dm (n48)** | **LANE only (SDF)** | **0.000415** | ~30 floats ≈ 1–2 KB / 600 fr | **post-R 0.000797 → SURVIVES R** |
| **FEED-du (n96)** | **HOOD only (static)** | **0.000737** | **56 B total / 600 fr** | **post-R 0.000677 → SURVIVES R** |
| FEED-ah | realization gate | ×170–350 | — | palette-paint→RGB→SegNet→R multiplies stored d_seg |

## NEW measurement — the RD curves (n96, direct-partition d_seg vs descriptor bytes)

**ARM A — lossless ceiling (LZMA-over-labels):** 872 B/frame indep, **693 B/frame temporal**, d_seg=0.
(FEED-af's context-arithmetic codec tightens this to 425 B/frame → **rate 0.17**, the true ceiling.)

**ARM B — coarse-seed eikonal/Voronoi** (verified: NN-upsample ≡ eikonal-EDT-argmax from grid seeds,
0.7% disagree — the literal "grow SDF from seeds → argmax → partition"):

| block | grid | d_seg | B/frame (temporal) | seg-half rate |
|---|---|---|---|---|
| 2 | 192×256 | 0.00555 | 257 | 0.103 |
| 3 | 128×171 | 0.00773 | 153 | 0.061 |
| 4 | 96×128 | 0.01064 | 84 | 0.034 |
| 8 | 48×64 | 0.01716 | 23 | 0.009 |
| 16 | 24×32 | 0.02531 | 5 | 0.002 |
| 32 | 12×16 | 0.04161 | 2.4 | 0.001 |

**ARM C — control points (cv2 contour + approxPolyDP) + free SDF-argmax** (n48):

| eps | d_seg | B/frame | verts | rate |
|---|---|---|---|---|
| 0.5 | 0.00279 | 1124 | 1092 | 0.449 |
| 1.0 | 0.00475 | 697 | 256 | 0.279 |
| 2.0 | 0.00769 | 504 | 157 | 0.201 |
| 8.0 | 0.01661 | 358 | 103 | 0.143 |

**ARM D — structured lane SDF (FEED-dm re-verify, n48, hard-band):** lane-attributable d_seg 0.00425
(FN 0.00192 + FP-from-road 0.00159), ~43 floats/frame. Consistent with FEED-dj hard-mask 0.00396; the
**continuous-band optimal form (FEED-dm) is 0.000415, post-R 0.000797** (cited — supersedes hard-band).

## The decisive structural finding — lossy partition coding is DOMINATED for EVERY free generator
The seg weight (100) vs rate weight (25/37,545,489 ≈ 6.66e-7/byte) gives a **break-even of only
Δd_seg/Δ(byte/frame) = 4.0e-6**. The coarse-seed curve trades at **2.1–8.4e-5 per byte (5–21× worse
than break-even)**; control-points similar. So **in LABEL SPACE you cannot trade d_seg for bytes** — the
lossless store (rate 0.17) dominates every lossy operating point. This **generalizes FEED-dj's
margin-simplify "lossy dominated" result to the eikonal-seed AND control-point generators** — it is a
property of the SCORE, not the codec.

**Consequence:** the rate-half is **NOT byte-bottlenecked**; a near-perfect partition is already
rate-affordable (~0.17). The bottleneck is **d_seg ACCURACY** + the **realization-through-R** gate.

## Per-class irreducible residual (coarse-seed block-2, frame 0; d_seg≈0.0055)
| class | area | attributable d_seg | dominant mode |
|---|---|---|---|
| Road | 22.4% | 0.00554 | FP 0.0049 = **boundary annulus** (road bleeds at coarse edges) |
| Lane | 0.71% | 0.00283 | FN 0.0023 = **thin markings vanish** (highest residual-per-area) |
| MyCar | 25.7% | 0.00130 | FN (hood edge) |
| Undrivable | 49.3% | 0.00139 | well-captured (large smooth) |
| Movable | 1.81% | 0.00071 | **small objects** missed |

The irreducible residual is concentrated in **(a) the all-class boundary annulus, (b) thin lane
markings/dash-gaps, (c) small movable objects** — i.e., the **~8-dim nonlinear lane-orbit long-tail**
(MEMORY flip-mass: Road ~50% / Lane ~19% / Undrivable ~13%). Free generators capture the large smooth
regions; the long-tail needs the LEARNED witness.

## Realization caveat (binding; do NOT overclaim)
ARM A/B/C are **direct-partition** d_seg (stored labels vs GT labels). The contest realizes through
RGB→SegNet→R; FEED-ah measured palette-painting multiplies this ×170–350. So the free generator MUST be
**SDF-based** (1-Lipschitz margin → R-surviving), not label-painting. The structured SDF components
(FEED-dm lane post-R 0.0008, FEED-du hood post-R 0.0007) are the existence proof that SDF representations
survive R at tiny bytes.

## VERDICT — VIABLE (structured-manifold form), LIMITED (generic label-space form)

- **Generic eikonal-seed / control-point recovery: LIMITED.** Recovers direct d_seg to ~0.0028–0.0055 at
  ~250–1124 B/frame (rate 0.10–0.45), i.e. 3–6× the frontier need (~6e-4–1e-3), and frontier accuracy
  needs near-lossless (~256 KB, rate 0.17). Lossy points are score-DOMINATED. Not the primary rate-half.
- **Structured-manifold descriptor: VIABLE — this IS the rate-half sufficient statistic.** Per-class SDF
  on the ~8-dim manifold coords achieves frontier-grade per-class d_seg (lane 4.2e-4, hood 7.4e-4),
  **R-surviving**, at **a few KB TOTAL for 600 frames (rate ≪ 0.01)** — it spends bytes on O(manifold)
  not O(boundary-entropy). The COUNTED payload = the manifold coords + a small LEARNED residual for the
  annulus/dash/small-movable long-tail; the SDF rasterizer + eikonal growth is the FREE generator.

**HEADLINE:** a tiny **structured** descriptor (~few KB / 600 frames, rate ≪ 0.01) + free SDF/eikonal
generator recovers per-class d_seg to **~7e-4 (R-surviving)**; **generic** label-space recovery
(eikonal-seed/control-point) gets only ~0.003 d_seg @ 250–1124 B/frame and is score-DOMINATED by the
lossless store (rate 0.17). **Irreducible residual = the all-class boundary annulus + lane dash-gaps +
small movable objects (~8-dim lane orbit)** — this is precisely the LEARNED payload the trained witness
must carry; everything else (large smooth regions) the free generator handles. The rate-half is
**byte-cheap and d_seg-bottlenecked**, so the GPU budget belongs on the residual/realization gap, NOT on
shrinking the descriptor.

### Wire-in (Catalog #125)
- Hook #1 sensitivity-map: ACTIVE — per-class residual identifies where capacity must route (annulus/lane/movable).
- Hook #2 Pareto: ACTIVE — the d_seg-vs-bytes curves are the rate-half RD frontier; lossy region pruned (dominated).
- Hook #3 bit-allocator: ACTIVE — structured-manifold O(coords) ≪ O(boundary-entropy) is the byte allocation rule.
- Hook #4 cathedral autopilot: N/A (advisory measurement).
- Hook #5 continual-learning: ACTIVE — confirms+generalizes FEED-dj "lossy dominated"; sizes learned residual.
- Hook #6 probe-disambiguator: N/A.
