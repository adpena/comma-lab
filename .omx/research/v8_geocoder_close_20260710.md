# v8 GEOCODER CLOSE — Road/Lane generator-coverage, the texture primitive REFUTED in composition (#394 UNIT A) — 2026-07-10

**Axis:** `[through-R] n600 · macOS-CPU advisory · REALIZED-through-R CPU-SegNet · NON-PROMOTABLE`.
**MEANS.** Pointer **contest-CPU 0.19110 UNMOVED** — it moves only through a byte-closed
`upstream/evaluate.py` n600 exact row. Everything here is a mask/through-R measurement that shapes
the v8 route; it is not a score.

**Operator directive (#394 UNIT A):** *"close the Road/Lane generator-coverage gap — the measured
53% of the v8 rate enemy."* Decisive new input: the texture price list
(`segnet_texture_perception_20260710`) — Road/Lane are texture-defined; the winning primitive is a
period-4 stem-Nyquist luminance grating.

---

## 0. HEADLINE (answer-first)

**The period-4 grating texture primitive is REFUTED as the v8 Road/Lane generator-coverage closer
(FORMULATION scope, MEASURED n600 through-R).** Placed in the REAL Road/Lane region shapes within
scene composition, the price-list-winning grating **destroys** both:

| arm (n600 through-R) | agg d_seg | Road | Lane | Undriv | Movable | MyCar |
|---|---:|---:|---:|---:|---:|---:|
| **scene-flat baseline** (matched control) | **0.07095** | 0.0165 | 0.1204 | 0.0032 | 0.4031 | 0.2354 |
| **grating Road/Lane** (the proposed carrier) | **0.29873** | **0.9985** | **1.0** | 0.076 | 1.0 | 0.043 |
| Δ (texture − flat) | **+0.22778** | +0.982 | +0.880 | — | — | — |

The whole-frame price-list tile WINS Road/Lane in ISOLATION (Road +8.336 / win 0.887; Lane +1.994 /
win 0.970) — but that is a **context-free artifact**. SegNet is a U-Net: its argmax is
context-dominated. In the real scene the scene-mean FLAT road colour ALREADY wins Road (d_seg
**0.0165**); injecting the period-4 grating throughout the region flips it 100%. **The v8 Road/Lane
carrier FILL is FLAT scene colour, not a texture.**

**verdict_scope:** FORMULATION (the whole-region global context-free grating fill). The texture-fill
FAMILY is NOT killed — reformulations queued (§4). Pointer UNMOVED.

---

## 1. What was built (deliverables a + b)

- **`tac.through_r.roadlane_texture_generator`** (a) — the composed per-class texture generator:
  `plan_from_palette` (scene-flat basins + grating overrides), `fill_partition_texture` (fills the
  GT `L*` partition with each class's `TextureSpec`), `byte_account_texture_fill` (exact COUNTED
  bytes; rule-118: grating STRUCTURE free, fitted colours+phase+orientation counted = **15.6 bytes
  whole-video**), `run_composed_generator_arm` (through-R via the canonical `palette_realization.run_arm`).
  `default_roadlane_grating_specs` = the MEASURED price-list winner (period-4, orient 135°, bright
  160 / dark 0, Road bright-on-dark, Lane reversed). 15 tests green.
- **`tac.boundary_math.movable_site_coder`** (b) — the Movable sparse-site geometric carrier:
  `extract_movable_sites` (connected-component boxes), `track_sites` (bounded-K Hungarian
  correspondence — reuses the #234 LAP discipline), `byte_account_sites` (tracked temporal-delta +
  zigzag + zlib vs raw-per-frame), `render_sites_to_mask` (geometry-coverage tell). 15 tests green.
- **`experiments/measure_v8_geocoder_close.py`** — the governed through-R driver (matched flat vs
  grating A/B + Movable byte-account + verdict rows).

## 2. What was measured (deliverable c — MEASURED, n600 through-R authority-scale)

**Scale ladder (all real R + frozen CPU-torch SegNet):** n6 smoke → n96 governed (peak 7.3 GiB,
67 s) → **n600 governed authority** (`is_n600=True`). The refutation is monotone across all three
(Road/Lane → 1.0 at every scale). n600 numbers in §0.

**Reading the baseline (DERIVED):** my scene-flat control uses ONE GLOBAL per-class mean colour →
agg 0.071, worse than the cited per-pair flat-paint floor 0.0416 (per-pair colour is better). The
A/B is still clean: both arms share the SAME global scene colours; the ONLY difference is Road/Lane
flat vs grating, so Δ = +0.228 is a matched-control delta, not a baseline-quality confound.

**Movable sparse-site carrier (MEASURED n600):** 2145 sites, K=9 concurrent slots, **6289 B**
(tracked temporal-delta + presence) vs **9094 B** raw-per-frame (correspondence-tracking saves
**31%**), box-IoU vs GT-Movable **0.743** (boxes over-cover the blobs ~26% — the honest lossy tell).
This is a real, byte-accounted geometry primitive for the Movable edge (SPEC_v8.1 §I clause-B).

## 3. The mechanism (DERIVED from the two measurements)

The price-list tile is a WHOLE-FRAME grating (no competing context) → evokes Road. The SAME grating
in a real region, bordered by flat Undrivable/Lane/Movable within real scene geometry, injects
period-4 high-contrast edges the surrounding context re-reads as not-Road. In composition the
context SUPPLIES the discrimination the isolated tile lacked → scene-flat wins Road (0.017). The
"Road/Lane need texture" finding is context-free; it does not transfer.

## 4. Constraint carved + reformulation queue (P10 / verdict-scope ladder)

**Carved (REMOVED / PINNED / RELOCATED):**
- REMOVES the price-list whole-frame grating from the v8 Road/Lane carrier design space (NO-GO).
- PINS the v8 Road/Lane carrier FILL to FLAT scene colour (Road covered flat at 0.017).
- RELOCATES the coverage residual to **Movable** (flat 0.403) + **MyCar** (0.235) per-frame COLOUR
  and **Lane** (0.120) thin-structure boundary jitter (#333 annulus) — **none is a texture gap**.
  The "53% Road/Lane generator-coverage enemy" is dominated by Movable/MyCar per-frame colour + Lane
  boundary, NOT a missing texture generator.

**Reformulation queue (untested — the texture-fill FAMILY lives):**
1. thin-band grating at the Lane centerline only (not the whole region);
2. grating restricted to the Road/Lane boundary annulus (#333) only;
3. scene-ADAPTED grating colours (not the context-free 160/0);
4. grating oriented to the LOCAL lane tangent (directional basis) not a global 135°;
5. per-PAIR (not global) flat colours (the 0.0416 per-pair floor beats the 0.071 global-flat);
6. a Movable/MyCar per-frame COLOUR carrier (the real residual).

## 5. Owed / honest non-coverage

- The scene-flat baseline is GLOBAL-mean (0.071), not the per-pair 0.0416 floor — the baseline has
  its own headroom (reformulation 5). The A/B delta is unaffected.
- Through-R d_seg is a NECESSARY mask/geometry signal, not the byte-closed exact score. The v8
  Road/Lane carrier (FLAT fill) still needs the geometry byte-close (SPEC_v8.1 §I horizon 4167 B +
  lane LBND2) + n600 exact eval to move the pointer.
- The Movable site carrier's box-IoU 0.743 means boxes over-cover; a contour/ellipse site or a
  per-site presence-gated fill is the tighter reformulation (not measured).

## 6. Triality legs

- **DAG:** FEED-u394a (this block).
- **equations:** `roadlane_grating_composition_refuted_v1`
  (`src/tac/canonical_equations/roadlane_grating_composition_refuted_20260710.py`,
  VERIFIED_VIA_EMPIRICAL_ANCHOR, registered).
- **mechanism/DSL:** the two generator modules (`roadlane_texture_generator`, `movable_site_coder`);
  no new trainer lever (a $0 measurement, not a training launch — the DSL leg is N/A for a
  measurement screen, per SPEC_v8.1 §PROGRAM precedent).
- **verdict:** `experiments/results/v8_geocoder_close_n600/verdict_v8_geocoder_texture_close.json`
  (NO-GO, FORMULATION, ANTAGONISTIC, is_negative=True + reformulation_queue).

## 7. #385 one-liner (v8 macro-rate triple update)

v8 geocoder close (#394A, n600 through-R): the price-list period-4 grating is a **NO-GO** as the
Road/Lane carrier fill (destroys both in composition, +0.228 vs matched scene-flat; Road already
wins flat 0.017) → the Road/Lane carrier fill is **FLAT** (cheap, ~15 B texture), and the 53%
Road/Lane "coverage enemy" RELOCATES to Movable/MyCar per-frame colour + Lane boundary jitter, NOT a
texture generator. Movable sparse-site carrier MEASURED (2145 sites → 6289 B tracked, IoU 0.743).
The macro-rate triple's Road/Lane term is a FLAT-fill + geometry-placement problem, not a texture-fill
gap — do NOT build the grating carrier.

## STORES CONSULTED

`segnet_texture_perception_20260710` (the price-list winner, context-free) ·
`mature_codec_toolbox_audit_20260710` (coverage-not-coding; the top-5 unexploited levers) ·
`SPEC_v8.1_20260709` + macro-rate addendum (the v8 carrier decomposition + 0.0585/0.131/0.0725
triple) · `tac.through_r.palette_realization` (flat-paint floor 0.0416, run_arm, decision geometry) ·
`tac.through_r.stem_perception` (TextureSpec / texture_dl_bits / price list) ·
`tac.boundary_math.lane_track_and_smooth` (#234 Hungarian correspondence, reused) ·
MEMORY L12 (witness capstone) / L17 (d_seg islands = LANE) / L66 (annulus = boundary jitter) /
L71 (analytic lane band) · CLAUDE.md §WITNESS CAPSTONE (chroma/texture as d_seg lever). No upstream
edits. Pointer 0.19110 UNMOVED (MEANS).
