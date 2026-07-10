# SegNet texture perception — how the frozen scorer reads texture, and the minimal per-class texture that wins through R (2026-07-10)

**Axis:** `[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]`. Every number here is
a **context-free decision-geometry probe** (whole-frame constant/parametric tiles), explicitly
**NON-n600** — a d_seg VERDICT is n600 only through `tac.through_r.measure_through_r`. **MEANS; the
pointer (contest-CPU 0.19110) is UNMOVED** — it moves only through byte-closed exact eval.

**Module:** `src/tac/through_r/stem_perception.py` (+ `tests/test_stem_perception.py`, 15 tests green).
**Artifact:** `experiments/results/stem_perception_20260710/{tile_responses.jsonl, price_list_analysis.txt, verdict_texture_price_list.json}`.
**Equations:** `segnet_stem_nyquist_alias_wall_v1` + `segnet_through_r_texture_price_list_v1`
(both `VERIFIED_VIA_EMPIRICAL_ANCHOR`).

**Origin (operator, 2026-07-10):** *"how SegNet perceives texture and how we can provide it precisely
what is optimal."* Context: the palette probe (`palette_realization`) proved the argmax is
texture-dominated — flat colour wins Road only 1/216, Lane 0/216 — and the trained mod32cap witness
(d_seg 0.0048) sits **8.7× below** the 0.0416 flat-paint floor because it *synthesizes texture*. So:
don't reproduce video texture — read the perceiver, then produce SegNet-OPTIMAL texture at MINIMAL rate.

---

## 1. The perceiver, read verbatim (Phase 1)

The frozen SegNet is `smp.Unet('tu-efficientnet_b2', classes=5)`. Its first layer — everything the
whole scorer's perception is built on — is `encoder.model.conv_stem`: **32 kernels of 3×3×3,
stride-2, no bias**, followed by `bn1` (`BatchNormAct2d`, eps 1e-5, **SiLU**). Read verbatim
(no upstream edit) via `extract_stem_filters` + `characterize_filters`:

| Property | MEASURED |
|---|---|
| kernels | 32 × (3 in × 3 × 3) |
| **kind** | **21/32 low-pass (colour/blur), 11/32 isotropic high-pass** |
| **colour** | **11 achromatic/luminance** + strong opponency: 8× R+G, 4× +R−B, 2× +G−B, +G−RB, +G−R, +B−RG, … |
| DC-fraction | mean 0.62, median 0.91 (most kernels are near-pure colour/blur) |
| **orientation** | **oriented_share = 0.0** — orientation is NOT resolvable at 3×3 |

**Read:** the B2 stem is a **colour + low-pass front-end**. It is not an oriented-edge bank —
orientation selectivity emerges in *later MBConv layers*, not the stem. Texture discrimination is
therefore a **downstream** property, gated at the front door by the stem's Nyquist.

### The alias wall (`stem_nyquist`)

Stride-2 halves 384×512 → 192×256, so the finest texture PERIOD that survives the stem is
`2·stride = 4` seg-input px. Seg-input is itself R's bilinear-down of camera 874×1164 → 384×512
(scale ≈ 2.27), so:

> **finest surviving period = 4 seg-input px ≈ 9.1 camera px.**

Texture finer than this aliases away **before the first MBConv even runs** — it cannot carry
class-discriminative signal. This is the load-bearing constant of the whole texture question.

`Law: segnet_stem_nyquist_alias_wall_v1.`

---

## 2. Per-class minimal sufficient texture — the PRICE LIST (Phase 2, through R)

568 cheapest-first tiles (64 flats + a bounded stripe/checker/gabor grid at periods {2,4,8,16} × 4
orientations × 24 colour pairs) pushed through the **real R** (`render_grid_to_camera_uint8` → uint8
→ SegNet `preprocess_input`) + frozen SegNet; per-class interior **signed margin**
`logit_c − max_{k≠c} logit_k`. Per class, the CHEAPEST (fewest description-length bits) texture that
WINS the argmax with positive margin:

| class | flat floor | cheapest winner (through R) | reading |
|---|---:|---|---|
| **Undrivable** | **+8.33 (WINS)** | flat, 15 bits, margin +8.33, win 1.000 | texture-free DEFAULT basin (sky/top); 496/568 tiles land here |
| **MyCar** | **+11.85 (WINS)** | flat, 15 bits, margin +11.85, win 1.000 | flat-winnable (ego-hood colour region) |
| **Movable** | **+0.31 (WINS)** | flat, 15 bits, margin +0.31, win 0.696 | weakly flat-winnable |
| **Road** | **−3.50 (LOSES)** | **stripe period-4 gray↔black, 40 bits, margin +8.34, win 0.887** | TEXTURE-defined |
| **Lane** | **−5.00 (LOSES)** | **stripe period-4 black↔gray, 40 bits, margin +1.99, win 0.970** | TEXTURE-defined |

### The Road + Lane headline (the decisive finding)

**Road and Lane have a NEGATIVE flat floor — 0/64 flats win either.** No constant colour classifies
them through R. The ONLY winning texture in the entire 568-tile sweep is a **period-4 (= the stem
Nyquist, ≈9 cam-px) HIGH-CONTRAST LUMINANCE grating**:

- exactly **1 of 96** period-4 stripes wins Road (+8.34), **1 of 96** wins Lane (+1.99);
- **NOTHING at period 2, 8, or 16 wins** — period-2 aliases below Nyquist; period-8/16 read as flat →
  the Undrivable basin;
- **polarity matters** — Road = bright-on-dark (gray-on-black), Lane = dark-on-bright (reversed).

This is the **texture-level statement of the capstone thesis**: the witness's binding job on Road/Lane
is **stem-Nyquist grating SYNTHESIS**, not colour selection. It also explains the palette finding
mechanically — the stem's low-pass filters read any flat region as the Undrivable default basin;
only a stem-Nyquist-frequency grating produces the edge responses that flip argmax to Road/Lane. And
it explains why the trained mod32cap witness beats the flat-paint floor 8.7× (§3).

`Law: segnet_through_r_texture_price_list_v1.`

### Rate / compliance boundary (rule-118)

The period-4 grating **STRUCTURE** is a generic parametric family — **free** as inflate.py code. Only
the fitted **colours + phase + orientation** per Road/Lane region are video-derived — **counted**
(~40 bits/generator here; a denser search sharpens the colour/phase, not the structure). This is the
rate half of the capstone: store the tiny fitted coefficients, generate the grating deterministically.

---

## 3. What did the witness learn (partial; part-3 comparison)

The trained mod32cap witness (`levelset_n600_witness_mod32cap_20260706T115554Z`, EMA-BEST) is a
coord-INR with a per-class **`palette (5,3)`** base + a dedicated **`out_tex` (3,96)** texture head.
Read from the checkpoint (weights only; no decode): the `out_tex` head is **ACTIVE but MODEST**
(‖W‖ mean 0.06, max 0.22) — i.e. the witness spends most of its budget on a flat per-class palette
base and adds a small texture perturbation on top. This is *consistent* with beating the flat-paint
floor 8.7× via modest texture, and with the price list (Road/Lane need only a narrow-band grating).

**DEFERRED (honest):** the direct spectral comparison — decode mod32cap BEST renders for Road/Lane
regions and check whether their power spectrum concentrates at the period-4 stem-Nyquist band vs the
optimal synthetic — requires a witness render decode that would **contend with the live witness run
(pid 55609)**. Routed as a #385 follow-up; the cheap-read above (active out_tex head + palette base)
is the load-bearing partial. Verdict scope: **FORMULATION** (characterization, no kill).

---

## 4. Route (v8 carriers + witness loss)

1. **v8 Road/Lane carrier = a period-4 grating primitive**, not a colour region. Store per-region
   {two colours, phase, orientation, extent}; generate the grating in inflate.py. The structure is
   free; only ~a few dozen bits/region are counted. Sisters: FEED-v8-roadlane, L71 analytic lane band.
2. **Witness loss implication:** a texture-DL regularizer should bias Road/Lane toward the period-4
   band (the finest surviving scale) rather than smooth/flat — the smooth-stage curriculum (which
   RAISES d_seg, per MEMORY) is exactly the wrong prior for these classes. Anti-alias below period-4
   is wasted (aliased away by the stem).
3. **Undrivable/MyCar/Movable** stay flat carriers (texture-free basins) — spend the texture budget
   only where it flips argmax (Road/Lane), per the costate duty-to-measure ranking.

---

## Stores consulted
`tac.through_r.palette_realization` (flat-paint floor, Road 1/216 Lane 0/216) · MEMORY L12 (witness
capstone) / L17 (d_seg islands = LANE ~8-dim manifold) / L71 (analytic lane band) · CLAUDE.md
§WITNESS CAPSTONE (chroma as d_seg lever, all-class directional basis) · DAG FEED-alldim /
FEED-v8-roadlane · #204 NTK band-pass / #141 margin-saliency / #213 matched-filter-on-stem (the
foundation this builds on). No upstream edits. Pointer 0.19110 UNMOVED (means).
