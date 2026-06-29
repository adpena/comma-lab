# R-SURVIVAL PHYSICS — the binding d_seg wall, deep-math + $0 measured (GAP 2 / DAG FEED-iw)

**UTC:** 20260629T182659Z · **evidence grade:** `[macOS research-signal]` (advisory; SegNet-FREE
resample isolation) · **score_claim:** false · **promotable:** false · **pointer:** 0.19110 UNMOVED.
**Tool:** `tools/r_survival_probe.py` · **data:** `.omx/research/r_survival_probe_n96.json` (n=96).

> means→ends: this is the binding d_seg wall toward sub-0.15. The deliverable is the **minimal
> R-invariant boundary representation** + the **measured survival curve** + **what the v2 survival term
> must do**. The headline is a REFRAME: the contest render operator R is **not** the binding wall.

---

## 0. TL;DR (the reframe)

contest `d_seg = mean[ argmax SegNet(R(witness)) != L* ]`, where
`R = render → bicubic↑874×1164 → uint8 → bilinear↓512×384`. The DAG-FEED-iw premise was that a
*geometrically-correct* partition FLIPS under R (the resample/uint8 kills the thin dashed lane). **Measured,
that premise is FALSE at scorer resolution.** Decomposing the wall:

| sub-wall | what it is | measured verdict |
|---|---|---|
| **(A) resample/quant survival** | does R's ↑874→uint8→↓384 destroy a correctly-rendered partition? | **BENIGN.** At render res ≥384, *every* rep (even naive palette) survives R to `d_seg < 1e-4`. R is a near-identity band-limited round-trip for content already resolved at 384. |
| **(B) capacity / render-res survival** | does the witness's *native* render resolution resolve the 2px lane? | **THE BINDING WALL.** Below 384, a hard-label / palette carrier loses the lane (box-averaged into road). An **SDF carrier survives ~8–10× better** and recovers ~full quality at render ≥192–320. |
| **(C) SegNet-reading survival** | does the frozen SegNet read the painted RGB *as* the partition? | the **other** half (the real palette ×170–350 death is HERE, not in A). Out of scope for this SegNet-free probe; measured by the trainer's `cpu_verdict_d_seg` realized-through-R verdict. |

So: **R is friendly; the wall is (B) witness render/representation capacity + (C) SegNet RGB-reading.**
The minimal R-invariant rep — a **wide-ramp 1-Lipschitz multi-class SDF (argmax of K signed-distance
fields)** — collapses sub-wall (B): it buys ~2× render-resolution headroom (SDF@192 ≈ hard@384), which is
exactly the headroom the small-basis / low-capacity witness needs.

---

## 1. Deep-math: the survival physics

### 1.1 The L*-is-at-384 observation (why R is benign)
`L*` (the GT partition, the d_seg reference) is `SegNet(gt)` argmax at **384×512 — the scorer
resolution**. The contest stores the reconstruction as a uint8 video at 874×1164, then the scorer
`preprocess_input` bilinear-downsamples to 384. So for a witness that renders **at or above 384**, R is
`render → ↑874 → uint8 → ↓384` = a round-trip that upsamples then downsamples back to the *same* grid the
target lives on. There is **no sub-384 information to alias away** — a 2px lane at 384 becomes ~4.6px at
874 and back to 2px. The only true losses are (i) the uint8 quant at 874 and (ii) the resample kernels'
mild low-pass, neither of which destroys a feature defined at the output grid. **Measured: confirms — every
rep ≈0 at @384.**

### 1.2 Nyquist / where the lane actually lives
Lane (class 1) geometry, measured on n=96 `L*`: width **median 2.0px**, mean 2.38px, p90 4.0px,
**76% of lane pixels ≤2px wide**, 0% ≤1px; lane area 0.59%. A 2px stripe has spatial period ~4px → it is
**above** the 384-Nyquist limit (period 2px). So the lane is *resolved* at 384 and round-trips through R
intact. The lane only falls **below Nyquist of the WITNESS** when the witness's native render res drops:
at render res `r`, the lane's effective width is `2·r/384` px → sub-Nyquist (≤1px) once `r ≲ 192`. **That
is the cliff**, and the measured cliff edge (§2) sits exactly there.

### 1.3 Gibbs: why hard/palette ring and SDF does not
`R` upsamples with **bicubic** (negative-lobe kernel). Bicubic interpolation of a *step* (the hard
indicator / palette level, |∇|→∞ at the boundary) **rings** — overshoot/undershoot lobes near the
discontinuity (Gibbs). After uint8 + downsample + argmax these lobes create/destroy boundary pixels. A
**1-Lipschitz SDF (|∇φ|=1, smooth, locally bandlimited)** has *constant* rate of change through the
boundary, so bicubic reconstructs it **accurately** — this is exactly Chlumský 2018's statement that SDF
"interpolation provides accurate reconstruction only where the rate of change is more or less constant."
The decision margin `m = φ_top1 − φ_top2` is ~**linear through zero**, so the argmax boundary (zero-crossing
of m) is placed at sub-pixel precision and is stable under interpolation.

### 1.4 The uint8 knife-edge favors a WIDE ramp
The uint8 quant at 874 has step 1/255. A **sharp** SDF ramp crosses the decision band (φ≈0) in <1px → only
1–2 transition samples → uint8 rounding jitters the zero-crossing. A **wide** ramp crosses over many px →
many transition samples → uint8 rounding *averages out* → the zero-crossing is placed below-uint8-precision
(it's encoded in the *sub-pixel pattern* of quantized samples, à la area-coverage AA). **Measured (§2.3):
wider ramp → strictly better lane survival, saturating at ramp half-width ≥ ~5px.**

### 1.5 The minimal R-invariant boundary representation
Combining 1.1–1.4, the boundary rep that minimizes `d_seg` through R is:

> **A multi-class 1-Lipschitz SDF carrier, `partition = argmax_k φ_k`, rendered with a WIDE decision
> ramp (half-width ≳ 5px ≈ the resample+uint8 footprint), at the highest native render res the byte
> budget allows (≥192 buys hard@384-equivalent survival).**

It is R-invariant because: smooth (no Gibbs in ↑874), 1-Lipschitz (linear margin → sub-pixel zero-crossing),
wide ramp (uint8 averages → sub-precision placement), and the partition is recovered by `argmax` which is
robust to symmetric kernel ringing. **R's bicubic/bilinear is then a *reconstruction filter* that places the
boundary MORE accurately than a hard resize** (measured: §2.2).

---

## 2. The measured survival curve (n=96, SegNet-free isolation)

`tools/r_survival_probe.py` encodes `L*` as a K=5 boundary-membership carrier in [0,255], pushes it through
the **contest-exact R** (torch `align_corners=False` bicubic↑874 → uint8 → bilinear↓384, matching
`apply_contest_faithful_roundtrip_nhwc`), and recovers the partition (argmax for membership/SDF carriers;
nearest-prototype for palette/rgb3). It compares to `L*` via the canonical
`bitmask_dseg.d_seg_reference`. **This isolates sub-walls (A)+(B); it does NOT run the SegNet (C).**

### 2.1 Capacity × rep grid — LANE (class 1) flip rate / total survival d_seg
| render res | palette lane% | hard lane% | **sdf lane%** | hard total | **sdf total** |
|---:|---:|---:|---:|---:|---:|
| 96  | 58.7 | 58.2 | 41.9 | 0.0160 | 0.0051 |
| 128 | 52.3 | 51.9 | 28.1 | 0.0122 | 0.0033 |
| 192 | 26.3 | 25.9 | **3.19** | 0.0063 | **0.00059** |
| 256 | 23.5 | 22.9 | **2.95** | 0.0061 | **0.00037** |
| 320 | 25.7 | 25.6 | **0.04** | 0.0062 | **0.00001** |
| 384 |  0.0 |  0.0 | 0.0 | ~0 | 0 |
| 874 |  0.09|  0.09| 0.0 | 1e-5 | 0 |

- **R is benign (A):** at render ≥384 every rep ≈0. Even naive single-channel **palette survives the
  resample** (@384 total 5e-5) — the famous palette ×170–350 death is therefore **NOT** a resample effect;
  it is sub-wall (C), the frozen SegNet reading an off-distribution flat-color image.
- **The cliff is render-res (B):** the lane flip rate explodes only as the *witness* render res drops below
  384. The lane is the binding class at every res (all other classes <0.2% once SDF@≥192).
- **SDF buys ~2× render-res headroom:** sdf@192 (total 5.9e-4) ≈ hard@384 (≈0). A small-basis witness
  rendering at 192 with an SDF carrier reaches the survival of a hard-label witness rendering at 384.

### 2.2 pre-R geometric vs rendering-survival (the task's decomposition)
- **hard/palette:** `survival ≈ pre-R` (e.g. hard@192: pre 0.00629, surv 0.00630). R adds ≈0 — the loss is
  the **low-res representation** (the lane is gone before R ever runs). The hard-label loss is **pure
  capacity**, R is neutral.
- **SDF:** `survival << pre-R-nearest-proxy` (sdf@192: pre 0.00555 nearest-proxy vs **surv 0.00059**). The
  nearest-resize proxy *overstates* the SDF's geometric cost ~10×; the true through-R survival is the floor,
  and **R's bicubic/bilinear actively HELPS** — it reconstructs the 1-Lipschitz zero-crossing better than a
  hard resize. (Intentional asymmetry: a label-predicting witness *must* use nearest — labels aren't
  interpolable; an SDF-predicting witness uses bicubic — the SDF *is* interpolable. The SDF wins precisely
  because it is a smooth interpolable field. This is the finding, not a methodology bias.)

### 2.3 SDF ramp-slope sweep @ render 192 — LANE flip rate
| slope (per px) | ramp half-width | lane flip% |
|---:|---:|---:|
| 192 | 0.66px | 10.30 |
| 96  | 1.3px  | 7.58 |
| 48  | 2.6px  | 3.19 |
| 24  | 5.3px  | 2.96 |
| 12  | 10.6px | 2.95 |

**Wider 1-Lipschitz ramp → strictly better lane survival, saturating at half-width ≥ ~5px** (slope ≤24).
A sharp SDF (slope 192) is **3.5× worse** on the lane than a wide one — confirms §1.4 (uint8 + downsample
footprint must be spanned by the ramp). The optimal-form SDF survival term wants a wide decision band,
not a knife-edge.

---

## 3. OSS / online grounding
- **Chlumský 2018, "Improved Corners with Multi-Channel Signed Distance Fields"** (CGF; `msdfgen`): single-
  channel SDF interpolation is accurate only where `|∇|` is ~constant; **sharp corners** (gradient
  discontinuities — e.g. lane **dash ends**) are where single-channel SDF degrades, and MSDF (median of
  RGB channels, ≥2 channels per edge) fixes them "by up to several orders of magnitude." → v2 refinement:
  the lane's dashes have corners; a per-class single SDF will round dash-ends. An **MSDF-style multi-channel
  encoding of the lane boundary** preserves dash corners.
- **Coverage / area anti-aliasing** (marching-squares SDF→polygon→area, DCAA coverage masks): the boundary
  should be stored as **fractional coverage spanning the resample footprint** — exactly the wide-ramp result
  of §2.3. Conservative rasterization = render the thin lane so its coverage is never quantized to 0.
- **Binary-mask resampling sampling-theory:** nearest-label downsampling drops sub-Nyquist (thin) classes;
  the cure is a smooth signed/coverage field — matches the hard-vs-SDF gap of §2.1.

---

## 4. What the v2 survival term must do (the actionable spec)
1. **Represent the partition as a 1-Lipschitz SDF level-set** (`argmax_k φ_k`), NOT hard labels / palette /
   per-class colors. Substrate already in-tree:
   `boundary_math/lever_b_levelset_generator.signed_distance_fields` + `lane_sdf_component` +
   `lever_b_levelset_generator` (the level-set witness). This probe gives them the quantitative survival
   curve + the prescription.
2. **Use a WIDE decision ramp** (margin band half-width ≳5px at render res; slope ≲24/px). Train the SDF so
   the top-2 margin `m=φ_top1−φ_top2` crosses zero over ≥ the resample+uint8 footprint. (Add a margin-band
   width regularizer / temperature on the SDF→logit ramp.)
3. **Render at the highest native res the byte budget allows; ≥192 is the knee** (SDF@192 ≈ hard@384).
   This is the resolution of the capacity-vs-rate trilemma for the *boundary*: the SDF rep lets a coarse
   (cheap) basis render the lane at sub-384 res and still survive R — ~2× effective-res headroom at equal
   bytes.
4. **Lane gets a dedicated channel** (MSDF-style multi-channel for dash corners; or a higher-res lane SDF):
   even SDF@192 leaves **3.19% lane flip** while every other class is <0.2%. The lane is the last holdout —
   it needs render ≥320 *or* a dedicated lane sub-channel. This is the ~8-dim lane-orbit long-tail, here
   localized to sub-wall (B).
5. **Keep R in-loop on the SDF carrier** (the trainer already does — `render_through_R_mlx`). The SURVIVAL
   term should be the realized-through-R `d_seg` on the SDF render (this probe's argmax model is the
   SegNet-free upper bound; the trainer's `cpu_verdict_d_seg` is the (C)-inclusive verdict).
6. **Do NOT spend bytes fighting R on the resample.** R is benign at scorer res. Spend the byte/capacity
   budget on (B) effective render-res of the SDF and (C) the SegNet-reading of the painted RGB.

---

## 5. Honest caveats / NO-FAKE
- SegNet-FREE: this probe models the SegNet as `argmax` of the membership carrier. It isolates sub-walls
  (A)+(B). The (C) SegNet-RGB-reading wall (the real palette ×170–350 death) is **separate** and is the
  trainer's realized-through-R job. **No contest score is claimed; pointer 0.19110 unmoved.**
- The carriers are built from `L*` (the correct partition) — this measures *survival of a correct
  partition*, the right question. A real witness must also *learn* `L*`; that's capacity/training, adjacent.
- Numbers are advisory `[macOS research-signal]` on n=96 cached GT; reproducible:
  `tools/r_survival_probe.py --n 96 --slopes 192,96,48,24,12 --render-res 96,128,192,256,320,384,874`.

## 6b. Scale-space / fluid-dynamics lens (heat-equation framing — tested, partially CORRECTED)
A lens proposed that R ≈ a DIFFUSION (the ↑874→↓384 anti-alias low-pass ≈ a heat kernel of effective σ);
"what survives R" = Koenderink/Witkin scale-space stability, and the SDF survives because its zero-level-set
is heat-stable. I TESTED it ($0, `--scale-space`); the lens is useful but the mechanism is REFINED:

1. **R's own kernel is small/benign** (edge-spread → effective σ): @384 σ≈0.38px, @874 σ≈0.52px, growing only
   to ≈1.2px @ render 96. An *isolated* thin bar survives R at every res (the bicubic ↑874 spreads it before
   the ↓384). So R's intrinsic diffusion is NOT the killer — consistent with §0(A) "R is benign."
2. **The heat-kernel claim for SDF survival is FALSIFIED for a thin minority class.** Applying the EXACT heat
   kernel (Gaussian blur of the carrier → argmax), the SDF does NOT beat hard on the lane — it is marginally
   WORSE (σ=1: hard 16.6% vs **SDF 26.8%** lane flip; σ=2: 74% vs 81%). Blurring averages the thin lane's
   small-magnitude φ_lane (≤~1, the lane is 2px) into its large-magnitude neighbors → the thin-class argmax
   collapses. The classic "level-set is heat-stable" result holds for a *single large region's* boundary
   (shift ≈ ½·κ·σ², small for low curvature), NOT for a thin minority class competing in multi-class argmax.
3. **The real mechanism is INTERPOLATION-exactness, not diffusion.** R = bicubic/bilinear INTERPOLATION
   (subsample→reconstruct) + a mild low-pass. The SDF wins R because bicubic/bilinear is **exact on the
   locally-linear 1-Lipschitz ramp** → the zero-crossing is reconstructed at sub-pixel precision across the
   resolution change. Decisive contrast at equal coarsening: HEAT(σ≈1) SDF lane 26.8% (loses) vs INTERP/R@192
   SDF lane 3.19% (wins). Same rep, different operator, opposite ranking → **R is interpolation-dominant.**

So: scale-space gives the right **Nyquist condition** (a width-w lane needs render res `r ≳ 384·(σ_target/w)`;
the measured cliff at r≈192 matches), but the rep-survival *ranking* is governed by interpolation-exactness on
1-Lipschitz ramps, not heat-kernel level-set stability. The v2 spec (§4) is unchanged and now mechanistically
grounded: use an SDF *because R interpolates* (and bicubic is exact on its linear ramp), render ≥192, wide ramp.
Data: `.omx/research/r_survival_probe_scalespace_n48.json`. (NOT-PESSIMISTIC: the lens was high-value — it
forced the decisive heat-vs-interp test that pinned the true mechanism. "rep vs test?": the heat test correctly
*distinguishes* the operators; neither rep nor R-test was wrong.)

## 6. Wire-in hooks (per Catalog #125)
1. sensitivity-map: ACTIVE — per-class survival-d_seg vs render-res is a per-axis sensitivity row (lane
   dominates). 2. Pareto: ACTIVE — render-res ↔ survival-d_seg is a rate↔distortion constraint for the
   boundary carrier (SDF@192 ≈ hard@384 = the dominating arm). 3. bit-allocator: ACTIVE — "spend on
   effective render-res of the SDF, not on resample-fighting." 4. cathedral autopilot: N/A (research probe,
   not archive-deployable). 5. continual-learning: this memo + JSON + DAG FEED-iw. 6. probe-disambiguator:
   `tools/r_survival_probe.py` IS the disambiguator between sub-walls (A) resample vs (B) capacity vs (C)
   SegNet-reading.
