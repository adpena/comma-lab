# HARD-FRAME MECHANISM ATLAS — visual analysis of the witness-own residual (#273)

**Source:** operator standing method 2026-07-16 (*"complement your analyses with a visual
analysis, looking at actual hard frames and pairs and tracing sensitivity and hardness
analysis to specific dynamics and failure modes visually"*) + the two same-day operator
mechanism namings (wet-hood specular rim; degraded/ragged lane markings). This arm traces
the `c2_witness_own_decomp_20260716` bucket table onto the actual imagery and NAMES the
physical dynamic at each hard region. **Pointer 0.19108 UNMOVED — MEANS** (c2 design
inputs). Axis: `[macOS-CPU advisory]`, frozen CPU-torch fp32 SegNet, bit-exact cached GT
(`gt_n600.npz`), frozen mod32cap EMA-best ep650 witness frames through exact R.
`research_only; score_claim=false; promotable=false`.

**Tool:** `tools/hard_frame_mechanism_atlas.py` (stages select/render/ladder/lane).
**Artifacts:** `experiments/results/hard_frame_mechanism_atlas_20260716/` — 22 extended
multipane montages + per-callout native-res crops + 22 resolution-ladder figures +
7 lane-degradation figures + 2 pose-path figures + selection/render/ladder/lane manifests
+ `scene_strip.png`. Operator copies: `~/Downloads/atlas_{montage_f518, crops_f074,
lane_f061, montage_f151, ladder_f061, scene_strip}.png`.

**Hard set (MEASURED selection from decomp rows):** worst overall {518, 74, 569} ·
roadlane {61, 192, 517, 84} · movable {575, 439, 546, 512} · horizon {137, 510, 110, 179}
· rim {522, 151, 442, 540} · easy controls {213, 393, 378}.

---

## 0. THE GLOBAL SCENE FACT (MEASURED — changes how every bucket reads)

**The entire 600-pair video is a NIGHT, WET-ROAD highway drive.** Mean luma 18–27/255 at
every sampled pair (0/75/…/599; `scene_strip.png`). Consequences, all visually verified:

- **Every separatrix runs at globally tiny GT margins** — Road-Lane median |m| 0.47–0.52
  logits AND Road-Undrivable 0.42 (n=7 hard pairs, `lane_manifest.json`): at night the
  whole scene sits within ~½ logit of flipping. Combined with the MEASURED Lane head gain
  (‖Δw‖ 3.75–4.01, flip d=|m|/‖Δw‖), per-frame boundary flicker is the *equilibrium
  behavior* of the frozen scorer on this footage, not a witness pathology.
- Wetness is scene-global: every light source paints a **specular streak column** on the
  pavement; the hood carries **visible red taillight reflections in most frames**
  (bottom of `scene_strip.png`).
- The "easy" control frames (d_seg 0.00198) are the SAME night scene with fewer light
  events/vehicles — residual structure identical (boundary flicker), only shorter.
  Difficulty is graded by traffic + glare events, not scene type.

## 1. Mechanism → model table (the deliverable)

Legend: mechanism labels are **NAMED-BY-EYE** unless marked MEASURED. "ladder" = flip-born
class from the exact A ker/range split (§3). "in c2?" = carried by the current
c2_surgical_warm dispositions (witness-own memo §4) — YES via which lever / OWED.

| bucket (% of residual) | named dynamic | evidence | deep-math object | cheapest treatment (train-least) | ladder | in c2? |
|---|---|---|---|---|---|---|
| **Road-Lane 66%** (a) | **Faded/ragged irregular dashes** (operator-named; VISUALLY CONFIRMED) | MEASURED: spacing CV 0.41–0.78 (comb ~0), blob tilt med ~62°, aspect med ~9.8, blobs 7–12/frame | irregular blob chain, NOT a periodic comb | **per-dash anchors** (arc-length s_i + extent/tilt) + trained phase stack matching the ACTUAL blobs | faithful | YES — c2 trained surgical stage; **#287 comb prior WRONG-FOR-THIS-VIDEO (confirmed)** |
| Road-Lane (b) | **Own-headlight illumination-cone gating** — paint exists only inside the static lit cone; each dash's margin is modulated as it advects through the cone edge; flips concentrate at the cone boundary (pairs 74#4, 522#3/#5: GT crops nearly black where GT still labels Lane) | NAMED-BY-EYE + crops | **static camera-frame multiplicative gain field g(x)** composing with advection: appearance = g(x)·φ(x−ξt) — a NEW structural factor for the (ξ,R) phase family | static luma-gate field in the render (near-0-byte: low-order radial/2D polynomial seed; the coord-INR carries static fields natively) | faithful | **OWED** — no explicit gate; trunk may learn it implicitly but unstructured |
| Road-Lane (c) | **Wet-road specular streak columns mimicking lane paint** — vertical bright reflection columns of scene lights read as Lane (Road→Lane false-positive side; pair 74#1 shows a textbook streak ringed by flips; 518#3–5) | NAMED-BY-EYE; colocalization measurement owed | **mirror transport** (highlight anchored to light source × camera, ~2× angular rate under ξ) — same specular-ξ family as the hood | witness must not brighten streak columns lane-like; cheap light-anchored vertical-streak seed; robustness term in phase stack, NOT new capacity | faithful | **OWED** — refines the L86 phase stack: needs TWO transport laws (surface ξ-advection AND mirror transport) |
| Road-Lane (d) | Taillight/headlight **bloom + retroreflector dot chains** washing out or mimicking dashes (74#2, 522#2) | NAMED-BY-EYE | transient glare events (light-anchored) | mostly GT-floor; phase-stack robustness; no new terms | faithful | YES (implicitly — no action) |
| Road-Lane (e) | **GT self-flicker / advection phase** — SegNet's OWN labels flip between the two GT frames of the SAME pair all along the lane lines | MEASURED: 840–1061 px/frame lane-band self-flips; **38–54% of witness lane-band residual sits on GT-self-flicker pixels** (7 hard pairs) | L85 GT sub-pixel advection phase, seen directly | **do not chase with capacity** — this share is at/near the label-noise floor; budget the recoverable Road-Lane share accordingly | — | YES (L85/L86 endgame) |
| **Movable border 21%** | **Motion-blur smear of close passing cars** + dark-on-dark rooflines + bloom halos + contact-shadow ambiguity (575#3/#5; 74#2 car ring; GT self-flicker also rings the car at pair 61) | NAMED-BY-EYE | ξ-tracked border with a **blur-width/soft-profile parameter** (crisp border is wrong under motion blur) | c2's ξ-tracked border profile + add blur-width to the profile family; part GT-floor | faithful | YES (border carriers); blur-width refinement OWED |
| **Horizon 15%** | **Night treeline vs dark sky ≈ zero-contrast boundary** + windshield droplet bokeh circles ON the horizon band + taillight bloom/tire spray crossing it (137, 151) | NAMED-BY-EYE; GT margin med 0.42 MEASURED | GT boundary itself soft/arbitrary at night | precision to Road side only (c2 disposition confirmed); do NOT spend on Undrivable-side texture — largely GT-floor | faithful | YES |
| **MyCar rim 9.6%** | **Specular reflections on the wet hood — CONFIRMED at night**: red taillight glows visibly smeared on the black hood; rim flickers full-width while the true boundary is static (persist 0.054) | NAMED-BY-EYE (operator mechanism, visually verified pair 151/518#2; 518#2 also shows the zero-contrast dark-rim variant) | static geometry + specular-ξ (mirror transport) appearance | **0-byte static rim freeze** (#139 clamp family) + specular appearance stays in the ξ-phase family — train NOTHING new | faithful | operator disposition (same-day); witness-side rim clamp still OWED |
| saddles 0.7% | — | — | — | no bytes (unchanged) | — | YES |

## 2. Sensitivity findings (VJP at the hard regions, all 22 frames)

- **luma_frac 0.83–0.88 on every hard frame** (VJP of summed margin deficit at all
  disagreeing px) — the luma-BT.601 cure-driver law holds exactly AT the hard regions,
  not just on random samples.
- The sensitivity heat is **spatially diffuse with a fine lattice imprint** (the shared
  resize A adjoint × stride-2 stem) — the non-local-cure law made visible: no hard region
  has a compact private cure footprint.

## 3. Resolution ladder (frozen_scorer_exact_factorization made visual; exact upstream ops)

CG orthogonal split of the camera-res witness–GT diff through the EXACT shared resize A
(modules.py:109, adjoint via autograd of the real call; residual |A·d_ker|/|A·d| ≈ 5.6e-6).

- **ker(A) fraction ≈ 52.8% of the witness's photometric deviation energy — on EVERY
  frame** (and 47–61% within hard callouts). Half the render's appearance deviation is
  structurally invisible to both scorers (blind B1) — the task-space vehicle exploits the
  blind complement by construction.
- **All 110 hard callouts classify "faithful"** (survival ratio ≈ 1.0; cam_rms 10–208 LSB):
  no camera-invisible flips and no blind-wasted flips. **The coordinator's sub-pixel-born
  vs blind classes DEGENERATE on this vehicle** because the render is non-photometric —
  the whole frame differs at camera res, so "camera-diff small but flip appears" cannot
  fire. On a PHOTOMETRIC vehicle (necessity/palette) the classes would separate; the
  witness's sub-pixel story lives in boundary GEOMETRY (1-px alternating flip bands), not
  diff magnitude.
- **Pose path (P2/P3, pairs 61/575):** the witness chroma diff is low-frequency, so 86.5%
  of its energy survives the 2×2 box-average — the pose-safety statement is specifically
  for FINE (<2px) chroma structure, correctly visualized in `posepath_f*.png`.

## 4. Contradictions / refinements routed to c2 (flagged LOUDLY where due)

1. **#287 dash-comb regularity assumption is WRONG-FOR-THIS-VIDEO** — MEASURED spacing
   CV 0.41–0.78 + visually irregular tilted elliptical blobs. Per-dash anchors replace the
   comb phase (degraded-lane memo consequence, now with numbers).
2. **Wet-road specular DOES re-explain part of Road-Lane flicker** (operator hypothesis
   confirmed on imagery): the Road→Lane false-positive side includes light-anchored streak
   columns. The L86 appearance-phase stack needs **two transport laws** — surface
   ξ-advection AND mirror transport (light-anchored, ~2× angular rate). Treating all
   appearance phase as surface-advected will mis-model the streaks.
3. **NEW structural prior — illumination-cone gate:** a static camera-frame multiplicative
   luma gain g(x) composing with advection, g(x)·φ(x−ξt). Near-free to carry (static
   low-order field); without it the phase stack attributes cone-edge margin modulation to
   per-dash appearance, wasting rate.
4. **GT-floor budget:** 38–54% of the witness lane-band residual sits on pixels where
   SegNet's label is not even stable across the two GT frames of the same pair. c2 should
   NOT budget the full 66% Road-Lane share as recoverable — the recoverable core is the
   coherent geometry (solid-line + near-dash placement); the far-field speckle is at the
   label-noise/advection floor. (Caveat: the overlap metric includes 1-frame advection, so
   it upper-bounds label noise — supporting, not proving, the floor.)
5. **Night-luma realization context:** at mean luma ~20 LSB, 1 uint8 LSB ≈ 5% relative
   luma — consistent with the evasion-arm finding that Lane cures need sub-LSB amplitudes
   (realization floor); night footage makes the uint8 lattice coarse relative to the
   evidence scale.

## 5. Round-1 review (own attack) + honest boundaries

- **NAMED-BY-EYE is an evidence class, labeled as such.** Cheap confirming measurements
  DONE: dash spacing CV/tilt/aspect · boundary margin medians · GT self-flicker + overlap
  · whole-video luma · ker/range split · luma_frac at hard px. OWED (cheap, named):
  specular-streak ↔ Road→Lane flip colocalization; margin-vs-cone-radial-position for the
  illumination gate; rim-flicker ↔ hood-reflection luma correlation.
- GT self-flicker uses SegNet(gt_f0) vs cached argmax(gt_f1) — includes ~1/20s real
  advection, honest upper bound only.
- Viewed in detail: 8/22 montages + 4 crop sheets + 2 lane figures + 1 ladder + 1 posepath
  + scene strip; remaining figures rendered and manifest-summarized but not eye-read
  (worst 569, roadlane 84/192/517 crops, movable 439/512/546, horizon 110/179/510, rim
  442/540, easy 378/393). The named mechanism set was already saturating (every newly
  opened figure repeated the same dynamics); risk of a missed distinct mechanism is
  nonzero and stated.
- The witness decomposed is mod32cap ep650 (not the c2 vehicle) — same caveat as the
  parent memo §7; mechanism namings are scene+scorer facts and transfer; bucket weights
  may shift.
- Selection is top-K per bucket (hard tail), not random — by design; easy controls
  confirm the structure is shared.

## 6. Triality + stores consulted

- **DAG:** FEED-visual-atlas appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **equations:** no new law registered — the atlas NAMES mechanisms feeding existing
  laws (witness_own_residual_decomposition_v1 · segnet_head_rank4_linear_flipdist_v1 ·
  L85/L86); the illumination-cone gate + mirror-transport refinement are recorded here +
  DAG as OWED design inputs to the c2 phase stack (law registration follows their first
  measured carrier, per FORMALIZATION_PENDING discipline: the gate/transport terms have
  no measured equation-grade anchor yet).
- **DSL:** no lever wired (analysis-only); owed items routed to c2 as design inputs.
- **STORES CONSULTED:** c2_witness_own_decomp (parent) · c2_perclass_stratum taxonomy ·
  frozen_scorer_exact_factorization (ladder stages) · segnet_recursive_fractal_factorization
  (head gain) · degraded_lane_markings memory · mycar_rim specular memory · L65/L68/L73/
  L80/L85/L86 · adversarial_evasion_fisher_null (realization floor) · #401 blind-coordinate.

**Pointer 0.19108 UNMOVED — MEANS.**
