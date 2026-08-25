# OPENPILOT DEPTH × THE MORSE-SMALE TEXTURE LAYER — witness-term design (answers §6(b)+(c)) — 2026-07-08

**Operator directive (verbatim, binding):** "openpilot and research tell us how to handle the depth and
parallax issues ... we may need to tweak our witness and level set and morse smale and terms to render
something more precisely segmented by class and depth as appropriate for conditioning with the necessary
information." This memo answers the two open dives fired by
`pose_legible_witness_aperture_design_20260708.md` §6: **(b)** openpilot depth, concretely, and **(c)** the
Morse-Smale-native texture layer. Pointer **contest-CPU 0.19110 UNMOVED — MEANS**. Design + $0 plan only;
nothing here is a score until byte-closed through `upstream/evaluate.py`. Every number tagged
MEASURED / DERIVED / PREDICTED.

STORES CONSULTED: `pose_legible_witness_aperture_design_20260708` (the §6 parent — the aperture diagnosis) ·
`pose_mladder_depthwarp_measured_20260708` (70649531f — **A0 1.685 / A2 1.486 / A2+ 1.223** MEASURED,
off-plane mass ≈0.5%, corr(d_pose,|t|) NEGATIVE) · `pose_taskspace_native_morse_smale_depth_warp_design`
(the per-region warp law + D1–D4 review corrections) · `openpilot_world_model_free_prior_v2` (lane-poly
oracle floor 0.00214 MEASURED; supercombo output inventory; **no per-pixel depth head**) ·
`comma_openpilot_crossref_polynomial_geometry` (K=910/(582,437), h=1.22, POLY_PATH_DEGREE=4, ground
homography) · `comma_openpilot_domain_tricks` (FastViT RepMixer local-to-mid RF = luma-parallax reader;
row-depth `v=910·1.22/d+437`; chroma-is-task-safe) · `SPEC_v8_perclass_decomposition` (edge-centric
per-class carriers, MERGE→DIFF→CORRECT, chroma-first/luma-reserved) · `SPEC_v75_optimal_single_trunk` §8
(operating contract) · `xi_pose_coder.py` (ξ store-nothing, ~2.7 KB MEASURED, derive-H FREE) ·
`stratified_depth_warp.py` (affine off-plane flow, bit-parity to A0) · CLAUDE.md L80 class order (MEASURED).

---

## 0. THE LOAD-BEARING REFRAME (read first — it decides what depth is FOR)

The measured ladder settled one thing hard: **depth-as-pose-steering is bounded ~10%.** A2+ (a 12-DOF warp
with an ORACLE off-plane mask, solving depth flow DIRECTLY to the PoseNet target) improved d_pose only
1.486→1.223 (−10%, MEASURED n8) over the 6-DOF floor, because off-plane parallax mass is ≈0.5%
(MEASURED). So **the pose win does NOT live in the depth field.** It lives — IF anywhere — in the
**texture layer restoring flow OBSERVABILITY** (the aperture diagnosis: a flat cartoon pair makes optical
flow unobservable except normal-to-boundary → PoseNet under-informed → d_pose floors ~1.7).

Therefore depth is re-scoped to its **RIGHT job**: not steering the 6 pose outputs, but making the
**per-cell advection flow CORRECT** so the texture we paint moves the way the real scene moves. Ground
cells (≈99.5% of observable flow) get EXACT flow from `H(ξ)` at **zero** stored depth; off-plane cells get
a small counted correction. This is the operator's "segmented by class **AND** depth": class gives the
partition (d_seg), depth gives the per-cell flow model (d_pose advection). The whole design is GATED on the
A0T $0 probe (aperture §4): if seeded texture advected by the free ground flow does not drop A0's 1.685,
the aperture diagnosis is wrong and pose reverts to a budget item — no depth or texture build fires.

---

## A. OPENPILOT DEPTH, CONCRETELY — the per-cell depth table (§6b)

### A.1 The geometry openpilot actually gives us (source-confirmed)

- **Intrinsics (CONFIRMED, HIGH):** EON/neo fcam 1164×874, focal 910.0, principal point at center →
  `K_native=[[910,0,582],[0,910,437],[0,0,1]]` (`common/transformations/camera.py`), byte-identical to
  the contest `frame_utils.py` (`camera_fl=910`, `camera_size=(1164,874)`). At the SegNet 512×384 grid
  (uniform 0.44× resize): `K_512≈[[400.3,0,256],[0,399.5,192],[0,0,1]]`. rule-118: GENERIC constant → FREE.
- **Ground plane (CONFIRMED, HIGH):** camera height **h=1.22 m**, pitch ≈ −0.02 rad, roll≡0; plane normal
  `n=[0,−cos p,−sin p]`. Ground depth is closed-form by row: **d(v) = fy·h/(v−cy) = 910·1.22/(v−437)**
  native (MEASURED row-formula, domain-tricks); horizon at v=cy (437 native / 192 at 512×384).
- **openpilot has NO per-pixel depth head (CONFIRMED MED-HIGH, absence-of-evidence).** supercombo emits
  `laneLines`(4)/`roadEdges`(2)/`position`/`pose[5755:5761]` + `leadsV3` (lead **dRel** distance) — depth
  is IMPLICIT in lane/edge/position z and lead distance, NOT a shippable depth raster. **So building/
  structure depth is NOT free from openpilot** — it is a small counted fit or a coarse geometric prior.
- **FastViT-T12 = RepMixer, local-to-mid receptive field (CONFIRMED, domain-tricks):** PoseNet recovers
  ego-motion from **local mid/high-freq LUMA parallax** across the 2-frame pair. This is the scorer-side
  fact that makes the texture layer luma-primary (§B.4).

### A.2 The minimal SUFFICIENT depth per cell — FLOW-correct, not pixel-accurate (DERIVED)

Plane+parallax (Irani–Anandan–Weinshall): real flow `u(p) = u_planar(H(ξ);p) + γ(p)·ê`, where
`γ(p) = (t_z/Z(p))·(perspective)` is the parallax MAGNITUDE and `ê` the epipolar direction. **PoseNet
reads flow, not depth** — so we never need metric Z(p). Two facts collapse the parametrization:
1. the ego trajectory is **rank-1 forward speed** (dim0 var ≈700× the next, MEASURED), so the epipole is
   the FOE ≈ the vanishing point (256,174 @512×384); `ê(p) = (p−FOE)/|p−FOE|` is **known from ξ, 0 params**.
2. within a Morse cell (a building facade, a car flank) inverse-depth is smooth → **affine inverse-depth**
   `1/Z(x,y) = α + βx + γy` (3 params/cell) captures the dominant slanted-planar structure.

**Minimal sufficient per-cell flow model = 3 params (affine 1/Z) × the KNOWN radial ê(ξ).** This is a
tighter, geometry-constrained refinement of the landed `stratified_depth_warp.affine_extra_flow` (6 free
du/dv params): factor the field into (counted 3-param inverse-depth) × (free radial epipolar direction from
ξ), fewer DOF, less overfit, and it drives sky→(1/Z→0)→rotation-only AUTOMATICALLY (D2 correction, no
brittle horizon hard-split).

### A.3 THE PER-CELL DEPTH TABLE (the §6b deliverable)

| Morse cell (class, MEASURED order) | depth / flow model | source | COUNTED bytes |
|---|---|---|---|
| **Road (0)** — 22.9% | ground plane, flow = `H(ξ)` exact, γ≡0 | closed-form EON calib (K, h=1.22, pitch) + stored ξ | **0** (derived, rule-118 FREE) |
| **Lane (1)** — 0.59% | same ground plane `H(ξ)` | same | **0** |
| **Undrivable-SKY (2, above horizon)** | ∞ → rotation-only `R(ξ)`, 1/Z→0 | auto from affine-1/Z fit (D2) | **0** |
| **Undrivable-STRUCTURE (2, finite vertical)** | affine 1/Z (3 params) × radial ê(ξ) | compress-time fit (no openpilot depth head) | **~3 fp16 × n_struct cells** (~30–120 B, ~0.5% mass) |
| **Movable (3) — tracked leads** — part of 1.56% | inverse-depth 1/Z from lead **dRel** | supercombo leadsV3 / comma2k19 (FREE compress-time) | **~0** (sourced) or ~2 B/lead |
| **Movable (3) — other islands** | one 1/Z scalar / island × radial ê(ξ) | compress-time fit | **~2 B/island** (sparse) |
| **MyCar / hood (4)** — 25.6% | identity (rigid w/ camera → flow≈0) | static, IoU 0.994 MEASURED (#139) | **0** |

**Honest bound (MEASURED):** ~87% of image area (Road+Sky+Hood) contributes **0 counted depth bytes** and
EXACT free flow. The counted depth (structure + movable, ~2% area, ~0.5% parallax mass) buys AT MOST ~10%
on the warp cap (A2+ ceiling, MEASURED). Depth is the advection-correctness companion, not the pose lever.

---

## B. THE MORSE-SMALE-NATIVE TEXTURE LAYER (§6c)

### B.1 Formalization — texture lives in the cell INTERIOR, d_seg-safe BY CONSTRUCTION

The v8 partition is `P(x)=argmax_c(φ_c(x)+b_c)`; separatrices = tie loci (DERIVED, never represented). The
**SDF-gauged margin field** `m(x) = φ_(1)(x) − φ_(2)(x)` (top-two gap; 1-Lipschitz under the eikonal gauge →
in pixel units) defines the safe interior `Ω_int = {x : m(x) > m_0}`. Texture is painted ONLY in `Ω_int`
with amplitude `a(x) ≤ κ·m(x)`:

- **d_seg-safe by construction, not by empirical margin (#141).** At a textured pixel the argmax flips only
  if the luma perturbation moves `φ_c` by more than `m(x)`; capping `a(x) ≤ κ·m(x)` (κ<1, through the R
  response) makes the flip IMPOSSIBLE. This is a HARD SDF gate, strictly stronger than the measured p50≈0.9
  margin headroom the aperture memo cited. Texture never touches a separatrix → **births no new tie locus →
  the length/persistence term is neutral** (same guarantee).
- **Seeded (rule-118 FREE):** texture `T(x) = g(x; seed)` — a deterministic generator (value/gradient
  noise, or a low-order Fourier sum) from ONE global `seed`. Zero counted bytes for realization; only the
  seed (few B) is stored. Same field every frame.
- **ξ-consistent advection (the aperture fix):** `f1_T(p) = f0_T(p − u_cell(p))` with `u_cell` the §A
  per-cell flow. frame0 texture is **SegNet-FREE** (SegNet reads `x[:,-1]`=frame1 only → f0 fully
  unconstrained, unlimited amplitude); frame1 texture is sub-margin in `Ω_int` and in the R-surviving band.
  The pair is consistent by construction → the 10.42 "mixed real+cartoon" pathology cannot recur.

### B.2 PLACEMENT — decode-side first, in-training only on measured evidence (RECOMMENDATION + reasoning)

**Recommend: build the texture layer DECODE-SIDE (post-argmax paint), gate an in-training escalation on a
measured shortfall.** Reasoning, decisive:

| | DECODE-SIDE (post-argmax paint) | IN-TRAINING term |
|---|---|---|
| d_seg training | **UNCHANGED** — every v7.5/v8 seal stays valid | RE-COUPLES d_seg↔d_pose in the gradient (v8-risk-2 theft channel reopens) |
| staged-training discipline (v8 §8) | respected (paint solved vs frozen scorer, separately) | VIOLATED — needs its own measured justification |
| risk | frame1 texture bounded by the SDF gate; f0 free | render can drift; τ/eikonal/length interactions live |
| what it can buy | the full aperture win IF PoseNet reads correct motion from a textured-but-fixed partition | co-adaptation (render moves so the warped pair better hits the pose target — the R1 0.0011 joint-training hypothesis) |
| A0T probe home | **native** (A0T is exactly a post-hoc paint on fixed renders) | n/a |

Decode-side is simplest, zero-risk to the hard-won d_seg, and is *exactly the A0T probe surface*. Escalate
to an in-training term ONLY if decode-side A0T is GREEN but insufficient to reach the pose target — and then
under the staged protocol (texture/paint stage vs the frozen scorer, never end-to-end through the
composite). This matches the aperture memo §5 ("LATE/decode-side layer OR in-training if co-adaptation pays
— probe decides") and the mladder synthesis ("the cure is a dedicated joint pose-descent RUN, not a
post-hoc carrier") — decode-side settles whether a RUN is even needed.

### B.3 The R-survival two-sided amplitude window (MEASURED constraints)

Texture must survive R (`bicubic↑874→uint8→bilinear↓512×384`, #204): **mid-spatial band**, amplitude
**≥1–2 uint8 steps** (below that, the round erases it — the L4 dash-erasure/Gibbs lesson) and **≤ κ·m(x)**
(above that, argmax flips). Two-sided window `a(x) ∈ [uint8_step, κ·m(x)]`. Where `κ·m(x) < uint8_step`
(the low-margin annulus — ~97% of d_seg lives there, MEASURED #333), the window is EMPTY → **that pixel gets
NO texture, stays flat** — automatically keeping the annulus untouched (where d_seg is decided) and texturing
only the safe high-margin interior (where flow observability is needed and free). This is a feature: the
same SDF gate that protects d_seg concentrates texture exactly in the interior the aperture argument needs.

### B.4 INTERACTION AUDIT — every term the texture layer could destabilize + its guard

- **τ-anneal (τ=ε=ħ, Maslov edge width #284):** as τ sharpens, the margin field `m(x)` changes → a
  sub-margin amplitude at a soft-τ step could flip at the hard-τ decode. **GUARD:** decode-side placement
  freezes τ at decode (interaction vanishes). If ever in-training, gate texture amplitude on the
  **hardest-τ (final) margin**, or add texture only in the final constant-τ turnpike stage (DAG FEED-v7seal:
  TAIL@τ0=τ_end). Never add texture during the annealing transient.
- **eikonal (|∇φ|=1, w=0.01):** the SDF gauge on the fields `φ_c`. Texture is an APPEARANCE layer on the
  painted luma, NOT on `φ` → decode-side cannot touch eikonal. **GUARD (in-training only):** stop-gradient
  from the texture/pose loss to the eikonal-constrained field parameters; route texture only to the
  paint/appearance head. A texture-loss path into `φ` would corrupt the gauge — forbid it structurally.
- **length / persistence (w=0.001; preserves lane dashes):** penalizes boundary length + low-persistence
  features. Texture in `Ω_int` births no new argmax edge (SDF gate) → **length term neutral**. **GUARD:**
  the interior gate IS the guarantee (a=0 wherever m<m_0); verify at decode that argmax(R(textured f1)) ==
  argmax(R(flat f1)) bit-for-bit (the aperture §4 d_seg guard: "measure, don't assume").
- **chroma / palette (#276, v8 §3 channel routing):** SegNet reads full RGB (chroma fully argmax-visible →
  a chroma perturbation costs MORE d_seg budget/unit than luma); PoseNet reads YUV6 with SUBSAMPLED chroma
  and is luma-dominated (FastViT local-luma-parallax reader, MEASURED). **GUARD:** texture is **LUMA-ONLY**;
  chroma stays the flat per-cell palette. This is EXACTLY v8's luma-reserved/chroma-first split — texture is
  a luma-warp-coherence layer living in the reserved luma channel, orthogonal to the chroma-first seg
  repairs (the near-triangular correction Jacobian, v8 §3) → they COMPOSE, do not fight.
- **temporal screw-consistency (v8-risk-4/P0 force):** texture advected by `u_cell(ξ)` shares the ONE
  coherent luma warp field → it REINFORCES screw-consistency rather than fighting it (luma stays one
  warp-structured field). No new guard; it is a companion.

---

## C. v7.5 / v8 COUPLING + minimal schema (§6b/c)

- **v8 (edge-centric per-class carriers) — NATIVE fit.** Each per-class carrier already owns its cell's
  support; the texture layer is a per-carrier sub-layer over that carrier's interior, advected by that
  class's depth model from the table. Depth augments each carrier: `{class_id, depth_model_id, params}` =
  the operator's "class AND depth." No new sharing — texture rides the support the carrier already paid for.
- **v7.5 (single trunk) — decode-side only, no trunk change.** The trunk produces the frozen argmax
  partition + the SDF margin field `m(x)` (already there: `φ_(1)−φ_(2)`) + ξ. Texture consumes those at
  decode. The trunk needs to EXPOSE `m(x)` and the per-cell class map (it does). Nothing in the trunk moves
  → v7.5 launches/seals unaffected; texture is a pure additive decode layer gated on A0T.

**Minimal schema (per connected Morse cell + global):**
```
per-cell (only cells 2-STRUCTURE and 3 need explicit rows; 0/1/4 derive depth_model from class):
  { class_id: uint8,               # already stored for d_seg (v8 field) — not new
    depth_model_id: uint8,         # {GROUND=0, SKY_ROT=1, HOOD_ID=2, LEAD_INVDEPTH=3, AFFINE_INVDEPTH=4}
    depth_params: fp16[k] }        # k=0 (ground/sky/hood) | k=1 (lead 1/Z) | k=3 (affine α,β,γ)
global:
  { texture_seed: uint32, texture_band: fp16[2], texture_amp_kappa: fp16 }   # ~10 B, drives g(x;seed)
reused (NOT new): xi payload ~2.7 KB (xi_pose_coder, delta_res, MEASURED)
```
`depth_model_id` for classes 0/1/4 is a deterministic function of `class_id` → only off-plane cells (2,3)
store rows. Everything is per-CLIP / slowly-varying (lanes+structure near-static in the ground frame),
ξ-propagated per-pair for free — **never per-pair-per-cell** (see §D firewall).

---

## D. RATE ACCOUNTING (honest) + the hide-data-in-code firewall (§6, NO-FAKE #6)

**Counted bytes (the honest account):**
| item | bytes | rule-118 |
|---|---|---|
| ξ trajectory (REUSED, not new) | ~2700 (MEASURED, delta_res n600) | COUNTED (already) |
| texture seed + band + κ | ~10 | seed COUNTED; **realization `g(x;seed)` FREE** |
| undrivable-structure affine 1/Z | ~30–120 (3 fp16 × ~5–20 cells, DERIVED) | COUNTED |
| movable 1/Z per island | ~2–40 (leads sourced free) | COUNTED |
| **NEW counted total** | **~50–170 B** | on top of existing ξ |

Rate contribution `25·(2700+~170)/37.5M ≈ 0.00191` vs ξ-only `≈0.00180` — **negligible increase (DERIVED)**.
The rate axis does NOT bind (D3: ~6.3 MB affordance). The prize is the pose term: warp cap d_pose≈1.5 →
`√(10·1.5)≈3.9` DOMINATES S; if the aperture fix reaches ancestor-class d_pose~1e-3 → `√(10·1e-3)≈0.1`.
That drop is the entire value — and it is **UNMEASURED, gated on A0T** (depth alone buys only ~10%, MEASURED).

**Texture realization is genuinely FREE (confirmed):** `g(x;seed)` is a generic deterministic generator
(same field every frame, advected by the counted flow) → the only stored entropy is the seed. The FLOW is
video-derived (ξ + depth params, COUNTED); the TEXTURE is generic (seed, FREE); the advection is generic
algorithm (FREE in inflate.py). Clean rule-118 line.

**hide-data-in-code firewall (NO-FAKE #6) — two named risks + guards:**
1. **Per-pair texture fit.** If the texture field were OPTIMIZED per-pair to hit the pose target and that
   per-pair field smuggled into inflate.py as "code" → the hide-data fake. **FORBIDDEN.** Guard: texture is
   SEEDED (one global seed → identical field all frames), carries ZERO per-pair video entropy; enforce via a
   byte-mutation smoke (mutate the seed → all frames change identically; no per-pair table exists).
2. **Per-pair-per-cell depth table (600× the cells).** A dense per-pair depth table hidden in code = fake;
   even stored it is expensive. **GUARD:** store per-CLIP (or low-rank temporal) depth params, ξ-propagate
   per-pair (the openpilot ground-frame near-static property makes this exact-enough). Any per-pair depth
   entropy stays COUNTED and is a red flag if it grows with P.

---

## E. THE GATE + BUILD OBLIGATIONS (means discipline)

- **Nothing builds before A0T (aperture §4, $0):** A0 (ground-`H(ξ)` warp, consistent pair) + seeded
  sub-margin texture advected by the same flow, measured vs A0's **1.685** (MEASURED) through the frozen
  CPU-torch PoseNet, n→600, real gt, through R; d_seg guard = argmax flips vs untextured must be ~0
  (MEASURE). If A0T ≈ 1.685, the aperture diagnosis is FALSIFIED at formulation level → pose reverts to a
  budget item; no depth/texture term ships. If A0T ≪ 1.685 → build decode-side, then A2T (solve on textured
  pair), then escalate to in-training only on measured shortfall.
- **DSL Lever obligation (build phase):** the texture layer + per-cell depth model land as
  `curriculum_dsl` `Lever` factories (texture_seed / band / κ, and the per-cell depth_model_id/params
  schema) — NEVER a hand-added trainer/decoder flag at finalize time (config-orphan confound). Register in
  `lever_registry`; surface via `completeness().unmapped`. Decode-side placement still registers (swept
  amp/band/seed = swept intent).
- **verdict_scope:** the A2/A2+ negatives are FORMULATION-level (post-hoc warp of a FIXED flat render) — this
  design changes WHAT is rendered (adds the missing flow-carrying texture), the door verdict_scope left open.
  No anchor discarded.

## TRIALITY / EQUATION
This memo is DESIGN — no measured anchor, so `morse_smale_stratified_parallax_dpose_v1` stays
COUNCIL-FLAGGED (anchor owed to A0T + byte-close). Commit tagged `[no-triality]` (design memo, no lever
fired yet); the DSL Lever + canonical-equation registration are OWED at the build phase per the obligations
above. Pointer **0.19110 UNMOVED — MEANS**.

---

## Observability surface

*(OBSERVABILITY-ADDENDUM 2026-08-25 — APPEND-ONLY per Catalog #110/#113. This
section is an INDEX into this memo's own content per Catalog #305's 6 facets;
it adds no new claim. Facets with no counterpart in this memo say so plainly.)*

1. **Per-layer inspection** — §A.3 "THE PER-CELL DEPTH TABLE" and §B.1 "Formalization — texture lives in the cell INTERIOR, d_seg-safe BY CONSTRUCTION" separate the depth layer from the texture layer, each inspectable per cell.
2. **Per-signal decomposition** — §D "RATE ACCOUNTING (honest) + the hide-data-in-code firewall" decomposes the term's byte cost; §B.3 "The R-survival two-sided amplitude window (MEASURED constraints)" decomposes the survivable amplitude range.
3. **Run-to-run diff** — §B.2 "PLACEMENT — decode-side first, in-training only on measured evidence" fixes where the term enters, so a build with the term differs from one without at exactly that placement; §C gives the minimal schema that makes two builds comparable.
4. **Post-hoc query** — named surfaces are `stratified_depth_warp.py`, `frame_utils.py`, `inflate.py` and openpilot's `common/transformations/camera.py`; the authority is `upstream/evaluate.py`.
5. **Cite-chain** — §A.1 "The geometry openpilot actually gives us (source-confirmed)" is source-confirmed rather than asserted, and 20 claims in this memo carry an explicit MEASURED label.
6. **Counterfactual hooks** — §B.4 "INTERACTION AUDIT — every term the texture layer could destabilize + its guard" is a per-interaction counterfactual with a named guard each; §E "THE GATE + BUILD OBLIGATIONS" is the pre-registered gate.
