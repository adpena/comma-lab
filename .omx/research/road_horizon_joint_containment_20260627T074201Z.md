# Sky/undrivable + road-as-complement components AND the FIRST JOINT-CONTAINMENT measure — FEED-dw (the FEED-dt integration-seal begun)

**UTC:** 2026-06-27T07:42:01Z
**Lane:** `road_horizon_joint_FEED_dw`
**Authority:** `[macOS-CPU advisory]` research-signal — `score_claim=false`, `promotable=false`,
`ready_for_exact_eval_dispatch=false`. $0 CPU-only (numpy + scipy; NO torch/MPS/GPU). NOT a byte-closed
row, NOT a trained-witness output. Frozen CPU-torch SegNet argmax partition (cached `lstars` in
`gt_n96.npz`, bit-exact per FEED-db; NO surrogate, NO new scorer pass). **GPU UNTOUCHED** (levelset
descent pid 72600/72602 + pose subagent a1116e516f ran concurrently; this task read ONLY `lstars`
~150 MB, n96; light memory — one per-frame SDF stack ~4 MB freed each iteration).

Operator 2026-06-27: complete the structured-region decomposition (lane FEED-ds + hood FEED-dv + **sky +
road**, this memo) and begin the FEED-dt integration-seal — its **highest-risk item**: *"do the
structured components COMPOSE without antagonism in ONE argmax?"* Additive, default-off; does NOT disturb
the critical path.

<!-- FORMALIZATION_PENDING: the canonical equations "sky/undrivable = static-majority-mask SDF (dominates
the horizon-line half-plane)", "road = level-set COMPLEMENT (a deep background field, NOT a const)", and
"locally-supported structured SDF components compose ADDITIVELY in one argmax" are queued for
tac.canonical_equations registration once a byte-closed witness row confirms the predicted d_seg benefit. -->

---

## 0. TL;DR (the decisive answers)

Measured n96, $0, frozen CPU-torch L*, isolation + joint via the validated lane/hood template (build
structured field → inject as phi_k → recompute the argmax vs L* → full per-class confusion decomposition).
All classes SELF-DETECTED from the data (no hardcoded index): **road=0, lane=1, sky=2, movable=3, hood=4**.

| variant (others ideal unless noted) | total d_seg | note |
|---|---|---|
| IDEAL (all 5 ideal SDF) | 0.000000 | harness baseline (argmax==L* exactly) |
| ISO lane (continuous band, FEED-ds optimal) | 0.000439 | reproduces FEED-ds (~0.00042) |
| ISO hood (static mask, FEED-dv) | 0.000737 | **reproduces FEED-du exactly → harness validated** |
| **ISO sky — static MASK** ✅ | **0.004827** | the sky optimal form |
| ISO sky — horizon LINE (2-float/frame) | 0.012696 | **2.6× WORSE → DOMINATED** (sky ≠ half-plane) |
| ISO road — CONSTANT complement | **0.000000** | **road IS the complement (free) when others are ideal** |
| **JOINT — structured MUTUAL (road IDEAL)** | **0.005993** | **≈ Σ isolations 0.006003 → ADDITIVE (antagonism ~0)** |
| JOINT — road-CONST complement | 0.014337 | road-const marginal **+0.008344** (the SOLE antagonist) |

**1. SKY/UNDRIVABLE = static-majority-mask SDF (ADOPT); the horizon-LINE is dominated (REJECT).** The
static mask (per-frame IoU mean **0.982**, min **0.952**, frac 0.494, rows [0,218]) captures the sky to
total **0.004827**; the per-frame 2-float horizon line is **2.6× worse** (0.012696, fit rms 10.2 px)
because the SegNet "undrivable" region is **not a clean half-plane** (buildings/structure/foliage poke into
it). Same template + verdict shape as the hood (continuous/mask beats the parametric gate). **Honest gap:**
sky shape FN **0.002409 is ABOVE the witness target 0.00087** — sky drifts ~9% (vs hood's ~1%), so the
static mask is good-and-cheap but its drift is a residual the witness must refine (≠ the hood's near-perfect
static capture). Cost **98 bytes** for the whole mask over 600 frames (0.16 B/frame, rate 6.5e-5).

**2. ROAD = the level-set COMPLEMENT — NO dedicated road component is needed, BUT a CONST is not enough.**
Road-as-complement is **free in isolation (0.000000)**: with the other classes given as deep ideal fields,
a constant phi_road wins **exactly** the complement of the positive regions → road falls out for free, no
polynomial/mask sidecar. **HOWEVER, in the JOINT a CONST road ANTAGONIZES (+0.008344 marginal, ratio
2.39):** once the other classes are *approximate* structured fields (shallow just inside their boundaries),
a const floor=0 cannot defend the road boundary — it claws/loses boundary pixels (road FN 0.0011→0.0075,
lane FP 0.0003→0.0050). **Verdict: road needs a DEEP field (the witness's learned, eikonal-regularized
phi_0), not a const and not a hand-built sidecar.** Road = the witness's base channel (0 video-derived
sidecar bytes), but it is *learned*, not free. The const is a $0 lower-bound probe, not the codec road.

**3. THE JOINT-CONTAINMENT MEASURE (FEED-dt's highest-risk item — answered):**
   - **(a) The structured components COMPOSE ADDITIVELY.** With road given deep (ideal), the joint of
     lane+hood+sky = **0.005993 ≈ Σ isolations 0.006003** (antagonism **−1e-5**, i.e. ZERO). The locally-
     supported SDF components do NOT fight in the shared argmax — exactly the structural-containment thesis
     (each phi_k is positive only inside its region; deep neighbors dominate just outside). **The ONLY
     antagonism is the road-CONST** (+0.008344); fixable by a deep road field.
   - **(b) Per-class containment HOLDS in the joint (road-ideal).** Every structured class's FN/FP in the
     joint ≈ its isolation value: lane 0.000134/0.000307 (iso 0.000133/0.000306), sky 0.002409/0.002419
     (iso identical), hood 0.000714/0.000102 (iso 0.000634/0.000102). **No class-A leaks into class-B**
     beyond isolation. (With road-CONST the containment breaks: road FN + lane FP blow up — the const can't
     hold boundaries.)
   - **(c) The RESIDUAL the witness must LEARN** = structured fine-boundary floor **0.005993** (dominated
     by the sky-mask drift 0.0024; the coarse EDT-rasterized fits place boundaries a few px off) **+ Movable
     class mass 0.015569** (class 3 has NO static structure → fully learned; if not given, +0.0268 d_seg).
     **Movable is the larger learned chunk.**

**Honest, load-bearing caveat:** the structured floor 0.005993 is **ABOVE the current frontier d_seg
0.00056**. The coarse EDT-rasterized structural decomposition is a **PRIOR/INIT, not the final d_seg** — it
gives the regions cheaply+contained+additively so the witness's capacity goes to the **boundary annulus +
Movable**, but the witness must still LEARN deep SDFs (eikonal) to tighten boundaries toward ~0.001.

**R-survival (n8):** ideal 0.000010; joint(road-IDEAL, sky-mask) **0.003920**; joint(road-CONST, sky-mask)
0.011036 — the clean (deep-road) structured joint R-survives at sub-floor; the const-road joint does not.

---

## 0b. NO-FAKE region self-detection (the mandatory first finding)

`classify_segnet_regions` detects ALL 5 comma10k roles from the REAL cached argmax (never hardcodes an
index — the discipline FEED-du/dv enforced after the FEED-dn luma-sort mislabel). Falling-rule:
hood = argmax(bottom_share·IoU) → sky = argmax(top_share·IoU) → road = argmax(area) of the rest →
lane = argmax(lane-lines fitted) of the last two (tiebreak: thinner = lane) → movable = last.

| cls | static IoU | area-scaled frac | top_share | bottom_share | lane-lines | maj rows | → role |
|---|---|---|---|---|---|---|---|
| 0 | 0.330 | 22.04 | 0.000 | 0.003 | 1 | [175,291] | **road** (largest mid band) |
| 1 | 0.000 | 0.57 | 0.000 | 0.001 | **5** | [189,223] | **lane** (most ground-plane lines) |
| 2 | 0.907 | 47.35 | **0.507** | 0.000 | 2 | [0,218] | **sky/undrivable** (static TOP) |
| 3 | 0.000 | 1.50 | 0.000 | 0.000 | 1 | [173,218] | **movable** (dynamic, no structure) |
| 4 | 0.963 | 24.55 | 0.000 | **0.975** | 1 | [282,383] | **hood** (static BOTTOM) |

Detected `{road:0, lane:1, sky:2, movable:3, hood:4}` — matches comma10k (CONFIRMED FEED-da + horizon-band
memo) AND is robust under arbitrary label permutation (a unit test relabels the classes and the roles
follow the pixels). The lane-vs-movable disambiguation is decisive: the lane fits **5** ground-plane lines
(the canonical IPM clusterer), movable fits 1.

---

## 1. The designs (each REUSES the lane/hood machinery — no duplication)

- **Sky static mask** = `compute_static_hood_mask(lstars, hood_cls=sky)` (class-agnostic) → majority-vote
  mask → `build_static_sky_sdf` (scipy-EDT, the SAME 1-Lipschitz construction as `signed_distance_fields`).
  ONE field for all 600 frames.
- **Sky horizon line** (the measured alternative) = per-frame `fit_horizon_line` (per-column top-contiguous
  sky boundary → robust degree-1 polyfit, building-silhouette outliers trimmed) → `rasterize_sky_above_line`
  half-plane → EDT. 2 floats/frame. **Measured dominated** — kept as the falsified alternative (observability).
- **Road complement** = `road_complement_field` = a single CONSTANT level (phi_road = c). Road wins where
  every locally-supported class SDF is below c → road = the COMPLEMENT. 0 video-derived bytes.
- **Joint** = inject {road-compl, lane(phi_1), sky(phi_2), hood(phi_4)} into the ONE ideal stack, keep
  Movable(phi_3) ideal, recompute the argmax. `decompose_full_confusion` gives the full per-class FN/FP +
  confusion (the joint-containment metric; the prior lane/hood decompose was road-vs-class only).

---

## 2. The $0 mechanism (what was actually measured) — NO-FAKE

Per frame (n96, frozen CPU-torch L* = `lstars[i]`, 384×512): `phi_ideal = signed_distance_fields(L,5)`
(argmax==L exactly → all-ideal baseline 0). Build the structured fields (sky-mask once; sky-line + lane
per frame; hood once; road const). Inject each (isolation: ONE channel; joint: all four structured,
movable ideal) via the class-agnostic `inject_lane_sdf(...,mode="replace")`. Recompute argmax; decompose
vs the REAL cached L* (bit-exact). The lane uses its OWN optimum (continuous band, `dash_gate=False`, per
FEED-ds) — OPTIMAL-FORM discipline: every lever compared at its own optimum. Real majority votes / real
polyfits / real scipy EDTs / real argmax / real disagreement; no stub, no surrogate, classes detected not
assumed.

Scripts: `experiments/measure_road_horizon_joint_containment.py` (delegates to
`src/tac/boundary_math/road_horizon_component.py`, which REUSES `lane_sdf_component` +
`hood_static_component`); JSON `experiments/results/road_horizon_joint_FEED-dw/n96.json`; 18 tests in
`src/tac/boundary_math/tests/test_road_horizon_component.py` (all green; hood's 17 still green).

---

## 3. Byte cost + integration spec (rule-118 boundary) — the multi-component codec

| component | structured form | COUNTED bytes (600 frames) | per-frame | status |
|---|---|---|---|---|
| lane (phi_1) | per-frame poly band SDF (FEED-ds) | ~1–2 KB | ~43 floats/frame | DONE |
| hood (phi_4) | single static mask SDF (FEED-dv) | 56 B | 0.093 B | DONE |
| **sky (phi_2)** | **single static mask SDF** | **98 B** | **0.16 B** | **this memo** |
| **road (phi_0)** | **the witness's LEARNED deep field** (complement) | **0 sidecar** | 0 | **this memo** |
| movable (phi_3) | the LEARNED dynamic residual | (learned, no sidecar) | — | witness job |

**FREE (inflate.py, rule-118):** re-expand the sky span-table/bitmap → EDT → phi_sky; the lane IPM+EDT
rasterizer; the hood EDT. **COUNTED:** lane poly coeffs + hood mask + sky mask. Total structured sidecar
≈ **1–2 KB** for road+lane+sky+hood regions (rate ~3e-4) — negligible vs the 177 KB frontier archive.

**Integration (additive / default-off), proposed flags on
`train_levelset_witness_realized_through_R_mlx.py`:** `--sky-static-component` (bool),
`--sky-static-mode {bias,replace}` (default `bias`), `--road-complement-level` (float, default 0.0 — but
prefer the LEARNED phi_0; the const is the probe, not the codec). Compose with the already-built
`--lane-sdf-component` (FEED-ds) + `--hood-static-component` (FEED-dv). **Critical wiring note from this
measure:** keep road as a deep LEARNED field (eikonal-regularized, `--eikonal-weight`) — do NOT replace
phi_0 with a const (the const is the SOLE joint antagonist). `mode="bias"` is the safe first integration
(keeps the learned head, pulls toward the static mask).

---

## 4. Verdict (compose, do not stack)

- **SKY:** ADOPT the static-majority-mask SDF (98 B). REJECT the horizon-line (2.6× worse — sky ≠ half-plane).
  Sky drift (FN 0.0024 > target) is a residual the witness refines — sky is good+cheap but NOT as tight as hood.
- **ROAD:** ADOPT road-as-complement as the *structural identity* (no dedicated sidecar) BUT realize it as a
  DEEP LEARNED phi_0, NOT a const (the const antagonizes the joint by +0.008344). The const is a $0 probe.
- **JOINT:** the structured components (lane+hood+sky) compose ADDITIVELY in one argmax (antagonism ~0) and
  their per-class containment HOLDS — **the FEED-dt highest-risk integration concern is answered POSITIVELY
  for the structured set.** The one integration risk (road-const) is identified and has a clean fix (deep
  road). PROCEED to wire the multi-component codec, with road learned-deep.

The capstone plan, sharpened by this measure: GIVE the witness the lane-SDF (phi_1) + hood-static (phi_4) +
sky-static (phi_2) as structured priors, let it LEARN a deep road phi_0 + the Movable phi_3 + the
boundary-annulus refinement. The structured set is contained, additive, and ~1–2 KB; the witness's real job
is **Movable (0.0156) + the all-class boundary annulus** — i.e. the genuinely hard part of the d_seg gate.

---

## 5. Honest caveats / NO-FAKE
- **Structured floor (0.005993) > frontier d_seg (0.00056).** The EDT-rasterized structural decomposition is
  a PRIOR/INIT, not the final d_seg. Its value is structural (cheap + contained + additive regions), freeing
  witness capacity for the boundary annulus + Movable — NOT a finished d_seg solution by itself.
- **Isolation/joint assume Movable + (in the additive test) road are IDEAL.** The real witness LEARNS them;
  the measured additivity + containment assume the neighbors are deep fields, which the Eikonal regularizer
  (`--eikonal-weight`) maintains. The road-CONST result is exactly what happens when a neighbor is NOT deep.
- **n96, not n600** (FEED-dt seal owes the n600 re-measure). Sky static IoU min 0.952 over n96 is decent but
  the RAV4 segment is mostly straight (FEED-dj) → turns/bumps under-sample sky drift; an n600 re-measure is
  owed before the byte-closed row.
- **Sky-line possibly underrated on curves:** on the mostly-straight segment the static mask wins; a
  pitching/turning segment might favor a per-frame line. The mask wins HERE (measured); re-check at n600.
- This is a PRIOR / structural lever at the representation level — the predicted exact-eval benefit requires
  a byte-closed trained-witness row (the next step). **Pointer UNMOVED 0.19110.**

## 6. Observability surface
- Per-variant full confusion (`FullDecomp`): total + per-class FN/FP + KxK confusion + `leak_into(k)` — the
  joint-containment metric; diff-able across variants (ideal / iso×6 / joint×4), queryable from the JSON.
- Region detection evidence (`RegionEvidence`): static_iou / frac / top·bottom·mid share / lane-lines /
  maj_row_span — the NO-FAKE self-detection audit surface (§0b table regenerated by the script).
- Static-mask diagnostics (`StaticHoodMask`, reused): mean/min per-frame IoU — sky staticity (0.982/0.952).
- Horizon-line diagnostics (`HorizonLine`): coeffs / deg / rms_row / n_fit_cols — why the line is dominated.
- Byte cost (`static_mask_byte_cost`, `sky_line_byte_cost`, `road_complement_byte_cost`).
- Reproducible CPU $0: `experiments/measure_road_horizon_joint_containment.py --n 96 --r-survival`; input:
  `gt_n96.npz` `lstars` only. 18 tests.

## 7. Wire-in (6-hook)
1. **sensitivity-map:** sky/undrivable ≈ 13–17% of d_seg flips (the road→sky/undrivable horizon transition,
   horizon-band memo); the static mask is its ~0-byte structural solver. Road is the complement (structural).
2. **Pareto:** sky 98 B + road 0 B (rate utterly non-binding) → the structured set (lane+hood+sky+road)
   ≈ 1–2 KB; frees the rate budget for the binding boundary-annulus residual.
3. **bit-allocator:** sky mask = a new COUNTED section (one 98-B blob); road = NO section (learned phi_0).
4. **cathedral:** the inflate.py span-table/bitmap→EDT sky rasterizer + the complement-road = FREE generic
   interpreter paths.
5. **continual-learning:** this memo + DAG FEED-dw + `road_horizon_joint_FEED-dw/n96.json`; answers the
   FEED-dt seal's HIGHEST-RISK joint-containment item (structured components ADD, not antagonize; road-const
   is the lone antagonist; residual = Movable 0.0156 + boundary 0.006).
6. **probe-disambiguator:** sky-MASK-vs-LINE arbitrated by the measurement (mask wins 2.6×); road-CONST-vs-
   road-DEEP arbitrated (deep wins; const antagonizes). No open interpretation remains.

## 8. Borrowed-substrate accounting
- **BORROWED (cited):** scipy EDT (same primitive as `signed_distance_fields`); the lane-SDF template
  (FEED-dm/ds, `lane_sdf_component`, REUSED class-agnostically); the static-mask template (FEED-du/dv,
  `hood_static_component`, REUSED class-agnostically); openpilot horizon-as-vanishing-point geometry
  (horizon-band memo); comma10k 5-class semantics.
- **OURS-ORIGINAL:** the sky/undrivable static-mask-vs-horizon-line head-to-head (mask dominates — sky ≠
  half-plane); ROAD-as-level-set-COMPLEMENT (a deep background field, 0 sidecar bytes; the const-probe shown
  to antagonize → road must be learned-deep); the JOINT-containment harness + full per-class confusion that
  MEASURES structured-component additivity/antagonism + the learned residual in ONE argmax; the data-driven
  full-region classifier (NO-FAKE self-detection of all 5 roles, robust under label permutation).
