# ddm_sv2 — SURVIVAL ENGINEERING + THE RE-BASE (task #888, 2026-08-02)

**Status:** MEASURED-FROM-EXISTING-RECEIPTS (Part 1) · DESIGN + PRE-REGISTERED FALSIFIER (Part 2)
**Axis:** all numbers below are `[macOS-CPU frozen-scorer advisory]` inherited from their source
receipts. `score_claim=false`, `promotion_eligible=false`, `pointer_moved=false`. No n600 scorer job
was fired by this arm.
**Denominator, used everywhere:** scored pixels n600 = 600 × 512 × 384 = **117,964,800**.

---

## 0. HEADLINE

The live "crux workload ≈ **2.7 M errors** of realization" is denominated on a described base that
is **not** the best measured described base in this repo. A better base was measured on 2026-07-23,
at n600, on-vehicle, **for 30 bytes, with `geometry_changed: false`** — i.e. a pure realization-side
fix that changes nothing about the description.

Re-based, the workload above box is **879,886 errors, not 2,709,004** — a **3.08× overstatement**.
**1,829,118 of the priced errors do not exist.**

The `159×` factor in this arm's own name is likewise **2.80× overstated**: against the best measured
described base the pipeline loss is **56.7×**, not 159×.

And the instrument I was chartered to build — a survival function measured through the real trip —
**already exists at n600, on-vehicle, and was never consumed** (`candidate_arms_constructed: []`).

So does the algebra. The exact reformulation of the binding stage is a **registered canonical
equation** since 2026-07-01 (`argmax_of_sdf_is_additively_weighted_power_diagram_v1`), with a live
donor at `power_diagram_witness.py:477` — and it is wired **nowhere in the describe loop**. Margin
becomes an exactly-differentiable signed distance to a hyperplane; no STE, no bias. Its exactness is
**measured, not assumed** — AUC 0.9987 with a named uncovered stratum (codim-2/3 Movable junctions,
27.0% of flip mass), which this memo leads with rather than buries.

**Net: three of the four things this arm was sent to build already exist and are unwired. The build
is smaller than the charter assumed; the debt is wiring, not construction.**

---

## 1. INSTRUMENT FAILURE FOUND IN MY OWN FIRST THREE SEARCHES (reported before any result)

My first searches for the confounded numbers returned **0 hits** for `0.024125` and `159×` — strings
that are literally in the source memo. `rg` skips **hidden** directories by default, and `.omx/` is
hidden. Every recursive search was **vacuous over `.omx/**` — 27,417 files, the single most important
scope** — and returned the same symbol a clean search would.

A later attempt with `--hidden` inside a shell variable (`--glob=!...` form) *also* silently returned
nothing. Only `rg --hidden <pat> .` as literal argv worked.

Caught by a positive control, not by inspection. **Denominators for every search below:** `.omx/research`
= 6,939 `.md`; repo `.py` = 8,030 (`src/tac` 7,213 + `experiments` 817); files under `.omx/research`
visible to the corrected walk = 27,417. Every negative in this memo is scoped, never universal.

---

## 2. PART 1 — THE RE-BASE

### 2.1 The base ladder, all on one denominator, all re-derived at source

| base | d_seg | errors | bytes | × vs solve | source (re-derived) |
|---|---:|---:|---:|---:|---|
| exact lattice solve | 1.52e-4 | 17,931 | — | 1.00× | `ddm_is1_directive4…md` |
| **box allowance (target)** | **0.00116** | **136,839** | — | 7.63× | `ddm_c1_…ledger….json` `box.d_seg_max` |
| **pt1 `global_amplitude_statistics_match`** | **0.008619** | **1,016,725** | **30** | **56.7×** | `ddm_pt1_measurement_receipt.json` `rows[3]` |
| pt1 `analytic_coverage_blend` | 0.020945 | 2,470,714 | 68,464 | 137.8× | `rows[2]` |
| pt1 `hard_camera_placement` | 0.021980 | 2,592,874 | 68,464 | 144.6× | `rows[1]` |
| pt1 `native_grid_flat_palette_control` | 0.022448 | 2,648,079 | 0 | 147.7× | `rows[0]` |
| W_seg "best DESCRIBED base" | 0.024125 | 2,845,843 | 138,031 | **158.7×** | `ddm_sc1_…md:101` |
| c1 composed control | 0.027470 | 3,240,528 | 133,941 | 180.7× | `ddm_c1_…ledger….json` `control` |
| inherited E2 flat paint | — | 3,349,482 | — | — | pt1 `inherited_e2_anchor_comparison` |

Arithmetic checks: `0.024125 × 117,964,800 = 2,845,955` vs cited 2,845,843 (0.004% — consistent).
`0.00116 × 117,964,800 = 136,839` ✓ exact. `1.52e-4 × 117,964,800 = 17,931` ✓.
`0.024125 / 1.52e-4 = 158.7` ✓ — **the `159×` label is correct for W_seg and only for W_seg.**

### 2.2 The re-base table

| artifact (file:line) | number it uses | what it becomes re-based | ΔWorkload |
|---|---|---|---|
| `pantheon_synergy_crux_synthesis_20260728.md:202` — *"the crux workload is ~2.7M errors of realization"* | 2,845,843 − 136,839 = **2,709,004** | 1,016,725 − 136,839 = **879,886** | **−1,829,118 (3.08× overstated)** |
| `ddm_ar1_archetype_codec_priced_spec_20260728.md:38` — *"⇒ 2.7 M-error realization workload"* (priced spec; residual stream OPEN) | same 2.7 M | same 0.88 M | **same 3.08×**; the OPEN residual stream is sized off this |
| `pantheon…:37-38` — *"description→RGB regeneration loses **159×**"* | 158.7× | **56.7×** | headline **2.80× overstated** |
| `ddm_pp1_direct_partition_pricing_20260728.md:132` — *"paint-face **0.0086 = 159×**"* | mis-pairs 0.0086 with 159× | 0.0086 → **56.7×**; 159× belongs to 0.024125 | **label error 2.80×**; see §2.4 — the Route-B verdict itself does **not** flip |
| `ddm_ee1_…capstone_20260728.md:225` (C5) | already flags *"paint negatives measured on ZERO-PARAMETER palettes"* | **consistent with this re-base** — ee1 reached the same scope-gap independently | no change; ee1 is the one live doc already re-based in spirit |
| `.omx/state/operator_p0_ledger.jsonl:449` (FEED-603) | records 2,845,843 as the confound *anchor* | correct as **history**; must not be read as live workload | none — historical, keep |

### 2.3 The airtight version (same control, same inputs, same scorer SHA)

The cross-formulation comparison above (W_seg vs pt1) mixes two formulations. The **within-receipt**
comparison does not — `rows[0]` and `rows[3]` share one control, one input set, one
`segnet_weights_sha256` (`68956e32…`), one `modules_sha256` (`065961ba…`):

- flat-palette control, **0 B**: 2,648,079 errors
- amplitude-statistics arm, **30 B**, `geometry_changed: false`: 1,016,725 errors
- **61.6% of the entire described-base error mass removed for 30 bytes, with the description's
  geometry untouched.**
- above box: 2,511,240 → 879,886 = **2.85× overstatement**, defensible without any cross-formulation
  assumption.

**This is exactly the outcome is1 §2 predicted.** The error mass was an artifact of the regeneration
path, not a property of the description family — and the cure was realization-side (match the solved
object's amplitude statistics) rather than description-side. It cost 30 bytes and was already on disk.

### 2.4 What does NOT change (adversarial check on my own headline)

- **pp1's Route-B DEAD verdict stands.** Its ~421 KB correction-support estimate is keyed to the
  *correct* base (0.0086); only the `159×` **label** attached to it is wrong. I checked whether the
  verdict flips: it does not. Reporting the label error without the verdict error.
- **`ddm_c1_composed_candidate_ledger_603_613_20260723.json` does not cite the confounded base.**
  Exact-searched for `2845843`, `2,845,843`, `0.024125`, `0.0241`, `0.0086` → **0 hits each**. Its
  own control is a *third*, worse base (3,240,528 errors). Scoped negative, that file only.
- The correction menus (`menu1`) and #366's workload: `menu1` codex findings cite the `0.0241` family;
  they inherit the confound label per is1 §2. **I did not re-price them** — flagged, not closed.
- The re-base **does not by itself put any base inside the box.** 879,886 still exceeds 136,839 by
  6.4×. This shrinks the priced workload by 3.08×; it does not solve the axis.

---

## 3. PART 2 — SURVIVAL ENGINEERING

### 3.1 The instrument already exists — and was never fired

`.omx/research/ddm_pt1_continuous_paint_ceiling_20260723/ddm_pt1_on_vehicle_survival_wall_receipt.json`

- `definition`: *"boundary_errors / boundary_sites after render-grid palette paint -> pinned bicubic
  camera uint8 -> real SegNet.preprocess_input bilinear down -> frozen SegNet argmax"* — **the full
  real trip**, not a proxy.
- `pair_count: 600`, `boundary_dilation: 1`, `batch_count: 38`, `batch_size: 16`.
- `measured_survival_wall_fraction: 0.26926624093456114`
- `boundary_errors: 1,387,404` / `boundary_sites: 5,152,536` → 0.269266 ✓ (arithmetic re-checked).
- **`candidate_arms_constructed: []`**

That last field is the finding. A survival wall measured through the real trip at n600, on-vehicle,
SHA-bound — **with zero arms constructed against it.** This is the grade-5 orphan class
(`built_elsewhere_unwired_is_p0`): a better instrument exists and live work does not consume it.

### 3.2 The margin-structural signature is already measured

From the same receipt, against the n600 denominator:

| quantity | value |
|---|---:|
| boundary sites | 5,152,536 = **4.368%** of scored px |
| boundary errors | 1,387,404 = **52.39%** of all errors |
| failure rate **at** boundary | **26.93%** |
| failure rate **off** boundary | 1,260,675 / 112,812,264 = **1.117%** |
| **enrichment** | **24.09×** |

**4.4% of pixels carry 52% of the errors, failing 24× more often than the rest.** This is the
strongest in-tree evidence that the loss is margin-structural — and it was sitting unfired.

Honest caveat, stated because it weakens my own case: *any* imperfect method concentrates error at
class boundaries, including the exact solve. Enrichment shows **where** to spend; it is not by itself
proof that a margin **constraint** converts. §3.5 is designed to settle exactly that.

### 3.3 The per-stage attribution also already exists

`ddm_pt1_measurement_receipt.json` → `mechanism_decomposition`, `disjoint_operational_attribution: true`:

| attributed stage | errors | note (verdict_scope preserved in receipt) |
|---|---:|---|
| `bn_se_amplitude_statistics` | **1,578,514** | corrected by the **30-byte** global mean/variance arm |
| `sub_cell_placement` | 117,055 | hard placement **inside** the target boundary band |
| `class_interaction` | 110,376 | hard placement **outside** the band |
| `texture_prior_or_region_erf` | 7,233 | routes to texture follow-up |

The "**WHICH STAGE killed it**" axis is therefore **measured, disjoint, and in-tree**. The dominant
term by 13× is amplitude statistics — a realization-side scalar mismatch, not geometry.

### 3.4 REFORMULATION, not STE — the exact algebra is already registered as a law

**Correction absorbed (operator 2026-08-02): do not build on STE.** STE *pretends* the gradient is
identity through a non-differentiable op; it is a **biased** estimator, and building the describe-loop
objective on it would re-create the estimate-vs-realized gap inside the cure. Reformulation changes
the algebraic form so the operation **is** differentiable and the gradient is **real**.

**And the stage I was first told to STE does not need it.** `is1` **exonerated quantization** — the
exact lattice solve passes the same uint8 gate at 17,931 errors. §2.1 independently corroborates:
the 30-byte amplitude-statistics arm moved d_seg 0.022448 → 0.008619 through the *same* uint8 gate.
Uint8 was never binding. **The correction makes the build smaller: reformulate the stage that binds
(argmax/margin), skip the one that does not (quantization).**

**The reformulation is already a REGISTERED CANONICAL EQUATION** —
`argmax_of_sdf_is_additively_weighted_power_diagram_v1`
(`src/tac/canonical_equations/witness_measured_findings_20260701.py:680`):

> *"IDENTITY: argmax of K=5 phi_k = additively-weighted power diagram in R^K; 1-skeleton =
> Morse-Smale separatrix (FEED-fh AUC 0.9987); codim-2/3 Movable junction = un-covered stratum."*

Live entry points, re-derived at file:line (not from the counts I was handed):

| surface | file:line | role |
|---|---|---|
| `affine_head_to_power_diagram(weight, bias, …)` | `src/tac/boundary_math/power_diagram_witness.py:477` | **the donor** — *"Derive the **exact** real-arithmetic quotient, then its float32 target"* |
| `power_diagram_argmax(phi, offsets)` | `src/tac/boundary_math/laguerre_logit_offset.py:77` | forward cell membership |
| `power_laguerre_labels(…)` | `src/tac/canonical_equations/closed_scorer_variational_de_20260721.py:107` | labels in the closed variational DE |
| `laguerre_ot_head_offset_v1` | `src/tac/canonical_equations/laguerre_ot_head_offset_20260709.py:107` | OT head-offset law |
| `lane_signed_distance(band_mask)` | `src/tac/boundary_math/lane_sdf_component.py:320` | the SDF primitive the identity is stated over |

Because argmax over affine functions **is** a power diagram, cell membership is an explicit affine
inequality and the cell boundary is a **hyperplane with exact normals** — so **margin becomes a signed
distance to a hyperplane with an exact analytic gradient.** No surrogate, no STE, no bias. #284's
τ→0 result (the witness IS tropical) says the softmax/τ family already trained is the **relaxation**
and this is its **exact endpoint**: both ends of one object are in hand.

**EXACTNESS IS MEASURED HERE, NOT ASSUMED — and it is not total.** The registered anchor reports
`separatrix_auc = 0.9987`, `residual = 1.3e-3`, and an explicitly named
**`uncovered_stratum: codim_2_3_movable_junction`**. So the identity is exact on codim-1 boundaries
and **has a known gap at codim-2/3 Movable junctions.** Movable is 27.0% of flip mass (pantheon §3),
so this residual is *not* negligible and must be measured on the real objects, not inherited. Any arm
adopting this form owes: agreement against the exact trip on a held sample, finite-difference gradient
agreement where feasible, and a reported residual at the Movable junction stratum. A reformulation
that silently drifts is worse than an STE known to be biased, because it would be trusted.

### 3.4b The remaining gap is an EMISSION path, not an instrument

- `tools/measure_ddm_pt1_continuous_paint_ceiling.py` (1,977 lines) computes the wall and emits only
  `measured_survival_wall_fraction`; `_independent_survival_wall()` (L214) **reads a scalar back**.
  Verified: no `per_element` / `per_site` emission path in that file.
- Per `built_new_machinery_instead_of_paying_identified_debt`: **the deliverable is an emission path
  on the existing pt1 measurement, not a new instrument.** The per-boundary-site vector is already
  computed internally to form the ratio; what is missing is writing it out as typed JSONL keyed by
  `(pair, class, site)` — realized margin after the trip · realized ΔS · which stage killed it (from
  the §3.3 disjoint attribution) — so #247's costate SENSE layer and the duty queue consume it. A
  survival function that lives only in a memo is the orphan class; per §3.1 the **scalar** version
  already became exactly that.

**Filter vs gradient:** v19/j11 accept-iff-realized is a **filter** — generation stays blind,
proposals are paid for then discarded. The per-element survival field in the power-diagram algebra is
what makes it a **gradient**: an exact derivative of the realized scored quantity per described
element, so generation aims at survival. **The realized trip remains the authority** regardless.

### 3.4c POSE LEGIBILITY IS AN EXPLICIT CONSTRAINT, not an afterthought

`ddm_uv1` (#889, landed 2026-08-02) measured that seg-only training does not merely fail to add pose
legibility — it **spends** it: `corr(f1_gr1, f1_ep854) = +0.119` with 99.7% of pixels changed, yet
d_seg *improves*, because SegNet reads only the **argmax** (invariant to palette deformation) while
PoseNet reads dense **photometric** correspondence (destroyed).

**That is my stage's failure mode stated in scorer terms.** The power-diagram reformulation is
*intrinsically argmax-only* — cell membership is invariant to any deformation preserving the
inequalities — so a describe objective built naively on it will reproduce uv1's failure exactly, and
would do so *while reporting an improving d_seg*. Binding consequence for the build: **carry pose
legibility as an explicit constraint in the descent (photometric correspondence term or a null-space
restriction), or measure its loss and report it.** An argmax-only reformulated objective that does not
do this is not a smaller build — it is a d_seg win that silently pays in d_pose, and §2.1's own
30-byte amplitude-statistics arm is a warning shot: it carries `geometry_changed: false` precisely
because it moved *photometric* statistics, the very channel PoseNet reads.

### 3.5 PRE-REGISTERED FALSIFIER — registered here, BEFORE any build

**Slice (pinned):** the SHA-bound n600 vehicle, `gt_n600.npz` sha `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`;
frozen SegNet sha `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`;
`modules.py` sha `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`;
`boundary_dilation = 1`; first **96 pairs** (18,874,368 scored px) for the bounded run, full 600 for
the confirming run.

**Test.** The 30-byte amplitude-statistics arm removed 61.6% of the error mass **without touching
geometry**. The margin-structural hypothesis predicts the *residual* (1,016,725 errors) is still
boundary-concentrated. So: **measure the boundary survival wall OF THE AS ARM**, not of the flat
control.

**Target factor.** Margin-structural ⇒ the AS-arm residual retains enrichment **≥ 8×**
(vs the control's 24.09×), i.e. the residual is still a boundary phenomenon and margin-aware placement
has something to bite on.

**FALSIFIED IF** the AS-arm residual enrichment is **< 4×** — the remaining loss would then be
spatially diffuse, *not* margin-structural, margin-constrained description cannot materially reduce
the (re-based, 56.7×) loss, and **calibration is the honest answer after all.**

This test reuses the existing instrument on an existing arm. It is the cheapest decisive experiment
available and it is pre-registered before any code.

**Second pre-registered falsifier — REFORMULATION EXACTNESS (registered before any build).** The
power-diagram form is adopted only if it is exact on the real objects:
- **Agreement:** `affine_head_to_power_diagram` cell labels vs the frozen-SegNet argmax on the held
  slice must agree at **≥ 99.5%** of scored sites (the registered anchor's separatrix AUC is 0.9987).
- **Gradient:** analytic signed-distance gradient vs finite differences, **relative error ≤ 1e-3** on
  a sampled set of boundary sites.
- **FALSIFIED IF** either fails, **or** if the `codim_2_3_movable_junction` uncovered stratum carries
  **> 10%** of the residual error mass on the slice — in which case the exact form does not cover the
  stratum that matters (Movable = 27.0% of flip mass) and must be reported as partial, not adopted
  silently.

**Third pre-registered constraint — POSE LEGIBILITY.** Any descent under §3.4c reports d_pose on the
same slice alongside d_seg. **FALSIFIED IF** d_seg improves while d_pose degrades beyond the
`joint_finish_d_pose_max = 0.001610` box value (`ddm_c1_…ledger….json` `box`) — that is uv1's failure
reproduced, and it must be reported as such rather than booked as a d_seg win.

**Positive control, required before publishing any survival number:** one element independently
verified to survive must register as surviving, and one verified to die must register as dying. Three
agents hit silent instrument failures in 48 h; §1 of this memo is the fourth. Controls first.

**Admission gate (belt and braces, charter move 5):** the gradient aims generation at survival; a
fail-closed margin floor — refuse any described element whose realized margin is below the trip's
measured perturbation magnitude — *guarantees* it. A gradient can be locally fooled; an admission
gate cannot.

### 3.6 STAGED — not fired (scope guard honored)

No n600 scorer job was fired. MAIN owns the slot; `uv1` is ahead. Staged command:

```
.venv/bin/python tools/measure_ddm_pt1_continuous_paint_ceiling.py \
  --config .omx/research/configs/ddm_pt1_continuous_paint_ceiling_20260723.json \
  --output .omx/research/ddm_sv2_as_arm_survival_wall_20260802/receipt.json \
  --measure-survival-wall
```
Requires a config variant whose `calibration_arm` is `global_amplitude_statistics_match` rather than
`pt1_native_grid_flat_palette_control` (current value, L: `calibration_arm` in the survival receipt).
Inputs: `gt_n600.npz`, 5,078,017,610 B, sha `cf8d8360…`. No archive is produced; this emits no score.

---

## 4. ROUND-1 ADVERSARIAL SELF-REVIEW — my own defects

1. **My first three searches were vacuous and I nearly published from them.** `.omx` is hidden; `rg`
   skipped 27,417 files and returned a clean-looking zero. Only a positive control caught it. §1.
2. **The cross-formulation claim (3.08×) is weaker than the within-receipt claim (2.85×).** W_seg and
   pt1 are different formulations. I lead with 3.08× because it is the number the live docs actually
   use, but **2.85× is the defensible one** and I say so in §2.3 rather than burying it.
3. **I did not re-price `menu1` or #366.** I flagged them as inheriting the confound and stopped.
   That is unfinished, not closed — stated plainly rather than implied complete.
4. **Enrichment is not proof.** §3.2 argues against my own headline: boundary concentration is
   expected of any imperfect method. §3.5 exists because §3.2 is not sufficient.
5. **I checked whether pp1's Route-B verdict flips and it does not.** Reporting a label error while
   explicitly *not* claiming the verdict error would have been the easy overclaim.
6. **Nothing here moved the pointer.** `pointer_moved: false`. This arm shrank a *priced workload*;
   it did not lower S. Per the means/ends firewall that is not goal progress, and I say so.
7. **The 30-byte AS arm is `[macOS-CPU frozen-scorer advisory]`, `research_only: true`,
   `promotion_eligible: false`, verdict_scope FORMULATION.** It is not a contest-axis result and the
   re-base inherits exactly that grade — the re-based workload is as advisory as the number it replaces.

---

8. **I initially accepted an STE-based build without questioning it.** STE is a biased estimator;
   building the describe objective on it would have re-created the estimate-vs-realized gap inside the
   cure. I did not catch this — the operator did. The registered power-diagram law
   (`witness_measured_findings_20260701.py:680`) has been in-tree since 2026-07-01 and I did not
   consult it before writing §3.4 the first time. **Proactive recall failure, mine.**
9. **The reformulation's exactness gap is real and I lead with it rather than burying it.** AUC 0.9987
   is not 1.0, and the uncovered stratum is *Movable* — 27.0% of flip mass. If that stratum dominates
   the residual, the exact form is partial for the axis that matters most.

---

## 5. WHAT THIS OWES NEXT

1. Fire §3.5's three pre-registered falsifiers (staged, needs the scorer slot): boundary enrichment of
   the AS-arm residual · reformulation exactness (labels + gradient + Movable stratum) · pose legibility.
2. Wire the **registered** power-diagram form (`affine_head_to_power_diagram`,
   `power_diagram_witness.py:477`) into the describe loop — **~127 file-hits of argmax reformulation
   exist and none is wired here.** #539 (`Build the POWER-DIAGRAM witness parametrization`, in_progress)
   is the natural donor; this arm should hand off to it rather than fork a parallel surface.
3. Emit per-element survival as typed JSONL from the **existing** pt1 path — not a new instrument.
4. Re-price `menu1` / #366 on the re-based denominator.
5. Correct the `159×` label at `pp1:132` and the 2.7 M workload at `pantheon:202` / `ar1:38`
   (append-only supersession; do not mutate the historical FEED-603 ledger row).

**CLOSING-ARTIFACT: .omx/research/ddm_sv2_survival_engineering_and_the_rebase_20260802.md**
