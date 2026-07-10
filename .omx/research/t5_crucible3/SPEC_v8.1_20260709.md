# SPEC — v8.1 INCREMENT-1a DECOUPLING SCREEN (the crucible-3 measurement vehicle) — 2026-07-09

**P7 DELIVERABLE of T5 CRUCIBLE-3 (task #380).** This SPEC is the SEALED synthesis
`SYNTHESIS_v3_v8_20260709.md` (SEALED at P6-R7, 3 consecutive clean; `tac.review_counter`
`crucible3_v8` sealed=True; SEAL commit `240a4ab1b`) rendered as an **executable, provenance-typed
measurement surface** + the operator-facing v8 half of the dual-chain comparison brief. **P7
TRANSCRIBES — it does not re-design.** Every decision here is taken from the sealed v3; where a
transcription needed a judgment call it is flagged in §OWED / §OPEN-ITEMS, never silently decided.

**What increment-1a IS (and IS NOT):** the crucible-2 P7 (`SPEC_v752`) compiled a TRAINING LAUNCH
(the v7.5.2 `WitnessProgram` → trainer argv). Crucible-3's increment-1a is DIFFERENT: it is a **$0,
paint-free, mask-level MEASUREMENT** — the decoupling PARTITION screen, an A/B of the decoupled
per-class-field arm vs a matched-compute shared-head CONTROL arm, both measured the SAME way
(composite tropical-argmax MASK d_seg vs the GT SegNet argmax `L*` on `gt_n600.npz`). It emits **NO
trainer argparse argv**, uses **NO GPU**, and its config is the `Inc1aScreenConfig` (§PROGRAM). It is
a **NECESSARY-condition** screen (mask-optimal ≠ score-optimal); the SUFFICIENT through-R test is 1b.

**STORES CONSULTED:** `SYNTHESIS_v3_v8_20260709.md` (SEALED, the sole design authority — incl.
§SEAL-BOUNDARY + POST-V2 SUPERSESSION LEDGER + §B overrides) · `docs/operating_manual_craft_handoff.md`
(answer-first / label-by-provenance / attack-own-conclusion / §8.4 no-plausible-summary) · crucible-2
`SPEC_v752_20260709.md` + `witness_autoconfig.derive_crucible_v752_config` (the P7 config-from-sealed
pattern) · PRIMARY code re-read: `src/tac/inc1a_harness/decoupling_screen.py` (the harness the config
REUSES) · `src/tac/canonical_equations/laguerre_ot_head_offset_20260709.py` (the `no_offset` LawRef
anchor) · `src/tac/through_r/scaffold_assembler.py` (`BC_MODES` / `N_SEG_CLASSES`). **Not a store: no
new measurement / GPU / training taken; run dirs read-only; #205 STOPPED.**

**Pointer contest-CPU 0.19110 UNMOVED — this SPEC is MEANS.** Everything here is `[macOS-CPU advisory ·
research-signal · NON-PROMOTABLE]`. The END is a byte-closed `upstream/evaluate.py` n600 exact row <
0.19110. **No launch fires when crucible-3 seals** (operator dual-chain directive): when BOTH chains
are sealed+approved, main presents the side-by-side brief and REQUESTS operator which-to-run GO
(#385). This SPEC is the v8 half of that brief. Remaining gap to sub-0.15 = **0.0411 S**; every
magnitude ÷0.0411.

**verdict_scope discipline:** every negative/dismissal carries `verdict_scope:
INSTANCE|FORMULATION|FAMILY` (default narrowest — a failed FORMULATION never kills a FAMILY).

---

## 0. HEADLINE (answer-first)

**increment-1a = the cheapest-falsifiable PARTITION screen: does decoupling the per-class fields
(∂φ_c/∂θ_{c'}=0) produce a better mask-level argmax partition than a matched-compute shared head?**
Measured paint-free (composite argmax MASK d_seg vs `L*` on gt_n600), as a pre-registered A/B with a
MEASURED-IN-RUN control baseline (NEVER run-1's 0.312), a byte-closed composite (F-P5-1), a measured
δ_mask kill floor (F-P5-P9-1), ≥3 seed replicates/arm (F-P5-2), and a temporal tie-flicker row
(F-P5-4). The Road↔Undriv carrier is the **lateral-capable 3-curve generator** (the near-break fix —
the single-valued horizon arc structurally cannot home the R6-MEASURED 97.54% unsupported-column
lateral undriv mass).

```
METRIC        : composite_argmax MASK d_seg vs L* (gt_n600.npz)   [PAINT-FREE; through-R is 1b]
GRID          : scorer-authoritative 512×384 (gt_n600 lstars); argmax AFTER the R downsample
DESIGN        : A/B — decoupled per-class fields vs MATCHED-COMPUTE shared-head CONTROL (both paint-free)
BASELINE      : the control arm mask d_seg, MEASURED IN-RUN (NEVER run-1's 0.312)
b_c MODE      : no_offset (SATURATED safe default; both flip arms + OT REFUTED at n600, §A.3) <!-- # VERDICT_SCOPE_OK: citation of FORMULATION-scoped anchors (scopes stated at their verdicts) -->
KILL FLOOR    : operative δ_mask = max(3.46e-6 [R7 MEASURED], in-run seed spread) — DERIVED-LIVE
SEED REPS     : >=3 per arm (the seed-spread instrument; the DOMINANT δ_mask component)
CARRIER       : lateral-capable 3-curve Road↔Undriv (top arc + x_L(y) + x_R(y)); GEOMETRIC-MINIMAL
TEMPORAL      : per-class tie-flicker across consecutive pairs (Lever-D #280), BOTH arms
FALSIFIES     : decoupling PARTITION quality (necessary). CANNOT falsify through-R survival (= 1b)
```

**Projected S-path (HONEST — §7):** increment-1a produces **NO S row** — it is a partition screen,
not a byte-closed eval. A 1a PASS is a NECESSARY condition for the v8 route; the SUFFICIENT test is 1b
(through-R) and the authority is a byte-closed n600 row. The shippable increment-1 rate is **0.135
(WASH-with-frontier)**; the sub-0.118 win rides the flip-weighted #226 waterfill r\* ∈ **[0.061,
0.135]** (P-C-gated, UNMEASURABLE in increment-1). Pose is **BANKED-as-artifact** (owed-14 #384 GREEN +
engage-on #383 BUILT + banked R1 dxi 0.127 / 7.2 KB), NOT solved-for-v8.

**Wall-clock:** increment-1a is a **$0 local measurement** (no GPU, no training). The BUILD it gates
(the lateral 3-curve carrier + the decoupled/control trained arms) is the cost; the screen itself is a
CPU pass over gt_n600.

---

## 1. §INCREMENT-1a CONFIG (the sealed §B `measure_1a` — transcribed, provenance-tagged)

Value-provenance ladder: **DERIVED-LIVE > DERIVED-AT-CONFIG > MEASURED-ANCHOR >
CONJECTURED/owed-measurement > HARDCODED-WITH-WAIVER**. TBD FORBIDDEN. Every constant below is
RESOLVED from its authority (LawRef anchor where one exists; the live `decoupling_screen` harness
constants otherwise — REUSE-not-rederive so a config↔harness drift is IMPOSSIBLE). The typed
`Inc1aScreenConfig` that carries this exact set is §PROGRAM.

| field | value | ladder class | authority |
|---|---|---|---|
| `metric` | composite_argmax MASK d_seg vs L* (gt_n600.npz) | design | sealed §B (PAINT-FREE; through-R = 1b) |
| `scorer_grid` | (512, 384) | MEASURED-ANCHOR | gt_n600 lstars (600,384,512) → W×H = post-R SegNet argmax AUTHORITY grid (§A.4-grid) |
| `generator_render_grid` | (1164, 874) | design | camera-res #149 PLACEMENT grid (map → scorer 512×384 BEFORE argmax; never conflate) |
| `bc_mode` | `no_offset` | MEASURED-ANCHOR | §A.3 #386 RULED; SATURATED safe default; per-edge b_c on FRESH v8 fields is the owed route (1b) |
| `no_offset_d_seg` | **0.0031436** | MEASURED-ANCHOR | LawRef `laguerre_ot_head_offset_v1` `GATE_N600_D_SEG_NO_OFFSET` (n600; the 0.00272 was the n24/n48 SUBSET) |
| `bc_never` | menon / ot_newton / flip_weighted / flip_median | MEASURED-ANCHOR | §A.3 both flip arms REFUTED (fw 0.0196734 = 6.3×; fm 0.0215612 = 6.9×) | <!-- # VERDICT_SCOPE_OK: citation of FORMULATION-scoped anchors (scopes stated at their verdicts) -->
| `n_seg_classes` | 5 | MEASURED-ANCHOR | live `scaffold_assembler.N_SEG_CLASSES` (Road/Lane/Undriv/Movable/MyCar) |
| `measure_byte_closed_composite` | True | design | F-P5-1: measure the SHIPPABLE composite, not pre-byte-close fields |
| `design` | A/B decoupled vs MATCHED-COMPUTE shared-head control | design | sealed §B: isolates DECOUPLING, not capacity; baseline IN-RUN (never 0.312) |
| `seed_replicates_per_arm` | 3 | DERIVED-AT-CONFIG | F-P5-2 / sealed §B: >=3 = the seed-spread instrument (min for a spread estimate) |
| `delta_mask_frame_sampling_floor` | 3.46e-6 | MEASURED-ANCHOR | R7 (P4_recess) SEM of the 600-frame mean flip fraction; live harness constant |
| `delta_mask_operative_spec` | max(3.46e-6, in-run seed spread) | DERIVED-LIVE | F-P5-P9-1/F-P5-2: DOMINANT seed component is in-run; `operative_delta_mask` REFUSES until it lands |
| `delta_mask_retired_proxy` | 0.0196 (RETIRED) | MEASURED-ANCHOR | F-P5-P9-1: the δ_R ~5600× category error; recorded as RETIRED, NEVER the floor |
| `kill_criterion` | decoupled > control + δ_mask (pre-registered) | design | sealed §B; evaluator = `decoupling_screen.evaluate_kill` |
| `carrier` | lateral-capable 3-curve (see §3) | DERIVED | §A.1 / §I I1b (F-P5-1 near-break fix) |
| `temporal` | per-class tie-flicker (see §5) | design | §A.8 (F-P5-4) |
| `residual_operating_point` | r\* RANGE [0.061,0.135]; shippable 0.135 WASH | design | §A.2 (F-P5-3) |

**Compiled state (MEASURED):** `derive_crucible_v8_inc1a_config(gt_n600, num_pairs=600)` →
`validate() == []` (0 violations); **`unknown_count() == 0`** (the 0-unknown bar, mirroring the
crucible-2 291-token-0-unknown compile smoke); `provenance_manifest()` schema `dsl_program_manifest.v1`,
`kind=measurement_screen`, 18 provenanced fields.

---

## 2. §THE KILL-GATE PROTOCOL (increment-1a items 3+4 — the paint-free mask-level screen)

The A/B, its baseline, its floor, and its verdict (evaluator = `decoupling_screen.evaluate_kill`, which
the config REUSES — the harness owns the verdict):

- **DESIGN:** decoupled per-class-field arm vs a **MATCHED-COMPUTE shared-head CONTROL** arm, sized so
  total params match within tolerance, SAME seed/epochs/curriculum → the A/B isolates **DECOUPLING**
  (∂φ_c/∂θ_{c'}=0), not capacity/compute. Both **PAINT-FREE** ⇒ the flat-paint 0.0064 confound is
  **EXCLUDED_BY_CONSTRUCTION**.
- **BASELINE:** the control arm's mask d_seg, **MEASURED IN-RUN** — NEVER borrowed from run-1's 0.312
  (the pre-actuation birth arm). (eightfold-P5: no borrowed baseline.)
- **δ_mask (the kill floor):** `operative δ_mask = max(3.46e-6 [R7 MEASURED frame-sampling floor],
  in-run seed spread)`. The DOMINANT component is training-SEED variance — NOT $0-measurable — measured
  IN-RUN from ≥3 control-arm seed replicates. `operative_delta_mask` **RAISES `DecouplingScreenError`**
  (→ `evaluate_kill` surfaces `VERDICT_REFUSED`) when ≥2 replicates exist but `seed_spread` is not
  supplied (the P2 seed-honesty guard: a kill fired on a floor that dropped a measurable seed component <!-- # MAGNITUDE_DISMISSAL_OK: DESCRIBES the P2 seed-honesty REFUSE guard (enforcement), not a dismissal -->
  would be firing on within-seed noise).
- **δ_mask=operative floor = max(3.46e-6, seed_spread), REFUSE on seed under-spec** — the config's
  declared 3 replicates make the operative value **DERIVED-LIVE**, honestly NOT a config-time number.
  The config `.validate()` asserts this contract holds (calls `operative_delta_mask(n=3, spread=None)`
  and requires it to RAISE).
- **VERDICTS (pre-registered):** `improvement = control.d_seg − decoupled.d_seg`. `> δ` →
  **DECOUPLING-CONFIRMED** (the ONLY gate-passing outcome; proceed to 1b); `< −δ` → **KILLED**
  (verdict_scope FORMULATION, NOT paradigm); `|·| ≤ δ` → **INCONCLUSIVE** (underpowered → measure the
  spread / more data); any toy / missing arm / under-spec floor → **REFUSED** (NO-FAKE / n600).
- **composite argmax vs L\*** — the composite is measured **BYTE-CLOSED** (F-P5-1): any top-arc ↔
  side-curve seam/overlap is captured in the MEASURED composite argmax (no idealized-union gap).
- **matched-compute shared-head CONTROL** — the control's `to_config_dict()` is the DSL authoring path
  P7 compiles against the live architecture (this harness never edits the DSL).

**Kill-gate thresholds (this SPEC ↔ the sealed doc, one-to-one):** floor = R7 3.46e-6 (not δ_R 0.0196);
seeds ≥3; grid 512×384; bc `no_offset` (never the flip/OT arms); measure byte-closed composite; REFUSE
on under-spec. The config `.validate()` enforces each (§PROGRAM).

---

## 3. §THE 3-CURVE LATERAL CARRIER (§I representation — clause-A non-derivability + clause-B min-dim)

**The near-break fix (F-P5-1):** the v2 Road↔Undriv carrier shipped the single-valued `_horizon_profile`
(per-column `y(x)`). P4 R6 MEASURED that **97.54%** of the 9.44M GT-Undrivable flip px is
**UNSUPPORTED-COLUMN lateral/side undrivable** (grass/building/off-road at the frame SIDES, in columns
with NO Road/Undriv horizon point) — which a single-valued `y(x)` **STRUCTURALLY cannot represent at
any capacity** (P9's worst class: a proxy with a structurally-blocked thing-itself path). The replacement
is the lateral-capable 3-curve generator (still GEOMETRIC-MINIMAL, still far cheaper than the dense
bulk SDF field):

| # (§I) | curve / lever | dim | mode | justification |
|---|---|---|---|---|
| I1 | **top arc `y(x)`** (`_horizon_profile`) | 4 coeffs (deg-3) + ξ intercept (charged once §G S1) | GEOMETRIC-MINIMAL | MEASURED: cubic/quad coeffs FROZEN 599/600 (\|Δ\|≈1e-7); only intercept moves. **dominant S 0.00277** (F-P5-6 PIN: code-emitted 4167 B, not memo 0.0032). 14.6× MEASURED. |
| I1b | **lateral extents `x_L(y)`, `x_R(y)`** (NEW, F-P5-1) | 2 per-ROW low-order curves (~4 coeffs each) + per-frame intercepts | GEOMETRIC-MINIMAL | **DERIVED + owed-measurement:** the multi-branch complement of the single-valued top arc; homes the R6-MEASURED 97.54% lateral mass. **carrier_total_S ∈ [0.0040, 0.0083]** DERIVED range (LOWER = side curves as ego-rigid as the top arc; UPPER = fully independent per-frame = 3× the 4167 B). **owed-measurement recess R8** (side-curve frozenness on gt_n600 UNMEASURED). Still ≪ the dense bulk SDF field (I10, 20–50 KB = S 0.13–0.33) ⇒ T1's demotion RATE law survives. |

- **clause-A (non-derivability / geometry-first):** the lateral extents are NOT representable by the
  top arc (R6 measured), NOR by the lane centerline (I2 = drivable geometry, a different object), NOR by
  Movable/hood — genuinely non-derivable off-road geometry. The seam gets an OWNER:
  `owns_explicitly: lateral_side_undrivable` (no residual-coder catch-all silently absorbs the frame-side
  undrivable). ANTI-REDUNDANT with the residual coder (P12).
- **clause-B (min-dim-or-waterfill):** every carrier is GEOMETRIC-MINIMAL (3 low-order curves) or
  KKT-waterfilled; `carrier_total_S` carries its owed-measurement caveat welded on, never asserted as
  the lateral's own measurement.
- **BUILD OWED (owed item 9):** extend `road_undriv_bulk_field.py` with `x_L(y)`/`x_R(y)`; ADD the §I I1b
  row; MEASURE the side-curve byte cost + frozenness on gt_n600 (recess R8 — turns the DERIVED range
  into a MEASURED anchor); PIN 1a to the byte-closed composite. This is a `build (carrier) + $0 measure`.

---

## 4. §THE MEASUREMENT GRID PIN (operator directive "flipping at correct resolution")

A spec silent on the grid breeds wrong-grid proxies (the same class as the δ_R unit-category error).
PINNED: the 1a composite argmax and `L*` are compared at the **scorer-authoritative 512×384 grid** —
the `gt_n600.npz` `lstars` grid `(600, 384, 512)` = the post-R SegNet argmax resolution. The analytic
generators may RENDER at camera-res 874×1164, but the render-grid → scorer-grid map (bilinear
downsample 874→384 / 1164→512, the R downsample leg) is applied and the **argmax is taken AFTER the
map, never before**. The 1b through-R row uses the SAME 512×384 grid via the real R operator. The
**INTENDED EXCEPTION is #149:** sub-pixel curve/lane PLACEMENT is authored at camera-res 874×1164 (a
PLACEMENT-grid convention) — the SCORER compare is ALWAYS 512×384. The two conventions
(placement-grid camera-res #149 vs compare-grid scorer 512×384) must NEVER be conflated. (P6-R3: a
sibling proxy-audit CONFIRMED the code surfaces are grid-correct.)

---

## 5. §THE TEMPORAL SECTION (§A.8 — F-P5-4, eightfold-P6 SEAL requirement)

- **PER-FRAME (existing):** composite-argmax MASK d_seg vs `L*` on gt_n600, per-frame-averaged, BOTH arms.
- **TEMPORAL-FLICKER ROW (new):** per-class **tie-flicker** across consecutive frame pairs — for each
  class-pair edge, the fraction of pixels whose argmax winner CHANGES between frame t and t+1 at the tie.
  The decoupled per-class fields can flicker **INDEPENDENTLY** at the argmax tie; a shared head cannot
  (it moves all classes' logits jointly). A decoupled arm that WINS static per-frame d_seg but LOSES
  temporal flicker (higher tie-flicker than the control) is a REAL finding (verdict_scope FORMULATION):
  the decoupling bought a better static partition at the cost of temporal coherence, which through-R
  penalizes (MEMORY L67: #205 CE-residual IS temporal flicker, 44% spikes = LANE).
- **INSTRUMENT (existing, consumed):** the **Lever-D temporal machinery** (#280) — a duty-to-measure
  consume, no new instrument built here.
- **SPEC_v8.1 owed (deferred, not increment-1 scope):** the fuller temporal story — slot-churn (Movable
  tracking coherence across GOP), GOP-keyframe structure, dash-phase = ego-distance (the dash-erasure
  temporal dual) — with an owning design principle (not facet-by-facet fixes).

---

## 6. §SEAL-BOUNDARY (carried forward VERBATIM as owed-gates — the honest non-coverage)

The v8-increment-1 design is SEALED as a *design*, NOT a measured result. This SPEC makes ZERO claim on
the pointer (0.19110 UNMOVED); it is `[macOS-CPU advisory · research-signal · NON-PROMOTABLE]` MEANS.
Explicit non-coverage (each is a pre-registered owed-gate, transcribed from SYNTHESIS_v3 §SEAL-BOUNDARY):

1. **Through-R survival (the SUFFICIENT test) is OUT.** 1a is a NECESSARY-condition partition screen;
   mask-optimal ≠ score-optimal. The through-R row is **1b**. A 1a PASS does not predict the n600 score.
2. **The I1b carrier byte cost is DERIVED, not MEASURED.** `carrier_total_S ∈ [0.0040, 0.0083]` is a
   derived range; the side-curve frozenness on gt_n600 is UNMEASURED (owed **recess R8**). The lower
   bound ASSUMES the side curves are as ego-rigid as the top arc (owed, not asserted).
3. **The "3-curve homes 97.54% lateral" coverage is DERIVED (the weakest link).** The 3-curve carrier can
   STRUCTURALLY represent lateral undriv (vs the single-valued arc which cannot); the actual Undriv-d_seg
   reduction is measured INDIRECTLY by the byte-closed 1a composite, NOT as a standalone number.
4. **r\* (the sub-frontier win) is UNMEASURABLE in increment-1.** Carried as a labeled RANGE **[0.061,
   0.135]**, P-C-gated. The SHIPPABLE increment-1 rate is **0.135 = WASH-with-frontier**, never a rate win.
5. **Lane-generator coverage (53% of the residual enemy) is a FAST-FOLLOW** (increment 2), NOT increment-1
   scope (F-P5-5); its own BUILD + byte-close + n600 A/B. The 0.135 wash is partly this weak-generator
   artifact, not v8's ceiling.
6. **The b_c route forward is OWED.** The GLOBAL post-hoc 5-scalar b_c is SATURATED at `no_offset` on the
   eroded trunk (#386 RULED, both flip arms REFUTED — FORMULATION/REGIME scope). Per-EDGE b_c on the FRESH
   v8 Stage-A decoupled fields (+ offsets-solved-JOINTLY-with-training) is the route (increment 1b).
7. **The de-share 0.0044 thing-itself is OWED** (dilate=2 INSTANCE; band 0.000–0.0069). Curve-relative
   REFUTED verdict HOLDS across the whole footprint band. <!-- # VERDICT_SCOPE_OK: citation of FORMULATION-scoped anchors (scopes stated at their verdicts) -->
8. **Pose is BANKED-as-artifact, NOT solved-for-v8.** owed-14 RESOLVED (#384 GREEN) + engage-on BUILT
   (#383) + banked R1 dxi (0.127 / 7.2 KB); STILL owed = per_class_dseg_basin_conjunct + f_basin_0.9.
9. **The δ_mask DOMINANT floor component is $0-UNMEASURABLE.** The R7 floor (3.46e-6) is a LOWER BOUND;
   the operative floor's seed-variance component is measured IN-RUN from ≥3 replicates (code-guarded).
10. **The auditor-A C1 directional-basis (−48%) flag is IRRELEVANT to this SPEC** (grep NONE; v8 makes no
    −48% / directional-basis claim).
11. **Two triality legs are OWED to the BUILD.** The DSL `Lever` fold (at the carrier BUILD) and the
    equations-leg canonical row (once R8 MEASURES the carrier rate) land with the build, not here.

---

## 7. §FALSIFIERS (pre-registered per build item — P7-philosophy)

| build item | falsifier (pre-registered) | verdict_scope |
|---|---|---|
| **increment-1a screen** | decoupled mask d_seg > control + δ_mask → CONFIRMED; < −δ_mask → KILLED (decoupling FORMULATION falsified) | FORMULATION (NOT paradigm) |
| **lateral 3-curve carrier** (owed-9) | inherits the 1a A/B falsifier; recess R8 = its measurement path (side-curve frozenness) | FORMULATION |
| **temporal tie-flicker** (§A.8) | a decoupled arm winning static d_seg but LOSING tie-flicker vs the control → the decoupling bought static-partition at a temporal-coherence cost | FORMULATION |
| **cannot-falsify (out of 1a)** | through-R survival (mask-optimal ≠ score-optimal) — the SUFFICIENT test is 1b, NOT 1a | — |

`falsifies` / `cannot_falsify` / `flat_paint_confound=EXCLUDED_BY_CONSTRUCTION` are pre-registered in
the config `falsifier` field (§PROGRAM). NO-FAKE: the screen REFUSES a verdict on any toy / missing arm
/ under-spec floor — a partition that cannot be measured at n600 cannot falsify anything.

---

## 8. §P12 COMPOSITION / INTERACTION SIGNS (per shipped lever — measured by the 1a A/B, never assumed additive)

| lever pair | interaction sign | authority |
|---|---|---|
| lateral carrier × residual coder | **ANTI-REDUNDANT** | §A.1 `owns_explicitly: lateral_side_undrivable` (the seam has an owner; no double-cover) |
| decoupled per-class fields × temporal coherence | **ANTAGONISTIC at the tie** | §A.8 tie-flicker (MEASURED both arms) — the pre-registered risk |
| lane-generator coverage (increment 2) × increment-1a | **ANTAGONISTIC-if-folded-early** | F-P5-5 (folding it BLOATS the cheapest-falsifiable row + DEFERS the screen) |
| b_c (`no_offset`) × the trained fields | **ORTHOGONAL** | closed-form, out of the scorer grad loop (risk-2 structural) |

Composition claims are MEASURED by the 1a A/B (in-run), never assumed additive (the Dykstra
non-additivity trap). P11 (strategic-surgical-harm) = **N/A-with-derivation**: increment-1 ships NO
deliberate-harm schedule element (erosion / homotopy island-birth / τ-anneal are v7.5.2/#205 curriculum
scope); Stage-A field DECOUPLING is a partition CHANGE, not an accepted local worsening.

---

## 9. §INCREMENTS 1b / 2 (the named fast-follows — NOT increment-1a scope)

- **increment 1b — per-EDGE b_c + through-R:** the GLOBAL post-hoc 5-scalar b_c is SATURATED at
  `no_offset` on the frozen eroded trunk (#386 RULED). The route forward is **per-EDGE b_c on the FRESH
  v8 Stage-A decoupled fields** (non-eroded by construction — S1's original per-edge form), + offsets
  solved JOINTLY-with-training. 1b is also the through-R SUFFICIENT test (the 1a NECESSARY screen's
  complement). params ride §I clause-B.
- **increment 2 — lane-generator coverage (FAST-FOLLOW):** the 40% off-curve Road/Lane residual (53% of
  the enemy) is a LANE-GENERATOR coverage gap, not a residual-coder gap (P4 R2: the generator-coverage
  lever DOMINATES the coder lever). Its own BUILD + byte-close + n600 A/B. Its params ride §I clause-B
  (GEOMETRIC-MINIMAL or KKT-waterfilled). Named so the P8 brief does not read the 0.135 wash as v8's
  ceiling.

---

## 10. §PROGRAM — the provenance-typed `Inc1aScreenConfig` (the executable measurement surface)

The increment-1a config is authored as a frozen, provenance-tagged `Inc1aScreenConfig` (NO trainer
argparse argv — this is a $0 measurement, not a launch). It:
- **RESOLVES every constant on the value-provenance ladder** from its authority: `no_offset_d_seg` via
  the LawRef anchor `laguerre_ot_head_offset_v1.GATE_N600_D_SEG_NO_OFFSET`; the R7 δ_mask floor + the
  RETIRED δ_R proxy via the live `decoupling_screen` harness constants; `bc_mode`/`n_seg_classes` via the
  live `scaffold_assembler` — REUSE-not-rederive, so a config↔harness/LawRef drift is IMPOSSIBLE.
- **`.validate()` returns [] (clean)** — fail-closed against the sealed doc + the live surfaces (bc =
  no_offset never the flip/OT arms · d_seg = the LawRef anchor · δ_mask floor = R7 not δ_R · seeds ≥3 ·
  grid 512×384 · byte-closed composite · lateral-capable carrier owns the seam · 0-unknown · the
  operative-floor REFUSE contract holds). `derive_...` RAISES on any violation.
- **`.unknown_count()` == 0** (the 0-unknown bar; TBD FORBIDDEN).
- **`.provenance_manifest()`** emits the DSL-provenance attestation (schema `dsl_program_manifest.v1`,
  `kind=measurement_screen`, per-field ladder class + provenance) a consumer reads.

**Module:** `tac.witness_autoconfig.derive_crucible_v8_inc1a_config` (the 4th DSL-provenance program
after v6/v7/v7.5.2). Authored by RESOLVING each constant from its authority + the sealed carrier/temporal/
residual spec dicts. Fail-CLOSED (`validate()==[]` at authoring).
**Test:** `src/tac/tests/test_crucible3_v8_inc1a_config.py` (16 tests, all PASS) — validate-clean +
0-unknown + LawRef value-identity + R7-not-retired-floor + no_offset-never-flip-arms + ≥3-seed-pin +
grid-pin + byte-closed + lateral-capable-carrier + kill-thresholds-match-sealed + manifest-per-field +
the REFUSE paths (refuted b_c arm / d_seg drift / TBD placeholder) + the operative-floor DERIVED-LIVE
contract.

**Validation result (MEASURED):** `derive_crucible_v8_inc1a_config(gt_n600, num_pairs=600)` →
`validate() == []` (0 violations); `unknown_count() == 0`; `provenance_manifest()` schema
`dsl_program_manifest.v1`, `kind=measurement_screen`, 18 provenanced fields, `no_offset_d_seg=0.0031436`
(LawRef), `delta_mask_frame_sampling_floor=3.46e-6` (R7 harness), `bc_mode=no_offset`,
`seed_replicates_per_arm=3`, carrier mode `horizon_poly_xi + lateral_extent_curves`.

**Config ↔ BUILD boundary (never-invent):** the config carries the SEALED spec (the carrier is a spec
dict `owed item 9`; the trained decoupled/control arms are the governed EVENT it gates). It compiles NO
flag that does not exist and trains nothing. `--pose-finish-engage-on`-class owed builds are v7.5.2
scope, not here.

---

## 11. §OWED (transcription judgment calls surfaced, NOT silently decided — operating-manual §5)

- **owed-9 (BUILD, F-P5-1):** the lateral-capable 3-curve carrier (`road_undriv_bulk_field.py` +
  `x_L(y)`/`x_R(y)`) + its §I I1b row + recess R8 (side-curve byte cost + frozenness on gt_n600). The
  config carries the SPEC; the BUILD is 1a-BLOCKING (a single-valued carrier hobbles the decoupled arm).
- **owed-11 (config, F-P5-2):** ≥3 seed replicates/arm in the 1a A/B (the config PINS `seed_replicates_per_arm=3`;
  the in-run seed-spread supply is the run's job — the operative floor is DERIVED-LIVE).
- **owed-R8 (recess):** turns `carrier_total_S ∈ [0.0040, 0.0083]` (DERIVED) into a MEASURED anchor.
- **owed-2 (FAST-FOLLOW, F-P5-5):** lane-generator coverage = increment 2 (§9), NOT increment-1.
- **two triality legs OWED to the BUILD:** the DSL `Lever` fold (carrier BUILD) + the equations-leg
  canonical row (once R8 MEASURES the carrier rate). This SPEC is a DSL-leg landing (the config); the
  carrier-BUILD folds the `Lever`, not here.

**NOT open (resolved by the sealed doc, transcribed):** the b_c safe default (`no_offset`, both flip arms
REFUTED, §A.3) · the R7 δ_mask floor (δ_R RETIRED) · the byte-closed composite (F-P5-1) · the grid pin <!-- # VERDICT_SCOPE_OK: citation of FORMULATION-scoped anchors (scopes stated at their verdicts) -->
(512×384) · the temporal tie-flicker section (§A.8). Each is UNAMBIGUOUS in the sealed doc and is DONE
in the typed config (verified by the test's assertions).

---

## 12. DUAL-CHAIN SEQUENCING (operator directive)

NO launch fires when crucible-3 seals. When BOTH chains are sealed + approved, main presents a
side-by-side {v7.5.2 `WitnessProgram` (a training launch) vs v8 increment-1a `Inc1aScreenConfig` (a $0
mask-level screen): projected S-path, owed-gates, risk register, wall-clock} and REQUESTS operator
which-to-run GO (#385). This SPEC's §6/§7/§8/§11 are the v8 half of that brief. **The P8 wall is:
both-sealed → comparison brief → operator which-to-run GO.**

---

## STORES CONSULTED (line)

`SYNTHESIS_v3_v8_20260709.md` (SEALED P6-R7, seal `240a4ab1b` — sole design authority; §SEAL-BOUNDARY +
POST-V2 SUPERSESSION LEDGER + §B) · `docs/operating_manual_craft_handoff.md` · crucible-2
`SPEC_v752_20260709.md` + `witness_autoconfig.derive_crucible_v752_config` (the P7 pattern) · PRIMARY:
`src/tac/inc1a_harness/decoupling_screen.py` (the harness REUSED) ·
`src/tac/canonical_equations/laguerre_ot_head_offset_20260709.py` (the `no_offset` LawRef anchor) ·
`src/tac/through_r/scaffold_assembler.py` (`BC_MODES`/`N_SEG_CLASSES`). $0 · no GPU · no training · run
dirs READ-ONLY · #205 STOPPED.

**Pointer 0.19110 UNMOVED — this SPEC is MEANS.** Only a byte-closed `upstream/evaluate.py` n600 row <
0.19110 moves it. increment-1a is a NECESSARY-condition PARTITION screen (mask-optimal ≠ score-optimal);
the SUFFICIENT through-R test is 1b; the authority is a byte-closed n600 row.

---

## MACRO-RATE ADDENDUM (append-only — 2026-07-10; operator "view the whole thing bringing micro to macro")

**Scope:** this addendum re-derives the increment-1 rate from the WHOLE-ARCHIVE macro object (scene
geometry ~8-dim + ego-screw ξ(t) already-banked 7.2 KB + per-class charts) instead of the §I MICRO
sum. It is a rate-half DERIVATION `[macOS-CPU advisory · NON-PROMOTABLE]`; it CHANGES NO sealed §1–§12
decision and moves NOTHING on the pointer (0.19110 UNMOVED). Full memo:
`.omx/research/v8_macro_rate_pass_20260710.md`. Sibling of clause-A lifted pairwise→whole-set.

**The measured negatives that shape it (both residual RE-CODING axes are DEAD):** (verdict_scope: FORMULATION per their anchors; reformulation queues OPEN) TEMPORAL ego-warp
residual coding NO-GO (`keyframe_ego_residual_coding_n600` n600: `intra` BEATS `ego_warp_R1` at all <!-- # VERDICT_SCOPE_OK: citation of FORMULATION-scoped anchors (scopes stated at their verdicts) -->
resolutions; verdict_scope FORMULATION full-frame) + SPATIAL curve-relative REFUTED (`residual_kit`
probe2; FORMULATION). ⇒ the 0.074 residual enemy is coding-irreducible; the macro view RELOCATES bytes
and REFRAMES the enemy as COVERAGE, it does not shrink the residual by re-coding.

**Revised triple (labeled; P2 floor + verdict_scope on each):** dominant 0.061 → ~0.0585 (R2 ego-rigid
ξ-charge −0.0025 DERIVED, horizon intercept = banked pose ξ, ego-rigid only); complete-lossless 0.135 →
**~0.131 FIRM** (R1 exhaustive whole-set dedup −0.001 MEASURED [probe1b, band 0.0007–0.00137, INSTANCE
gt_n600 dilate=2] + R2 + R4 joint-coding −0.0005 DERIVED) → ~0.11–0.12 IF the owed R3 hood
single-static-store (candidate −[0,0.017] DERIVED-owed; the 0.028→0.0202 = 1.4×-only tell) and/or the
increment-2 R5 generator-coverage (−[0.015,0.02], the enemy's true home = Road/Lane 0.042 = 53% of
enemy) land; residual enemy 0.074 → **~0.0725 FIRM** (barely moves — CONFIRMS coding-irreducible).

**Consequence for this SPEC:** firm macro complete ~0.131 is STILL a WASH vs the 0.118 frontier — the
§0/§6/§SEAL-BOUNDARY-4 r\* RANGE [0.061, 0.135] and "shippable = WASH" verdict STAND. The macro pass
does NOT itself cross 0.118; it LOCATES the sub-frontier path (owed R3 $0 hood measure + increment-2 R5
coverage BUILD, §9). No sealed decision changes. Triality: DAG FEED-macro-rate + equations anchor
`v8_macro_rate_pass_exhaustive_dedup_and_egocharge_measured_20260710`; DSL N/A (rate derivation).

---

## GEOCODER-CLOSE + OWED-9 RECONCILIATION ADDENDUM (append-only — 2026-07-10; #398A "unlock v8")

**Scope:** this addendum RECONCILES the §1/§3 carrier table + §5/§8/§9 to the measured revisions
that landed AFTER the P7 transcription: the #394A geocoder-close (Road/Lane texture REFUTED → FLAT
fill; Movable sparse-site coder MEASURED), the macro-rate pass (both residual RE-CODING axes DEAD),
the #386 flip-bc gate (global b_c SATURATED at no_offset), and this unit's owed-9 lateral-carrier
BUILD (MEASURED FORMULATION NEGATIVE at the analytic level). It CHANGES NO sealed §1–§12 decision
and moves NOTHING on the pointer (**0.19110 UNMOVED**); it corrects the byte table + retires the
refuted pieces so the #385 brief is honest. `[macOS-CPU advisory · research-signal · NON-PROMOTABLE]`.

### Revised per-class byte table (provenance per entry; supersedes the §1/§3 pre-measurement view)

| edge / carrier | dominant S | complete S | FILL / carrier | provenance |
|---|---|---|---|---|
| Road/Lane | 0.0275 | 0.0695 (res 0.042) | **FLAT scene colour** (~15.6 B texture, rule-118) + geometry placement (horizon 4167 B + lane LBND2). **Texture grating REFUTED** | MEASURED n600 through-R `roadlane_grating_composition_refuted_v1` (#394A: grating +0.228 vs matched flat; Road wins flat 0.017). res 0.042 = **coverage gap → increment-2 R5**, NOT a coder gap |
| Road/Undriv (horizon top arc, I1) | **0.00277** (4167 B) | 0.0221 (res 0.0189) | deg-3 horizon poly + ξ intercept (ego-rigid) | MEASURED `horizon_poly_xi_byte_cost` (§I I1 PIN). Macro R2: dominant → ~0.0003 if intercept charged to banked pose ξ |
| Road/Undriv lateral extents (I1b, owed-9) | — | — | x_L(y), x_R(y) 2 low-order curves | **MEASURED R8 (this unit): deg-2 6426 B / S 0.00428 · deg-3 8773 B / S 0.00584** — lands in the §I I1b DERIVED range [0.0040, 0.0083] (**range CONFIRMED**). BUT the analytic convex-envelope form is a **d_seg NEGATIVE** (+0.019; fit residual ~20 px) — see the owed-9 verdict below |
| Movable (Road/Mov + Undriv/Mov) | 0.00344 | 0.0209 (res 0.0175) | **sparse-site coder MEASURED 6289 B** (2145 sites, K=9, tracked temporal-delta, 31% vs raw; box-IoU 0.743) | MEASURED n600 `movable_site_coder` (#394A). The 53% "Road/Lane coverage enemy" RELOCATES here (per-frame Movable/MyCar colour) + Lane boundary jitter — **not a texture gap** |
| Road/MyCar (hood, I4) | 0.0202 | 0.0202 (res 0) | one static silhouette (majority-vote) | MEASURED. Macro R3 CANDIDATE −[0, 0.017] owed $0 hood single-static-store measure (the 1.4×-only-vs-bitmap tell) |
| Lane/* (3 rows) | 0.007 | 0.007 | already tiny | MEASURED |
| **T-half texture trunk (#395)** | — | — | 375-param texture trunk (BUILT) | sibling build (READ-only here) |
| **Pose** | — | — | frame_0 carrier (Unit C DOF) + banked R1 dxi | d_pose contribution **0.127 / ξ_eff 7.2 KB** BANKED-as-artifact (MEMORY L68); NOT solved-for-v8 |

**Macro triple (from the macro-rate addendum, unchanged): dominant ~0.0585 · complete-lossless ~0.131
FIRM · residual enemy ~0.0725 FIRM** → **WASH vs 0.118 frontier**. Sub-frontier requires the owed R3
hood $0 measure and/or the increment-2 R5 generator-coverage BUILD. **The r\* RANGE [0.061, 0.135]
and "shippable = WASH" verdict STAND** (§0/§6/§SEAL-BOUNDARY-4 unchanged).

### owed-9 lateral 3-curve carrier — BUILT + MEASURED (this unit, #398A)

- **BUILT ($0, real, tested):** `road_undriv_bulk_field._lateral_extents` (per-row leftmost/rightmost
  Road column = the ego-rigid drivable envelope, blob-count-agnostic) + `lateral_extent_poly_byte_cost`
  (the R8 anchor) + `analytic_smoke.analytic_lateral_undriv_field` + the `--include-lateral` A/B flag.
  45 tests green, ruff F clean.
- **R8 MEASURED (turns §I I1b DERIVED → MEASURED anchor):** deg-2 6426 B (S 0.00428) · deg-3 8773 B
  (S 0.00584) — CONFIRMS the DERIVED [0.0040, 0.0083] range. **SEAL-BOUNDARY item 2 (I1b byte cost
  DERIVED-not-MEASURED) is now CLOSED for the byte-cost half.**
- **d_seg VERDICT — FORMULATION NEGATIVE (MEASURED n600):** the naive convex leftmost/rightmost
  envelope, as an ANALYTIC field, HURTS the decoupled floor: **horizon-only 0.100403 → +lateral(deg-2)
  0.119845 / (deg-3) 0.119060** (Road error 0.021 → 0.089 as the smooth low-order poly band cuts into
  true road; fit residual ~20 px — the jagged envelope is NOT a low-order poly, unlike the smooth
  ~1.5 px horizon). **verdict_scope: FORMULATION** (the hard analytic convex-hull envelope), NOT the
  lateral-carrier FAMILY. `--include-lateral` DEFAULT OFF; the canonical floor stays horizon-only 0.100403.
- **What this CARVES:** the lateral undriv field must be **margin-aware / learned** (the trained
  decoupled arm's per-class field), NOT a hard geometric poly hull. SEAL-BOUNDARY item 3 (the "3-curve
  homes 97.54% lateral" coverage) is **DERIVED-then-REFUTED-for-the-analytic-form**: the trained-field
  form is the open route (increment-1b). The lateral REPRESENTATION exists; the analytic INSTANTIATION
  is a measured no.

### #386 remainder reconciliation (retired-honestly / stands)

| #386 piece | status | verdict_scope |
|---|---|---|
| flip-weighted / flip-median b_c | BUILT + MEASURED n600; both **REFUTED** (6.3× / 6.9× worse); global b_c **SATURATED at no_offset** | FORMULATION (per-EDGE b_c on FRESH v8 Stage-A fields = the open 1b route) |
| de-share (Lever-1) | BUILT + MEASURED; **CONFIRMED S ≈ 0.0044** (folded into the 0.135→0.131 macro complete) | — (stands) |
| curve-relative δ(s) coder (Lever-2) | BUILT + MEASURED; **REFUTED** (0.99× / 0.90×) | FORMULATION. **REPLACEMENT per the flat-fill pin: NONE-as-a-coder** — the residual is coding-irreducible (macro: temporal ego-warp NO-GO + spatial curve-relative REFUTED). The reframe is **COVERAGE not coding** (increment-2 R5 lane generator) + de-share (CONFIRMED) + flat-fill placement |
| inc1a harness (5 deliverables) | BUILT; $0 analytic floor 0.100403 MEASURED | — (stands; owed-9 lateral extends it) |
| Movable sparse-site coder + roadlane texture generator (#394A) | BUILT + MEASURED; texture NO-GO, Movable 6289 B | FORMULATION (texture whole-region grating) |

**Nothing in #386 is left half-built.** Every piece is BUILT + MEASURED + verdicted; the refuted
formulations carry their reformulation queues (per-edge b_c, coverage-not-coding, margin-aware lateral).

Triality: DAG FEED-v8unlock · equations anchor `v8_lateral_extent_carrier_r8_measured_and_analytic_negative_20260710`
(appended to `v8_geometric_rate_decomposition_v1`) · DSL N/A (measurement + geometry primitive; no trainer
Lever — the trained decoupled-field lever is owed to the arm BUILD). Pointer 0.19110 UNMOVED.
