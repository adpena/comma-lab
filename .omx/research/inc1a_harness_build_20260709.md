# increment-1a HARNESS — the paint-free MASK-level DECOUPLING PARTITION SCREEN (BUILD, 2026-07-09)

**Task:** #386 increment-1a HARNESS builder (operator GO "Build all unbuilt"). **Surface:** new package
`src/tac/inc1a_harness/`. **Axis:** `[macOS advisory · research-signal · NON-PROMOTABLE]` — HARNESS = MEANS.
**Pointer contest-CPU 0.19110 UNMOVED.** The arms are trained in a later governed EVENT, NOT here.

## Answer-first
Built the apparatus that runs increment-1a EXACTLY as `SYNTHESIS_v2_v8_20260709.md` §A.4 (P3b F1) redefines
it: a **paint-free MASK-level decoupling partition screen** (composite tropical argmax vs GT SegNet argmax
L\* on `gt_n600.npz`), run as an **A/B against a matched-compute shared-head CONTROL arm**, with a
**pre-registered numeric kill** (decoupled mask d_seg must beat control's by > δ_mask). I did NOT resurrect
the through-R v1 form. **$0 smoke MEASURED: analytic-composite MASK d_seg = 0.100403 (n600)**, labeled
`[analytic-generators, no-trained-fields]` — a harness validation + first honest floor row, NOT the 1a verdict.

## STORES CONSULTED
`docs/operating_manual_craft_handoff.md` · `SYNTHESIS_v2_v8_20260709.md` (§A.4 1a-redefinition · §B
`measure_1a` · §G dedup audit · §H consumed surfaces · §I representation table · §E R7 δ_mask) ·
`P3_redteam_verdict_20260709.md` (F1 near-break + the "mask-optimal PROXY" latent fix) · CLAUDE.md
(OPERATOR PRIORITY: n600/no-toys, save-memories, class-order discipline; NO-FAKE; triality) · MEMORY.md
CURRENT-STATE · **PRIMARY CODE re-derived, not memo-trusted:** `laguerre_logit_offset.py`
(`power_diagram_argmax` @77, `solve_head_offsets` @398 with {menon,ot_newton,flip_weighted,flip_median},
`flip_median_offsets` @289 — the #386 median solver DOES exist, correcting the P3 "no median" claim which
was true only at P3's commit) · `bitmask_dseg.py` (`d_seg_reference` — the exact evaluate.py functional) ·
`perclass_verdict.py` (`per_class_flip_stats`/`per_class_dseg_fields`, `CLASS_NAMES`) · `movable_deshare.py`
(`detect_seg_roles` self-detecting class order) · `road_undriv_bulk_field.py` (`_horizon_profile`) ·
`lane_sdf_component.py` (`build_structured_lane_sdf`) · `hood_static_component.py`
(`compute_static_hood_mask`/`build_static_hood_sdf`) · `gt_n600.npz` (`lstars` (600,384,512) int64 0-4;
verified via numpy) · `reports/delta_R_noise_floor.json` (δ_R = 0.019590163230895963, the through-R p95 proxy).

## The 5 deliverables (all BUILT)
1. **Composite-argmax assembler** (`composite_assembler.py`): stacks per-class `CarrierField`s into a
   (H,W,K) field (Road complement sea + deep-negative for unmodeled classes) and runs the BUILT
   `power_diagram_argmax`. b_c **pluggable** over the #386 `solve_head_offsets` dispatcher — `no_offset`
   DEFAULT (S5-V6 safe; never ot_newton), `flip_weighted` (M-a MASS-OT), `flip_median` (M-b Hamming median,
   the #386 landing), `menon`/`ot_newton`. `reconcile_partition` = bounded linearized alternating projection
   (re-solve b_c with `pred` fed back to the current partition) with **max-iter cap + per-iter d_seg-
   monotonicity REJECT**; DEFAULT `max_iter=0` = single-pass tropical argmax; **NO Dykstra-convergence claim**
   (S3 honesty — SegNet-argmax pre-image is non-convex).
2. **Mask-level d_seg meter** (`mask_dseg_meter.py`): per-class + aggregate, n600, deterministic, numpy-ref;
   REUSES `d_seg_reference` (aggregate = exact evaluate.py functional) + `per_class_flip_stats` (per-class,
   flips attributed to GT class; sum identity holds). `require_n600` DEFAULT-ON — REFUSES a subset for any
   verdict-informing measurement (OPERATOR PRIORITY allergic-to-toys).
3. **Matched-compute CONTROL-arm spec + config** (`decoupling_screen.py` `ControlArmSpec`): same trunk, ONE
   shared 5-class head, params matched to the decoupled arm within `param_tolerance` (default ±5%), same
   seed/epochs/curriculum. **Matching rule + provenance stated**; `to_config_dict()` = the DSL authoring path
   (P7 compiles the typed WitnessProgram; this harness never edits `witness_dsl/`). Isolates DECOUPLING, not
   capacity/compute.
4. **Kill-criterion evaluator** (`decoupling_screen.py` `evaluate_kill`): reads both arms →
   {DECOUPLING-CONFIRMED (improvement > δ_mask; the ONLY gate-pass) / KILLED-at-δ_mask (worse by > δ_mask →
   FORMULATION falsified, NOT paradigm) / INCONCLUSIVE-below-floor (|Δ| ≤ δ_mask → underpowered, measure R7) /
   REFUSED (arm missing/toy/not-n600)}. δ_mask = δ_R 0.0196 PROXY with the caveat welded (NOT a measured mask
   floor — R7 owed).
5. **$0 smoke** (`analytic_smoke.py`): composes {deg-3 horizon poly, structured lane band, static hood} +
   Road complement over n600 → **0.100403 MEASURED**. Runnable: `python -m tac.inc1a_harness.analytic_smoke --json`.

## The measured smoke (the honest floor row)
`[analytic-generators, no-trained-fields]` · **agg MASK d_seg 0.100403** · n600 · 29 s · deterministic.

| class | mask d_seg | flip share |
|---|---|---|
| Road | 0.0212 | 4.9% |
| Lane | 0.362 | 2.1% |
| Undrivable | 0.162 | **79.7%** |
| Movable | **1.0 (unmodeled)** | 12.3% |
| MyCar (hood) | 0.0039 | 1.0% |

The shape is exactly F4-predicted: the single-valued horizon under-covers **lateral/multi-valued** Road↔Undriv,
so **Undrivable carries 79.7% of the flips** — the F4 lateral-undriv under-coverage is now EMPIRICALLY VISIBLE.
Movable=1.0 because it has no analytic SDF generator (folds to Road complement). Hood is near-perfect (static
mask). This is a HARNESS VALIDATION + a first honest floor row; **NOT the 1a verdict**.

## Clauses + NO-FAKE
- **Clause A (dedup/geometry-first):** every `CarrierField` states its UNIQUE geometric home (G1 horizon /
  G2 lane / G4 hood); duplicate-home is a raised error. The assembler stores NOTHING derivable — deterministic
  code = rule-118 FREE; only the callers' carrier params carry bytes.
- **Clause B (§I min-dim):** each carrier tagged GEOMETRIC-MINIMAL (I1 4-coeff horizon · I2 lane poly · I4 one
  static hood mask); the `CarrierField` constructor rejects any other representation mode.
- **NO-FAKE:** the assembler ACTUALLY composes (power-diagram argmax on real stacked fields — verified by the
  smoke's sensible per-class numbers); the meter ACTUALLY measures argmax disagreement (the exact functional);
  the kill evaluator REFUSES toys/non-n600. The 0.100403 is MEASURED n600, never borrowed.

## Own round-1 adversarial review (attacked before handoff)
- **Assumed keys/units:** `lstars` int64 0-4 (verified via numpy); `SegRoles` fields; `power_diagram_argmax`
  (phi (...,K), offsets (K,)); `per_class_flip_stats` takes LISTS (I pass lists). Class-order: the meter maps
  per-class by CLASS_NAMES[c] on the RAW GT index — the codebase convention; Movable=1.0 (index 3) confirms
  `detect_seg_roles.movable == 3 == CLASS_NAMES["Movable"]`, so the self-detected roles align with the label
  map (canonical comma10k order per the class-order discipline). Consistent.
- **Class not instance:** assembler/meter/kill are all GENERIC (any carriers/partitions/arms), not
  instance-hardwired.
- **Would tests pass if broken:** `test_meter_known_fixture` hard-codes the expected d_seg from a manual flip
  count; `test_no_offset_equals_argmax` checks the exact primitive; kill tests check exact verdict labels — all
  would fail on a broken meter/assembler/evaluator.
- **Reconcile guarantee:** monotone vs the SAME mode's single-pass start (reconcile never WORSENS the chosen
  bc_mode), NOT vs no_offset — the test asserts exactly that (the first test version compared the wrong
  baseline and was corrected). Using gt in the b_c solve is legitimate (flip_median/flip_weighted are designed
  to; the offset is a byte-free decode-time calibration).
- **Known limitation (honest):** the analytic smoke does not exercise the reconcile loop or non-default b_c
  (those are unit-tested separately); it is the assembler+meter integration on real n600 data, which is the
  mission's item-5 scope.

## Verification
23/23 harness tests pass; ruff F-clean; consumed-module regression (perclass_verdict + boundary_math seg_core)
31/31 pass (only ADDED files — no edits to siblings). Import graph clean.

## Triality
- **DAG** = `FEED-inc1a-harness` (this build + the measured 0.100403).
- **DSL** = the CONTROL-arm config authoring path (`ControlArmSpec.to_config_dict`); P7 compiles the typed
  `WitnessProgram`. No `witness_dsl/` edit here (sibling territory).
- **Equations** = **N/A-until-the-A/B-measures**: the decoupling law (decoupled-beats-control-by-δ_mask) is
  FORMALIZATION_PENDING until TRAINED arms produce a measured row. The analytic floor is a harness-validation
  instance, not a law — registering it as an equation would be premature.

Pointer 0.19110 UNMOVED. #205 untouched. This is MEANS — the END is a byte-closed `upstream/evaluate.py` n600
row < 0.19110, which arrives only after the arms are trained (governed EVENT) and the kill criterion fires
DECOUPLING-CONFIRMED → 1b through-R → byte-close.

## Canonical equations (Catalog #344)
# FORMALIZATION_PENDING: harness build memo — no measured rows; the increment-1a kill-gate law registers with the gate's first measured arm (control arm currently REFUSED-not-$0-rulable).
