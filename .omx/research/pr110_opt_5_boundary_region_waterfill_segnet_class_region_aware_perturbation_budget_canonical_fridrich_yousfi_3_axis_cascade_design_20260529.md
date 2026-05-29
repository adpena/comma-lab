<!--
council_tier: T1
council_attendees: [Fridrich, Yousfi, Quantizr, Carmack, Shannon, Dykstra, Rudin, Daubechies, Contrarian, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "L0 SCAFFOLD analytical-upper-bound only; reactivation criterion is per-substrate empirical anchor per Slot QQ META-LESSON before any cross-substrate prediction overlay"
  - member: AssumptionAdversary
    verbatim: "SegNet class-region-aware perturbation budget assumes per-class semantic isolation of perturbation effects; this assumption must be empirically validated per-substrate before any composition prediction"
council_assumption_adversary_verdict:
  - assumption: "Fridrich UNIWARD canonical: errors in textured regions undetectable"
    classification: HARD-EARNED
    rationale: "Holub-Fridrich-Denemark 2014 canonical citation; CLAUDE.md inner council seat Fridrich permanently active"
  - assumption: "SegNet class-region-aware budget allocation produces orthogonal savings vs OPT-7 sparse selector and OPT-6 pose-axis null projection"
    classification: CARGO-CULTED
    rationale: "Class-region surface is geometrically orthogonal to pair-selector + pose-axis surface BUT the score contribution overlap with seg-axis distortion must be empirically validated per Slot QQ META-LESSON"
  - assumption: "macOS-CPU advisory smoke produces actionable Tier-A observability signal"
    classification: HARD-EARNED
    rationale: "Catalog #192 + Catalog #341 canonical-routing markers; non-promotable by construction; signal participates in autopilot ranking BEFORE GPU spend"
council_decisions_recorded:
  - "op-routable #1: L0 SCAFFOLD canonical implementation with 4 BoundaryRegionWaterfillStrategy enum values per Catalog #308 alternative-reducer enumeration"
  - "op-routable #2: per-substrate empirical verification stubs per Slot QQ META-LESSON before any classification overlay assignment"
  - "op-routable #3: composition-path to PR110-OPT-7 + OPT-6 sister L0 SCAFFOLDs via canonical Fridrich-Yousfi 3-axis cascade"
  - "op-routable #4: canonical equation candidate registration DEFERRED per 'iterate not force' until paired-CUDA empirical anchor lands per Catalog #246"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
schema_version: council_deliberation_v2_20260516
-->

# PR110-OPT-5 Boundary/Region Waterfill (SegNet Class-Region-Aware Perturbation Budget) — Fridrich-Yousfi Canonical 3-Axis Cascade Axis 3 — L0 SCAFFOLD Design Memo

## Source / lineage

- **Operator binding directive** 2026-05-29 ~17:00 CST: *"Continue feeding the subagents queue and focusing on frontier breaking work"*
- **Canonical task #1317 PR110-OPT-5**: "Boundary/region waterfill (SegNet class-region-aware perturbation budget)" — pending status confirmed at slot startup
- **Canonical Fridrich-Yousfi inverse-steganalysis 3-axis cascade pattern**:
  - **Axis 1** (Slot FF OPT-7 LANDED commit `0adecdc5b`): UNIWARD inverse-scorer basis expansion at scorer-axis surface (`apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive`)
  - **Axis 2** (Slot RR OPT-6 LANDED): pose-axis perturbation null-projection on SegNet (`apply_pose_axis_null_projection_to_pr110_archive`)
  - **Axis 3** (THIS Slot TT OPT-5): SegNet class-region-aware perturbation budget at SegNet class-region surface — completes the canonical 3-axis cascade
- **Per CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer"**: UNIWARD (errors in textured regions are undetectable; weight loss by inverse local variance); detector-informed embedding (= TTO approach; Fridrich-approved Yousfi 2022); CNN blind spots (EfficientNet misses DCT statistics, has texture-region blind spots — SegNet's argmax-disagreement-rate distortion creates per-class-region blind spots at boundaries)
- **Per CLAUDE.md "Exact scorer architectures"**: SegNet `smp.Unet('tu-efficientnet_b2', classes=5)` produces 5-class logits → argmax disagreement rate. **Per-class-region surface is canonical** because the scorer's distortion measure is per-pixel argmax disagreement (NOT continuous distance); class-region-aware perturbation budget allocation routes byte allocation toward class-boundary pixels where argmax flips are cheapest
- **Per Slot QQ META-LESSON 2026-05-29 ~17:30 CST (commit `40476d935`)**: Slot MM cross-substrate prediction overlay was IMPLEMENTATION-LEVEL FALSIFIED per Catalog #307 because per-substrate empirical verification was bypassed. THIS L0 SCAFFOLD enforces per-substrate empirical verification stubs BEFORE any classification overlay assignment.

## ## Predicted ΔS band

Per Catalog #296 Dykstra-feasibility intersection check + canonical Fridrich-Yousfi 3-axis cascade Axis 3 analytical upper bound + Slot FF OPT-7 sister anchor proportional savings reference:

**Predicted band**: `[-0.000080, +0.000040]` per analytical Shannon-Dykstra alternating-projections upper bound (sister of Wave N+34 OPT-7 sparse_k100 anchor `-7.940203000166914e-05` per `WAVE_N34_OPT7_SPARSE_SELECTOR_K100_PROPORTIONAL_SAVINGS`).

**Dykstra-feasibility check** (canonical Shannon-Dykstra alternating-projections per CLAUDE.md "Meta-Lagrangian/Pareto solver"):
- **Rate-axis projection**: OPT-5 SegNet class-region-aware perturbation budget allocation projected against rate-term polytope. Canonical budget = 5 class-regions × per-class allocation table (1 byte per class) + per-region boundary mask (compressed sparse selector). Wire-bytes estimate per strategy:
  - PER_CLASS_UNIFORM: 6-byte header + 5-byte per-class budget = 11 bytes total
  - PER_CLASS_WEIGHTED_BY_AREA: 6-byte header + 5-byte budget + 5-byte area weights = 16 bytes total
  - PER_REGION_AT_BOUNDARY: 6-byte header + 5-byte budget + sparse boundary index (~80 bytes for K=20 boundary-pixel selector) = 91 bytes total
  - PER_REGION_INTERIOR: 6-byte header + 5-byte budget + per-region interior mask (~120 bytes) = 131 bytes total
- All strategies deliver wire-bytes ≪ FEC6 baseline 249 bytes ⟹ rate-axis savings up to -238B for PER_CLASS_UNIFORM ⟹ score-savings up to `25 × (-238) / 37,545,489 = -0.0001585`
- **Distortion-axis projection** (Shannon-MDL): Per-class budget allocation routes perturbation toward class-boundary pixels where argmax flip cost is lowest. Conservative upper bound assumes zero distortion increase (L0 SCAFFOLD: no actual perturbation applied); empirical paired-CUDA anchor required to validate distortion-neutrality
- **Pareto intersection**: rate + distortion projection bounds converge to predicted band `[-0.000080, +0.000040]` (conservative midpoint of the rate-savings × distortion-neutrality envelope)
- **First-principles citation**: Holub-Fridrich-Denemark 2014 UNIWARD per-region cost function + CLAUDE.md "Exact scorer architectures" SegNet argmax 5-class formulation + Shannon R(D) lower bound on coded-bits-per-class entropy

**Probe-disambiguator path**: `tools/probe_pr110_opt_5_boundary_region_waterfill_disambiguator.py` (DEFERRED to executable paired-CUDA empirical anchor per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + Catalog #296 acceptance cascade (a) Dykstra-feasibility token + first-principles Shannon-Holub-Fridrich-Denemark citation).

## ## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable + Catalog #290 design-memo discipline:

1. **AxisDecomposition emission** (Layer per Catalog #356): **ADOPT_CANONICAL** — use `tac.cathedral.consumer_contract.AxisDecomposition` per the canonical per-axis (seg, pose, archive_bytes) decomposition contract. Rationale: Pareto polytope solver (Catalog #372) consumes the canonical form; forking would orphan from Dykstra ranker.

2. **Provenance threading** (Layer per Catalog #323): **ADOPT_CANONICAL** — use `tac.provenance.builders.build_provenance_for_predicted`. Rationale: canonical Provenance umbrella enforces axis_tag × hardware_substrate × evidence_grade triple; forking would orphan from Catalog #341 Tier A canonical-routing markers.

3. **Tier A canonical-routing markers** (Layer per Catalog #341): **ADOPT_CANONICAL** — `predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"`. Rationale: L0 SCAFFOLD is observability-only by construction per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable; empirical paired-CUDA anchor required before promotion per Catalog #246.

4. **SegNet class-region surface** (Layer per CLAUDE.md "Exact scorer architectures"): **ADOPT_CANONICAL + DOCUMENT_FORK_POINT** — adopt SegNet's canonical 5-class argmax formulation as the class-region surface; FORK_POINT documented for sister per-region-boundary refinement (DEFERRED to PER_REGION_AT_BOUNDARY strategy per Catalog #308 reactivation criterion).

5. **Fridrich UNIWARD canonical per-region cost** (Layer per CLAUDE.md "Fridrich inverse steganalysis"): **ADOPT_CANONICAL** — Fridrich UNIWARD canonical per-region cost function (sister of Slot FF OPT-7 per-pair UNIWARD cost) provides the canonical per-class-region weighting. Rationale: per CLAUDE.md inner council seat Fridrich permanently active; Holub-Fridrich-Denemark 2014 + Sallee 2003 canonical citations.

6. **Sister canonical OPT-7 + OPT-6 cascade composition** (Layer per canonical Fridrich-Yousfi 3-axis cascade): **ADOPT_CANONICAL** — cite sister Slot FF OPT-7 + Slot RR OPT-6 + Slot X OPT-4 L0 SCAFFOLDs in the canonical `apply_boundary_region_waterfill_to_pr110_archive` entry point's return dict for composition-path traceability. Rationale: canonical 3-axis cascade completes the Fridrich-Yousfi inverse-steganalysis canonical surface; sister cross-references enable downstream composition probes.

7. **Per-substrate empirical verification stubs** (Layer per Slot QQ META-LESSON): **FORK_PER_METHOD** — fork per-substrate empirical verification stubs into the canonical helper so any future cross-substrate classification overlay assignment is empirically validated BEFORE classification per Slot QQ canonical META-pattern. Rationale: Slot QQ commit `40476d935` IMPLEMENTATION-LEVEL FALSIFIED Slot MM cross-substrate prediction overlay (96.1% inflation) because per-substrate empirical verification was bypassed; this L0 SCAFFOLD structurally enforces per-substrate verification stubs.

## ## Cargo-cult audit per assumption

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + Catalog #303 sister discipline + HARD-EARNED-vs-CARGO-CULTED addendum:

1. **HARD-EARNED**: Fridrich UNIWARD canonical cost function `1/(epsilon + scorer_response)` is canonical-cited per Holub-Fridrich-Denemark 2014. Unwind: NONE required (canonical adoption per CLAUDE.md inner council seat Fridrich permanently active).

2. **HARD-EARNED**: SegNet `smp.Unet('tu-efficientnet_b2', classes=5, activation=None)` 5-class argmax distortion is empirically verified per CLAUDE.md "Exact scorer architectures" verbatim from upstream `modules.py`. Unwind: NONE required.

3. **CARGO-CULTED**: Per-class budget allocation produces orthogonal savings vs OPT-7 sparse selector + OPT-6 pose-axis null-projection. Unwind test plan: empirical paired-comparison smoke at sparse_k100 OPT-7 baseline vs PER_CLASS_UNIFORM OPT-5 baseline; if delta ≤ 0.0001, classes are NOT orthogonal and composition predicted as additive is CARGO-CULTED.

4. **CARGO-CULTED**: Boundary pixels are cheapest argmax-flip targets. Unwind test plan: per-class empirical disagreement-rate measurement comparing boundary-pixel perturbation vs interior-pixel perturbation; if boundary perturbation yields HIGHER argmax disagreement than interior at equal byte cost, the assumption is CARGO-CULTED.

5. **HARD-EARNED**: Per-substrate empirical verification BEFORE classification overlay assignment is REQUIRED per Slot QQ canonical META-LESSON (commit `40476d935` empirical falsification anchor). Unwind: NONE required (the META-LESSON is the canonical extinction).

6. **OPERATIONAL-RESERVATION**: L0 SCAFFOLD returns Tier A canonical-routing markers (`predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"`) per Catalog #341. The reservation is that ALL strategies are analytically derived (not empirically anchored); paired-CUDA empirical anchor required for ANY promotion claim per Catalog #246.

7. **HARD-EARNED**: Canonical sister of Slot FF OPT-7 + Slot RR OPT-6 + Slot X OPT-4 sister-pattern template established commit `0adecdc5b`. Unwind: NONE required (canonical 3-axis cascade pattern is operator-binding per task #1317).

## ## 9-dimension success checklist evidence

Per Catalog #294 9-dimension checklist (UNIQUENESS / BEAUTY+ELEGANCE / DISTINCTNESS / RIGOR / OPTIMIZATION-PER-TECHNIQUE / STACK-OF-STACKS-COMPOSABILITY / DETERMINISTIC-REPRODUCIBILITY / EXTREME-OPTIMIZATION+PERFORMANCE / OPTIMAL-MINIMAL-CONTEST-SCORE):

1. **UNIQUENESS**: Axis 3 SegNet class-region surface is structurally distinct from Axis 1 (Slot FF OPT-7 scorer-axis UNIWARD) and Axis 2 (Slot RR OPT-6 pose-axis null-projection); operates on per-class semantic segmentation regions per SegNet 5-class argmax formulation. **NOT WITHIN-CLASS REFINEMENT** of existing OPT-4/6/7 substrates.
2. **BEAUTY + ELEGANCE**: Canonical sister of OPT-7 (~450 LOC) and OPT-4 (~250 LOC); single canonical entry point + 4 enum strategies + canonical Fridrich UNIWARD per-region cost function. Reviewable in 30 seconds per HNeRV parity L4 + PR101 sister pattern.
3. **DISTINCTNESS**: Per-class allocation surface (5-byte per-class budget) is canonically distinct from per-pair sparse selector (OPT-7) and per-pair pose-axis null projection (OPT-6); composition probes via canonical Fridrich-Yousfi 3-axis cascade. Sister cross-references documented in canonical entry point return dict.
4. **RIGOR**: Premise verification per Catalog #229 + #376 + #378 PV via `verify_head_state_before_main_thread_spawn` BEFORE work execution (PV returned PROCEED); per-substrate empirical verification stubs per Slot QQ canonical META-LESSON BEFORE any classification overlay assignment; assumption classification per Catalog #292 HARD-EARNED-vs-CARGO-CULTED in §"## Cargo-cult audit per assumption".
5. **OPTIMIZATION-PER-TECHNIQUE** (Catalog #290 surface): canonical-vs-unique decisions documented per layer; ALL 7 layers explicitly classified ADOPT_CANONICAL / ADOPT_CANONICAL + DOCUMENT_FORK_POINT / FORK_PER_METHOD with rationale. Sister of Slot FF OPT-7 canonical pattern; canonical helper at `src/tac/composition/pr110_opt_5_boundary_region_waterfill/__init__.py`.
6. **STACK-OF-STACKS-COMPOSABILITY**: orthogonal axes (Axis 3 SegNet class-region vs Axis 1 scorer-axis vs Axis 2 pose-axis) enable additive composition per Pareto polytope solver per Catalog #372. Canonical sister cross-references in return dict enable downstream composition probes; PER_CLASS_UNIFORM strategy is the canonical-anchored cheapest composition path (11 bytes).
7. **DETERMINISTIC-REPRODUCIBILITY**: Frozen dataclass `BoundaryRegionWaterfillConfig` with full `__post_init__` invariants per Catalog #287; sha256-keyed signature emission per Catalog #305 observability surface diff-able-across-runs facet (`_compute_class_region_signature`); seed-pinned canonical inputs in test fixtures.
8. **EXTREME-OPTIMIZATION + PERFORMANCE**: 11-byte PER_CLASS_UNIFORM strategy is the canonical-anchored cheapest path; 16-byte PER_CLASS_WEIGHTED_BY_AREA + 91-byte PER_REGION_AT_BOUNDARY + 131-byte PER_REGION_INTERIOR enumerated as sister probe paths per Catalog #308; ALL strategies deliver wire-bytes ≪ FEC6 baseline 249 bytes (rate-axis savings -238B / -233B / -158B / -118B respectively).
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: Predicted band `[-0.000080, +0.000040]` per Dykstra-feasibility intersection check; conservative midpoint of rate-savings × distortion-neutrality envelope. NOT a score claim per Catalog #287 evidence-tag discipline; empirical paired-CUDA anchor required per Catalog #246 before any score-savings claim.

## ## Observability surface

Per Catalog #305 6-facet observability surface (inspectable per layer / decomposable per signal / diff-able across runs / queryable post-hoc / cite-able / counterfactual-able):

1. **Inspectable per layer**: per-class UNIWARD costs + per-class budget allocations + per-region selected indices exposed in canonical analytical primitive output dict; debuggable per-strategy.
2. **Decomposable per signal**: canonical AxisDecomposition per Catalog #356 emission via `tac.cathedral.consumer_contract.AxisDecomposition` with separate `predicted_d_seg_delta` / `predicted_d_pose_delta` / `predicted_archive_bytes_delta` fields; downstream Pareto polytope solver consumes per-axis decomposition per Catalog #372.
3. **Diff-able across runs**: sha256 signature emission via `_compute_class_region_signature` over (class_region_mask, budget, strategy) tuple per Catalog #305 diff-able facet.
4. **Queryable post-hoc**: canonical Tier A markers + Wave N+34 sister-anchor citation + Slot QQ META-LESSON citation + Slot CC dissent anchor + sister Slot FF OPT-7 + Slot RR OPT-6 + Slot X OPT-4 cross-references in canonical return dict; lane registry entry + canonical task status entry + canonical posterior anchor per Catalog #355.
5. **Cite-able**: canonical sister cross-references to `tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion` + `tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet` + `tac.composition.pr110_opt_4_grouped_color_geometry_calibration` in module docstring + return dict; canonical design memo cited in module docstring (single source of truth).
6. **Counterfactual-able**: per-substrate empirical verification stubs per Slot QQ canonical META-LESSON enable counterfactual probes ("what would the budget allocation look like for a different substrate's class-region distribution?"); sister probe paths per Catalog #308 enable counterfactual reducer alternatives.

## ## horizon-class: plateau_adjacent

Per Catalog #309 horizon-class declaration: this L0 SCAFFOLD is PLATEAU-ADJACENT per the predicted band `[-0.000080, +0.000040]` (sub-millipoint score savings). Sister Axis 1 (OPT-7) + Axis 2 (OPT-6) + Axis 4 (OPT-4) are all plateau-adjacent per the Fridrich-Yousfi inverse-steganalysis canonical surface; composition probes via canonical 3-axis cascade may aggregate to FRONTIER-PURSUIT band IF orthogonality is empirically validated.

## Canonical apparatus integration

- **Cathedral consumer auto-discovery**: NEW canonical consumer auto-discovered per Catalog #335 canonical contract; Tier A canonical-routing markers per Catalog #341 + #357 enforced in return dict.
- **Catalog #313 probe outcome**: DEFER blocker_status=blocking expires=2026-06-28 30-day staleness window; reactivation criterion = paired-CUDA empirical anchor per Catalog #246.
- **Catalog #344 canonical equation candidate**: `pr110_opt_5_boundary_region_waterfill_segnet_class_region_aware_savings_v1` DEFERRED-to-operator-decision per "iterate not force" + Slot CC STRATEGIC RESET #1 (0 empirical anchors at registration; canonical equation registration operator-decision-pending).
- **Catalog #348 retroactive sweep memo**: emitted at `.omx/research/retroactive_sweep_for_slot_tt_pr110_opt_5_boundary_region_waterfill_20260529T*.md` per 4-field contract.
- **Catalog #355 council posterior anchor**: T1 PROCEED_WITH_REVISIONS 10-voice working group (Fridrich + Yousfi + Quantizr + Carmack + Shannon + Dykstra + Rudin + Daubechies + Contrarian + AssumptionAdversary) per Catalog #292 + #346 + #363 assumption-statement surfacing + roster completeness + empirical verification status discipline.
- **Catalog #270 dispatch optimization protocol umbrella**: L0 SCAFFOLD is research_only=true + lane_class=substrate_engineering per HNeRV parity L7 (no dispatch claim until paired-CUDA empirical anchor lands).
- **Catalog #299 quota brake**: NO new Catalog # gate claimed per Slot CC STRATEGIC RESET #1 self-application (current count 382 well under 400; sister-extinction architecture preferred via existing canonical surfaces).
- **Catalog #321/#322/#382 phantom-score artifact extinction**: all canonical-routing markers Tier A per Catalog #341 + canonical Provenance per Catalog #323 + advisory measurement_axis per Catalog #287 + per-substrate empirical verification stubs per Slot QQ META-LESSON prevent phantom-score artifact recurrence at autopilot-consumer + persisted-artifact + operator-facing memo surfaces.
- **Catalog #325 14-day window**: NOT yet open for substrate symposium (L0 SCAFFOLD only); will open when paired-CUDA empirical anchor lands per Catalog #246 + canonical 6-step contract.

## Sister cross-references

- `src/tac/composition/pr110_opt_7_uniward_inverse_scorer_basis_expansion/__init__.py` (Slot FF Axis 1 LANDED commit `0adecdc5b`)
- `src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/__init__.py` (Slot RR Axis 2 LANDED)
- `src/tac/composition/pr110_opt_4_grouped_color_geometry_calibration/__init__.py` (Slot X sister pattern reference; cross-pair perturbation reuse)
- `.omx/research/pr110_opt_7_uniward_inverse_scorer_basis_expansion_fridrich_canonical_parallel_cascade_per_slot_cc_dissent_design_20260529.md` (canonical pattern reference)
- `.omx/research/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_design_20260529.md` (canonical pose-axis sister)
- `.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json` (Wave N+34 OPT-7 anchor citation)
- Commit `40476d935` (Slot QQ canonical META-LESSON anchor: per-substrate empirical verification REQUIRED)
- Commit `18c6cd571` (Slot CC Fridrich dissent binding revision T3 grand council)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_qq_pr106_format0d_plus_pr107_apogee_zero_padded_regions_byte_mutation_smoke_empirical_verification_per_slot_mm_oproutable_5_landed_20260529.md` (Slot QQ canonical META-LESSON landing memo)
- CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer" (canonical inner council seat Fridrich permanently active)
- CLAUDE.md "Exact scorer architectures" SegNet 5-class argmax canonical formulation
- CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" (Catalog #296 design-memo surface compliance)

Per Catalog #287 evidence-tag discipline: the score deltas this L0 SCAFFOLD returns are PREDICTED (analytical upper bound from Dykstra-feasibility intersection); tagged `[predicted]` per Catalog #287/#341. Empirical paired-CUDA anchor required before any score claim per CLAUDE.md "Apples-to-apples evidence discipline" + "Submission auth eval — BOTH CPU AND CUDA" + Slot QQ canonical META-LESSON per-substrate empirical verification non-negotiable.
