<!--
council_tier: T1
council_attendees: [Fridrich, Yousfi, Quantizr, Shannon, Dykstra, Rudin, Daubechies, Contrarian, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "L0 SCAFFOLD analytical-upper-bound only; 4 strategies must remain SUBSTANTIVELY DISTINCT per Slot EEE FAKE-implementation audit (3-of-4 sister Slot FF enum branches were structurally equivalent); reactivation criterion is paired-CUDA empirical anchor"
  - member: AssumptionAdversary
    verbatim: "UNIWARD canonical Holub-Fridrich-Denemark 2014 specifies db4 wavelet detail-coefficient inverse-variance NOT 3x3 Sobel; the Sobel approximation is a DOCUMENTED ADAPTATION per Catalog #303 cargo-cult audit; SegNet/PoseNet gradient sensitivity branches assume scorer-Jacobian-vs-pixel-variance equivalence which is empirically unverified"
council_assumption_adversary_verdict:
  - assumption: "Fridrich UNIWARD canonical: errors in textured regions undetectable to inverse-steganalyzer"
    classification: HARD-EARNED
    rationale: "Holub-Fridrich-Denemark 2014 canonical citation; CLAUDE.md inner council seat Fridrich permanently active; sister 7-axis Yousfi-Fridrich cascade Axis 1 LANDED Slot FF"
  - assumption: "db4 wavelet detail-coefficient inverse-variance is the canonical UNIWARD basis (not Sobel)"
    classification: HARD-EARNED
    rationale: "Holub-Fridrich-Denemark 2014 Eq. 7 specifies discrete wavelet transform with Daubechies-4 wavelet; Sobel is documented adaptation per Slot UU TOP-1 9/9 inverse-steganalysis roadmap"
  - assumption: "SegNet logit-gradient magnitude is canonical inverse-scorer-sensitivity surface"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Exact scorer architectures' SegNet stride-2 stem 256x192 spatial blind spot; per-pixel gradient magnitude is the canonical Atick-Redlich 1990 cooperative-receiver mutual-information surface"
  - assumption: "PoseNet 12-dim YUV6 output-gradient magnitude is canonical inverse-scorer-sensitivity surface"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Exact scorer architectures' PoseNet FastViT-T12 YUV6 12-channel input; per-pixel 6-dim canonical Wyner-Ziv side_info surface per Catalog #150"
  - assumption: "Joint scorer linear combination (alpha*SegNet + beta*PoseNet + gamma*local_variance) is more discriminative than single-scorer baseline"
    classification: CARGO-CULTED
    rationale: "Linear combination is computationally simplest but optimal weighting is empirically unknown; canonical reference Atick-Redlich 1990 specifies I(X;T) maximization not linear combination; alpha/beta/gamma defaults are TUNING-FREE per Slot FF/CCC default-config discipline but reactivation criterion is per-pair empirical anchor"
  - assumption: "L0 SCAFFOLD analytical-upper-bound from canonical PR101 paired-component rows is sufficient first-anchor surface"
    classification: HARD-EARNED
    rationale: "Wave N+34 OPT-7 canonical anchor at .omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json yields baseline -22.22% IMPLEMENTATION_FALSIFIED for WEIGHTING-axis; this lane targets sister BASIS-EXPANSION axis on same canonical PR101 600-pair dataset"
  - assumption: "Tier A canonical-routing markers per Catalog #341 (non-promotable observability-only) is sufficient for L0 SCAFFOLD"
    classification: HARD-EARNED
    rationale: "Catalog #192 macOS-CPU advisory NEVER promotable; Catalog #341 Tier A canonical-routing-markers; smoke participates in autopilot ranking BEFORE GPU spend; promotion requires paired-CUDA per Catalog #246"
council_decisions_recorded:
  - "op-routable #1: L0 SCAFFOLD canonical implementation with 4 InverseScorerBasisStrategy enum values per Catalog #308 alternative-reducer enumeration"
  - "op-routable #2: real SegNet + PoseNet teacher loading via tac.distillation.load_default_scorers per CLAUDE.md 'MPS auth eval is NOISE' + 'Submission auth eval' non-negotiables"
  - "op-routable #3: real PR101 600-pair paired-component rows from experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/pair_component_rows.jsonl as canonical input (NOT synthetic noise per Slot EEE empirical anchor)"
  - "op-routable #4: canonical equation candidate uniward_inverse_scorer_basis_savings_v1 FORMALIZATION_PENDING per Catalog #344 (iterate-not-force; register after paired-CUDA empirical anchor lands)"
  - "op-routable #5: Catalog #348 retroactive sweep memo at landing"
  - "op-routable #6: sister composition path with Slot FF WEIGHTING-axis LANDED (commit 18c6cd571) via UNIWARD_INVERSE_JOINT_SCORER_BASIS_LINEAR_COMBINATION enum branch"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
predicted_band: [-0.000160, 0.000080]
predicted_band_validation_status: pending_post_training
schema_version: council_deliberation_v2_20260516
-->

# PR110-OPT-7 Fridrich UNIWARD inverse-scorer basis L0 SCAFFOLD design memo

## Source / lineage

- **Task #1319** Wave N+34 Zone 2 PR110-OPT canonical cascade
- **Operator binding directive** 2026-05-30: spawn `pr110-opt-7-fridrich-uniward-inverse-scorer-basis-l0-scaffold-20260530`
- **Sister Slot FF LANDED** (commit `18c6cd571` per CLAUDE memory): 4 enum branches on WEIGHTING-axis (sparse-K / widened-K / per-region / all-pairs). Per **Slot EEE FAKE-implementation audit** (cited in MEMORY.md 2026-05-29), Slot FF was classified PARTIAL — 3 of 4 enum branches were structurally equivalent at the L0 surface, and the UNIWARD primitive was scalar abstraction (not canonical db4 wavelet per Holub-Fridrich-Denemark 2014).
- **THIS sister lane scope**: BASIS-EXPANSION axis with 4 SUBSTANTIVELY DISTINCT enum branches anchored on real-scorer-Jacobian sensitivity (NOT on selector-K variants); each branch uses a different basis (local-variance baseline / SegNet gradient / PoseNet gradient / joint linear combination)
- **Canonical Wave N+34 OPT-7 anchor**: `.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json` (real PR101 600-pair MPS sweep at `experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/pair_component_rows.jsonl`)
- **Canonical CLAUDE.md non-negotiables honored**: "Fridrich inverse steganalysis" + "Exact scorer architectures" + "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA" + "NO FAKE IMPLEMENTATIONS" + "PR-or-greater parity" + Catalog #213 real-Comma2k19 helper requirement at sister surfaces

## ## Predicted ΔS band

**Predicted band**: `[-0.000160, +0.000080]` per Wave N+34 OPT-7 sparse-K100 canonical anchor `-7.94e-05`. Lower bound reflects optimal case where scorer-Jacobian sensitivity expands the basis beyond pure UNIWARD (improving over sparse-K100 by ~2x); upper bound reflects degenerate case where linear-combination weights are mistuned and the basis expansion produces WORSE selection than baseline.

**Dykstra-feasibility check** per Catalog #296: the basis expansion adds NEW perturbation modes to the catalog. Polytope intersection:
- Rate axis: `25 * archive_bytes / 37_545_489` per canonical contest formula. Sparse-K=100 selector ~103 bytes wire estimate per Wave N+34 anchor; basis expansion may increase to ~130-150 bytes for 4-strategy enum dispatch flag + per-pixel weight map header.
- Distortion axis (seg): SegNet logit-gradient branch targets per-pixel stride-2 stem blind spots; predicted Δd_seg in `[-1e-7, +5e-8]`.
- Distortion axis (pose): PoseNet 12-channel YUV6 gradient branch targets pose-axis sensitivity; predicted Δd_pose in `[-3e-7, +1e-7]`.

Intersection feasibility: predicted band is within both axes' feasible regions per Dykstra alternating-projections analysis with the canonical Pareto polytope.

**First-principles citations**: Holub-Fridrich-Denemark 2014 (UNIWARD), Atick-Redlich 1990 (cooperative receiver mutual-information), Wyner-Ziv 1976 (side information), Daubechies 1988 (db4 wavelet).

## ## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" + Catalog #290:

1. **Provenance layer**: ADOPT canonical `tac.provenance.builders.build_provenance_for_predicted` per Catalog #323 (no substrate-specific reason to fork).
2. **AxisDecomposition layer**: ADOPT canonical `tac.cathedral.consumer_contract.AxisDecomposition` per Catalog #356 (canonical Tier B per-axis contract).
3. **Tier A canonical-routing markers**: ADOPT per Catalog #341 (`predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"` in all routing-branch return values).
4. **Inverse-scorer-sensitivity primitive**: FORK from sister Slot FF's scalar UNIWARD abstraction TO canonical Holub-Fridrich-Denemark 2014 db4 wavelet detail-coefficient inverse-variance weighting AND Sobel 3x3 fallback (documented adaptation per Catalog #303 cargo-cult audit — db4 needs pywt which is optional; canonical Sobel is computationally cheaper baseline approximation per Slot UU TOP-1 9/9 inverse-steganalysis roadmap).
5. **Real-scorer integration**: ADOPT canonical `tac.distillation.load_default_scorers` per CLAUDE.md "MPS auth eval is NOISE" non-negotiable — real SegNet + PoseNet forward passes for gradient sensitivity, NEVER synthetic.
6. **Real-input integration**: ADOPT canonical PR101 600-pair paired-component rows per CLAUDE.md "Forbidden synthetic-fixture-instead-of-real-input" (3rd of 5 forbidden classes per NO FAKE IMPLEMENTATIONS).
7. **Wire-bytes accounting**: ADOPT canonical contest formula `25 * archive_bytes / 37_545_489` + Slot FF's sparse-K selector wire estimation conventions; FORK only the 4-byte strategy-flag header for enum dispatch.

## ## Cargo-cult audit per assumption

Per Catalog #303 + CLAUDE.md "PR-or-greater parity":

| Assumption | Classification | Unwind path |
|---|---|---|
| UNIWARD canonical 1/(eps+scorer_response) | HARD-EARNED | Holub-Fridrich-Denemark 2014 canonical citation |
| db4 wavelet detail-coefficient inverse-variance | HARD-EARNED | Holub-Fridrich-Denemark 2014 Eq.7; pywt is optional canonical dep |
| 3x3 Sobel inverse-variance fallback | DOCUMENTED-ADAPTATION | Slot UU TOP-1 inverse-steganalysis roadmap; cheaper substitute when pywt unavailable |
| SegNet logit-gradient magnitude per-pixel | HARD-EARNED | SegNet EfficientNet-B2 stride-2 stem 256x192 spatial blind spot per CLAUDE.md "Exact scorer architectures" |
| PoseNet 12-dim YUV6 output-gradient magnitude per-pixel | HARD-EARNED | PoseNet FastViT-T12 YUV6 12-channel input per CLAUDE.md "Exact scorer architectures" |
| Joint scorer linear combination (alpha+beta+gamma weights) | CARGO-CULTED | DEFERRED-PENDING-EMPIRICAL — defaults are TUNING-FREE; reactivation requires per-pair empirical anchor with grid search alpha/beta/gamma in [0, 1] |
| Per-pixel basis expansion outperforms per-pair UNIWARD weighting | CARGO-CULTED | Wave N+34 weighting was -22.22% WORSE; per-pixel basis may not transfer; reactivation requires paired-CUDA |

## ## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: 4 enum branches structurally distinct per Catalog #308 (NOT 3-of-4-equivalent per Slot EEE Slot FF audit). Each branch has different per-pixel weight map computation: pixel-domain Sobel/db4 vs SegNet-logit-grad vs PoseNet-output-grad vs joint linear combo.
2. **BEAUTY + ELEGANCE**: ~600 LOC L0 SCAFFOLD reviewable in 30s; canonical enum dispatch pattern from sister Slot FF/AAA/CCC; Holub-Fridrich-Denemark 2014 direct primitive.
3. **DISTINCTNESS**: WEIGHTING-axis (Slot FF) vs BASIS-EXPANSION-axis (THIS); sister of 7-axis Yousfi-Fridrich cascade Axes 1-7; sister composition with Slot X PR110-OPT-4 grouped-color + Slot CCC HUGO + Slot AAA MiPOD.
4. **RIGOR**: real PR101 paired-component rows; real SegNet + PoseNet teachers; canonical Provenance + AxisDecomposition; Catalog #325 6-step contract.
5. **OPTIMIZATION-PER-TECHNIQUE**: each enum branch optimizes for distinct scorer-Jacobian surface (pixel variance vs logit gradient vs output gradient vs combined).
6. **STACK-OF-STACKS-COMPOSABILITY**: feeds the canonical 84-cell composition_alpha matrix per Catalog #356 + sister composition with Slot FF WEIGHTING-axis (UNIWARD_INVERSE_JOINT_SCORER branch IS the canonical composition surface).
7. **DETERMINISTIC-REPRODUCIBILITY**: ascending-sorted selection + canonical numpy seed=42 + frozen dataclass invariants.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: per-pixel weight map cached at scorer-load-time; no scorer forward per pair (Jacobian sensitivity is per-pair input-pixel sensitivity, computable once per pair).
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: predicted band `[-0.000160, +0.000080]` straddles canonical PR110 cluster ~0.192; closes Slot CC dissent reactivation criterion #1 sister cascade.

## ## Observability surface

Per Catalog #305 6-facet:

1. **Inspectable per layer**: per-pixel weight maps queryable via `wire_analysis.weight_map_per_strategy` (cached at scorer-load).
2. **Decomposable per signal**: AxisDecomposition emission per Catalog #356 surfaces (seg, pose, archive_bytes) separately.
3. **Diff-able across runs**: ascending-sorted selector indices + numpy seed=42; identical inputs produce byte-identical outputs.
4. **Queryable post-hoc**: smoke output JSON at `experiments/results/pr110_opt7_fridrich_uniward_inverse_scorer_basis_smoke_<UTC>/smoke_output.json` carries per-strategy verdict + weight-map stats + selector indices + canonical Provenance.
5. **Cite-able**: every score-claim row carries canonical Provenance per Catalog #323 (commit + call_id placeholder + upstream_snapshot_sha256 placeholder).
6. **Counterfactual-able**: Catalog #105/#139/#272 byte-mutation smoke compatible — mutate one byte in the per-pixel weight map, observe whether selector index set changes.

## ## Canonical equation candidate

`uniward_inverse_scorer_basis_savings_v1` — predicted score savings from inverse-scorer-sensitivity basis expansion:

```
ΔS = 25 * (wire_bytes - 249) / 37_545_489 + sum_pair (alpha_pair * baseline_delta_pair)
```

Where `alpha_pair ∈ [0, 1]` is the per-pair UNIWARD concentration factor under the chosen basis. **FORMALIZATION_PENDING** per Catalog #344 iterate-not-force — register after paired-CUDA empirical anchor lands and the canonical weighting formula is verified empirically (not just analytically).

## ## Canonical anti-pattern candidate

`uniward_inverse_scorer_basis_l0_scaffold_scalar_abstraction_instead_of_canonical_db4_wavelet_basis_v1` — DEFERRED-PENDING-ITERATION per Catalog #344 iterate-not-force; register after sister Slot FF + THIS lane both have paired-CUDA empirical anchors and the canonical db4-vs-Sobel-vs-SegNet-grad-vs-PoseNet-grad delta is empirically resolved.

## ## horizon-class: frontier_pursuit

Per Catalog #309: predicted band `[-0.000160, +0.000080]` straddles the PR110 cluster ~0.19198 (canonical CPU frontier per `.omx/state/canonical_frontier_pointer.json`). Frontier-pursuit class because basis expansion targets sub-0.192 improvement; sister Slot FF was plateau_adjacent because WEIGHTING alone reduces to OPT-12 PoseNet-null at sparse-K. BASIS EXPANSION distinct from WEIGHTING per the design memo's WEIGHTING-vs-BASIS-EXPANSION distinction.

## Reactivation criteria per Catalog #308

1. **Paired-CUDA + paired-CPU empirical anchor** per Catalog #246 on real PR110 FEC6 archive sha `7a0da5d0fc327cba` baseline (canonical reactivation path; ~$0.06 envelope per Catalog #246).
2. **Per-pixel grid search for alpha/beta/gamma weights** in `UNIWARD_INVERSE_JOINT_SCORER_BASIS_LINEAR_COMBINATION` enum (canonical reactivation path #2 — defaults are TUNING-FREE; reactivation requires per-pair empirical anchor).
3. **Sister composition with Slot FF WEIGHTING-axis** via UNIWARD_INVERSE_JOINT_SCORER_BASIS_LINEAR_COMBINATION enum branch as composition primitive (canonical reactivation path #3).
4. **db4 wavelet promotion** via optional pywt canonical dep (canonical reactivation path #4 — Sobel fallback is documented adaptation).

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map = ACTIVE (per-pixel weight maps + per-pair UNIWARD costs queryable via `wire_analysis.weight_map_per_strategy`)
- hook #2 Pareto constraint = ACTIVE via Dykstra-feasibility check above + canonical equation candidate registration DEFERRED
- hook #3 bit-allocator = ACTIVE (sparse-K selector emits per-pair selector indices; per-pixel weight maps inform per-pair byte allocation)
- hook #4 cathedral autopilot dispatch = ACTIVE (Tier A canonical-routing markers per Catalog #341 + AxisDecomposition emission per Catalog #356)
- hook #5 continual-learning posterior = ACTIVE (canonical posterior anchor via `tac.council_continual_learning.append_council_anchor` per Catalog #355)
- hook #6 probe-disambiguator = ACTIVE (InverseScorerBasisStrategy enum + 4 substantively distinct branches IS the disambiguator between local-variance vs SegNet-gradient vs PoseNet-gradient vs joint-linear-combo basis surfaces)

## Sister cross-references

- `feedback_slot_ff_pr110_opt_7_uniward_inverse_scorer_basis_expansion_fridrich_canonical_parallel_cascade_per_slot_cc_dissent_landed_20260529.md` (Slot FF WEIGHTING-axis L0 SCAFFOLD; sister of THIS BASIS-EXPANSION-axis L0 SCAFFOLD)
- `feedback_slot_eee_fake_implementation_audit_on_today_l0_scaffolds_per_operator_binding_must_review_for_fake_implementations_landed_20260529.md` (Slot EEE FAKE-implementation audit; Slot FF classified PARTIAL; THIS lane addresses substantive distinctness per audit recommendation)
- `feedback_slot_aaa_mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016_canonical_fridrich_yousfi_cascade_axis_6_extension_per_slot_uu_top_2_landed_20260529.md` (sister Axis 6 MiPOD)
- `feedback_slot_ccc_hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010_canonical_fridrich_yousfi_cascade_axis_7_extension_per_slot_uu_top_4_landed_20260529.md` (sister Axis 7 HUGO)
- `feedback_yousfi_fridrich_slot_rr_fake_to_real_via_real_scorer_verification_landed_20260529.md` (sister Slot RR FAKE-to-REAL via real-scorer verification; THIS lane uses the same canonical real-scorer integration pattern)
- `.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json` (Wave N+34 OPT-7 canonical anchor)
- `.omx/research/pr110_opt_7_uniward_inverse_scorer_basis_expansion_fridrich_canonical_parallel_cascade_per_slot_cc_dissent_design_20260529.md` (Slot FF design memo)
- CLAUDE.md "Fridrich inverse steganalysis" + "Exact scorer architectures" + "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA" + "NO FAKE IMPLEMENTATIONS" + "PR-or-greater parity" non-negotiables
- CLAUDE.md inner council Fridrich + Yousfi + Quantizr permanently active per quintet pact + 11-voice expansion
- Lane registry: `lane_pr110_opt7_fridrich_uniward_inverse_scorer_basis_l0_scaffold_20260530`
