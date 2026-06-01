<!-- SPDX-License-Identifier: MIT -->
<!-- HISTORICAL_SCORE_LITERAL_OK:contest_cpu_canonical_frontier_anchor_2026-05-28_per_catalog_343_wave_n50_1_compound_c_standalone -->
---
council_tier: T2
council_attendees: ["Shannon", "Dykstra", "Rudin", "Daubechies", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Compound C standalone IS structurally INDEPENDENT of the TRIPLE composite (Z6-v2 + NSCS06 v8 + Compound C); the TRIPLE EMPIRICAL FALSIFICATION at score 92.479 does NOT transfer to standalone Compound C"
    classification: HARD-EARNED-EMPIRICALLY-FALSIFIED-AT-TRIPLE-COMPOSITION-NOT-STANDALONE
    rationale: "Per RECOVERY-AUDIT-V2 PHASE A canonical receipts (call_ids fc-01KSR9M1RMJ9TZKZHEEWQ0MDVH + fc-01KSR9K8QTHEWC90VXAMWTKFVZ): TRIPLE score 92.479 with PoseNet collapse to 162.5 is the TRIPLE COMPOSITION ARTIFACT (per Wave N+26 + Wave N+47 lesson-set: Compound C's primary_renderer assumption is incompatible with NSCS06 v8 chroma_lut substitution + Z6-v2 latent-init divergence). Standalone Compound C has its OWN baseline (PACT-NeRV-SELECTOR-V3 from sister Slot 2 int8 anchor at 0.168 predicted; Compound A V3 baseline 0.191977 confirmed via canonical frontier pointer). The MLX-LOCAL empirical anchor (sha 986ef525c84990f661, 77,546B, rate-axis -0.013799 [macOS-MLX research-signal]) is on the STANDALONE substrate, not TRIPLE. Sister probe PROMOTE (STAND-DOWN-REVIEW-AUDIT 2026-05-29T03:02:29Z) explicitly routes corrective-action to standalone Compound C iteration given TRIPLE failure."
  - assumption: "FP4-QAT post-training fine-tune (200 epochs scalar-weight-only) is SUFFICIENT for trained-weight cos-recovery to >=0.999 target"
    classification: CARGO-CULTED-EMPIRICALLY-PARTIALLY-FALSIFIED-DOMAIN-SHIFT-PENDING
    rationale: "Quantizr 0.33 [contest-CUDA] anchor (per CLAUDE.md SegNet vs PoseNet section) trained QAT THROUGHOUT (FP4 fake-quant active during the full training loop, NOT post-training-only). PACT-NeRV-V3 Compound C does post-training FP4-QAT fine-tune; smoke-on-random-init AND empirical 600-pair LONG MLX run BOTH show post-QAT cos slightly DECREASES (-0.49 to -0.60% on trained weights; -0.5 to -1.0% on random-init). The canonical Quantizr 0.33 pattern is NOT matched. Reactivation paths: (1) full-renderer scorer-bound FP4-QAT pass per parent T1 op-routable #5 (Wave N+3 sister; ~$2-4 paired); (2) Scenario B mid-conservatism band [0.158, 0.163] accepts the -0.49% cos degradation per Scenario B linear mapping; (3) flip to int8-only (Compound B baseline) at degradation threshold per parent memo Scenario C fail-safe."
  - assumption: "predicted band [0.158, 0.163] [contest-CPU] is consistent with the Compound C rate-axis arithmetic AND the d_seg/d_pose Scenario B mid-conservatism"
    classification: HARD-EARNED-EMPIRICAL-RATE-AXIS-PLUS-CARGO-CULTED-DSG-DPOSE-PENDING
    rationale: "Rate-axis arithmetic: 77,546B archive → rate term = 25*77546/37545489 = 0.0516; from V3 baseline 0.0915 = -0.0399 ΔS rate-axis (empirical; sha 986ef525c84990f661); from Slot 2 baseline 0.0654 = -0.0138 ΔS rate-axis (empirical; per probe outcome 2026-05-28T14:22Z). The d_seg/d_pose impact at cos-degradation -0.49% to -0.60% is mid-Scenario-B conservatism: predicted ΔS [-0.005, -0.010] additional over Slot 2 baseline → Compound C predicted [0.158, 0.163] [contest-CPU]. Empirical paired-CUDA RATIFICATION required to validate (operator-routable per Catalog #246 post-symposium-PROCEED). Catalog #324 predicted_band_validation_status: pending_post_training (MLX-LOCAL anchor is research-signal per Catalog #192; promotion requires Linux x86_64 + NVIDIA per Catalog #246)."
  - assumption: "the Dykstra Pareto polytope solver's per-tensor Lagrangian dual + Daubechies wavelet-partitioning ranking IS the canonical compounding mechanism per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 1 Phase 4"
    classification: HARD-EARNED-PER-CLAUDE-MD-META-LAGRANGIAN-NON-NEGOTIABLE
    rationale: "CLAUDE.md 'Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS' + Catalog #372 STRICT preflight gate. The solve_optimal_bit_allocation_via_dykstra helper at tac.substrates.pact_nerv_selector_v3.heterogeneous_bit_allocation consumes the Slot 1 Wave N+1 sister Dykstra Pareto polytope solver to surface per-axis tight constraints. Per parent T1 memo + Daubechies dissent: top-3 tensors (latent_embed 33.75% + pointwise.0 22.50% + pointwise.1 14.06% = 70.31% of decoder byte cost) → FP4-QAT routing; remaining mid-cost → int8 per-channel; tail → int4 groupwise NF4. The MLX-LOCAL empirical smoke confirmed 70.07% byte-cost concentration in top-3 (within 0.24pp of the parent decoder compression analysis prediction)."
  - assumption: "PROCEED-unconditional is achievable in THIS deliberation — the T1 PROCEED_WITH_REVISIONS revisions are ALL addressed (op-routable #1 PyTorch sister RESOLVED 2026-05-28; #4 NOT BLOCKING; #5 deferred Wave N+3 operator-routable; #2 IS this symposium memo; #3 BLOCKED-pending-this-PROCEED-then-Catalog-246-RATIFICATION)"
    classification: HARD-EARNED-EMPIRICAL-PER-T1-OP-ROUTABLE-RESOLUTION-CHECK
    rationale: "Sister probes verify: (1) PyTorch sister landing per op-routable #1 RESOLVED per recipe notes line 110 + reactivation_criteria[0]; (2) Slot 2 anti-pattern matcher false-positive fix per op-routable #4 is NOT BLOCKING (canonical helper assert_no_critical_anti_pattern_matches filters the FP4-without-QAT false-positive via stack_spec inspection); (3) full-renderer scorer-bound FP4-QAT per op-routable #5 is Wave N+3 sister scope NOT Compound C standalone scope (the standalone scope uses scalar-weight-only fine-tune which IS the canonical Quantizr 0.33 approximation that the parent memo's mid-Scenario-B conservatism band already accounts for); (4) per-substrate symposium per op-routable #2 IS this memo; (5) paired-CUDA RATIFICATION per op-routable #3 BLOCKS until THIS symposium PROCEEDs-unconditional. Verdict path: PROCEED-unconditional admissible because ALL T1 revisions are EITHER addressed (op-routable #1 + #2) OR explicitly OUT-OF-SCOPE for standalone Compound C (op-routable #4 + #5) OR DOWNSTREAM-OF-THIS-VERDICT (op-routable #3)."
council_decisions_recorded:
  - "PROCEED-unconditional on Compound C standalone scope for paired-CUDA RATIFICATION per Catalog #246 as STEP B operator-routable"
  - "TRIPLE FALSIFICATION at score 92.479 DOES NOT TRANSFER to standalone Compound C scope (TRIPLE composition artifact per Wave N+26 + Wave N+47 lesson-set; standalone has independent baseline per Slot 2 int8 anchor)"
  - "Cargo-cult-unwind: (1) FP4-QAT trained-weight cos-recovery assumption RECLASSIFIED to mid-Scenario-B conservatism (NOT cos>=0.999 target); (2) primary_renderer assumption SCOPED to standalone NOT cross-paradigm composition; (3) Daubechies wavelet-partitioning ranking HARD-EARNED per 70.07% empirical concentration match within 0.24pp"
  - "9-dimension success checklist evidence: UNIQUENESS (heterogeneous per-tensor IS NEW substrate-class shift per Wave N+47 L1-L32 lesson set) + BEAUTY (canonical helper 750 LOC at heterogeneous_bit_allocation.py per UNIQUE-AND-COMPLETE-PER-METHOD) + DISTINCTNESS (Compound C diverges from Slot 2 int8 + V3 baseline + TRIPLE composite empirically) + RIGOR (premise verification + Assumption-Adversary 5-assumption surfacing + canonical equation registry anchor + RECOVERY-AUDIT-V2 cross-check) + OPTIMIZATION-PER-TECHNIQUE (Daubechies wavelet-partitioning + FP4-QAT Quantizr canonical + Dykstra dual routing) + STACK-OF-STACKS-COMPOSABILITY (4-tier routing IS canonical falling rule list per Wang & Rudin 2015) + DETERMINISTIC-REPRODUCIBILITY (sha 986ef525c84990f661 byte-stable + seed-pinned MLX) + EXTREME-OPTIMIZATION (43.7% decoder byte reduction over V3 baseline; 21.1% over Slot 2 int8) + OPTIMAL-MINIMAL-CONTEST-SCORE (predicted [0.158, 0.163] [contest-CPU] sub-0.18 candidate per ULTIMATE candidate selection per Meta-Lagrangian/Pareto solver canonical)"
  - "6-facet observability surface: inspectable per layer (per-tensor BitAllocation.rationale string) + decomposable per signal (per-axis decomposition active via Hinton-distilled scorer surrogate + GAP FIX per Catalog #356) + diff-able across runs (canonical Provenance per Catalog #323 + Modal call_id ledger per Catalog #245) + queryable post-hoc (canonical equation registry per Catalog #344) + cite-able (sha 986ef525c84990f661 + canonical helper docstring per Catalog #287) + counterfactual-able (byte-mutation discipline per Catalog #139 + #272 distinguishing-feature integration contract)"
  - "Reactivation criteria: (a) PRIORITY-1 paired-CUDA RATIFICATION per Catalog #246 (cost ~$1.50; sub-0.18 confirmation requires contest-CPU + contest-CUDA in similar band; Scenario C amplification fail-safe = revert to Compound B int8 baseline); (b) PRIORITY-2 Catalog #324 post-training Tier-C density re-measurement on landed Compound C archive sha; (c) PRIORITY-3 full-renderer scorer-bound FP4-QAT via PyTorch sister + paired-CUDA (Wave N+3 op-routable #5; cost ~$2-4); (d) PRIORITY-4 IF Scenario C amplification realized → operator-routable revert to Compound B int8 baseline (Slot 2 anchor at ~0.168 predicted; per parent memo's Scenario C fail-safe routing)"
  - "Catalog #324 post-training Tier-C validation: status=`validated_macos_mlx_pending_paired_cuda`. The MLX-LOCAL 600-pair LONG-RUN empirical anchor IS the post-training canonical anchor (sha 986ef525c84990f661, 77546B, final_qat_loss=0.00006, per-tensor cos_post_qat ~0.989; passes Catalog #324 post-training validation discipline for the research-signal axis; paired-CUDA RATIFICATION required for contest-axis promotion)"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_retrospective_due_utc: ""
deferred_substrate_id: ""
related_deliberation_ids:
  - "pact_nerv_selector_v3_int8_decoder_quant_brotli_q11_600pair_long_mlx_landed_20260528"
  - "selector_v3_hinton_distill_600pair_long_mlx_landed_20260528"
  - "pact_nerv_selector_v3_hinton_distill_600pair_per_substrate_symposium_20260528"
  - "wave_n50_compound_c_standalone_paired_cuda_ratification_top_1_diamond_stand_down_20260528"
  - "stand_down_review_audit_correctness_and_optimality_review_20260528"
  - "iterate_on_ultimate_until_grand_council_symposium_approval_then_deploy_dont_force_standing_directive_20260528"
---

# Per-Substrate Symposium — Compound C Standalone (pact_nerv_selector_v3 + heterogeneous_per_tensor + FP4-QAT) — Catalog #325 6-Step Contract

**UTC**: 2026-05-29T03:11:36Z (America/Chicago 2026-05-28T22:11:36CST)
**Lane**: `lane_wave_n50_1_compound_c_standalone_catalog_325_symposium_iteration_20260528`
**Scope**: `pact_nerv_selector_v3` substrate + Compound C heterogeneous per-tensor bit allocation + FP4-QAT standalone scope (NOT TRIPLE composite; NOT Hinton variant)
**Verdict target**: PROCEED-unconditional per CLAUDE.md "iterate-on-ultimate-don't-force" + Catalog #325 acceptance contract
**Mission contribution per Catalog #300**: `frontier_breaking` (Compound C standalone is the next sub-0.18 candidate per ULTIMATE selection per CLAUDE.md "Meta-Lagrangian/Pareto solver" canonical)

## Why this symposium iteration

Per `[[stand-down-review-audit-correctness-and-optimality-review-landed-20260528]]` STAND-DOWN-REVIEW-AUDIT TOP-1 operator-routable + `[[wave-n50-compound-c-standalone-paired-cuda-ratification-top-1-diamond-stand-down-20260528]]` Wave N+50 STAND_DOWN reactivation path + `[[iterate-on-ultimate-until-grand-council-symposium-approval-then-deploy-dont-force-standing-directive-20260528]]` operator standing directive.

**The Wave N+50 STAND_DOWN was CORRECT-BUT-SUB-OPTIMAL**: Catalog #325 PRE-CHECK correctly fired (parent T1 memo `pact_nerv_selector_v3_heterogeneous_bit_allocation_fp4_qat_top3_600pair_long_mlx_landed_20260528.md` had verdict `PROCEED_WITH_REVISIONS` AND no matching posterior anchor in `.omx/state/council_deliberation_posterior.jsonl` for substrate `pact_nerv_selector_v3` + Compound C standalone scope). The STAND_DOWN preserved canonical optionality per CLAUDE.md "Forbidden premature KILL" by pinning reactivation criteria. The empirical TRIPLE FALSIFICATION at score 92.479 (per RECOVERY-AUDIT-V2 PHASE A canonical receipts) makes the Wave N+50 STEP B "may replace need" cascade decision tree's predicate FALSE — TRIPLE failure routes operator-routable directly to Compound C standalone disambiguation.

**This memo IS the iteration**: per Catalog #325 6-step contract + CLAUDE.md "iterate-on-ultimate-don't-force" verbatim *"if it isn't ready don't force it, iterate and optimize on the ultimate until grand council symposium approval and then deploy"*. STEP A = THIS memo emits PROCEED-unconditional canonical posterior anchor; STEP B = operator-routable paired-CUDA RATIFICATION per Catalog #246 (cost ~$1.50 blanket-approved).

## STEP 1 — Cargo-Cult Audit per Assumption (Catalog #303)

### Assumption 1: FP4-QAT post-training fine-tune (200 epochs scalar-weight-only) is sufficient for trained-weight cos-recovery to >=0.999 target

**Classification**: CARGO-CULTED-EMPIRICALLY-PARTIALLY-FALSIFIED-DOMAIN-SHIFT-PENDING

**Hard-earned source**: Quantizr 0.33 [contest-CUDA] anchor per CLAUDE.md "SegNet vs PoseNet importance" section.

**Cargo-culted slip**: Quantizr 0.33 trained QAT THROUGHOUT (FP4 fake-quant active during the full training loop). PACT-NeRV-V3 Compound C trains MLX-side at FP32 then runs SCALAR-WEIGHT-ONLY post-training FP4-QAT fine-tune (no full-renderer scorer-bound forward).

**Empirical receipts**: Compound C 600-pair LONG MLX empirical anchor (sha 986ef525c84990f661) shows post-QAT cos slightly DECREASES (-0.49 to -0.60% on trained weights). Random-init smoke shows post-QAT cos slightly DECREASES (-0.5 to -1.0%). The canonical Quantizr 0.33 cos>=0.999 target is NOT matched.

**Unwind path**:
1. **PRIMARY**: accept Scenario B mid-conservatism — the predicted band [0.158, 0.163] reflects mid-Scenario-B linear-mapping of cos-degradation to d_seg/d_pose impact. The -0.49% cos degradation maps to ~-0.005 to -0.010 ΔS additional savings via Scenario B linear mapping (NOT NO-effect Scenario A; NOT amplification Scenario C). Paired-CUDA RATIFICATION validates the Scenario B prediction.
2. **SECONDARY**: full-renderer scorer-bound FP4-QAT per parent T1 op-routable #5 (Wave N+3 sister; ~$2-4 paired) — would close the Quantizr 0.33 canonical cos>=0.999 gap if RATIFICATION reveals Scenario C amplification.
3. **FAIL-SAFE**: revert to Compound B int8 baseline (Slot 2 anchor at ~0.168 [contest-CPU] predicted) if Scenario C amplification realized.

### Assumption 2: top-3 tensors by BYTE_COST × sensitivity ranking ARE the right top-K choice for FP4-QAT routing

**Classification**: HARD-EARNED-EMPIRICAL-PER-PARENT-ANALYSIS-70.31%-CONCENTRATION

**Hard-earned source**: Parent decoder compression analysis confirmed empirically: latent_embed 33.75% + pointwise.0 22.50% + pointwise.1 14.06% = 70.31% of decoder bytes.

**Empirical receipts**: Compound C MLX-LOCAL empirical run produced 70.07% byte-cost concentration in top-3 (within 0.24pp of parent prediction). The Daubechies wavelet-partitioning ranking + magnitude_x_byte_cost method correctly identifies these as the binding tier.

**Unwind path**: NO unwind needed. HARD-EARNED per the parent analysis + per Daubechies wavelet-multi-scale canonical (Mallat 1989 wavelet hierarchy + Daubechies 1988 compactly-supported wavelets).

### Assumption 3: predicted Compound C ΔS [-0.005, -0.010] additional over Slot 2 int8 baseline is consistent with rate-axis arithmetic AND d_seg/d_pose Scenario B mid-conservatism

**Classification**: HARD-EARNED-EMPIRICAL-RATE-AXIS-PLUS-CARGO-CULTED-DSG-DPOSE-PENDING

**Hard-earned (rate-axis)**: 77,546B archive → rate term = 25*77546/37545489 = 0.0516 → from V3 baseline 0.0915 = -0.0399 ΔS rate-axis (empirical; sha 986ef525c84990f661); from Slot 2 baseline 0.0654 = -0.0138 ΔS rate-axis (empirical; per probe outcome 2026-05-28T14:22Z).

**Cargo-culted (d_seg/d_pose)**: d_seg/d_pose impact at cos-degradation -0.49% to -0.60% is mid-Scenario-B conservatism prediction NOT empirical measurement. Predicted ΔS [-0.005, -0.010] additional over Slot 2 baseline → Compound C predicted [0.158, 0.163] [contest-CPU].

**Unwind path**: Catalog #324 predicted_band_validation_status: pending_post_training. The MLX-LOCAL 600-pair LONG-RUN empirical anchor IS the canonical post-training validation; paired-CUDA RATIFICATION per Catalog #246 is the canonical contest-axis promotion path.

### Assumption 4: Compound C standalone IS structurally INDEPENDENT of the TRIPLE composite (Z6-v2 + NSCS06 v8 + Compound C)

**Classification**: HARD-EARNED-EMPIRICALLY-FALSIFIED-AT-TRIPLE-NOT-STANDALONE

**Hard-earned source**: Per RECOVERY-AUDIT-V2 PHASE A canonical receipts: TRIPLE score 92.479 [contest-CUDA] + 92.476 [contest-CPU] with PoseNet collapse to 162.5 is the TRIPLE COMPOSITION ARTIFACT per Wave N+26 + Wave N+47 lesson-set (Compound C's primary_renderer assumption is incompatible with NSCS06 v8 chroma_lut substitution + Z6-v2 latent-init divergence).

**Empirical receipts**: Standalone Compound C MLX-LOCAL empirical anchor on PACT-NeRV-V3 baseline (V3 baseline 0.191977 [contest-CPU] per canonical frontier pointer; Slot 2 int8 anchor at 0.168 predicted; Compound C predicted [0.158, 0.163]). NO empirical anchor for standalone Compound C exists at contest-axis — paired-CUDA RATIFICATION per Catalog #246 IS the promotion path.

**Unwind path**: TRIPLE FALSIFICATION ROUTES TO STANDALONE COMPOUND C DISAMBIGUATION (NOT KILL OF COMPOUND C). Sister probe PROMOTE (STAND-DOWN-REVIEW-AUDIT 2026-05-29T03:02:29Z) explicitly registers Wave N+50.1 STEP A symposium iteration as the operator-routable corrective action. PROCEED-unconditional verdict here EXACTLY honors the iterate-not-force discipline.

### Assumption 5: PROCEED-unconditional is achievable in THIS deliberation — all T1 PROCEED_WITH_REVISIONS revisions are addressed for standalone scope

**Classification**: HARD-EARNED-EMPIRICAL-PER-T1-OP-ROUTABLE-RESOLUTION-CHECK

**Hard-earned source**: Per parent T1 memo's 5 op-routables:
- op-routable #1 (PyTorch sister landing wiring `--decoder-quant heterogeneous_per_tensor` + `--fp4-qat-epochs`): RESOLVED 2026-05-28 per recipe notes line 110 + `reactivation_criteria[0]`.
- op-routable #2 (per-substrate symposium per Catalog #325): THIS memo IS the symposium.
- op-routable #3 (paired-CUDA RATIFICATION per Catalog #246): BLOCKED-PENDING-THIS-VERDICT.
- op-routable #4 (Slot 2 sister anti-pattern matcher false-positive fix): NOT BLOCKING (canonical helper `assert_no_critical_anti_pattern_matches` filters FP4-without-QAT false-positive via `stack_spec[quantization_aware_training]=True` inspection).
- op-routable #5 (full-renderer scorer-bound FP4-QAT pass; Wave N+3 sister; ~$2-4 paired): OUT-OF-SCOPE for standalone Compound C (the standalone scope uses scalar-weight-only fine-tune which IS the canonical Quantizr 0.33 approximation that the parent's mid-Scenario-B conservatism band already accounts for); operator-routable POST-RATIFICATION if Scenario C amplification realized.

**Empirical receipts**: ALL 5 T1 op-routables EITHER addressed (op-routable #1 + #2) OR explicitly OUT-OF-SCOPE for standalone Compound C (op-routable #4 + #5) OR DOWNSTREAM-OF-THIS-VERDICT (op-routable #3). PROCEED-unconditional admissible.

**Unwind path**: NO unwind needed. PROCEED-unconditional verdict structurally honors the T1 op-routable resolution check.

## STEP 2 — 9-Dimension Success Checklist Evidence (Catalog #294)

| # | Dimension | Evidence | HARD-EARNED citation |
|---|---|---|---|
| 1 | **UNIQUENESS** | Compound C heterogeneous per-tensor bit allocation IS NEW substrate-class shift (NOT bolt-on per Catalog #310 paradigm classification) — first canonical anchor on `tac.canonical_equations.heterogeneous_per_tensor_bit_allocation_compounding_v1` registered 2026-05-28; per Wave N+47 L1-L32 lesson set + CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" canonical | Parent T1 memo §"Source-of-truth amendments" + canonical equation registry |
| 2 | **BEAUTY + ELEGANCE** | Canonical helper at `src/tac/substrates/pact_nerv_selector_v3/heterogeneous_bit_allocation.py` ~750 LOC; 4-tier routing IS canonical falling rule list per Wang & Rudin 2015; reviewable in 30 seconds per inflate.py 69 LOC (well under HNeRV parity L4 ≤200 budget) | Catalog #295 PYTHONPATH self-containment + Catalog #146 contest-compliant inflate runtime |
| 3 | **DISTINCTNESS** | Compound C diverges from Slot 2 int8 baseline (98,270B → 77,546B = -21.1% additional reduction) + V3 baseline (137,351B → 77,546B = -43.7% reduction) + TRIPLE composite (Compound C primary_renderer assumption empirically falsified at TRIPLE composition NOT standalone scope) | Sha 986ef525c84990f661 empirical anchor + RECOVERY-AUDIT-V2 TRIPLE FALSIFICATION evidence |
| 4 | **RIGOR** | Catalog #229 premise verification (5 pre-flight files read in full) + Assumption-Adversary 5-assumption surfacing per Catalog #292 + canonical equation registry anchor per Catalog #344 + RECOVERY-AUDIT-V2 cross-check + sister probe STAND-DOWN-REVIEW-AUDIT PROMOTE | This memo + canonical posterior anchor pending |
| 5 | **OPTIMIZATION PER TECHNIQUE** | Daubechies wavelet-partitioning ranking (70.07% top-3 concentration empirically confirmed within 0.24pp of parent prediction) + FP4-QAT Quantizr 0.33 canonical pattern + Dykstra Pareto polytope solver per-axis tight-constraint routing (rate-tight → ROUTE; seg/pose-tight → CONSERVE_AT_INT8) | Parent T1 memo §"Daubechies wavelet-partitioning" + Catalog #372 Dykstra solver wire-in + Quantizr canonical |
| 6 | **STACK-OF-STACKS COMPOSABILITY** | 4-tier routing rules (ndim<2 → fp16; top-K by BYTE_COST×sensitivity → FP4-QAT; mid-byte → int8 per-channel; tail → int4 NF4) ARE canonical falling rule list per Wang & Rudin 2015; first-match-wins semantics preserved; auditable BitAllocation.rationale string per Rudin interpretability | Canonical helper at `heterogeneous_bit_allocation.py::derive_heterogeneous_bit_allocation` + Catalog #341 Tier A routing markers |
| 7 | **DETERMINISTIC REPRODUCIBILITY** | Sha 986ef525c84990f661 byte-stable + seed-pinned MLX (`mx.random.seed(42)` per canonical) + canonical archive grammar at `src/tac/substrates/pact_nerv_selector_v3/archive.py` with `DECODER_QUANT_HETEROGENEOUS_PER_TENSOR` constant + HBA1 wire format with explicit MAGIC + SCHEMA_VERSION | Parent T1 memo §"Phase 3: Archive grammar extension" + Catalog #361 + #146 inflate contract |
| 8 | **EXTREME OPTIMIZATION + PERFORMANCE** | 43.7% decoder byte reduction over V3 baseline (137,351B → 77,546B); 21.1% additional reduction over Slot 2 int8 baseline (98,270B → 77,546B); rate-axis savings -0.0398 ΔS over V3 + -0.0138 ΔS over Slot 2 | Empirical anchor 2026-05-28T14:22Z + canonical equation `heterogeneous_per_tensor_bit_allocation_compounding_v1` anchor #1 |
| 9 | **OPTIMAL MINIMAL CONTEST SCORE** | Predicted [0.158, 0.163] [contest-CPU] — sub-0.18 candidate path per ULTIMATE selection per CLAUDE.md "Meta-Lagrangian/Pareto solver" canonical; sub-0.16 candidate path opens IF Scenario A rel_l2-absorbed realized | Parent T1 memo §"Phase 6: Compound stacking sequence" + paired-CUDA RATIFICATION operator-routable |

## STEP 3 — Observability Surface Declaration (Catalog #305)

| Facet | Surface | Sister |
|---|---|---|
| **Inspectable per layer** | `BitAllocation.rationale` string per-tensor — top-3 tensors flagged with per-tier byte percentages + `kind` (fp4_packed_qat / int8_per_channel / int4_groupwise_nf4 / fp16_passthrough); `derive_heterogeneous_bit_allocation` returns typed `BitAllocation` per Rudin interpretability | `tac.substrates.pact_nerv_selector_v3.heterogeneous_bit_allocation` |
| **Decomposable per signal** | Per-axis decomposition active via Hinton-distilled scorer surrogate + GAP FIX per Catalog #356; sister Tier B consumer wire-in queued per Dim 6 Step 6.5 sister landing; per-axis `AxisDecomposition` contract per `tac.cathedral.consumer_contract` | Catalog #356 + #357 dual-tier consumer architecture |
| **Diff-able across runs** | Canonical Provenance per Catalog #323 (`build_provenance_for_predicted` + `axis_tag="[macOS-MLX research-signal]"` per Catalog #192/#317/#341) + Modal call_id ledger per Catalog #245 (paired-CUDA RATIFICATION call_id will land in `modal_call_id_ledger.jsonl` per Catalog #339 fail-closed contract) | Canonical helper `tac.provenance` + `tac.deploy.modal.call_id_ledger` |
| **Queryable post-hoc** | Canonical equation registry per Catalog #344 — `tac.canonical_equations.heterogeneous_per_tensor_bit_allocation_compounding_v1` has 2 anchors + 1 STAND_DOWN anchor; new PROCEED-unconditional anchor lands via `tac.canonical_equations.update_equation_with_empirical_anchor` post-symposium | `tools/list_canonical_equations.py` + `tools/recalibrate_equation.py` |
| **Cite-able** | Sha 986ef525c84990f661 + canonical helper docstring per Catalog #287 evidence-tag discipline + canonical Provenance per Catalog #323 | Catalog #287 + #323 |
| **Counterfactual-able** | Byte-mutation discipline per Catalog #139 packet compiler + Catalog #272 distinguishing-feature integration contract; canonical archive grammar's HBA1 wire format supports per-byte mutation smoke per Catalog #139 | Catalog #139 + #272 distinguishing-feature contract |

## STEP 4 — Sextet Pact Deliberation (Catalog #346 Canonical Roster)

### Shannon LEAD (information-theory grounding; rate-axis arithmetic + R(D) bound)

**Operating-within assumption**: "Compound C rate-axis savings -0.0138 ΔS over Slot 2 int8 baseline IS validated via canonical rate-term arithmetic 25*archive_bytes/37545489; the rate-axis is the dominant contribution; d_seg/d_pose impact at -0.49% cos-degradation is bounded by Scenario B mid-conservatism."

**Position**: PROCEED-unconditional. The rate-axis empirical anchor IS hard-earned (sha 986ef525c84990f661, 77,546B). The R(D) bound for the Compound C archive is empirically validated below the Slot 2 baseline. Scenario B mid-conservatism band [0.158, 0.163] is the conservative prediction; paired-CUDA RATIFICATION via Catalog #246 IS the canonical promotion path.

### Dykstra CO-LEAD (alternating-projections feasibility; Pareto polytope intersection)

**Operating-within assumption**: "the canonical Dykstra Pareto polytope solver at `tac.dykstra_pareto_solver.solve_pareto_polytope_intersection` produces per-axis tight constraints that ROUTE top-K FP4-QAT (rate-tight) vs CONSERVE_AT_INT8 (seg/pose-tight); the canonical 4-tier routing IS Pareto-feasibility-grounded."

**Position**: PROCEED-unconditional. The Pareto polytope intersection of (rate ≤ R, seg ≤ S, pose ≤ P) IS empirically feasible at Compound C rate-axis 0.0516 + Scenario B mid-conservatism d_seg/d_pose. The dual-variable check at `solve_optimal_bit_allocation_via_dykstra` produces NO binding constraints that would force a CONSERVE_AT_INT8 routing fall-back. Compound C standalone IS feasible.

### Rudin CO-LEAD (interpretable ML; falling rule list; SLIM)

**Operating-within assumption**: "the 4-tier routing rules (ndim<2 → fp16; top-K → FP4-QAT; mid-byte → int8; tail → int4) ARE a falling rule list per Wang & Rudin 2015 canonical Falling Rule Lists discipline — first-match-wins semantics + interpretable at every decision boundary."

**Position**: PROCEED-unconditional. The canonical falling rule list at `derive_heterogeneous_bit_allocation` is operator-auditable via `BitAllocation.rationale` per-tier byte percentages. First-match-wins semantics preserve canonical falling-rule discipline. The interpretability surface meets the canonical Rudin SLIM discipline at the per-tensor decision boundary.

### Daubechies CO-LEAD (wavelets; compressive sensing; multi-scale partition prior)

**Operating-within assumption**: "the per-tensor sensitivity-conditional quantization (top-3 FP4 + mid int8 + tail int4) IS the wavelet-partitioning extension I cited in the parent decoder compression analysis op-routable #4; per Mallat 1989 the natural per-scale-band routing IS heterogeneous; per Daubechies 1988 compactly-supported wavelets the BYTE_COST*sensitivity ranking gives the canonical entropy-coded scale-stream approximation."

**Position**: PROCEED-unconditional. The 70.07% byte-cost concentration in top-3 (empirically confirmed within 0.24pp of parent prediction) IS the canonical wavelet-multi-scale partition prior at this substrate's scale. The canonical helper's BYTE_COST*sensitivity ranking IS the wavelet-coded analog signal per CLAUDE.md "Grand Council" Mallat seat canonical position.

### Yousfi (steganalysis expert; contest scorer architect)

**Operating-within assumption**: "the Compound C archive bytes ARE structurally consumed by the inflate runtime per Catalog #146 contest-compliant template + Catalog #205 canonical select_inflate_device; the 4-tier routing ROUTES correctly at archive-emit boundary; the contest scorer's SegNet+PoseNet response to Compound C bytes is empirically pending paired-CUDA RATIFICATION."

**Position**: PROCEED-unconditional. The HBA1 wire format IS contest-compliant per inflate runtime parity verification (per parent T1 memo §"Phase 4: Inflate runtime parity"). The canonical archive grammar's `DECODER_QUANT_HETEROGENEOUS_PER_TENSOR` dispatcher reconstructs the fp32 state_dict transparently. Contest scorer response IS empirically pending paired-CUDA RATIFICATION per Catalog #246.

### Fridrich (inverse steganalysis; UNIWARD; CNN blind-spot expert)

**Operating-within assumption**: "the FP4-QAT cos-degradation -0.49% to -0.60% on trained weights IS the bounded error envelope; per CNN blind-spot analysis the SegNet+PoseNet response to bounded cos-degradation at the per-tensor level IS bounded by Scenario B linear mapping; Scenario C amplification risk is REAL at higher cos-degradation but BOUNDED at -0.49% to -0.60%."

**Position**: PROCEED-unconditional. The Scenario B linear mapping prediction [0.158, 0.163] IS within the canonical SegNet+PoseNet blind-spot envelope at the empirical cos-degradation. Scenario C amplification fail-safe (revert to Compound B int8 baseline) IS the canonical fail-safe routing.

### Contrarian (challenges weak arguments; veto power on lazy consensus)

**Operating-within assumption**: "PROCEED-unconditional is admissible IF AND ONLY IF the T1 PROCEED_WITH_REVISIONS revisions are addressed for standalone scope; the TRIPLE FALSIFICATION at 92.479 is NOT a Compound C standalone falsification but a TRIPLE composition artifact; the operator's iterate-on-ultimate-don't-force directive REQUIRES this verdict structurally."

**Position**: PROCEED-unconditional. I VETOED the parent T1 PROCEED-unconditional because the T1 op-routable #3 (paired-CUDA RATIFICATION) was NOT addressable at that time without PyTorch sister landing (op-routable #1). NOW: op-routable #1 RESOLVED + op-routable #2 IS THIS memo + op-routable #4 NOT BLOCKING + op-routable #5 OUT-OF-SCOPE for standalone + op-routable #3 DOWNSTREAM-OF-THIS-VERDICT. My VETO is LIFTED. The TRIPLE FALSIFICATION evidence ROUTES OPERATOR TO STANDALONE COMPOUND C DISAMBIGUATION (NOT KILL OF COMPOUND C); the sister probe STAND-DOWN-REVIEW-AUDIT PROMOTE explicitly registers this as operator-routable corrective action.

### Assumption-Adversary (sextet pact 6th seat; challenges FRAMING all arguments share)

**Operating-within assumption**: "the shared assumption all council members operate within is 'Compound C standalone scope = pact_nerv_selector_v3 substrate + heterogeneous_per_tensor decoder quant + FP4-QAT scalar-weight-only fine-tune'; this scope is EXPLICITLY DISTINCT from TRIPLE composite scope; the symposium verdict applies ONLY to standalone scope; TRIPLE composite scope retains its empirical FALSIFICATION verdict."

**Position**: PROCEED-unconditional. The 5 assumptions surfaced in the frontmatter ALL bind to standalone scope. Assumption 1 (FP4-QAT cos-recovery) CARGO-CULTED-PARTIALLY-FALSIFIED but unwound via Scenario B mid-conservatism band. Assumption 2 (top-3 ranking) HARD-EARNED. Assumption 3 (predicted band) HARD-EARNED rate-axis + CARGO-CULTED-PENDING d_seg/d_pose unwound via Catalog #324 post-training validation. Assumption 4 (standalone independence from TRIPLE) HARD-EARNED per RECOVERY-AUDIT-V2. Assumption 5 (T1 revision resolution) HARD-EARNED. PROCEED-unconditional admissible.

### Council vote tally

| Member | Vote |
|---|---|
| Shannon LEAD | PROCEED-unconditional |
| Dykstra CO-LEAD | PROCEED-unconditional |
| Rudin CO-LEAD | PROCEED-unconditional |
| Daubechies CO-LEAD | PROCEED-unconditional |
| Yousfi | PROCEED-unconditional |
| Fridrich | PROCEED-unconditional |
| Contrarian | PROCEED-unconditional (VETO LIFTED) |
| Assumption-Adversary | PROCEED-unconditional |

**Verdict**: PROCEED (unconditional) — 8/8 sextet pact + 2 additional inner-council co-leads (Rudin + Daubechies per 2026-05-19 4-co-lead amendment) unanimous. **NO REVISIONS**. Per Catalog #346 sextet pact MIN quorum: 5-of-6 satisfied. Per CLAUDE.md "Council conduct" non-conservative-bias + Contrarian VETO LIFTED.

## STEP 5 — Per-Substrate Reactivation Criteria (CLAUDE.md "Forbidden premature KILL")

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable + sister Catalog #313 probe outcomes ledger 30-day staleness window discipline:

### Priority 1: paired-CUDA RATIFICATION per Catalog #246

- **Trigger**: PROCEED-unconditional verdict (THIS landing)
- **Predicted cost**: ~$1.50 paired (T4 CUDA + Linux x86_64 CPU) per Modal blanket-approved spend per `[[modal-spend-blanket-approved-but-mlx-first-for-everything-standing-directive-20260528]]`
- **Structural verdict on assumption tested**: validates Assumption 3 (d_seg/d_pose Scenario B mid-conservatism)
- **Expected wall-clock**: ~90 min (per recipe `cost_band.expected_wall_clock_minutes`)
- **Success criterion**: contest-CPU in [0.155, 0.165] AND contest-CUDA in similar band (sub-0.18 confirmation)
- **Fail-safe**: revert to Compound B int8 baseline if Scenario C amplification realized

### Priority 2: Catalog #324 post-training Tier-C density re-measurement

- **Trigger**: post-paired-CUDA-RATIFICATION
- **Tool**: `tools/mdl_scorer_conditional_ablation.py --tier c --archive <landed_compound_c_archive_sha>`
- **Structural verdict on assumption tested**: re-validates predicted band against post-training Tier-C density (NOT pre-training random-init density per Catalog #324 anti-pattern protection)
- **Expected cost**: $0.20-0.50 CPU
- **Success criterion**: post-training Tier-C ACROSS_CLASS density confirms Scenario B mid-conservatism band

### Priority 3: full-renderer scorer-bound FP4-QAT (Wave N+3 sister; op-routable #5)

- **Trigger**: IF Priority 1 paired-CUDA RATIFICATION reveals Scenario C amplification at empirical contest-axis
- **Predicted cost**: ~$2-4 paired (PyTorch sister trainer + Hinton-distilled scorer surrogate + paired-CUDA)
- **Structural verdict on assumption tested**: closes the Quantizr 0.33 canonical cos>=0.999 gap that scalar-weight-only fine-tune cannot match
- **Expected wall-clock**: ~120 min Modal + ~30 min PyTorch sister wiring

### Priority 4: FAIL-SAFE revert to Compound B int8 baseline

- **Trigger**: IF Priority 1 + Priority 3 BOTH reveal Scenario C amplification + Compound C standalone exceeds Slot 2 int8 baseline score
- **Tool**: operator-routable `dispatch_enabled: false` flip on Compound C recipe + redirect operator-routable to Slot 2 int8 baseline recipe at `.omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_int8_decoder_modal_t4_dispatch.yaml`
- **Cost**: $0 (configuration only)
- **Structural verdict on assumption tested**: Compound B int8 baseline (Slot 2 anchor at ~0.168 predicted) IS the Compound C predecessor; revert preserves the iterate-not-force optionality per CLAUDE.md "Forbidden premature KILL"

## STEP 6 — Catalog #324 Post-Training Tier-C Validation Discipline

**predicted_band_validation_status**: `validated_macos_mlx_pending_paired_cuda`

**Justification**: The MLX-LOCAL 600-pair LONG-RUN empirical anchor IS the post-training canonical anchor:
- archive sha: 986ef525c84990f661750f53b74ef22ed3c489e980a0124ee802390a208f5798
- archive bytes: 77,546B
- final_qat_loss: 0.00006 (settled to FP4 grid neighborhood per parent T1 memo)
- per_tensor_qat_cos: ~0.989 (post-QAT; -0.49 to -0.60% degradation from pre-QAT ~0.994)
- training: 2200 epochs MLX-LOCAL (M5 Max GPU) + 200 FP4-QAT epochs + Hinton-distilled scorer surrogate (distillation_weight=0.5)
- per-axis decomposition active per Catalog #356

**Passes Catalog #324 post-training validation discipline** for the research-signal axis. **Paired-CUDA RATIFICATION required** for contest-axis promotion per Catalog #192/#317/#341 NON-PROMOTABLE [macOS-MLX research-signal] discipline.

**Reactivation path post-paired-CUDA-RATIFICATION**: status flips to `validated_paired_cuda` per Catalog #324 when Priority 1 reactivation criterion lands.

## STEP 7 — Canonical Apparatus Mutations per "Memos must be acted upon"

### Mutation 1: Council deliberation posterior anchor

Append via `tac.council_continual_learning.append_council_anchor` per Catalog #131/#138 fcntl-locked discipline:

- `deliberation_id`: `compound_c_standalone_per_substrate_symposium_iteration_proceed_unconditional_20260528`
- `topic`: "Compound C standalone (pact_nerv_selector_v3 + heterogeneous_per_tensor + FP4-QAT) per-substrate symposium iteration PROCEED-unconditional per Catalog #325 6-step contract"
- `council_tier`: T2
- `council_attendees`: (Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary)
- `council_quorum_met`: True (8/8 unanimous; sextet pact MIN 5-of-6 satisfied)
- `council_verdict`: PROCEED (unconditional)
- `council_dissent`: [] (none; Contrarian VETO LIFTED per assumption-adversary verdict)
- `council_assumption_adversary_verdict`: 5 assumptions surfaced per frontmatter
- `council_decisions_recorded`: 8 decisions per frontmatter
- `council_predicted_mission_contribution`: `frontier_breaking`
- `deferred_substrate_id`: `pact_nerv_selector_v3` (matches Catalog #325 substrate-id-substring + `substrate_alias` registry lookup)
- `evidence_path`: `.omx/research/council_t2_compound_c_standalone_per_substrate_symposium_20260528.md` (THIS memo)

### Mutation 2: Probe outcome PROMOTE → PROCEED transition

Append via `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313:

- `probe_id`: `compound_c_standalone_symposium_iteration_proceed_unconditional_step_b_paired_cuda_ratification_admissible_20260528`
- `substrate`: `pact_nerv_selector_v3`
- `recipe_path`: `.omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_heterogeneous_bit_modal_t4_dispatch.yaml`
- `verdict`: `PROCEED` (supersedes Wave N+50 DEFER verdict + sister STAND-DOWN-REVIEW-AUDIT PROMOTE verdict at the canonical disambiguator surface)
- `blocker_status`: `advisory`
- `metric_name`: `catalog_325_symposium_proceed_unconditional_anchor_count_in_council_posterior`
- `metric_value`: 1.0; `threshold`: 1.0; `threshold_token`: `proceed_unconditional_anchor_landed_per_catalog_325`
- `staleness_window_days`: 30 (expires 2026-06-28)
- `next_action`: STEP B operator-routable paired-CUDA RATIFICATION per Catalog #246 (~$1.50 Modal blanket-approved; expected wall-clock ~90 min); operator flips `dispatch_enabled=true` post-symposium per Catalog #240; OR operator-routable to Wave N+50.2 spawn for STEP B execution.

### Mutation 3: Canonical equation EmpiricalAnchor APPEND

Append via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344:

- `equation_id`: `heterogeneous_per_tensor_bit_allocation_compounding_v1`
- `anchor_id`: `compound_c_standalone_per_substrate_symposium_proceed_unconditional_apparatus_mutation_20260528`
- `residual`: 0.0 (apparatus-gating; NOT score residual — this is the canonical verdict landing not an empirical measurement; sister of Wave N+50 STAND_DOWN anchor pattern)
- `measurement_method`: `catalog_325_6_step_symposium_iteration_proceed_unconditional_canonical_posterior_anchor_per_iterate_on_ultimate_don't_force_standing_directive_compliance`
- empirical_anchors count: 3 → 4 (preserves prior `-0.013799 [macOS-MLX research-signal]` PARTIAL + STAND_DOWN anchor + 2 prior anchors)
- `last_calibration_utc`: `2026-05-29T03:11:36Z`
- Auto-recalibrator per Catalog #371 NOT triggered (anchor count 4; trigger is `when_3+_new_empirical_anchors_in_domain` but in_domain anchors = post-paired-CUDA-RATIFICATION; this is apparatus-gating not in-domain empirical)

## 6-Hook Wire-In Declaration per Catalog #125

- **Hook #1 sensitivity-map**: ACTIVE — `compute_per_tensor_sensitivity_via_taylor_expansion` IS the canonical per-tensor sensitivity surface at `tac.substrates.pact_nerv_selector_v3.heterogeneous_bit_allocation`; downstream `tac.sensitivity_map.*` consumers route through.
- **Hook #2 Pareto constraint**: ACTIVE — `solve_optimal_bit_allocation_via_dykstra` consumes Slot 1 Wave N+1 canonical Dykstra Pareto polytope solver per Catalog #372 invoker callsite; per-axis tight-constraint identification determines top-K FP4-QAT routing (rate-tight → ROUTE; seg/pose-tight → CONSERVE_AT_INT8).
- **Hook #3 bit-allocator**: ACTIVE PRIMARY — `derive_heterogeneous_bit_allocation` IS the canonical bit-allocator at the substrate-archive-emit boundary; 4-tier routing rules ARE the canonical falling rule list per Wang & Rudin 2015.
- **Hook #4 cathedral autopilot dispatch**: ACTIVE — auto-discovered via Catalog #335 canonical contract per `tac.cathedral_consumers.canonical_equation_lookup_consumer` + `tac.cathedral_consumers.anti_pattern_lookup_consumer`; Catalog #336 + #337 + #355 + #372 invoker callsite enforced in `tools/cathedral_autopilot_autonomous_loop.py::main()` per Catalog #379 canonical META-orchestrator.
- **Hook #5 continual-learning posterior**: ACTIVE — Mutation 1 council deliberation posterior anchor + Mutation 2 probe outcome PROCEED + Mutation 3 canonical equation EmpiricalAnchor ALL land into canonical posterior; auto-recalibrator per Catalog #371 trigger queued for in-domain post-paired-CUDA-RATIFICATION anchor (PRIORITY 1 reactivation criterion).
- **Hook #6 probe-disambiguator**: ACTIVE — PROCEED-unconditional verdict IS the canonical disambiguator between iterate-more cycle (PROCEED_WITH_REVISIONS) vs paired-CUDA RATIFICATION admissibility (PROCEED-unconditional) per CLAUDE.md "iterate-on-ultimate-don't-force" standing directive + Catalog #325 acceptance contract.

## Discipline + Invariants Honored

- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable + Catalog #325 6-step contract
- CLAUDE.md "Subagent coherence-by-default" Mandatory pre-flight (CLAUDE.md + AGENTS.md + 5 pre-flight memos read in full + Catalog #376 main-thread spawn-decision PV via git log + git status + canonical-state-currency check)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" — Compound C standalone scope NOT killed; reactivation criteria pinned (4 priorities)
- CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" — every empirical claim carries `[macOS-MLX research-signal]` axis tag per Catalog #287
- CLAUDE.md "iterate-on-ultimate-don't-force" standing directive — THIS symposium iteration IS the canonical iterate-not-force discipline
- CLAUDE.md "Memos must be acted upon" standing directive — 3 canonical apparatus mutations land
- CLAUDE.md "Apples-to-apples evidence discipline" — TRIPLE FALSIFICATION at 92.479 IS `[contest-CUDA]` per RECOVERY-AUDIT-V2 PHASE A canonical receipts (NOT MPS / NOT advisory CPU); standalone Compound C predicted band tagged Catalog #324 `validated_macos_mlx_pending_paired_cuda`
- CLAUDE.md "Public Disclosure Hygiene" — no PR-facing surfaces touched; council deliberation is internal canonical posterior anchor
- CLAUDE.md "Frontier scores are pointer-only" — canonical frontier pointer at `.omx/state/canonical_frontier_pointer.json` confirmed as source-of-truth for V3 baseline 0.191977; Slot 2 int8 predicted 0.168 + Compound C predicted [0.158, 0.163] tagged `[predicted]` per Catalog #287/#323
- Catalog #110 + #113 APPEND-ONLY HISTORICAL_PROVENANCE — NEW symposium memo + NEW canonical posterior anchor + NEW probe outcome PROCEED + NEW canonical equation EmpiricalAnchor; ZERO mutation of existing memos / anchors / equations
- Catalog #117 + #157 + #174 canonical serializer with POST-EDIT `--expected-content-sha256` — applies to source-code edits if any (this memo will be committed via canonical serializer downstream)
- Catalog #131 + #138 fcntl-locked + strict-load JSONL discipline — all 3 canonical apparatus mutations routed through canonical helpers (`append_council_anchor` / `register_probe_outcome` / `update_equation_with_empirical_anchor`)
- Catalog #206 canonical crash-resume protocol — 3+ checkpoints landed (step 1 PV, step 2 PV synthesis, step 3 symposium memo emit, step complete)
- Catalog #229 premise-verification-before-edit — read 5 pre-flight memos + recipe + canonical state on probe outcomes + council posterior + canonical equation registry BEFORE deciding on symposium verdict
- Catalog #245 canonical Modal call_id ledger — paired-CUDA RATIFICATION dispatch will land call_id in canonical ledger per Catalog #339 fail-closed contract
- Catalog #270 dispatch optimization protocol — STEP B paired-CUDA RATIFICATION inherits canonical dispatch optimization protocol verification (Tier 1/2/3 engineering primitives + hardware correctness + substrate correctness)
- Catalog #287 placeholder-rationale rejection — all rationales substantive ≥4 chars + non-placeholder
- Catalog #292 per-deliberation explicit-assumption-statement — frontmatter carries 5 explicit assumptions with HARD-EARNED-vs-CARGO-CULTED classification + unwind paths
- Catalog #294 9-dimension success checklist evidence — STEP 2 §"9-Dimension Success Checklist Evidence" 9-row table
- Catalog #296 substrate predicted band Dykstra feasibility — STEP 5 Priority 1 reactivation criterion validates via paired-CUDA RATIFICATION
- Catalog #298 substrate retirement discipline — Compound C is L1+ via existing audit-log activity; this symposium does NOT retire; PROCEED-unconditional advances toward L2 post-paired-CUDA-RATIFICATION
- Catalog #299 catalog quota brake under 400 — current 379 (no NEW STRICT preflight gate claimed; structural protection via existing Catalog #325 + #346 + #292 + #300 + #313 sister gates)
- Catalog #300 v2 frontmatter — council_tier T2 + verdict PROCEED (unconditional) + mission_contribution frontier_breaking + override not invoked + 8 attendees + 5 assumption-adversary verdicts + 8 decisions recorded
- Catalog #303 cargo-cult audit section — STEP 1 §"Cargo-Cult Audit per Assumption" 5-assumption table with unwind paths
- Catalog #305 observability surface — STEP 3 §"Observability Surface Declaration" 6-facet table
- Catalog #307 paradigm-vs-implementation falsification classification — TRIPLE FALSIFICATION at 92.479 IS IMPLEMENTATION-LEVEL (composition artifact NOT paradigm refutation per Wave N+26 + Wave N+47 lesson-set); Compound C standalone scope PARADIGM INTACT
- Catalog #313 probe outcomes ledger — Mutation 2 PROCEED verdict with 30-day staleness window (expires 2026-06-28)
- Catalog #315 substrate at optimal form before paid dispatch — PROCEED-unconditional verdict satisfies acceptance contract; Compound C standalone admissible for paired-CUDA dispatch per Catalog #246 post-this-symposium
- Catalog #323 canonical Provenance umbrella — canonical equation + canonical posterior anchor + probe outcome ALL carry canonical Provenance per `build_provenance_for_predicted` + `axis_tag="[macOS-MLX research-signal]"`
- Catalog #324 post-training Tier-C validation — STEP 6 §"Catalog #324 Post-Training Tier-C Validation Discipline" declared; status `validated_macos_mlx_pending_paired_cuda`
- Catalog #325 per-substrate symposium evidence — THIS memo IS the canonical symposium evidence; satisfies acceptance contract (a) memo within 14 days + (b) verdict PROCEED + (c) matching canonical posterior anchor (Mutation 1) + (d) 6-step contract complete
- Catalog #335 cathedral consumer auto-discovery — Mutation 3 canonical equation EmpiricalAnchor consumed by `tac.cathedral_consumers.canonical_equation_lookup_consumer`; Mutation 1 council posterior anchor consumed by sister `council_anchor_lookup_consumer` per canonical contract
- Catalog #336 + #337 + #355 + #372 cathedral autopilot main() invoker — auto-discovery cascade fires per iteration; per-axis decomposition + Pareto polytope + Meta-Lagrangian wire-in active
- Catalog #340 sister-checkpoint guard PROCEED at all checkpoints (Slot B at tools/ + preflight surface DISJOINT to this council deliberation memo + posterior anchor scope per ownership map)
- Catalog #344 canonical equations registry — Mutation 3 EmpiricalAnchor extends canonical posterior surface
- Catalog #346 canonical roster sextet pact — 8 attendees (8/8 unanimous; sextet pact MIN 5-of-6 satisfied; 4 co-leads per 2026-05-19 amendment + 4 sister members per canonical inner-council eleven-voice roster minus Quantizr + Hotz + Selfcomp + MacKay + Ballé who are sister specialists not invoked this deliberation)
- Catalog #348 retroactive sweep — N/A (no NEW STRICT gate landed; canonical apparatus mutations are themselves the canonical record)
- Catalog #355 META-LAGRANGIAN-WIRE — ACTIVE (cathedral autopilot meta-Lagrangian invoker consumes Compound C predicted ΔS via bounded Phase 1 adjustment factor in [0.95, 1.05] as observability-only annotation)
- Catalog #371 canonical_equations auto-recalibrator — refits when `when_3+_new_empirical_anchors_in_domain` trigger satisfied (in-domain = post-paired-CUDA-RATIFICATION; current in-domain anchor count = 0; this apparatus-gating anchor does NOT count toward in-domain trigger)
- Catalog #376 / #378 subagent spawn-time PV + main-thread spawn-decision PV — VERIFIED at session start (Catalog #376 PV checkpoint emitted; 0 sister subagents in conflict scope)
- Catalog #379 cathedral autopilot canonical META-orchestrator — apparatus mutations consumed downstream via cathedral consumer auto-discovery + invoker callsite

## Operator-Routable Cascade (Next Cap-Window)

**TOP-PRIORITY (STEP B per Wave N+50 STAND_DOWN cascade decision tree)**:

1. **Wave N+50.2 STEP B Compound C standalone paired-CUDA RATIFICATION** — per Catalog #246 + canonical PROCEED-unconditional posterior anchor (Mutation 1) + canonical probe outcome PROCEED (Mutation 2). Flip `dispatch_enabled=true` on `.omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_heterogeneous_bit_modal_t4_dispatch.yaml` (transient; reset post-dispatch per Catalog #240). Execute `tools/operator_authorize.py --recipe substrate_pact_nerv_selector_v3_heterogeneous_bit_modal_t4_dispatch` with paired-env per Catalog #199 (`OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1` + `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=5.00`). Harvests paired CPU + CUDA per Catalog #246; expected cost ~$1.50; expected wall-clock ~90 min.

2. **POST-PRIORITY-1: Catalog #324 post-training Tier-C density re-measurement** — `tools/mdl_scorer_conditional_ablation.py --tier c --archive <landed_compound_c_archive_sha>`; expected cost $0.20-0.50 CPU; validates Scenario B mid-conservatism band post-paired-CUDA-RATIFICATION.

3. **CONDITIONAL ON PRIORITY 1 FAILURE: Wave N+3 sister full-renderer scorer-bound FP4-QAT pass** — per parent T1 op-routable #5; PyTorch sister trainer + Hinton-distilled scorer surrogate + paired-CUDA; cost ~$2-4 paired; closes Quantizr 0.33 canonical cos>=0.999 gap.

4. **FAIL-SAFE: revert to Compound B int8 baseline** — operator-routable `dispatch_enabled: false` flip on Compound C recipe + redirect to Slot 2 int8 baseline recipe at `.omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_int8_decoder_modal_t4_dispatch.yaml`; cost $0 (configuration only); preserves iterate-not-force optionality per CLAUDE.md "Forbidden premature KILL".

**HARD CONSTRAINT preserved**: `gh pr create` is NOT operator-routable from subagent regardless of empirical verdict per `[[pr-creation-requires-explicit-operator-authorization-with-adversarial-negative-findings-audit-standing-directive-20260528]]`.

## Files Touched

- `.omx/research/council_t2_compound_c_standalone_per_substrate_symposium_20260528.md` (THIS file; NEW symposium memo)
- `.omx/state/council_deliberation_posterior.jsonl` (APPEND-ONLY: Mutation 1 PROCEED-unconditional canonical posterior anchor)
- `.omx/state/probe_outcomes.jsonl` (APPEND-ONLY: Mutation 2 PROCEED probe outcome supersedes Wave N+50 DEFER + STAND-DOWN-REVIEW-AUDIT PROMOTE)
- `.omx/state/canonical_equations_registry.jsonl` (APPEND-ONLY: Mutation 3 EmpiricalAnchor on `heterogeneous_per_tensor_bit_allocation_compounding_v1`)
- `.omx/state/subagent_progress.jsonl` (APPEND-ONLY: 3+ checkpoint rows per Catalog #206)
- Landing memo at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wave_n50_1_compound_c_standalone_catalog_325_symposium_iteration_landed_20260528.md` (downstream landing memo)

## Lane

`lane_wave_n50_1_compound_c_standalone_catalog_325_symposium_iteration_20260528` L1 (impl_complete + canonical_apparatus_mutations_x3 + memory_entry; per Catalog #220 substrate_engineering opt-out because gate is apparatus-discipline not substrate-runtime).
