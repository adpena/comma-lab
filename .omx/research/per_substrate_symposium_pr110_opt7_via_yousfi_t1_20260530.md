# Per-substrate symposium: PR110-OPT-7 via Yousfi-T1 L1 PROMOTION

**council_tier:** T2
**council_attendees:** [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Rudin, Daubechies, MacKay, PR95Author]
**council_quorum_met:** true
**council_verdict:** PROCEED_WITH_REVISIONS
**council_dissent:**
  - member: Contrarian
    verbatim: "Phase C smoke runs on SYNTHETIC random-noise frames + synthetic vulnerability map; the actual Yousfi-T1 canonical 600-pair fp64 anchor (vulnerability_ratio=363.54) is used at canonical-anchor-mode but the Phase C smoke uses synthetic-mode for test determinism. Phase D paired-CUDA RATIFICATION REQUIRES use_canonical_pose_vulnerability_anchor=true + real upstream/videos/0.mkv frames; without that the predicted band is planning-only."
  - member: Assumption-Adversary
    verbatim: "The 5-helper composition is a BINDING-DEPTH claim per the PR-or-greater parity directive. The claim 'composing 5 canonical helpers produces sub-frontier score' is HARD-EARNED for each individual helper (each has empirical anchors) but CARGO-CULTED for the COMPOSITION (no paired-CUDA empirical anchor on the composed substrate yet; Phase D operator-routable). Therefore predicted_band_validation_status MUST be pending_post_training per Catalog #324."
council_assumption_adversary_verdict:
  - assumption: "PR110-OPT-7 L0 SCAFFOLD inverse-scorer basis produces score-relevant per-pair selectors"
    classification: HARD-EARNED
    rationale: "Wave N+34 canonical anchor unweighted aggregate ΔS=-0.001170 vs UNIWARD-weighted aggregate ΔS=-0.000910 at .omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json"
  - assumption: "Yousfi-T1 Deliverable A pose-vulnerability map vulnerability_ratio ~363x is canonical"
    classification: HARD-EARNED
    rationale: "Empirical anchor: build_default_pose_vulnerability_map_from_canonical_anchor() on canonical 600-pair fp64 tensor at .omx/state/master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy yields vulnerability_ratio=363.54"
  - assumption: "Yousfi-T1 Deliverable B PoseNet MAE-V surrogate (582 params) is numerically faithful to FastViT-T12 PoseNet on real video frames"
    classification: CARGO-CULTED
    rationale: "Surrogate is random-initialized at L1 PROMOTION; FORWARD-PASS parity vs real FastViT-T12 PoseNet has not been measured. The surrogate is PORTABLE (numpy-native) which IS the canonical contract for L1 + research-only; paired-CUDA RATIFICATION measures actual FastViT-T12 distortion delta."
  - assumption: "Yousfi-T1 Deliverable C YUV6 chroma-subsampled perturbation preserves luma EXACTLY in YUV6 space"
    classification: HARD-EARNED
    rationale: "Empirically verified: Phase C smoke produces luma_preservation_max_abs_drift_yuv6=0.0 per the canonical contract test. luma drift in YUV6 = 0 by construction (only U_sub, V_sub channels are modified)."
  - assumption: "alaska canonical Y0_UV color branch matches SegNet stride-2 stem 256x192 blind spot"
    classification: HARD-EARNED
    rationale: "Y0_UV = (Y0, U, V) at YUV6 half-resolution (192x256 grid) matches the EfficientNet-B2 stride-2 stem 256x192 blind spot per CLAUDE.md 'Exact scorer architectures' canonical mapping"
  - assumption: "5-helper composition produces SUPER_ADDITIVE per-byte savings vs PR110 fec6 baseline"
    classification: CARGO-CULTED
    rationale: "Composition score-impact is empirically unmeasured. Phase D paired-CUDA RATIFICATION is the canonical reactivation surface. Per Catalog #322 sister anti-pattern (autopilot adjustment derived from phantom composition_alpha): substrate composition_alpha can ONLY be claimed AFTER paired-CUDA anchor lands; predicted_band [-0.0030, 0.0010] is planning-only."
council_decisions_recorded:
  - "op-routable #1: Phase D operator-attended paired-CUDA RATIFICATION dispatch via .venv/bin/python tools/operator_authorize.py --recipe substrate_pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1_modal_t4_dispatch --estimated-cost-usd 0.30 (cost band ~$0.30; envelope per Catalog #246 paired CPU+CUDA)"
  - "op-routable #2: Phase E canonical equation registration via tac.canonical_equations.register_canonical_equation per Catalog #344 with sister anti-pattern + FORMALIZATION_PENDING per the empirical-anchor canonical pattern"
  - "op-routable #3: post-paired-CUDA Tier-C density measurement via tools/mdl_scorer_conditional_ablation.py --tier c on the post-training archive sha + replace predicted_band_validation_status pending_post_training with validated_post_training per Catalog #324"
  - "op-routable #4: IF empirical paired-CUDA score is within predicted band [-0.0030, 0.0010] → ratify canonical equation pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1_savings_v1; IF NOT → IMPLEMENTATION-LEVEL falsification per Catalog #307 + reactivation paths per Catalog #308"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""

---

## Symposium memo per Catalog #325 6-step contract

This memo satisfies the canonical 6-step per-substrate symposium contract for
the PR110-OPT-7 via Yousfi-T1 L1 PROMOTION substrate per CLAUDE.md
"PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"
non-negotiable + Catalog #325 STRICT preflight gate enforcement.

## Cargo-cult audit per assumption (Catalog #303)

Per the operator's binding 2026-05-30 standing directive on cargo-cult audit
discipline: every substrate design memo MUST enumerate each cargo-culted
assumption + the canonical unwind path.

### Cargo-cult #1: 5-helper composition produces SUPER_ADDITIVE per-byte savings

- **HARD-EARNED-vs-CARGO-CULTED:** CARGO-CULTED at L1 PROMOTION; HARD-EARNED
  for each individual helper.
- **Unwind path:** Phase D operator-attended paired-CUDA RATIFICATION dispatch
  per op-routable #1 measures the composition's actual score delta vs PR110
  fec6 baseline.
- **Reactivation criteria per Catalog #308:** IF paired-CUDA empirical paired
  CPU/CUDA shows ΔS within predicted band [-0.0030, 0.0010], composition is
  HARD-EARNED-EMPIRICALLY-VERIFIED + canonical equation ratifies. IF NOT,
  alternative compositions per the canonical anti-pattern unwind paths.

### Cargo-cult #2: PoseNet MAE-V surrogate (582 params) faithful to FastViT-T12

- **HARD-EARNED-vs-CARGO-CULTED:** CARGO-CULTED at L1 PROMOTION
  (random-initialized surrogate); HARD-EARNED-PROVENANCE (the surrogate IS
  canonical per Yousfi-T1 Deliverable B contract).
- **Unwind path:** N+50 trainer wave fits the surrogate weights to real
  FastViT-T12 PoseNet predictions on real upstream/videos/0.mkv frames per
  the canonical distillation pattern.
- **Reactivation criteria per Catalog #308:** IF distilled surrogate
  L1-distance < 0.1 vs real FastViT-T12, promote to HARD-EARNED. IF NOT,
  alternative surrogate architectures.

### Cargo-cult #3: alaska Y0_UV branch is canonical optimal vs other branches

- **HARD-EARNED-vs-CARGO-CULTED:** CARGO-CULTED at L1 PROMOTION (Y0_UV chosen
  per CLAUDE.md SegNet stride-2 stem blind-spot mapping but not empirically
  swept vs YUV6_full / UV_only / etc.).
- **Unwind path:** Sister sweep wave dispatches Y0_UV / YUV6_full / UV_only /
  Y_only / Y0 alone strategies in 5-arm sweep; canonical optimal per empirical
  paired-CUDA score.

### Cargo-cult #4: chroma_perturbation_magnitude=4.0 is canonical optimal

- **HARD-EARNED-vs-CARGO-CULTED:** CARGO-CULTED (canonical default; not swept).
- **Unwind path:** Magnitude sweep wave dispatches 1.0/2.0/4.0/8.0/16.0 in
  5-arm sweep; canonical optimal per empirical paired-CUDA score + visual
  distortion threshold.

### Cargo-cult #5: vulnerable_pair_budget=100 (of 600) is canonical optimal

- **HARD-EARNED-vs-CARGO-CULTED:** CARGO-CULTED (16.67% canonical ratio chosen
  for parsimony; not swept).
- **Unwind path:** Budget sweep wave dispatches 50/100/200/400/600 in 5-arm
  sweep; canonical optimal per empirical paired-CUDA score + rate cost
  intersection.

## 9-dimension success checklist evidence (Catalog #294)

### Dimension 1: UNIQUENESS

The substrate is BINDING-DEPTH UNIQUE per the PR-or-greater parity directive:
no sister substrate binds the canonical 5 primitives (alaska + Yousfi-T1 A+B+C
+ PR110-OPT-7) simultaneously. Sister L0 SCAFFOLD at
`tac.composition.pr110_opt_7_fridrich_uniward_inverse_scorer_basis` is the
inverse-scorer basis ALONE; THIS L1 PROMOTION is the canonical composition.

### Dimension 2: BEAUTY + ELEGANCE

PR-or-greater parity: substrate package = 4 files + 41/41 dedicated tests.
The substrate-engineering binding pattern follows the canonical Slot CCC
template (research_only=true; lane_class=substrate_engineering;
canonical Tier A markers per Catalog #341; canonical Provenance per #323).
NOT compressed to ≤350 LOC bolt-on budget per HNeRV parity L7 substrate-
engineering exception.

### Dimension 3: DISTINCTNESS

Substantively distinct per Slot EEE NO FAKE IMPLEMENTATIONS gate:
- Distinct color branches produce distinct alaska slices (test verified).
- Distinct rng_seeds produce distinct vulnerability maps (test verified).
- Distinct chroma strategies produce distinct perturbation summaries
  (test verified).

### Dimension 4: RIGOR

- Premise verification: Yousfi-T1 helpers verified per the canonical
  `compute_per_pair_pose_vulnerability_map` empirical anchor on the
  canonical 600-pair fp64 tensor (vulnerability_ratio=363.54).
- Adversarial review: this T2 symposium with 10-voice council
  (Shannon/Dykstra/Yousfi/Fridrich/Contrarian/Assumption-Adversary/
  Rudin/Daubechies/MacKay/PR95Author).
- Assumption classification: 5 cargo-culted assumptions enumerated above
  per Catalog #303 + 5 hard-earned-vs-cargo-culted classifications per
  the addendum.
- Empirical anchor: Phase C MLX-LOCAL N=100 smoke GREEN per 7/7
  substantive axes.

### Dimension 5: OPTIMIZATION PER TECHNIQUE

Canonical-vs-unique decision per layer (Catalog #290 sister):

- alaska color separation: ADOPT canonical (sister of canonical
  ColorBranchSliceStrategy.Y0_UV; serves PR-or-greater parity per L1).
- Yousfi-T1 Deliverable A pose-vulnerability map: ADOPT canonical
  (canonical 600-pair fp64 anchor IS the canonical pair-selection
  prior).
- Yousfi-T1 Deliverable B PoseNet MAE-V surrogate: ADOPT canonical
  (582 params numpy-portable IS the canonical PoseNet surrogate).
- Yousfi-T1 Deliverable C YUV6 chroma perturbation: ADOPT canonical
  (luma preservation EXACT 0.0 in YUV6 IS the canonical contract).
- PR110-OPT-7 inverse-scorer basis L0 SCAFFOLD: ADOPT canonical
  (4-strategy enum IS the canonical inverse-scorer basis).

### Dimension 6: STACK-OF-STACKS-COMPOSABILITY

The substrate composes 5 canonical helpers per the PR-or-greater parity
binding-depth discipline. Each helper carries canonical Tier A markers per
Catalog #341 so the composition IS bounded by per-helper non-promotability.
Composition surface = pair-selection intersection (vulnerable + uniward).

### Dimension 7: DETERMINISTIC REPRODUCIBILITY

`rng_seed=42` canonical default; numpy / mlx seeds threaded through.
Synthetic-vulnerability path is deterministic per seed; canonical-anchor
path is deterministic per the canonical 600-pair fp64 tensor sha256.

### Dimension 8: EXTREME OPTIMIZATION + PERFORMANCE

MLX-first per CLAUDE.md "MLX portable-local-substrate authority" + operator
standing directive 2026-05-30. $0 macOS-MLX advisory training + paired-CUDA
RATIFICATION at ~$0.30 envelope per Catalog #246. Tier 1+2+3 engineering
hygiene per Catalog #270 (MLX-canonical primitives; PyTorch fp16/tf32 N/A).

### Dimension 9: OPTIMAL MINIMAL CONTEST SCORE

Predicted band [-0.0030, 0.0010] per the 5-helper composition Dykstra
feasibility check per Catalog #296. Paired-CUDA RATIFICATION required for
score-claim authority per Catalog #246.

## Observability surface declaration (Catalog #305)

6-facet canonical observability per CLAUDE.md "Max observability — non-
negotiable":

1. **Inspectable per layer:** Each of the 5 helpers exposes its own summary
   in `PR110OPT7ViaYousfiT1Result.{pose_vulnerability_summary,
   alaska_color_slice, inverse_scorer_basis_summary,
   chroma_perturbation_summary, posenet_surrogate_summary}`.

2. **Decomposable per signal:** Per-helper invocation receipts in
   `canonical_helpers_invoked` allow downstream consumers to verify each
   helper was invoked + its per-helper output drift.

3. **Diff-able across runs:** rng_seed determinism + canonical Provenance
   inputs_sha256 enable run-to-run diff at the helper-invocation level.

4. **Queryable post-hoc:** training_stats.json artifact + per-pair selector
   indices + cross-reference matrix all queryable via standard JSON tooling.

5. **Cite-able:** Cross-reference matrix maps EACH helper to its source
   commit (alaska=61a91a48e / Yousfi-T1=3d027ecf9 / PR110-OPT-7=3fd28b5b2)
   + canonical_helper_invocation field in Provenance per Catalog #323.

6. **Counterfactual-able:** Each helper exposes a config field so the
   substrate's composition can be ablated by varying ONE config field at a
   time (single-axis ablation surface per the canonical sister test pattern).

## Sextet pact deliberation (canonical 6-of-6 + grand council)

- **Shannon LEAD:** PROCEED. The substrate's R(D) framing is canonical: 5
  helpers each contribute to per-pair selector cost minimization; canonical
  rate term = 25 * archive_bytes / 37545489. The substrate's distinguishing
  contribution is the canonical Fridrich UNIWARD inverse-scorer weighting
  on the alaska Y0_UV color branch with Yousfi-T1 pose-axis priors.
- **Dykstra CO-LEAD:** PROCEED_WITH_REVISIONS. The 5-helper composition
  requires Dykstra alternating-projections feasibility check on the joint
  polytope (per Catalog #296). The predicted band [-0.0030, 0.0010] is
  derivative; Phase D paired-CUDA RATIFICATION provides the canonical
  empirical anchor.
- **Yousfi:** PROCEED. The Yousfi-T1 Deliverables A+B+C are canonical;
  binding into ONE substrate per the canonical inverse-steganalysis
  framework is the canonical compounding pattern.
- **Fridrich:** PROCEED. UNIWARD canonical universal embedding cost per
  Holub-Fridrich-Denemark 2014; canonical inverse-scorer basis 4-strategy
  enum per PR110-OPT-7 L0 SCAFFOLD; canonical adoption.
- **Contrarian:** PROCEED_WITH_REVISIONS (verbatim above).
- **Assumption-Adversary:** PROCEED_WITH_REVISIONS (verbatim above; 5
  cargo-culted assumptions enumerated per Catalog #303).
- **Rudin:** PROCEED. The substrate's canonical contract is interpretable:
  each of the 5 helpers exposes its summary + the composition surface
  (vulnerable + uniward intersection) is auditable per Catalog #305.
- **Daubechies:** PROCEED. The alaska Y0_UV canonical branch IS the
  canonical multi-scale partition prior for the substrate's YUV6
  representation.
- **MacKay (memorial):** PROCEED. The substrate's canonical Provenance
  per Catalog #323 + canonical equations registry per Catalog #344
  ensures the information-theoretic framing is queryable post-hoc.
- **PR95Author:** PROCEED. The substrate-engineering binding-depth
  discipline per PR-or-greater parity is canonical; the substrate binds
  5 canonical primitives deeply per the canonical PR95-family pattern.

10-voice consensus: PROCEED_WITH_REVISIONS per Contrarian + Assumption-
Adversary binding revisions encoded in the recipe's
`predicted_band_validation_status: pending_post_training` per Catalog #324.

## Per-substrate reactivation criteria pinned (Catalog #325 step 5)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this
substrate has 4 canonical reactivation paths per Catalog #308 if Phase D
paired-CUDA RATIFICATION returns negative:

1. **Alternative Fridrich weighting basis:** swap canonical JOINT linear
   combination for UNIWARD_INVERSE_SEGNET_GRADIENT_SENSITIVITY OR
   UNIWARD_INVERSE_POSENET_GRADIENT_SENSITIVITY single-strategy variants;
   re-fire paired-CUDA RATIFICATION (cost ~$0.30 per arm).

2. **Alternative alaska color branch:** swap canonical Y0_UV for YUV6_full
   OR UV_only OR Y_only branches; re-fire paired-CUDA RATIFICATION.

3. **Alternative Yousfi-T1 chroma perturbation strategy:** swap canonical
   JOINT_ATICK_REDLICH for LOCAL_VARIANCE_WEIGHTED OR
   SEGNET_GRADIENT_WEIGHTED OR POSENET_GRADIENT_WEIGHTED_VIA_MAE_V;
   re-fire paired-CUDA RATIFICATION.

4. **Alternative PoseNet MAE-V surrogate initialization:** distill the
   surrogate weights to match real FastViT-T12 PoseNet via N+50 trainer
   wave per the canonical distillation pattern; re-fire paired-CUDA
   RATIFICATION.

Each alternative is a SISTER-EXTINCTION ARCHITECTURE per Catalog #299
gate consolidation discipline + the canonical Catalog #313 probe-outcomes
ledger pattern.

## Catalog #324 post-training Tier-C validation discipline declared

`predicted_band_validation_status: pending_post_training` per the canonical
recipe declaration. Post paired-CUDA RATIFICATION lands, Tier-C density
measurement via `tools/mdl_scorer_conditional_ablation.py --tier c` on the
post-training archive sha replaces `pending_post_training` with
`validated_post_training` + artifact path.

## Cross-references

- Phase A substrate package: `src/tac/substrates/pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1/`
- Phase B trainer: `experiments/train_substrate_pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1.py`
- Phase C MLX-LOCAL smoke: `experiments/results/pr110_opt7_via_yousfi_t1_l1_promotion_smoke_20260530T205259Z/`
- Phase D recipe: `.omx/operator_authorize_recipes/substrate_pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1_modal_t4_dispatch.yaml`
- L0 SCAFFOLD landing: `feedback_pr110_opt7_fridrich_uniward_inverse_scorer_basis_l0_scaffold_landed_20260530.md`
- Yousfi-T1 landing: `feedback_yousfi_cascade_tier_1_pose_axis_canonical_helpers_a_plus_b_plus_c_landed_20260530.md`
- alaska landing: `feedback_alaska_yousfi_canonical_pattern_extraction_landed_20260530.md`
- Deferred-items feeder audit: `feedback_deferred_items_feeder_audit_post_alaska_m9v3_yousfi_tier_1_wave_landed_20260530.md`
- L1 PROMOTION landing memo: `feedback_pr110_opt7_l1_promotion_via_yousfi_t1_landed_20260530.md`
- Retroactive sweep per Catalog #348: `.omx/research/retroactive_sweep_for_pr110_opt7_l1_promotion_via_yousfi_t1_20260530.md`
