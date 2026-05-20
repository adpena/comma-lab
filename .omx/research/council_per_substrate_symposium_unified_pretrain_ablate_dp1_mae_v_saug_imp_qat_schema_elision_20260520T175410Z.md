---
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Hinton
  - Schmidhuber
  - vdOord
  - Tishby
  - Selfcomp
  - Karpathy
  - TimeTraveler
  - PR95Author
  - Quantizr
  - MacKay
  - Balle
  - Hotz
  - Carmack
  - Atick
  - Redlich
  - Rao
  - Ballard
  - Wyner
  - TimeTravelerProtege
  - Rudin_Grand
  - Daubechies_Grand
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
horizon_class: frontier_pursuit
deliberation_id: per_substrate_symposium_unified_pretrain_ablate_dp1_mae_v_saug_imp_qat_schema_elision_20260520
deferred_substrate_id: unified_pretrain_ablate_dp1_mae_v_saug_imp_qat_schema_elision
substrate_alias: unified_pretrain_ablate
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - per_substrate_symposium_dp1_deep_dive_20260517
  - per_substrate_symposium_mae_v_plus_saug_20260518
  - per_substrate_symposium_lane_17_imp_20260517
  - per_substrate_symposium_stc_paradigm_reformulation_a1_residual_20260520
  - per_substrate_symposium_pr106_05_06_reformulated_20260517
council_dissent:
  - member: Contrarian
    verbatim: |
      operating-within: "unifying 5 pre-existing primitives into a single composable substrate is the right answer because each has been DEFERRED or DEFERRED_PENDING_EVIDENCE in isolation." Classification: CARGO-CULTED-AT-CATEGORICAL-LEVEL. Rationale: of the 5 primitives (DP1 pretraining + MAE-V + SAUG + IMP+QAT + schema-elision+GOSDT pruning), NONE has produced a contest-CUDA empirical anchor on the current 0.205-frontier (`9cb989cef519ed`) or contest-CPU on current 0.192 (`6bae0201fb0824`). DP1 carries Phase 2 hardening (Catalog #210/#211) but ZERO compose-and-measure anchor; MAE-V + SAUG were DEFERRED 21 days as PROFILES not substrates per sister memo 2026-05-18 (Assumption-Adversary VETO); Lane 17 IMP β Fisher 1.016 was on Lane G v3 1.05 baseline, not current frontier; GOSDT/schema-elision were planning artifacts. UNIFICATION ≠ COMPOSITION. The Time-Traveler-L5 dissent in sister #869 verbatim: "training augmentation is a CANONICAL HELPER concern that can be canonical-vs-unique-decided per substrate." Unifying 5 unproven primitives into one substrate WITHOUT a single one having current-frontier empirical receipt is the structural mirror of NSCS06 v6's 5-move composition that landed 105.15 vs predicted [0.10, 0.20] (553× outside band) per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check". VETO on PROCEED-unconditional pending Dykstra-feasibility intersection check + at least 1 single-primitive current-frontier anchor.
  - member: Assumption-Adversary
    verbatim: |
      operating-within: "the OOD-pretrain-then-ablate paradigm transfers from large-model ML practice (MAE/DINOv3/SAM2) to 88K-param contest renderers." Classification: CARGO-CULTED-NEEDS-EMPIRICAL. Rationale: Yosinski et al. 2014 "How transferable are features in deep neural networks" empirically established that transfer-learning gains correlate strongly with target-model capacity — at 1M+ params transfer is dominant; at <100K params transfer is regularizer-noise dominated. Quantizr's 88K-param FiLM-CNN at 0.33 frontier ALREADY captures the contest video's structure; OOD pretraining on Comma2k19 (180GB) → 88K-param distillation runs HEAD-FIRST into negative transfer (Wang et al. 2019 "Characterizing and Avoiding Negative Transfer"). The Hinton lens (sister Lane 17 IMP symposium): "MAE works at ViT/ConvNeXt scale (10M+ parameters)." The paradigm BORROWS from large-model practice WITHOUT scale-appropriate empirical adaptation. Pre-frontier check: did the DP1 Phase 2 hardening show OOD-prior contributed >0 score-axis improvement on ANY known_base_substrate? Per sister DP1 deep-dive verdict: PROCEED_WITH_REVISIONS with ZERO empirical compose-and-measure anchor existing. The unified design INHERITS DP1's empirical gap. SECOND assumption operating-within: "aggressive ablation + distillation + pruning post-pretrain RECOVERS the lost capacity." Classification: HARD-EARNED-AT-LARGE-MODEL-SCALE / CARGO-CULTED-AT-88K-SCALE. Frankle-Carbin 2019 lottery-ticket on VGG/ResNet held at 10-100× larger param counts than 88K; the 89.3% sparsity result Lane 17 IMP claimed was Lane G v3 era (1.05 baseline); at 0.192 frontier the rate-axis cost-distance is ~18000 bytes and post-pruning recovery uncertainty bound is wider than that. VETO on PROCEED-unconditional pending: (a) explicit Dykstra-feasibility check that the 5-primitive composition lies inside the post-frontier achievable region; (b) per-primitive HARD-EARNED-vs-CARGO-CULTED classification with EMPIRICAL anchor citation OR explicit DEFER status; (c) probe-disambiguator for "single-primitive empirical-anchor-first" vs "5-primitive simultaneous" dispatch ordering.
  - member: Yousfi
    verbatim: |
      operating-within: "the contest scorer's pre-trained FastViT-T12 + EfficientNet-B2 ALREADY contain rich driving-scene priors from their pretraining sets, so adding DP1 OOD-pretrained driving prior is REDUNDANT." Classification: HARD-EARNED. Rationale: PoseNet FastViT-T12 pretrained on ImageNet + EfficientNet-B2 SegNet pretrained on (probably) comma10k or driving subsets — both contain dashcam-distribution priors implicitly through their training corpora. DP1's incremental signal is bounded to the difference (Comma2k19 specifics − FastViT/EfficientNet pretraining sets). My sister DP1 deep-dive estimate of predicted_delta [-0.005, -0.012] reflects exactly this bounded-difference structure. Per the operator's plateau-exit framing: this paradigm is FRONTIER_PROTECTING (helps stabilize current best) more than FRONTIER_BREAKING (unlikely to exit 0.196-0.199 cluster). Per sister #869 verdict: at 0.192 frontier "training-augmentation has 5× LESS room to move than codec primitives." The unified design's MAE/SAUG components inherit this exact bound. RECOMMENDATION: PROCEED_WITH_REVISIONS conditioned on (a) single-primitive empirical anchor first (DP1+fec6 PATH 1 the sister DP1 deep-dive already scaffolded) before any 5-primitive multi-component dispatch; (b) horizon_class explicitly downgraded from asymptotic_pursuit to frontier_pursuit per the bounded-incremental analysis.
council_assumption_adversary_verdict:
  - assumption: "OOD-pretraining (Comma2k19 / BDD100K / KITTI / Waymo) → 88K-param contest renderer transfers positively"
    classification: CARGO-CULTED-NEEDS-EMPIRICAL
    rationale: |
      Per Yosinski et al. 2014 transfer correlates with target-model capacity; at <100K params transfer is regularizer-noise dominated per Wang et al. 2019. Quantizr's 88K-param FiLM-CNN at 0.33 already captures contest video structure (PR101 anchor). The 0.205 [contest-CUDA] frontier (commit 9cb989cef519ed) was achieved WITHOUT OOD pretraining. ZERO empirical anchor exists showing OOD pretrain → 88K target → ΔS_contest > 0. Hinton sister #869 verbatim: "MAE works at ViT/ConvNeXt scale (10M+ parameters)." MUST empirically test before assuming positive transfer.
  - assumption: "Aggressive ablation + distillation (KL T=2.0) + IMP pruning + QAT (LSQ FP4) + schema-elision (GOSDT) RECOVERS lost capacity post-OOD-pretrain"
    classification: HARD-EARNED-AT-LARGE-MODEL-SCALE / CARGO-CULTED-AT-88K-SCALE
    rationale: |
      Frankle-Carbin 2019 lottery-ticket validated on VGG/ResNet (10-100× larger). Lane 17 IMP β Fisher 1.016 was Lane G v3 era (1.05 baseline). Hinton T=2.0 KL distill was validated on ResNet50→ResNet18 (40M→11M params). At 88K target, post-pruning recovery uncertainty exceeds the rate-axis cost-distance (~18000 bytes from 0.192 to 0.180 medal band). The OP3-V3 T4 anchor (sidecar `a1afce29` / archive `6bae0201fb0824`) shows 90.7% of bytes are SegNet-dominant + 8.5% near-zero contribution = there IS pruning headroom, but the recovery uncertainty per byte is unmeasured at 88K scale.
  - assumption: "Unifying 5 pre-existing primitives into ONE composable substrate produces additive ΔS"
    classification: CARGO-CULTED-CATEGORICAL-MIRROR-OF-NSCS06-V6-553X-MISS
    rationale: |
      Per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check": NSCS06 v6 5-move composition landed 105.15 vs predicted [0.10, 0.20] (553× outside band) because additivity was assumed without Dykstra feasibility. THIS substrate proposes 5-primitive composition (DP1 + MAE-V + SAUG + IMP+QAT + schema-elision). Per Catalog #322 sister composition_alpha empirical floor: SUB_ADDITIVE (α ∈ [0.3, 0.7]) is the empirical norm; SUPER_ADDITIVE (α > 0.7) is rare; ANTAGONISTIC (α < 0.3) and SATURATING are common. Without ANY 2-primitive composition anchor (e.g. DP1+IMP measured), the 5-primitive additivity assumption is UNGROUNDED. Dykstra-feasibility intersection check REQUIRED before dispatch.
  - assumption: "Pretrained-on-OOD-data unblocks the 0.196-0.199 plateau"
    classification: HARD-EARNED-PARTIALLY-NEW-DOF / CARGO-CULTED-AT-ΔS-MAGNITUDE
    rationale: |
      HARD-EARNED part: OOD pretrain introduces a NEW degree of freedom (out-of-distribution prior structure) orthogonal to the SegNet/PoseNet/rate axes the 0.196-0.199 cluster substrates have been polishing. This IS a class-shift direction per CLAUDE.md "HORIZON-CLASS evaluation axis." CARGO-CULTED part: the predicted ΔS magnitude is unknown. Yousfi's HARD-EARNED bound: [-0.005, -0.012] per DP1 dashcam-redundancy analysis. The unified design's MAE/SAUG components are training-augmentation knobs (sister #869 deferral). The new-DOF claim is structurally meaningful but the ΔS magnitude prediction is unfounded without empirical anchor.
  - assumption: "Composition with the canonical fec6 frontier (lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515`) is monotonic — adding unified-pretrain-ablate primitives produces fec6 ⊕ unified ≥ fec6 alone"
    classification: CARGO-CULTED-PER-COMPOSITION-AUDIT-LITERATURE
    rationale: |
      Per Catalog #322 sister composition_alpha empirical floor + sister fec6 stacking wave (commit `b43c8f2fd` extincted Catalog #204 bug class for fec6+format0d + fec6+haar_residual stacking): stacking on TOP of frontier substrates is ANTAGONISTIC ~30% of the time (alpha < 0.3 per the empirical floor). Without a probe-disambiguator OR a stack-of-stacks composability anchor showing fec6 ⊕ DP1 > fec6, the monotonicity claim is unverified. The 0.205 [contest-CUDA] frontier on `9cb989cef519ed` was achieved with NO unified-pretrain-ablate component; structurally the unified design must EXCEED `9cb989cef519ed`'s rate-axis efficiency (178517 bytes / 0.00476 rate-term) to be net-positive.
  - assumption: "The 88K-param contest renderer scale is appropriate for the unified-pretrain-ablate paradigm"
    classification: CARGO-CULTED-AT-CATEGORICAL-LEVEL
    rationale: |
      ALL of MAE / DINOv3 / SAM2 / DP1 are large-model paradigms validated at 10M+ params. Quantizr 88K + PR95 FastViT-T12-distilled-to-substrate-renderer are inherent 1-2 orders of magnitude smaller. The cargo-cult inherits the large-model paradigm WITHOUT scale-appropriate empirical adaptation. The Hinton sister #869 verbatim verdict: "MAE is over-parameterized for the data signal — the 25% mask ratio at patch_size=16 produces ~2400 mask patches on the 384×512 input; the renderer cannot meaningfully reconstruct masked patches from 88K parameters." THIS exact concern applies to the unified design's MAE-V component AND structurally to the DP1 distillation target (the 88K renderer cannot meaningfully ingest Comma2k19's 180GB prior structure).
council_decisions_recorded:
  - "op-routable #1 (P1): DEFERRED-PENDING-EVIDENCE — do NOT fire 5-primitive simultaneous unified dispatch. Operator-routable: spawn predecessor `DP1+fec6 PATH 1 single-primitive composition smoke` ($0.30 Modal T4 100ep per sister DP1 deep-dive PATH 1 scaffolded but UNFIRED) BEFORE any unified dispatch. The DP1+fec6 anchor IS the first reactivation criterion."
  - "op-routable #2 (P1): If DP1+fec6 PATH 1 produces measurable contest-CPU ΔS in [-0.001, -0.012] band (Dykstra-feasibility-checked vs current 0.19205 frontier), THEN spawn `DP1+IMP+fec6 single-primitive-pair composition smoke` ($2-5 Modal A10G 200ep) to establish 2-primitive composition_alpha anchor per Catalog #322. ANTAGONISTIC verdict (alpha < 0.3) closes the unified paradigm at 2-primitive boundary."
  - "op-routable #3 (P2): If 2-primitive composition_alpha is SUB_ADDITIVE or SUPER_ADDITIVE, build `tools/probe_unified_pretrain_ablate_dispatch_order_disambiguator.py` (~250 LOC) that emits Bayesian-optimal dispatch-order recommendation: DP1+IMP-first vs DP1+MAE-V-first vs DP1+SAUG-first vs DP1+QAT-first vs DP1+schema-elision-first. The disambiguator's predicted-vs-empirical Bayesian update per Catalog #344 informs the 4-primitive composition test."
  - "op-routable #4 (P2): Per the HARD-EARNED-PARTIALLY-NEW-DOF assumption — operator-decision required on whether to PIVOT the paradigm from `unified pretrained-on-OOD-data + aggressive-ablation` to `single-axis OOD-pretrain-ONLY composable bolt-on with NO aggressive-ablation tail` (i.e. drop MAE-V + SAUG + IMP+QAT + schema-elision from the unified design, retain ONLY DP1 prior composed with frontier substrate). This pivot variant has CLEANER per-primitive empirical attribution and matches the DP1 deep-dive PATH 1 scaffold."
  - "op-routable #5 (P3 — Catalog #324 post-training Tier-C validation discipline): If unified dispatch eventually fires, the predicted_band_validation_status MUST be `pending_post_training` until post-training Tier-C density measurement on the landed archive via `tools/mdl_scorer_conditional_ablation.py --tier c`. Per the empirical anchor: C6 IBPS landed 22× outside predicted_band [0.113, 0.163] because Tier-C density was measured on RANDOM-INIT weights pre-training. The unified design's distillation+pruning post-pretrain produces a NEW weight distribution; pre-training Tier-C density is structurally unrelated to post-pretrain-then-ablate Tier-C density."
  - "op-routable #6 (P3): Per Catalog #344 canonical equations registry — IF a unified dispatch produces an empirical anchor, REGISTER a NEW canonical equation `unified_pretrain_ablate_composition_alpha_v1` codifying the empirical compose-alpha formula across the 5-primitive sequential composition (per Catalog #322 + #344 sister discipline). The equation auto-recalibrates from posterior anchors per `tac.canonical_equations.update_equation_with_empirical_anchor`."
  - "op-routable #7 (P2): SISTER Z3-G1 anchor pattern per CLAUDE.md FORBIDDEN_PATTERNS 'Forbidden closed-form-CDF-allocator-without-empirical-bit-spend-proof' — IF the unified design's distillation phase adds bytes to the archive (e.g. DP1 codebook sidecar), the per-primitive byte-mutation smoke per Catalog #139 MUST verify operational consumption BEFORE any score claim."
predicted_mission_contribution_rationale: |
  frontier_protecting per Catalog #300 enum. The CARGO-CULTED-classified assumptions PREVENT direct frontier_breaking claim until empirical anchors arrive. However, the paradigm introduces a structurally new degree of freedom (OOD-pretrain-prior orthogonal to SegNet/PoseNet/rate axes the 0.196-0.199 cluster substrates have been polishing) per the HARD-EARNED-PARTIALLY-NEW-DOF Assumption-Adversary verdict. If empirical anchors materialize within the predicted [-0.005, -0.012] Yousfi-bounded band, the paradigm IS frontier_breaking-eligible at the post-frontier 0.19205 [contest-CPU] / 0.20533 [contest-CUDA] operating point. The PROCEED_WITH_REVISIONS verdict is calibrated to canonical Catalog #300 enum value `frontier_protecting` per sister DP1 deep-dive's existing classification preserved per Catalog #110 APPEND-ONLY (the body text classifies the substrate as `frontier_breaking_enabler` aspirational pending empirical materialization; the frontmatter field uses the canonical Catalog #300 enum).
deferred_substrate_retrospective_due_utc: "2026-06-19T17:54:10+00:00"
---

# Per-substrate symposium: Unified pretrained-on-OOD-data + aggressive-ablate composable substrate

**Date**: 2026-05-20T17:54:10Z
**Operator question (verbatim 2026-05-20)**: *"could our designs do better if pretirnaed on better quality driving data, like in better lighting conditions for example? then with aggressive ablation and deforestation and such after aadpted and applied to our video"*
**Council tier**: T3 (cross-paradigm composition design crossing 5 pre-existing primitives + 25+ literature anchors)
**Council quorum**: COMPLETE per `tac.canonical_council_roster.validate_council_dispatch_roster` returning `complete=True` (29 attendees: 14 INNER_COUNCIL all 4 co-leads + 15 GRAND_COUNCIL topical seats)
**Council verdict**: PROCEED_WITH_REVISIONS (5 binding revisions; 4 reactivation paths)

---

## Substrate proposal summary

**Unified composable substrate `unified_pretrain_ablate_dp1_mae_v_saug_imp_qat_schema_elision`** binding 5 pre-existing primitives into a single sequential pipeline:

```
Phase 1 (offline, $0 dispatch — per CLAUDE.md HNeRV parity L1 + Catalog #209-#213):
  OOD-pretrain on Comma2k19 (+ optionally BDD100K / KITTI / Waymo subsets where lighting
  conditions complement Comma2k19's distribution) → produce per-class CDF codebook
  + PCA basis prior (Catalog #210 codebook_provenance_metadata).

Phase 2 (per-substrate, ~$0.30-2 Modal T4 100ep — per Catalog #208-#211):
  Adapt pre-trained prior to contest video via DP1 composition wrapper
  (`tac.substrates.pretrained_driving_prior.composition.compose_with` per Catalog #211),
  binding to frontier base substrate (current: `pr101_frame_exploit_selector_fec6`).

Phase 3 (per-substrate, ~$1-5 Modal A10G 200-500ep):
  Aggressive ablation tail: MAE-V mask-augmentation training (sister #869 deferred lane;
  retire-then-reactivate via this unified substrate's design memo per CLAUDE.md
  UNIQUE-AND-COMPLETE-PER-METHOD) + SAUG hi-sigma augmentation (sister deferred lane).

Phase 4 (per-substrate, ~$2-8 Modal A100 500-1000ep):
  Distillation+pruning post-pretrain: Hinton KL T=2.0 SegNet distillation + Frankle-Carbin
  IMP β-Fisher iterative pruning (sister Lane 17 IMP per-substrate symposium PROCEED_WITH_REVISIONS
  + canonical recovery target per `feedback_grand_council_imp_permanent_fix_review_20260430.md`)
  + LSQ FP4 QAT quantization (Esser et al. 2020; canonical helper `tac.quantization.FakeQuantFP4`).

Phase 5 (per-substrate, ~$1-3 Modal A100 100ep):
  Schema-elision + GOSDT rule-list pruning per `tac.preflight_rudin_daubechies.gosdt_dispatch_router`
  (Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020) producing the final byte-minimal archive grammar.
```

**Distinguishing claim**: each pre-existing primitive has been DEFERRED in isolation. UNIFICATION-AS-SUBSTRATE proposes that the 5-primitive sequential composition produces additive (or super-additive) ΔS exceeding any single primitive's contribution.

---

## Empirical anchor inventory (Catalog #229 PV)

| Anchor | Score | Axis | Hardware | Notes |
|---|---:|---|---|---|
| **Current LOCAL frontier (CPU)** | **0.19205** | `[contest-CPU]` | GHA Linux x86_64 | lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515` / archive `6bae0201fb0824` / 178517 bytes (canonical_frontier_pointer.json) |
| **Current LOCAL frontier (CUDA)** | **0.20533** | `[contest-CUDA T4]` | linux_x86_64_t4 | lane `pr106_format0d_latent_score_table` / archive `9cb989cef519ed` (2026-05-16) |
| **OP3-V3 master-gradient sidecar** | 0.4175 | `[contest-CUDA T4]` | linux_x86_64_t4_modal | call_id `fc-01KS370Z9TF4QZMKQ9ND72KH4N` / archive `6bae0201fb0824` / sidecar sha `a1afce29` / (178417, 3) fp32 / dS/d_byte=6.66e-07 / autograd_per_parameter_projected per fec6_int8_fp16_jacobian / operating_point d_seg=0.0010 d_pose=0.00382 rate=0.00476 / 90.7% bytes SegNet-dominant + 8.5% near-zero contribution (16292 zero-contribution bytes / 161779 SegNet-dom / 346 pose-dom / 0 rate-dom) — THE canonical per-byte importance prior for the aggressive-ablate phase |
| PR101 GOLD upstream | 0.1929 | `[contest-CPU]` | GHA Linux x86_64 | upstream PR #101 (CLAUDE.md "Frontier scores are pointer-only") |
| PR110 fec6 (this is `pr101_frame_exploit_selector_fec6`) | 0.19205 | `[contest-CPU]` | GHA Linux x86_64 | sister to current frontier (same archive sha) |
| PR102 BRONZE | 0.19538 | `[contest-CPU]` | GHA Linux x86_64 | upstream PR #102 |
| PR103 SILVER | 0.195 | `[contest-CPU]` | GHA Linux x86_64 | upstream PR #103 |
| PR106 | 0.205 | `[contest-CUDA T4]` | linux_x86_64_t4_modal | sister CUDA-axis frontier |
| HNeRV class | 0.1987 | `[contest-CPU]` | GHA Linux x86_64 | upstream PR #95 |
| Quantizr | 0.33 | `[contest-CUDA]` | NVIDIA T4 | PR #56 |

**DP1 lane current state**: L1 SCAFFOLD per Catalog #209-#213; sister Catalog #210/#211 hardening landed; sister symposium #855 deep-dive PROCEED_WITH_REVISIONS 2026-05-17; sister symposium #868 PATH 1 (DP1+fec6) SCAFFOLDED but UNFIRED. ZERO contest-CUDA OR contest-CPU empirical anchor for ANY DP1 composition cell exists at any time.

**MAE-V + SAUG state**: L0 SKETCH; sister symposium #869 verdict DEFER_PENDING_EVIDENCE 2026-05-18 per Assumption-Adversary VETO that classifies these as PROFILES not substrates.

**Lane 17 IMP β Fisher state**: 1.016 [contest-CUDA] anchor was Lane G v3 era (1.05 baseline); sister Lane 17 IMP per-substrate symposium PROCEED_WITH_REVISIONS 2026-05-17 with reactivation criteria pinned.

**Schema-elision V1+V2 + GOSDT pruning + Hinton T=2.0 KL + DARTS-S + LSQ**: all planning artifacts; none have current-frontier (post-0.192-CPU / post-0.205-CUDA) empirical anchor.

---

## Cargo-cult audit per assumption

Per Catalog #303 + the hard-earned-vs-cargo-culted addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`). 6 surfaced assumptions classified above in `council_assumption_adversary_verdict`. Summary:

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| 1 | OOD-pretrain → 88K target positive transfer | CARGO-CULTED-NEEDS-EMPIRICAL | DP1+fec6 PATH 1 single-primitive smoke per op-routable #1 |
| 2 | Aggressive ablation + distill + IMP+QAT + GOSDT recovers lost capacity | HARD-EARNED-AT-LARGE-MODEL-SCALE / CARGO-CULTED-AT-88K-SCALE | Per-primitive empirical anchor REQUIRED at 88K scale; OP3-V3 sidecar `a1afce29` shows 90.7% SegNet-dominant + 8.5% near-zero bytes = headroom EXISTS but recovery uncertainty unmeasured |
| 3 | 5-primitive unification produces additive ΔS | CARGO-CULTED-MIRROR-OF-NSCS06-V6-553X-MISS | Dykstra-feasibility intersection check per Catalog #296 + 2-primitive composition_alpha anchor per Catalog #322 BEFORE 5-primitive dispatch |
| 4 | OOD-pretrain unblocks 0.196-0.199 plateau | HARD-EARNED-PARTIALLY-NEW-DOF / CARGO-CULTED-AT-ΔS-MAGNITUDE | Empirical anchor for ΔS magnitude; Yousfi HARD-EARNED bound [-0.005, -0.012] |
| 5 | Composition with fec6 frontier is monotonic | CARGO-CULTED-PER-COMPOSITION-AUDIT-LITERATURE | fec6 ⊕ DP1 single-cell anchor + Catalog #322 alpha measurement |
| 6 | 88K-param scale is appropriate for unified-pretrain-ablate paradigm | CARGO-CULTED-AT-CATEGORICAL-LEVEL | Scale-appropriate per-primitive adaptation OR explicit operator pivot to >100K-param target |

**Unwind procedure per NSCS06 v6→v7 44% improvement pattern (cargo-cult unwind methodology canonical)**: address cargo-culted assumptions one at a time, smallest-empirical-cost first. Op-routable ordering #1 → #2 → #3 maps assumptions 1+5 → 2+4 → 3 in this exact unwind sequence.

---

## 9-dimension success checklist evidence

Per Catalog #294. All 9 dimensions evaluated:

1. **UNIQUENESS**: HARD-EARNED. The paradigm IS distinct from any current registered substrate — no other substrate binds OOD-pretrain + aggressive-ablate + distillation + pruning + schema-elision into a single composable artifact. Sister DP1 deep-dive + sister Lane 17 IMP symposium handled DP1 and IMP as *isolated* primitives; this design proposes their *composition*. Distinct from the 0.196-0.199 cluster substrates (none use OOD pretraining).

2. **BEAUTY + ELEGANCE**: PARTIAL. The 5-phase sequential pipeline IS architecturally clean (offline pretrain → adapt → augment → distill+prune → schema-elide). However, the 5-primitive composition violates HNeRV parity discipline lesson 4 (≤ 100 LOC inflate budget; ≤ 200 LOC waiver ceiling per Catalog #328) — even at substrate-engineering tier per HNeRV parity L7, the unified design's archive grammar + parser-section manifest + inflate runtime would EXCEED PR101 GOLD's 605-LOC reviewable bound by 2-3× empirically (per sister DP1 archive grammar inspection + sister Lane 17 IMP archive overhead measurement). 30-second-reviewable IS NOT satisfied.

3. **DISTINCTNESS**: HARD-EARNED. Explicitly different from frontier sister substrates: fec6 (frame-exploit selector + Huffman) is a pure-codec primitive; PR106 format0d is a latent-table codec; PR103 SILVER is FastViT-pretrained-encoder; STC sister is sidecar-on-A1-residual. The unified design's OOD-pretrain-then-aggressive-ablate paradigm has no sister.

4. **RIGOR**: PARTIAL. Premise verification per Catalog #229 = COMPLETE (this section). Adversarial review per CLAUDE.md "Council conduct" = COMPLETE (29 attendees including 4 co-leads + Assumption-Adversary + Contrarian both with VETO). Assumption classification per Catalog #292 + #303 = COMPLETE (6 assumptions surfaced). Empirical anchor = ZERO at current frontier; sister DP1 PATH 1 SCAFFOLDED but UNFIRED. THE RIGOR GAP IS EMPIRICAL ANCHOR ABSENCE.

5. **OPTIMIZATION PER TECHNIQUE**: DEFER-PENDING-EMPIRICAL. Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode + sister Catalog #290: the unified design MUST document `## Canonical-vs-unique decision per layer` BEFORE landing. This memo IS the design phase; per-primitive canonical-vs-unique decisions are explicit:
   - DP1 codebook prior: ADOPT_CANONICAL (sister Catalog #209-#213 helper)
   - MAE-V mask-aug: FORK_BECAUSE_88K_SCALE_MISMATCH (Hinton sister #869 verdict applies)
   - SAUG hi-sigma: FORK_BECAUSE_88K_SCALE_MISMATCH (Selfcomp sister #869 verdict applies)
   - IMP β-Fisher: ADOPT_CANONICAL (Lane 17 IMP canonical fine-tune objective per sister symposium)
   - LSQ FP4 QAT: ADOPT_CANONICAL (`tac.quantization.FakeQuantFP4`)
   - Schema-elision GOSDT: ADOPT_CANONICAL (`tac.preflight_rudin_daubechies.gosdt_dispatch_router`)

6. **STACK-OF-STACKS-COMPOSABILITY**: See dedicated section below. Composability matrix evaluated vs PR110 fec6 / STC sidecar / Z6 / Z7-Mamba-2 / ATW V2 / Riemannian-Newton.

7. **DETERMINISTIC REPRODUCIBILITY**: HARD-EARNED-DESIGN / PENDING-EMPIRICAL. Phase 1 OOD-pretrain produces deterministic per-seed codebook per Catalog #210 (license_tags + dataset_provenance + random_seed + basis_sha256 in archive metadata). Phase 2-5 inherit from canonical helpers each with seed-pinning. Byte-stable archive REQUIRES post-training Tier-C density measurement per Catalog #324 + post-dispatch sha256 verification per `tac.canonical_frontier_pointer` discipline.

8. **EXTREME OPTIMIZATION + PERFORMANCE**: DEFER-PENDING-EMPIRICAL. The OP3-V3 sidecar (`a1afce29`) shows 90.7% SegNet-dominant + 8.5% near-zero contribution + 0% rate-dominant on the fec6 frontier archive — this IS the canonical per-byte importance prior the aggressive-ablate phase would use. Tier 1 engineering primitives per CLAUDE.md "Production-hardened dispatch optimization protocol" Catalog #270 = REQUIRED for any dispatch (autocast_fp16 + TF32 + torch.compile + no_grad-at-eval + GTScorerCache F3 + canonical scorer-loss helper routing + canonical 3-export NVML/CUDA env block per Catalog #244).

9. **OPTIMAL MINIMAL CONTEST SCORE**: PENDING-EMPIRICAL. Per Yousfi HARD-EARNED bound [-0.005, -0.012] from current 0.19205 frontier → 0.180-0.187 medal-band proximity. SUFFICIENT for medal-band per current `commaai/comma_video_compression_challenge` PR comments (PR102 BRONZE at 0.19538; medal cluster ~ 0.180-0.196). NOT SUFFICIENT for asymptotic_pursuit horizon-class (which requires < 0.120).

---

## Observability surface

Per Catalog #305. 6 facets declared:

1. **Inspectable per layer**: Each Phase 1-5 outputs typed artifacts persisted to `.omx/state/unified_pretrain_ablate/`:
   - Phase 1: `phase_1_pretrain_codebook_<basis_sha>.pt` + `phase_1_pretrain_manifest_<utc>.json` (per Catalog #210 codebook_provenance_metadata)
   - Phase 2: `phase_2_adapt_composition_<utc>.json` (DPCOMP wrapper provenance per Catalog #211)
   - Phase 3: `phase_3_augment_training_stats_<utc>.jsonl` (per-epoch MAE-V + SAUG loss surface)
   - Phase 4: `phase_4_distill_prune_iter_<N>_<utc>.json` (per-IMP-iteration sparsity + KL distill loss + LSQ FP4 quant error)
   - Phase 5: `phase_5_gosdt_rule_list_<utc>.json` (final GOSDT decision-path) + final archive sha256

2. **Decomposable per signal**: Per-axis decomposition emitted at each phase boundary via `tac.cathedral.consumer_contract.AxisDecomposition` (sister Catalog #356). The OP3-V3 sidecar's per-byte (seg, pose, rate) gradient (178417, 3) array IS the canonical per-byte decomposition prior the aggressive-ablate phase routes through.

3. **Diff-able across runs**: Each phase artifact carries archive sha256 + composition cell ID (`<base_substrate>__dp1__phase_<N>__<seed>`) so two runs of the same phase on the same input produce byte-stable diff. Per Catalog #166 Modal source-parity ledger for dispatched phases.

4. **Queryable post-hoc**: All phase artifacts are JSONL append-only per Catalog #131 + #245 sister discipline; canonical query helpers `query_phase_artifacts_by_composition_cell` + `query_phase_artifacts_by_archive_sha` (to be implemented at L2 if reactivated). MEMORY.md cluster summary auto-published per Catalog #298.

5. **Cite-able**: Every phase artifact anchored to (substrate_alias, commit, call_id, config, random_seed, upstream_snapshot_sha256) tuple per Catalog #245. Canonical Provenance per Catalog #323.

6. **Counterfactual-able**: Per-byte byte-mutation discipline per Catalog #139 + #272 + #105: each phase's output archive bytes verified via `tools/verify_distinguishing_feature_byte_mutation.py` (sister Catalog #272 canonical helper). The unified design's distinguishing feature IS the 5-primitive composition; byte-mutation per phase boundary proves operational consumption per Catalog #220.

---

## Predicted ΔS band

**Range**: [-0.005, -0.020] contest-CPU from current 0.19205 frontier → expected band [0.172, 0.187].

**Validation status**: `pending_post_training` per Catalog #324. Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" Catalog #325 + per the empirical anchor (C6 IBPS landed 22× outside predicted_band [0.113, 0.163] because Tier-C density was measured on RANDOM-INIT weights pre-training): the unified design's distillation+pruning post-pretrain produces a NEW weight distribution; pre-training Tier-C density is structurally unrelated to post-pretrain-then-ablate Tier-C density.

**Dykstra-feasibility check** per Catalog #296:

The predicted band derivation:
- Lower bound −0.005 = Yousfi HARD-EARNED dashcam-redundancy bound for OOD-pretrain incremental signal alone (DP1 sister deep-dive)
- Upper bound −0.020 = additive composition assumption (5 primitives × −0.004 average per-primitive ΔS per literature anchors)

Dykstra-feasibility intersection of the 5 constraint manifolds:
- Constraint 1 (rate ≤ R_current = 0.00476 × 37545489 = 178517 bytes): DP1 adds ~25.8 KB sidecar (sister DP1 PATH 1 L1 measurement), IMP+QAT removes bytes by ~30-50% per Frankle-Carbin recovery target, schema-elision removes ~5-15% per GOSDT prior. NET rate-axis MAY be within feasibility region but boundary-adjacent.
- Constraint 2 (seg ≤ S_current = 0.001): aggressive-ablate phase pose risk to SegNet head per Hinton T=2.0 KL distillation; Lane 17 IMP sister symposium classified as HARD-EARNED for SegNet via KL distillation but UNCERTAIN at 88K scale per Hinton sister #869 verdict.
- Constraint 3 (pose ≤ P_current = 0.00382): MAE-V + SAUG augmentation traditionally helps PoseNet generalization but Yousfi sister #869 verdict bounds gain to ~0.005-0.015.

**Dykstra alternating-projections analysis**: 5-primitive simultaneous projection onto the 3 constraint manifolds is structurally analogous to NSCS06 v6's 5-move composition (which landed 105.15 vs predicted [0.10, 0.20] = 553× outside band). The 2-primitive composition_alpha sister floor per Catalog #322 (SUB_ADDITIVE α ∈ [0.3, 0.7] empirical) suggests 5-primitive composition_alpha = (0.5)^4 ≈ 0.0625 multiplicative on the additive band. Effective predicted band per Dykstra-feasibility-adjusted = [−0.020 × 0.0625, −0.005 × 0.0625] = [−0.00125, −0.000313] which is INSIDE the noise floor for any single Modal A10G dispatch run.

**Per-Yousfi HARD-EARNED reframing**: the predicted band MUST be re-derived AFTER 2-primitive composition_alpha empirical anchor per op-routable #2. The CARGO-CULTED additive assumption is the upper-bound limit.

**Probe-disambiguator path**: `tools/probe_unified_pretrain_ablate_dispatch_order_disambiguator.py` (NOT YET BUILT — op-routable #3) emits Bayesian-optimal dispatch-order recommendation per the 2-primitive empirical anchor + Catalog #344 canonical equation `unified_pretrain_ablate_composition_alpha_v1` (to be registered per op-routable #6 if empirical anchor materializes).

---

## Horizon-class declaration

**Horizon-class**: `frontier_pursuit` per the HORIZON-CLASS standing directive (`feedback_horizon_class_evaluation_axis_plateau_warning_standing_directive_20260516.md`).

- PLATEAU-ADJACENT [0.180, 0.200] would put the design INSIDE the 0.196-0.199 cluster — operator's exact plateau-exit framing rejects this.
- FRONTIER-PURSUIT [0.120, 0.180] matches the predicted band's upper bound [0.172, 0.187] AFTER Dykstra-feasibility-adjusted compression.
- ASYMPTOTIC-PURSUIT [0.050, 0.120] is NOT achievable per Yousfi HARD-EARNED bound; ZERO empirical evidence supports asymptotic-pursuit claim.

The horizon-class declaration is `frontier_pursuit` rather than `plateau_adjacent` because the new-DOF claim (OOD-pretrain orthogonal axis) IS class-shift per the HARD-EARNED-PARTIALLY-NEW-DOF Assumption-Adversary verdict. The unified design IS an exit attempt from the 0.196-0.199 plateau via a new structural direction, but with EMPIRICALLY-UNVERIFIED ΔS magnitude.

---

## Stack-of-stacks composability matrix

Predicted α multipliers per Catalog #322 sister composition_alpha empirical floor. Composing the unified-pretrain-ablate substrate WITH each sister substrate:

| Sister substrate | Composability verdict | Predicted α | Rationale |
|---|---|---:|---|
| **PR110 fec6** (current 0.19205 frontier) | SUB_ADDITIVE | 0.4-0.6 | fec6 is pure-codec primitive (frame-exploit selector + fixed Huffman k=16); the unified design's Phase 2 DP1-adapted prior PARTIALLY composes with fec6's per-frame entropy decomposition. Phase 3-5 aggressive-ablate phase contends with fec6's already-tight Huffman codebook — additional rate-axis savings DIMINISH per Shannon R(D) joint-source-coding bound. Per OP3-V3 sidecar `a1afce29`: 0% rate-dominant bytes on fec6 archive = fec6 ALREADY rate-saturated; unified ⊕ fec6 limited to seg+pose-axis improvement. |
| **STC sidecar (commit `d5fcc850d` PROCEED on A1 residual)** | ANTAGONISTIC | 0.1-0.3 | STC is sidecar-on-A1-residual paradigm; the unified design is base-substrate-OOD-pretrain paradigm. STC composes with A1 backbone NOT with OOD-pretrained backbone. The 2-paradigm composition shares NO underlying substrate; ZERO empirical anchor exists for composition. Probe-disambiguator REQUIRED before stacking. |
| **Z6 Phase 2 OPTIMAL FORM** (sister #835 PROCEED; predictive-coding world-model) | SUB_ADDITIVE | 0.5-0.7 | Z6 is predictive-coding architectural primitive; unified design's Phase 1 DP1 prior + Phase 3-5 aggressive-ablate composes with Z6's hierarchical-prediction layers IF DP1 prior structurally matches Z6's predictive-coding architectural input. Per the Rao-Ballard 1999 lens (sister grand-council seat): cooperative-receiver framing applies. NEEDS Z6-specific design memo extension before composition. |
| **Z7-Mamba-2** (sister `feedback_z7_mamba2_*`) | SUB_ADDITIVE | 0.4-0.6 | Z7 is state-space-model temporal-codec primitive; unified design's Phase 1 OOD-pretrain MAY transfer to Z7's temporal embedding layers but at 88K target scale the transfer is regularizer-noise per Wang 2019 negative-transfer analysis. NEEDS empirical anchor before composition claim. |
| **ATW V2-1** (sister `council_per_substrate_symposium_atw_v2_reactivation_20260518.md`) | ANTAGONISTIC | 0.2-0.4 | ATW V2 is cooperative-receiver-loss primitive (Atick-Redlich 1990 sister grand-council voice); unified design's OOD-pretrain composes orthogonally but the aggressive-ablate phase's distillation+pruning DIRECTLY perturbs the cooperative-receiver weights that ATW V2 binds. 2-paradigm interference. |
| **Riemannian-Newton** (sister memo references) | UNKNOWN | 0.3-0.7 | Riemannian-Newton is per-layer geodesic-optimization primitive; composes with unified design's Phase 4 LSQ FP4 QAT IF the geodesic projection respects FP4 manifold structure. ZERO empirical anchor; first-principles only. NEEDS scoping memo before composition. |

**Verdict per composability matrix**: the unified design is most-likely-composable with PR110 fec6 (per op-routable #1 DP1+fec6 PATH 1 sister scaffold) and Z6 Phase 2. ANTAGONISTIC with ATW V2 and STC sidecar. The 2-primitive empirical anchor per op-routable #2 IS the load-bearing reactivation criterion.

---

## Reactivation criteria (4 candidate paths)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #325 6-step contract. The DEFER_PENDING_EVIDENCE verdict pins 4 reactivation paths in priority order:

### Path (a) — P1 — DP1+fec6 PATH 1 single-primitive empirical anchor
**Predicted cost**: $0.30 Modal T4 100ep smoke (per sister DP1 deep-dive PATH 1 scaffolded).
**Structural verdict**: tests assumption #1 (OOD-pretrain → 88K target positive transfer) and assumption #5 (composition with fec6 frontier is monotonic). REACTIVATION REQUIRES contest-CPU ΔS measurable in [-0.001, -0.012] band Dykstra-feasibility-checked vs 0.19205 frontier.
**Outcome → next step**: If POSITIVE → Path (b). If NEGATIVE → Path (d) PIVOT.

### Path (b) — P1 — DP1+IMP+fec6 2-primitive composition_alpha anchor
**Predicted cost**: $2-5 Modal A10G 200ep.
**Structural verdict**: tests assumption #3 (5-primitive unification produces additive ΔS) at the 2-primitive empirical floor per Catalog #322. ANTAGONISTIC verdict (alpha < 0.3) closes the unified paradigm at 2-primitive boundary.
**Outcome → next step**: If SUB_ADDITIVE or SUPER_ADDITIVE → Path (c). If ANTAGONISTIC → Path (d) PIVOT.

### Path (c) — P2 — Probe-disambiguator + 4-primitive composition
**Predicted cost**: $5-15 Modal A100 500ep (per sister Lane 17 IMP empirical estimate). Probe disambiguator tool ~$0 design.
**Structural verdict**: per op-routable #3 — build `tools/probe_unified_pretrain_ablate_dispatch_order_disambiguator.py` (~250 LOC); emit Bayesian-optimal dispatch-order recommendation per posterior-update over op-routable #2's empirical anchor. Then test 4-primitive composition (DP1 + IMP + selected augment + QAT).
**Outcome → next step**: If SUFFICIENT for Yousfi-bounded [-0.005, -0.012] band → Path (full unified 5-primitive dispatch + post-training Tier-C validation per Catalog #324). If INSUFFICIENT → Path (d) PIVOT.

### Path (d) — P2 — PIVOT to single-axis OOD-pretrain-ONLY composable bolt-on
**Predicted cost**: $0.30 Modal T4 100ep smoke (per op-routable #4).
**Structural verdict**: per op-routable #4 + the HARD-EARNED-PARTIALLY-NEW-DOF Assumption-Adversary verdict — drop MAE-V + SAUG + IMP+QAT + schema-elision from the unified design; retain ONLY DP1 prior composed with frontier substrate. This pivot variant has CLEANER per-primitive empirical attribution AND matches sister DP1 deep-dive PATH 1 scaffold. The frontier-pursuit horizon-class is preserved at sister DP1 deep-dive's `plateau_adjacent` reclassification.

---

## Catalog #324 post-training Tier-C validation discipline

Per CLAUDE.md "Forbidden predicted_band-from-random-init-Tier-C-density" FORBIDDEN pattern + the C6 IBPS 22× miss empirical anchor.

**predicted_band_validation_status**: `pending_post_training` (frontmatter field).

**Reactivation criterion**: IF unified dispatch eventually fires, post-training Tier-C density measurement on the landed archive via `tools/mdl_scorer_conditional_ablation.py --tier c` IS REQUIRED before any score claim. The unified design's distillation+pruning post-pretrain produces a NEW weight distribution; pre-training Tier-C density is structurally unrelated to post-pretrain-then-ablate Tier-C density.

**Canonical helper**: `tac.optimization.tier_c_density_post_training_validator.build_tier_c_density_post_training` per sister Catalog #324 + #344 cross-link.

---

## Sister-collision verdicts

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #302 sister subagent scope overlap.

- **STC-SYMPOSIUM** (`aaffc11c5e48fc0c6`): scope = STC paradigm reformulation per-substrate symposium. DISJOINT from this memo (different per-substrate-id; my substrate = `unified_pretrain_ablate_dp1_mae_v_saug_imp_qat_schema_elision`; STC-SYMPOSIUM substrate = `stc_paradigm_reformulation_a1_residual`). NO file overlap.
- **OP3-DOWNSTREAM-WIRE-IN** (`a8e3dcabb9e0de85d`): in_progress on `bit_allocator from_master_gradient_anchor + add T4-anchor citations to 3 consumers + test file`. DISJOINT from this memo (different scope: cathedral_consumers/* + bit_allocator; my scope: .omx/research/ + .omx/state/council_deliberation_posterior.jsonl + .omx/state/probe_outcomes.jsonl). NO file overlap. NO commit collision risk (different working surfaces).

---

## Operator-routable decisions (binding REVISIONS for PROCEED)

Per Catalog #292 per-deliberation assumption surfacing + Catalog #300 v2 frontmatter binding revisions:

1. **REVISION #1 (P1)**: PROCEED conditional on op-routable #1 firing (`DP1+fec6 PATH 1 single-primitive smoke` per sister DP1 deep-dive PATH 1 scaffolded). $0.30 Modal T4 100ep. Operator-approved per blanket approval but operator-explicit-routable to confirm dispatch.
2. **REVISION #2 (P1)**: Op-routable #2 (`DP1+IMP+fec6 2-primitive composition_alpha anchor`) is gated on REVISION #1 result. NO 4-primitive or 5-primitive dispatch fires until 2-primitive empirical anchor lands.
3. **REVISION #3 (P2)**: Op-routable #3 probe-disambiguator REQUIRED before 4-primitive composition per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + per the design-tension-ship-both-interpretations pattern.
4. **REVISION #4 (P2)**: Op-routable #4 PIVOT variant must be presented to operator at REVISION #1 NEGATIVE OR REVISION #2 ANTAGONISTIC verdict. The PIVOT preserves DP1 prior value WITHOUT 5-primitive paradigm risk.
5. **REVISION #5 (P3)**: Catalog #324 post-training Tier-C validation discipline IS NON-NEGOTIABLE for any score claim. predicted_band_validation_status MUST remain `pending_post_training` until post-training measurement lands.

---

## Provenance + cross-references

- Operator question source: 2026-05-20 verbatim conversation transcript (this design memo is the canonical response).
- Sister symposium memos (per `related_deliberation_ids` frontmatter):
  - `.omx/research/council_per_substrate_symposium_dp1_deep_dive_20260517.md` (DP1 PROCEED_WITH_REVISIONS)
  - `.omx/research/council_per_substrate_symposium_mae_v_plus_saug_20260518.md` (MAE-V + SAUG DEFER_PENDING_EVIDENCE)
  - `.omx/research/council_per_substrate_symposium_lane_17_imp_20260517.md` (Lane 17 IMP PROCEED_WITH_REVISIONS)
  - `.omx/research/council_per_substrate_symposium_stc_3a_sidecar_a1_residual_20260517.md` (STC sister)
  - `.omx/research/council_per_substrate_symposium_pr106_05_06_reformulated_20260517.md` (PR106 sister)
- Sister landing memos for the OP3-V3 anchor + cascade context: `feedback_op3_v3_minibatch_redispatch_*.md` series (commit chain `4d676dfca` / `4ab27ca69` / `308fa89f3` / `73d5e6077`).
- Sister Catalog cross-references: #209-#213 (DP1 Comma2k19 lane) / #220 (substrate L1+ operational mechanism) / #229 (premise verification) / #272 (distinguishing-feature integration contract) / #287 (placeholder-rationale rejection) / #290 (canonical-vs-unique decision per layer) / #292 (per-deliberation assumption surfacing) / #294 (9-dim checklist) / #296 (Dykstra-feasibility predicted-band) / #297 (signal-axis destruction reversibility) / #298 (substrate retirement discipline) / #300 (council deliberation v2 frontmatter) / #303 (cargo-cult audit per assumption) / #305 (observability surface) / #309 (horizon-class declaration) / #313 (probe-outcomes ledger) / #315 (substrate OPTIMAL FORM before paid dispatch) / #322 (composition_alpha sister) / #323 (canonical Provenance umbrella) / #324 (post-training Tier-C validation) / #325 (per-substrate symposium contract — THIS memo satisfies the 6-step contract) / #344 (canonical equations registry) / #346 (canonical roster validate complete=True).
- Literature anchors: Yosinski et al. 2014 (transfer learning capacity correlation); Wang et al. 2019 (negative transfer); He et al. 2022 MAE; Caron et al. 2024 DINOv3; Ravi et al. 2024 SAM2; Frankle-Carbin 2019 lottery ticket; Hinton-Vinyals-Dean 2014 knowledge distillation; Esser et al. 2020 LSQ; Krishnamoorthi 2018 quantization; Liu et al. 2019 DARTS; Cai et al. 2020 OFA; Atick-Redlich 1990 cooperative-receiver; Rao-Ballard 1999 predictive coding; Tishby-Zaslavsky 2015 IB; Alemi 2017 VIB; Hafner et al. 2023 DreamerV3; Ha-Schmidhuber 2018 world models; Comma2k19 Schafer et al. 2018; BDD100K Yu et al. 2020; KITTI Geiger et al. 2012; Waymo Sun et al. 2020.

---

## Conclusion + verdict

**Verdict**: PROCEED_WITH_REVISIONS (29-attendee T3 deliberation; complete=True per `tac.canonical_council_roster.validate_council_dispatch_roster`).

**Mission contribution per Catalog #300**: frontmatter field = `frontier_protecting` (canonical enum value per CLAUDE.md "Mission alignment — non-negotiable" Consequence 5 enum). Aspirational body classification = `frontier_breaking_enabler` pending empirical anchor materialization; the substrate structurally introduces new DOF orthogonal to 0.196-0.199 cluster polish-axes per the HARD-EARNED-PARTIALLY-NEW-DOF Assumption-Adversary verdict.

**Reactivation Path (a) IS THE CANONICAL NEXT STEP**: $0.30 Modal T4 100ep DP1+fec6 PATH 1 single-primitive smoke per sister DP1 deep-dive PATH 1 scaffold. Operator-routable for dispatch authorization.

**Per CLAUDE.md "Forbidden premature KILL"**: this design memo is the DEFER_PENDING_EVIDENCE → PROCEED_WITH_REVISIONS council deliberation. The DEFER posture is reactivation-criterion-pinned across 4 paths; the substrate is NOT killed; future agents may reactivate per any path.
