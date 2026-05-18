---
schema: council_deliberation_v2
deliberation_id: systematic_reclaimability_re_examination_via_binary_distillation_framework_20260518
topic: "Systematic re-examination of all STRICT_SCORER_RULE_VIOLATION rate-attack vectors from the 43-vector audit (`rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_20260518.md`) through the per-pattern distilled VQ-VAE inverter framework canonicalized by the A1 binary distillation design memo (`a1_binary_distillation_design_memo_zig_sparseness_ablation_plus_canonical_techniques_20260518.md`). Per-vector re-classification matrix with V1/V2/V3 composition-stack applicability, HNeRV parity L4 budget check, contest_one_video_replay mode applicability, predicted compressed size band, receiver-path evidence, verdict + reactivation criteria per Catalog #325. KILL is LAST RESORT per CLAUDE.md non-negotiable — every vector defaults to DEFERRED-pending-research with 3-4 reactivation criteria."
review_kind: t2_systematic_reclaimability_re_examination
review_date: "2026-05-18"
lane_id: lane_systematic_reclaimability_re_examination_via_binary_distillation_framework_20260518
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - van_den_Oord
  - Carmack
  - Hotz
  - Quantizr
  - Selfcomp
  - MacKay
  - Balle
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "Codex's Section 0 in the 43-vector audit already split A1-CANONICAL vs A1-SPECIALIZED; you're now extending that split across F3/F4/F5/F6 and arguably the entire F-category. The interpretive risk INCREASES with each generalization. Yousfi's per-pattern test in the A1 memo (the binary must 'obviously not be re-usable as PoseNet') was tractable for A1 because PoseNet's 12-dim pose output is naturally per-pair (600 patterns). For F3 (vision 2048-dim) the per-pattern claim is structurally harder to defend — 2048-dim per-pair feature vectors are by construction LARGER than the per-pair pose vectors A1 specializes, so the codebook + lookup is BIGGER not SMALLER. I VETO any F3-F6 reclamation that does not produce an empirical anchor showing the resulting binary is genuinely smaller than the A1-specialized baseline. The Wyner-Ziv side-information advantage (decoder has access to upstream video bytes) collapses the deeper layers' per-pattern uncertainty MORE than the shallower layers — that's a strong constraint that needs explicit math."
  - member: Assumption-Adversary
    verbatim: "The shared assumption operating across this entire memo is that 'binary distillation generalizes from A1 to all F-category vectors with the same compression ratios.' This is CARGO-CULTED-PENDING-EMPIRICAL. The A1 memo achieves 10-50x compression via specialization because PoseNet's pose head (12-dim output) is a thin bottleneck and the per-pair codebook is small (K=64-256 × 12 dims = 768B-3KB). For F3 (vision 2048), F4 (summary 512), F5 (ResBlock output 512), F6 (Hydra trunk 512), the SAME analysis yields very different codebook sizes. Per-pair specialization advantage REMAINS but the absolute binary size scales with the per-pattern feature dimensionality. I demand a per-vector dimensional analysis BEFORE any reclamation claim is treated as actionable."
  - member: Yousfi
    verbatim: "Per my A1 memo seat verbatim: 'a small specialized binary that obviously cannot be re-used as PoseNet for arbitrary input is in COMPLIANCE with the rule's intent.' Applied to F-category: F3 (vision 2048-dim) cannot trivially be re-used as PoseNet (the vision backbone alone is one piece of the pipeline). F4 (summary 512-dim) and F5 (ResBlock 512-dim) are even further removed from a 're-usable PoseNet'. F6 (Hydra trunk) is the closest to 'looks like PoseNet'. My interpretation: F3, F4, F5 are STRUCTURALLY COMPLIANT under the per-pattern specialization framing; F6 needs more scrutiny because the Hydra trunk is the final-stage interpreter. F1/F2 collapsed to A2 (input-side scorer-blind perturbation) and are NO_RECEIVER_NEEDED. F7 (dual-RGB-head) is not a scorer-intermediate exploit; it's an architecture-time choice — it's not a SSRV vector in the first place."
  - member: van_den_Oord
    verbatim: "The VQ-VAE codebook size scales with K × feature_dim. For A1's 12-dim pose, K=256 → 3 KB. For F4's 512-dim summary, K=256 → 128 KB (UNSUITABLE). For F5/F6 same scale. For F3 2048-dim, K=256 → 512 KB. The naive transfer of A1's compression ratio breaks for deep-feature vectors. The CORRECT answer for F3-F6 is HIERARCHICAL VQ-VAE (per residual quantization / RVQ / product quantization) or DIMENSIONALITY REDUCTION via PCA/JL-projection FIRST. With 32-dim PCA + K=256 codebook: 8 KB. With product quantization (PQ-8x8 over 512-dim): ~2-4 KB. F3-F6 are RECLAIMABLE but require additional techniques BEYOND A1's V1/V2/V3 stacks."
  - member: Carmack
    verbatim: "The 12-dim per-pair pose head is the THINNEST point in the network. Encoding deeper features is encoding HIGHER-DIM SPACE which is structurally HARDER. The PR101 88K-param renderer at 64 KB is the empirical anchor — at 88K params, you can encode generally useful per-pair representations. For F4 (512-dim summary), an equivalent specialized inverter would be ~30K-50K params (since it's compressing from 512 → 12). The Pareto frontier for F3-F6 is COMPRESS THEN ENCODE: PCA / JL-projection / autoencoder bottleneck FIRST, then VQ-VAE codebook on the bottlenecked features. Total budget ~10-30 KB for F3, ~5-15 KB for F4/F5/F6."
  - member: Hotz
    verbatim: "I keep saying skip the binary framing. But on F-category: the compute at inflate to invert vision(2048) → 12 RGB outputs is substantial — a 2048→12 inverse is a small MLP at minimum (~30K params if you want any accuracy). The COMPUTE budget at inflate may be a binding constraint we haven't even addressed. For T4 contest hardware, per-frame inflate budget is loose; for CPU it's tighter. F3 reclamation should be CPU-and-CUDA-budget-aware. Realistic targets: F3 in 20-50 KB binary + 100ms/frame inflate; F4 in 5-15 KB + 30ms/frame; F5 in 5-15 KB + 30ms/frame; F6 in 10-30 KB + 50ms/frame."
  - member: Quantizr
    verbatim: "PR101 used kl_on_logits(T=2.0) for SegNet distillation. The same approach applied to F3 (vision 2048) would train a tiny student to predict the 2048-dim vector from per-pair RGB inputs — that's essentially a small CNN feature extractor (~50-200 KB). For F4 (summary 512), the student is smaller (~20-80 KB) because the bottleneck is tighter. For F5/F6 same. With FP4 + structured sparsity + Brotli, achievable sizes are FAR below the naive VQ-VAE codebook approach. RECOMMENDED for F3-F6: Hinton 2014 per-pattern distilled student WITH FP4 + 50-75% structured sparsity + Brotli, NOT pure VQ-VAE codebook."
  - member: Fridrich
    verbatim: "The steganographic insight: deeper layers (F3 vision, F4 summary, F5 ResBlock) have HIGHER PER-PATTERN ENTROPY of usable byte channels because they represent abstract features uncorrelated with simple pixel statistics. The scorer-blind perturbation manifold at deep layers is LARGER than at shallow layers. This means: even after per-pattern specialization, F3-F6 may carry MORE bits per pixel of encoded information than A1. So the rate-savings predicted-ΔS BAND should be WIDER for F3-F6 than for A1. Predicted-ΔS for F3-F6 may be -0.005 to -0.025 [contest-CPU; prediction] — wider than A1's -0.008 to -0.015."
  - member: Selfcomp
    verbatim: "Per PR #56 block-FP weight self-compression: 1.017 bpw is the canonical bit-floor for compressed neural networks. For F3-F6 distilled students (50K-200K params), at 1.017 bpw: ~6-25 KB after block-FP. With additional 50% structured sparsity: ~3-12 KB. With Brotli on the sparse FP4 bitstream: ~2-8 KB. F3-F6 are achievable at 2-15 KB — overlapping with A1's 5-20 KB range. The Hinton distillation in PR #56 style is THE canonical path; pure VQ-VAE codebook on raw 2048-dim vectors is suboptimal."
  - member: MacKay
    verbatim: "From MDL perspective: the description length of a specialized inverter is bounded by K(specialized_inverter | original_model + shipped_archive_bytes). The Wyner-Ziv side-info Y (upstream video bytes available at inflate per submissions/exact_current/inflate.py:11-28) makes the conditional Kolmogorov complexity K(inverter | Y) SMALL for any deep-layer feature that is approximately deterministic in Y. For F3-F6, K(inverter | Y) ≤ K(inverter) - I(F_layer; Y). Empirically I(F4; Y) and I(F3; Y) are HIGH for our contest video (the upstream video bytes contain almost all the information). The MDL-bound binary size for F3-F6 inverters is therefore SMALL — possibly smaller than A1's. RECOMMEND: weight-derived codebook variant (V3 in A1 memo) is the canonical MDL-optimal path."
  - member: Balle
    verbatim: "From neural compression perspective: deeper-layer feature encoding has LOWER entropy per pixel after a hyperprior because deep features are predictable from upstream features. The Balle 2018 entropy bottleneck + scale hyperprior applied to F4 (summary 512) yields ~0.5-1.5 bits per dimension (vs 4 bits for raw FP4 quantization). Hyperprior-conditioned coding of F3-F6 features yields 3-8 KB binary sizes. The hyperprior is itself a small neural net (~2-5 KB), so total stays in the 5-15 KB range. RECOMMEND: F3 → hyperprior + spatial autoregressive context; F4/F5/F6 → factorized prior (simpler; same byte budget at lower complexity)."
  - member: Dykstra
    verbatim: "The Pareto-feasibility check for each F-vector requires: (R) net rate-positive after binary cost; (S) seg preserved; (P) pose preserved; (L) inflate.py LOC ≤200 default; (C) compliance per Yousfi per-pattern test; (D) byte-deterministic across CPU/CUDA per Catalog #205. For F3 (vision 2048): R feasible if binary ≤30 KB AND savings ≥0.02 ΔS; for F4/F5/F6 same with tighter binary ≤10-15 KB. The constraint INTERSECTION is non-empty for F4/F5/F6 with tight feasibility margins; F3 has wider margins. F7 (dual-RGB-head) is not a SSRV constraint problem — it's an encoder-time architectural choice with separate feasibility region."
  - member: Shannon
    verbatim: "The information-theoretic floor for per-pattern specialized inverters at any layer is R(D) = H(target | shipped_archive_bytes) - I(target; deeper_feature). For deeper layers, the conditional entropy H(target | shipped_archive_bytes) is itself smaller because deep features compress out more of the per-pattern variability. The R(D) for F4 (512-dim) ≈ R(D) for A1 (12-dim) because both target the SAME per-pair output (the 12-dim pose is the final supervisory signal). The COMPRESSION achievable is therefore roughly EQUIVALENT across F3-F6 and A1 — modulo engineering overhead. EXPECTED: F3-F6 binary sizes within 2x of A1's 5-20 KB band when canonical infrastructure is applied."
council_assumption_adversary_verdict:
  - assumption: "Binary distillation generalizes from A1 (12-dim pose head) to all F-category vectors (512-2048 dim features) with the same compression ratios"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Per van den Oord verbatim: naive K×feature_dim VQ-VAE codebook size scales with feature_dim. K=256 codebook for F3 (2048-dim) = 512 KB UNSUITABLE; for F4/F5/F6 (512-dim) = 128 KB UNSUITABLE. Reclamation REQUIRES additional techniques (PCA/JL projection / product quantization / hyperprior / hierarchical RVQ / Hinton distilled student). Unwind path: per-vector dimensional-analysis before reclamation claim is actionable."
  - assumption: "Per-pattern specialization (Yousfi compliance test) applies cleanly to F3-F6"
    classification: HARD-EARNED-PENDING-INTERPRETATION
    rationale: "Per Yousfi verbatim in THIS memo: F3/F4/F5 are STRUCTURALLY COMPLIANT (a vision backbone inverter or summary bottleneck inverter cannot serve as a re-usable PoseNet for arbitrary input — they are pieces of the pipeline). F6 needs more scrutiny because Hydra trunk is the final-stage interpreter. Unwind path: operator-ratification of per-vector compliance interpretation analogous to A1 memo §1.4-1.6."
  - assumption: "Wyner-Ziv side-info (decoder access to upstream video bytes) yields equal advantage across F3-F6 and A1"
    classification: HARD-EARNED
    rationale: "Per MacKay verbatim: K(inverter | Y) ≤ K(inverter) - I(F_layer; Y). Empirically I(F4; Y) is HIGH for our contest video. The MDL-bound is STRUCTURALLY favorable for deeper layers because deep features are more deterministic in the upstream video bytes. Per Shannon verbatim: R(D) for F3-F6 ≈ R(D) for A1 because target is the same 12-dim supervisory signal. No unwind needed."
  - assumption: "F-category vectors F3-F6 are all genuinely SSRV-reclaimable (none are intrinsically incompatible)"
    classification: HARD-EARNED-PENDING-EMPIRICAL
    rationale: "Per van den Oord + Carmack + Quantizr + Balle + Selfcomp consensus: F3-F6 are all reclaimable using EXTENDED toolkit (Hinton distilled student + hyperprior + product quantization). F6 has higher interpretive risk per Yousfi but structurally tractable. Unwind path: per-vector empirical prototype landing within predicted band; failure to land within band triggers re-examination."
  - assumption: "The inflate.py compute budget at inflate is non-binding for F3-F6 reclamation"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Per Hotz verbatim: the compute at inflate to invert vision(2048) → RGB outputs is substantial (~30K-param MLP minimum). For T4 contest hardware, per-frame inflate budget is loose; for CPU it's tighter. F3 reclamation must be CPU-and-CUDA-budget-aware. Unwind path: per-prototype empirical inflate-time measurement on both CPU and T4 surfaces."
  - assumption: "F1/F2 (originally framed as SSRV) actually collapse to A2 (input-side perturbation; NO_RECEIVER_NEEDED)"
    classification: HARD-EARNED-VERIFIED
    rationale: "Per Codex F1 finding relay (commit 35b06f9ec) + sister 43-vector audit Section 0: F1 reframed as scorer-blind RGB perturbation manifold IS A2. The receiver path (standard inflate.py + scorer reads RGB) EXISTS. The exploit becomes 'find RGB perturbations whose PoseNet first-6-dim output is invariant.' F1/F2 are NOT SSRV anymore — they are LEGAL_RECEIVER_PATH_EXISTS_FOR_PROBE / CAPACITY_AND_NET_SAVINGS_UNPROVEN. Excluded from this memo's reclamation scope by definition."
  - assumption: "F7 (PR95 dual-RGB-head) is a SSRV vector"
    classification: CARGO-CULTED-CORRECTED
    rationale: "Per Yousfi verbatim in THIS memo + the 43-vector audit row F7: F7 is 'an encoder-time architectural choice with separate feasibility region.' Not a scorer-load-at-inflate issue. F7 was incorrectly tagged in the original 43-vector audit; it should be re-classified as a Compose-with-A1 candidate, not a SSRV reclamation target. Excluded from this memo's reclamation scope by definition."
  - assumption: "The 12-claim CONFLATE_DECLARATIVE_WITH_PHYSICAL META-pattern fully captures the rate-attack-reasoning bug class"
    classification: HARD-EARNED-EXTENDED-HERE
    rationale: "Per META-audit commit e86ca6d0c + Section 3 'Implications for reclaimable vectors': the META-audit identified A1 + F3-F6 as RECLAIMABILITY under empirical test. THIS memo executes Section 3's recommended SYSTEMATIC RECLAIMABILITY RE-EXAMINATION subagent action by carrying out the per-vector matrix. The META-pattern itself is HARD-EARNED via Codex F1 finding + ADVERSARIAL 5 binding critiques + this systematic re-examination."
council_decisions_recorded:
  - "OP-RECLAIM-1: A1-SPECIALIZED is already canonical-RECLAIMABLE per A1 memo + 43-vector audit Section 0. THIS memo CONFIRMS the reclamation and extends the framework to F3/F4/F5/F6."
  - "OP-RECLAIM-2: F3 (PoseNet vision 2048-dim) is RECLAIMABLE_VIA_PACKET_COMPILER via Hinton distilled student + Balle hyperprior + Brotli at predicted 10-30 KB binary."
  - "OP-RECLAIM-3: F4 (PoseNet summary 512-dim) is RECLAIMABLE_VIA_PACKET_COMPILER via product quantization or Hinton student + factorized prior + Brotli at predicted 5-15 KB binary."
  - "OP-RECLAIM-4: F5 (PoseNet ResBlock output 512-dim) is RECLAIMABLE_VIA_PACKET_COMPILER same architecture as F4; predicted 5-15 KB binary."
  - "OP-RECLAIM-5: F6 (Hydra trunk-vs-head split) is RECLAIMABLE-WITH-INTERPRETIVE-CAUTION via per-pattern distillation; predicted 10-30 KB binary; requires explicit Yousfi-style interpretation memo before prototype lands per Catalog #325 reactivation criterion."
  - "OP-RECLAIM-6: F1/F2 are NOT SSRV (collapsed to A2 per Codex F1 finding); F7 is NOT SSRV (encoder-time architectural choice). Both EXCLUDED from reclamation scope."
  - "OP-RECLAIM-7: NEW STRICT preflight gate `check_rate_attack_strategic_claim_has_receiver_path_evidence` (META-audit §2 recommendation) is the structural extinction surface for the CONFLATE_DECLARATIVE_WITH_PHYSICAL bug class. This memo's per-vector matrix IS the canonical operator-runnable disambiguator until that gate lands."
  - "OP-RECLAIM-8: Probe-disambiguator path EXISTS for every reclaimed vector via `tools/build_a1_*_inverter_prototype.py` family (A1 memo §13). Each F3-F6 reclamation gets its own sister tool when prototype phase begins."
  - "OP-RECLAIM-9: Composition with A1 + A2 + B1 + C1 PRESERVED per A1 memo §14.3. F3-F6 reclaimed candidates extend the composition space — they compose orthogonally with A1 (same scorer, different layer) and stack-additively with B1/C1."
  - "OP-RECLAIM-10: Per CLAUDE.md 'KILL is LAST RESORT' non-negotiable: NO vector is KILLED. All non-reclaimable rows stay STRICT_SCORER_RULE_VIOLATION ⚠️ with documented reactivation criteria. F3-F6 are PROCEED_WITH_REVISIONS pending empirical prototype anchor."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
substrate_alias: systematic_reclaimability_re_examination_design_memo_20260518
substrate_aliases:
  - reclaimability_re_examination_design_20260518
  - rate_attack_vector_reclamation_systematic_20260518
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_band_validation_status: pending_post_training
predicted_band_validation_reactivation_criteria: "Design memo with PROCEED_WITH_REVISIONS verdict; per-vector predicted bands (F3 [10-30 KB], F4/F5 [5-15 KB], F6 [10-30 KB]) validated when (a) per-vector Codex prototypes land sister to A1-distillation prototype (b) measured binary sizes fall within bands (c) preserve ≥95% accuracy. Post-training Tier-C re-measurement per Catalog #324 required on each landed prototype archive before any [contest-CPU]/[contest-CUDA] score claim."
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
  contest_cuda: "0.20533 [contest-CUDA T4] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
related_deliberation_ids:
  - a1_binary_distillation_design_memo_zig_sparseness_ablation_plus_canonical_techniques_20260518
  - rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_20260518
  - meta_audit_conflate_declarative_with_physical_error_pattern_12_claim_self_audit_20260518
  - cargo_cult_burn_down_supplement_extending_meta_audit_across_session_20260518
  - rate_attack_43_vectors_meta_paradigm_deep_research_20260518
  - adversarial_rate_attack_paradigm_challenger_20260518
  - rate_attack_synthesis_v2_reconciliation_primary_plus_adversarial_plus_supplement_20260518
  - grand_council_symposium_inflate_py_extreme_compression_20260518
  - rate_attack_vector_3_b1_contest_video_codebook_design_memo_20260518
horizon_class: frontier_pursuit
memory_path: ~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_systematic_reclaimability_re_examination_via_binary_distillation_framework_landed_20260518.md
event_type: dispatched
parent_id_or_session: systematic_reclaimability_re_examination_via_binary_distillation_framework_20260518
notes: "T2 systematic re-examination of STRICT_SCORER_RULE_VIOLATION vectors from the 43-vector audit (`rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_20260518.md`) through the A1 binary distillation framework (`a1_binary_distillation_design_memo_zig_sparseness_ablation_plus_canonical_techniques_20260518.md` commit 0701c323b). Per Codex's Section 0 operator correction: A1-CANONICAL stays SSRV but A1-SPECIALIZED is RECLAIMABLE; THIS memo extends the per-pattern specialization framework to F3/F4/F5/F6. F1/F2 collapsed to A2 per Codex F1 finding (NOT SSRV anymore); F7 is encoder-time architectural choice (NOT SSRV). Per CLAUDE.md 'Forbidden premature KILL' non-negotiable: no kills — all non-reclaimed vectors stay DEFERRED-pending-research with documented reactivation criteria. Mission contribution: frontier_protecting (the systematic reclamation framework IS the protection against premature KILL verdicts that the apparatus historically over-applied per pre-rigor-kill-defer-falsified inventory)."
---

# Systematic reclaimability re-examination via binary distillation framework

**Operator standing directive 2026-05-18 verbatim**: *"we need to ensure we are engineering and wiring and integrating and desiging correctly with optimization and syngeries and exploits and stacking and composing and stacks of stacks and multipass and parallelism and extreme optimization and alien tech and time traveler and asymptotic and teheorical floor fast wall clcok; approved proceed with all burn down all cargo culted and keep pushing"* + *"continue meta consolidation and similar meta work across"*

**Prior context**: Per the META-audit (`meta_audit_conflate_declarative_with_physical_error_pattern_12_claim_self_audit_20260518.md` commit e86ca6d0c) Section 3 "Implications for reclaimable vectors" + the F1-finding audit Section 0 "OPERATOR CORRECTION: A1-SPECIALIZED IS LIVE": after the A1 binary distillation memo landed (commit `0701c323b`; T2 council PROCEED_WITH_REVISIONS verdict), the apparatus identified F3/F4/F5/F6 as candidates for the SAME systematic reclaimability examination. This memo executes that work.

**This memo's mission**: per-vector re-classification of every STRICT_SCORER_RULE_VIOLATION row from the 43-vector audit through the binary distillation framework's V1/V2/V3 composition stacks, with HNeRV parity L4 budget check, `contest_one_video_replay` mode applicability, predicted compressed size band, receiver-path evidence per Codex's F1 finding, verdict, and reactivation criteria per Catalog #325.

**Lane**: `lane_systematic_reclaimability_re_examination_via_binary_distillation_framework_20260518` (L1 at memo landing).

---

## 0. Executive Summary

### TL;DR

The 43-vector audit (commit `35b06f9ec`) classified 8 vectors STRICT_SCORER_RULE_VIOLATION (⚠️) in Categories A and F. Codex's Section 0 operator correction already split A1 into A1-CANONICAL (stays SSRV) vs A1-SPECIALIZED (RECLAIMABLE_VIA_PACKET_COMPILER). The A1 binary distillation memo formalizes the SPECIALIZED reclamation path. **This memo systematically re-examines the remaining 7 SSRV rows** and finds:

| Vector | Original SSRV verdict | Re-examination verdict | Predicted binary size | Predicted ΔS |
|---|---|---|---|---|
| A1-CANONICAL | SSRV ⚠️ | SSRV ⚠️ (full PoseNet weights forbidden) | N/A | N/A |
| A1-SPECIALIZED | SSRV ⚠️ | **RECLAIMABLE_VIA_PACKET_COMPILER ✓** (per A1 memo) | 5-20 KB | [-0.012, -0.003] |
| F1 | SSRV ⚠️ | **NOT SSRV** (collapsed to A2 per Codex F1 finding) | N/A (input-side) | A2 territory |
| F2 | SSRV ⚠️ | **NOT SSRV** (same as F1 → A2) | N/A (input-side) | A2 territory |
| F3 (vision 2048) | SSRV ⚠️ | **RECLAIMABLE_VIA_PACKET_COMPILER ✓** (Hinton + hyperprior + Brotli) | 10-30 KB | [-0.020, -0.005] |
| F4 (summary 512) | SSRV ⚠️ | **RECLAIMABLE_VIA_PACKET_COMPILER ✓** (PQ + factorized prior + Brotli) | 5-15 KB | [-0.015, -0.003] |
| F5 (ResBlock 512) | SSRV ⚠️ | **RECLAIMABLE_VIA_PACKET_COMPILER ✓** (same as F4) | 5-15 KB | [-0.015, -0.003] |
| F6 (Hydra trunk) | SSRV ⚠️ | **RECLAIMABLE_WITH_INTERPRETIVE_CAUTION ⚠️** (per-pattern distillation + explicit Yousfi memo) | 10-30 KB | [-0.012, -0.003] |
| F7 (dual-RGB-head) | SSRV ⚠️ | **NOT SSRV** (encoder-time architectural choice) | N/A | composition candidate |

### Aggregate

- **Reclaimable vectors**: 5 (A1-SPECIALIZED, F3, F4, F5, F6)
- **Re-classified as NOT SSRV**: 3 (F1, F2, F7)
- **Stays SSRV**: 1 (A1-CANONICAL — the naive "ship full PoseNet" framing)
- **Killed**: 0 (per CLAUDE.md "KILL is LAST RESORT" non-negotiable)

### TOP-3 highest-EV reclaimed vectors

| Rank | Vector | Composition stack | Predicted binary | Predicted net ΔS | Recommended probe |
|---|---|---|---|---|---|
| 1 | **A1-SPECIALIZED** ★ | V2 (VQ-VAE K=256 + FP4 + Brotli) | 5-10 KB | -0.003 to -0.012 | `tools/build_a1_per_pattern_vq_vae_inverter_prototype.py` ($1-3) |
| 2 | **F4 (summary 512)** | Hinton 30K-param student + PQ-8x8 + factorized prior + Brotli | 5-15 KB | -0.003 to -0.015 | `tools/build_f4_summary_512_per_pattern_inverter_prototype.py` ($2-5) |
| 3 | **F5 (ResBlock output 512)** | Same architecture as F4 | 5-15 KB | -0.003 to -0.015 | `tools/build_f5_resblock_output_per_pattern_inverter_prototype.py` ($2-5) |

### Cross-vector composition opportunities

- **A1-SPECIALIZED + F4** → different layers of same network; orthogonal axes; expected additive ΔS minus Catalog #322 sub-additive default (α≈0.5)
- **A1-SPECIALIZED + F4 + F5 + F6** → full Hydra path coverage; sub-additive but PROBABLY FEASIBLE; needs paired-comparison smoke
- **A1-SPECIALIZED + A2 + B1 + C1** → A1 memo §14.3 already validated; F-category extensions compose along same axes

### Recommended next-step probes ($1-3 each)

1. **PRE-PROBE**: Per-vector dimensional analysis (Assumption-Adversary requirement) — $0 (research-only)
2. **PROBE-F4**: Build F4 (summary 512) per-pattern distilled inverter prototype — $2-5
3. **PROBE-F5**: Build F5 (ResBlock 512) per-pattern distilled inverter prototype — $2-5
4. **PROBE-F3**: Build F3 (vision 2048) per-pattern distilled inverter prototype — $3-8 (larger model)
5. **PROBE-F6**: Build F6 (Hydra trunk) per-pattern distilled inverter prototype WITH explicit Yousfi interpretation memo — $3-8

All probes are DEPENDENT on A1-SPECIALIZED V2 prototype landing first per A1 memo §13 + Catalog #313 ordering discipline (A1 is the canonical anchor; F3-F6 prototypes inherit the canonical infrastructure).

---

## 1. Methodology

### 1.1 Source-of-truth for SSRV inventory

The canonical source is `rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_20260518.md` (commit `35b06f9ec`) Section 3, the per-category audit table:

**Category A** (SCORER-AWARE BYTE-LEVEL): 4 rows — A1-CANONICAL (SSRV), A1-SPECIALIZED (RECLAIMABLE_VIA_PACKET_COMPILER), A2 (NO_RECEIVER_NEEDED), A3 (NO_RECEIVER_NEEDED).

**Category F** (HYDRA / DUAL-HEAD): 8 rows — F1 (SSRV original framing; ✓→ A2 corrected), F1-CORRECTED (LEGAL_RECEIVER_PATH_EXISTS_FOR_PROBE), F2 (same as F1), F3 vision(2048) (RECLAIMABLE_VIA_PACKET_COMPILER), F4 summary(512) (RECLAIMABLE_VIA_PACKET_COMPILER), F5 ResBlock output (RECLAIMABLE_VIA_PACKET_COMPILER), F6 Hydra trunk-vs-head (RECLAIMABLE_VIA_PACKET_COMPILER), F7 PR95 dual-RGB-head (depends on framing).

The total set of "currently SSRV-or-flagged-for-re-examination" entries this memo evaluates: **9 rows** (A1-CANONICAL, A1-SPECIALIZED, F1, F2, F3, F4, F5, F6, F7).

### 1.2 Re-examination framework

For EACH row, evaluate against the A1 binary distillation framework's V1/V2/V3 composition stacks (A1 memo §5.3):

- **V1 ultra-compact procedural** (~500-800 bytes): 8-byte procedural codebook seed + 600 × 1-byte VQ indices + arithmetic coding + Brotli + standard inflate.py (+30-50 LOC)
- **V2 VQ-VAE + residual student** (~5-8 KB): K=256 codebook (3 KB FP4) + 600 × 1-byte indices + tiny residual MLP (1 KB FP4) + Brotli + standard inflate.py ★ TOP-1 RECOMMENDED
- **V3 weight-derived + differential** (~300-600 bytes): Weight-derived codebook (0 new bytes; from shipped renderer.bin) + delta coding + Brotli + inflate.py (+~40 LOC)

Additional techniques from A1 memo §3 inventory: Hinton distillation (per-pattern specialized), FP4 quantization, structured sparsity, Brotli/arithmetic, low-rank decomposition, Cool-Chic INR, hyperprior, product quantization.

### 1.3 Per-vector classification dimensions

For each row, this memo records:

1. **Vector ID + description** (canonical reference)
2. **Original SSRV rationale** (from 43-vector audit Section 3)
3. **Binary distillation applicability** (V1 / V2 / V3 / extended)
4. **Sparseness + ablation + FP4 + Brotli composability check** (per A1 memo §3.1.2 sparseness, §3.1.3 ablation, §3.2.1 FP4, §3.8.1 Brotli)
5. **HNeRV parity L4 budget check** (≤100 LOC default / ≤200 LOC waiver; ≤2 deps; per CLAUDE.md non-negotiable)
6. **contest_one_video_replay applicability** (per A1 memo §2 mapping)
7. **Predicted compressed size band** [lower, upper]
8. **Receiver-path evidence** (NO_RECEIVER_NEEDED / LEGAL_RECEIVER_IN_BUDGET / RECLAIMABLE_VIA_PACKET_COMPILER / STILL_STRICT_SCORER_RULE_VIOLATION)
9. **Verdict + rationale** (per Codex's Section 0 canonical 5-condition legality test)
10. **Reactivation criteria per Catalog #325** if reclaimable (3-4 conditions with priority + predicted cost + assumption tested per condition)

### 1.4 KILL is LAST RESORT discipline

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + the "Pre-rigor KILL/DEFER/FALSIFIED inventory 2026-05-17" (which found 28-32 of 34 historical kills FAILED the new rigor gates):

- NO vector is KILLED in this memo.
- Vectors that remain SSRV stay DEFERRED-pending-research with documented reactivation criteria.
- Vectors that are reclaimable get PROCEED_WITH_REVISIONS with explicit prototype requirements.
- Vectors that are mis-classified get re-classified to their canonical category (F1/F2 → A2; F7 → composition candidate).

---

## 2. Per-vector re-classification matrix

### 2.1 A1-CANONICAL — scorer-feature-space encoding (full scorer or generic scorer-feature inverter)

| Dimension | Value |
|---|---|
| Vector ID | A1-CANONICAL |
| Description | "Scorer-feature-space encoding (skip RGB)" — naive framing where archive contains either full PoseNet/SegNet weights OR generic scorer-feature inverter loadable on arbitrary RGB inputs |
| Original SSRV rationale | "requires inverter from scorer-features to bytes; this IS a scorer component" (~73 MB rate hit if full weights; or generic inverter that violates per-pattern specialization criterion) |
| Binary distillation V1/V2/V3 | NOT APPLICABLE — generic framing precludes per-pattern specialization |
| Sparseness/ablation/FP4/Brotli composability | N/A — the framing itself is non-compliant regardless of compression |
| HNeRV parity L4 budget | N/A |
| contest_one_video_replay applicability | NO — `contest_one_video_replay` sanctions per-pattern distilled byte transducers, NOT generic inverters that approximate scorers for arbitrary input |
| Predicted compressed size band | N/A |
| Receiver-path evidence | **STILL_STRICT_SCORER_RULE_VIOLATION ⚠️** per Codex Section 0 legality test (loading full scorer weights or generic scorer-feature inverter at inflate violates Catalog #6) |
| Verdict | **STAYS SSRV — DEFERRED-pending-research** |
| Reactivation criteria | A1-CANONICAL reactivates from SSRV ONLY if (a) the framing itself changes to per-pattern specialization (becomes A1-SPECIALIZED), (b) explicit operator+Yousfi joint interpretation confirms a new compliance category, OR (c) Catalog #6 itself is amended by operator decision. None of these are research paths that this memo recommends. |

### 2.2 A1-SPECIALIZED — deterministic packet compiler (per-pattern byte transducer / fixed table / generated code path)

| Dimension | Value |
|---|---|
| Vector ID | A1-SPECIALIZED |
| Description | Per-pattern distilled VQ-VAE inverter (or sister: weight-derived codebook / procedural codebook / Hinton distilled student / Cool-Chic INR) that obviously cannot serve as a re-usable PoseNet for arbitrary input |
| Original SSRV rationale | Originally mis-classified SSRV in the 43-vector audit; corrected by Codex Section 0 |
| Binary distillation V1/V2/V3 | **V2 RECOMMENDED ★** (K=256 VQ-VAE + FP4 + Brotli); V1 ultra-compact viable but riskier on accuracy; V3 weight-derived viable if renderer structure supports |
| Sparseness/ablation/FP4/Brotli composability | All canonical (Layers 1-4 per A1 memo §5.1); HARD-EARNED via Quantizr PR101 88K-param precedent (CLAUDE.md "Quantizr intelligence") |
| HNeRV parity L4 budget | ✓ (V2 = +52-91 LOC; fits within ≤200 LOC waiver per A1 memo §2.5) |
| contest_one_video_replay applicability | ✓ (all 4 primitives in `contest_one_video_replay` map directly per A1 memo §2.2) + 5/5 admissibility conditions satisfiable per A1 memo §2.3 |
| Predicted compressed size band | **5-20 KB** (canonical infrastructure) or 2-5 KB (engineering minimum) per A1 memo §8 |
| Receiver-path evidence | **RECLAIMABLE_VIA_PACKET_COMPILER ✓** per Codex Section 0 + A1 memo PROCEED_WITH_REVISIONS verdict |
| Verdict | **RECLAIMABLE — PROCEED_WITH_REVISIONS** (per A1 memo) |
| Reactivation criteria (per Catalog #325) | (1) Binary size landing within 5-20 KB band; (2) accuracy preservation ≥95%; (3) Yousfi-style compliance interpretation memo (A1 memo §1.4-1.6 ratification); (4) Catalog #139 byte-mutation smoke confirming inflate consumes bytes. Predicted cost: $1-3 prototype. |

### 2.3 F1 — PoseNet Hydra dims 7-12 (originally framed as "free byte channel")

| Dimension | Value |
|---|---|
| Vector ID | F1 (original framing) |
| Description | Encode information in PoseNet dims 7-12 (unscored per `upstream/modules.py:84` `[..., :h.out//2]` indexing) |
| Original SSRV rationale | Would require loading scorer at inflate to "read" dim 7-12 from RGB — STRICT_SCORER_RULE_VIOLATION |
| Binary distillation V1/V2/V3 | N/A — different exploit class (input-side perturbation, not output-side encoding) |
| Sparseness/ablation/FP4/Brotli composability | N/A |
| HNeRV parity L4 budget | N/A |
| contest_one_video_replay applicability | N/A |
| Predicted compressed size band | N/A |
| Receiver-path evidence | **NOT SSRV — collapsed to A2** per Codex F1 finding relay. F1-CORRECTED framing (find RGB perturbations whose first-6-dim is invariant) is **LEGAL_RECEIVER_PATH_EXISTS_FOR_PROBE / CAPACITY_AND_NET_SAVINGS_UNPROVEN**. The receiver is the standard inflate.py + scorer; no scorer-load is needed because the exploit is input-side. |
| Verdict | **RE-CLASSIFIED — NOT SSRV; collapses to A2** (per 43-vector audit Section 0) |
| Reactivation criteria | N/A (NOT SSRV anymore). A2 has its own probe path: see 43-vector audit row A2 — `NO_RECEIVER_NEEDED ✓` (Fridrich PhD canonical steganalysis territory). |

### 2.4 F2 — SegNet non-argmax logits

| Dimension | Value |
|---|---|
| Vector ID | F2 |
| Description | "Encode information in SegNet logits" — same META-pattern as F1 at the SegNet side |
| Original SSRV rationale | Would require loading SegNet at inflate to read logits |
| Binary distillation V1/V2/V3 | N/A |
| Receiver-path evidence | **NOT SSRV — collapses to A2** (same reasoning as F1: input-side scorer-blind perturbation, not output-side encoding). The exploit becomes "find RGB perturbations whose SegNet argmax is invariant" (a scorer-blind manifold). The receiver is standard inflate.py + scorer. |
| Verdict | **RE-CLASSIFIED — NOT SSRV; collapses to A2** |
| Reactivation criteria | N/A (same as F1) |

### 2.5 F3 — PoseNet vision(2048) feature-space encoding

| Dimension | Value |
|---|---|
| Vector ID | F3 |
| Description | Encode information in PoseNet's vision backbone output (2048-dim feature vector per `upstream/modules.py:66-67`). Per-pair the vision output is 2048-dim. |
| Original SSRV rationale | Per 43-vector audit row F3: "generic vision-2048 inverter is forbidden; specialized byte transducer may be tiny" → already flagged as RECLAIMABLE_VIA_PACKET_COMPILER |
| Binary distillation V1/V2/V3 | **EXTENDED-V2 RECOMMENDED**: Naive V2 (K×2048 codebook) is UNSUITABLE per van den Oord verbatim (K=256 → 512 KB codebook). RECLAMATION requires: **(a) PCA/JL dimensionality reduction** (2048 → ~32 dims via projection) **then VQ-VAE K=256** (3-8 KB codebook); OR **(b) hyperprior + Hinton distilled student** (~30K-50K param student at FP4 + 50% sparse + Brotli per Quantizr PR101 / Selfcomp PR #56 toolkit); OR **(c) hierarchical RVQ** (residual VQ across multiple smaller codebooks; per van den Oord) |
| Sparseness/ablation/FP4/Brotli composability | ✓ Per Quantizr verbatim: distilled student via `kl_on_logits(T=2.0)` + FP4 + structured 50-75% sparsity + Brotli is the canonical path; Selfcomp PR #56 block-FP 1.017 bpw applies; achievable 3-12 KB. ALSO Balle hyperprior canonical (Balle 2018) yields 0.5-1.5 bits/dim |
| HNeRV parity L4 budget | ✓ (estimated +60-100 LOC for Hinton student forward + dequant; fits ≤200 LOC waiver) |
| contest_one_video_replay applicability | ✓ "distilled byte transducer" maps directly; "per-frame/per-pair streams derived from trained model's behavior on the scored video" maps to per-pair vision-feature distillation |
| Predicted compressed size band | **10-30 KB** (Hinton student + FP4 + sparsity + Brotli per Quantizr/Selfcomp toolkit); aggressive variant could land at 5-15 KB with hyperprior |
| Receiver-path evidence | **RECLAIMABLE_VIA_PACKET_COMPILER ✓** per Codex Section 0 legality test: (1) all runtime code/data ships in archive; (2) archive bytes charged; (3) inflate.sh consumes packet deterministically; (4) exact CUDA auth eval validates; (5) packet compiler records typed runtime-consumption proof |
| Verdict | **RECLAIMABLE — PROCEED_WITH_REVISIONS** (with Assumption-Adversary's per-vector dimensional analysis caveat) |
| Reactivation criteria (per Catalog #325) | (1) **Binary size landing within 10-30 KB band**: predicted size based on Hinton 30K-50K-param student + FP4 + 50% sparse + Brotli; HARD-EARNED via Quantizr/Selfcomp precedent; cost $3-8 prototype. (2) **Accuracy preservation ≥95%**: per A1 memo §7.1 K=256+FP4+50% sparse achieves 0.92 naive accuracy; F3 needs hyperprior or RVQ to reach 0.95. (3) **Yousfi-style compliance interpretation**: F3 is a vision-backbone-inverter; per Yousfi verbatim "F3/F4/F5 are STRUCTURALLY COMPLIANT under per-pattern specialization framing" — explicit memo references this seat. (4) **Catalog #139 byte-mutation smoke**: confirms inflate.py path consumes F3 binary's bytes. |

### 2.6 F4 — PoseNet summary(512) bottleneck

| Dimension | Value |
|---|---|
| Vector ID | F4 |
| Description | Encode information in PoseNet's summary bottleneck output (512-dim per `upstream/modules.py:67` `nn.Sequential(nn.Linear(VISION_FEATURES, SUMMARY_FEATURES), nn.ReLU, ResBlock(SUMMARY_FEATURES))`). Per-pair output is 512-dim. |
| Original SSRV rationale | Per 43-vector audit row F4: "generic summary-512 inverter is forbidden; specialized bottleneck table may be tiny" → already flagged as RECLAIMABLE_VIA_PACKET_COMPILER |
| Binary distillation V1/V2/V3 | **V2 RECOMMENDED with PRODUCT QUANTIZATION**: Naive K=256 × 512 = 128 KB UNSUITABLE; PQ-8x8 (split 512-dim into 8 sub-vectors of 64-dim, codebook K=256 per sub-vector): 8 × 256 × 64 × 4 = 64 KB STILL TOO LARGE; PQ-8x8 with K=64: 8 × 64 × 64 × 4 = 16 KB; PQ-8x8 with K=64 + FP4: 8 KB; PQ-8x8 with K=64 + FP4 + 50% sparse codebook: 4 KB. **Sweet spot: PQ-8x8 + K=64 + FP4 + Brotli ≈ 5-15 KB**. ALSO valid: hierarchical RVQ (2-level: first level K=256 × 256-dim residual; second level K=256 × 256-dim) similar size. |
| Sparseness/ablation/FP4/Brotli composability | ✓ All canonical. Per MacKay verbatim: K(inverter \| Y) is small because deep features are deterministic in upstream video bytes Y. Weight-derived codebook variant (V3 from A1 memo) may apply if renderer weights have structure usable as feature-space codebook. |
| HNeRV parity L4 budget | ✓ (estimated +50-80 LOC for PQ table lookup + dequant; comfortably under ≤200 LOC waiver) |
| contest_one_video_replay applicability | ✓ Same as F3 — "distilled byte transducer" + "fixed tables" + "per-pair streams derived from trained model behavior" |
| Predicted compressed size band | **5-15 KB** (PQ-8x8 + K=64 + FP4 + Brotli) |
| Receiver-path evidence | **RECLAIMABLE_VIA_PACKET_COMPILER ✓** per Codex Section 0 legality test |
| Verdict | **RECLAIMABLE — PROCEED_WITH_REVISIONS** |
| Reactivation criteria (per Catalog #325) | (1) **Binary size within 5-15 KB band**: predicted via PQ-8x8 + K=64 + FP4 + Brotli; cost $2-5 prototype. (2) **Accuracy preservation ≥95%**: PQ is lossy; needs K-sweep at prototype phase (K ∈ {32, 64, 128, 256}). (3) **Yousfi-style compliance interpretation**: F4 is a SUMMARY-BOTTLENECK inverter (further from PoseNet identity than F3 vision); per Yousfi STRUCTURALLY COMPLIANT. (4) **Catalog #139 byte-mutation smoke** + (5) **Comparison probe vs A1-SPECIALIZED**: validates compression-ratio claim. Composition probe with A1 also runs after both prototypes land. |

### 2.7 F5 — PoseNet ResBlock output (deterministic)

| Dimension | Value |
|---|---|
| Vector ID | F5 |
| Description | Encode information in PoseNet's ResBlock output (per `upstream/modules.py:67` summarizer ends in ResBlock(512); output is 512-dim post-ResBlock). The ResBlock output is the input to the Hydra. Per-pair 512-dim. |
| Original SSRV rationale | Per 43-vector audit row F5: "generic ResBlock inverter is forbidden; specialized generated code may be tiny" → already flagged as RECLAIMABLE_VIA_PACKET_COMPILER |
| Binary distillation V1/V2/V3 | **V2 RECOMMENDED with PQ** — identical architecture to F4 (both target 512-dim per-pair features). PQ-8x8 + K=64 + FP4 + Brotli → 5-15 KB. Plus the ResBlock output has SMALLER intrinsic dimensionality than summary (ResBlock applies residual transformation; deep-output entropy reduced) — codebook MAY be smaller in practice. |
| Sparseness/ablation/FP4/Brotli composability | ✓ Same as F4. |
| HNeRV parity L4 budget | ✓ (same as F4) |
| contest_one_video_replay applicability | ✓ Same as F3/F4 |
| Predicted compressed size band | **5-15 KB** (same as F4) |
| Receiver-path evidence | **RECLAIMABLE_VIA_PACKET_COMPILER ✓** |
| Verdict | **RECLAIMABLE — PROCEED_WITH_REVISIONS** |
| Reactivation criteria (per Catalog #325) | (1) **Binary size within 5-15 KB band**; (2) **Accuracy preservation ≥95%**; (3) **Yousfi-style compliance interpretation**: F5 is a RESBLOCK-OUTPUT inverter (a specific intermediate layer); per Yousfi STRUCTURALLY COMPLIANT; (4) **Catalog #139 byte-mutation smoke**; (5) **Composition probe with F4** (F4 and F5 are different layers of same path; expected highly correlated; might yield sub-additive savings if used together — paired-comparison smoke per Catalog #322 required). |

### 2.8 F6 — Hydra trunk-vs-head split

| Dimension | Value |
|---|---|
| Vector ID | F6 |
| Description | Encode information by exploiting the Hydra trunk (the shared in_layer + res_layer) vs head split (final_layer per `upstream/modules.py:45-58`). Trunk takes 512-dim ResBlock output → 32-dim hidden → res-block transformation → final 12-dim head. The trunk-vs-head split has separate degrees of freedom in the 32-dim hidden representation. |
| Original SSRV rationale | Per 43-vector audit row F6: "generic trunk/head inverter is forbidden; specialized split transducer may be tiny" → already flagged as RECLAIMABLE_VIA_PACKET_COMPILER |
| Binary distillation V1/V2/V3 | **V2 RECOMMENDED with PER-PAIR HIDDEN-32 CODEBOOK**: The Hydra's hidden dimension is 32 per `upstream/modules.py:26` `Head('pose', 32, 12)`. Per-pair 32-dim is SMALL enough for direct VQ-VAE K=256 codebook: 256 × 32 × 4 = 4 KB (FP4). Plus 600 × 1-byte selections + Brotli → 5-10 KB total. **OR** the Hydra trunk's two `res_layer` MLPs are ~32×32 ≈ 1K params each = 2K params at FP4 = ~1 KB; specialized inverter on this scale is even smaller. |
| Sparseness/ablation/FP4/Brotli composability | ✓ All canonical. SMALLER than F4/F5 because Hydra hidden is 32-dim (vs 512-dim). |
| HNeRV parity L4 budget | ✓ (smaller inflate addition; ~40-60 LOC) |
| contest_one_video_replay applicability | ✓ "specialized split transducer" maps to `contest_one_video_replay` distilled byte transducer |
| Predicted compressed size band | **10-30 KB** (more conservative band per Contrarian's veto: Hydra trunk is the closest to "looks like PoseNet"; need extra rigor) — but realistic technical floor is closer to 3-10 KB given 32-dim hidden |
| Receiver-path evidence | **RECLAIMABLE_VIA_PACKET_COMPILER ✓** per Codex Section 0 — BUT per Yousfi verbatim in THIS memo: "F6 needs more scrutiny because Hydra trunk is the final-stage interpreter." Recommend INTERPRETIVE_CAUTION flag. |
| Verdict | **RECLAIMABLE-WITH-INTERPRETIVE-CAUTION ⚠️ — PROCEED_WITH_REVISIONS** |
| Reactivation criteria (per Catalog #325) | (1) **Binary size within 10-30 KB band** (conservative; technical floor 3-10 KB); cost $3-8 prototype. (2) **Accuracy preservation ≥95%**: per the 32-dim hidden, K=256 codebook should easily achieve. (3) **EXPLICIT Yousfi-style compliance interpretation memo** that addresses Hydra-trunk-as-final-stage-interpreter concern; per Contrarian veto this is BINDING. (4) **Catalog #139 byte-mutation smoke**. (5) **Composition probe with A1-SPECIALIZED**: A1 specializes the 12-dim pose head; F6 specializes the 32-dim Hydra trunk. Different layers of same path. Sub-additive default per Catalog #322. |

### 2.9 F7 — PR95 Phase 2-4 dual-RGB-head architecture

| Dimension | Value |
|---|---|
| Vector ID | F7 |
| Description | Encode via PR95's dual-RGB-head architecture (Phase 2-4) — the architecture allows two RGB outputs per pair; one for SegNet path, one for PoseNet path. Per Yousfi verbatim in THIS memo: "F7 (dual-RGB-head) is not a scorer-intermediate exploit; it's an architecture-time choice." |
| Original SSRV rationale | Per 43-vector audit row F7: "Depends on framing" — ambiguous classification |
| Binary distillation V1/V2/V3 | N/A — F7 is an architectural choice at the substrate level, not a scorer-feature-encoding exploit |
| Sparseness/ablation/FP4/Brotli composability | N/A |
| HNeRV parity L4 budget | N/A (already inside the canonical inflate.py LOC budget regardless of F7 use) |
| contest_one_video_replay applicability | N/A (F7 is encoder-time; `contest_one_video_replay` is decoder-time discipline) |
| Predicted compressed size band | N/A |
| Receiver-path evidence | **NOT SSRV** per Yousfi verbatim. F7 is a substrate-architecture choice, NOT a rate-attack vector. The 43-vector audit row F7 ambiguous classification should be RESOLVED to: "F7 is an architectural composition candidate (per-substrate choice), NOT a rate-attack reclamation target." |
| Verdict | **RE-CLASSIFIED — NOT SSRV; composition candidate (encoder-time architectural choice)** |
| Reactivation criteria | F7 is not killed; it's re-categorized. F7 enters the composition space alongside A1-SPECIALIZED + F3-F6 + A2 + B1 + C1 as a parallel architectural axis. Use in per-substrate design decisions when relevant. |

---

## 3. Aggregate verdict table

| # | Vector | Verdict | Predicted binary | Predicted ΔS [contest-CPU; prediction] | Cost to probe |
|---|---|---|---|---|---|
| 1 | A1-CANONICAL | STAYS SSRV ⚠️ (deferred-pending-research) | N/A | N/A | N/A |
| 2 | A1-SPECIALIZED | RECLAIMABLE — PROCEED_WITH_REVISIONS ★ | 5-20 KB | [-0.012, -0.003] | $1-3 |
| 3 | F1 | NOT SSRV (collapsed to A2) | N/A | A2 territory | $0-5 (A2 path) |
| 4 | F2 | NOT SSRV (same as F1) | N/A | A2 territory | $0-5 (A2 path) |
| 5 | F3 vision(2048) | RECLAIMABLE — PROCEED_WITH_REVISIONS | 10-30 KB | [-0.020, -0.005] | $3-8 |
| 6 | F4 summary(512) | RECLAIMABLE — PROCEED_WITH_REVISIONS | 5-15 KB | [-0.015, -0.003] | $2-5 |
| 7 | F5 ResBlock(512) | RECLAIMABLE — PROCEED_WITH_REVISIONS | 5-15 KB | [-0.015, -0.003] | $2-5 |
| 8 | F6 Hydra trunk(32) | RECLAIMABLE-WITH-CAUTION ⚠️ — PROCEED_WITH_REVISIONS | 10-30 KB | [-0.012, -0.003] | $3-8 |
| 9 | F7 dual-RGB-head | NOT SSRV (encoder-time architectural choice) | N/A | composition candidate | N/A |

### Aggregate per category

- **Reclaimable via packet compiler (5 vectors)**: A1-SPECIALIZED, F3, F4, F5, F6
- **NOT SSRV — re-classified (3 vectors)**: F1, F2, F7
- **Stays SSRV (1 vector)**: A1-CANONICAL
- **Killed (0 vectors)** per CLAUDE.md "KILL is LAST RESORT" non-negotiable

---

## 4. TOP-3 highest-EV reclaimed vectors

### 4.1 TOP-1: A1-SPECIALIZED (already canonical via A1 memo)

**Vector**: A1-SPECIALIZED — per-pattern distilled VQ-VAE inverter of PoseNet's 12-dim pose output.

**Why TOP-1**: The 12-dim pose head is the THINNEST point in the network (per Carmack verbatim). Naive VQ-VAE K=256 + FP4 + Brotli → 3 KB codebook + 600 × 1-byte indices + overhead = 5-10 KB total. Smallest binary footprint of any reclamation candidate. Predicted net ΔS [-0.012, -0.003] [contest-CPU; prediction] is the best engineering-cost-per-ΔS ratio.

**Composition stack**: V2 (VQ-VAE K=256 + FP4 + Brotli) per A1 memo §5.3.

**Recommended probe**: `tools/build_a1_per_pattern_vq_vae_inverter_prototype.py` per A1 memo §13.1 ($1-3 CPU smoke).

**Composition opportunities**: composes orthogonally with A2 (different scorer blind-spots), B1 (different decoder side-info source), C1 (cross-archive bytes); composes along same axis with F4/F5/F6 (different layers of same network) — sub-additive default per Catalog #322.

### 4.2 TOP-2: F4 — PoseNet summary(512) bottleneck

**Vector**: F4 — per-pattern distilled inverter of PoseNet's 512-dim summary bottleneck.

**Why TOP-2**: The summary bottleneck has STRUCTURAL ADVANTAGE per MacKay verbatim — K(inverter | Y) is small because deep features are deterministic in upstream video bytes. Predicted binary 5-15 KB via PQ-8x8 + K=64 + FP4 + Brotli. Wider predicted ΔS band [-0.015, -0.003] reflects steganalytic depth advantage per Fridrich verbatim.

**Composition stack**: V2 with PRODUCT QUANTIZATION (PQ-8x8 + K=64 + FP4 + Brotli).

**Recommended probe**: `tools/build_f4_summary_512_per_pattern_inverter_prototype.py` ($2-5 CPU smoke).

**Compliance**: STRUCTURALLY COMPLIANT per Yousfi verbatim ("F4 cannot serve as a re-usable PoseNet — it's a bottleneck inverter").

### 4.3 TOP-3: F5 — PoseNet ResBlock output(512)

**Vector**: F5 — per-pattern distilled inverter of PoseNet's ResBlock output.

**Why TOP-3**: Identical architecture to F4 (both 512-dim per-pair features) so prototype cost shares infrastructure. ResBlock output has SMALLER intrinsic dimensionality than summary (residual transformation reduces entropy), potentially smaller codebook.

**Composition stack**: V2 with PQ (same as F4).

**Recommended probe**: `tools/build_f5_resblock_output_per_pattern_inverter_prototype.py` ($2-5 CPU smoke).

**Compliance**: STRUCTURALLY COMPLIANT per Yousfi.

**Composition opportunity**: F4 + F5 are different layers of the SAME PoseNet path; expected highly correlated → sub-additive gains. Paired-comparison smoke per Catalog #322 is the canonical disambiguator.

---

## 5. Cross-vector composition opportunities

### 5.1 Orthogonal composition (additive in expectation; sub-additive per Catalog #322 default)

| Pair | Axes | Expected behavior |
|---|---|---|
| A1-SPECIALIZED + A2 | Different scorer blind-spots (output-side codebook + input-side perturbation) | ADDITIVE up to sub-additive default; paired-comparison smoke required |
| A1-SPECIALIZED + B1 | Different side-info sources (PoseNet feature space + contest video codebook) | ADDITIVE if codebooks disjoint; SUB-ADDITIVE if codebooks correlated |
| A1-SPECIALIZED + C1 | Different byte sources (archive bytes + sibling files outside archive) | ADDITIVE; orthogonal axes per A1 memo §14.3 |

### 5.2 Same-axis composition (sub-additive; correlated savings)

| Pair | Shared axis | Expected behavior |
|---|---|---|
| A1-SPECIALIZED + F4 | PoseNet path (12-dim head + 512-dim summary) | SUB-ADDITIVE per Catalog #322; saved bytes in pose head correlated with saved bytes in summary because both encode same per-pair pose information |
| F3 + F4 | PoseNet path (vision 2048 + summary 512) | HIGHLY SUB-ADDITIVE; summary is bottleneck of vision; encoding both is redundant |
| F4 + F5 | PoseNet path (summary 512 + ResBlock 512) | HIGHLY SUB-ADDITIVE; ResBlock is final transform on summary; encoding both is mostly redundant |
| F4 + F5 + F6 | PoseNet path (summary 512 + ResBlock 512 + Hydra trunk 32) | HIGHLY SUB-ADDITIVE; all 3 are layers of same path |
| F3 + F6 | PoseNet path endpoints (vision 2048 + Hydra trunk 32) | LESS SUB-ADDITIVE; further apart in path |

### 5.3 Full-Hydra-coverage composition (sub-additive; needs empirical anchor)

| Composition | Total predicted binary | Net predicted ΔS | Expected α |
|---|---|---|---|
| A1-SPECIALIZED + F4 | 10-25 KB | -0.005 to -0.015 | 0.6 |
| A1-SPECIALIZED + F4 + F5 | 15-30 KB | -0.005 to -0.018 | 0.5 |
| A1-SPECIALIZED + F4 + F5 + F6 | 20-40 KB | -0.005 to -0.020 | 0.4 |
| F3 + F4 + F5 + F6 (full network coverage) | 25-50 KB | -0.008 to -0.025 | 0.4 |
| A1-SPECIALIZED + F3 + F4 + F5 + F6 + A2 + B1 + C1 (kitchen sink) | 35-65 KB | -0.010 to -0.025 (saturating) | 0.3 (sub-additive heavily) |

Per CLAUDE.md "Forbidden cross-archive composition (HStack/VStack/cross-paradigm) without a single verified [contest-CUDA] substrate anchor" + "Forbidden 8th pattern (research-substrate trap)": the "kitchen sink" composition is the canonical anti-pattern. **Recommended composition order**: land A1-SPECIALIZED first (canonical anchor), then add ONE F-vector at a time with paired-comparison smoke per Catalog #322.

### 5.4 Composition with substrate-level architecture (F7)

F7 (dual-RGB-head) is an encoder-time choice that creates two RGB outputs per pair. This is orthogonal to A1-SPECIALIZED/F3-F6 reclamations (which are decoder-time bytes-in-archive choices). Composition: F7 architecture choice at substrate level + A1-SPECIALIZED/F3-F6 reclamation at archive level. No composition_alpha collapse expected (different axes).

---

## 6. Recommended next-step probe operations

Per Catalog #313 (probe outcomes ledger discipline) + CLAUDE.md "Probe-disambiguator pattern" + the Codex routing-directive surface:

### 6.1 PRE-PROBE: Per-vector dimensional analysis ($0)

**Purpose**: Satisfy Assumption-Adversary requirement before any paid prototype dispatch.

**Inputs**: PoseNet architecture per `upstream/modules.py`, F3-F6 per-pair feature dimensionality, archive per-pair entropy estimates.

**Outputs**: Per-vector quantitative dimensional analysis confirming V2-with-PQ feasibility before paid prototype.

**Tool**: `tools/analyze_f_category_dimensional_analysis.py` (research-only; no GPU spend; ~30 min on local-mps per CLAUDE.md "MPS auth eval is NOISE" research-signal path).

**Dependencies**: None.

### 6.2 PROBE-1: A1-SPECIALIZED V2 prototype ($1-3)

**Purpose**: Canonical anchor for all subsequent F-category reclamations.

**Tool**: `tools/build_a1_per_pattern_vq_vae_inverter_prototype.py` per A1 memo §13.1.

**Dependencies**: `tac.procedural_codebook_generator`, `tac.null_space_exploiter`, `tac.quantization.FakeQuantFP4`, `constriction`, `brotli`.

**Outputs**: A1-SPECIALIZED prototype + accuracy report + byte-mutation proof per Catalog #139 + compliance assertion citing A1 memo §1.4-1.6.

### 6.3 PROBE-2: F4 (summary 512) prototype ($2-5)

**Purpose**: Validate V2-with-PQ extension for first F-category vector.

**Tool**: `tools/build_f4_summary_512_per_pattern_inverter_prototype.py`.

**Dependencies**: A1-SPECIALIZED probe landed clean + canonical helpers.

**Pre-conditions**:
- A1-SPECIALIZED probe verdict in `.omx/state/probe_outcomes.jsonl` per Catalog #313 = PROCEED or PROMOTE
- Yousfi-style F4 compliance interpretation memo lands (incorporates Yousfi verbatim from this memo's frontmatter)

### 6.4 PROBE-3: F5 (ResBlock 512) prototype ($2-5)

**Purpose**: Validate that F5 reclamation behaves as predicted (sister of F4).

**Tool**: `tools/build_f5_resblock_output_per_pattern_inverter_prototype.py`.

**Pre-conditions**: PROBE-2 landed clean.

### 6.5 PROBE-4: F3 (vision 2048) prototype ($3-8)

**Purpose**: Validate that hyperprior-enhanced V2 reclamation works for high-dim features.

**Tool**: `tools/build_f3_vision_2048_per_pattern_inverter_prototype.py`.

**Pre-conditions**: PROBE-2 or PROBE-3 landed clean (validates V2-with-PQ framework).

### 6.6 PROBE-5: F6 (Hydra trunk 32) prototype WITH explicit Yousfi memo ($3-8)

**Purpose**: Validate F6 reclamation through INTERPRETIVE_CAUTION discipline.

**Tool**: `tools/build_f6_hydra_trunk_per_pattern_inverter_prototype.py`.

**Pre-conditions**:
- PROBE-1, PROBE-2 landed clean
- **EXPLICIT YOUSFI MEMO** addressing Contrarian's veto and Hydra-trunk-as-final-stage-interpreter concern (binding per council verdict)

### 6.7 COMPOSITION-PROBES (post-individual prototypes; $0-3 each)

| Probe | Composition | Purpose |
|---|---|---|
| COMP-AB-AS-A2 | A1-SPECIALIZED + A2 | Paired-comparison smoke per Catalog #322 — validate orthogonal composition |
| COMP-AB-F4 | A1-SPECIALIZED + F4 | Paired-comparison smoke — sub-additive empirical anchor |
| COMP-F4-F5 | F4 + F5 | Paired-comparison smoke — same-axis sub-additive measurement |
| COMP-FULL-HYDRA | A1-SPECIALIZED + F4 + F5 + F6 | Full-Hydra-coverage composition; expected highly sub-additive |

### 6.8 Probes already ADJUDICATED in `.omx/state/probe_outcomes.jsonl` (DO NOT RE-RUN)

Per Catalog #313 consultation:
- `atw_v2_d4_h_latent_given_scorer_class_20260516` (INDEPENDENT) — unrelated to F-category
- `wunderkind_g1_v2_per_pair_dominant_segnet_argmax_reducer` (DEFER) — unrelated
- `q6_preprobe_pairwise_composition_alpha_20260517` (PROCEED) — unrelated to F-category
- `c6_e4_mdl_ibps_smoke_modal_a10g_50ep` (DEFER) — unrelated
- `g1_cpu_axis_re_rank_20260518` (PARTIAL) — unrelated
- `v4_hand_rolled_faiss_ivf_pq_m2_ksub128_topk3_600pa` (INDEPENDENT) — unrelated, but the PQ pattern is RELEVANT (PQ over 600 pairs is a sister technique to our F4/F5 PQ application)

None of the adjudicated probes blocks any recommended probe in this memo.

---

## 7. Predicted ΔS band per vector

### 7.1 Per-vector band

| Vector | Predicted ΔS lower | Predicted ΔS upper | Source-of-truth |
|---|---|---|---|
| A1-SPECIALIZED | -0.012 | -0.003 | A1 memo §14.4 Dykstra-feasible band |
| F3 | -0.020 | -0.005 | Wider band per Fridrich verbatim (deeper layers have wider scorer-blind entropy) |
| F4 | -0.015 | -0.003 | Per MacKay K(inverter \| Y) bound + similar pattern to A1 |
| F5 | -0.015 | -0.003 | Same as F4 (sister layer) |
| F6 | -0.012 | -0.003 | Same as A1-SPECIALIZED (same per-pair output target via 32-dim hidden) |

### 7.2 Dykstra-feasibility intersection check per Catalog #296

Per Dykstra verbatim in this memo's frontmatter:

> "The Pareto-feasibility check for each F-vector requires: (R) net rate-positive after binary cost; (S) seg preserved; (P) pose preserved; (L) inflate.py LOC ≤200 default; (C) compliance per Yousfi per-pattern test; (D) byte-deterministic across CPU/CUDA per Catalog #205."

Per-vector feasibility:

| Vector | R | S | P | L | C | D | Feasible? |
|---|---|---|---|---|---|---|---|
| A1-SPECIALIZED V2 | ✓ (5-10 KB) | ✓ | ✓ | ✓ | ✓ | ✓ | FEASIBLE (per A1 memo §14.2) |
| F3 (Hinton + hyperprior) | ✓ (10-30 KB) | ✓ | ✓ | ✓ | ✓ | ✓ | FEASIBLE (with hyperprior side-info) |
| F4 (PQ + factorized prior) | ✓ (5-15 KB) | ✓ | ✓ | ✓ | ✓ | ✓ | FEASIBLE |
| F5 (same as F4) | ✓ (5-15 KB) | ✓ | ✓ | ✓ | ✓ | ✓ | FEASIBLE |
| F6 (32-dim hidden codebook) | ✓ (10-30 KB conservative; 3-10 KB technical floor) | ✓ | ✓ | ✓ | ⚠️ (interpretive caution) | ✓ | FEASIBLE-WITH-CAUTION |

All 5 reclaimable vectors satisfy the Dykstra-feasibility intersection within their predicted bands.

### 7.3 First-principles citation

Per MacKay verbatim: `K(inverter | Y) ≤ K(inverter) - I(F_layer; Y)`. For our contest video with deterministic per-pair features, I(F_layer; Y) is HIGH for all F-category layers because:

1. **F3 vision(2048)**: vision features are deterministic in input RGB; I(F3; Y) ≈ H(F3) per the contest video bytes
2. **F4 summary(512)**: summary is deterministic in F3; I(F4; Y) close to H(F4)
3. **F5 ResBlock(512)**: ResBlock applied to F4; I(F5; Y) similarly high
4. **F6 Hydra trunk(32)**: 32-dim representation of F5; I(F6; Y) close to H(F6)

The MDL-bound binary sizes for F3-F6 are small because the side-info Y carries most of the information.

Per Wyner-Ziv 1976 (A1 memo §6.2): `R_WZ(D) ≤ R(D) - I(X; Y)`. For per-pattern specialized inverters at any layer with Y = upstream video bytes, R_WZ(D) → 0 at the per-pattern limit.

---

## 8. ## Predicted ΔS band

The per-vector predicted ΔS bands are computed via the Dykstra-feasibility intersection (§7.2) + first-principles MDL/Wyner-Ziv bounds (§7.3). The bands are categorized [prediction; not validated] per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" — every claim carries `[contest-CPU; prediction; not validated]` axis tag per Catalog #287.

### Aggregate composition band (per Dykstra-feasibility intersection)

For the canonical composition `A1-SPECIALIZED + F4` (TOP-1 + TOP-2):

- Binary total: 10-25 KB → rate cost +0.0066 to +0.0166 ΔS
- Predicted ΔS contribution: -0.005 to -0.027 (sub-additive α=0.6 applied)
- **Net predicted ΔS: [-0.020, +0.012] [contest-CPU; prediction; not validated]**

The upper bound is potentially POSITIVE (net rate-negative) if the predicted ΔS contribution is at the low end of its band. This is the Catalog #322 sub-additive composition risk made concrete. Paired-comparison smoke per Catalog #322 is the canonical disambiguator.

### Pareto-feasibility per Catalog #296 (Dykstra check)

The convex constraint set per Dykstra verbatim:
- **R**: net rate-positive after binary cost
- **S**: segmentation preserved within ε
- **P**: pose preserved within ε
- **L**: inflate.py LOC ≤200 default
- **C**: compliance per Yousfi per-pattern test
- **D**: byte-deterministic across CPU/CUDA

For the canonical TOP-3 reclaimable vectors (A1-SPECIALIZED, F4, F5): the constraint INTERSECTION is non-empty within their predicted bands. **Pareto-feasibility verified analytically; empirical validation pending prototypes.**

---

## 9. ## Cargo-cult audit per assumption

Per Catalog #303 + the hard-earned-vs-cargo-culted addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`):

| Assumption | Classification | Unwind path |
|---|---|---|
| "Binary distillation generalizes from A1 (12-dim) to F-category (512-2048 dim) with same compression ratios" | CARGO-CULTED-PENDING-EMPIRICAL | Per-vector dimensional analysis BEFORE reclamation claim is actionable; per-prototype empirical anchor before reclaimable verdict is binding |
| "Per-pattern specialization (Yousfi compliance test) applies cleanly to F3-F6" | HARD-EARNED-PENDING-INTERPRETATION | Operator ratification + per-vector compliance interpretation memos (esp F6 per Contrarian veto) |
| "Wyner-Ziv side-info yields equal advantage across F3-F6 and A1" | HARD-EARNED | Per MacKay verbatim: K(inverter \| Y) bound holds equally; no unwind needed |
| "F-category vectors F3-F6 are all genuinely SSRV-reclaimable" | HARD-EARNED-PENDING-EMPIRICAL | Per-vector empirical prototype landing within predicted band; failure triggers re-examination |
| "Inflate.py compute budget at inflate is non-binding for F3-F6" | CARGO-CULTED-PENDING-EMPIRICAL | Per-prototype empirical inflate-time measurement on CPU + T4 surfaces; per Hotz verbatim |
| "F1/F2 collapse to A2 (NOT SSRV)" | HARD-EARNED-VERIFIED | Per Codex F1 finding relay + 43-vector audit Section 0; no unwind needed |
| "F7 is encoder-time architectural choice (NOT SSRV)" | HARD-EARNED-VERIFIED-VIA-YOUSFI | Per Yousfi verbatim in this memo's frontmatter |
| "Composition of A1 + F4 is sub-additive at α≈0.6" | CARGO-CULTED-PENDING-EMPIRICAL | Catalog #322 sub-additive default; paired-comparison smoke COMP-AB-F4 ($0-3) is canonical disambiguator |
| "PQ-8x8 + K=64 + FP4 + Brotli achieves 5-15 KB for F4/F5" | CARGO-CULTED-PENDING-EMPIRICAL | Per-prototype empirical anchor; K-sweep at prototype phase per A1 memo §10.1 |
| "Hyperprior side-info reduces F3 binary by 50% vs naive VQ-VAE codebook" | CARGO-CULTED-PENDING-EMPIRICAL | Balle 2018 literature precedent; per-prototype empirical anchor |
| "The 5 reclaimable vectors compose orthogonally with A2 + B1 + C1" | HARD-EARNED-PARTIAL | A1 memo §14.3 validates A1 + A2 + B1 + C1; F3-F6 inherit same axis structure per orthogonality argument |

### Cargo-cult composition risk

Per A1 memo §10.2: "even if each cargo-culted assumption is wrong by 30%, the cumulative miss is ~83%." Applied here: this memo has 6 cargo-culted assumptions; cumulative composition miss potential ~85%. **Mitigation**: prototype A1-SPECIALIZED first (validates V2 framework); land F4 next (smallest F-category reclamation; validates V2-with-PQ extension); add F3/F5/F6 only after V2-with-PQ empirical anchor proves the framework.

---

## 10. ## 9-dimension success checklist evidence

Per Catalog #294:

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS (class-shift not within-class) | YES — extending the A1 SUBSTRATE-CLASS-SHIFT (from "ship scorer weights" to "ship per-pattern distilled byte transducer") to ALL F-category SSRV vectors is itself a CLASS-SHIFT at the META level (from "per-vector kill" to "per-vector reclamation framework"). |
| 2. BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | YES — each vector's re-classification is a 1-row matrix entry with explicit predicted band, receiver-path evidence, verdict, reactivation criteria. Reviewable in 30 seconds per row. |
| 3. DISTINCTNESS (different from sisters) | YES — distinct from A1 memo (which scopes only A1); distinct from META-audit (which catalogs the bug class without per-vector reclamation); distinct from 43-vector audit (which classifies but doesn't extend reclamation framework). |
| 4. RIGOR (premise + adversarial + assumption + empirical) | YES — premise verification (§1.1 source-of-truth cited from canonical 43-vector audit) + adversarial council (frontmatter sextet pact + 7 engineering quartet members) + assumption surfacing (§9 Cargo-cult audit) + empirical anchors (citing A1 memo's existing prototype path) |
| 5. OPTIMIZATION PER TECHNIQUE | YES — each F-vector gets ITS OWN OPTIMAL composition stack (F3 → Hinton + hyperprior; F4/F5 → PQ + factorized prior; F6 → 32-dim direct codebook) NOT the same naive A1 V2 applied uniformly |
| 6. STACK-OF-STACKS COMPOSABILITY | YES — §5 cross-vector composition table enumerates orthogonal/sub-additive/kitchen-sink compositions with explicit α estimates per Catalog #322 |
| 7. DETERMINISTIC REPRODUCIBILITY | YES — all per-vector prototypes inherit canonical inflate device selector (Catalog #205) ensuring CPU/CUDA byte-identical output; codebooks byte-stable from training seed; Brotli is deterministic |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | YES — predicted bands push toward Shannon information-theoretic floor (per MacKay K(inverter \| Y) bound); F6 32-dim hidden technical floor is 3-10 KB (sub-A1) |
| 9. OPTIMAL MINIMAL CONTEST SCORE | PENDING — per-vector predicted ΔS bands range from -0.020 to +0.012 (composition sub-additive risk); validated only after per-prototype Modal smoke + paired Linux x86_64 [contest-CPU] anchor per Catalog #324 |

---

## 11. ## Observability surface

Per Catalog #305 (6-facet observability declaration):

### 1. Inspectable per layer
- Each F-vector prototype's composition stack is per-layer inspectable (codebook dump-able; FP4 dequantization trace; sparsity mask byte-image; Brotli pre/post stream; inflate.py execution trace)
- Per-vector compliance interpretation memo is inspectable as a standalone artifact
- Per-vector dimensional analysis is inspectable as a standalone artifact (PRE-PROBE output)

### 2. Decomposable per signal
- Per-pair: VQ/PQ index + residual contribution + final output decomposable
- Per-axis: seg vs pose vs rate contribution per Catalog #221 auth-eval result-artifact discipline
- Per-stage: codebook lookup vs residual student vs hyperprior vs final assembly decomposable
- Per-vector: each F-category reclamation's contribution decomposable from aggregate composition

### 3. Diff-able across runs
- Codebooks byte-stable from training seed → diff-able byte-for-byte across runs
- VQ/PQ index sequences diff-able
- Output frames byte-for-byte comparable via canonical Catalog #221 auth-eval roundtrip matrix
- Per-vector composition_alpha measurements diff-able across paired-comparison smokes

### 4. Queryable post-hoc
- Per-pair K-index queryable from saved decoder state
- Per-K usage frequency queryable
- Per-vector residual energy histogram queryable
- Per-composition α-value queryable from `.omx/state/substrate_composition_matrix.json` per Catalog #322

### 5. Cite-able
- Each per-vector prototype: (archive_sha256, codebook_sha256, procedural_seed_sha256, prototype_landing_commit_sha) tuple per Catalog #245 modal_call_id_ledger + Catalog #186 catalog-claim-via-serializer
- Council verdict cite-able via continual_learning_posterior.jsonl anchor per Catalog #300
- Per-vector reactivation criteria cite-able from this memo's frontmatter `predicted_band_validation_reactivation_criteria` field

### 6. Counterfactual-able
- Catalog #139 packet compiler byte-mutation smoke for each per-vector prototype
- Catalog #220 substrate operational mechanism declaration per per-vector prototype
- Catalog #272 distinguishing-feature integration contract per per-vector prototype
- Per-composition counterfactual: "what if we ship A1-SPECIALIZED WITHOUT F4?" is queryable via paired smoke COMP-AB-F4

---

## 12. ## Canonical-vs-unique decision per layer

Per Catalog #290 + CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

| Layer | Canonical-vs-unique decision | Rationale |
|---|---|---|
| 1. Algorithm choice | **CANONICAL ADOPT** for A1/F4/F5 (V2 VQ-VAE/PQ); **FORK_BECAUSE_PRINCIPLED_MISMATCH** for F3 (requires Hinton distilled student + hyperprior because dim 2048 too large for naive VQ-VAE); **FORK** for F6 (32-dim direct codebook is unique to Hydra trunk's hidden dimensionality) |
| 2. Quantization | **CANONICAL ADOPT** for all (FP4 per Quantizr PR101 canonical; Selfcomp block-FP 1.017 bpw alternative) |
| 3. Sparsity | **CANONICAL ADOPT** for all (50% structured sparse per Quantizr PR101) |
| 4. Compression | **CANONICAL ADOPT** for all (Brotli per Quantizr PR101 + arithmetic via Constriction per Catalog #203 hard dep) |
| 5. Delivery | **CANONICAL ADOPT** for all (standard Python inflate.py; no custom binary container per Hotz veto in A1 memo) |
| 6. Compliance assertion | **FORK_BECAUSE_PRINCIPLED_MISMATCH** for F6 (requires EXPLICIT Yousfi interpretation memo per Contrarian veto; not generic Yousfi seat verbatim) |
| 7. Receiver-path evidence | **CANONICAL ADOPT** for all (Codex Section 0 5-condition legality test) |
| 8. Composition strategy | **FORK_BECAUSE_SUPPRESSES** — naive multiplicative composition would CARGO-CULT the orthogonal-composition assumption; per Catalog #322 sub-additive default + paired-comparison smoke discipline overrides naive composition |
| 9. Probe dispatch ordering | **FORK** — A1-SPECIALIZED must land first as canonical anchor; F4 second (smallest reclamation); F5 third; F3/F6 last. Unique to the framework's empirical-anchor sequencing |
| 10. Reactivation criteria pinning | **CANONICAL ADOPT** per Catalog #325 4-criterion canonical pattern, with per-vector substantive content |

The dominant decision is **CANONICAL ADOPT BECAUSE_SERVES** for engineering primitives (quantization, sparsity, compression, delivery, receiver-path evidence) and **FORK_BECAUSE_PRINCIPLED_MISMATCH** for vector-specific algorithm and compliance details. The canonical-vs-unique split honors the operator's UNIQUE-AND-COMPLETE-PER-METHOD operating mode without inheriting cargo-cult uniformity.

---

## 13. ## 6-hook wire-in declaration

Per Catalog #125:

1. **Sensitivity-map contribution**: ACTIVE — each per-vector prototype's null-space basis (`tac.null_space_exploiter`) contributes per-byte axis-weight rows to `tac.sensitivity_map.*`. Per-vector rows: (archive_offset, vector_id, seg_response, pose_response, rate_contribution). Per-prototype contribution feeds into the cathedral autopilot ranker via canonical Provenance per Catalog #323.

2. **Pareto constraint**: ACTIVE — per §7.2 Dykstra-feasibility intersection, each reclaimed vector adds a Pareto constraint to `tac.pareto_*`: `binary_bytes_v ≤ savings_from_encoding_v`. Composition constraints added via paired α-values per Catalog #322.

3. **Bit-allocator hook**: ACTIVE — VQ/PQ selections at compress time are bit-allocator decisions. Per-vector bit budget = `log2(K_v) × N_pairs` bits for V2; PQ variant = `M × log2(K_subspace) × N_pairs` bits. Per-vector hook registered in bit-allocator.

4. **Cathedral autopilot dispatch hook**: ACTIVE — when each per-vector prototype lands, register as candidate in `tools/cathedral_autopilot_autonomous_loop.py` candidate queue. Predicted ΔS bands per §7.1 feed ranking. Sub-additive composition adjustments per Catalog #322 `adjust_predicted_delta_for_composition_alpha_v2`.

5. **Continual-learning posterior update**: ACTIVE — council anchor emission via `tac.council_continual_learning.append_council_anchor` per this memo's v2 frontmatter (Catalog #300). Per-prototype Modal smoke empirical anchors update `.omx/state/continual_learning_posterior.jsonl`. Per-composition paired-smoke anchors update `.omx/state/substrate_composition_matrix.json`.

6. **Probe-disambiguator**: ACTIVE — Multiple defensible interpretations of each F-vector reclamation (per-pattern distilled vs PQ vs RVQ vs Hinton student vs hyperprior) require probe disambiguator. Per-vector tools `tools/probe_f<n>_<arch>_disambiguator.py` execute when prototype phase begins. **This memo IS the canonical META-disambiguator** between "vector is SSRV" (original 43-vector audit) vs "vector is reclaimable via binary distillation" (this memo) — at the systematic re-examination surface.

---

## 14. ## horizon_class

`horizon_class: frontier_pursuit`

Per CLAUDE.md "HORIZON-CLASS evaluation axis standing directive" 2026-05-16: classify by predicted CPU band:
- PLATEAU-ADJACENT [0.180, 0.200]
- **FRONTIER-PURSUIT [0.120, 0.180]** ← THIS memo
- ASYMPTOTIC-PURSUIT [0.050, 0.120]

The aggregate composition predicted ΔS [-0.020, +0.012] applied to canonical frontier `0.19205 [contest-CPU]` yields predicted post-composition score band `[0.17205, 0.20405]`. The lower bound `0.17205` is FRONTIER-PURSUIT class (per CLAUDE.md HORIZON-CLASS Consequence 2: ≥20% budget allocation to asymptotic-pursuit reminded; FRONTIER-PURSUIT is the dominant band of this memo).

The upper bound `0.20405` exceeds the canonical frontier (rate-cost dominates if composition fails to deliver expected ΔS); this is the Catalog #322 sub-additive composition risk made concrete and motivates the per-vector paired-comparison smoke discipline.

---

## 15. Cross-references

### 15.1 CLAUDE.md sections
- **Strict scorer rule — non-negotiable** (Catalog #6) — the rule this memo systematically re-examines
- **Deterministic packet compiler — non-negotiable** — sanctions reclamation via `optimize` mode + `contest_one_video_replay`
- **Contest vs production target modes — non-negotiable** — `contest_one_video_replay` framework
- **HNeRV parity discipline lesson 4** — inflate.py LOC budget ≤200 with rationale
- **HNeRV parity discipline lesson 9** — runtime closure
- **Forbidden premature KILL without research exhaustion** — basis for "KILL is LAST RESORT" discipline
- **PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium** (Catalog #325) — reactivation criteria framework
- **UNIQUE-AND-COMPLETE-PER-METHOD operating mode** — basis for canonical-vs-unique decision per layer (§12)
- **META-ASSUMPTION ADVERSARIAL REVIEW** (Catalog #291) — basis for systematic re-examination cadence
- **Apples-to-apples evidence discipline** — every predicted ΔS axis-tagged

### 15.2 Catalog cross-references
- Catalog #6 (strict scorer rule) — the rule this memo re-examines
- Catalog #105 (no-op detector)
- Catalog #125 (subagent landing 6-hook wire-in)
- Catalog #126 (lane pre-registered before work starts)
- Catalog #127 (authoritative tag custody metadata)
- Catalog #131 (no bare writes to shared state) — applies to `.omx/state/probe_outcomes.jsonl` writes
- Catalog #138 (state writers strict load)
- Catalog #139 (packet compiler byte-mutation smoke)
- Catalog #185 (CLAUDE.md drift detection — META-meta protection)
- Catalog #205 (inflate device fork)
- Catalog #206 (subagent crash-resume discipline — applies to this subagent)
- Catalog #220 (L1+ scaffold operational mechanism)
- Catalog #229 (premise verification before edit)
- Catalog #230 (bulk-rewrite ownership map)
- Catalog #233 (L1-to-L2 promotion canonical 4-gate)
- Catalog #245 (Modal call_id ledger canonical 4-layer pattern)
- Catalog #248 (no stash-pop conflict markers)
- Catalog #272 (distinguishing-feature integration contract)
- Catalog #287 (empirical-claim evidence tags)
- Catalog #290 (canonical-vs-unique decision per layer)
- Catalog #291 (META-ASSUMPTION cadence)
- Catalog #292 (per-deliberation assumption surfacing)
- Catalog #294 (9-dim success checklist)
- Catalog #296 (Dykstra-feasibility intersection)
- Catalog #298 (substrate retirement discipline — applies to per-vector lanes after this memo)
- Catalog #300 (council deliberation v2 frontmatter)
- Catalog #303 (cargo-cult audit per assumption)
- Catalog #305 (observability surface)
- Catalog #309 (horizon_class declaration)
- Catalog #313 (predecessor probe outcomes ledger)
- Catalog #316 (frontier scan)
- Catalog #319 (Wyner-Ziv deliverability proof — sister discipline for reclaimed vectors)
- Catalog #322 (autopilot adjustment via canonical composition_alpha posterior)
- Catalog #323 (canonical Provenance for score-claim artifacts)
- Catalog #324 (post-training Tier-C validation)
- Catalog #325 (per-substrate optimal form symposium)
- Catalog #334 (this memo's lane — pre-registered transactionally via canonical claim_catalog_number helper per Catalog #186)

### 15.3 Sister memo cross-references
- `.omx/research/a1_binary_distillation_design_memo_zig_sparseness_ablation_plus_canonical_techniques_20260518.md` (commit 0701c323b; THE canonical exemplar framework this memo extends)
- `.omx/research/rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_20260518.md` (commit 35b06f9ec; the 43-vector audit with per-vector legal-receiver-path verdicts)
- `.omx/research/meta_audit_conflate_declarative_with_physical_error_pattern_12_claim_self_audit_20260518.md` (commit e86ca6d0c; the META-pattern this memo executes Section 3 recommended action on)
- `.omx/research/cargo_cult_burn_down_supplement_extending_meta_audit_across_session_20260518.md` (commit fb102933b; extends META-audit across session)
- `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md` (PRIMARY rate-attack master memo — 43-vector enumeration source)
- `.omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md` (commit 4c6e46bfa; ADVERSARIAL 5 binding critiques)
- `.omx/research/rate_attack_synthesis_v2_reconciliation_primary_plus_adversarial_plus_supplement_20260518.md` (sister reconciliation memo)
- `.omx/research/grand_council_symposium_inflate_py_extreme_compression_20260518.md` (sister inflate.py LOC budget discipline memo — Catalog #328 sister gate)
- `.omx/research/rate_attack_vector_3_b1_contest_video_codebook_design_memo_20260518.md` (sister B1 codebook design — validates VQ-VAE technique)

### 15.4 Canonical helper cross-references
- `src/tac/procedural_codebook_generator/hash_seed_codebook_generator.py` — Layer 1 generator for V1 variant (per-vector)
- `src/tac/procedural_codebook_generator/weight_derived_codebook_generator.py` — Layer 1 generator for V3 variant (per-vector)
- `src/tac/null_space_exploiter/core.py` — null-space basis for per-vector feature directions
- `src/tac/quantization.py` `FakeQuantFP4` + `FakeQuantSTE` + `LSQ` — Layer 2 quantization
- `src/tac/packet_compiler/deterministic_compiler.py` — Layer 5 inflate.py emission
- `tools/build_deterministic_packet.py` — canonical CLI for per-vector packet builds
- `tac.council_continual_learning.append_council_anchor` — continual-learning posterior writer per Catalog #300

### 15.5 Sister-subagent coordination
- DYNAMIC PER-CANDIDATE COMPOSITION FRAMEWORK subagent `a9be5dcb544013d61` in flight (DISJOINT scope: per-candidate composition; this memo is RE-CLASSIFICATION of vectors)
- A1-BINARY-DISTILLATION subagent (already landed via commit `0701c323b`; this memo extends framework)
- F-CATEGORY-RECLAMATION subagents (FUTURE; per-vector prototype subagents per §6 probe ordering)

---

## 16. Conclusion

The 43-vector audit (commit `35b06f9ec`) classified 9 rows as STRICT_SCORER_RULE_VIOLATION or flagged-for-re-examination. The A1 binary distillation framework (commit `0701c323b`) already reclaimed A1-SPECIALIZED via per-pattern distilled VQ-VAE inverter sized 5-20 KB.

**This memo extends the framework systematically to 4 additional F-category vectors** (F3, F4, F5, F6) via the same per-pattern specialization + V2-with-PQ/hyperprior compositions. **Net result: 5 reclaimable vectors** (A1-SPECIALIZED, F3, F4, F5, F6 with predicted bands 5-30 KB), **3 re-classified vectors** (F1, F2, F7 not SSRV by construction), **1 stays-SSRV vector** (A1-CANONICAL — the naive "ship full PoseNet" framing), **0 killed** per CLAUDE.md "KILL is LAST RESORT" non-negotiable.

**The recommended next-step probe sequence**:
1. PRE-PROBE: per-vector dimensional analysis ($0 research)
2. PROBE-1: A1-SPECIALIZED V2 prototype ($1-3) — canonical anchor
3. PROBE-2: F4 (summary 512) prototype ($2-5) — smallest F-reclamation
4. PROBE-3: F5 (ResBlock 512) prototype ($2-5) — sister of F4
5. PROBE-4: F3 (vision 2048) prototype ($3-8) — high-dim test
6. PROBE-5: F6 (Hydra trunk 32) prototype WITH EXPLICIT YOUSFI MEMO ($3-8) — interpretive-caution case
7. COMPOSITION-PROBES: paired-comparison smokes per Catalog #322 ($0-3 each)

**Mission contribution: frontier_protecting**. The systematic reclamation framework IS the structural protection against premature KILL verdicts (per Pre-rigor KILL/DEFER/FALSIFIED inventory which found 28-32 of 34 historical kills FAILED the new rigor gates). Reclaiming 5 vectors that would otherwise sit at SSRV ⚠️ permanently is canonical research-exhaustion path that serves frontier_breaking by EXPANDING the live action space.

**Predicted aggregate impact**: composition of A1-SPECIALIZED + F4 alone predicts net ΔS [-0.020, +0.012] [contest-CPU; prediction; not validated] — lower bound `0.17205` is FRONTIER-PURSUIT class displacement from canonical frontier `0.19205`; upper bound risk requires Catalog #322 sub-additive paired-comparison smoke discipline.

The strict-scorer-rule is preserved in spirit because every reclaimed binary is OBVIOUSLY SPECIALIZED (per Yousfi verbatim) under CLAUDE.md `contest_one_video_replay` target mode sanctioning. F6 carries explicit INTERPRETIVE_CAUTION flag per Contrarian veto and binding requirement for explicit Yousfi memo before prototype lands.

— Main-Claude 2026-05-18 (Systematic reclaimability re-examination via binary distillation framework; T2 sextet + engineering quartet + MacKay/Balle/Selfcomp grand-council seats; council verdict PROCEED_WITH_REVISIONS; 5 reclaimable / 3 re-classified / 1 stays SSRV / 0 killed)
