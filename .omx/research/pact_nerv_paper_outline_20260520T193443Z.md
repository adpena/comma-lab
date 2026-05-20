<!-- SPDX-License-Identifier: MIT -->
# Pact-NeRV: A 15+ Variant Taxonomy + Empirical α Composability Matrix for Single-Video Compression — paper outline draft

**Authors:** Alejandro Peña (Pact lab; corresponding) + sister-subagent-developed apparatus components attribution per CLAUDE.md "User PR Attribution" + canonical helper attribution per `tac.*` module authorship; the paper documents apparatus internals as collaborative research output.
**Date:** 2026-05-20
**Target venue:** TBD — operator-routable. Candidates: arXiv preprint (primary); NeurIPS workshop on compression / efficient ML / neural codecs; ICLR workshop on real-world ML; sister of comma.ai compression challenge writeup.
**Sister of:** PACT-NERV-DESIGN-SYMPOSIUM Stage 6 paper-write op-routable #6 + FILM-FAMILY-RESEARCH Section 10 5-recommendation framing + CROSS-CANDIDATE-SENSITIVITY-COMPARISON diagnostic + the META engineering vision per `docs/meta_engineering_vision.md`.

Per CLAUDE.md "Strategic Secrecy" non-negotiable: this outline is INTERNAL DESIGN; paper publication requires operator routing per submission_dir + canonical compliance gates. The paper's contribution is the **methodology** + **empirical apparatus** + **15+ variant ablation as ablation contribution**; it does NOT depend on Pact-NeRV beating the contest frontier.

**Editorial voice per MG-19 + MG-15 + MG-16:** Yousfi/Hotz-voice — direct, empirically-grounded, honest about limitations, no marketing claims. Solo-developer scope explicit. Collaboration call at the end.

---

## 1. Introduction (META engineering vision)

**Length:** ~600 words

**Substance:**

- Frame the **single-video compression problem** as the canonical instance of the comma.ai compression challenge.
- Establish the **META engineering vision per `docs/meta_engineering_vision.md`**: building a coherent apparatus that converts research literature + empirical signals into per-method optimal substrate engineering.
- Establish the **PR101 GOLD HNeRV-class baseline (0.193 [contest-CPU])** as the empirical floor the apparatus is designed to systematically improve from.
- Establish the **canonicalization paradox** per the 2026-05-15 operator retrospective (CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"): excessive canonical-helper-sharing produces a flat plateau (the 0.196-0.199 cluster); per-method unique-and-complete engineering breaks out.
- **Thesis statement**: Pact-NeRV is the canonical synthesis of (a) the apparatus's empirical-priors-as-conditioning (canonical equations + cathedral consumers + per-pair difficulty + per-class chroma + ego-motion signals), (b) HNeRV-class architectural primitives (multi-layer FiLM + adaLN-Zero), (c) the 8-dimensional ULTIMATE decomposition that operationalizes per-mission cost-benefit decisions, and (d) the cross-substrate sensitivity analysis that empirically anchors composition_alpha predictions.

**Key insight to frame**: the contest scorer's `0.193` plateau is not a fundamental rate-distortion limit; it is the canonical floor produced by within-class refinement of the saturated 178158-byte HNeRV backbone. Breaking through requires either SELECTOR-PARADIGM-EXTENSIONS (CROSS-CANDIDATE finding #1) or CROSS-CODEC composition (CROSS-CANDIDATE finding #3).

---

## 2. Pact-NeRV variant taxonomy — STAIRCASE + DAG GRAPH

**Length:** ~1500 words (figure-heavy)

**Substance:**

### 2.1 The 15+ variants (with table)

Reproduce the Section 4 variant taxonomy table from the research memo. Categorize variants into 4 groups:
1. Bleeding-edge architecture (Mamba / MoE / DiffusionDistilled / Dreamer / NeuralCodecE2E) — 5 variants
2. Mid-LOC apparatus-aligned (DistilledScorer / VQ / Bayesian / MultiModal / DiffusionTrajectory) — 5 variants
3. SELECTOR-PARADIGM-EXTENSIONS (SELECTOR-V2/V3/V4 / IA3-Multi / AsymmetricBoundary) — 5 variants
4. CROSS-CODEC composition (CROSS-CODEC-A/B + NeuralCodecE2E-CROSS) — 3 variants

### 2.2 The 21-step STAIRCASE methodology

Reproduce the Section 8 STAIRCASE methodology from the research memo. Each step is dependency-gated + empirically-anchored. Show how the staging discipline prevents the NSCS06-v6 553x-miss anti-pattern (composition without per-primitive empirical anchor).

### 2.3 The DAG GRAPH visualization

Reproduce the Section 9 DAG Mermaid graph. Highlight:
- 6 PRIORITY 1 nodes (green) — the empirically-grounded SELECTOR + CROSS-CODEC critical paths
- 7 PRIORITY 2 nodes (yellow) — within-Pact-NeRV-family + sister symposiums
- 8 PRIORITY 3 nodes (orange) — high-LOC + asymptotic-pursuit + paper-only

Highlight ULTIMATE end-state nodes (red) with cardinality 8.

### 2.4 LOC efficiency curve (Carmack-razor)

Per-variant LOC plot vs predicted ΔS — empirical Carmack-razor curve. SELECTOR variants are 6.5x more LOC-efficient than PR101 GOLD baseline; high-LOC variants (Mamba / Dreamer / NeuralCodecE2E) are below PR101 baseline per-LOC unless asymptotic-pursuit pays off.

---

## 3. Empirical α composability matrix methodology

**Length:** ~1200 words

**Substance:**

### 3.1 The cross-substrate sensitivity comparison method

Reproduce the CROSS-CANDIDATE diagnostic methodology:
- 21-pair similarity matrix via per-pair top-K=32 Jaccard L1 + per-axis Pearson on common prefix
- Classification taxonomy: SUB_ADDITIVE / SUPER_ADDITIVE / ANTAGONISTIC / SUB_ADDITIVE_PARTIAL / INDETERMINATE / INSUFFICIENT_DATA
- Canonical equation registration per Catalog #344: `cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1`

### 3.2 Cross-hardware drift (CROSS-CANDIDATE finding #2)

Document the empirical drift signal: fec6 top-1% leverage 6.41% (macOS-CPU advisory) → 11.11% (CUDA T4) = 73% concentration delta on SAME archive bytes. Show that this implies composition_alpha predictions consumed by autopilot ranker MUST carry paired CPU+CUDA anchors before being trusted.

### 3.3 Sister CPU-CUDA-WRITEUP canonical equations

Document the 5 canonical equations supporting the methodology:
- `per_byte_leverage_uniformly_distributed_v1` (top-1% leverage band 6.4%-11.11%)
- `mps_drift_architecture_class_dependent_v1` (cross-hardware drift factor ~30x sister)
- `cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1` (composition classification)
- `pr101_vs_fec6_byte_leverage_distribution_v1` (PR101 GOLD vs fec6 frontier delta)
- `per_pair_master_gradient_score_impact_taylor_v1` (per-pair sensitivity Taylor expansion)

### 3.4 The 3 CROSS-CANDIDATE findings (per Section 5 of research memo)

Document the 3 headline findings + their concrete implications for Pact-NeRV design:
1. BACKBONE SATURATION → SELECTOR-PARADIGM-EXTENSIONS variants 11-15
2. CROSS-HARDWARE DRIFT → STAIRCASE Step 17 paired-anchor prerequisite
3. SUPER_ADDITIVE PR106 ↔ PR101 → 8th ULTIMATE-CROSS-CODEC dimension

---

## 4. MULTIPLE ULTIMATE empirically-winning compositions

**Length:** ~1400 words

**Substance:**

### 4.1 The 8-dimensional ULTIMATE decomposition

Reproduce the Section 10 + 11 ULTIMATE descriptions. Emphasize:
- ULTIMATE-FRONTIER critical path: S11 → S12 → S13 + S16 ($1.70; predicted [0.170, 0.187])
- ULTIMATE-CROSS-CODEC critical path: S16 PROCEED-unconditional per Atick ($0.80; predicted [-0.010, -0.003])
- ULTIMATE-EFFICIENCY 6.5x Carmack-razor baseline beat
- ULTIMATE-PAPER independent of contest score per Catalog #300 Mission Alignment

### 4.2 Per-ULTIMATE Dykstra-feasibility check

Document the per-ULTIMATE convex-feasibility analysis per Boyd's lens (CLAUDE.md "Council conduct" 4-co-lead amendment). Show that the SELECTOR-EXTENSION axes + CROSS-CODEC axis form a non-empty convex polytope intersection at the contest scorer's receptive field decomposition.

### 4.3 MINIMUM_VIABLE per ULTIMATE shortcuts

Per ULTIMATE: the cheapest path that still satisfies the dimension. Operator-facing table — given a budget + a target ULTIMATE, pick the minimum-cost step subset.

### 4.4 Empirical α matrix per ULTIMATE-FRONTIER path

Show the populated empirical α matrix for the SELECTOR-EXTENSION + CROSS-CODEC critical path; classify each pair per Catalog #322 cascade. Show that the polytope intersection achieves the predicted band.

---

## 5. Cargo-cult-unwound design methodology + adversarial grand council symposium discipline

**Length:** ~1000 words (methodological contribution)

**Substance:**

### 5.1 The Catalog #303 cargo-cult audit per assumption framework

Document the canonical pattern: every substrate design memo carries `## Cargo-cult audit per assumption` section that enumerates each substrate-design assumption + HARD-EARNED-vs-CARGO-CULTED classification per the addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`) + unwind-test plan.

### 5.2 The Catalog #325 per-substrate symposium 6-step contract

Document the canonical contract:
1. Cargo-cult audit per Catalog #303
2. 9-dim checklist evidence per Catalog #294
3. Observability surface declaration per Catalog #305
4. Sextet pact deliberation + grand council attendees per topic per Catalog #292+#300+#346
5. Per-substrate reactivation criteria pinned per Catalog #301
6. Catalog #324 post-training Tier-C validation discipline

### 5.3 The NSCS06 v6→v7 44% improvement empirical anchor

Document the canonical cargo-cult-unwind success story (per `.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md`): NSCS06 v6 landed 105.15 (553x outside predicted band [0.10, 0.20]); v7 unwound 4 of 7 cargo-culted assumptions in ONE iteration and achieved 58.89 (44% improvement) without paid dispatch waste. This validates the methodology empirically.

### 5.4 The 4-co-lead inner council structure

Per CLAUDE.md "Council conduct amendment 2026-05-19": Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD. Each co-lead covers an orthogonal axis (information-theory + optimization-feasibility + interpretable-ML + multi-scale wavelet partition prior). The methodology depends on the 4-co-lead structure for the meta-Lagrangian/Pareto solver dependency.

---

## 6. Empirical receipts

**Length:** ~1000 words (figure-heavy)

**Substance:**

### 6.1 PR101 GOLD vs fec6 backbone-equivalence empirical anchor

Reproduce the CROSS-CANDIDATE Section 3 per-byte delta diagnostic:
- PR101 GOLD: 0.19538 [contest-CPU]; 178158 bytes
- fec6 frontier: 0.19205 [contest-CPU]; 178417 bytes (+259 bytes)
- Per-axis aggregate sensitivity: IDENTICAL to 4 sig figs (seg sum_abs = 1.5838e-1; pose sum_abs = 1.1291e-1)

**Implication**: backbone is saturated; the +0.00333 advantage comes ENTIRELY from selector overhead.

### 6.2 Cross-hardware drift empirical anchor

Reproduce the CROSS-CANDIDATE Section 4 cross-hardware drift table:
- Advisory mean (6 substrates): 6.38% top-1% leverage
- CUDA T4 (1 authoritative): 11.11% top-1% leverage (+73%)

**Implication**: cross-hardware drift is REAL; composition_alpha predictions require paired anchors.

### 6.3 SUPER_ADDITIVE PR106 ↔ PR101/A1/fec6 empirical anchor

Reproduce the CROSS-CANDIDATE Section 2 SUPER_ADDITIVE signature:
- `pr101_gold` ↔ `pr106_format0d`: seg ρ = -0.076 / pose ρ = -0.094 (SUPER_ADDITIVE)
- `pr106_format0d` ↔ `fec6_frontier_cuda_t4`: seg ρ = -0.083 / pose ρ = -0.078 (SUPER_ADDITIVE)

**Implication**: cross-codec composition is structurally orthogonal per cooperative-receiver framing.

### 6.4 Apparatus statistics

Document the apparatus's empirical infrastructure:
- 47+ cathedral consumers auto-discovered per Catalog #335
- 6+ canonical equations registered per Catalog #344
- 295+ STRICT preflight gates per CLAUDE.md "Meta-bug class catalog"
- 11+ master-gradient anchors across 7 sister substrates per `tac.master_gradient.load_anchors_lenient`

---

## 7. Honest qualifications + collaboration call

**Length:** ~500 words (per MG-15 + MG-16 voice)

**Substance:**

### 7.1 Solo-developer scope explicit

Per MG-15 + MG-16 voice (Yousfi/Hotz direct + no marketing claims): the Pact lab is solo-developer scope. The apparatus's 47+ cathedral consumers + 295+ STRICT preflight gates were built over ~6 months of single-operator engineering with Claude Code assistance. The methodology + empirical apparatus are reproducible per the `docs/superpowers/` runbooks; the absolute score numbers depend on contest-specific scorer architecture (FastViT-T12 + EfficientNet-B2) + scorer-specific receptive field analysis.

### 7.2 What this paper does NOT claim

- Pact-NeRV does NOT necessarily beat HNeRV/PR101 on the contest scorer (the paper contribution is the methodology + ablation, not a score claim)
- The 8-dimensional ULTIMATE decomposition is NOT a universal framework (it is specific to the comma.ai compression challenge + the apparatus's empirical-prior surface)
- Cross-substrate sensitivity comparison via master-gradient sidecars is NOT a contest-CUDA-grade method (it is observability-only per Catalog #341; paired-CUDA anchors are required for promotion)
- The 15+ variant taxonomy is NOT exhaustive (the space of conditioning + selector + cross-codec compositions is much larger; we focus on the empirically-anchored subset)

### 7.3 Open questions

- Does the SUPER_ADDITIVE cross-codec class signature generalize beyond PR106/PR101/fec6 to other neural-vs-traditional codec pairs?
- Does the +259 bytes / +0.00333 fec6 selector empirical ratio extrapolate cleanly to selectors with K > 32 + per-pair-difficulty side-info?
- Does the canonical equation `cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1` predict α correctly across non-NeRV-class substrates?
- Can the 8-dimensional ULTIMATE decomposition be operationalized for non-compression tasks?

### 7.4 Collaboration call

The Pact apparatus is open-source per the operator's standing OSS posture. The CLAUDE.md non-negotiables + canonical helpers + cathedral consumers are reproducible. Researchers + practitioners interested in the methodology + cross-substrate sensitivity analysis + 8-dimensional ULTIMATE framework are invited to:
- Use the apparatus's canonical helpers for sister problems (the framework is task-agnostic at the discipline level)
- Contribute new cathedral consumers per Catalog #335 (the auto-discovery paradigm enables organic extension)
- Replicate the cross-substrate sensitivity comparison on sister neural codec benchmarks (UVG / Kodak / etc.)
- Adapt the per-substrate symposium discipline for other research-engineering decision surfaces

---

## 8. References

- HNeRV (Chen et al. 2023) — arXiv:2304.02633 + github.com/haochen-rye/HNeRV
- FiLM (Perez et al. 2018) — arXiv:1709.07871
- TeNeRV / Temporal NeRV ablations — sister citations
- CLADENorm (CLADE) — third-party-empirical
- Atick-Redlich cooperative-receiver (1990) — canonical reference
- Tishby-Zaslavsky information bottleneck (1999) — arXiv:physics/0004057
- DiT adaLN-Zero (Peebles & Xie 2023) — arXiv:2212.09748
- DreamerV3 (Hafner et al. 2023) — arXiv:2301.04104
- Mamba (Gu & Dao 2023) — arXiv:2312.00752
- IA3 (Liu et al. 2022) — arXiv:2205.05638
- VQ-VAE (van den Oord et al. 2017) — arXiv:1711.00937
- Ballé hyperprior (Ballé et al. 2018) — arXiv:1802.01436
- Score Distillation Sampling (Poole et al. 2022) — arXiv:2209.14988
- Falling Rule Lists (Wang & Rudin 2015) — canonical Rudin reference
- GOSDT (Lin et al. 2020) — canonical GOSDT reference
- comma.ai compression challenge — github.com/commaai/comma_video_compression_challenge

---

## 9. Appendix A — apparatus architecture diagram

Show the canonical apparatus architecture:
- `tac.*` package canonical helpers (cathedral_consumers / canonical_equations / provenance / council_continual_learning / etc.)
- `.omx/state/*.jsonl` canonical ledgers (modal_call_id_ledger / continual_learning_posterior / canonical_equations_registry / probe_outcomes / etc.)
- 295+ STRICT preflight gates per `src/tac/preflight.py`
- 47+ cathedral consumers auto-discovered per Catalog #335 paradigm
- 36-attendee council per `tac.canonical_council_roster`

## 10. Appendix B — STAIRCASE methodology pseudocode

Document the canonical staging discipline as pseudocode:

```
for step in STAIRCASE.steps:
    if not all(prereq.empirical_anchor_landed for prereq in step.prereqs):
        defer(step, reactivation_criterion=step.prereqs.missing)
        continue
    if step.dispatch_envelope.cost > operator_budget:
        defer(step, reactivation_criterion="operator_budget_increase")
        continue
    council_verdict = per_substrate_symposium(step.variant)
    if council_verdict not in {PROCEED, PROCEED_WITH_REVISIONS}:
        defer(step, reactivation_criterion=council_verdict.reactivation_criteria)
        continue
    empirical_anchor = dispatch(step)
    register_canonical_equation(empirical_anchor)
    update_composition_alpha_matrix(empirical_anchor)
    if step.ultimate_eligibility_satisfies(target_ultimate):
        target_ultimate.advance(step)
```

## 11. Appendix C — DAG GRAPH (Mermaid)

Reproduce Section 9 of the research memo.

## 12. Appendix D — Per-variant LOC + predicted ΔS table

Reproduce Section 4 variant taxonomy table.

## 13. Appendix E — Per-ULTIMATE critical path + cost envelope + MINIMUM_VIABLE shortcut

Reproduce Section 9 critical path per ULTIMATE table.

---

**Anchor memos cross-referenced:**
- `.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md` (THIS landing's parent research memo)
- `.omx/research/film_family_alternatives_bleeding_edge_research_20260520T184150Z.md` (FILM-FAMILY-RESEARCH)
- `.omx/research/council_per_substrate_symposium_pact_nerv_score_axis_aware_foveated_ego_motion_full_stack_synergy_eval_roundtrip_20260520T185500Z.md` (PACT-NERV-DESIGN-SYMPOSIUM)
- `.omx/research/cross_candidate_sensitivity_comparison_diagnostic_20260520T192204Z.md` (CROSS-CANDIDATE)
- `docs/meta_engineering_vision.md` (META engineering vision)
- `docs/superpowers/` (reproducibility runbooks)

**Editorial cadence (operator-routable):**

1. Drafting cycle 1 (this outline)
2. Drafting cycle 2 — expand each section to full prose (~6000 words total target)
3. Drafting cycle 3 — incorporate sister-subagent edits + per-Pact-NeRV-stage empirical anchors as they land
4. Drafting cycle 4 — Yousfi/Hotz-voice tone audit + MG-19 editorial discipline pass
5. Final submission — operator routing to arXiv preprint + NeurIPS/ICLR workshop submission
