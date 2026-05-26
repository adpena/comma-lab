---
schema: substrate_design_memo_v1
deliberation_id: path_3_j_mdl_ibps_substrate_design_20260526
topic: "Path 3 J=MDL-IBPS DISCRETE-CATEGORICAL-MINE-HYBRID L0 SCAFFOLD design memo (Phase 3 of 3) per operator binding directives 2026-05-26"
review_kind: l0_scaffold_design_memo_T2
review_date: "2026-05-26"
lane_id: lane_path_3_j_mdl_ibps_information_bottleneck_cargo_cult_first_20260526
substrate_id: path_3_j_mdl_ibps
substrate_alias: mdl_ibps_j
parent_substrate_id: c6_e4_mdl_ibps
deferred_substrate_id: path_3_j_mdl_ibps
horizon_class: frontier_pursuit
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - MacKay
  - Tishby
  - Zaslavsky
  - Higgins-memorial
  - Belghazi-memorial
  - Hafner
  - Contrarian
  - Assumption-Adversary
  - PR95Author
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
predicted_band_validation_status: pending_post_training
predicted_band: null
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
research_only: true
dispatch_enabled: false
related_deliberation_ids:
  - path_3_j_mdl_ibps_cargo_cult_audit_of_c6_scaffold_20260526
  - path_3_j_mdl_ibps_substrate_design_decision_20260526
catalog_anchors: [124, 125, 146, 164, 192, 205, 215, 220, 226, 229, 240, 244, 270, 287, 290, 292, 294, 295, 296, 303, 305, 309, 317, 324, 325, 341, 344, 1265]
mission_contribution: frontier_breaking_enabler
---

# Path 3 J=MDL-IBPS DISCRETE-CATEGORICAL-MINE-HYBRID — Design memo (L0 SCAFFOLD)

**Predecessors:**
- Phase 1: `.omx/research/path_3_j_mdl_ibps_cargo_cult_audit_of_c6_scaffold_20260526.md` (11 CCs classified)
- Phase 2: `.omx/research/path_3_j_mdl_ibps_substrate_design_decision_20260526.md` (Path (a) DISCRETE-CATEGORICAL-MINE-HYBRID chosen)

**Architecture summary:**

```
Per-pair categorical-index modulation m_i ∈ {0, 1, ..., K-1}^G  [K=16; G=12; 48 bits per pair]
   |
   v   (Gumbel-Softmax reparametrization for training; argmax for inference)
   v
   FiLM modulation: scale_i, shift_i = linear_film_proj(one_hot(m_i)) → R^(HIDDEN_DIM * 2)
   |
   v
Procedural coord-MLP base F_φ:
    Input: (x, y, t) ∈ [0,1]^2 × {0,1}  (normalized pixel coord + frame_index)
    Sinusoidal positional encoding: coord → R^(POS_DIM × 2 × 3)
    Hidden layers: 3 × HIDDEN_DIM=64 with FiLM modulation per layer:
        h ← sin(linear(h) * scale_i + shift_i)
    Output: linear(h) → R^3 → sigmoid → rgb in [0, 1]^3
   |
   v   per pixel (x, y) ∈ [0, 384) × [0, 512)
   v
Stack: rgb_0, rgb_1 ∈ (B, 3, 384, 512)  [FULL contest scorer resolution; CC-J-6 unwind]
   |
   v
Score-aware loss (canonical Catalog #164 routing):
    L_score = score_pair_components(rgb_pair, gt_pair, segnet, posenet, eval_roundtrip=True)
   |
   v
MINE-based IB regularizer:
    I(z; frames) ≥ MINE_lower_bound(critic_θ(z, frames))  [Belghazi 2018 DV representation]
    L_IB = β · I_lower_bound  [empirical β-sweep {1e-5, 1e-4, 1e-3, 1e-2}; CC-J-3 unwind]
   |
   v
Sparse-Laplacian regularizer on FiLM matrices (MacKay-canonical):
    L_sparse = λ_sparse · |W_film|_1

Total loss:
    L = L_score + L_IB + L_sparse
```

## ## Canonical-vs-unique decision per layer (per Catalog #290)

[Reproduced verbatim from Phase 2 memo §Canonical-vs-unique decision per layer table.]

## ## Cargo-cult audit per assumption (per Catalog #303)

[See Phase 1 audit at `.omx/research/path_3_j_mdl_ibps_cargo_cult_audit_of_c6_scaffold_20260526.md` for the canonical 11-CC audit table. Phase 3 inherits all classifications; no new CCs surfaced during Phase 2 decision.]

## ## 9-dimension success checklist evidence (per Catalog #294)

[See Phase 2 memo §9-dimension success checklist evidence (Path (a) DISCRETE-CATEGORICAL-MINE-HYBRID evidence).]

## ## Predicted ΔS band (per Catalog #296)

**Predicted ΔS band:** `NULL pending Phase 3 L0 SCAFFOLD MLX smoke + Stage 4 post-training Tier-C ablation + Stage 5 paid-dispatch full-run.`

Per CC-J-1 unwind: NO predicted band is claimed; ANY band claimed in a future per-substrate symposium per Catalog #325 + operator-frontier-override per Catalog #199 MUST cite ONE of:
- (a) Dykstra-feasibility intersection on (rate, seg, pose, archive-bytes) polytope per Catalog #296
- (b) Shannon R(D) bound + first-principles derivation
- (c) probe-disambiguator path `tools/probe_path_3_j_mdl_ibps_*_disambiguator.py` (Phase 3 follow-on)

Same-line waiver `# PREDICTED_BAND_VIBES_OK:<rationale>` REJECTED per Catalog #296 strictness; the C6 v1 random-init Tier-C extrapolation IS the canonical anti-pattern this gate exists to prevent.

`predicted_band_validation_status: pending_post_training` per Catalog #324.

## ## Observability surface (per Catalog #305)

[See Phase 2 memo §Observability surface (6-facet inspection / decomposition / diff / query / cite-chain / counterfactual).]

## ## NEW 3-axis discipline per AMENDMENT #3

[See Phase 2 memo §NEW 3-axis discipline (Axis 1 math+scientific+engineering; Axis 2 MLX drift minimization; Axis 3 numpy portability).]

## ## Horizon-class classification (per Catalog #309)

**horizon_class: frontier_pursuit**

[See Phase 2 memo §Horizon-class classification for full justification.]

## ## Catalog #124 archive-grammar 8 fields (declared inline)

- ``archive_grammar``: monolithic single-file ``0.bin`` (MDLIBPS-J1 grammar)
- ``parser_section_manifest``: 32-byte header (magic ``MIBJ1\x00`` + version u8 + K=16 u8 + G=12 u8 + HIDDEN_DIM u16 + NUM_HIDDEN_LAYERS u8 + POS_DIM u8 + NUM_PAIRS u16 + EVAL_H u16 + EVAL_W u16 + BASE_BLOB_LEN u32 + MINE_BLOB_LEN u32 + INDICES_BLOB_LEN u32 + META_BLOB_LEN u32 + reserved u8 × 3) + BASE_BLOB_LEN bytes brotli(q=9) procedural-coord-MLP + FiLM-proj state_dict + MINE_BLOB_LEN bytes brotli(q=9) MINE critic state_dict (provenance only; not consumed at inflate) + INDICES_BLOB_LEN bytes brotli(q=9) per-pair categorical indices (NUM_PAIRS × G × 4 bits packed → NUM_PAIRS × G / 2 bytes) + META_BLOB_LEN bytes sorted-keys JSON utf-8 (scale_modulation, num_pairs, eval_hw, schema_version, ...)
- ``inflate_runtime_loc_budget``: ≤200 LOC substrate-engineering waiver (procedural-coord-MLP forward + FiLM modulation + sigmoid + canonical bilinear → contest HW per Catalog #205 device-fork helper)
- ``runtime_dep_closure``: torch + brotli only (HNeRV parity L4 ≤ 2 deps; canonical Catalog #146 + #205 + #295 self-containment)
- ``export_format``: MDLIBPS-J1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``MDLIBPSJScoreAwareLoss`` routes through canonical ``score_pair_components`` per Catalog #164; eval-roundtrip mandatory
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity L7); procedural-coord-MLP + FiLM + MINE + categorical-indices composition is substrate engineering
- ``no_op_detector_planned``: emit/parse roundtrip preserves bytes byte-for-byte; INDICES_BLOB + BASE_BLOB are frame-affecting at inflate per Catalog #220/#272 distinguishing-feature contract; MINE_BLOB is provenance/training-only (Catalog #220 declaration: NOT score-affecting at inflate); META_BLOB is parse/config gate

## ## 6-hook wire-in declaration (per Catalog #125)

- **hook #1 sensitivity-map** = ACTIVE: per-pair categorical-index distribution is the per-pair sensitivity primitive; register `sensitivity_map.path_3_j_mdl_ibps_v1` (planned post-Stage-1 smoke)
- **hook #2 Pareto constraint** = ACTIVE: `(rate_categorical_indices ≤ 4 KB) ∩ (rate_base_decoder ≤ 50 KB) ∩ (rate_film_matrices ≤ 10 KB) ∩ (S(θ) ≥ canonical frontier per .omx/state/canonical_frontier_pointer.json - ε)`; register `tac.pareto.mdl_ibps_j_v1`
- **hook #3 bit-allocator** = ACTIVE: β + sparse-Laplacian λ_sparse + categorical alphabet K + group count G are the bit-allocator knobs; register `bit_allocator.mdl_ibps_j_v1`
- **hook #4 cathedral autopilot dispatch** = ACTIVE: cathedral consumer at `tac.cathedral_consumers.mdl_ibps_j_routing_consumer/` (Phase 3 follow-on per Catalog #335 canonical contract) routes substrate candidates per Catalog #341 canonical non-promotable markers
- **hook #5 continual-learning posterior** = ACTIVE: every empirical anchor (per-arm β + Tier-C density + final-score decomposition) emits canonical posterior anchor per Catalog #300 v2 frontmatter via `tac.council_continual_learning.append_council_anchor`
- **hook #6 probe-disambiguator** = ACTIVE: β-sweep probe `tools/probe_path_3_j_mdl_ibps_beta_sweep_disambiguator.py` (Phase 3 follow-on) emits per-arm score decomposition for empirical β-optimum derivation per Catalog #1265 MLX-first gate + post-training Tier-C verdict per Catalog #324

## ## Reactivation criteria (per CLAUDE.md "Forbidden premature KILL")

- Stage 0-2 MLX-first smokes (FREE; CPU): MUST pass Catalog #1265 gate threshold 0.001 BEFORE Stage 3 paid CUDA authorization
- Stage 3 Modal A10G β-sweep ($20-60): authorization requires per-substrate symposium per Catalog #325 + operator-frontier-override per Catalog #199
- Stage 4 post-training Tier-C ablation (FREE; CPU): verdict ACROSS_CLASS (density < 0.30) → Stage 5 authorized; verdict WITHIN_CLASS (density ≥ 0.70) → Catalog #307 implementation-level falsification + Catalog #308 sister-path enumeration; NO KILL
- Stage 5+ paid dispatches: per-arm Tier-C re-measurement per Catalog #324 mandatory; paired contest-CUDA + contest-CPU per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"

## ## Cross-references

- Phase 1: `.omx/research/path_3_j_mdl_ibps_cargo_cult_audit_of_c6_scaffold_20260526.md`
- Phase 2: `.omx/research/path_3_j_mdl_ibps_substrate_design_decision_20260526.md`
- Parent C6 substrate: `src/tac/substrates/c6_e4_mdl_ibps/` (preserved per Catalog #110/#113)
- Sister A=DreamerV3 RSSM: `src/tac/substrates/dreamer_v3_rssm/` (categorical posterior K=256 × G=24 reference)
- Sister K=COIN++: `src/tac/substrates/coin_pp_implicit_neural_representation/` (procedural+continuous-FiLM reference; 3-axis discipline)
- Sister F=Z8 hierarchical: `src/tac/substrates/z8_hierarchical_predictive_coding/` (hierarchical IB binding reference)
- CLAUDE.md Catalogs #124, #125, #146, #164, #192, #205, #215, #220, #226, #229, #240, #244, #270, #287, #290, #292, #294, #295, #296, #303, #305, #309, #317, #324, #325, #341, #344, #1265
- Tishby & Zaslavsky 2015 IB; Rissanen 1978 MDL; Belghazi 2018 MINE; Jang 2016 Gumbel-Softmax; Higgins 2017 β-VAE; Hafner 2024 DreamerV3; Perez 2017 FiLM; MacKay 2003 sparse-Laplacian; Olshausen-Field 1996 sparse coding

---

**Status:** PHASE 3 DESIGN MEMO LANDED 2026-05-26. L0 SCAFFOLD code files land in same commit batch under `src/tac/substrates/mdl_ibps_j_discrete_categorical_mine_hybrid/`. Landing memo follows at `.omx/research/path_3_j_mdl_ibps_L0_scaffold_landed_20260526.md`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>


# HIERARCHICAL_PREDICTIVE_CODING_QUADRUPLE_OK:design_memo_references_hierarchical_predictive_coding_in_cross_reference_or_partial_subset_context_NOT_as_primary_substrate_binding_all_four_Rao_Ballard_Mallat_DreamerV3_WynerZiv_canonical_primitives_simultaneously_per_catalog_312_pattern_i_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
