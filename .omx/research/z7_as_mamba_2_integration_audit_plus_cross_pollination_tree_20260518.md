---
name: z7-as-mamba-2-integration-audit-plus-cross-pollination-tree-20260518
metadata:
  node_type: memory
  council_tier: T1
  council_attendees:
    - Quantizr
    - Contrarian
    - Assumption-Adversary
  council_quorum_met: false
  council_verdict: AUDIT_COMPLETE
  council_predicted_mission_contribution: rigor_overhead
  council_override_invoked: false
  council_override_rationale: ""
  horizon_class: asymptotic_pursuit
  deferred_substrate_id: time_traveler_l5_z7_mamba2
  substrate_aliases:
    - z7_mamba2
    - z7_as_mamba_2
  predicted_dispatch_risk: 0
  originSessionId: lane_z7_as_mamba_2_full_landing_20260518
  related_deliberation_ids:
    - council_symposium_z7_as_mamba_2_full_landing_20260518
    - z7_integration_audit_20260518
---

# Z7-as-Mamba-2 INTEGRATION AUDIT + CROSS-POLLINATION TREE 2026-05-18

**Lane**: `lane_z7_as_mamba_2_full_landing_20260518`
**Substrate**: `time_traveler_l5_z7_mamba2`
**Scope**: Audit Z7-Mamba-2 against the 6 mandatory canonical wire-in hooks per CLAUDE.md "Subagent coherence-by-default" non-negotiable AND enumerate composition opportunities with sister substrates (DP1, PR101_lc_v2, PR106_format0d, ATW V2-1, lane_g_v3+siren, TOP-3 reclamation, A1-SPECIALIZED).

This memo extends the sister audit `.omx/research/z7_integration_audit_20260518.md` (Z7-Mamba-2 + Z7-LSTM unified) at the per-substrate FULL LANDING level — focused on cross-pollination opportunities BETWEEN Z7-Mamba-2 and the rest of the lattice.

## TL;DR (60 seconds)

| Composition target | Composability | Estimated α | Sister substrate ref |
|---|---|---|---|
| **DP1 (pretrained driving prior)** | **HIGH-PAIR-CANDIDATE** (codebook init for Mamba-2 state) | TBD (untested) | `tac.substrates.pretrained_driving_prior` |
| **PR101_lc_v2** (HNeRV-class) | MEDIUM-PAIR-CANDIDATE (score-aware curriculum) | TBD | `submissions/pr101_lc_v2_clone/` |
| **PR106_format0d** (latent score table) | MEDIUM-PAIR-CANDIDATE (per-pair score-table augmentation) | TBD | `tac.substrates.pr106_format0d_latent_score_table` |
| **ATW V2-1** (Atick-Tishby-Wyner codec) | HIGH-PAIR-CANDIDATE (per-region SegNet softmax channel) | TBD | `tac.symposium_impls.atw_codec_atick_tishby_wyner_v1` (queued V2-1) |
| **lane_g_v3 + siren topology** (Stage 2 stack pending) | LOW-PAIR-CANDIDATE (architecture-level substitute, not composition) | N/A | `lane_super_additive_lane_g_v3_siren_topology_integration_20260517` |
| **TOP-3 reclamation** (A1-SPECIALIZED + F4 + F5) | MEDIUM-PAIR-CANDIDATE (per-pattern inverter feed) | TBD | `experiments/results/top_3_reclamation_local_cpu_smoke_20260519T003210Z` |

**Net**: Z7-Mamba-2 has 4 HIGH/MEDIUM-PAIR-CANDIDATE composition opportunities. The single most-promising is **DP1 + Z7-Mamba-2** (DP1's codebook initializes Mamba-2's selective-projection matrices A/B/C) — this is the canonical "alien tech" composition per operator standing directive 2026-05-18.

## 1. Six-hook wire-in audit (per Catalog #125)

This complements `.omx/research/council_symposium_z7_as_mamba_2_full_landing_20260518.md` §7 with operational detail.

### Hook 1: Sensitivity-map contribution (`tac.sensitivity_map.*`)

**Canonical module**: `src/tac/sensitivity_map/{__init__.py,axis_weights.py,wyner_ziv_reweight.py}` (35.2 KB + 16.3 KB + 16.1 KB)

**Z7-Mamba-2 specific signal**: Mamba-2 selective-projection gradient norms (`A_proj`, `B_proj`, `C_proj`, `dt_proj`, `in_proj`, `out_proj`) ARE the per-tensor importance signal. The selective state-space mechanism naturally produces input-conditional sensitivity weights at each timestep.

**Wire-in path**: Register `tac.sensitivity_map.time_traveler_l5_z7_mamba2.*` at PATH C 100ep auth_eval anchor per per-substrate symposium PROCEED-unconditional. Until then, DEFERRED-N/A.

### Hook 2: Pareto constraint (`tac.boosting.pareto_front`)

**Z7-Mamba-2 specific constraint**: `mamba2_residual_entropy ≤ ε_residual` on the convex feasibility region. The Pareto frontier consumes empirical anchors; without a 100ep auth_eval baseline, the constraint cannot be calibrated. DEFERRED-N/A.

### Hook 3: Bit-allocator hook

**Z7MCM2 archive own byte budget**: per-tensor int8 quantization for latent_init/residuals/ego_motion (1 byte/element); fp16 zlib for encoder/decoder/predictor state_dicts (variable compression). The canonical `tac.bit_allocator.*` wire-in deferred to Wave-N+1 trainer-build per parent symposium Revision #6.

### Hook 4: Cathedral autopilot dispatch (STRUCTURALLY ACTIVE)

**Recipe path**: `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml`. Loaded by `tools/cathedral_autopilot_autonomous_loop.py`. Currently filtered out by `research_only=true + dispatch_enabled=false` per Catalog #240 + Catalog #325 per-substrate symposium discipline.

**Activation criteria**: per-substrate symposium PROCEED-unconditional (THIS memo is PROCEED_WITH_REVISIONS; PATH C 100ep anchor required to elevate to PROCEED-unconditional).

### Hook 5: Continual-learning posterior (ACTIVE)

**Canonical helper**: `tac.council_continual_learning.append_council_anchor`. THIS landing emits the sister symposium memo `council_symposium_z7_as_mamba_2_full_landing_20260518.md` which carries the canonical YAML frontmatter consumed by the posterior.

### Hook 6: Probe-disambiguator (ACTIVE)

- **identity_predictor mode**: Mamba2Predictor with `identity_predictor=True` returns z_prev unchanged with 0 trainable parameters; canonical sister to Z6/Z7-GRU identity-predictor pattern.
- **static_capacity_control**: `_static_control_archive_pair` in trainer emits same-byte-budget identity-predictor archive; PATH C 100ep paired-comparison anchors recurrent-Mamba-2-WIN at ΔS ≥ 0.005 on contest-CUDA = canonical Z7-Mamba-2 promotion criterion.
- **Sister probe tool**: `tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` (Z7-LSTM canonical; ready to extend with `--substrate z7_mamba2` flag at next iteration).

## 2. Per-substrate cross-pollination matrix

This is a 7×7 hypergraph following the canonical pattern from commit `38db94424` (dynamic per-candidate composition framework design memo). Each row/column is a candidate substrate; intersection = composition opportunity classification.

### Composition opportunity #1: DP1 + Z7-Mamba-2 (HIGH-PAIR-CANDIDATE)

**Mechanism**: DP1's codebook (`tac.substrates.pretrained_driving_prior.codebook.distill_codebook`) produces ~5-10 KB binary distilled codebook from out-of-distribution Comma2k19 dashcam data. The codebook IS a feature-space prior that could initialize:

- **Initial latent**: `Z7Mamba2PredictiveCodingSubstrate.latent_init` parameter could be set to the DP1 codebook centroid most relevant to the contest video's first-pair domain (instead of `torch.randn() * 0.02`).
- **Mamba-2 input projection**: `Mamba2Predictor.input_projection` weight could be initialized to project from (z_prev + ego_motion) space into a basis aligned with DP1 codebook subspace (warm-start from out-of-distribution priors).

**Predicted composition α**: TBD; likely SUPER-ADDITIVE if DP1 codebook captures generic dashcam visual priors that contest video's ego-motion-conditioned next-frame prediction can exploit. Sister of A-STACK Pareto composition pattern.

**Cost to test**: $0 GPU (DP1 distillation is local OoD-only; Z7-Mamba-2 substrate just consumes the resulting codebook bytes). Estimated +200-500 LOC for the integration.

**Reactivation criterion**: After PATH C Z7-Mamba-2 100ep anchor lands, run paired Z7-Mamba-2 (random init) vs Z7-Mamba-2-DP1-init exact eval at same archive bytes. ΔS ≥ 0.005 = SUPER-ADDITIVE confirmed.

### Composition opportunity #2: ATW V2-1 + Z7-Mamba-2 (HIGH-PAIR-CANDIDATE)

**Mechanism**: ATW V2-1 (Atick-Tishby-Wyner codec; per-region SegNet softmax channel queued per cargo-cult-unwind audit) emits per-region SegNet softmax logits as a sidecar. Z7-Mamba-2's ego_motion buffer (8-dim) could be EXTENDED with N additional channels carrying compressed per-region SegNet softmax features at each pair.

**Predicted composition α**: TBD; canonical OUTPUT-OUTPUT-conditioning pattern per Atick-Redlich 1990 cooperative-receiver framework. Sister to Z6 Candidate 4c canonical FiLM-cond scorer-logit-compressed channel.

**Cost to test**: $0 GPU at MPS proxy; ATW V2-1 already designed in `.omx/research/atw_codec_v1_cargo_cult_unwind_design_20260516.md`. Estimated +300-800 LOC.

**Reactivation criterion**: After ATW V2-1 design memo per-region selection completes (channel #1 SegNet softmax was ranked #1 by Atick), the canonical Z7-Mamba-2-ATW-V2-1 stack is the canonical "Z7 + cooperative-receiver" composition.

### Composition opportunity #3: PR101_lc_v2 + Z7-Mamba-2 (MEDIUM-PAIR-CANDIDATE)

**Mechanism**: PR101_lc_v2 is the HNeRV-class FRONTIER substrate (0.193 PR101 GOLD baseline). Its score-aware training curriculum (β-IB Lagrangian + λ_R warmup) could replace Z7-Mamba-2's current `--loss-mode score_aware` simple Lagrangian for Wave-N+2 training.

**Predicted composition α**: MEDIUM; likely SATURATING since both substrates already use canonical score_pair_components_dispatch loss. Possible SUPER-ADDITIVE if PR101's curriculum better aligns Mamba-2's per-pair MI bottleneck with contest score components.

**Cost to test**: $0 GPU; pure design-memo work + ~200-400 LOC training loop refactor.

### Composition opportunity #4: PR106_format0d + Z7-Mamba-2 (MEDIUM-PAIR-CANDIDATE)

**Mechanism**: PR106_format0d is the per-pair score-table latent augmentation that landed 0.20533 [contest-CUDA] anchor. The per-pair score table could augment Z7-Mamba-2's `residuals` parameter — providing a sister low-rate channel for per-pair score-correction at inflate time.

**Predicted composition α**: MEDIUM; sister to per-pair residual concept but at different abstraction level (score-table at inflate-time per-pair tone-mapping vs Mamba-2 residual at training-time per-pair latent correction).

### Composition opportunity #5: lane_g_v3 + siren topology + Z7-Mamba-2 (LOW-PAIR-CANDIDATE)

**Mechanism**: lane_g_v3 + siren topology is the recent SUPER_ADDITIVE candidate (α=4.74 per sister #823 anchor, BUT flagged FALSE_SIGNAL ARTIFACT per byte-identity ROOT CAUSE). The siren topology is an architecture-level substitute, NOT a composition with Z7-Mamba-2 — they would replace each other as substrates rather than stack.

**Verdict**: NOT a composition; documented for completeness.

### Composition opportunity #6: TOP-3 reclamation + Z7-Mamba-2 (MEDIUM-PAIR-CANDIDATE)

**Mechanism**: TOP-3 reclamation just validated 3-of-3 advisory positive (per `tools/build_top_1_a1_local_cpu_advisory_smoke.py` + sister TOP-2/TOP-3 tools at commit `ade065879`). The binary-distillation framework is available as a composition stack for Z7-Mamba-2 IF Z7-Mamba-2 produces a per-pair signal that the per-pattern inverter binary can consume.

**Predicted composition α**: TBD; cross-substrate-class. Requires PATH C 100ep Z7-Mamba-2 first.

### Composition opportunity #7: A1-SPECIALIZED + Z7-Mamba-2 (MEDIUM-PAIR-CANDIDATE)

**Mechanism**: A1-SPECIALIZED is the current canonical CPU-axis frontier (0.19205 [contest-CPU] per archive sha `6bae0201`). A1-SPECIALIZED's FEC6 fixed Huffman k=16 latent encoding could serve as a sister low-rate channel composable with Z7-Mamba-2's recurrent latent stream.

## 3. Composition_alpha N-way analysis per Catalog #322

Per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden autopilot-adjustment-derived-from-phantom-provenance-composition-alpha" + Catalog #322 sister to Catalog #321:

The 7 composition opportunities listed above are ALL TBD α until empirical paired-comparison anchors land. Per the canonical 3-cascade decision tree (per Catalog #319 sister gate Q2/Q3 landing 2026-05-17):

1. **CASCADE 1 (PRIMARY)**: Lagrangian-dual `OptimalPerPairTreatmentPlan` sidecar (none exists yet for Z7-Mamba-2; gate fires only when PATH C 100ep anchor produces master-gradient evidence).
2. **CASCADE 2 (DELIVERABILITY)**: `DeliverabilityProof` (none exists yet; pre-empirical).
3. **CASCADE 3 (PASSTHROUGH)**: 1.0× factor (canonical default until a Cascade 1 or 2 evidence row lands).

All 7 composition opportunities are at CASCADE 3 PASSTHROUGH until empirical anchors land. NO false-authority composition α can leak into autopilot ranking from this memo.

## 4. Predicted dispatch sequencing

Per Z7 parent symposium Revision #6 cascade + per-substrate symposium PATH A → PATH B → PATH C ordering (this memo's sister §4):

1. **Phase 0** (CURRENT): Substrate BUILT + design memo + symposium memo + integration audit. dispatch_enabled=false. $0 spent.
2. **Phase 1** (NEXT — $0 cost): PATH A MPS proxy training on M5 Max + sister Z7-LSTM same-config MPS proxy training. Compare proxy-loss-stream + archive-byte-distinguishing-feature consumption + identity-predictor delta. If Z7-Mamba-2 wins by margin > 0.005 proxy delta, escalate.
3. **Phase 2** ($5-15): PATH B Modal T4 50ep smoke. Harvest archive sha. Run Catalog #324 post-training Tier-C density validation. If density signature matches canonical pattern, escalate.
4. **Phase 3** ($20-30): PATH C Modal A100 100ep full + paired contest-CPU/CUDA exact eval. If score lands within [0.167, 0.184] AND beats Z7-LSTM at same archive bytes, promote to L2 + sensitivity-map wire-in.
5. **Phase 4** (composition): Add DP1 codebook init OR ATW V2-1 per-region SegNet channel; PATH B → PATH C cycle for the composition.

## 5. Operator-routable next-step

- **op-routable #1 ($0)**: Update recipe `predicted_band_validation_status: research_prior_prebuild` → `pending_post_training` per Catalog #324 since substrate is now COMPLETE.
- **op-routable #2 ($0)**: Run PATH A MPS proxy: `Z7_MAMBA2_DEVICE=mps Z7_MAMBA2_TRAINER_MODE=timing_smoke bash scripts/remote_lane_substrate_time_traveler_l5_z7_mamba_2.sh` on local M5 Max; harvest MPS-research-signal stats; compare vs Z7-LSTM same-config.
- **op-routable #3 ($0)**: Pre-register `lane_z7_mamba2_plus_dp1_codebook_init_composition_20260520` at L0 for the canonical Phase-4 composition target.
- **op-routable #4 (council)**: Convene Wave-N+1 per-substrate symposium on landed Z7-Mamba-2 archive bytes (NOT on this scaffold/landing smoke); operator-routable to apparatus when ready.

## 6. Cross-references

- Sister symposium memo: `.omx/research/council_symposium_z7_as_mamba_2_full_landing_20260518.md`
- Parent design memo: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- Z7-Mamba-2 + Z7-LSTM unified audit: `.omx/research/z7_integration_audit_20260518.md`
- Parent Z7 symposium: `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md`
- DEEP-RESEARCH-WAVE TOP-5 #2: `.omx/research/comprehensive_research_wave_20260518.md`
- Dynamic per-candidate composition framework: commit `38db94424` design memo
- A-STACK Pareto composition pattern: `tac.optimization.substrate_composition_matrix.canonical_substrate_inventory`
- Cathedral autopilot ranker: `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2`
- TOP-3 reclamation framework: `experiments/results/top_3_reclamation_local_cpu_smoke_20260519T003210Z`
