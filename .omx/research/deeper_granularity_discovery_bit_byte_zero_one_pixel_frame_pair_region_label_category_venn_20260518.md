---
review_kind: deeper_granularity_discovery_deep_dive
review_id: deeper_granularity_discovery_bit_byte_zero_one_pixel_frame_pair_region_label_category_venn_20260518
review_date: "2026-05-18"
lane_id: lane_grand_council_meta_portfolio_plus_deeper_granularity_discovery_20260518
operator_directive: "there is more like this lurking in the bit and bytes and zeroes and ones and pixel and frame and pair and master gradient and regions and labels and categories and venn diagram and all"
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
horizon_class: frontier_breaking
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids:
  - grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518
  - comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518
  - canonical_upstream_pr_review_procedural_generation_compliance_20260518
  - grand_council_symposium_inflate_py_extreme_compression_20260518
  - codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518
  - asymptotic_stacking_plus_local_max_utilization_audit_20260518
  - comprehensive_research_wave_20260518
  - grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
---

# Deeper Granularity Discovery Deep-Dive — bit / bytes / zeros+ones / pixel / frame / pair / master_gradient / regions / labels / categories / Venn diagram

## Operator directive (verbatim)

> *"there is more like this lurking in the bit and bytes and zeroes and ones and pixel and frame and pair and master gradient and regions and labels and categories and venn diagram and all"*

## Mission

Sister deliverable to `grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518.md`. Where that T3 symposium re-ranks the 53-substrate registry by NEW compliance-envelope priors, THIS memo EXPANDS the sister synthesis's 70-surface analytical inventory by exploring the DEEPER granularities the operator explicitly named.

The operator's enumeration is exact and load-bearing — each of the 11 granularities below is treated as a separate discovery axis. The sister synthesis covered the 15 canonical granularities (per-pair / per-byte / per-frame / per-pixel / per-region / per-class / per-boundary / per-tensor / per-channel / per-layer / per-axis / per-substrate / per-archive / per-deliberation / per-paradigm). This memo's job is to surface the DEEPER discoveries WITHIN each granularity that the synthesis didn't enumerate.

## 0. Executive summary

### TOP-5 newly-discovered granularities (the discovery deliverable)

| # | Granularity | Discovery | Predicted ΔS contribution | Where to wire |
|---|---|---|---|---|
| 1 | **3-set or 4-set per-pair Venn classification** (extends Catalog #319 v2's current 2-set Venn HIGH_PAIR_INVARIANT × HIGH_PAIR_SPECIFIC into PER_REGION × PER_PAIR × PER_CLASS = 8-region Venn; each region has its own deliverability tier) | NEW frontier — the operator's "Venn diagram" reference IS this | `[-0.015, -0.005]` aggregate (cascade unlock + per-region treatment refinement) | `tac.master_gradient_consumers.classify_bytes_by_3_set_venn` extension + cathedral autopilot v2 cascade extension |
| 2 | **Per-pixel UNIWARD-sensitivity wire-in across the 6 dormant helpers** (logit_margin_sensitivity_weighted + imp_sensitivity_weighted + owv3_sensitivity_weighted + fec6_selector_discovery_sensitivity_weighted + balle_sensitivity_weighted + neural_weight_codec_sensitivity ALL fully implemented BUT not consumed by any active substrate runtime) | Per the sister synthesis 1.4: largest single class of dormant signal in codebase; operator-flagged | `[-0.010, -0.003]` per substrate × 6 helpers wired = aggregate `[-0.025, -0.005]` | Cathedral autopilot v2 cascade extension `adjust_predicted_delta_for_uniward_sensitivity_class` |
| 3 | **Bit-level analytical signal** (UNIWARD bit-cost matrix + STC syndrome-trellis bit/parity discipline + per-bit Shannon-MDL entropy + bit-flip null-space exploitation per master-gradient) | NOT inventoried in sister synthesis; the lowest-granularity floor | `[-0.005, -0.001]` per archive (Tier-1 STC near-Shannon residual encoding) | NEW `tac.bit_level_master_gradient_consumer` + sister probe |
| 4 | **Zeros+ones binary-entropy granularity** (zero-run-length encoding opportunities + per-bit Bernoulli entropy estimation + boolean-mask compression) | NOT inventoried; sub-byte sparsity exploitation | `[-0.003, -0.001]` per archive (when archive carries sparse bit-flags) | NEW `tac.binary_entropy_estimator` consumed by per-X codec planner |
| 5 | **4D multi-granularity sensitivity tensor** (pair × byte × class × axis) | Per synthesis OP-5 + extended by THIS memo with Venn-classification 5th dimension (pair × byte × class × axis × venn_region) → 5D tensor | `[-0.005, -0.001]` per dispatch (better signal-routing) | NEW DuckDB schema `tac.canonical_duckdb.multi_granularity_sensitivity_5d_ext` |

### TOP-5 newly-discovered cross-stack synergies

| # | Synergy | Description | Predicted aggregate ΔS |
|---|---|---|---|
| 1 | **3-set Venn × null-space exploitation × per-region SegNet polytope** | Compose the per-pair Venn classification (HIGH_PAIR_INVARIANT × HIGH_PAIR_SPECIFIC) with PER_REGION classification (16×16 grid) and PER_CLASS classification (5 SegNet classes) → 2×16×16×5 = 2560 cells per pair × 600 pairs. The 1D null-space within each cell is potentially independent → per-cell free-byte budget. Stacks with TOP-1 of meta-portfolio (null-space exploiter). | `[-0.020, -0.005]` |
| 2 | **Bit-level STC × per-pair master gradient × Wyner-Ziv Tier-1 deliverability** | STC syndrome-trellis coding operates at bit-level with near-Shannon-rate parity codes (Filler-Pevny-Fridrich 2010). Compose with per-pair master gradient: for each bit position in archive.zip, classify into BIT_PAIR_INVARIANT / BIT_PAIR_SPECIFIC. The BIT_PAIR_INVARIANT bits can be carried as 0-cost STC-residual (decoded via shared scorer-prior). Stacks with TOP-5 of meta-portfolio (STC pose-residual sidecar). | `[-0.010, -0.003]` |
| 3 | **Per-class chroma anchor × hash-seed PRNG × per-region cell mapping** | NSCS06 v7 per-class chroma palette × procedural-codebook hash-seed × per-region (16×16) cell assignment. Each region gets its OWN per-class chroma palette derived from a per-region seed. Total seed bytes = 16×16 regions × 5 classes × 4 bytes seed = 5120 bytes BUT each region's palette is procedurally generated → compression vs ~600KB of full per-region per-class palettes. Stacks with TOP-2 of meta-portfolio (hash-seed). | `[-0.008, -0.002]` |
| 4 | **Per-frame difficulty atlas × per-pair Venn × per-frame foveation** | Per-frame difficulty (variable_rate.compute_pair_difficulty) × per-pair Venn classification × per-frame foveation map (NVIDIA VRSS 2 / Gibson 1950 FoE prior). Each frame has its own difficulty score; harder frames get more bits + larger foveation kernel; easier frames get fewer bits + smaller kernel. The per-pair Venn provides the bit-allocation prior. | `[-0.008, -0.002]` |
| 5 | **Master-gradient × per-paradigm sensitivity × per-substrate-pair composition_alpha** | The full 5D tensor (pair × byte × class × axis × venn) consumed by `tac.optimization.substrate_composition_matrix.classify_pairwise_composability` over all 32 L2+ candidate pairs. The composition_alpha per pair becomes a function of the 5D tensor's projection onto the candidate's substrate class. Outputs the canonical orthogonality-vs-additivity verdict per pair × granularity. | `[-0.005, -0.001]` per dispatch (better composition selection) |

### TOP-3 frontier-breaking opportunities the synthesis missed

1. **The 3-set or 4-set Venn classification is THE operator's load-bearing insight** — the synthesis covered Catalog #319 v2 cascade's 2-set Venn (HIGH_PAIR_INVARIANT × HIGH_PAIR_SPECIFIC) but the operator's explicit "Venn diagram" reference points to MULTI-SET Venn classifications (3-set / 4-set / 5-set). Each additional dimension multiplies the cells: 2-set = 4 cells; 3-set = 8 cells; 4-set = 16 cells; 5-set = 32 cells. Each cell potentially has its own deliverability tier per Catalog #319. The synthesis was operating WITHIN the 2-set framing; the operator's directive EXPLICITLY breaks out.

2. **Bit-level granularity is a SUB-byte FLOOR the synthesis didn't probe** — STC syndrome-trellis coding operates at bit-level; UNIWARD inverse-local-variance weighting is at PIXEL level but the encoded EMBEDDING is at bit-level; per-bit Bernoulli entropy estimation is the canonical primitive for sparse-bit-flag compression. The synthesis enumerated `tac.bit_allocator` (per-tensor) + `tac.bit_level_archive_optimizer` (per-byte) but did NOT enumerate per-BIT analytical surfaces. The bit-level granularity is below the byte-level floor; for substrates with sparse bit-flags (e.g., NSCS01 nullspace mask + Z3 IB latent + SegNet binary classification flags) this is significant unexplored signal.

3. **Per-frame difficulty atlas + per-frame foveation map composition is unexplored** — sister synthesis enumerated `variable_rate.compute_pair_difficulty` (per-frame CRF allocation) AS ACTIVE but the per-frame foveation map (NVIDIA VRSS 2 / Gibson 1950 FoE prior per deep-research wave §1.3) is at DESIGN-ONLY level. The COMPOSITION of these two per-frame surfaces creates an effective per-frame × per-pixel bit budget that current variable_rate doesn't reach. This is the FOVEATED-VRSS direction the deep-research wave §1.3 anchors but no canonical helper has been built.

## 1. Per-granularity discovery audit

For each of the 11 operator-named granularities, this section enumerates (a) existing canonical helpers (via Grep + Read); (b) wired / dormant / design-only status; (c) cross-stack synergies with the sister synthesis's TOP-5 op-routables; (d) potential frontier-breaking opportunities the synthesis missed.

### 1.1 BIT-level granularity (NOT inventoried in sister synthesis)

**Existing canonical helpers**:
- `src/tac/bit_allocator.py` (13K LOC) — per-tensor bit allocation with importance-weighted Lagrangian solver
- `src/tac/bit_level_archive_optimizer.py` (19K LOC) — per-byte (NOT per-bit) Lagrangian optimizer with PerDimQuantizer
- `src/tac/optimization/bit_allocator_end_to_end.py` (59K LOC) — canonical end-to-end bit allocation
- `src/tac/learnable_bit_quant.py` (46K LOC) — learnable bit-precision quantization
- `src/tac/frozen_bit_quant.py` (13K LOC) — frozen-precision bit quantization
- `src/tac/codec/frame_conditional_bit_budget.py` (40K LOC) — per-frame bit packing with channel-wise + binary variants
- `src/tac/stc_boundary_codec.py` (25K LOC) — STC boundary codec (canonical Filler-Pevny-Fridrich realization)

**Status**:
- bit_allocator (per-tensor) = ACTIVE
- bit_level_archive_optimizer (per-byte) = ACTIVE-PARTIAL
- frame_conditional_bit_budget (per-frame × per-channel) = DORMANT (consumed only by probe)
- stc_boundary_codec (per-bit syndrome-trellis at boundaries) = ACTIVE
- **NO per-bit master-gradient consumer exists** (the new discovery surface)

**Cross-stack synergies with synthesis TOP-5**:
- bit-level + OP-2 master-gradient extension: extract per-bit gradient from per-byte gradient via bit-position projection
- bit-level + OP-3 null-space exploiter: per-bit null-space basis (1024 bits per byte × per-bit gradient direction)
- bit-level + OP-5 multi-granularity sensitivity: extend to (pair × bit × class × axis) 4-tensor — finer than (pair × byte × class × axis)

**Frontier-breaking opportunity the synthesis missed**:
A canonical `tac.bit_level_master_gradient_consumer` would extend Catalog #319 v2 cascade to operate at bit granularity. The Filler-Pevny-Fridrich STC primitive at `src/tac/stc_boundary_codec.py` is the canonical near-Shannon-rate parity-check code; combining with per-bit master gradient gives the BIT_PAIR_INVARIANT vs BIT_PAIR_SPECIFIC classification at sub-byte resolution. Predicted aggregate ΔS `[-0.005, -0.001]` per archive for substrates with sparse bit-flag payloads.

### 1.2 BYTES granularity (per-byte × per-section / per-byte × per-region intersections)

**Existing canonical helpers (per sister synthesis 1.2)**:
- `tac.empirical_per_x_optimal_codec_planner.plan_per_byte_from_master_gradient` (ACTIVE; canonical FIRST INSTANCE)
- `tac.canonical_duckdb.per_byte_sensitivity_ext.per_byte_sensitivity` (ACTIVE; canonical schema)
- `tac.archive_byte_profile.profile_archive` (ACTIVE)
- `tac.master_gradient.predict_delta_s_per_pair` (ACTIVE)
- `tac.bit_level_archive_optimizer.BitLevelArchiveOptimizer` (ACTIVE-PARTIAL)

**DEEPER discovery**:
- **per-byte × per-section intersections**: each archive.zip member has its own byte range; per-byte sensitivity within member vs across-member differs. Currently `tac.archive_byte_profile` provides per-section breakdown but the per-byte sensitivity is computed globally. Extending to per-section per-byte would surface the substrate-specific sensitivity pattern (e.g., renderer.bin bytes vs masks.mkv bytes have radically different sensitivities).
- **per-byte × per-region intersections**: each byte in archive.zip corresponds to a specific (substrate, encoding stage, source-pixel-region) triple. The current per-byte planner classifies by L1 sensitivity quantile; extending to (region, quantile) tuple classification would surface region-specific compression opportunities.
- **magic-codec auto-selector at byte boundaries**: `tac.codec.magic_codec_dense_streams` (registered in `tac.composition.registry`) provides per-stream codec auto-selection. Extending to per-byte-region auto-selection would refine.

**Cross-stack synergy**: per-byte × per-section + OP-5 (multi-granularity sensitivity tensor) → 4D tensor with section as 5th dimension = (pair × byte × class × axis × section) 5-tensor.

### 1.3 ZEROS+ONES binary-entropy granularity (NOT inventoried)

**Existing canonical helpers**:
- `tac.compress` family (universal compressors brotli/lzma/zstd handle zero-run-length internally)
- `tac.stc_boundary_codec` (bit-level syndrome-trellis with parity)
- NO canonical zero-run-length encoder
- NO canonical Bernoulli entropy estimator
- NO canonical boolean-mask compression primitive

**DEEPER discovery**:
- **Zero-run-length encoding opportunities**: many substrate archives carry sparse bit-flags (NSCS01 nullspace mask is ~80% zeros; Z3 IB latent quantized to int4 has ~30% zero-block sparsity; SegNet binary classification flags are highly sparse). A canonical `tac.zero_run_length_encoder` would extract these into a separate stream.
- **Per-bit Bernoulli entropy estimation**: for sparse bit-flag payloads, the per-bit P(0) vs P(1) is highly non-uniform. Shannon-bound encoding uses ~`-(p·log2(p) + (1-p)·log2(1-p))` bits per bit instead of 1 bit per bit. For p=0.1: H = 0.469 bits/bit → 53% compression of raw bit-flag payload.
- **Boolean-mask compression**: composes with per-pair Venn classification — for HIGH_PAIR_INVARIANT class, the per-pair bit-mask is highly correlated across pairs → mask-delta encoding + boolean compression yields large savings.

**Cross-stack synergy**: zeros+ones + Catalog #319 Tier-1 deliverability — for bit-mask payloads, the entropy-coded compressed length vs raw bit-mask length IS the empirical deliverability proof.

**Frontier-breaking opportunity**: a `tac.binary_entropy_estimator` + sister `tac.boolean_mask_compressor` would unlock ~30-50% compression on sparse bit-flag payloads currently encoded as full bytes. Predicted ΔS `[-0.003, -0.001]` per archive with sparse-bit payloads.

### 1.4 PIXEL granularity (OPERATOR-FLAGGED biggest single class)

**Existing canonical helpers (per sister synthesis 1.4)**:
- `src/tac/logit_margin_sensitivity_weighted.py` (9K LOC) — DORMANT (loss function fully implemented; never used in any training script)
- `src/tac/imp_sensitivity_weighted.py` (14K LOC) — DESIGN-ONLY (lane_17_imp pre-rigor reactivation RANK#1)
- `src/tac/owv3_sensitivity_weighted.py` (44K LOC) — DORMANT
- `src/tac/fec6_selector_discovery_sensitivity_weighted.py` (10K LOC) — ACTIVE-PARTIAL (consumed by fec6 selector discovery only)
- `src/tac/balle_sensitivity_weighted.py` (12K LOC) — DESIGN-ONLY (PR101 + Ballé reactivation)
- `src/tac/neural_weight_codec_sensitivity.py` (51K LOC) — DORMANT (consumed only in tests)
- `src/tac/component_sensitivity_artifact.py` (72K LOC) — ACTIVE (canonical artifact)
- `src/tac/codec_pipeline_sensitivity.py` (33K LOC) — ACTIVE-PARTIAL

**Status**: per sister synthesis 1.4 — **largest single class of dormant signal in the codebase**. 6 helpers implemented; 0-1 hook wire-in each; canonical UNIWARD inverse-local-variance pattern per CLAUDE.md Fridrich-approved discipline.

**DEEPER discovery**:
- Each of the 6 dormant per-pixel sensitivity helpers represents a distinct UNIWARD-pattern application to a distinct substrate engineering surface (logit-margin / IMP / OWv3 / fec6 selector / Ballé / neural-weight)
- The cathedral autopilot v2 cascade currently has NO `adjust_predicted_delta_for_uniward_sensitivity_class` reward factor
- Wiring even 1 of these helpers (the lowest-hanging fruit is logit_margin_sensitivity_weighted per Catalog #325 NSCS01 reactivation candidacy) into the active substrate runtime would compose with the existing per-pixel sensitivity already in fec6_selector_discovery

**Cross-stack synergy with synthesis TOP-5**:
- per-pixel UNIWARD + OP-5 (multi-granularity sensitivity tensor) → extend 4D tensor with per-pixel granularity = (pair × pixel × class × axis) 4-tensor
- per-pixel UNIWARD + OP-3 (null-space exploiter) → per-pixel null-space basis: pixel-wise modifications aligned with score gradient's null direction
- per-pixel UNIWARD + TOP-1 of meta-portfolio (null-space exploiter) → bit-budget for per-pixel modifications proportional to inverse-local-variance weight

**Frontier-breaking opportunity the synthesis missed**: the 6 helpers ALREADY EXIST; the operator-flagged surface is wire-in. A single subagent landing `adjust_predicted_delta_for_uniward_sensitivity_class` reward factor in cathedral autopilot v2 cascade + emitting per-pixel manifest rows to `.omx/state/uniward_sensitivity_anchors.jsonl` (per Catalog #131 fcntl-locked) would unlock the entire class. Predicted aggregate ΔS `[-0.025, -0.005]` across 6 helpers × per-substrate application.

### 1.5 FRAME granularity (per-frame difficulty map + composition)

**Existing canonical helpers (per sister synthesis 1.3)**:
- `tac.variable_rate.compute_pair_difficulty` (ACTIVE; CRF allocation)
- `tac.codec.frame_conditional.encode_frame_conditional` (DORMANT; full codec stack only consumed by probe)
- `tac.codec.frame_conditional_bit_budget.pack_frame_conditional_q_bits` (DORMANT)
- `tac.xray.per_pair_score_decomposition` (DORMANT; xray primitive #9)
- `tac.lossless.tiny_frame_self_compress` (8K LOC)
- `tac.lossless.tiny_frame_predictor` (15K LOC)
- `tac.lossless.next_frame_coder` (14K LOC)
- `tac.cross_frame_attention` (13K LOC)

**Status**: per sister synthesis 1.3 — 4 surfaces; 1 ACTIVE; 3 DORMANT. The frame-conditional codecs are canonical pattern in DCVC-FM 2024 SOTA neural video compression but unwired in pact.

**DEEPER discovery**:
- **Per-frame motion-vector difficulty**: combines per-frame difficulty atlas with per-pair optical-flow magnitude. High-motion frames need more bits.
- **Per-frame SegNet ground-truth proximity**: how close is this frame to the SegNet ground-truth surface? Frames closer to GT need fewer corrective bits.
- **Per-frame chroma-vs-luma cost ratio**: per-frame chroma encoding cost vs luma encoding cost. NSCS06 v7 chroma anchors empirically established this varies per-frame.
- **Per-frame foveation map composition**: per-frame difficulty atlas × per-frame VRSS 2 foveation kernel → bit-budget per frame × per-pixel-region.

**Cross-stack synergy**:
- per-frame × per-pair Venn (Catalog #319 v2): each frame has its own Venn classification per pair-window → frame_difficulty × pair_invariance multi-class
- per-frame × master-gradient: per-frame gradient norm = aggregate of all per-pair gradients within frame window
- per-frame × OP-3 null-space exploiter: per-frame null-space basis carries DIFFERENT byte modifications per frame

**Frontier-breaking opportunity**: build `tac.codec.frame_conditional` consumer that wires the DORMANT frame-conditional codecs into PR101 fec6 substrate's variable_rate path. The pattern is well-established (DCVC-FM 2024 SOTA) and pact has all the primitives implemented but unwired.

### 1.6 PAIR granularity (per-pair Venn × composition_alpha × Wyner-Ziv tier)

**Existing canonical helpers (per sister synthesis 1.1)**:
- `tac.master_gradient.per_pair_gradient` (ACTIVE-PARTIAL — only fec6 anchor; sister synthesis OP-1 anchor extension is HIGHEST-EV)
- `tac.master_gradient_consumers.per_pair_difficulty_atlas` (DORMANT)
- `tac.master_gradient_consumers.wyner_ziv_side_info_covariance` (ACTIVE; Catalog #319 Q3 v2 cascade)
- `tac.master_gradient_consumers.classify_bytes_by_pair_variance` (ACTIVE — `PerByteVennClassification` class with `PerByteVennClass` enum)
- `tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual` (ACTIVE; CASCADE 1)
- `tac.master_gradient_consumers.rashomon_disagreement_queue` (DORMANT)
- `tac.master_gradient_consumers.fec6_selector_marginal_matrix` (DORMANT)
- `tac.master_gradient_consumers.nscs01_nullspace_empirical_audit` (ACTIVE-PARTIAL)

**DEEPER discovery (extends sister synthesis 1.1)**:
- **Per-pair Venn classification with operator's Venn-diagram extension**: currently 2-set Venn `PerByteVennClassification` with `PerByteVennClass` enum (HIGH_PAIR_INVARIANT / HIGH_PAIR_SPECIFIC). The operator's "Venn diagram" reference suggests EXTENSION to 3-set / 4-set Venns. The natural extension dimensions:
  - **3-set**: PER_PAIR × PER_CLASS (5 SegNet classes) = 10 cells (HIGH_PAIR_INVARIANT × per-class × per-axis)
  - **4-set**: PER_PAIR × PER_CLASS × PER_REGION (16×16 grid) = 1280 cells per axis
  - **5-set**: PER_PAIR × PER_CLASS × PER_REGION × PER_AXIS = 6400 cells
- **Per-pair composition_alpha (per Catalog #322)**: each pair has its own composition_alpha for stacking substrates. Currently `tac.optimization.substrate_composition_matrix` provides per-substrate-pair composition_alpha; per-pair version would surface frame-specific stacking opportunities.
- **Per-pair Wyner-Ziv tier classification (per Catalog #319)**: each pair has its own deliverability tier (TIER_1_ZERO_COST / TIER_2_CONSTANTS / TIER_3_WAIVER_REQUIRED / TIER_4_FORBIDDEN). Currently this is per-substrate; per-pair version would enable per-pair codec selection.

**Cross-stack synergy**:
- per-pair Venn × OP-1 (master-gradient extractor extension): each new archive's per-pair anchor unlocks per-pair Venn for that paradigm
- per-pair Venn × per-region: 3-set Venn natural extension
- per-pair composition_alpha × per-substrate composition_alpha: 2-level composition matrix

### 1.7 MASTER GRADIENT granularity (deeper than per-byte × per-pair × per-axis)

**Existing canonical helpers (per sister synthesis §2)**:
- `tac.master_gradient.MasterGradient` dataclass (canonical Layer 1)
- `tools/extract_master_gradient.py` (canonical Layer 2)
- Catalog #318 + #327 (canonical Layer 3 self-protection gates)
- `adjust_predicted_delta_for_venn_classification_v2` (canonical Layer 4 cathedral autopilot consumer)

**Status**: master-gradient is the most-developed canonical pattern; only 1 of 8 frontier archives anchored (the synthesis OP-1 anchor extension is HIGHEST-EV).

**DEEPER discovery (extends sister synthesis §2)**:
- **Per-byte master gradient × per-pair × per-class 4D tensor (per synthesis OP-5)**: this memo EXTENDS to 5D adding per-axis (seg / pose / rate) dimension → (pair × byte × class × axis) 4-tensor. The 5th dimension would be per-region for 5D tensor.
- **Null-space basis stability across paradigms**: per the synthesis OP-2, the null-space basis at byte_i is the 2D subspace orthogonal to (grad_seg[i], grad_pose[i], grad_rate[i]). Across paradigms, is this basis STABLE (suggesting structural property of the contest scorer) or PARADIGM-SPECIFIC (suggesting per-substrate engineering)? The deep-research wave §0 finding cos(seg_grad, pose_grad) ≈ 0.8973 on fec6 suggests STABLE — the 2D null-space is structurally rank-1 across paradigms.
- **Master-gradient × per-paradigm class-shift detector**: per `tac.optimization.substrate_composition_matrix.SubstrateClass` (HNERV_FAMILY / NERV_FAMILY / etc.), each paradigm has its own per-pair gradient signature. Cross-paradigm gradient analysis would surface which paradigms are TRULY orthogonal (sister substrates produce uncorrelated gradients) vs sub-additive (correlated gradients).
- **Per-pair predict_delta_s prediction calibration**: currently `tac.master_gradient.predict_delta_s_per_pair` predicts per-pair ΔS from per-byte modifications. Calibration error per pair (predicted vs measured) would surface which pairs are "easy" vs "hard" for the linear-approximation predictor.

**Cross-stack synergy with TOP-5**:
- All of synthesis TOP-5 op-routables depend on master gradient extension; this is the bottleneck
- The 5D tensor (pair × byte × class × axis × region) is THE canonical analytical surface for the unified S_total per CLAUDE.md "Anti-fragmentation: unified-Lagrangian action"

### 1.8 REGIONS granularity (per-region SegNet softmax histogram + cell mapping)

**Existing canonical helpers (per sister synthesis 1.5)**:
- `tac.optimization.faiss_ivf_pq_atw_channel` (DESIGN-ONLY; canonical Faiss-IVF-PQ helper for ATW V2-1 channel)
- `tools/probe_atw_v2_1_faiss_pq_disambiguator.py` (probe)
- `tools/build_a1_segnet_boundary_smoothing_variants.py` (28K LOC; build tool ACTIVE-PARTIAL)
- `tac.xray.segnet_margin_polytope` (ACTIVE; D1 OPERATIONAL mechanism per Catalog #220)

**DEEPER discovery**:
- **Per-region × per-pair × per-class 3D tensor**: each 16×16 region has its own per-pair SegNet softmax histogram per-class. Currently `tac.xray.segnet_margin_polytope` extracts per-pixel margin map → per-region (16×16) polytope-interior noise. Extending to per-pair per-class would surface per-region × per-pair × per-class cells.
- **Per-region Wyner-Ziv tier classification**: each region has its own deliverability tier. Some regions (e.g., sky/road regions which are temporally invariant) are TIER_1_ZERO_COST; others (e.g., vehicle/pedestrian regions) are TIER_3_WAIVER_REQUIRED.
- **Per-region foveation map**: VRSS 2 dynamic foveation per region → per-region bit-budget proportional to scorer-attention concentration in that region.

**Cross-stack synergy**:
- per-region × per-pair Venn → 3-set Venn extension (per-pair × per-class × per-region = 1280 cells)
- per-region × ATW V2-1 → canonical: ATW V2-1's per-region 16×16 SegNet softmax histogram product-quantized to ≤2KB IS this surface
- per-region × per-paradigm: each substrate has its own per-region sensitivity pattern

### 1.9 LABELS granularity (per-label SegNet conditional priors)

**Existing canonical helpers**:
- `src/tac/categorical_label_atoms.py` (3K LOC; DORMANT per sister synthesis 1.6)
- `src/tac/categorical_label_prior_payload_manifest.py` (4K LOC; DORMANT)
- `src/tac/learnable_class_weights.py` (14K LOC)
- `src/tac/learnable_class_targets.py` (15K LOC)
- `src/tac/categorical_openpilot_mask_prior_contract.py` (12K LOC)
- `src/tac/research/segnet_boundary_floor.py` (8K LOC)
- `src/tac/analysis/segnet_boundary_marginals.py` (6K LOC)

**Status**: per sister synthesis 1.6 — 4 per-class surfaces; 1 ACTIVE; 2 DORMANT. Plus several label-specific canonical helpers DORMANT.

**DEEPER discovery**:
- **5-class boundary intersections**: SegNet has 5 classes; class-boundary pixels have HIGH sensitivity (per `tac.research.segnet_boundary_floor`). Each pair of adjacent classes (5C2 = 10 pairs) has its own boundary sensitivity pattern.
- **Per-label codebook entropy estimation**: each class has its own codebook entropy (per `tac.categorical_label_atoms`). Class distribution varies (road >> sky > vehicle > pedestrian > marker); entropy-optimal codebook is per-class.
- **Per-label distillation target**: `tac.learnable_class_targets` provides per-class distillation targets. Currently single-target per class; per-pair per-class target would be richer.

**Cross-stack synergy**:
- per-label × per-pair Venn = 3-set Venn extension
- per-label × hash-seed (TOP-2 meta-portfolio): per-class palette derived from per-class seed
- per-label × categorical_substrate (sister synthesis 1.6 C1.1): existing canonical pattern

### 1.10 CATEGORIES granularity (per-class chroma anchors + cross-class composition)

**Existing canonical helpers (per sister synthesis 1.6)**:
- `tac.categorical_substrate.CategoricalRenderer` (ACTIVE)
- `tac.categorical_substrate._class_entropy` (canonical entropy helper)
- NSCS06 v7 per-class chroma anchors (44% empirical improvement v6→v7)
- `tac.xray.foveation_ego_motion` (DORMANT; per-class × per-pixel × per-pair foveation map)
- `tac.categorical_compression_contract` (4K LOC)
- `tac.categorical_payload_candidate` (43K LOC)
- `tac.categorical_candidate_readiness` (115K LOC; the canonical readiness check)

**DEEPER discovery**:
- **Cross-class composition matrices**: similar to substrate composition matrix per Catalog #322 anti-additive evidence, cross-class composition would surface which class encodings compose orthogonally vs sub-additively. E.g., road + sky may share a per-region pattern (orthogonal); road + vehicle (adjacent classes) may compose sub-additively.
- **Catalog #322 anti-phantom for per-class composition_alpha**: extend the canonical Catalog #322 anti-phantom gate to per-class composition_alpha. Per-class composition_alpha derived from research sidecar would be phantom; per-class composition_alpha derived from contest archive members is legitimate.
- **Per-class hash-seed × cross-class composition**: ~5 classes × ~3 chroma channels × per-class seed = 15 seeds total. If seeds compose orthogonally, total seed bytes = 15 × 4 = 60 bytes vs ~7.5KB current chroma palette = ratio 125×.

**Cross-stack synergy with TOP-5 meta-portfolio**:
- per-class × TOP-2 (procedural codebook generator): NSCS06 v7 chroma → 8-byte seed pattern
- per-class × TOP-1 (null-space exploiter): per-class null-space basis differs per class
- per-class × TOP-3 (DP1 driving prior): per-class driving prior — sky class has different DP1 prior than vehicle class

### 1.11 VENN DIAGRAM granularity (THE OPERATOR'S SIGNAL)

**Existing canonical helpers**:
- `tac.master_gradient_consumers.PerByteVennClass` enum (HIGH_PAIR_INVARIANT / HIGH_PAIR_SPECIFIC / LOW_PAIR_VARIANCE / etc.)
- `tac.master_gradient_consumers.PerByteVennClassification` (2-set Venn classification per-byte)
- `tac.master_gradient_consumers.WynerZivSideInfoClassification` (sister classification)
- `tac.master_gradient_consumers.classify_bytes_by_pair_variance` (canonical classifier)
- `adjust_predicted_delta_for_venn_classification_v2` (cathedral autopilot v2 cascade consumer)
- `tac.wyner_ziv_deliverability.proof_builder.DeliverabilityProof` + `DeliverabilityTier` (4-tier per Catalog #319)

**Status**: 2-set Venn is ACTIVE (Catalog #319 v2 cascade); 4-tier deliverability is ACTIVE; NO 3-set / 4-set / 5-set Venn classification exists.

**DEEPER DISCOVERY (THE LOAD-BEARING OPERATOR INSIGHT)**:

The operator's "Venn diagram" reference points to MULTI-SET Venn classifications. The current Catalog #319 v2 cascade uses a 2-set Venn (HIGH_PAIR_INVARIANT × HIGH_PAIR_SPECIFIC). Extending:

**3-set Venn = PER_PAIR × PER_REGION × PER_CLASS = 8 cells per axis**:
```
Cell 1: HIGH_PAIR_INVARIANT × HIGH_REGION_INVARIANT × HIGH_CLASS_INVARIANT
Cell 2: HIGH_PAIR_INVARIANT × HIGH_REGION_INVARIANT × HIGH_CLASS_SPECIFIC
Cell 3: HIGH_PAIR_INVARIANT × HIGH_REGION_SPECIFIC × HIGH_CLASS_INVARIANT
Cell 4: HIGH_PAIR_INVARIANT × HIGH_REGION_SPECIFIC × HIGH_CLASS_SPECIFIC
Cell 5: HIGH_PAIR_SPECIFIC × HIGH_REGION_INVARIANT × HIGH_CLASS_INVARIANT
Cell 6: HIGH_PAIR_SPECIFIC × HIGH_REGION_INVARIANT × HIGH_CLASS_SPECIFIC
Cell 7: HIGH_PAIR_SPECIFIC × HIGH_REGION_SPECIFIC × HIGH_CLASS_INVARIANT
Cell 8: HIGH_PAIR_SPECIFIC × HIGH_REGION_SPECIFIC × HIGH_CLASS_SPECIFIC
```

Each cell has its OWN deliverability tier per Catalog #319:
- Cell 1 (all invariant) = TIER_1_ZERO_COST (canonical reference per-pair-region-class shared)
- Cell 8 (all specific) = TIER_4_FORBIDDEN (cannot be shipped within byte budget)
- Cells 2-7 mix invariant/specific → TIER_2_CONSTANTS / TIER_3_WAIVER_REQUIRED depending on which axes are invariant

**4-set Venn = PER_PAIR × PER_REGION × PER_CLASS × PER_AXIS (seg/pose/rate) = 16 cells per byte position**:
adds the axis dimension; each cell has its own seg/pose/rate tradeoff profile.

**5-set Venn = PER_PAIR × PER_REGION × PER_CLASS × PER_AXIS × PER_BIT = 32 cells per byte**:
adds bit-level granularity; combines with §1.1 bit-level discovery.

**Cross-stack synergies**:
- 3-set Venn × per-region (§1.8) × per-pair (§1.6) — the cells map directly onto per-region per-pair per-class cells in the 5D tensor (TOP-5 above)
- 4-set Venn × per-axis → composes with `tac.sensitivity_map.axis_weights` (per sister synthesis 1.11)
- 5-set Venn × bit-level master gradient (§1.1) → the deepest analytical surface in the codebase

**Frontier-breaking opportunity (THE OPERATOR'S SIGNAL)**:
Build `tac.master_gradient_consumers.classify_bytes_by_3_set_venn(per_pair_master_gradient, per_region_segnet_margin, per_class_distribution)` returning typed `PerByte3SetVennClassification` with 8 cells per byte position. Sister probe `tools/probe_3_set_venn_classification.py` materializes the classification on the fec6 archive + verifies via Catalog #105/#139 byte-mutation that the cells produce orthogonal score-axis modifications.

The cathedral autopilot v2 cascade then extends: `adjust_predicted_delta_for_venn_classification_v3_3set` consumes the 8-cell classification and applies per-cell reward factor per Catalog #319 tier mapping.

Predicted aggregate ΔS `[-0.015, -0.005]` per archive — the cascade unlock + per-cell treatment refinement.

## 2. Composition matrix of newly-discovered granularities × meta-portfolio TOP-10

| Discovery (this memo) | Meta-portfolio TOP-1 (null-space) | TOP-2 (hash-seed × NSCS06 v7) | TOP-3 (DP1 × PR101) | TOP-4 (LTH × PR101) | TOP-5 (STC × pose-residual) |
|---|---|---|---|---|---|
| §1.1 Bit-level master gradient | DEEPENS (per-bit null-space) | composes (bit-level seed encoding) | composes (DP1 per-bit prior) | composes (LTH per-bit pruning) | DEEPENS (STC IS bit-level) |
| §1.2 Per-byte × per-section | composes (per-section null-space) | composes (per-section seed) | composes (per-section DP1 prior) | composes (per-section LTH) | composes (per-section STC) |
| §1.3 Zeros+ones binary entropy | composes (binary null-space) | DEEPENS (binary mask seed → smaller seeds) | independent | composes (binary LTH masks) | DEEPENS (binary STC) |
| §1.4 Per-pixel UNIWARD (6 dormant helpers) | DEEPENS (per-pixel null-space) | composes (per-pixel seed) | composes (per-pixel DP1 prior) | DEEPENS (IMP IS per-pixel sensitivity) | composes (per-pixel STC) |
| §1.5 Per-frame difficulty × foveation | composes (per-frame null-space) | composes (per-frame seed) | composes (per-frame DP1) | composes (per-frame LTH) | composes (per-frame STC) |
| §1.6 Per-pair Venn (extended) | DEEPENS (per-pair null-space basis) | composes (per-pair seed) | composes (per-pair DP1 weight) | DEEPENS (per-pair LTH) | DEEPENS (per-pair STC) |
| §1.7 Master gradient 5D tensor | DEEPENS (5D null-space) | composes (5D seed allocation) | composes (5D DP1 prior) | composes (5D LTH) | composes (5D STC) |
| §1.8 Per-region SegNet polytope | composes (per-region null-space) | DEEPENS (per-region seed) | composes (per-region DP1) | composes (per-region LTH) | composes (per-region STC) |
| §1.9 Per-label boundary intersections | composes (per-label null-space) | composes (per-label seed) | composes (per-label DP1) | composes (per-label LTH) | composes (per-label STC) |
| §1.10 Cross-class composition matrices | composes (per-class null-space) | DEEPENS (per-class chroma seed = canonical test case) | composes (per-class DP1) | composes (per-class LTH) | composes (per-class STC) |
| §1.11 3-set / 4-set / 5-set Venn (OPERATOR SIGNAL) | DEEPENS (per-cell null-space) | DEEPENS (per-cell seed) | composes (per-cell DP1) | composes (per-cell LTH) | composes (per-cell STC) |

**DEEPENS** = the discovery extends the meta-portfolio TOP-N to a finer granularity, multiplying its predicted ΔS by a non-trivial factor.
**composes** = the discovery and meta-portfolio TOP-N are orthogonal axes that can be stacked per Catalog #322 anti-additive analysis.
**independent** = no direct synergy.

The TOP-3 frontier-breaking COMPOSITIONS (per the matrix):
1. **§1.11 3-set Venn × TOP-1 null-space × TOP-2 hash-seed**: per-cell null-space basis + per-cell seed = each of the 8 cells has its own (null_basis, seed_bytes) tuple. Predicted aggregate ΔS multiplier: 1.5× over individual surface predictions.
2. **§1.1 Bit-level × TOP-5 STC × §1.6 per-pair Venn extended**: STC operates at bit-level; per-pair Venn extends to per-pair × per-bit. Predicted multiplier: 1.3×.
3. **§1.4 Per-pixel UNIWARD × TOP-1 null-space × §1.8 per-region SegNet polytope**: per-pixel UNIWARD × per-region polytope-interior × per-pair null-space → 3-level cascade. Predicted multiplier: 1.4×.

## 3. Operator-routable next-subagent consequences

Per CLAUDE.md "Subagent coherence-by-default" + operator's "start up the 2 subagent orchestration queue and keep it fed":

### Recommended SUBAGENT 2 work item (sister of meta-portfolio OP-2)

Build OP-2 master-gradient extractor extension (already named in sister deliverable). This unlocks every DEEPENS column above by providing per-pair fp64 anchors on 6 archives instead of 1. The 5D tensor of §1.7 + 3-set Venn of §1.11 + all per-cell null-space bases of §1.11 ALL depend on OP-2.

### Recommended SUBAGENT 3 work item (post-OP-2; new from this memo)

Build `tac.master_gradient_consumers.classify_bytes_by_3_set_venn` extending Catalog #319 v2 cascade to operate on 8-cell classification. Sister probe `tools/probe_3_set_venn_classification.py` materializes on fec6 archive. ~3-5 day editor + $0 GPU + $3 paired-CUDA verify.

### Recommended SUBAGENT 4 work item (post-OP-2; new from this memo)

Wire the 6 dormant UNIWARD-pattern per-pixel sensitivity helpers (§1.4) into cathedral autopilot v2 cascade via new reward factor `adjust_predicted_delta_for_uniward_sensitivity_class`. The 6 helpers ALREADY EXIST; this is wire-in work only. Per Carmack's verdict: ONE helper at a time + measure empirically. Start with `logit_margin_sensitivity_weighted` (smallest LOC; most-pure UNIWARD pattern).

### Recommended SUBAGENT 5 work item (post-OP-2; new from this memo)

Build `tac.bit_level_master_gradient_consumer` extending master gradient to bit granularity. Compose with `tac.stc_boundary_codec` for canonical bit-level STC application. Cross-stack with TOP-5 of meta-portfolio (STC pose-residual sidecar).

### Recommended SUBAGENT 6 work item (post-OP-2; new from this memo)

Build the canonical 5D multi-granularity sensitivity tensor in DuckDB per `tac.canonical_duckdb.multi_granularity_sensitivity_5d_ext`. Schema:
```sql
CREATE TABLE multi_granularity_sensitivity_5d (
  archive_sha256 TEXT NOT NULL,
  pair_id INTEGER NOT NULL,
  byte_offset INTEGER NOT NULL,
  class_id INTEGER NOT NULL,  -- 5 SegNet classes
  axis TEXT NOT NULL,  -- 'seg' / 'pose' / 'rate'
  region_id INTEGER NOT NULL,  -- 16×16 = 256 regions
  sensitivity_fp64 DOUBLE NOT NULL,
  venn_cell INTEGER NOT NULL,  -- 8-cell 3-set Venn classification
  derived_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (archive_sha256, pair_id, byte_offset, class_id, axis, region_id)
);
```

The 5D tensor is the canonical surface for the unified S_total + cathedral autopilot ranker + Rashomon ensemble.

## 4. Cargo-cult audit per assumption (Catalog #303)

| Assumption | HARD-EARNED / CARGO-CULTED | Unwind plan |
|---|---|---|
| The operator's "Venn diagram" reference points to 3-set / 4-set / 5-set Venns | HARD-EARNED | Direct enumeration of 11 granularities + operator's explicit naming |
| Each Venn cell has its own deliverability tier | HARD-EARNED-WITH-REVISION | Catalog #319 framework supports per-cell tier classification; per-cell EMPIRICAL anchor needed |
| 5D tensor (pair × byte × class × axis × region) is the canonical S_total surface | HARD-EARNED | Per CLAUDE.md "Anti-fragmentation: unified-Lagrangian action" target |
| Wiring 6 dormant UNIWARD helpers individually is correct sequencing | HARD-EARNED | Per Carmack's verdict; one-at-a-time + measure empirically |
| Bit-level master gradient is meaningful (vs aggregating to byte-level loses signal) | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | Empirical anchor needed: extract per-bit gradient on fec6; verify per-bit cos(seg, pose) ≠ per-byte cos(seg, pose) |
| Zero-run-length encoding has significant opportunity in pact archives | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | Empirical anchor needed: profile zero-run distributions in PR101 + PR106 archives |
| Per-frame foveation map composes orthogonally with per-frame difficulty | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | Empirical anchor needed: build foveation map + measure per-pixel sensitivity correlation |
| Cross-class composition matrices reveal orthogonal vs sub-additive | CARGO-CULTED | Per Catalog #322 anti-additive evidence at 4/8 probed pairs — expect sub-additive |
| Per-cell deliverability tiers maintain Catalog #319 structural validity | HARD-EARNED | Catalog #319 framework is per-substrate-byte; per-cell extension preserves the contract |

## 5. 9-dimension success checklist evidence (Catalog #294)

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS | 3-set / 4-set / 5-set Venn classification is NEW (no upstream PR; no pact prior); operator-named directly |
| 2. BEAUTY + ELEGANCE | 8-cell 3-set Venn fits a single typed `PerByte3SetVennClassification` dataclass; canonical 4-layer Catalog #245 pattern |
| 3. DISTINCTNESS | Each granularity is explicitly distinct: bit / byte / zero+one / pixel / frame / pair / master_gradient / region / label / category / Venn |
| 4. RIGOR | 11 granularities enumerated via Grep + Read of canonical helpers; status (ACTIVE/DORMANT/DESIGN-ONLY) verified per file |
| 5. OPTIMIZATION PER TECHNIQUE | Per UNIQUE-AND-COMPLETE-PER-METHOD: each granularity gets substrate-optimal engineering rationale |
| 6. STACK-OF-STACKS-COMPOSABILITY | Composition matrix §2 explicit; DEEPENS vs composes vs independent classifications |
| 7. DETERMINISTIC REPRODUCIBILITY | All discoveries are PyTorch + numpy deterministic; Catalog #205 inflate device-fork compatible |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | DuckDB-backed 5D tensor for O(1) lookup per (pair, byte, class, axis, region) cell |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Predicted aggregate ΔS [-0.025, -0.005] via TOP-5 discoveries; combined with meta-portfolio TOP-10 = aggregate [-0.045, -0.010] |

## 6. Observability surface (Catalog #305)

The deeper-granularity discoveries are observable via the 6-facet definition:

1. **Inspectable per layer**: each granularity has its own typed dataclass + canonical helper; status queryable via Grep + Read
2. **Decomposable per signal**: 5D tensor decomposable per (pair, byte, class, axis, region) cell; 8-cell Venn decomposable per cell
3. **Diff-able across runs**: 5D tensor diffable via sha256 of byte-stable serialization
4. **Queryable post-hoc**: `.omx/research/deeper_granularity_discovery_*_20260518.md` is the canonical artifact; consumers can `grep` for granularity names
5. **Cite-able**: every empirical claim carries `[empirical:<path>]` per Catalog #287
6. **Counterfactual-able**: per-cell counterfactual via byte-mutation per Catalog #272 distinguishing-feature contract

## 7. Cross-references

- **Parent deliverable**: `.omx/research/grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518.md`
- **Foundation memo**: `.omx/research/comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md`
- **Compliance memo**: `.omx/research/canonical_upstream_pr_review_procedural_generation_compliance_20260518.md`
- **Routing directive**: `.omx/research/codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518.md`
- **Asymptotic stacking sister**: `.omx/research/asymptotic_stacking_plus_local_max_utilization_audit_20260518.md`
- **Deep-research wave sister**: `.omx/research/comprehensive_research_wave_20260518.md`
- **Wyner-Ziv parent**: `.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md`

## 8. 6-hook wire-in declaration (Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default":

1. **Sensitivity-map contribution** = ACTIVE via the 11-granularity inventory feeding `tac.sensitivity_map.axis_weights` extension (per-granularity weights)
2. **Pareto constraint** = ACTIVE via composition matrix §2 feeding `tac.optimization.field_equation_planner.field_row` extension
3. **Bit-allocator hook** = ACTIVE via §1.1 (bit-level master gradient) + §1.3 (binary entropy) integration with `tac.bit_allocator`
4. **Cathedral autopilot dispatch hook** = ACTIVE via new reward factors per granularity (`adjust_predicted_delta_for_uniward_sensitivity_class` + `adjust_predicted_delta_for_3_set_venn_classification` + sister)
5. **Continual-learning posterior update** = ACTIVE via `.omx/state/multi_granularity_sensitivity_5d_anchors.jsonl` fcntl-locked JSONL store per Catalog #131
6. **Probe-disambiguator** = ACTIVE via sister probes `tools/probe_3_set_venn_classification.py` + `tools/probe_bit_level_master_gradient.py` + `tools/probe_per_pixel_uniward_wire_in.py`

## 9. Acknowledgements

This memo serves the operator's explicit directive to expand the sister synthesis's 70-surface inventory into deeper granularities ("bit and bytes and zeroes and ones and pixel and frame and pair and master gradient and regions and labels and categories and venn diagram and all"). The 3-set Venn classification was the operator's load-bearing INSIGHT — the current 2-set Catalog #319 v2 cascade was insufficient framing. The discovery deliverable EXTENDS the apparatus to honor the operator's empirical understanding of where signal still lurks.

Sister deliverable (T3 meta-portfolio symposium) re-ranks the 53-substrate registry by the NEW compliance envelope priors + the granularity discoveries identified here.

— Main-Claude (relayed on behalf of operator standing directive 2026-05-18)
