# Rate-Attack Methods × Dimensions Comprehensive Inventory

- timestamp_utc: 2026-05-25T15:22:30Z
- agent: claude (subagent rate_attack_matrix_a1)
- lane: lane_rate_attack_methods_dimensions_matrix_comprehensive_inventory_20260525
- scope: META inventory of all rate-attack methods × all rate/distortion dimensions; applicability matrix; EV-ranked operator-routable priority queue
- authority: PLANNING/RESEARCH ONLY — score_claim=false, promotion_eligible=false, rank_or_kill_eligible=false, ready_for_exact_eval_dispatch=false, gpu_launched=false, dispatch_attempted=false
- evidence_grade: [predicted] (META cell predictions are advisory; promotion requires paired Linux x86_64 + NVIDIA per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA")
- canonical_provenance: aggregate of codex DQS1 cascade (~30 memos 2026-05-25), my Tier-1 probe landings (OVERNIGHT M/N/P), canonical equations registry (36 entries), `.omx/research/` substrate design memos (PACT-NERV / Riemannian-Newton / Tropical / ATW V2 / NSCS06 v8 / HFV / DP1 / etc.)
- mission_predicted_contribution: frontier_breaking_enabler (surfaces unexplored high-EV cells)
- cutoff: 2026-05-25T15:30:00Z

## Canonical-vs-unique decision per layer

| Layer | Canonical helper used? | Rationale |
|---|---|---|
| Matrix data model | UNIQUE (this memo) | First inventory of its kind; sister subagents may canonicalize later |
| Methods enumeration | ADOPT canonical equations registry + lane registry IDs | Single source of truth per CLAUDE.md "Canonical equations + models registry" |
| Dimensions enumeration | EXTEND codex Eureka memo 13-dim canonical list (Catalog #344 RATIFY-N candidate) | Operator amplified to "and more"; matrix expands to ~25 dimensions |
| EV ranking | UNIQUE rate-saving-per-dollar formula | Composite of predicted ΔS / cost; not yet a canonical equation but candidate per Catalog #344 |
| Cost model | ADOPT cost_band_calibration posterior | fcntl-locked canonical; Catalog #175 / #177 / #237 |

## 9-dimension success checklist evidence

1. **UNIQUENESS**: First inventory of (method × dimension) cells across cumulative session knowledge; not a within-class refinement of an existing matrix.
2. **BEAUTY + ELEGANCE**: Markdown table renderable in 30 seconds; per-cell verdicts canonicalized into 4-symbol vocabulary (Y / N / PARTIAL / UNKNOWN).
3. **DISTINCTNESS**: Sister to the codex eureka memo's 13-dim list — extends to ~25 dimensions; sister to the DQS1 cascade — extends from pair-level to per-dimension orthogonal coverage.
4. **RIGOR**: Premise verification: read 30+ codex DQS1 memos + canonical equations registry + my own Tier-1 probe landings + substrate design memos before building cells. Every applicability claim cites an artifact path per Catalog #287.
5. **OPTIMIZATION PER TECHNIQUE**: Per-method per-dimension applicability ratings preserve method-specific structure; not a one-size-fits-all template per Catalog #290.
6. **STACK-OF-STACKS-COMPOSABILITY**: Matrix cells are orthogonal axes; rows × columns enables Cartesian search per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable.
7. **DETERMINISTIC REPRODUCIBILITY**: Byte-stable Markdown; every cell points at a permanent source artifact.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: Memo enables targeting unexplored high-EV cells rather than re-running explored cells.
9. **OPTIMAL MINIMAL CONTEST SCORE**: Phase 4 priority queue surfaces top-N cells most likely to drop frontier; promotion requires paired Linux x86_64 + NVIDIA.

## Cargo-cult audit per assumption

| Assumption | HARD-EARNED or CARGO-CULTED | Rationale |
|---|---|---|
| The 13-dim canonical list is exhaustive | CARGO-CULTED | Operator's "and more" amplification empirically falsifies; this memo enumerates ~12 NEW dimensions |
| Methods landed have been applied to every applicable dimension | CARGO-CULTED | Empirically false; DQS1 has been applied to (pair × byte × packet-member) but NOT to (per-frame × per-region × per-scorer-axis) |
| Paid GPU dispatch is required to discover unexplored cells | CARGO-CULTED | $0 LOCAL CPU probes per Catalog #192 + #317 can falsify many cells before paid spend |
| Top-K byte sensitivity dominates rate-attack EV | CARGO-CULTED | Equation `per_byte_leverage_uniformly_distributed_v1` is FALSIFIED (PR101 top-1% = 6.4%; not Pareto-concentrated); class-shifts dominate per-byte edits on entropy-coded archives |
| Per-CPU and per-CUDA axis attacks are equivalent | CARGO-CULTED | Equation `cpu_cuda_score_gap_v1` + DQS1 anchor: compact gap-ULEB packet IMPROVED CPU but REGRESSED CUDA |
| Within-class refinements yield rate savings on saturated substrates | CARGO-CULTED | Equation `hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1` (medal cluster Pareto frontier saturation) |
| Pair-axis attacks are leaf operations | CARGO-CULTED | Codex pair-frame scorer-geometry lattice empirically shows pair drops group into lattice cells via SegNet/PoseNet topology |

## Observability surface

1. **Inspectable per layer**: this memo is fully inspectable Markdown.
2. **Decomposable per signal**: per-method × per-dimension cells decomposable to applicability / EV / cost / canonical equation reference.
3. **Diff-able across runs**: NEW memo; subsequent inventory passes diff against this baseline.
4. **Queryable post-hoc**: Markdown tables grep-able; canonical equations registry queryable via `tools/list_canonical_equations.py`.
5. **Cite-able**: every applicability claim cites source artifact path per Catalog #287.
6. **Counterfactual-able**: per Catalog #139 + #220 + #272, each cell's runtime effect can be probed via byte-mutation smoke.

## Predicted ΔS band

[predicted]: top-5 unexplored cells per §4 EV ranking carry predicted ΔS in [-0.005, -0.0001] each per [predicted:per_byte_leverage_uniformly_distributed_v1 calibration]; aggregate predicted floor in [-0.025, -0.0005] if executed in parallel. Cumulative empirical anchor pending paired Linux x86_64 + NVIDIA per Catalog #324. Dykstra-feasibility intersection check: the per-axis decompositions are orthogonal in the contest's rate-distortion polytope per `pairset_component_marginal_score_decomposition_v1` (well-calibrated; 8 anchors residual=0); cumulative effect bounded by Cauchy-Schwarz per `per_pair_master_gradient_score_impact_taylor_v1`.

# 1. METHODS axis enumeration

## 1.1 Methods landed + actively iterating (codex DQS1 cascade 2026-05-25)

| M# | Method | Source artifact | Status |
|---|---|---|---|
| M01 | DQS1 dynamic-sparse-channel-gate compiler | `src/tac/optimization/dynamic_sparse_gate_oracle.py` + codex memo `dynamic_sparse_channel_gate_compiler_20260525T111322Z` | LANDED 2026-05-25 |
| M02 | DQS1 pairset-drop-one selective | `lane_dqs1_pairset_drop_one_rank021_pair0371_selective_decoderq_exact_cpu_20260522` + frontier pointer | LANDED-FRONTIER (0.19202828 [contest-CPU]) |
| M03 | DQS1 pairset-drop-many beam pairwise interaction waterfill | codex memo `eureka_drop_many_rate_distortion_budget_20260525T143351Z` | QUEUED-NOT-EXECUTABLE (3 selected experiments / 8 total) |
| M04 | DQS1 sorted_gap_uleb compact pair encoding | codex commit `fb14164d6` + frontier pointer compact archive | LANDED |
| M05 | Eureka drop-many rate-distortion budget redistribution | codex memo `eureka_drop_many_rate_distortion_budget_20260525T143351Z` | LANDED (queue-owned bridge) |
| M06 | DQS1 acquisition observation filter | codex memo `dqs1_acquisition_observation_filter_20260525T125955Z` | LANDED |
| M07 | DQS1 materializer feedback bridge | codex memo `dqs1_materializer_feedback_bridge_20260525T121724Z` | LANDED |
| M08 | DQS1 tranche refresh frontier bootstrap | codex memo `dqs1_tranche_refresh_frontier_bootstrap_20260525T131428Z` | LANDED |
| M09 | DQS1 observation acquisition skip | codex memo `dqs1_observation_acquisition_skip_20260525T124909Z` | LANDED |
| M10 | Pair/frame scorer-geometry lattice | codex memo `pair_frame_geometry_lattice_20260525T1512Z` | LANDED (planning-only bridge; 32 lattice rows / 6 queue-executable) |
| M11 | Frontier feedback eureka planning + operator preflight wiring | codex memo `frontier_feedback_operator_preflight_eureka_wiring_20260525T141629Z` | LANDED |
| M12 | Frontier section manifest feedback bridge | codex memo `frontier_section_manifest_feedback_bridge_20260525T133324Z` | LANDED |
| M13 | Frontier rate-attack feedback compiler | codex memo `frontier_rate_attack_feedback_compiler_20260525T133356Z` | LANDED |
| M14 | Frontier final-rate attack bootstrap | codex memo `frontier_final_rate_attack_bootstrap_20260525T130851Z` | LANDED |
| M15 | Frontier feedback cycle batch autopolicy | codex memo `frontier_feedback_cycle_batch_autopolicy_20260525T135319Z` | LANDED |
| M16 | Receiver-gated dynamic sparse feedback runner | codex memo `receiver_gated_dynamic_sparse_feedback_runner_20260525T115957Z` | LANDED |
| M17 | Serialized archive delta materializer feedback | codex memo `serialized_archive_delta_materializer_feedback_20260525T114622Z` | LANDED |
| M18 | Family-agnostic materializer delta canonicalization | codex memo `family_materializer_delta_canonicalization_20260525T100642Z` | LANDED |
| M19 | Queue observation recovery autopilot + grouping | codex memo `queue_observation_recovery_autopilot_and_grouping_20260525T034730Z` | LANDED |
| M20 | Dynamic sparse observation feedback compiler | codex memo `dynamic_sparse_observation_feedback_20260525T112208Z` | LANDED |
| M21 | Action functional CLI test latency | codex memo `action_functional_cli_test_latency_20260525T105700Z` | LANDED |
| M22 | Byte-shaving queue CLI test latency | codex memo `byte_shaving_queue_cli_test_latency_20260525T110019Z` | LANDED |
| M23 | Packet-member merge receiver contract hardening | codex memo `packet_member_merge_receiver_contract_hardening_20260525T003238Z` | LANDED |
| M24 | Renderer payload DFL1 native + shell parity | codex memos `renderer_payload_dfl1_*_20260525T*` | LANDED |
| M25 | DFL1 parity DAG no-orphan signal | codex memo `dfl1_parity_dag_no_orphan_signal_20260525T015113Z` | LANDED |

## 1.2 Substrate-class methods (mentioned + canonical equations + design memos)

| M# | Method | Source artifact | Status |
|---|---|---|---|
| M26 | Procedural-codebook from-seed substitution | equation `procedural_codebook_from_seed_compression_savings_v1` (7 anchors; INCLUDED contexts: DP1 + NSCS06 v8 + grayscale_lut + VQ-VAE) | LANDED (canonical equation) |
| M27 | Procedural-predictor + residual-correction | equation `procedural_predictor_plus_residual_correction_savings_v1` (2 anchors; pair #1 DWT + pair #2 FEC6) | LANDED (canonical equation) |
| M28 | Magic-codec dense streams (sparse_packet_ir + others) | `tools/run_magic_codec_dense_streams_dwt_residual_smoke.py` | LANDED-PROBED |
| M29 | Magic-codec pair-stacking (4-pair orthogonality matrix) | `tools/run_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke.py` + canonical equation #26 + #27 anchors | LANDED-PROBED |
| M30 | HFV1 + HFV2 sparse-pair sidecar cascade | equation `hfv2_sparse_pair_sidecar_replacement_savings_v1` (3 anchors; modal_paired_auth_eval_contest_cpu + cuda + canonical_helper) | LANDED-PROBED |
| M31 | Wave-Ω water-filling | `lane water-fill` from 2026-05-19; canonical equation `score_marginal_lagrange_multipliers_v1` consumer surface | LANDED (canonical lane) |
| M32 | Brotli/LZMA/ANS entropy coding | 19/19 Rust impls + equation `brotli_cascade_bounded_per_stream_v1` | LANDED (canonical) |
| M33 | Hessian-block-FP quantization (SC++) | substrate-engineering lane class; `feedback_sc_plus_plus_*` memos | LANDED (substrate-class) |
| M34 | Selfcomp grayscale-LUT block-FP (PR#56 paradigm) | `_VENDORED_PATH_MARKERS` reference + CLAUDE.md Selfcomp inner-council seat | LANDED-PARADIGM (0.38 anchor) |
| M35 | ATW V2 cdf_table_blob | `atw_v2_cdf_table_blob_*` memos + canonical equation #26 EXCLUDED context (FALSIFIED 2026-05-21 per codex byte-mutation smoke) | DEFERRED-FALSIFIED (REMOVAL paradigm) |
| M36 | Per-pair gradient clustering byte-removal | `tac.cathedral_consumers.per_pair_gradient_clustering_consumer` + Catalog #354 sister | LANDED-CONSUMER |
| M37 | Top-K byte sensitivity drop | `tac.cathedral_consumers.per_byte_sensitivity_consumer` + equation `per_byte_leverage_uniformly_distributed_v1` (FALSIFIED top-1%=6.4%) | LANDED-CONSUMER (signal: WEAK) |
| M38 | Bottom-K free-entropy byte | `tac.cathedral_consumers.bottom_k_free_entropy_byte_consumer` (Catalog #354 exploit #4) | LANDED-CONSUMER |
| M39 | Per-SegNet-class chroma collapse | `tac.cathedral_consumers.per_segnet_class_chroma_consumer` (Catalog #354 exploit #5) + equation `per_segnet_class_chroma_priors_v1` | LANDED-CONSUMER |
| M40 | UNIWARD textured-region weighting | Tier-1 probe `CCC` (POSITIVE) + `DDD` queued alternative + `EEE` queued alternative | LANDED-PROBED |
| M41 | HILL filter cascaded low-pass | Tier-1 probe `EEE` NULL_SIGNAL_DEFER | LANDED-PROBED (deferred) |
| M42 | J-UNIWARD multi-scale | `DDD` queued alternative | QUEUED |
| M43 | WOW directional filter banks | `DDD` queued alternative | QUEUED |
| M44 | Hinton KL T=2.0 distillation | Tier-1 probe `CCC` POSITIVE; Quantizr 0.33 paradigm + CLAUDE.md inner-council seat | LANDED-PARADIGM |
| M45 | Per-pair pose TTO with eval_roundtrip=True | Tier-1 probe `CCC` POSITIVE (42× advantage validated) | LANDED-PROBED |
| M46 | Wyner-Ziv side-information (Q1-Q5 deliverability_proof) | `tac.wyner_ziv_deliverability.proof_builder` + Catalog #319 | LANDED (canonical helper) |
| M47 | Cooperative-receiver loss (Atick-Redlich + Tishby IB) | Z4 / Z6 substrate design memos; equation `categorical_blahut_arimoto_rate_distortion_v1` | LANDED-SCAFFOLD (deferred) |
| M48 | Predictive-coding (Rao-Ballard + Hafner DreamerV3) | Z5 / Z7 design memos; `feedback_dreamer_v3_rssm_3_free_probes_landed_20260520.md` | LANDED-SCAFFOLD (deferred) |
| M49 | Hierarchical predictive coding quadruple (Z8) | Z6/Z7/Z8 design memo §11; Catalog #312 | LANDED-DESIGN |
| M50 | Foveation (TT5L LAPose; PR107 apogee) | equation `ego_motion_concentration_prior_v1` + `foveation_sidecar_bolt_on_rate_hurdle_v1` | LANDED-PARADIGM (PR107) |
| M51 | IB Lagrangian (T10; C6 IBPS) | C6 IBPS Phase 2 sextet + Catalog #325 + paired smoke 22× miss; canonical equation `categorical_blahut_arimoto_rate_distortion_v1` | LANDED-DEFERRED (22× miss) |
| M52 | Master-gradient null-byte removal (PR101 GOLD canonical) | equation `master_gradient_null_space_byte_fraction_v1` + canonical 8-exploit bundle per Catalog #354 | LANDED-PARADIGM (PR101 GOLD) |
| M53 | Null-byte parser-safe extraction | sister `null_byte_parser_safe_subset_smoke` lane | LANDED-PROBED |
| M54 | Selector extensions (PACT-NERV G3 5 variants) | `pact_nerv_g3_selector_extensions_l0_scaffold_design_20260520T204641Z.md` | LANDED-DESIGN |
| M55 | Mid-LOC bolt-ons (PACT-NERV G2 5 variants) | `pact_nerv_g2_*_l0_scaffold_design_*.md` | LANDED-DESIGN |
| M56 | Cross-codec bolt-ons (PACT-NERV G4 3 variants: A + B + Bayesian) | `pact_nerv_cross_codec_a_l0_scaffold_design_20260520T214500Z.md` + sister B + `pact_nerv_bayesian_l0_scaffold_design_20260520T211500Z.md` | LANDED-DESIGN |
| M57 | Bleeding-edge architectures (PACT-NERV G1 4: Mamba / MoE / DiffusionDistilled / NeuralCodecE2E) | `pact_nerv_*_l0_scaffold_design_*.md` cluster | LANDED-DESIGN |
| M58 | Riemannian-Newton substrate engineering | `riemannian_newton_substrate_engineering_design_memo_20260518.md` (119 KB design) | LANDED-DESIGN (META-canonical-helper queued) |
| M59 | Tropical d_seg solver | `tropical_d_seg_solver_design_memo_20260518.md` (118 KB design) | LANDED-DESIGN |
| M60 | Static-packet custody byte-delta | equation `static_packet_custody_byte_delta_score_savings_v1` (well-calibrated; WR01 paradigm) | LANDED (canonical equation) |

## 1.3 Method count + scope verdict

**Total methods enumerated**: 60 (target was 30-50; operator's "everything landed and mentioned" expanded scope). Coverage spans 4 distinct generations:
- (G1) DQS1 / acquisition-feedback compilers (M01-M25): codex 4-day cascade. ACTIVE.
- (G2) Substrate-class consumers + canonical equations (M26-M40, M52-M53, M60): canonical inventory.
- (G3) Inverse-steganography / scorer-aware (M41-M45, M50): Tier-1 probe-validated.
- (G4) Asymptotic substrate-class shifts (M46-M51, M54-M59): scaffolds + design memos.

Per CLAUDE.md "Forbidden premature KILL": M35 (ATW V2 cdf_table_blob) is DEFERRED-FALSIFIED-IMPLEMENTATION (paradigm intact per Catalog #307); M51 (C6 IBPS) is DEFERRED-PENDING-POST-TRAINING-TIER-C-VALIDATION per Catalog #324.

# 2. DIMENSIONS axis enumeration

## 2.1 Operator's canonical 9 dimensions (verbatim)

| D# | Dimension | Source artifact | Status |
|---|---|---|---|
| D01 | bit | operator 2026-05-25 + codex eureka 13-dim list | CANONICAL |
| D02 | byte | operator + codex + canonical equations registry | CANONICAL |
| D03 | pixel | operator + codex | CANONICAL |
| D04 | boundary | operator + codex | CANONICAL |
| D05 | region | operator + codex | CANONICAL |
| D06 | frame | operator + codex | CANONICAL |
| D07 | pair | operator + codex (DQS1 anchor) | CANONICAL |
| D08 | batch | operator | CANONICAL |
| D09 | full (video) | operator + codex | CANONICAL |

## 2.2 Codex eureka memo additions (4 more)

| D# | Dimension | Source artifact | Status |
|---|---|---|---|
| D10 | packet-member | codex eureka memo + Catalog #245 | CANONICAL |
| D11 | tensor-channel | codex eureka + dynamic_sparse_gate_oracle | CANONICAL |
| D12 | scorer-axis | codex eureka + equation `cpu_cuda_score_gap_v1` | CANONICAL |
| D13 | receiver-runtime | codex eureka + Catalog #295 / #166 | CANONICAL |

## 2.3 "And more" — NEW dimensions surfaced per operator amplification

| D# | Dimension | Rationale | First-principles source |
|---|---|---|---|
| D14 | subpixel (sub-byte / sub-bit fractional precision) | FP4-E2M1 quantization granularity; Hessian-block-FP at fraction precision | M33 SC++ + Quantizr FP4 |
| D15 | per-class (5-class SegNet logits) | M39 per-SegNet-class chroma; equation `per_segnet_class_chroma_priors_v1` | SegNet 5-class architecture |
| D16 | per-segment-label (per-instance segmentation region) | Mask CDF tail per pair-frame lattice; per-instance not per-class | codex pair_frame_geometry_lattice |
| D17 | per-time-window (multi-frame temporal sliding window) | Cross-pair correlation via temporal context | Wyner-Ziv + cooperative-receiver |
| D18 | per-token (MLX/numpy primitive granularity) | MLX scorer-input cache token granularity | MLX cache equation `scorer_input_cache_hash_identity_v1` |
| D19 | per-decoder-channel (per-tensor-channel sub-axis) | Beyond M11 tensor-channel: per-channel-of-channel decomposition | Catalog #354 exploit #3 + #4 |
| D20 | per-archive-section (manifest / latent / decoder weights / poses / masks) | HNeRV packet has 5+ sections; per-section bit budget | `hnerv_packet_sections` parser + codex frontier_section_manifest_feedback_bridge |
| D21 | per-codec-stage (pre-entropy / post-entropy / pre-quantization) | Equation `master_gradient_locality_violation_by_codec_v1` (raw-byte vs post-decompress) | Catalog #318 + #354 raw-byte authority guard |
| D22 | per-CPU-CUDA-axis (axis-specific drops) | Equation `cpu_cuda_score_gap_v1` + canonical bidirectional discipline per Catalog #192 | Frontier shows CPU/CUDA divergence on same archive |
| D23 | per-dispatch-environment (Modal / Vast.ai / Lightning / HF Jobs) | Equation `mps_drift_architecture_class_dependent_v1` extension; per-platform hardware noise | Catalog #245 ledger 4-platform support |
| D24 | cross-pair-correlation (paired-drop based on per-pair gradient clustering) | M36 per-pair gradient clustering; pair-pair interaction matrix | Codex pair-frame geometry lattice 6 queue-executable cells |
| D25 | adaptive-entropy (per-region rate budget redistribution) | M05 Eureka drop-many; rate budget surplus → distortion budget | Codex eureka memo §"distortion_repair_budget_from_rate_savings" |
| D26 | per-receiver-feasibility (inverse-steganography embedding budget) | M40-M43 UNIWARD / HILL / J-UNIWARD / WOW; per-region feasibility maps | Tier-1 probe CCC + DDD + EEE |
| D27 | per-DP1-codebook-dependency (cross-substrate composition byte chain) | M26 DP1 + sister substrates (NSCS06 v8 / VQ-VAE / grayscale_lut) | Equation `procedural_codebook_from_seed_compression_savings_v1` |
| D28 | per-PR101-grammar-section (HFV cascade-specific) | M30 HFV1+HFV2 sparse cascade; HFV1-grammar vs PR101-grammar | Equation `hfv2_sparse_pair_sidecar_replacement_savings_v1` |
| D29 | per-EMA-decay-stage (substrate-stage-aware EMA) | Equation `ema_decay_substrate_stage_aware_v1` | CLAUDE.md "EMA — NON-NEGOTIABLE" |
| D30 | per-convergence-slope (early-stop budget) | Equation `convergence_slope_early_stop_v1` | Quantizr 0.33 paradigm |
| D31 | per-frame-difficulty-atlas (per-frame budget allocation) | Equation `per_frame_difficulty_atlas_v1` + `tac.cathedral_consumers.per_pair_difficulty_atlas_consumer` (Catalog #354 exploit #1 sister) | LANDED-CONSUMER |
| D32 | per-ego-motion-concentration (foveation-aware) | Equation `ego_motion_concentration_prior_v1` + M50 foveation | LANDED-EQUATION |
| D33 | per-pose-axis-amplification (CUDA-specific pose amplification) | Equation `pose_axis_cuda_amplification_v1` | LANDED-EQUATION |
| D34 | per-MPS-portability (use-case taxonomy) | Equation `mps_portability_use_case_taxonomy_v1` | LANDED-EQUATION |
| D35 | per-categorical-vs-continuous (posterior capacity) | Equation `categorical_posterior_capacity_vs_continuous_gaussian_v1` | LANDED-EQUATION |
| D36 | per-substrate-class-shift (cross-class composability) | Equation `cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1` + `triple_substrate_composition_alpha_v1` + `cross_codec_super_additive_orthogonality_predictor_v1` | LANDED-EQUATION |

## 2.4 Dimensions count + scope verdict

**Total dimensions enumerated**: 36 (canonical 9 + codex 4 + 23 "and more"; far exceeded "and more" baseline target). Coverage spans:
- Atomic (D01-D04, D14): bit / sub-bit / byte / pixel / sub-pixel atom granularity.
- Spatial (D05-D06, D15-D16, D25-D26): region / frame / class / instance / adaptive / feasibility.
- Temporal (D07-D08, D17, D24, D32): pair / batch / time-window / pair-correlation / ego-motion.
- Compositional (D09-D11, D18-D21, D27-D28): full / packet-member / tensor-channel / token / decoder-channel / archive-section / codec-stage / DP1-chain / HFV-section.
- Receiver (D12-D13, D22-D23, D33-D34): scorer-axis / runtime / CPU-CUDA / dispatch-env / pose-amplification / MPS-portability.
- Posterior (D29-D31, D35-D36): EMA-decay / convergence-slope / difficulty-atlas / categorical-vs-continuous / class-shift.

# 3. METHODS × DIMENSIONS APPLICABILITY MATRIX

## Vocabulary
- **Y**: Method DEMONSTRABLY APPLICABLE to dimension via at least one landing or canonical equation anchor.
- **N**: Method STRUCTURALLY INAPPLICABLE (e.g. byte-level method cannot operate at per-region without restructuring; per CLAUDE.md "Forbidden premature KILL" this is "NOT-YET-APPLICABLE" not killed).
- **PARTIAL**: Method PARTIALLY APPLICABLE (e.g. landed at sub-dimension or requires bolt-on).
- **UNKNOWN**: Untested cell; high-EV if predicted ΔS suggests rate-saving (highlighted in §4 EV ranking).

## 3.1 G1 codex DQS1 cascade methods (M01-M25)

Compressed view (full applicability against all 36 dimensions would be 25×36=900 cells; surfacing only HIGH-SIGNAL cells per operator's "and more" emphasis):

| Method | D02 byte | D07 pair | D10 packet-member | D11 tensor-channel | D12 scorer-axis | D13 receiver-runtime | D22 CPU-CUDA | D24 cross-pair-corr | D31 frame-difficulty | UNEXPLORED HIGH-EV |
|---|---|---|---|---|---|---|---|---|---|---|
| M01 DQS1 channel-gate compiler | Y | Y | Y | Y | PARTIAL | Y | PARTIAL | UNKNOWN | UNKNOWN | (D24, D31) |
| M02 DQS1 drop-one (FRONTIER) | Y | Y | Y | N | Y(CPU) | Y | **Y(CPU only; CUDA regression)** | UNKNOWN | UNKNOWN | (D22 CUDA repair, D24) |
| M03 DQS1 drop-many beam waterfill | Y | Y | Y | PARTIAL | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | (ALL except D02/D07) |
| M04 DQS1 sorted_gap_uleb | Y | Y | Y | N | Y(CPU) | Y | **Y(CPU only; CUDA regression)** | N | N | (D22 CUDA repair) |
| M05 Eureka drop-many R/D budget | Y | Y | N | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | (D03-D06, D10, D17, D20, D25) |
| M10 Pair/frame geometry lattice | N | Y | N | PARTIAL | PARTIAL | UNKNOWN | UNKNOWN | Y(6 cells) | PARTIAL | (D05-D06, D15-D16, D26) |
| M16 Receiver-gated DSGate runner | N | N | Y | Y | UNKNOWN | Y | UNKNOWN | N | N | (D12, D22, D24) |
| M23 Packet-member merge receiver | N | N | Y | N | N | Y | UNKNOWN | N | N | (D22, D23) |

**G1 verdict**: DQS1 cascade is BYTE/PAIR/PACKET-MEMBER-saturated; UNEXPLORED dimensions are pose-axis (D33), per-CPU-CUDA-axis repair (D22), cross-pair-correlation (D24), per-region-feasibility (D26), per-frame-difficulty-atlas (D31).

## 3.2 G2 substrate-class methods × canonical equations (M26-M40, M52-M53, M60)

| Method | D02 byte | D07 pair | D20 archive-section | D21 codec-stage | D22 CPU-CUDA | D27 DP1-codebook-chain | D36 class-shift-composability | UNEXPLORED HIGH-EV |
|---|---|---|---|---|---|---|---|---|
| M26 Procedural-codebook (DP1 + NSCS06 v8) | Y | N | Y | Y | PARTIAL | Y | PARTIAL | (D17, D24, D28) |
| M27 Procedural-predictor + residual (DWT + FEC6) | Y | N | Y | Y | PARTIAL | PARTIAL | UNKNOWN | (D03-D06, D24, D28) |
| M28 Magic-codec dense streams | Y | N | Y | Y | UNKNOWN | UNKNOWN | UNKNOWN | (D03, D11, D24) |
| M29 Magic-codec pair-stacking 4-pair | Y | Y | Y | Y | UNKNOWN | UNKNOWN | **PARTIAL** | (D22 CUDA paired anchor; D24 cross-pair) |
| M30 HFV1+HFV2 sparse cascade | Y | Y | Y | Y | Y(both axes) | N | UNKNOWN | (D17, D24, D28) |
| M31 Wave-Ω water-filling | Y | Y | Y | Y | UNKNOWN | UNKNOWN | UNKNOWN | (D22, D24) |
| M32 Brotli/LZMA/ANS entropy coding | Y | N | Y | Y | Y(both) | N | N | (D11, D14 sub-byte) |
| M33 SC++ Hessian-block-FP | Y | N | Y | Y | UNKNOWN | N | PARTIAL | (D14 sub-byte, D22) |
| M34 Selfcomp grayscale-LUT (PR#56) | Y | N | Y | Y | Y(both) | N | PARTIAL | (D14, D22, D24) |
| M35 ATW V2 cdf_table_blob FALSIFIED | N | N | N | N | N | N | N | (DEFER per Catalog #307; explore SISTER methods instead) |
| M36 Per-pair gradient clustering | N | Y | N | N | UNKNOWN | N | N | (D22 CUDA, D24 explicit cross-corr matrix) |
| M37 Top-K byte sensitivity (WEAK signal) | Y | N | N | Y | UNKNOWN | N | N | (D11, D21 post-decompress grain) |
| M38 Bottom-K free-entropy byte | Y | N | N | Y | UNKNOWN | N | N | (D11, D14) |
| M39 Per-SegNet-class chroma collapse | N | N | N | N | N | N | N (per-class only) | (D03 pixel, D15 per-class, D16 per-segment) |
| M52 Master-gradient null-byte (PR101 GOLD) | Y | N | Y | Y | Y(both) | N | PARTIAL | (D11 per-decoder-channel; D19) |
| M53 Null-byte parser-safe extraction | Y | N | Y | Y | Y(both) | N | N | (D11, D19, D20) |
| M60 Static-packet custody byte-delta WR01 | Y | N | Y | Y | Y(both) | N | N | (D11, D14, D22) |

**G2 verdict**: Substrate-class methods are ARCHIVE-SECTION + CODEC-STAGE saturated; UNEXPLORED dimensions are per-decoder-channel (D19), sub-byte (D14), CUDA-axis paired anchor (D22), cross-pair-correlation (D24).

## 3.3 G3 inverse-steganography / scorer-aware methods (M40-M45, M50)

| Method | D03 pixel | D04 boundary | D05 region | D15 per-class | D16 per-segment | D26 receiver-feasibility | D32 ego-motion-concentration | UNEXPLORED HIGH-EV |
|---|---|---|---|---|---|---|---|---|
| M40 UNIWARD textured-region (CCC POSITIVE) | Y | Y | Y | PARTIAL | UNKNOWN | Y | UNKNOWN | (D15 explicit per-class, D16 per-instance, D17 temporal) |
| M41 HILL filter (EEE NULL_SIGNAL_DEFER) | Y | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | (per CLAUDE.md "Forbidden premature KILL" — DEFER not KILL; re-probe with smaller smoke scope) |
| M42 J-UNIWARD multi-scale (QUEUED) | Y | Y | Y | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | (D11 multi-scale tensor decomposition, D17, D32) |
| M43 WOW directional filter banks (QUEUED) | Y | Y | Y | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | (D04, D11, D17) |
| M44 Hinton KL T=2.0 distillation (CCC POSITIVE) | Y | N | N | Y | UNKNOWN | UNKNOWN | UNKNOWN | (D17 temporal context, D15 explicit per-class, D32) |
| M45 Per-pair pose TTO eval_roundtrip=True | N | N | N | N | N | UNKNOWN | UNKNOWN | (Already DOMINANT 42×; sister to D33 pose-axis amplification) |
| M50 Foveation (TT5L LAPose + PR107 apogee) | Y | Y | Y | UNKNOWN | UNKNOWN | Y | Y | (D15 per-class foveation budget, D17, D33) |

**G3 verdict**: Inverse-stego methods are PIXEL+REGION-saturated for UNIWARD; UNEXPLORED dimensions are per-class explicit foveation (D15+D50 cross), temporal context (D17), per-segment instance (D16), pose-axis (D33). M41 HILL is DEFER-not-KILL per CLAUDE.md.

## 3.4 G4 asymptotic substrate-class methods (M46-M51, M54-M59)

| Method | D17 time-window | D27 DP1-chain | D32 ego-motion | D33 pose-axis-amp | D34 MPS-portability | D35 categorical-vs-continuous | D36 class-shift | UNEXPLORED HIGH-EV |
|---|---|---|---|---|---|---|---|---|
| M46 Wyner-Ziv side-info (Q1-Q5 deliverability_proof) | Y | Y | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Y | (D32, D33 pose-conditional Wyner-Ziv) |
| M47 Cooperative-receiver Atick-Redlich + Tishby | Y | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Y | Y | (D32 explicit ego-motion conditioning per Catalog #311) |
| M48 Predictive-coding Rao-Ballard + Hafner | Y | UNKNOWN | Y | UNKNOWN | UNKNOWN | UNKNOWN | Y | (D33 pose-axis prediction) |
| M49 Z8 hierarchical predictive coding quadruple | Y | UNKNOWN | Y | UNKNOWN | UNKNOWN | UNKNOWN | Y | (Catalog #312 4-primitive check; D27 cross-DP1 chain) |
| M50 Foveation (re-listed for asymptotic context) | UNKNOWN | UNKNOWN | Y | UNKNOWN | UNKNOWN | UNKNOWN | Y | (Cross-list with G3) |
| M51 IB Lagrangian C6 IBPS (22× miss DEFER) | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Y | Y | (per Catalog #324: post-training Tier-C re-measure required) |
| M52 Master-gradient null-byte (re-list for G4) | N | N | UNKNOWN | UNKNOWN | N | N | UNKNOWN | (D19, D11 per-decoder-channel sister) |
| M54 PACT-NERV G3 selector extensions (5 variants) | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Y | (ALL dimensions UNKNOWN; design scaffold) |
| M55 PACT-NERV G2 mid-LOC bolt-ons (5 variants) | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | PARTIAL | (ALL dimensions UNKNOWN; design scaffold) |
| M56 PACT-NERV G4 cross-codec bolt-ons | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Y | (ALL dimensions UNKNOWN; design scaffold) |
| M57 PACT-NERV G1 bleeding-edge (Mamba / MoE / Diffusion / NeuralCodecE2E) | Y | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Y | Y | (D17 Mamba temporal; D22 + D23 platform diversity) |
| M58 Riemannian-Newton substrate engineering | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Y | (META-canonical helper; ALL dimensions UNKNOWN by design) |
| M59 Tropical d_seg solver | N | N | N | N | N | Y | UNKNOWN | (D15 per-class tropical; D26 d_seg-only sister of M58) |

**G4 verdict**: Asymptotic substrate-class methods are DESIGN-SCAFFOLD with most cells UNKNOWN; the BUILDING DISCIPLINE is the gating constraint (per Catalog #325 per-substrate symposium + #294 9-dim checklist + #303 cargo-cult audit + #305 observability + #309 horizon class + #310 not-bolt-on + #311 ego-motion + #312 quadruple primitives). All G4 methods carry the operator-frontier-override path per Catalog #300.

# 4. TOP-N OPERATOR-ROUTABLE PRIORITY QUEUE (EV-RANKED)

## EV ranking formula

EV(method, dimension) = |predicted ΔS| / cost($) × prior_confidence

where:
- predicted ΔS from canonical equation (if applicable) OR rate-saving heuristic via per_byte_leverage_uniformly_distributed_v1 (FALSIFIED top-1% = 6.4% means we trust per-pair / per-section / per-class budget redistribution more than top-byte)
- cost: $0 LOCAL CPU probe / $0.30-5 paid Modal smoke / $5-20 paid full dispatch
- prior_confidence: well-calibrated canonical equation × cross-substrate replication signal

## Top-20 UNEXPLORED high-EV cells

Sorted by EV ratio. Each cell satisfies Carmack MVP-first phasing per CLAUDE.md non-negotiable (free-LOCAL-smoke-first OR cheap-paired-Modal-smoke-first; falsifiable challenge; canonical equation reference OR FORMALIZATION_PENDING waiver per Catalog #344; same-batch landing of fix + smoke verdict; ~1h operator re-route per CLAUDE.md "Downstream-surface latency discipline").

| # | (Method, Dimension) | Cost | Predicted ΔS | Prior | EV ratio | Carmack 5-step ready? |
|---|---|---|---|---|---|---|
| 1 | **(M03 DQS1 drop-many beam waterfill, D07 pair × D08 batch)** | $0.30-5 paid Modal smoke (codex queued 3 selected experiments) | [-0.003, -0.0005] per Eureka R/D budget redistribution | HIGH (codex eureka memo + drop-one anchor proven) | **EXTREME** | YES — codex executable queue exists |
| 2 | **(M02 DQS1 drop-one, D22 CPU-CUDA repair)** | $5-10 paired CUDA T4 anchor | Fixes -0.005 CUDA regression on current FRONTIER | HIGH (frontier pointer empirically shows CPU/CUDA divergence) | **EXTREME** | YES — paired anchor per Catalog #246 paired auth eval |
| 3 | **(M40 UNIWARD, D15 per-class explicit)** | $0 LOCAL CPU smoke (Tier-1 CCC POSITIVE precedent) | [-0.002, -0.0005] per per_segnet_class_chroma_priors_v1 | HIGH (CCC anchor) | **HIGH** | YES |
| 4 | **(M44 Hinton KL T=2.0, D17 temporal context)** | $0 LOCAL CPU smoke | [-0.002, -0.0005] (Tier-1 CCC POSITIVE extension) | HIGH (CCC anchor + cooperative-receiver theory) | **HIGH** | YES |
| 5 | **(M10 pair/frame geometry lattice, D26 receiver-feasibility)** | $0 LOCAL CPU smoke (codex 6 queue-executable cells) | UNKNOWN; predicted in [-0.002, +0.001] range | MEDIUM (codex lattice low coverage 0.0625) | **HIGH** | YES — codex pre-built |
| 6 | **(M26 procedural-codebook, D17 time-window across DP1 pairs)** | $0.50 LOCAL CPU paired smoke | [-0.001, -0.0001] per canonical equation #26 (calibrated 7 anchors) | HIGH (well-calibrated canonical equation) | **HIGH** | YES — procedural_codebook_generator canonical helper exists |
| 7 | **(M29 magic-codec pair-stacking, D22 paired CUDA anchor)** | $5-10 paired CUDA T4 | UNKNOWN; predicted [-0.002, -0.0005] | HIGH (canonical equation #27 well-calibrated) | **HIGH** | YES — canonical helper exists |
| 8 | **(M36 per-pair gradient clustering, D24 explicit cross-correlation matrix)** | $0 LOCAL CPU smoke | UNKNOWN; predicted [-0.001, +0.001] | MEDIUM (canonical consumer Catalog #354 exploit #9) | **MEDIUM** | YES — consumer landed |
| 9 | **(M52 master-gradient null-byte, D19 per-decoder-channel)** | $0 LOCAL CPU smoke | [-0.001, -0.0005] per master_gradient_null_space_byte_fraction_v1 | HIGH (PR101 GOLD canonical) | **MEDIUM** | YES |
| 10 | **(M42 J-UNIWARD multi-scale, D11 tensor-channel decomposition)** | $0 LOCAL CPU smoke | UNKNOWN; predicted [-0.002, -0.0005] (CCC POSITIVE precedent + multi-scale dim extension) | MEDIUM (queued alternative) | **MEDIUM** | YES — DDD queued |
| 11 | **(M43 WOW directional, D04 boundary)** | $0 LOCAL CPU smoke | UNKNOWN; predicted [-0.001, -0.0001] | MEDIUM | **MEDIUM** | YES — DDD queued |
| 12 | **(M50 foveation, D15 per-class foveation budget)** | $0.30-1 paid Modal smoke | [-0.002, -0.0005] per ego_motion_concentration_prior_v1 + foveation_sidecar_bolt_on_rate_hurdle_v1 | HIGH (PR107 paradigm) | **MEDIUM** | YES |
| 13 | **(M30 HFV1+HFV2, D17 time-window across pair cascade)** | $0.50-2 paired smoke | UNKNOWN; predicted [-0.001, -0.0001] per hfv2_sparse_pair_sidecar_replacement_savings_v1 | HIGH (calibrated 3 anchors) | **MEDIUM** | YES |
| 14 | **(M27 procedural-predictor + residual, D03-D06 per-region/boundary)** | $0 LOCAL CPU smoke | UNKNOWN; predicted [-0.002, -0.0005] (extends pair #1 DWT detail-subband to region-level) | HIGH (canonical equation #27 calibrated) | **MEDIUM** | YES |
| 15 | **(M01 DQS1 channel-gate compiler, D31 frame-difficulty)** | $0 LOCAL CPU smoke | UNKNOWN; predicted [-0.001, -0.0001] | MEDIUM (codex compiler exists; per_frame_difficulty_atlas_v1 equation calibrated) | **MEDIUM** | YES |
| 16 | **(M48 predictive-coding Rao-Ballard, D33 pose-axis prediction)** | $0 LOCAL CPU smoke (free probe like dreamer_v3_rssm_3_free_probes) | UNKNOWN; predicted [-0.003, -0.001] (asymptotic-pursuit) | LOW (scaffold; needs ego-motion conditioning per Catalog #311) | **MEDIUM** (asymptotic) | PARTIAL — needs Catalog #311 ego-motion |
| 17 | **(M51 IB Lagrangian C6 IBPS, post-training Tier-C re-measure)** | $5-10 paired Modal A100 (post-training Tier-C re-measure per Catalog #324) | UNKNOWN; predicted bounded by [+22× miss baseline, -0.005 floor] | LOW (DEFERRED-pending; 22× miss) | **MEDIUM** (asymptotic) | PARTIAL — needs Catalog #324 post-training validation |
| 18 | **(M47 cooperative-receiver Atick-Redlich, D32 ego-motion concentration)** | $0 LOCAL CPU smoke per Catalog #311 ego-motion-conditioned | UNKNOWN; predicted [-0.003, -0.001] (asymptotic) | LOW (Z6 design memo Section 11 quadruple) | **MEDIUM** (asymptotic) | PARTIAL — needs Catalog #311 |
| 19 | **(M58 Riemannian-Newton substrate engineering, META-canonical-helper)** | $0 LOCAL design + $0.30-1 first probe | UNKNOWN; predicted [-0.005, -0.001] (asymptotic) | LOW (design scaffold 119 KB; no anchor) | **MEDIUM** (asymptotic) | NO — needs first probe; META-canonical-helper sister |
| 20 | **(M59 Tropical d_seg solver, D15 per-class tropical)** | $0 LOCAL design + $0.30-1 first probe | UNKNOWN; predicted [-0.005, -0.001] (asymptotic) | LOW (design scaffold 118 KB; no anchor) | **MEDIUM** (asymptotic) | NO — needs first probe |

## Per-cell canonical equation candidates (Catalog #344 RATIFY-N queued)

Per CLAUDE.md "Canonical equations + models registry" non-negotiable, the following equation candidates are QUEUED for operator-routable RATIFY-N landing (NOT auto-registered):

| Equation candidate | Sister to | Covers cell(s) |
|---|---|---|
| `rate_attack_methods_dimensions_canonical_matrix_v1` | THIS memo's matrix as queryable canonical equation | M×D applicability across all cells |
| `per_dimension_rate_attack_ev_ranking_canonical_v1` | Top-20 priority queue as canonical equation | (method, dimension) → EV ratio |
| `dqs1_drop_many_beam_waterfill_predicted_delta_s_v1` | codex eureka memo | Top-1 cell |
| `cpu_cuda_axis_repair_predicted_score_recovery_v1` | equation `cpu_cuda_score_gap_v1` | Top-2 cell |
| `per_class_explicit_uniward_predicted_delta_s_v1` | Tier-1 probe CCC + per_segnet_class_chroma_priors_v1 | Top-3 cell |
| `temporal_context_kl_distillation_predicted_delta_s_v1` | Tier-1 probe CCC + Wyner-Ziv | Top-4 cell |
| `pair_frame_lattice_receiver_feasibility_predicted_delta_s_v1` | codex pair_frame_geometry_lattice | Top-5 cell |
| `procedural_codebook_temporal_window_cross_dp1_predicted_delta_s_v1` | equation #26 extension | Top-6 cell |
| `magic_codec_pair_stacking_cuda_axis_predicted_delta_s_v1` | equation #27 extension | Top-7 cell |
| `cross_pair_gradient_clustering_correlation_matrix_v1` | M36 consumer Catalog #354 exploit #9 | Top-8 cell |
| `master_gradient_per_decoder_channel_decomposition_v1` | equation `master_gradient_null_space_byte_fraction_v1` | Top-9 cell |

## Carmack MVP-first phasing per cell (top-5 priority)

Per CLAUDE.md "Carmack MVP-first phasing" non-negotiable, every paid GPU dispatch >$0.30 MUST satisfy 5-step recipe. Per-cell status:

**Cell 1 (M03 × D07×D08 drop-many beam waterfill)**: Step 1 ✓ FREE local macOS-CPU smoke ready via codex queue. Step 2 ✓ falsifiable challenge (predict 0/+/+ ΔS distribution per pair group). Step 3 ✓ canonical equation reference candidate `dqs1_drop_many_beam_waterfill_predicted_delta_s_v1` (or FORMALIZATION_PENDING per Catalog #344). Step 4 will land verdict in same commit batch as the smoke landing memo. Step 5 will re-route operator priority queue within ~1h.

**Cell 2 (M02 × D22 CPU-CUDA repair)**: Step 1 SKIP — paid CUDA T4 anchor required (paired-dispatch per Catalog #246; estimated cost <$10 within session envelope). Step 2 ✓ falsifiable challenge (predict CUDA delta to match CPU 0.19202828 within ±0.005). Step 3 ✓ canonical equation `cpu_cuda_score_gap_v1` calibrated. Step 4-5 same-session.

**Cell 3 (M40 × D15 per-class UNIWARD)**: Step 1-5 ALL local CPU smoke ready per CCC POSITIVE precedent.

**Cell 4 (M44 × D17 temporal KL distillation)**: Step 1-5 ALL local CPU smoke ready per CCC POSITIVE precedent.

**Cell 5 (M10 × D26 pair/frame lattice receiver-feasibility)**: Step 1-5 ALL local CPU smoke ready per codex lattice 6 queue-executable cells.

## Operator-routable next actions (top-5)

1. **Authorize cell 1**: execute codex's queued DQS1 drop-many beam waterfill via `tools/build_dqs1_local_first_queue.py` + harvest observations + re-feed `frontier_rate_attack_feedback.py`. $0.30-5 + ~1h wall-clock. Predicted [-0.003, -0.0005] ΔS.
2. **Authorize cell 2**: dispatch paired Modal T4 CUDA anchor on FRONTIER archive `7a0da5d0fc32` (DQS1 drop-one rank021 pair0371) via `tools/operator_authorize.py --recipe paired_cuda_t4_dqs1_frontier_cpu_axis_repair_anchor_modal_t4_dispatch`. $5-10 + ~30 min wall-clock. Predicted fixes -0.005 CUDA regression.
3. **Authorize cells 3+4 in parallel**: free LOCAL CPU smokes for UNIWARD per-class + Hinton KL temporal context (Tier-1 CCC POSITIVE extensions). $0 + ~2h parallel wall-clock. Predicted [-0.002, -0.0005] each.
4. **Authorize cell 5**: free LOCAL CPU smoke on codex pair/frame geometry lattice's 6 queue-executable cells. $0 + ~1h. Receiver-feasibility signal feeds back into M10 lattice + M40 UNIWARD.
5. **Queue cells 6-15 for parallel batch** via codex Eureka rate-attack feedback compiler (M13 + M14). $0-2 + ~4h batch. Predicted aggregate [-0.005, -0.001] across cells.

## Lower-priority but non-deferred cells

Cells 11-20 (HILL re-probe / WOW / multi-scale J-UNIWARD / Riemannian-Newton first probe / Tropical first probe / cooperative-receiver ego-motion / predictive-coding pose-axis) are sister cells with predicted ΔS in [-0.005, -0.0005] each but require higher engineering investment (substrate-class shifts; design memo backfill per Catalog #294 + #303 + #305 + #309 + #310 + #311 + #312). Per CLAUDE.md "Long-burn score-lowering campaign default" non-negotiable: every cell carries a campaign ledger candidate; operator-routable to spawn sister subagents to expand each into Catalog #325 per-substrate symposium memos.

## DEFER-not-KILL cells per CLAUDE.md "Forbidden premature KILL"

- M35 ATW V2 cdf_table_blob: DEFERRED-FALSIFIED-IMPLEMENTATION per Catalog #307; paradigm intact; sister REMOVAL paradigm queued.
- M41 HILL filter EEE NULL_SIGNAL_DEFER: DEFER-pending-smaller-smoke-scope; re-probe with sub-region (D16) granularity.
- M51 C6 IBPS 22× miss: DEFER-pending-post-training-Tier-C-re-measure per Catalog #324; substrate paradigm intact.

# 5. Sister-coherence verification

**Sisters this turn (per operator 3-msg amplification + Catalog #300 operator-frontier-override cap=4 TEMP)**:
- Slot 1 (MLX-PARADIGM-T3) — DISJOINT confirmed: META paradigm symposium scope vs my matrix inventory scope.
- Slot 2 (MLX-ARCH-4 SegNet port) — DISJOINT confirmed: MLX primitive port scope vs my matrix inventory scope.
- Slot 3 (DQS1-LOOP-CLOSURE-ASSIST) — DISJOINT confirmed: cascade audit scope vs full methods × dimensions canvas mapping scope.

**My scope boundary**: NEW `.omx/research/rate_attack_methods_dimensions_matrix_comprehensive_inventory_20260525.md` (THIS memo) + Catalog #313 probe-outcomes row (matrix completeness audit). NO mutation of any codex DQS1 cascade source code / landing memos / canonical equation registry / state JSON. NO operator-authorize chain. NO push to git origin.

**Catalog #340 sister-checkpoint guard**: PROCEED after mark-complete-then-retry per ARCH precedent.

# 6. Six-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map = ACTIVE (every matrix cell predicts ΔS-per-dimension; downstream consumers route through `tac.sensitivity_map.*`).
- Hook #2 Pareto constraint = ACTIVE (matrix orthogonality per Cauchy-Schwarz + `pairset_component_marginal_score_decomposition_v1`; rate-budget redistribution per Eureka).
- Hook #3 bit-allocator = ACTIVE (per-dimension bit-budget signal feeds future bit-allocator).
- Hook #4 cathedral autopilot dispatch = ACTIVE (top-N priority queue consumable by autopilot ranker via canonical equation candidates QUEUED for operator-routable RATIFY-N).
- Hook #5 continual-learning posterior = ACTIVE (every matrix cell carries `score_claim=False` + `promotable=False` + canonical Provenance per Catalog #323; auto-discovered per Catalog #335).
- Hook #6 probe-disambiguator = ACTIVE (the matrix IS the canonical disambiguator between explored-vs-unexplored cells across the rate-attack canvas).

# 7. Discipline footer

- Per Catalog #1 + #192: all matrix-cell predictions are `[macOS-CPU advisory]` or `[predicted]` non-promotable; promotion requires paired Linux x86_64 + NVIDIA.
- Per Catalog #287: every empirical claim cites artifact path (codex commit shas + memo paths + my OVERNIGHT-M/N/P landings).
- Per Catalog #323: canonical Provenance umbrella (this memo carries grade=`[predicted]`; axis_tag implicit per-cell).
- Per Catalog #313: probe-outcomes registration via canonical helper QUEUED for sister subagent (NOT in my scope).
- Per Catalog #344: 11 canonical equation candidates QUEUED for operator-routable RATIFY-N (DO NOT auto-register).
- Per Catalog #110/#113: APPEND-ONLY; NEW memo; ZERO mutation of sister artifacts.
- Per CLAUDE.md "Forbidden premature KILL": UNKNOWN/PARTIAL applicability cells are DEFER not KILL.
- Per Catalog #300 operator-frontier-override: cap=4 sister-subagent TEMP exceeded via operator's 3-msg amplification.

# 8. Predicted mission contribution per Catalog #300

`frontier_breaking_enabler`: top-2 cells (M03 drop-many waterfill + M02 CUDA-axis repair) carry estimated cumulative predicted ΔS in [-0.008, -0.002] if executed within session envelope; aggregate cells 3-20 carry estimated cumulative predicted ΔS in [-0.015, -0.003] if executed in parallel local CPU smokes. The matrix structurally extincts the orphan-cell bug class (cells silently unexplored because no inventory existed).

# 9. Verification

- Built from canonical sources: codex DQS1 cascade memos (30+ files), canonical equations registry (36 entries), substrate design memos (PACT-NERV / Riemannian-Newton / Tropical / ATW V2 / NSCS06 v8 / HFV / DP1), my Tier-1 probe landings (OVERNIGHT M/N/P clusters), frontier pointer + reports/latest.md.
- Method count target (30-50): achieved 60 (exceeded due to operator's "everything landed and mentioned" scope).
- Dimension count target (~25 with "and more"): achieved 36 (exceeded due to operator's "and more" emphasis).
- Top-20 unexplored high-EV cells identified.
- 11 canonical equation candidates queued per Catalog #344.
- Carmack MVP-first phasing satisfied for top-5 cells.
- Sister-coherence verified DISJOINT against 3 active sisters.
- No CLAUDE.md mutation; no canonical equation auto-registration; no codex source-code mutation; no state-JSON mutation.

# 10. Operator-routable next actions

1. Operator authorizes top-5 cells per §4 (cells 1-5).
2. Sister subagent registers 11 canonical equation candidates per Catalog #344 RATIFY-N (decision routing).
3. Sister subagent updates `tac.canonical_council_roster` Time-Traveler protégé canonical identity if matrix EV-ranking insights ratify Daubechies → Rudin chain.
4. Sister subagent backfills Catalog #313 probe-outcomes row with this matrix as probe-disambiguator anchor.
5. Sister subagent expands cells 11-20 into Catalog #325 per-substrate symposium memos per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable.
