# Comprehensive codebase research sweep

Snapshot at 2026-05-20 / 05:05Z of every substrate, technique, tool, and research artifact in the pact repository — the universe behind PR #110 (`hnerv_fec6_fixed_huffman_k16`). This is the canonical internal accounting against which the tight public overview (`docs/comma_lab_overview.md`) is calibrated. Cites CLAUDE.md non-negotiables, Catalog # gates, and canonical-helper paths freely (internal artifact).

Headline counts: 52 substrate packages under `src/tac/substrates/` + 47 cathedral consumers under `src/tac/cathedral_consumers/` + 698 tools + 235 STRICT preflight catalog gates + 93 operator-authorize recipes + 122 design memos + 1038 lanes in `.omx/state/lane_registry.json` (Level 0=290, Level 1=658, Level 2=90, Level 3=1) + 11 canonical equations in the registry + 2238 research memos. PR #110 ships ~1140 LOC; the apparatus that produced it is ~3 orders of magnitude larger.

---

## Contents

- [1. Empirical anchors](#1-empirical-anchors)
- [2. Substrate inventory](#2-substrate-inventory)
- [3. Technique inventory](#3-technique-inventory)
- [4. Tooling and meta-engineering](#4-tooling-and-meta-engineering)
- [5. Methodology and discipline](#5-methodology-and-discipline)
- [6. Research lineage and canonical references](#6-research-lineage-and-canonical-references)
- [7. Empirical-vs-design honest accounting](#7-empirical-vs-design-honest-accounting)

---

## 1. Empirical anchors

End-to-end measurements actually run on contest-1:1 hardware. Score literals axis-tagged. Sources: `.omx/state/canonical_frontier_pointer.json`, `.omx/state/modal_call_id_ledger.jsonl`, `.omx/state/continual_learning_posterior.jsonl`, inventory memo Section B.

| Name | Class | Score [axis tag] | Hardware | Archive sha[:12] | Date | Lane id |
|---|---|---|---|---|---|---|
| `hnerv_fec6_fixed_huffman_k16` (PR #110) | within-HNeRV-family bolt-on stack | `0.192051 [contest-CPU]` + `0.226210 [contest-CUDA T4]` paired | Modal Linux x86_64 + Modal Tesla T4 | `6bae0201fb08` | 2026-05-15 | `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515` |
| `hnerv_ft_microcodec` (PR #101 replay) | within-HNeRV-family | `0.192845 [contest-CPU]` | Modal Linux x86_64 | n/a (upstream) | bot-recomputed | sister anchor |
| `lane_a1_*` paired | A1 substrate engineering | `0.19285 [contest-CPU]` + paired CUDA | Modal | `87ec7ca5f2f3` | 2026-05-13 | `lane_a1_*` |
| `pr106_format0d_latent_score_table` | within-HNeRV-family CUDA-side | `0.205330 [contest-CUDA T4]` | Modal T4 | `9cb989cef519` | 2026-05-15 | `lane_pr106_format0d_latent_score_table` |
| Earlier within-HNeRV-family iterations | bolt-on | multiple `[contest-CUDA T4]` `0.95-1.45` | Modal T4 | various | various | retired-config pre-PR101-substrate |
| `c6_ibps` 50ep IB smoke | information-bottleneck | `final_score = 3.04 [contest-CUDA A10G]` vs design-time band `[0.113, 0.163]` | Modal A10G | `be06a4b09726` | 2026-05-17 | `substrate_c6_e4_mdl_ibps` |
| `nscs06` Strip-Everything v6→v7 | composition substrate | v6 `105.15` → v7 `58.89 [contest-CUDA T4]` (44% improvement, one cargo-cult-unwind iteration) | Modal T4 | various | 2026-05-15→16 | `lane_nscs06_*` |
| `lane_g_v3` historical | within-class | `1.05 [contest-CUDA T4]` | Vast.ai 4090 | n/a | pre-HNeRV-family | `lane_g_v3` |
| `apogee_int4` PTQ smoke | low-bit weight quantization | `1.42866394 [contest-CUDA T4]` | Modal T4 | n/a | FALSIFIED-at-naive-PTQ | `lane_apogee_intN` |

Honest summary: outside the HNeRV-family local cluster (`0.192-0.21` band on CPU), every other paradigm empirically tested either falsifies at a specific implementation config (NOT at the paradigm class — per CLAUDE.md "Forbidden premature KILL without research exhaustion") or has not yet been pushed end-to-end to a paired CPU + CUDA anchor on contest-1:1 hardware.

---

## 2. Substrate inventory

52 substrate packages under `src/tac/substrates/`. Each has a corresponding `experiments/train_substrate_*.py` trainer and `.omx/operator_authorize_recipes/substrate_*.yaml` recipe.

### 2.1 Within-HNeRV-family (canonical substrate base)

| Substrate | Class | Status | Lane id | Design memo |
|---|---|---|---|---|
| `pr101_lc_v2_clone` | HNeRV ft microcodec clone | L2 INTEGRATION (canonical post-PR101 base) | `lane_pr101_*` | inventory C.6 |
| `sane_hnerv` | HNeRV parity faithful baseline | L1 SCAFFOLD | `lane_sane_hnerv_*` | `feedback_sane_hnerv_*` |
| `pr95_lora_dora` | LoRA/DoRA adapter on PR95 HNeRV | L1 SCAFFOLD | `lane_pr95_lora_dora_*` | n/a |
| `a1` | A1 substrate-engineering paired-CPU+CUDA anchor | L2 INTEGRATION (paired empirical) | `lane_a1_*` | various |
| `a1_plus_lapose` | A1 + pose-axis codec | L2 SCAFFOLD | `lane_a1_plus_lapose` | inventory C.5 |
| `a1_plus_wavelet_residual` | A1 + Daubechies wavelet residual sidechannel | L2 SCAFFOLD | `lane_a1_plus_wavelet_residual` | inventory C.7 |

### 2.2 NeRV-family expansion beyond HNeRV (8 substrates per inventory C.6)

| Substrate | Class | Status |
|---|---|---|
| `tc_nerv` | Temporal-Convolutional NeRV | L1 SCAFFOLD |
| `block_nerv` | Block-decomposed NeRV | L1 SCAFFOLD |
| `ff_nerv` | Feature-grid NeRV | L1 SCAFFOLD |
| `ds_nerv` | Deformable-Scene NeRV | L1 SCAFFOLD (research_only after API crash) |
| `hi_nerv` | Hierarchical NeRV | L1 SCAFFOLD (research_only after API crash) |
| `e_nerv` (architecture file only) | Sister NeRV variant | L0 design |
| `ego_nerv` | Ego-motion-conditioned NeRV | L0 design |
| `nervdc` | Sister NeRV variant | L0 design |

### 2.3 Predictive-coding world models (inventory C.1)

| Substrate | Class | Status |
|---|---|---|
| `time_traveler_l5_z6` | Z6 multi-layer FiLM (depth=3, ~300K) primary predictive-receiver substrate | L1 SCAFFOLD (Wave 2 driver mode-routing bug closed at gate; reactivation queued) |
| `time_traveler_l5_z7_lstm_predictive_coding` | Z7 LSTM/GRU temporal predictor | L0 design |
| `time_traveler_l5_z7_mamba2` | Z7-Mamba-2 selective state-space (Dao-Gu 2024) | L1 SCAFFOLD |
| `time_traveler` + `time_traveler_l5_autonomy` | Z6/Z7/Z8 substrate skeletons | L0-L1 scaffolds |
| `c1_world_model_foveation` | DreamerV3 sister with foveation | L1 SCAFFOLD research_only |

Z8 hierarchical canonical quadruple (Daubechies wavelet + Mallat multi-resolution + Rao-Ballard hierarchy + Wyner-Ziv side-info) is design-only — Catalog #312 strict gate enforces the quadruple presence.

### 2.4 Cooperative-receiver framings (inventory C.2)

| Substrate | Class | Status |
|---|---|---|
| `z4_cooperative_receiver_loss` | Bolt-on objective term | L1 SCAFFOLD (full path council-gated) |
| `wyner_ziv_cooperative_receiver` | Wyner-Ziv side-information sister | L1 SCAFFOLD |
| `atw_codec_v1` | Atick-Tishby-Wyner triple V1 | folded into V2 |
| `atw_codec_v2` | ATW V2 with Faiss-IVF-PQ per-region SegNet softmax | L1 SCAFFOLD; ATW V2 D4 conditioning probe returned INDEPENDENT (MI=0.006385 bits/symbol) per probe-outcomes ledger Catalog #313 |

### 2.5 Information-Bottleneck framings (inventory C.3)

| Substrate | Class | Status |
|---|---|---|
| `c6_e4_mdl_ibps` | C6 IBPS canonical Path B quadruple | EMPIRICAL FALSIFIED at 24-dim latent; paradigm intact; reactivation = latent-dim sweep + β_ib calibration |
| `tishby_ib_pure` | Tishby IB-pure variant | L0 design |

### 2.6 Pretrained driving priors (inventory C.4)

| Substrate | Class | Status |
|---|---|---|
| `pretrained_driving_prior` (DP1) | Comma2k19 OOD-codebook + Wyner-Ziv composition | L1 SCAFFOLD Phase 2 (Catalog #209 enforces canonical Comma2k19 frame iterator; Catalog #210 enforces codebook provenance) |
| `driving_prior_world_model` | DP1 + world-model variant | L0 design |

### 2.7 Pose-axis / foveation / spatial-sparse (inventory C.5)

| Substrate | Class | Status |
|---|---|---|
| `a1_plus_lapose` | LAPose pose codec composition | L2 SCAFFOLD |
| `sar_coherent_pose_pairs` | Spatial-AR coherent pose pairs | L1 SCAFFOLD |
| `sabor_boundary_only_renderer` | SegNet-class-boundary-only renderer | L1 SCAFFOLD |
| (RAFT-derived poses in `src/tac/raft_radial_pose.py` + `src/tac/raft_pose.py`) | optical-flow-derived pose channel | LANDED helper |
| (TT5L V2 foveation) | telescopic foveation revival | DESIGN-ONLY (V1 REFUSE verdict; V2 staged probes) |

### 2.8 Non-NeRV substrate architectures (inventory C.7)

| Substrate | Class | Status |
|---|---|---|
| `cool_chic` | Cool-Chic coordinate-based hierarchical codec | DESIGN-ONLY (open export-contract gate) |
| (`C3` from Kim et al. 2023) | C3 single-image/video neural compression | DESIGN-ONLY (open export-contract gate) |
| `wavelet` | Daubechies wavelet residual sidechannel | L1 SCAFFOLD |
| `hybrid_renderer_residual` | Composition substrate | L1 SCAFFOLD |
| `siren` | SIREN sinusoidal coordinate MLP (Sitzmann et al. 2020) | L1 SCAFFOLD (smoke timeout at 1h T4; reactivation = longer-budget 4090) |
| `vq_vae` | Discrete-latent codec (van den Oord 2017) | L1 SCAFFOLD |
| `grayscale_lut` | Selfcomp PR #56 grayscale-LUT extension | L1 SCAFFOLD design-only beyond what PR #56 shipped |
| `coord_mlp_residual_sidecar` | Coordinate-MLP residual sidecar | L1 SCAFFOLD |
| `rudin_floor_interpretable_ml` | Rudin SLIM / falling-rule-list interpretable-ML substrate | L1 SCAFFOLD |
| (`src/tac/mae_mask_aug.py` + Lane MAE-V) | MAE mask augmentation (He et al. 2021 arXiv:2111.06377) | LANDED helper; Lane MAE-V task #197 completed |
| (CLADE per Tan et al. 2021 arXiv:2012.04644) | class-adaptive denormalization | L1 SCAFFOLD (`src/tac/categorical_*.py` + `src/tac/dp_sims_renderer.py`) |
| (SPADE per Park et al. 2019 arXiv:1903.07291) | spatially-adaptive denormalization | L1 SCAFFOLD (`src/tac/dp_sims_renderer.py` default conditioning) |
| (Quantizr-faithful in `experiments/` historical) | PR-56-lineage faithful reimplementation | Historical: `[contest-CUDA T4]` 0.33-0.41 |
| (Diffusion renderer) | speculative | DESIGN-ONLY |

### 2.9 Codec primitives + entropy coding (inventory C.8)

| Substrate / primitive | Class | Status |
|---|---|---|
| `tac.codec.wyner_ziv_layer` | Wyner-Ziv layer (canonical) | LANDED + Catalog #321/#322 phantom-savings refusal |
| `tac.wyner_ziv_deliverability.proof_builder` | Deliverability proof builder | LANDED |
| Hierarchical Wyner-Ziv composition (== Z8) | Daubechies+Mallat+Rao-Ballard+Wyner-Ziv quadruple | DESIGN-ONLY (Catalog #312 strict gate) |
| `stc_v2` (STC-Dasher arithmetic-coding maximalism) | Syndrome-trellis pushed to arithmetic-coder limit | L1 SCAFFOLD |
| `z3_balle_hyperprior_bolton` + `z3_g1_*` + `balle_renderer` + `balle_*` codec/transform helpers | Ballé hyperprior (CompressAI primitives registered per Catalog #169) | L1 SCAFFOLD |
| `self_compress_nn` + `src/tac/block_fp_codec.py` + `src/tac/block_fp_jfg.py` | Selfcomp block-FP (PR #56 lineage) | LANDED helpers |
| (Hessian-block-FP) | Sister using Hessian-weighted blocks | SCAFFOLDED helper |
| `src/tac/uniward_delta.py` + `src/tac/wavelet_variance.py` + `src/tac/saliency_inversion.py` + `src/tac/score_gradient_param_saliency.py` | UNIWARD texture-aware encoding (Holub-Fridrich-Denemark 2014) | LANDED helpers |
| (water-bucket filling Lane Ω-W per tasks #233 SHIPPED + #244 V2 + #272 inflate handler + #356 V3 launch-ready) | water-filling bit allocation | LANDED |

### 2.10 Self-compression family (inventory C.9)

| Substrate | Class | Status |
|---|---|---|
| `self_compress_nn` (SC++ SegMap + KL distill) | KL-distilled SegNet surrogate (Hinton 2014 distillation) | L1 SCAFFOLD |
| (MDL FP4 TTO) | MDL-optimal FP4 via test-time optimization | SCAFFOLDED |
| `lane_17_imp` | Iterative magnitude pruning | L2 SCAFFOLD (council symposium deferred dispatch pending cycle-0 empirical) |

### 2.11 Composition substrates + stacking (inventory C.10)

| Substrate | Class | Status |
|---|---|---|
| `nscs06_carmack_hotz_strip_everything` + `nscs06_v8_path_b_wavelet` | Strip-Everything per-class chroma anchor | EMPIRICAL at v7 `58.89 [contest-CUDA T4]` (44% improvement v6→v7 in one cargo-cult-unwind iteration); v8 Path B queued |
| `nscs01_nullspace_split_renderer` | Renderer split along nullspace direction | L1 SCAFFOLD |
| `nscs02_downsampled_renderer` | Downsampled-input renderer | L1 SCAFFOLD |
| `nscs03_end_to_end_balle_joint_codec` | Ballé hyperprior + end-to-end-trained renderer | L1 SCAFFOLD |
| (`stack_of_stacks` recipe-level composition framework) | Composition recipe framework | LANDED |
| `s2sbs_byte_stuffing` | Sample-to-sample byte-stuffing | L1 SCAFFOLD |
| `d1_segnet_margin_polytope` | SegNet margin polytope substrate | L1 SCAFFOLD |
| `d4_wyner_ziv_frame_0` | Wyner-Ziv frame-0 substrate | L1 SCAFFOLD |
| `yucr` | YUCR composition substrate | L1 SCAFFOLD |
| `v8_learned_compression_faiss` | V8 Faiss IVF-PQ learned compression (`substrate_v8_learned_compression_faiss_modal_a100_smoke.yaml`) | L1 SCAFFOLD |

### 2.12 Higher-order optimization framings (inventory C.11)

| Primitive | Class | Status |
|---|---|---|
| (Riemannian-Newton substrate engineering) | Manifold-aware second-order | DESIGN-ONLY |
| (Tropical d_seg solver) | Tropical-algebra segmentation distortion solver | DESIGN-ONLY |
| (`src/tac/solvers/` Joint-ADMM coordinator per Boyd 2011 ADMM) | Cross-substrate consensus | SCAFFOLDED |
| (3-set Venn classification: high-pair-invariant / pair-specific / per-pair) | Per-pair classifier | EMPIRICAL |

---

## 3. Technique inventory (cross-cutting)

Per-paradigm class techniques that span multiple substrates:

### 3.1 Distillation lineage (Hinton 2014 arXiv:1503.02531)

- `src/tac/freezing/frozen_teacher_distillation.py` — canonical KL-on-logits T=2.0 distillation primitive
- `src/tac/substrates/self_compress_nn/` SC++ SegMap + KL distill
- Council `D` design memo per CLAUDE.md "EMA — NON-NEGOTIABLE" (kl_on_logits during specific training phases)

### 3.2 Telescopic foveation revival

- TT5L V2 foveation + LAPose redesign per inventory C.5
- `src/tac/raft_radial_pose.py` + `src/tac/raft_pose.py` RAFT-derived focus-of-expansion-aware pose decomposition
- `src/tac/lapose_motion_atom_allocator.py` pose-axis dedicated codec

### 3.3 Engineered corrections (Slot 16 lineage)

- `src/tac/engineered_corrections.py` + `src/tac/engineered_corrections_v2.py` + `src/tac/engineered_correction_readiness.py`
- MPS-engineering corrections: Kahan summation, softmax epsilon, fp32 matmul (canonical equation `mps_drift_architecture_class_dependent_v1`)

### 3.4 Freezing toolkit (`src/tac/freezing/`)

- 8 focused helpers per the operator-canonical pattern: `pose_gradient_stop_after_warmstart` + `lora_style_renderer_adapter` + `frozen_teacher_distillation` + `swa_checkpoint_averaging` + `lottery_ticket_extraction` + `ema_freeze_at_eval` + `__init__.py` canonical surface + `src/tac/training_curriculum/__init__.py` freezing-after-warm-start
- Consumed by `src/tac/substrates/a1_plus_wavelet_residual/score_aware_loss.py`

### 3.5 Score-aware loss (canonical helper)

- `src/tac/substrates/_shared/score_aware_common.py` — canonical `score_pair_components` consumed by every substrate trainer (Catalog #164 enforces canonical scorer-preprocess routing)
- `src/tac/training/score_weighted_reconstruction_loss.py` per-pair-weighted loss

### 3.6 Streaming master-gradient

- `src/tac/training/streaming_master_gradient_hook.py` (~17.6K) — training-time streaming master-gradient hook
- `src/tac/streaming_prediction/` — streaming prediction ledger + Kalman + dashboard (Slot MG-5, Catalog #351)

---

## 4. Tooling and meta-engineering

### 4.1 Cathedral autopilot

- `tools/cathedral_autopilot_autonomous_loop.py` (316K) — primary autonomous loop with `discover_and_register_consumers` auto-discovery (Catalog #336) + `rerank_candidates_via_master_gradient` invocation (Catalog #337)
- `tools/cathedral_autopilot.py` (134.5K) — original entry point
- `tools/cathedral_autopilot_activation_summary.py`
- `tools/cathedral_autopilot_meta_lagrangian_bridge.py`
- 47 cathedral consumers under `src/tac/cathedral_consumers/` per canonical Protocol contract (Catalog #335): `analytical_solve_extinctions_consumer` / `atom_consumer` / `bit_allocator_per_pair_consumer` / `bit_level_score_critical_bits_consumer` / `bottom_k_free_entropy_byte_consumer` / `canonical_equation_lookup_consumer` / `contest_exploits_consumer` / `contest_oracle_consumer` / `cpu_axis_optimal_consumer` / `early_stopping_consumer` / `ema_decay_formula_consumer` / `engineered_correction_targeting_consumer` / `experimental_extinctions_consumer` / `findings_lagrangian_consumer` / `formula_extinctions_consumer` / `gradient_informed_decoder_pruning_consumer` / `hf_jobs_dispatcher_consumer` / `information_theoretic_floor_consumer` / `master_gradient_aggregate_consumer` / `master_gradient_per_pair_consumer` / `mps_diagnostic_consumer` / `mps_gap_experiment_consumer` / `mps_viable_prescreen_consumer` (Catalog #341) / `multi_granularity_comparison_consumer` (Catalog #352) / `packetir_candidate_queue_consumer` / `per_byte_sensitivity_consumer` / `per_pair_coding_budget_allocation_consumer` / `per_pair_difficulty_atlas_consumer` / `per_pair_gradient_clustering_consumer` / `per_pair_kkt_residuals_consumer` / `per_pair_lagrangian_lambda_bisection_consumer` / `per_pair_lora_supervision_signal_consumer` / `per_pair_pareto_envelope_consumer` / `per_pair_volterra_cross_terms_consumer` / `per_segnet_class_chroma_consumer` / `procedural_codebook_generator_consumer` / `risk_adjusted_ranking_consumer` (Catalog #349) / `score_lagrangian_consumer` / `score_weighted_reconstruction_error_consumer` / `solvers_consumer` / `streaming_prediction_consumer` (Catalog #351) / `substrate_fit_diagnostic_consumer` / `top_k_byte_sensitivity_consumer` / `uncertainty_weighted_loss_consumer` / `unified_action_consumer` / `utility_curves_consumer`

### 4.2 Master-gradient extractor + per-pair Lagrangian planner

- `tools/extract_master_gradient.py` — per-pair / per-byte / per-tensor master-gradient extractor (CPU-only against landed archive)
- `tools/master_gradient_xray.py` — XRAY viz tool
- `tools/audit_master_gradient_feasibility.py` + `tools/audit_master_gradient_wire_in_coverage.py` + `tools/backfill_master_gradient_score_axis_dominance.py` + `tools/build_master_gradient_*` (operator-plan + brotli-operator-candidate + trust-region-candidates)
- `src/tac/master_gradient.py` + `src/tac/master_gradient_consumers.py` + `src/tac/master_gradient_comparison/multi_granularity.py` (Slot MG-3)
- Per-pair Lagrangian planner: `src/tac/master_gradient_consumers.load_optimal_plan_for_archive` (Catalog #319 v2 cascade)
- Authority boundary (Catalog #318): canonical `CandidateModificationSpec` + `grammar_aware_operator` only; raw-byte FD forbidden

### 4.3 Deterministic packet compiler (CLAUDE.md non-negotiable)

- `src/tac/packet_compiler/deterministic_compiler.py` (40.5K) + `deterministic_compiler_cli.py` + `golden_vectors/` + companions: `balle_hyperprior.py` / `cheng2020.py` / `cooperative_receiver_grammars.py` / `custom_binary_container.py` / `dynamic_video_adaptive_packet.py` / `factorized_prior.py` / `magic_codec.py` / `magic_codec_dense_streams.py` / `pr100_schema_driven_decoder.py` / `pr101_conv4_storage_perms.py`
- Catalog #158 `check_deterministic_compiler_canonical_use` — refuses new packet-compilation surfaces bypassing the canonical compiler

### 4.4 Canonical equations registry

- `tac.canonical_equations` package (~1700 LOC) + fcntl-locked JSONL append-only at `.omx/state/canonical_equations_registry.jsonl` per Catalog #344
- 11 registered equations: `brotli_cascade_bounded_per_stream_v1` / `canonical_frontier_pointer_v1` / `convergence_slope_early_stop_v1` / `cpu_axis_optimal_archive_selector_v1` / `ema_decay_substrate_stage_aware_v1` / `master_gradient_locality_violation_by_codec_v1` / `mps_drift_architecture_class_dependent_v1` / `per_byte_leverage_uniformly_distributed_v1` / `per_pair_loss_weighting_optimal_v1` / `per_pair_master_gradient_score_impact_taylor_v1` / `score_marginal_lagrange_multipliers_v1`
- Operator CLI: `tools/list_canonical_equations.py` + `tools/recalibrate_equation.py`
- BayesianPosterior helper at `src/tac/canonical_equations/bayesian_posterior_update.py` (Slot MG-2)

### 4.5 Modal call-id ledger + harvester

- `tac.deploy.modal.call_id_ledger` — fcntl-locked APPEND-ONLY at `.omx/state/modal_call_id_ledger.jsonl` per Catalog #245 (canonical 4-layer pattern: schema + register + update_outcome + query)
- `tools/harvest_modal_calls.py` + `tools/parallel_harvest_actuator.py` — closes the spawn-and-lose failure mode (CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE")
- `register_dispatched_call_id_fail_closed` per Catalog #339 (silent-no-spawn extinction)
- Catalog #330 `check_modal_harvesters_record_call_id_outcome`

### 4.6 Subagent crash-resume + checkpoint protocol

- `tools/subagent_checkpoint.py` + fcntl-locked JSONL at `.omx/state/subagent_progress.jsonl` per Catalog #131/#206
- Catalog #302 `check_sister_subagent_scope_overlap_via_checkpoint_jsonl` — extincts edit-time-checkpoint collision
- Catalog #314 + #340 sister-checkpoint guards (DETECT + PREVENT bidirectional)
- `src/tac/commit_safety/sister_checkpoint_guard.py` + `tools/check_sister_checkpoint_before_git_add.py`

### 4.7 235 STRICT preflight gates

- `src/tac/preflight.py` (canonical) + `tools/preflight_hook.py`
- 235 cataloged gates (Catalog #1-#348 numbering with gaps; #348 most recent)
- META-meta gates: #118 duplicate-number + #159 catalog-text-matches-strict-value + #176 strict-callsites-have-CLAUDE.md-row + #185 LIVE_COUNT drift + #186 catalog-claim-via-serializer + #299 catalog quota brake (≤#400)
- Newest landings (2026-05-18→19): #335-#348 cathedral consumer Protocol contract + canonical equations + frontier pointer + canonical roster helper + sister checkpoint guard + retroactive sweep gate

### 4.8 Frontier pointer canonical

- `tac.canonical_frontier_pointer` + `.omx/state/canonical_frontier_pointer.json` per CLAUDE.md "Frontier scores are pointer-only — NON-NEGOTIABLE"
- `tools/refresh_canonical_frontier.py` (human-readable + `--json` + `--strict` + `--update-upstream`)
- Catalog #316 `check_reports_latest_md_not_stale_vs_canonical_frontier` + Catalog #343 `check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded`
- Auto-refresh on Modal/HF Jobs dispatch completion

### 4.9 Empirical per-X optimal codec planner (DuckDB unification)

- `src/tac/empirical_per_x_optimal_codec_planner/` + `src/tac/canonical_duckdb/`
- `tac.master_gradient_consumers.load_optimal_plan_for_archive` reads per-pair plans

### 4.10 Probe outcomes ledger

- `tac.probe_outcomes_ledger` + `.omx/state/probe_outcomes.jsonl` per Catalog #313 (4-layer canonical pattern mirroring #245)
- `tools/check_predecessor_probe_outcome.py` (canonical operator CLI)
- 7-verdict taxonomy: INDEPENDENT / KILL / DEFER / PROMOTE / PROCEED / PARTIAL / OPERATOR_REVIEW_REQUIRED
- 3-status taxonomy: blocking / advisory / expired (30-day staleness window per #298)
- Runtime wire-in: `tools/operator_authorize.py::_check_predecessor_probe_outcome`

### 4.11 Canonical Provenance helper

- `tac.provenance` package (~1500 LOC, 5 files: contract / builders / validator / adapters / __init__) per Catalog #323
- Frozen `Provenance` + `ScoreClaim` dataclasses + 6-kind + 8-grade taxonomy
- `tools/audit_provenance_compliance.py` (operator audit; classifies 6 violation categories)
- `docs/provenance_canonical_usage.md` developer guide

### 4.12 Wyner-Ziv deliverability proof builder

- `tac.wyner_ziv_deliverability.proof_builder` per Catalog #319 v2 cascade
- `DeliverabilityProof` + `DeliverabilityTier` 4-tier IntEnum (TIER_1_ZERO_COST / TIER_2_CONSTANTS / TIER_3_WAIVER_REQUIRED / TIER_4_FORBIDDEN)
- Sidecar persistence at `.omx/state/wyner_ziv_deliverability/proof_<sha[:12]>_<utc>.json`
- Catalog #321 phantom-savings refusal + Catalog #322 autopilot composition_alpha refusal

### 4.13 Pre-dispatch codex adversarial review automation

- `tools/run_codex_review_for_dispatch.py` per Catalog #271 + #281 (fail-closed invocation-error) + #282 (cache key includes dirty-tree fingerprint) + #283 (operator_authorize fails closed on missing helper)
- Cache schema: composite SHA-256 of `(git_HEAD_sha, recipe_sha, trainer_sha, dirty_tree_fingerprint, untracked_relevant_fingerprint)` keyed JSONL at `.omx/state/codex_pre_dispatch_review_cache.jsonl` with 1h TTL
- 4-verdict taxonomy: approve / advisory / needs-attention / no-ship / invocation-error
- Cost gate: only invokes codex when `estimated_cost_usd > $1`

### 4.14 Operator-authorize harness

- `tools/operator_authorize.py` — canonical entry point (Catalog #176 + #162)
- 30-second local pre-deploy harness `tools/local_pre_deploy_check.py` (8 checks per Catalog #243 + #279 fail-closed)
- Routes: `_dispatch_modal` + `_dispatch_local_mps` + `_dispatch_local_cpu` + `_dispatch_lightning` + `_dispatch_vastai`
- `--target {auto|modal|vastai|lightning|local|local-mps|local-cpu}` one-arg toggle per Catalog #317
- Paired-env bypass discipline per Catalog #199 + #202 (CONFIRMED + BUDGET_USD; CLEAN_HEAD + TRUSTED_SENTINELS_CLEAN_VERIFIED)
- Recipe validation: Catalog #152 (required input files) + #270 (canonical dispatch optimization protocol — Tier 1/2/3 umbrella) + #240 (recipe-vs-trainer-state consistency)

### 4.15 Lane registry + dispatch claims

- `.omx/state/lane_registry.json` (1038 lanes; 7-gate maturity; Level 0/1/2/3) per Catalog #90
- `.omx/state/active_lane_dispatch_claims.md` per Catalog #131 + `tools/claim_lane_dispatch.py` (canonical with `prune` archival)
- `tools/lane_maturity.py` (canonical mutation surface)

### 4.16 Sister library `adpena/tac`

- Public OSS extraction of task-aware compression primitives
- 56-test pytest suite + CI matrix (Python 3.11/3.12) — CI was 14d red on missing test paths; Slot O recovery in flight
- MIT license + pre-commit + ruff + canonical pyproject

### 4.17 Other infrastructure

- `tac.canonical_council_roster` per Catalog #346 — 4-co-lead structure (Shannon + Dykstra + Rudin + Daubechies) + Grand-Council 20-seat roster
- `tac.canonical_task_status` per Catalog #331 — single source of truth for task lifecycle
- `tac.codex_to_claude_inbox` per Catalog #333 — bidirectional Codex-Claude channel
- `tac.council_continual_learning` per Catalog #300 v2 frontmatter
- `tools/operator_briefing.py` summary wire-in
- `tools/audit_*.py` family (~12 audit tools)
- `src/tac/cathedral/consumer_contract.py` — canonical Protocol + validate helper
- `src/tac/symposium_impls/` — 9 symposium implementation packages (Catalog #265-#269)
- `src/tac/preflight_rudin_daubechies/` — interpretable preflight ranker (SLIM + falling-rule-list + Rashomon + GOSDT per Catalog #273-#278)
- `src/tac/autopilot_rudin_daubechies/` — autopilot sister of preflight composite per Catalog #250-#255

---

## 5. Methodology and discipline

### 5.1 Cargo-cult-unwind methodology

- NSCS06 v6 → v7 = 44% improvement on `[contest-CUDA T4]` (105.15 → 58.89) in ONE iteration via cargo-cult-unwind of 4-of-7 cargo-culted assumptions
- Operator-pioneered meta-engineering pattern
- Canonical procedure: enumerate substrate-design assumptions → classify HARD-EARNED-vs-CARGO-CULTED → design unwind probe per CARGO-CULTED → apply unwinds → re-test sextet → iterate to PROCEED-unconditional
- Companion: `docs/cargo_cult_unwind_methodology.md`
- Catalog #303 enforces per-substrate cargo-cult audit section in design memos

### 5.2 HARD-EARNED vs CARGO-CULTED classification

- Sister addendum: `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`
- HARD-EARNED: cite canonical reference + empirical anchor that validated transfer to dashcam-video context — preserve
- CARGO-CULTED: inherited from canonical reference without empirical testing of transfer — eligible for unwind probe
- Per-deliberation surfacing required at T2+ per Catalog #292
- Assumption-Adversary sextet-pact council seat per "Council conduct" Fix-7 amendment

### 5.3 Per-substrate optimal-form symposium discipline

- Catalog #315 (parent OPTIMAL FORM gate) + Catalog #325 (per-substrate symposium evidence sub-surface)
- Canonical 6-step contract: (1) cargo-cult audit per #303 + (2) 9-dim checklist per #294 + (3) observability surface per #305 + (4) sextet pact deliberation + (5) per-substrate reactivation criteria pinned + (6) Catalog #324 post-training Tier-C validation discipline
- 14-day symposium memo recency window required for paid dispatch >$0.30

### 5.4 Recursive 3-clean-pass adversarial review

- 3 consecutive clean rounds → SEAL (canonical)
- D-1 operator-declared SEAL alternative when external-adversary unanimous + Contrarian SUPER-VETO + 7-day cool-down + operator explicit close
- Per-round assumption-challenge axis (item 8 of protocol) per CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW"

### 5.5 4-tier council hierarchy

- T1 Working Group (unbounded; 1-3 members) / T2 Inner-Skunkworks (5-of-6 sextet; ≤3/day) / T3 Full Grand Council (5-of-6 sextet + ≥12-of-20 grand; ≤3/week) / T4 Symposium (6-of-6 sextet + ≥16-of-20 grand + specialist seats; ≤2/30d)
- Canonical roster (Catalog #346): 4 co-leads Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD; 11-voice inner council (added PR 95 author 2026-05-19); 20-seat Grand Council
- Catalog #291 (per-session META-ASSUMPTION cadence) + Catalog #292 (per-deliberation assumption surfacing) + Catalog #300 (v2 frontmatter discipline)
- Continual-learning wire-in: `tac.council_continual_learning.append_council_anchor` → `.omx/state/council_deliberation_posterior.jsonl`

### 5.6 8-axis bug-class extinction (orthogonal surfaces)

For substrate L2+ promotion / dispatch correctness:
- design-memo: #290 canonical-vs-unique + #294 9-dim checklist + #303 cargo-cult audit + #305 observability surface + #296 Dykstra-feasibility predicted-band
- runtime-effect: #220 substrate L1+ scaffold operational mechanism + #272 distinguishing-feature integration contract + #297 signal-axis destruction reversibility probe
- promotion-discipline: #233 L1→L2 canonical 4-gate
- retirement-discipline: #298 substrate L1 not stale dispatch
- council-discipline: #300 v2 frontmatter + #292 per-deliberation + #291 per-session cadence + #346 roster complete
- iteration-discipline: #315 OPTIMAL FORM before paid dispatch + #325 per-substrate symposium evidence
- post-training-validation: #324 predicted band post-training Tier-C validation
- per-substrate-symposium-evidence: #325

### 5.7 Master-gradient consumer canonical contract + auto-discovery

- Catalog #335 `check_cathedral_consumer_directory_package_exposes_canonical_contract`
- Catalog #336 `check_cathedral_autopilot_main_invokes_discover_and_register_consumers`
- Catalog #337 `check_rerank_candidates_via_master_gradient_invokes_consumers`
- Catalog #341 sister at MPS-prescreen routing markers
- Convention-over-configuration paradigm shift: 47 consumers auto-discovered without manual ranker-cascade edits

### 5.8 Canonical Provenance + axis-tagged evidence-grade discipline

- Catalog #323 META-class umbrella refusing score-claim rows without canonical Provenance sub-object
- 6 sister surfaces extincted at: #287 docstring overstatement + #249 misleading device-named output dirs + #319 autopilot Wyner-Ziv reweight + #321 phantom WZ savings from research sidecar + #322 autopilot composition_alpha phantom + #185 META-meta-meta drift

### 5.9 Strict-flip atomicity rule + Live count: 0 discipline

- New gates land warn-only initially; flipped strict in same commit batch that drives live count to 0
- Catalog #185 META-meta-meta refuses CLAUDE.md "Live count: 0" claims when underlying gate returns >0
- Catalog #299 catalog quota brake at #400 forces operator "stop and consolidate" pause

### 5.10 Other meta-engineering

- Premise-verification-before-edit pattern (Catalog #229) — every bulk-edit ≥3 commits requires verdict table or reproducer-script
- Subagent coherence-by-default (CLAUDE.md non-negotiable + Catalog #125 6-hook wire-in declaration in landing memos)
- Operator-frontier-override at all tiers (CLAUDE.md "Mission alignment" Consequence 1) — bypasses quorum + tie-break + recusal with operator-verbatim quote in `council_override_rationale` frontmatter

---

## 6. Research lineage and canonical references

Operator framing: "techniques we started seeing with quantizr's first sub 0.X submission and that have been enhanced and honed and added to since then."

### 6.1 Upstream PR lineage

| PR | Author | Title | Score | What we built on |
|---|---|---|---|---|
| #56 | @szabolcs-cs | Selfcomp (PR56 origin; "SegNet fit using same trick as Quantizr. (Idependent idea)") | sub-0.X first | block-FP + grayscale-LUT + SegMap |
| #95 | @AaronLeslie138 | hnerv_muon — HNeRV substrate first arrival | medal-class | HNeRV decoder; sister sidecar pattern |
| #98 | @EthanYangTW | iterations | medal-class | sister |
| #100 | @BradyMeighan | hnerv_lc_v2 sister sidecar pattern | medal-class | LC schema pattern |
| #101 | @SajayR | hnerv_ft_microcodec GOLD | `0.192845 [contest-CPU]` | Our substrate base; FEC6 + fixed-Huffman bolt-ons stack on this |
| #102 | @EthanYangTW | hnerv_lc_v2_scale095_rplus1 | `0.19538 [contest-CPU]` (silver) | sister |
| #103 | @rem2 | hnerv_lc_ac | `0.195 [contest-CPU]` | sister |
| #107 | apogee | our apogee submission | (historical) | own predecessor |
| #108 closure | @YassineYousfi | competitive-OR-innovative rubric | n/a | Canonical post-deadline gate; cited in our PR #110 body |
| #110 (this) | adpena | hnerv_fec6_fixed_huffman_k16 | `0.192051 [contest-CPU]` (-0.000794 vs PR #101 GOLD) | FEC6 frame-exploit selector + fixed-Huffman k=16 |

### 6.2 Academic lineage by area

- **Steganalysis**: Fridrich (Binghamton DDE Lab) + Yousfi PhD + Filler STC + Kodovsky Rich Models; Holub-Fridrich-Denemark 2014 UNIWARD; Yousfi-Fridrich 2020 OneHot
- **Compression theory**: Shannon R(D) + Cover-Thomas 2006 *Elements of Information Theory* + Ballé 2017 end-to-end + Ballé 2018 scale hyperprior + Ladune 2023 Cool-Chic + Kim 2023 C3
- **Predictive coding + cooperative receivers**: Atick-Redlich 1990 + Atick-Redlich 1992 + Rao-Ballard 1999 + Tishby-Zaslavsky 2015 + Wyner-Ziv 1976 + Hafner 2023 DreamerV3
- **Implicit neural representations**: Sitzmann 2020 SIREN + Chen 2021 NeRV + Chen 2023 HNeRV + Mildenhall 2020 NeRF
- **Multi-scale + wavelets**: Daubechies 1988 + Mallat 1989 + Mallat 2009 *A Wavelet Tour of Signal Processing*
- **Convex optimization + ADMM**: Boyd 2011 ADMM + Boyd-Vandenberghe 2004 *Convex Optimization* + Dykstra 1983 alternating projections
- **Interpretable ML**: Rudin SLIM (Ustun-Rudin 2016) + Wang-Rudin 2015 Falling Rule Lists + Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020 GOSDT
- **Distillation**: Hinton-Vinyals-Dean 2014 KD + Hu et al. 2021 LoRA + Frankle-Carbin 2019 Lottery Ticket + Izmailov 2018 SWA
- **Semantic conditioning**: Park et al. 2019 SPADE + Tan et al. 2021 CLADE
- **Dashcam dataset**: Schafer et al. 2018 comma2k19
- **State-space models**: Dao-Gu 2024 Mamba-2
- **Optical flow + ego-motion**: Teed-Deng 2020 RAFT + Longuet-Higgins 1981 + Gibson 1950
- **Masked reconstruction**: He et al. 2021 MAE
- **VQ-VAE**: van den Oord 2017
- **Information bottleneck**: Tishby-Pereira-Bialek 1999 + Tishby-Zaslavsky 2015

---

## 7. Empirical-vs-design honest accounting

Per operator's framing "I think a lot of our research hasn't been fully iterated on and implmeneted and given experimental rigor" + the inventory memo's Section F "What is stuck":

### 7.1 Empirically validated on contest-1:1 hardware (paired CPU + CUDA)

- `hnerv_fec6_fixed_huffman_k16` (PR #110) — paired
- `lane_a1_*` — paired
- `hnerv_ft_microcodec` (PR #101 replay) — single-axis CPU (bot-recomputed)
- `pr106_format0d_latent_score_table` — single-axis CUDA
- `nscs06` Strip-Everything v7 — single-axis CUDA (not yet paired CPU)

**Count: ~5 substrates / lanes with empirical contest-axis anchors.**

### 7.2 Empirically validated as IMPLEMENTATION-falsified (paradigm intact)

- `c6_ibps` — 24-dim IB collapse, paradigm intact, reactivation = latent-dim sweep
- `apogee_int4` PTQ — naive PTQ falsified, paradigm intact, reactivation = QAT/LSQ/per-channel
- `nscs06` v6 — implementation-level (7 cargo-cults), unwound to v7
- `wunderkind_g1` v2 reducer / `atw_v2_d4` cooperative-receiver / `z6` FiLM ego-motion / `nscs01_nullspace_split` / `nscs06_v8_path_b` — 5 dispatch failures at lifted-trainer form per Catalog #315 anchor; ALL marked DEFER not KILL per CLAUDE.md "Forbidden premature KILL without research exhaustion"

**Count: ~10 substrates with implementation-falsified-paradigm-intact verdict.**

### 7.3 Scaffolded L1/L2 without empirical contest-axis anchor

- Most of the 52 substrate packages (NeRV-family expansion / Z6/Z7/Z8 / cooperative-receiver / IB / DP1 / pose-axis / non-NeRV / composition / higher-order optimization)
- 658 Level-1 lanes + 90 Level-2 lanes total in registry

**Count: ~50+ substrates SCAFFOLDED but not empirically validated end-to-end.**

### 7.4 Design-only / research_only

- TT5L V2 / FF foveation / SAR / DreamerV3 RSSM bridge / Z7-LSTM / Z7-Mamba-2 / Z8 hierarchical / DP1+PR101 / E4 MDL-IBPS / Tishby IB-pure / Cool-Chic / C3 / Quantizr-faithful / Diffusion renderer / Riemannian-Newton / Tropical d_seg
- 195 lanes marked `research_only` in registry
- 122 design memos in `.omx/research/*_design_*.md`

**Count: ~25 paradigms in DESIGN-ONLY state.**

### 7.5 What would bridge the gap

- ~$15-$300 per honest substrate attempt (Section D inventory)
- Per-substrate optimal-form symposium per Catalog #315/#325 BEFORE paid dispatch (14-day recency)
- Cargo-cult-unwind iteration to PROCEED-unconditional verdict (NSCS06 v6→v7 anchor: 44% improvement in ONE iteration)
- Score-axis surrogate distillation (per inventory F.3) to amortize GPU cost
- Paired CPU + CUDA hardware anchors per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"

The honest summary: the apparatus is large (52 substrates + 47 cathedral consumers + 235 catalog gates + 11 canonical equations + 698 tools) and the discipline is strict (8-axis bug-class extinction + 4-tier council + cargo-cult-unwind methodology + canonical Provenance + canonical equations registry). The empirical validation cost has not been paid across most of the substrate canvas. The submission packet (~1140 LOC) is the validated tip of a ~3-orders-of-magnitude-larger iceberg. Most of the iceberg is scaffolded infrastructure that has not yet been run end-to-end.

---

## Cross-links

- Public overview: `docs/comma_lab_overview.md` (the canonical tight Yousfi/Hotz-voice public introduction)
- Asymptotic-floor candidate inventory: `docs/asymptotic_floor_candidate_inventory.md` (v2 amendment-required signal-density at 4372 words)
- Standout candidates spotlight: `docs/standout_undersold_candidates_spotlight.md` (MG-8 prose-format 10 candidates × 120-180 words; predates the operator's tighter-voice redirect)
- Companion methodology + tooling tours: `docs/cargo_cult_unwind_methodology.md` + `docs/strict_preflight_catalog_summary.md` + `docs/canonical_equations_tour.md` + `docs/master_gradient_extractor_tour.md`
- PR #110 anchor: https://github.com/commaai/comma_video_compression_challenge/pull/110

---

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map = N/A (this is a documentation-positioning artifact, not algorithmic signal)
- hook #2 Pareto constraint = N/A
- hook #3 bit-allocator = N/A
- hook #4 cathedral autopilot dispatch = N/A (internal accounting; autopilot does not consume this artifact)
- hook #5 continual-learning posterior = N/A (informational; no posterior contribution)
- hook #6 probe-disambiguator = N/A (no disambiguator role)
