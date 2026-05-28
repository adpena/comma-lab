---
council_tier: T1
council_attendees: []
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Operator critique 2026-05-28 verbatim 'some of what you have the subagents working on may exist in part or in full already, including some of the rate attack stuff' empirically falsifies the main-thread default of spawning N subagents without comprehensive existing-work PV through canonical Catalog #376 verify_head_state_before_spawn helper."
    classification: HARD-EARNED
    rationale: "Empirical receipts: today alone STAND_DOWNs landed for cascade_a_fec10 (Cascade A FEC10 ALREADY implemented + dispatched V14 paired + canonical equation REGISTERED), z4_atick_redlich_cooperative_receiver (sister z4_atick_redlich_substrate_scaffold owns identical scope), operator_override_review_paper (sister rudin_daubechies owns review-memo content). The bug class IS operationally real; not a hypothetical."
  - assumption: "Operator standing directive 'Implement all ~42 substrate _full_main MLX-first at $0 with dispatch_enabled:false' empirically overstates ~42 by 2-3x. Actual scaffold-only count is 11 (raises NotImplementedError) + ~5 MLX/L2 variants without main()."
    classification: HARD-EARNED-via-empirical-AST-scan
    rationale: "AST-scan of all 100 train_substrate_*.py shows 78 have _full_main IMPLEMENTED; only 11 raise NotImplementedError; 11 have NO _full_main but those 11 all have main() (different architecture, not scaffold). The '~42' figure is operator's intuition from prior session; empirical count is dramatically smaller."
  - assumption: "Main-thread Agent-tool spawn does NOT route through Catalog #376 verify_head_state_before_spawn helper. The canonical helper exists + is unit-tested but is NOT structurally invoked at spawn-time in main-thread context."
    classification: HARD-EARNED-via-architectural-PV
    rationale: "Catalog #376 is the SUBAGENT-spawn discipline gate (checks subagent's OWN first-checkpoint PV evidence post-spawn) NOT the main-thread spawn-decision gate (checks parent's PV BEFORE spawn). Today's STAND_DOWNs prove main-thread spawn-decision lacks structural enforcement. The gap is empirical."
  - assumption: "Approximately 75 cathedral consumers already exist; auto-discovery via Catalog #335 is operational. Approximately 85 canonical equations + 23 anti-patterns are already registered. The MAIN bottleneck is NOT 'build more apparatus' but 'route work through existing apparatus FIRST'."
    classification: HARD-EARNED-via-empirical-inventory
    rationale: "Verified empirically via .venv/bin/python query of canonical_equations registry (85) + canonical_anti_patterns registry (23) + ls src/tac/cathedral_consumers (75 non-trivial packages). The apparatus is mature; the gap is its NON-INVOCATION at spawn-decision time."
council_decisions_recorded:
  - "op-routable #1 PRIMARY: register canonical anti-pattern main_thread_subagent_spawn_without_catalog_376_verify_head_state_before_spawn_pv_v1 per Catalog #344 — operator-routed (not auto-landed because anti_pattern registry is currently in-flight per sister wave_n6_triple_pairedcuda_20260528 + wave_n12 last touched 4h ago + cap=1-per-turn applies)"
  - "op-routable #2 STRUCTURAL: propose NEW STRICT preflight gate check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state per CLAUDE.md 'Bugs must be permanently fixed AND self-protected against' canonical 2-landing pattern — DEFER to operator because new Catalog # would require concurrent canonical helper landing that DOES the spawn-decision-time PV (Catalog #376 cannot self-cover the parent-side surface)"
  - "op-routable #3 SISTER COORDINATION: emit DEFER probe outcome for in-flight z5_substrate_scaffold (PRIMARY canonical anchor LANDED via sister already; sister checkpoint shows step 5 author-landing-memo NOW); DEFER for slot_z5_rao_ballard_mlx_local_scaffold (DUPLICATE per Variant 1)"
  - "op-routable #4 SISTER COORDINATION: emit DEFER for z4_cooperative_receiver_atick_redlich (predecessor of z4_atick_redlich_cooperative_receiver_substrate_first_anchor which already STOOD DOWN); PROCEED for z4_atick_redlich_substrate_scaffold (canonical owner, 60% budget remaining per prompt)"
  - "op-routable #5 SISTER COORDINATION: emit DEFER for cascade_a_fec10_second_order_markov_landing_respawn (Cascade A FEC10 already implemented per V14 paired Modal + canonical equation cascade_a_fec10_hybrid_adaptive_blend_savings_v1 registered + STAND_DOWN memo already drafted at .omx/research/cascade_a_fec10_second_order_markov_stand_down_audit_20260528.md)"
  - "op-routable #6 META-PATTERN-ANCHOR: register canonical equation main_thread_spawn_pv_gap_pre_catalog_376_extension_v1 per Catalog #344 — operator-routed; this audit IS empirical anchor 1 (today's STAND_DOWN count = 4: cascade_a_fec10 + z4_atick + operator_override_review_paper + N+5 prior wave); empirical anchor 2 = predecessor a31b7f53 spawn pattern; empirical anchor 3 = future STAND_DOWN within 14 days would trigger Catalog #371 auto-recalibration"
  - "op-routable #7 RETROACTIVE SWEEP per Catalog #348: when (op-routable #2) NEW STRICT gate is operator-approved, emit retroactive_sweep_for_catalog_<N>_<utc>.md citing the 4-field contract (bug-class symptom signature / pre-fix window / historical-STAND-DOWN search results / per-finding RE-EVAL-priority assignment)"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - z4_stand_down_sister_coherence_audit_20260528T220707Z
  - cascade_a_fec10_second_order_markov_stand_down_audit_20260528
  - operator_override_review_paper_STAND_DOWN_per_sister_convergence_20260528
related_lanes:
  - lane_operator_critique_existing_work_audit_20260528
  - lane_z4_atick_redlich_cooperative_receiver_substrate_scaffold_first_600pair_mlx_local_anchor_20260528
  - lane_cascade_a_fec10_second_order_markov_landing
canonical_pattern: ECOSYSTEM_PV_AUDIT_PHASE_1_thru_5_OPERATOR_CRITIQUE_RESPONSE
session_id: audit_existing_work_critique_20260528
schema_version: council_deliberation_v1_20260516
---

# Operator Critique Existing-Work Audit — 5-Ecosystem PV Response

## Operator critique (verbatim, 2026-05-28)

> "some of what you have the subagents working on may exist in part or in full already, including some of the rate attack stuff"

## Top-line verdict

**OPERATOR CRITIQUE EMPIRICALLY RATIFIED.** Comprehensive existing-work PV across 5 ecosystems confirms substantial duplicate / partial-overlap subagent work TODAY. The META-pattern is **main-thread agent-spawn-decision lacking structural Catalog #376 verify_head_state_before_spawn invocation**. Catalog #376 covers the SUBAGENT-side first-checkpoint PV evidence; the PARENT-side spawn-decision PV remains UNSTRUCTURAL. This audit memo IS the operational follow-up; canonical apparatus mutations are operator-routable (op-routable #1-#7) rather than auto-landed because of (a) in-flight sister activity on anti-pattern + canonical-equation registries and (b) the cap=1-per-turn throttle directive.

## Phase 1: RATE-ATTACK ECOSYSTEM survey

### Existing inventory (substantial)

**Codec primitives** (`src/tac/codec/`):
- `a6_selfcomp_blockfp_hyperprior_compose.py` (30.9KB)
- `charm_range_coder.py` (20.4KB) — range coder
- `cooperative_receiver/` subdir
- `cost_curves.py`
- `dual_layer_stc_av1_codec.py`
- `factorized_hnerv_codec.py` (37.5KB)
- `frame_conditional_bit_budget.py` (40.4KB)
- `frame_conditional.py`
- `jscc/` subdir
- `per_cluster_codec_sharing.py`
- `per_tensor_codecs.py`
- `pose_filler_stc_codec.py` (24.2KB)
- `pr101_polymorphic.py` (45.2KB) — substantial
- `rel_err.py`
- `syndrome_trellis_codec.py` (16.6KB) — STC
- `wyner_ziv_layer.py` (44.5KB) — Wyner-Ziv canonical

**NOTE**: `src/tac/codec/fec_family/` directory **DOES NOT EXIST**. FEC primitives are scattered across `tools/build_fec*.py` (NOT consolidated as a Python package). This is a structural gap, but FEC functionality IS implemented at the tool surface.

**Builder tools** (selected highlights from `tools/`):
- `tools/build_fec6_plus_format0d_extra_packet.py` (17.4KB) — FEC6 stacking variant
- `tools/build_fec6_plus_haar_residual_packet.py` (17.0KB) — FEC6 stacking variant
- `tools/build_feca_selector_reparameterized_candidate.py` (4.1KB)
- `tools/build_pr101_frame_exploit_selector_packet.py` (95.5KB) — **substantial frame-exploit selector**
- `tools/build_pr101_fec6_packetir_candidate_queue.py` (5.2KB)
- `tools/build_pr101_nonlocal_sweep_packets.py` (38.1KB) — non-local sweep canonical
- `tools/pilot_difficulty_weighted_fec6_smoke.py` (7.1KB) — FEC6 smoke
- `tools/pr101_fec6_sub0192_cpu_component_selector.py` (28.0KB) — FEC6 CPU selector
- `tools/pr101_fec6_wrapper_profile.py` (13.5KB)
- `tools/profile_pr101_fec6_escape_routes.py` (21.0KB)
- `tools/profile_pr101_fec6_source_payload_anatomy.py` (2.7KB)

**Markov / arithmetic / Cascade**:
- `tools/append_fec8_markov_2nd_order_anchors.py` (10.8KB)
- `tools/measure_fec8_markov_2nd_order_p19_bucket_extension.py` (23.6KB)
- `tools/register_cascade_a_fec10_hybrid_adaptive_blend_canonical_equation_20260526.py` (8.7KB)
- `tools/pr101_markov_transition_table_cost.py` (10.2KB)
- `tools/pr101_markov1_aac_codec.py` (15.8KB)
- `tools/pr101_adaptive_arithmetic_coding.py` (11.7KB)
- `tools/probe_pr103_arithmetic_retarget.py` (9.2KB)
- `tools/materialize_pr103_arithmetic_histogram_candidate.py` (4.3KB)
- `tools/plan_pr103_arithmetic_transform.py` (2.9KB)
- `tools/audit_arithmetic_qint_optimality.py` (2.3KB)

**DQS1**:
- `tools/build_dqs1_local_first_queue.py` (28.2KB)
- `tools/build_dqs1_local_first_harvest_observations.py` (5.0KB)
- `experiments/results/dqs1_drop_many_build_1_pairwise_interaction_matrix_population_20260525/` — landed
- `experiments/results/dqs1_drop_many_build_1c_greedy_heuristic_alternative_reducer_20260525/` — landed

**Cascade families**:
- `experiments/results/cascade_a_fec10_*` — **MULTIPLE LANDED** (V14 paired CPU + CUDA)
- `experiments/results/cascade_b_catalyst_*` — multiple landed
- `experiments/results/cascade_c_prime_*` — multiple landed
- `experiments/results/cross_family_candidate_portfolio_*` — multiple landed
- `experiments/results/fec8_rate_packet_budget_bridge_*` — landed

### Canonical equation coverage (rate-attack subset, 21 of 85 total)

```
- brotli_cascade_bounded_per_stream_v1
- master_gradient_locality_violation_by_codec_v1
- ema_decay_substrate_stage_aware_v1
- cpu_axis_optimal_archive_selector_v1
- categorical_blahut_arimoto_rate_distortion_v1
- cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1
- pr101_vs_fec6_byte_leverage_distribution_v1
- hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1
- cross_codec_super_additive_orthogonality_predictor_v1
- triple_substrate_composition_alpha_v1
- scorer_conditional_joint_rate_distortion_floor_v1
- hnerv_class_substrate_geometry_saturation_v1
- foveation_sidecar_bolt_on_rate_hurdle_v1
- markov_context_selector_stream_compression_savings_v1
- cascade_a_fec10_hybrid_adaptive_blend_savings_v1     <-- ALREADY REGISTERED
- fec8_1st_order_markov_static_variant_b_savings_v1
- fec8_2nd_order_true_markov_variant_a_savings_v1
- wyner_ziv_pipeline_stage_codec_decoder_side_canonical_y_savings_v1
- triple_substrate_composition_orthogonal_pose_axis_savings_v1
- wyner_ziv_cross_substrate_composition_y_pose_axis_savings_v1
- api_rate_limit_burst_envelope_predicts_simultaneous_spawn_crash_v1
```

### Verdict for rate-attack ecosystem

**OPERATOR CRITIQUE EMPIRICALLY RATIFIED for rate-attack**. The Cascade A FEC10 + FEC8 Markov 1st/2nd-order + FEC6 selector ecosystem is **substantially implemented + canonical-equation-registered**. The in-flight subagent `cascade_a_fec10_second_order_markov_landing_respawn_20260528` (step 1) is DUPLICATE WORK per Variant 1 STAND_DOWN — already correctly recognized + STAND_DOWN audit memo drafted at `.omx/research/cascade_a_fec10_second_order_markov_stand_down_audit_20260528.md` (30.3KB landed already).

**Operator-routable**: confirm STAND_DOWN-audit-resume sister completion + mark `cascade_a_fec10_second_order_markov_landing_respawn_20260528` complete via canonical posterior.

## Phase 2: CLASS-SHIFT SUBSTRATE ECOSYSTEM survey

### Substrate trainer inventory (100 trainers across `experiments/train_substrate_*.py`)

**Implementation status** (AST-scan of `_full_main` function bodies):

| Status                                    | Count | Notes                                                       |
|-------------------------------------------|-------|-------------------------------------------------------------|
| `_full_main` IMPLEMENTED                  | 78    | Production paid-dispatch-ready                              |
| `_full_main` raises `NotImplementedError` | 11    | Scaffold-only per Catalog #240 research_only opt-out      |
| NO `_full_main` (uses `main()` directly)  | 11    | Different architecture; production-ready (NOT scaffold)     |
| **Total**                                 | **100** | (~89% production-ready; only 11 genuine scaffolds)      |

**The 11 genuine scaffolds** (raise NotImplementedError):
1. `train_substrate_atw_v2_1.py`
2. `train_substrate_c1_world_model_foveation.py`
3. `train_substrate_e_nerv.py`
4. `train_substrate_ego_nerv.py`
5. `train_substrate_nervdc.py`
6. `train_substrate_nirvana_cascading_nerv_mlx.py`
7. `train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py`
8. `train_substrate_pr101_with_dp1_prior_regularizer.py`
9. `train_substrate_time_traveler_l5_tt5l_v2.py`
10. `train_substrate_tishby_ib_pure.py`
11. `train_substrate_uniward_per_instance_multi_scale_wavelet_segnet.py`

### IMPORTANT EMPIRICAL CORRECTION OF OPERATOR'S SAVED STANDING DIRECTIVE

**Operator's saved standing directive** (memory entry):
> "Implement all ~42 substrate `_full_main` MLX-first at $0 with `dispatch_enabled:false` gate preserved (Catalog #325)."

**Empirical reality**: only **11** trainers have unimplemented `_full_main` (raising NotImplementedError). The "~42" figure is approximately **3.8x overstated** relative to disk state.

This is NOT a falsification of the directive's intent (MLX-first + dispatch_enabled:false gate preserved) — it's a correction of the implementation-target count. Future MLX-first scaffold waves should target the **11 actually-unimplemented trainers**, not "all 42".

Sister consideration: the 11 trainers WITHOUT `_full_main` (boost_nerv_pr110_residual_mlx_l2, cascade_c_prime_frame_1_segnet_waterfill, mdl_ibps_j_discrete_categorical_mine_hybrid_mlx_l2, nirvana_cascading_nerv_mlx_l2, nscs06_v8_chroma_lut_mlx_l2, s2sbs_byte_stuffing, stack_of_stacks, z6_predictive_coding_mlx, z6_predictive_coding_mlx_l2, z7_mamba2_v2_fresh_substrate_mlx_l2, z7_mamba2_v2_mlx) all have `main()` and many are MLX-variants of existing PyTorch trainers. These are NOT scaffolds; they are mature MLX-architecture variants.

### Class-shift substrate dir inventory (60+ substrate packages)

Substrate packages exist for: a1, a1_plus_lapose, a1_plus_wavelet_residual, atw_codec_v1, atw_codec_v2, atw_v2_cooperative_receiver_v2, balle_renderer, block_nerv, boost_nerv, boost_nerv_pr110_residual, c1_world_model_foveation, c6_e4_mdl_ibps, cascade_c_prime_frame_1_segnet_waterfill, coin_plus_plus, coin_pp_implicit_neural_representation, cool_chic, coord_mlp_residual_sidecar, d1_segnet_margin_polytope, d4_wyner_ziv_frame_0, dreamer_v3_rssm, driving_prior_world_model, ds_nerv, faiss_ivf_pq_residual, ff_nerv, grayscale_lut, hi_nerv, hinton_distilled_scorer_surrogate, hybrid_renderer_residual, mdl_ibps_j_discrete_categorical_mine_hybrid, nirvana, nirvana_cascading_nerv, nscs01_nullspace_split_renderer, nscs02_downsampled_renderer, nscs03_end_to_end_balle_joint_codec, nscs06_carmack_hotz_strip_everything, nscs06_v8_chroma_lut, nscs06_v8_path_b_wavelet, pact_nerv_asymmetric_boundary, pact_nerv_bayesian, pact_nerv_cross_codec_a, pact_nerv_cross_codec_b, pact_nerv_diffusion_distilled, pact_nerv_diffusion_trajectory, pact_nerv_distilled_scorer, pact_nerv_ia3, pact_nerv_ia3_multi, pact_nerv_mamba, pact_nerv_moe, pact_nerv_multi_modal, pact_nerv_neural_codec_e2e, pact_nerv_neural_codec_e2e_cross, pact_nerv_selector_v2/v3/v4, pact_nerv_vq, pr101_lc_v2_clone, pr95_lora_dora, **time_traveler_l5_z4 (NEW today)**, time_traveler_l5_z5 (NEW today), time_traveler_l5_z6, time_traveler_l5_z7_lstm_predictive_coding, time_traveler_l5_z7_mamba2, **z4_cooperative_receiver_loss (EXISTING)**, z5_predictive_coding_world_model, z6_v2_cargo_cult_unwind, z7_mamba2_v2_fresh_substrate, z8_hierarchical_predictive_coding.

### Z4 substrate overlap (CRITICAL DUPLICATE-DETECTION ANCHOR)

**Pre-existing**: `src/tac/substrates/z4_cooperative_receiver_loss/`
- Files: `__init__.py` (10.2KB) / `architecture.py` (12.2KB) / `archive.py` (18.4KB) / `inflate.py` (6.3KB) / `score_aware_loss.py` (8.5KB) / `tests/` — FULL implementation dated 2026-05-14 → 2026-05-19.

**NEW today**: `src/tac/substrates/time_traveler_l5_z4/`
- Files: `__init__.py` (5.7KB; created 17:03 today) / `architecture.py` (11.1KB; created 17:04 today) / `tests/` empty — PARTIAL scaffold.

**Verdict**: COMPLEMENTARY-EXTEND (NOT duplicate). New `time_traveler_l5_z4/` is INTENTIONAL Z4 v2 rewrite per (a) 8th MLX-FIRST standing directive 2026-05-28 + (b) 11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27 + (c) Catalog #311 spatial Atick-Redlich form admissible without ego-motion conditioning. The new path is `time_traveler_l5_z4/` (semantically a SPATIAL retinal MI form per Atick-Redlich 1990) while existing `z4_cooperative_receiver_loss/` is the original generic cooperative-receiver loss term form.

**Within Z4-v2 itself**: the two in-flight Z4 subagents (`z4_atick_redlich_substrate_scaffold_20260528` step 3 vs `z4_cooperative_receiver_atick_redlich_20260528` step 1) — the step-3 sister IS the canonical owner; the step-1 sister has ALREADY STOOD DOWN per `z4_atick_redlich_cooperative_receiver_substrate_first_anchor_20260528` checkpoint at 22:06.

### Z5 substrate overlap (CRITICAL DUPLICATE-DETECTION ANCHOR)

**Pre-existing**: `src/tac/substrates/z5_predictive_coding_world_model/` (last touched 2026-05-27)

**NEW today**: `src/tac/substrates/time_traveler_l5_z5/` (touched 2026-05-28 17:02)

**In-flight sisters**:
- `z5_substrate_scaffold_20260528` step 5 — author landing memo NOW; PRIMARY canonical anchor LANDED per probe outcome at 22:08:18 (`verdict=PROCEED substrate=time_traveler_l5_z5 eq=z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1`)
- `slot_z5_rao_ballard_mlx_local_scaffold_20260528` step 3 — earlier checkpoint but DUPLICATE per Variant 1

**Verdict**: PRIMARY canonical anchor ALREADY LANDED. `slot_z5_rao_ballard_mlx_local_scaffold` is DUPLICATE per Variant 1 STAND_DOWN.

### Verdict for class-shift substrate ecosystem

**OPERATOR CRITIQUE PARTIALLY RATIFIED for substrate**. Most class-shift substrate scaffolds (Z4, Z5, Z6, Z6-v2, Z7-Mamba-2, Z8, Compound C, NSCS06 v8) ALREADY EXIST. The "~42 trainers to implement" figure is overstated 3.8x. In-flight Z4 + Z5 sisters have ALREADY-LANDED canonical anchors + STAND_DOWN recognition.

## Phase 3: CATHEDRAL CONSUMER auto-discovery state

### Inventory: **75 production cathedral consumer packages** under `src/tac/cathedral_consumers/`

(79 entries minus `__init__.py`, `__pycache__`, `_example_consumer`, `README.md` = 75 production packages)

Categories:
- **Master-gradient consumers** (Catalog #354 exploit bundle): per_byte_sensitivity / top_k_byte_sensitivity / bottom_k_free_entropy_byte / per_segnet_class_chroma / substrate_fit_diagnostic / information_theoretic_floor / bit_level_score_critical_bits / per_pair_gradient_clustering / master_gradient_aggregate / master_gradient_per_pair / cross_substrate_master_gradient_analyzer / score_weighted_reconstruction_error / mlx_per_pair_master_gradient / multi_granularity_comparison
- **Pareto / Lagrangian / Findings consumers**: dykstra_pareto_solver / findings_lagrangian / per_pair_lagrangian_lambda_bisection / per_pair_kkt_residuals / pareto_polytope_unified_solver / score_lagrangian / unified_action / per_pair_pareto_envelope / per_pair_volterra_cross_terms / meta_resurrection_audit_v2
- **Submission pipeline consumers**: submission_bundle_builder / submission_linter / submission_compliance / pr_submission_compliance / paired_auth_eval / submission_compliance_consumer
- **Auto-discovery + canonical-equation consumers**: canonical_equation_lookup / anti_pattern_lookup / framework_agnostic_lookup / auto_trigger_similarity_after_master_gradient_anchor / atom / risk_adjusted_ranking
- **MPS / framework consumers**: mps_diagnostic / mps_gap_experiment / mps_viable_prescreen / framework_agnostic_lookup
- **Substrate-specific consumers**: ema_decay_formula / cpu_axis_optimal / cross_codec_orthogonality_predictor / cross_substrate_similarity / pact_nerv_ultimate_composition_selector / null_byte_codebook_candidate / packetir_candidate_queue / procedural_codebook_generator / procedural_codebook_savings / engineered_correction_targeting / experimental_extinctions / formula_extinctions / analytical_solve_extinctions
- **Operator / scoring / observability consumers**: contest_oracle / contest_exploits / archive_grammar_builder / utility_curves / early_stopping / uncertainty_weighted_loss / streaming_prediction / per_frame_sensitivity / xray_cuda_score_input_hardening / per_pair_lora_supervision_signal / hf_jobs_dispatcher / venn_risk_composition / wr01_static_packet_custody / per_pair_difficulty_atlas / domain_prior / distilled_scorer_surrogate_canonical_equation / per_pair_coding_budget_allocation / tt5l_sideinfo / uniward_invariant_enumerator / solvers / bit_allocator_per_pair / compression_pipeline_readiness / gradient_informed_decoder_pruning

### Verdict for cathedral consumer ecosystem

**OPERATOR CRITIQUE PARTIALLY RATIFIED for consumers**. 75 production consumers already exist; the auto-discovery infrastructure per Catalog #335 is operational. The MAIN gap is NOT "build more consumers" but "ensure new substrate / equation work AUTO-PROPAGATES through existing consumers via Catalog #335". The Wave N+12 landing memo "NO NEW cathedral consumer file needed" pattern (auto-discoverable canonical equation/anti-pattern lookup consumers absorb new artifacts) is the canonical default.

## Phase 4: CANONICAL EQUATION + ANTI-PATTERN REGISTRY survey

### Canonical equations: **85 registered** (sample IDs already enumerated in Phase 1)

Coverage:
- Rate-attack: 21
- Class-shift / substrate / MLX: 21
- Pareto / Lagrangian / Dykstra: 6
- Cathedral consumer / autopilot: 8
- Validation / observability / discipline: 14
- Composition / cross-substrate: 8
- Domain-specific (foveation / pose / SegNet): 7

### Canonical anti-patterns: **23 registered**

```
- brotli_plus_lzma_chained_anti_pattern_v1
- cross_paradigm_test_without_per_axis_decomposition_v1
- docstring_overstatement_without_evidence_tag_v1
- fp4_packed_without_qat_cos_collapse_v1
- lzma_on_already_brotli_saturated_compounding_v1
- mamba_state_space_training_nan_at_specific_epoch_without_grad_clip_v1
- mlx_trainer_pytorch_sister_duplicated_implementation_v1
- modal_dispatch_local_projector_vs_worker_extraction_root_divergence_v1
- operator_shared_js_rendered_spa_url_inaccessible_to_webfetch_v1
- phantom_score_directory_naming_lie_v1
- predecessor_working_tree_uncommitted_handoff_v1
- predicted_band_from_random_init_tier_c_v1
- quantize_then_svd_corrupted_low_rank_v1
- rank_1_problem_spec_synergy_tautology_v1
- silent_no_spawn_modal_dispatch_v1
- simultaneous_multi_subagent_spawn_rate_limit_cascade_v1
- source_selector_inherited_predicted_score_mean_v1
- subagent_spawn_without_head_state_premise_verification_v1         <-- Catalog #376 sister anti-pattern
- substrate_trainer_uses_pytorch_default_without_mlx_first_consideration_v1
- transient_tmp_path_in_persisted_artifact_v1
- wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface_v1
- wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1
- wyner_ziv_y_derivable_from_x_at_byte_level_structural_ceiling_v1
```

### CRITICAL GAP IDENTIFIED

The existing `subagent_spawn_without_head_state_premise_verification_v1` covers the **SUBAGENT-side first-checkpoint PV evidence** (Catalog #376 STRICT gate on `.omx/state/subagent_progress.jsonl` rows). It does NOT cover the **PARENT-MAIN-THREAD spawn-decision time PV** before the Agent-tool call returns.

**This is the gap operator's critique surfaced**. The canonical helper `tac.discipline_anti_pattern_guards.verify_head_state_before_spawn` exists + is unit-tested. The gap is **structural enforcement that the main-thread parent-agent INVOKES the helper BEFORE every NEW Agent-tool spawn**.

### Proposed NEW canonical anti-pattern (op-routable #1)

```
anti_pattern_id: main_thread_subagent_spawn_without_catalog_376_verify_head_state_before_spawn_pv_v1
severity: HIGH
canonical_unwind_path: |
  Before every NEW Agent-tool spawn from the main-thread context, the parent
  agent MUST invoke tac.discipline_anti_pattern_guards.verify_head_state_before_spawn(
    declared_scope=[<list of file globs / paths>],
    lookback_minutes=60,
  ) and route on the SpawnGuardVerdict.recommendation:
  - PROCEED: spawn approved
  - DUPLICATE_HEAD_STATE: read recent commit bodies + sister landing memos
    BEFORE spawn; pivot scope to COMPLEMENTARY-EXTEND or DEFER
  - SISTER_IN_FLIGHT: coordinate via Catalog #230 ownership map OR stand
    down per Catalog #229 PV pattern

empirical_falsifications:
  - id: today_4_stand_downs_20260528
    receipt: |
      Today (2026-05-28) alone produced 4 STAND_DOWN audits:
      (1) cascade_a_fec10_second_order_markov_stand_down_audit_20260528 — Cascade A FEC10
          ALREADY implemented + V14 paired Modal dispatched + canonical equation
          cascade_a_fec10_hybrid_adaptive_blend_savings_v1 registered. Predecessor crash +
          respawn re-discovered existing work.
      (2) z4_stand_down_sister_coherence_audit_20260528T220707Z — z4_atick_redlich_cooperative_
          receiver_substrate_first_anchor STAND_DOWN per sister z4_atick_redlich_substrate_
          scaffold_20260528 owning identical scope.
      (3) operator_override_review_paper_STAND_DOWN_per_sister_convergence_20260528 — paper
          review STAND_DOWN per Variant 1 (sister 6.6 min earlier, identical scope, identical
          WebFetch blocker).
      (4) pr111_paired_cuda_ratification_refire_stand_down_disjoint_yield_to_sister_20260528 —
          PR111 sister yield to sister.

      All 4 detected at STAGING via Catalog #340 sister-checkpoint guard (POST-EDIT detection)
      rather than at SPAWN via Catalog #376 (PRE-EDIT prevention). The PARENT-side spawn-
      decision PV is structurally unwired.

  - id: prior_session_z5_dual_sister_anchor
    receipt: |
      slot_z5_rao_ballard_mlx_local_scaffold_20260528 vs z5_substrate_scaffold_20260528 —
      both launched against z5_predictive_coding_world_model variant. PRIMARY z5_substrate_
      scaffold landed empirical anchor (loss 0.362 -> 0.148, 59% reduction, archive sha
      ceb614f6c0d2784f) per probe outcome 22:08:18Z; sister DUPLICATE.

severity_weight: 1.0
last_falsification_utc: 2026-05-28T22:22:43Z
```

### Proposed NEW canonical equation (op-routable #6)

```
equation_id: main_thread_spawn_pv_gap_pre_catalog_376_extension_v1
name: Main-thread agent-spawn-decision PV gap predicts STAND_DOWN rate
one_line_summary: |
  Without structural enforcement that main-thread parent-agent invokes
  Catalog #376 verify_head_state_before_spawn BEFORE every NEW Agent-tool
  spawn, the STAND_DOWN rate scales linearly with the number of in-flight
  sister subagents.
latex_form: |
  \text{P}(\text{STAND\_DOWN}_{\text{new\_spawn}} | N_{\text{in\_flight\_sisters}})
  \geq \frac{N_{\text{in\_flight\_sisters}}}{\text{S}_{\text{total\_substrates}}}
  \text{ when } \neg \text{Catalog\_376\_PV\_invoked\_at\_main\_thread}
domain_of_validity:
  apparatus_state: in_flight_sister_count_geq_2
  spawn_context: main_thread_parent_agent_Agent_tool_call
  pv_check_status: catalog_376_not_invoked_at_main_thread
canonical_producers:
  - tac.discipline_anti_pattern_guards.verify_head_state_before_spawn  # canonical helper
canonical_consumers:
  - <PROPOSED> check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state  # NEW STRICT gate
  - tools/operator_authorize.py  # if main-thread spawn is gated through operator-authorize
empirical_anchors:
  - anchor 1: today's 4 STAND_DOWNs (this audit memo's empirical receipt)
  - anchor 2: prior-session predecessor a31b7f53 spawn pattern (z4)
  - anchor 3 (PENDING): next STAND_DOWN within 14 days would trigger Catalog #371 auto-recalibration
```

## Phase 5: OPERATOR-ROUTABLE consolidation + sister coordination

### In-flight subagent verdicts (per Catalog #335 sister-coherence variant taxonomy)

| Subagent | Step | Last Checkpoint | Verdict | Action |
|---|---|---|---|---|
| `audit_existing_work_critique_20260528` (THIS) | 3 | 22:22 | PROCEED | Author audit memo |
| `z5_substrate_scaffold_20260528` | 5 | 22:17 | PROCEED (LANDED) | Confirm landing memo land |
| `cascade-a-fec10-stand-down-audit-resume-20260528` | 1 | 22:15 | PROCEED | Resume the STAND_DOWN audit complete |
| `cascade_a_fec10_second_order_markov_landing_respawn_20260528` | 1 | 22:12 | DUPLICATE (STAND_DOWN per sister at 22:15) | Mark complete; close lane |
| `z4_atick_redlich_substrate_scaffold_20260528` | 3 | 22:04 | PROCEED (canonical owner) | Continue 60% remaining budget |
| `lane_wave_n6_triple_paired_cuda_ratification_respawn_20260528` | 5 | 21:49 | PROCEED | Poll Modal call_ids |
| `z4_cooperative_receiver_atick_redlich_20260528` | 1 | 20:57 | DUPLICATE (predecessor of z4_atick step-1 STAND_DOWN) | Mark complete |
| `slot_z5_rao_ballard_mlx_local_scaffold_20260528` | 3 | 20:55 | DUPLICATE (sister z5_substrate_scaffold step 5 ahead) | Mark complete; close lane |
| `wave_n6_triple_pairedcuda_20260528` | 4 | 20:48 | PROCEED (DEFER row already emitted) | Close lane after memo |
| `slot_pr111_paired_cuda_refire_20260528` | 1 | 20:32 | PROCEED (canonical owner, disjoint from sisters) | Continue |
| `operator_override_review_paper_rudin_daubechies_20260528` | 1 | 20:28 | PROCEED (canonical owner) | Continue |

### Operator-routable next actions (consolidated)

**OP1 PRIMARY** — Register canonical anti-pattern `main_thread_subagent_spawn_without_catalog_376_verify_head_state_before_spawn_pv_v1` via `tac.canonical_anti_patterns.register_anti_pattern(...)`. Operator-routed because anti_pattern registry is in-flight per sister wave_n6_triple_pairedcuda + wave_n12. Per cap=1-per-turn directive: NOT auto-landed in this audit pass.

**OP2 STRUCTURAL** — Propose NEW STRICT preflight gate `check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state` per canonical 2-landing pattern (CLAUDE.md "Bugs must be permanently fixed AND self-protected against"). The gate would scan parent-agent invocation contexts for evidence of `verify_head_state_before_spawn` calls before Agent-tool spawn-mandate emission. NEW Catalog # would require concurrent canonical helper landing that DOES the spawn-decision-time PV from the main-thread context (Catalog #376 cannot self-cover the parent-side surface). DEFER to operator decision.

**OP3 SISTER COORDINATION (Z5)** — Operator-routable: confirm `z5_substrate_scaffold_20260528` step 5 landing memo complete; mark `slot_z5_rao_ballard_mlx_local_scaffold_20260528` complete as DUPLICATE per Variant 1 STAND_DOWN.

**OP4 SISTER COORDINATION (Z4)** — Operator-routable: continue `z4_atick_redlich_substrate_scaffold_20260528` step 3 to completion (canonical owner); mark `z4_cooperative_receiver_atick_redlich_20260528` complete as DUPLICATE.

**OP5 SISTER COORDINATION (Cascade A FEC10)** — Operator-routable: confirm `cascade-a-fec10-stand-down-audit-resume-20260528` step 1 STAND_DOWN audit memo land; mark `cascade_a_fec10_second_order_markov_landing_respawn_20260528` complete.

**OP6 META-PATTERN-ANCHOR** — Register canonical equation `main_thread_spawn_pv_gap_pre_catalog_376_extension_v1` via `tac.canonical_equations.register_canonical_equation(...)`. Operator-routed; this audit IS empirical anchor 1.

**OP7 RETROACTIVE SWEEP (Catalog #348)** — When OP2 NEW STRICT gate is operator-approved, emit `retroactive_sweep_for_catalog_<N>_<utc>.md` citing the 4-field contract (bug-class symptom signature / pre-fix window / historical-STAND-DOWN search results across `.omx/research/*stand_down*.md` / per-finding RE-EVAL-priority assignment).

**OP8 SIGNAL** — Operator's saved standing directive "Implement all ~42 substrate `_full_main` MLX-first" should be CORRECTED to "Implement the 11 actually-unimplemented `_full_main` substrate scaffolds MLX-first" based on AST-scan empirical reality. The "~42" figure is approximately 3.8x overstated.

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map**: ACTIVE — audit identifies which existing apparatus surfaces are saturated vs which have gaps (apparatus-internal sensitivity)
- **Hook #2 Pareto constraint**: N/A
- **Hook #3 bit-allocator**: N/A
- **Hook #4 cathedral autopilot dispatch**: ACTIVE — sister coordination signals route through canonical posterior anchors; DUPLICATE work prevented at PRE-edit time when OP1-OP7 land
- **Hook #5 continual-learning posterior**: ACTIVE — proposed canonical equation `main_thread_spawn_pv_gap_pre_catalog_376_extension_v1` would feed the canonical posterior with empirical anchors per Catalog #344 + #371 auto-recalibration
- **Hook #6 probe-disambiguator**: ACTIVE — proposed STRICT gate (OP2) would BE the structural disambiguator between main-thread-spawn-with-Catalog-376-PV-invoked vs main-thread-spawn-without-PV

## Cargo-cult audit per assumption (per Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| "Main-thread spawn already covered by Catalog #376" | CARGO-CULTED | Catalog #376 covers SUBAGENT-side first-checkpoint PV evidence (post-spawn), NOT main-thread parent-side spawn-decision PV (pre-spawn). |
| "~42 substrate trainers need MLX-first implementation" | CARGO-CULTED | Empirical AST-scan: only 11 raise NotImplementedError. ~42 is operator intuition, not disk reality. |
| "Sister-checkpoint guard (Catalog #340) at STAGING is sufficient" | CARGO-CULTED | STAGING-time PREVENT catches the bare `git add` but does NOT prevent the wasted token spend + duplicate scaffold work that happens BEFORE staging. |
| "Cathedral consumer auto-discovery (Catalog #335) is enough" | HARD-EARNED | Verified empirically: 75 production consumers + Wave N+12 "NO NEW cathedral consumer file needed" pattern. The auto-discovery genuinely works. |
| "Canonical equation registry (Catalog #344) absorbs new findings" | HARD-EARNED | 85 registered equations + Catalog #371 auto-recalibration trigger operationally enforces continual-learning. |

## Predicted ΔS band

Not applicable. This audit produces apparatus-maintenance signal, not score-lowering signal. Dykstra-feasibility check: not applicable to defensive PV gate landing.

## Observability surface (per Catalog #305)

- **Inspectable per layer**: 5-phase audit explicitly enumerates ecosystem inventory per layer (rate-attack codec / class-shift substrate / cathedral consumer / canonical equation registry / canonical anti-pattern registry).
- **Decomposable per signal**: per-subagent verdict table classifies DUPLICATE vs PROCEED vs STAND_DOWN.
- **Diff-able across runs**: future audit memos can compare ecosystem inventory growth (rate-attack equation count, anti-pattern count, cathedral consumer count, in-flight subagent count).
- **Queryable post-hoc**: canonical posterior anchors (when OP1 + OP6 land) make the META-pattern queryable via `tac.canonical_anti_patterns.query_anti_patterns(...)` + `tac.canonical_equations.query_equations(...)`.
- **Cite-able**: every finding cites a canonical surface (commit sha + canonical equation ID + canonical anti-pattern ID + probe outcome row + sister checkpoint timestamp).
- **Counterfactual-able**: if Catalog #376 PV is invoked at main-thread spawn-decision time, the predicted STAND_DOWN rate drops; this audit + OP1 + OP6 would make that prediction testable.

## Discipline

- Catalog #229 PV: 5-phase ecosystem survey (rate-attack codec / class-shift substrate / cathedral consumer / canonical equation registry / canonical anti-pattern registry)
- Catalog #117/#157/#174/#235/#289 canonical serializer (THIS memo will be staged via canonical serializer with POST-EDIT --expected-content-sha256 by the parent agent)
- Catalog #206 crash-resume (4 checkpoints landed: 22:17 step 1 → 22:21 step 2 → 22:22 step 3 → step complete)
- Catalog #110/#113 APPEND-ONLY (NEW memo + zero mutation of existing forensic artifacts)
- Catalog #131/#138 (no direct writes to fcntl-locked JSONL canonical state; all proposed canonical-state mutations operator-routed per cap=1-per-turn)
- Catalog #287 placeholder-rationale rejection (all rationales in v2 frontmatter are substantive non-placeholder ≥4 chars)
- Catalog #292/#300/#346 council discipline (v2 frontmatter complete; assumption-adversary verdicts surfaced)
- Catalog #294/#296/#303/#305 design-memo discipline (9-dim checklist evidence + cargo-cult audit + observability surface declared)
- Catalog #313 probe outcomes (proposed OP3/OP4/OP5 sister coordination probes operator-routed)
- Catalog #323 canonical Provenance (every empirical receipt cites source path + canonical helper)
- Catalog #335 cathedral consumer auto-discovery (Phase 3 inventory verifies operational)
- Catalog #340 sister-checkpoint guard (PROCEED confirmed at PV before authoring)
- Catalog #341 Tier A canonical-routing markers (this audit's predicted_delta_adjustment=0.0 per defensive validator gate semantics)
- Catalog #344/#371 canonical equations + auto-recalibration (OP6 canonical equation proposal feeds the registry)
- Catalog #348 retroactive sweep (OP7 retroactive sweep memo deferred to operator approval of OP2 NEW STRICT gate)
- Catalog #376 verify_head_state_before_spawn (centerpiece of this audit — the gap THIS memo surfaces is at the PARENT-side spawn-decision time, NOT the SUBAGENT-side first-checkpoint time the existing gate covers)
- Catalog #373 anti-pattern acknowledgment in compound stacking (N/A — this audit does not propose a compound stack)
- CLAUDE.md "Results must become system intelligence": the audit produces 8 operator-routable items rather than retrospective prose
- CLAUDE.md "Subagent coherence-by-default": sister coordination signals enumerated per in-flight subagent
- CLAUDE.md "Forbidden premature KILL": DUPLICATE subagents marked DEFER-pending-sister-completion, NOT killed
- CLAUDE.md "Mandatory crash-resume protocol": 4 checkpoints landed
- MLX-FIRST 8th standing directive: respected (Z5 + Z4 v2 + Z6-v2 sisters all MLX-first per directive)
- "memos-must-be-acted-upon" standing directive: ALL canonical apparatus mutations are operator-routable OP1-OP7 with explicit canonical helper invocation paths; nothing is retrospective prose-only
- Cap=1-per-turn throttle directive: NO canonical-state mutations from this audit pass; all OP1-OP7 are operator-routed; only THIS audit memo is written

## Mission contribution per Catalog #300

`apparatus_maintenance`. The audit closes the operator-critique-pre-spawn-existing-work-PV-gap structurally via OP1-OP7. Indirectly enables `frontier_breaking` work by preventing future wasted-spawn cycles that would otherwise consume token budget that could go to actual substrate work.

## Closing

The operator's critique is empirically ratified. The structural gap is at the parent-main-thread spawn-decision surface, not at the subagent-first-checkpoint surface. The 7 operator-routable items (OP1-OP7) constitute the canonical apparatus mutation to close the gap. Per cap=1-per-turn + sister-in-flight throttle, this audit memo is the only landing this pass; OP1-OP7 are operator-routed for subsequent canonical apparatus landings.

The "~42 substrate trainers" figure should be CORRECTED in the operator's standing directive to "11 actually-unimplemented" per empirical AST-scan reality.
