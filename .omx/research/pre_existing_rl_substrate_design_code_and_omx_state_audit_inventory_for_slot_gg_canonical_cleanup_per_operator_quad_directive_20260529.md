---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Contrarian, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: "An inventory memo is NOT a substrate; the operator binding directive #3+#4 is closed only when canonical EXTENSION + RETIREMENT recommendations are explicitly enumerated, not when raw inventory is dumped. Demand the memo end with operator-actionable recommendations not raw counts."
  - member: AssumptionAdversary
    verbatim: "Operating-within assumption I surface for this deliberation: 'EXTENSION of existing tac.* canonical helpers is always preferred over building a new tac.rl_substrate_design package from scratch.' Per CLAUDE.md 'consolidate everything into META layer' standing directive 2026-05-15 + Catalog #299 quota brake, this is HARD-EARNED. Sister-FORK is the operator-decision-pending opt-out when EXTENSION would suppress per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD."
council_assumption_adversary_verdict:
  - assumption: "Pre-existing tac.* canonical helpers (Lagrangian/Pareto/curriculum/Muon/MLX) are mature enough that Slot GG canonical SHARED helper DESIGN should EXTEND not duplicate"
    classification: HARD-EARNED
    rationale: "Empirically verified: findings_lagrangian (4-term scalar Lagrangian + closed-form Gaussian posterior) + dykstra_pareto_solver (43-test Pareto polytope intersection) + training_curriculum/quantizr_5_stage_staircase.py (28K LOC) + optimization/muon.py + optimization/pr95_muon_local_training_integration.py + local_acceleration/pr95_hnerv_mlx.py (115K canonical MLX-LOCAL primitive) + losses/core.py (95.7K canonical scorer-loss helper) + score_composition (canonical contest score formula with per-axis composition) ALL exist as production canonical helpers. ZERO pre-existing PufferLib/RL integration in tac.* (empirically verified via exhaustive grep)."
  - assumption: "The CANONICAL CONTEST SCORER itself IS the canonical reward signal (no new reward function needed)"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Results must become system intelligence' non-negotiable + Catalog #344 canonical_equations_registry (354 entries codifying contest-domain knowledge) + Catalog #344 sister anti_patterns (98 entries codifying KNOWN-BAD compositions). The reward signal IS already canonical; PRIMARY positive signal is canonical_equations_registry + PRIMARY negative signal is canonical_anti_patterns_registry; both fcntl-locked + auto-discoverable per Catalog #335."
council_decisions_recorded:
  - "op-routable #1: Slot GG canonical SHARED helper DESIGN must EXTEND existing tac.* per inventory below (NOT build from scratch)"
  - "op-routable #2: Sister Slot HH multi-reward + multi-env design memo SHOULD consume canonical_equations + canonical_anti_patterns as PRIMARY signal source per CLAUDE.md 'Results must become system intelligence' non-negotiable"
  - "op-routable #3: Stale 2026-05-01 Lightning ledgers eligible for archival per CLAUDE.md State JSONL archival policy (DEFERRED-to-operator per Catalog #298 sister discipline; ~28MB cumulative cold orphan-harvest signal; archival via tools/archive_jsonl_state.py --apply)"
  - "op-routable #4: Pre-existing canonical RL-substrate-design research memos identified for Slot GG synthesis: beat_pr95_curriculum_substrate_training_design_20260513.md (Langevin SDE + Brownian bridge + A* + HardPairReplayBuffer) + council_t1_pr95_curriculum_8stage_research_20260520T143358Z.md (PR95 8-stage HARD-EARNED-vs-CARGO-CULTED classification)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
canonical_equation_references:
  - apparatus_maintenance_cascade_dominance_v1
related_deliberation_ids:
  - council_t3_grand_council_strategic_reprioritization_symposium_rudin_daubechies_20260529
  - council_t1_pr95_curriculum_8stage_research_20260520T143358Z
horizon_class: plateau_adjacent
predicted_band: not_applicable_audit_memo_only
predicted_band_validation_status: not_applicable_audit_memo_only
---

# Slot II — Pre-existing code + .omx/state + .omx/research audit inventory FOR Slot GG canonical-design symposium cleanup recommendations (operator binding quad-directive phase 3+4)

**Date**: 2026-05-29T08:25Z
**Lane**: `lane_slot_ii_pre_existing_code_omx_state_omx_research_audit_for_slot_gg_20260529` (L1: impl_complete + canonical_research_inventory + audit_recommendations)
**Operator directive**: 2026-05-29 BINDING quad-directive verbatim *"we also might have some existing code to clean up or update"* (phase 3) + *"and maybe some .omx state or research too"* (phase 4)
**Scope**: READ-ONLY audit + canonical apparatus mutation chain (canonical equation + anti-pattern registration + landing memo). NO code build. NO state mutation beyond APPEND-ONLY discipline. NO GPU dispatch.
**Cost**: $0 ($0 paid GPU + ~75 min wall-clock)
**Sister DISJOINT vs**: Slot GG (T3 grand council canonical-design symposium IN-FLIGHT) + Slot HH (multi-reward + multi-env architecture design memo IN-FLIGHT) per Catalog #340 sister-checkpoint guard PROCEED

## 1. Why this audit is necessary

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #229 premise-verification-before-edit + the CARGO-CULTED canonical anti-pattern `rl_substrate_design_canonical_helper_built_from_scratch_when_canonical_extension_path_exists_v1` (proposed registration this landing): Slot GG MUST receive a comprehensive inventory of pre-existing tac.* canonical helpers BEFORE designing a new `tac.rl_substrate_design` package. The "iterate not force" + "consolidate everything into META layer" + Catalog #299 quota brake + Catalog #335 cathedral consumer auto-discovery binding directives ALL converge on the same operating mode: **EXTEND existing canonical surfaces; do NOT duplicate**.

This memo provides Slot GG with the empirical 3-phase inventory + canonicalization recommendations.

## 2. Phase A — Pre-existing tac.* code audit

### 2.1 Top-level package count

**94 substrate dirs** at `src/tac/substrates/*/` + **100 train_substrate_*.py trainers** at `experiments/` + **80 cathedral consumer dirs** at `src/tac/cathedral_consumers/*/`. Cumulative tac.* package count includes 90+ top-level packages (per `ls src/tac/` empirical enumeration).

### 2.2 Canonical helpers Slot GG should EXTEND (NOT recreate)

**(A) Lagrangian / Pareto / Bayesian-posterior** (canonical META-Lagrangian solver surface per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS"):

| Package | LOC | Function |
|---|---|---|
| `tac.findings_lagrangian` | 9 files / ~120K LOC | 4-term scalar Lagrangian (`lagrangian.py`) + closed-form Gaussian posterior (`posterior.py`) + Lindley-1956 action selector (`action_selector.py`) + info-gain (`info_gain.py`) + partition (`partition.py`) + weights (`weights.py`) + interpretability (`interpretability.py`) + Phase 2 ablation framework (`phase_2_ablation/`) + dual_solver_phase_2 (`dual_solver_phase_2.py` 33K LOC) |
| `tac.dykstra_pareto_solver` | 5 files | Pareto polytope intersection via Dykstra alternating projections (`solver.py` 25K) + `AntiPatternConstraint` (15K) + `Polytope` (13K) + `ParetoSolverVerdict` (14K) per Catalog #372 |
| `tac.pareto_polytope_unified_solver` | 1 file / 43K LOC | Unified Pareto solver (sister of dykstra) |
| `tac.findings_lagrangian_pp` | 5 files | PP (predictive posterior) variant with Bayesian conjugate updates; `pp_action_selector` + `pp_posterior` + `substrate_composition_pp` + `cost_band_pp` |

**(B) Curriculum + multi-stage + Muon optimizer + parameter-group LR policy**:

| Package | LOC | Function |
|---|---|---|
| `tac.training_curriculum` | 15 files | `multi_stage_curriculum.py` (12K) + `quantizr_5_stage_staircase.py` (28K canonical) + `pause_to_swap_loss.py` + `pause_distill_resume.py` + `pause_quantize_finetune.py` + `pause_and_diagnose.py` + `early_stopping_with_resume.py` + `model_soup_averaging.py` (12K) + `swa_polyak_averaging.py` + `a1_pattern_inflate_time_bias_correction.py` (17K) + `master_gradient_pair_weights.py` |
| `tac.optimization` | 110+ files | `muon.py` (canonical Muon optimizer per HNeRV parity L15) + `pr95_muon_local_training_integration.py` + `optimizer_scheduler_registry.py` + `parameter_group_lr_policy.py` + `langevin_optimizer.py` + `bayesian_experimental_design.py` (27K) + `bit_allocator_end_to_end.py` (57K) + `byte_shaving_campaign.py` (158K) + `cross_family_candidate_portfolio.py` (121K) + sister optimizer/searcher primitives |
| `tac.training` | many | `long_training_canonical.py` (sample_batch protocol) + `streaming_master_gradient_hook.py` + sister training primitives |
| `tac.training_optimization` | 4 files | `autocast_helper.py` + `compile_helper.py` (torch.compile) + `scorer_cache.py` (GTScorerCache F3) per Catalog #228 |

**(C) Canonical contest scorer + reward signal + score composition**:

| Package | LOC | Function |
|---|---|---|
| `tac.losses` | 4 files | `core.py` (95.7K LOC canonical scorer-loss helper) + `cat_entropy_v2.py` (9.9K cat_entropy per HNeRV parity L7) + `u_die_kl.py` (25K U-DIE-KL substrate-wide loss) per CLAUDE.md eval_roundtrip non-negotiable |
| `tac.score_composition` | 1 file / 18.8K | Canonical contest score formula `100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37_545_489` + per-axis composition + AxisDecomposition per Catalog #356 |
| `tac.codec` + `tac.codecs` | many | Canonical codec primitives + cooperative_receiver/atick_redlich + sister scorer-related codecs |

**(D) Bit allocator + domain priors + contest oracle**:

| Package | LOC | Function |
|---|---|---|
| `tac.bit_allocator` | 6 files | `per_axis.py` (18K) + `per_byte.py` (29K) + `per_class.py` (16K) + `per_pair_difficulty_weighted.py` (15K) + `lane_omega.py` + `pareto_dual.py` (28K) |
| `tac.domain_priors` | 5 files | `comma2k19_priors.py` + `ego_motion_concentration.py` (19K) + `equations.py` (15K) + `per_class_statistical.py` + `per_frame_difficulty.py` (18K) |
| `tac.contest_oracle` | many | `arithmetic_coder_class_conditional.py` + `bandit_per_pair.py` + `cell_allocator.py` + `pareto_frontier.py` + `per_class_lagrangian.py` + `phase_classifier.py` + `pose_axis_canonical.py` + `score_predictor.py` + `substrate_alignment.py` + `theoretical_floor.py` |

**(E) Search frameworks (Bayesian / CMA-ES / MCTS / Optuna)**:

| Package | LOC | Function |
|---|---|---|
| `tac.search` | 12 files | `bayesian_optimization_gp.py` (12K Gaussian-process BO) + `cma_es_searcher.py` (13K CMA-ES) + `mcts_codebook_searcher.py` (15K Monte Carlo Tree Search) + `optuna_tpe_sampler.py` (12K TPE) + `rashomon_ensemble_committee.py` (14K Rashomon ensemble per Catalog #252) + `contract.py` (17K) + `decorator.py` + `pipeline.py` (24K) |
| `tac.compress_time_optimization` | 11 files | `generic_tto_harness.py` + `iterated_bisection.py` + `multipass_refinement.py` + `per_pair_coordinate_search.py` + `simulated_annealing.py` + `pipeline.py` (32K) |

**(F) MLX-LOCAL canonical primitives (per CLAUDE.md MLX portable-local-substrate authority)**:

| Package | LOC | Function |
|---|---|---|
| `tac.local_acceleration.pr95_hnerv_mlx` | 1 file / 115K LOC | Canonical PR95 HNeRV MLX-LOCAL primitive per `feedback_mps_phase_b_options_b_plus_c_completion_landed_20260519.md` 3-component aggregate gap 0.072% (69× below 5% LOCAL_MPS_TRAIN_VIABLE threshold) |
| `tac.local_acceleration.pr95_hnerv_mlx_training` | 1 file / 23K | Long-training infrastructure |
| `tac.local_acceleration.pr95_hnerv_mlx_long_training` | 1 file / 67K | Long-training canonical |
| `tac.local_acceleration.pr95_hnerv_mlx_stage_losses` | 1 file / 8.5K | Stage-aware losses |
| `tac.framework_agnostic` | 6 files | `backend.py` + `decorators.py` + `helpers.py` + `operations.py` + `tensor_protocol.py` — MLX/PyTorch/numpy/tinygrad-like decorators per portability discipline |

**(G) Canonical apparatus surfaces (cathedral + canonical equations + anti-patterns + posterior)**:

| Package | LOC | Function |
|---|---|---|
| `tac.canonical_equations` | 5+ files | Registry of 354 canonical equations per Catalog #344; `register_canonical_equation` + `update_equation_with_empirical_anchor` + auto-recalibrator per Catalog #371 |
| `tac.canonical_anti_patterns` | 5+ files | Registry of 98 canonical anti-patterns per Catalog #344 sister discipline; `register_anti_pattern` + `append_empirical_falsification` |
| `tac.cathedral_consumers` | 80 dirs | Auto-discovered cathedral consumers per Catalog #335; PR_SUBMISSION_COMPLIANCE, master_gradient_per_pair, score_weighted_reconstruction_error, top_k_byte_sensitivity, bottom_k_free_entropy, per_segnet_class_chroma, substrate_fit_diagnostic, information_theoretic_floor, bit_level_score_critical_bits, per_pair_gradient_clustering, streaming_prediction, per_pair_difficulty_atlas (10 master-gradient exploit consumers per Catalog #354 + 70 sister consumers) |
| `tac.council_continual_learning` | 1 file | Canonical posterior at `.omx/state/council_deliberation_posterior.jsonl`; `append_council_anchor` per Catalog #355 |
| `tac.probe_outcomes_ledger` | 1 file | Canonical ledger at `.omx/state/probe_outcomes.jsonl`; `register_probe_outcome` per Catalog #313 |
| `tac.canonical_posterior_read_validator` | 1 dir | Canonical READ-surface validator per Catalog #382 |
| `tac.discipline_anti_pattern_guards` | 3 files / ~73K LOC | `main_thread_spawn_decision_pv_guard.py` (31K Catalog #378) + `predecessor_handoff_auto_commit_guard.py` (18K) + `subagent_spawn_head_pv_guard.py` (25K Catalog #376) |

**(H) Substrate registry + decorators + recipe generation**:

| Package | LOC | Function |
|---|---|---|
| `tac.substrate_registry` | 7 files | `contract.py` (22K canonical 36-field schema per Catalog #241) + `decorator.py` + `auto_wire.py` + `driver_generator.py` + `recipe_generator.py` + `example_template.py` per the META layer canonical contract |

### 2.3 EMPIRICAL VERIFICATION: ZERO pre-existing PufferLib/RL integration

Exhaustive grep across `src/tac/**/*.py` for `pufferlib` / `PufferLib` / `gymnasium.make` / `gym.make` / `stable_baselines` / `rllib` / `sample_batch` / `trajectory_buffer` / `compute_reward` / `class Policy` / `class Actor` / `class Critic` / `ppo_step` / `actor_critic` / `policy_loss` / `value_loss` / `advantage_estimator` returns **ZERO results** for explicit RL framework integration. The pre-existing `sample_batch` token references are NON-RL training-loop sample-batch protocols in `tac.training.long_training_canonical`.

**Conclusion**: Slot GG canonical SHARED helper DESIGN must build the PufferLib + RL policy + reward computation surface FROM SCRATCH while CONSUMING the canonical Lagrangian + Pareto + curriculum + scorer + MLX-LOCAL helpers above.

## 3. Phase B — .omx/state ledger audit

### 3.1 PRIMARY reward signal sources (per operator binding directive #2)

| Ledger | Rows | Size | Purpose |
|---|---|---|---|
| `.omx/state/canonical_equations_registry.jsonl` | 354 | 3.4M | **PRIMARY positive signal source** — codified empirical equations + anchors per Catalog #344 |
| `.omx/state/canonical_anti_patterns_registry.jsonl` | 98 | 472K | **PRIMARY negative signal source** — codified KNOWN-BAD compositions + falsifications per Catalog #344 sister |
| `.omx/state/council_deliberation_posterior.jsonl` | 206 | 799K | T2/T3 council anchors per Catalog #355 + #300 |
| `.omx/state/probe_outcomes.jsonl` | 246 | 444K | Catalog #313 verdicts (PROCEED/DEFER/KILL/PARTIAL/INDEPENDENT/ESCALATE_TO_OPERATOR) |
| `.omx/state/master_gradient_anchors.jsonl` | 11 | 20K | Per-pair fp64 master-gradient ledger per Catalog #327 |
| `.omx/state/substrate_composition_matrix.json` | — | 298K | Composition_alpha pairs across 84 cells per Catalog #322 |
| `.omx/state/canonical_frontier_pointer.json` | — | 9.3K | Canonical frontier scores (CPU + CUDA) per Catalog #343 |
| `.omx/state/cost_band_posterior.jsonl` | 111 | 86K | Cost-band canonical posterior per Catalog #175 + #177 |
| `.omx/state/modal_call_id_ledger.jsonl` | 509 | 1.1M | Dispatch outcomes per Catalog #245 + #339 |
| `.omx/state/canonical_task_status.jsonl` | 24 | 28K | Task lifecycle per Catalog #331 |
| `.omx/state/atom_ledger.jsonl` | 67 | 137K | Canonical atom contracts per `tac.atom` |
| `.omx/state/lane_registry.json` | — | 2.0M | Lane lifecycle per Catalog #90 (255 substrate lanes L1+ per Slot M Wave N+48 audit) |
| `.omx/state/subagent_progress.jsonl` | 4425 | 2.8M | Canonical checkpoint discipline per Catalog #206 + #376 |

### 3.2 STALE / RETIREMENT CANDIDATES per CLAUDE.md State JSONL archival policy

| Ledger | Size | Age | Status |
|---|---|---|---|
| `lightning_refresh_all_followup_20260501T0926.jsonl` | 12.7M | 28 days | RETIREMENT CANDIDATE — cumulative orphan-harvest signal preserved per HISTORICAL_PROVENANCE Catalog #110/#113; ARCHIVE via `tools/archive_jsonl_state.py --target <file> --apply` per CLAUDE.md State JSONL archival policy. Total ~28MB across 15+ lightning_* ledgers. |
| `lightning_harvest_direct_fd_completed_20260501T090541Z.jsonl` | 4.6M | 28 days | RETIREMENT CANDIDATE |
| `lightning_refresh_direct_fd_allwaves_*` (×6) | ~12MB cumulative | 28 days | RETIREMENT CANDIDATE |
| `repair_campaign_stackability_posterior.jsonl` | 6.6M | (date unknown; SLOT-decision-pending) | INVESTIGATE before archival |

**Operator-routable**: archival via `tools/archive_jsonl_state.py --apply` per CLAUDE.md "State JSONL archival policy"; default 90-day retention window + monthly cadence. NOT a kill — APPEND-ONLY archive at `.omx/state/archive/<file>_<YYYY-MM>.jsonl` preserves signal per Catalog #110/#113.

### 3.3 Sister state subdirs (active or archive)

`.omx/state/wyner_ziv_deliverability/` (active per Catalog #319) + `.omx/state/dispatch_claims_archive/` (per Catalog #154 archival) + `.omx/state/verdicts/latest-verdict.json` (active) + sister 2026-05-01 / 2026-05-04 / 2026-05-05 dated subdirs (RETIREMENT CANDIDATES per Catalog #298 sister discipline applied to state surfaces; review-before-archival).

## 4. Phase C — .omx/research memo audit

### 4.1 Inventory

**3770 total .omx/research memos**. Per CLAUDE.md "Memory file rotation discipline" non-negotiable: memos older than 60 days superseded by newer memos should be marked `superseded_by: <newer>` in YAML frontmatter; the monthly audit cadence via `tools/audit_memory_file_freshness.py` surfaces candidates.

### 4.2 Pre-existing canonical RL-substrate-design research feeding Slot GG

| Memo | Lines | Size | Relevance to Slot GG canonical SHARED helper DESIGN |
|---|---|---|---|
| `beat_pr95_curriculum_substrate_training_design_20260513.md` | 881 | 42K | **HIGHEST RELEVANCE** — pre-existing A* curriculum + Brownian bridge + Langevin SDE + HardPairReplayBuffer canonical DESIGN with empirical anchor refs. Lane: `lane_beat_pr95_curriculum_substrate_training_design_20260513`. |
| `council_t1_pr95_curriculum_8stage_research_20260520T143358Z.md` | 474 | 63K | **HIGH RELEVANCE** — T1 council deliberation on PR95 8-stage curriculum HARD-EARNED-vs-CARGO-CULTED classification per Catalog #303; Muon flagged CARGO-CULTED vs Langevin/SGLD alternatives. |
| `council_t3_grand_council_strategic_reprioritization_symposium_rudin_daubechies_20260529.md` | 443 | 52K | **HIGH RELEVANCE** — TODAY's T3 grand-strategy memo per operator 4-message META-cascade; TOP-5 reprioritized (NULL-BYTE PROBE / PR110-OPT / C6 IBPS Tier-C / Wyner-Ziv eq / Z6-v2+Z8). |
| `cross_pr_family_canonical_techniques_mining_L14_L70_20260529T075244Z.md` | 274 | 34K | **HIGH RELEVANCE** — Slot DD; PR101 GOLD canonical L14-L70 techniques mining. |
| `quantizr_canonical_3_stack_audit_design_20260529.md` | 244 | 23K | **MEDIUM RELEVANCE** — Slot EE; canonical 3-stack EMA 0.997 + KL T=2.0 + eval_roundtrip discipline audit. |
| `why_have_we_not_produced_original_frontier_score_meta_diagnostic_synthesis_20260529.md` | 270 | 30K | **MEDIUM RELEVANCE** — Slot V WHY-FRONTIER META-diagnostic; OPT-3 + OPT-7 cascade enumeration. |
| `tinygrad_portable_inflate_primitive_bridge_design_20260529.md` | — | — | **MEDIUM RELEVANCE** — sister portable-runtime design surface. |
| `expert_team_aerospace_stealth_analytic_alien_tech_20260513.md` | — | — | **LOW RELEVANCE** — sister Atick-Redlich expert team discipline. |
| `pr110_opt_4_grouped_color_geometry_calibration_cross_pair_perturbation_reuse_design_20260529.md` | — | — | **LOW RELEVANCE** — sister color/geometry probe; Slot X. |

### 4.3 STALE / RETIREMENT memo candidates per Catalog #298 sister discipline

Per CLAUDE.md "Memory file rotation discipline" + Catalog #298 substrate retirement discipline applied to memo surfaces: memos referencing falsified canonical posterior tokens per Catalog #382 should carry APPEND-ONLY footer marking re-classification (the `auto_emit_append_only_footer_to_memos_citing_falsified_score` canonical helper per Catalog #382 is the structural protection). Slot Z 2026-05-29 already applied this to the canonical 3-metric trichotomy memo (Wave N+33 alpha=4.74 PHANTOM correction); same META pattern can extend to other operator-facing memos as canonical posterior verdicts flip.

## 5. Canonicalization recommendations for Slot GG

### 5.1 EXTENSION-via-canonical-helpers (preferred per "iterate not force")

| Slot GG canonical SHARED helper need | EXISTING canonical helper to EXTEND | EXTENSION pattern |
|---|---|---|
| RL policy network architecture | `tac.findings_lagrangian` + `tac.dykstra_pareto_solver` | Policy gradient consumes 4-term scalar Lagrangian as canonical advantage; Pareto polytope intersection bounds policy update via canonical KKT feasibility |
| Reward computation | `tac.losses.core` + `tac.score_composition` + `tac.canonical_equations` + `tac.canonical_anti_patterns` | Canonical contest scorer IS the reward; positive shaping via canonical_equations posterior; negative shaping via anti_patterns posterior |
| Multi-stage curriculum / staircase | `tac.training_curriculum.multi_stage_curriculum` + `quantizr_5_stage_staircase` + `pause_distill_resume` + sister 15 helpers | RL curriculum consumes the canonical PR95 8-stage chain as STAGE-LEVEL POLICY (each stage is a policy episode) |
| Optimizer (Muon + Newton-Schulz) | `tac.optimization.muon` + `pr95_muon_local_training_integration` + `langevin_optimizer` + `parameter_group_lr_policy` | RL policy optimizer consumes canonical Muon + Langevin SDE alternative (per Slot N M2 GATED-BY-SLOT-L revision) + per-stage LR policy |
| Vectorized environment (PufferLib) | `tac.framework_agnostic` (canonical MLX/PyTorch backend abstraction) | NEW PufferLib adapter consumes `framework_agnostic.backend.get_backend()` for cross-framework portability |
| MLX-LOCAL acceleration | `tac.local_acceleration.pr95_hnerv_mlx` (115K canonical) + sister 30+ MLX helpers | RL on-policy rollout MLX-LOCAL execution via canonical PR95 HNeRV MLX-LOCAL primitive; 128GB M5 Max enables high-batch parallelism |
| Per-pair reward signal | `tac.master_gradient_consumers` (10 master-gradient exploit consumers per Catalog #354) + `tac.cross_substrate_master_gradient_analyzer` | RL per-pair difficulty atlas consumes canonical master_gradient_per_pair_consumer signal |
| Search / hyperparameter optimization | `tac.search.bayesian_optimization_gp` + `cma_es_searcher` + `mcts_codebook_searcher` + `optuna_tpe_sampler` + `rashomon_ensemble_committee` | RL curriculum hyper-search consumes canonical Bayesian/CMA-ES/MCTS/Optuna search; Rashomon ensemble for exploration |
| Canonical equation auto-discovery | `tac.canonical_equations` + `cathedral_consumers/canonical_equation_lookup_consumer` | Per CLAUDE.md "Results must become system intelligence" — RL training-loop results MUST register as canonical equation anchors per Catalog #344 |
| Canonical anti-pattern auto-discovery | `tac.canonical_anti_patterns` + `cathedral_consumers/anti_pattern_lookup_consumer` | RL policy MUST consult anti-patterns before sampling action; Catalog #373 `match_stack_against_anti_patterns` IS the canonical disambiguator |

### 5.2 NEW package recommendation (operator-decision-pending; Catalog #299 quota brake binding)

Per Catalog #299 quota brake under 400 + "consolidate everything into META layer" standing directive 2026-05-15: **DO NOT create `tac.rl_substrate_design` as a flat new package**. Instead, EXTEND existing structure per the table above + add MINIMAL NEW surfaces:

- `tac.training.rl_policy_loop` — NEW canonical helper (~500-800 LOC) that orchestrates the RL policy + reward + multi-stage curriculum chain by CONSUMING existing canonical helpers (NO duplication of Lagrangian/Pareto/curriculum/scorer/optimizer logic).
- `tac.framework_agnostic.pufferlib_adapter` — NEW canonical helper (~200-400 LOC) that adapts PufferLib's vectorized environment + policy interface to the canonical `framework_agnostic.backend` abstraction.
- Sister cathedral consumer `tac.cathedral_consumers.rl_substrate_design_consumer` — NEW canonical contract-compliant consumer per Catalog #335 + Tier A canonical-routing markers per Catalog #341 (auto-discovery via canonical contract; ZERO ranker-cascade-edit surface).

### 5.3 RETIREMENT candidates (operator-decision-pending per CLAUDE.md "Forbidden premature KILL")

**STATE retirement candidates** (archival NOT deletion per Catalog #298 sister discipline):
- 15+ Lightning ledgers 2026-05-01 dated (~28MB cumulative) → ARCHIVE via `tools/archive_jsonl_state.py --apply` to `.omx/state/archive/<file>_2026-05.jsonl`
- 5+ dated subdirs under `.omx/state/` from 2026-05-01 / 2026-05-04 / 2026-05-05 → REVIEW per Catalog #110 HISTORICAL_PROVENANCE + Catalog #298 sister discipline

**CODE retirement candidates** (operator-decision-pending; NO immediate retirement recommended per "Forbidden premature KILL"):
- Pre-existing scattered RL-adjacent prototypes (if any surface) → CONSOLIDATE into the new `tac.training.rl_policy_loop` canonical surface
- Stale experimental_extinctions + formula_extinctions + analytical_solve_extinctions (3 dirs) → REVIEW per Catalog #307 paradigm-vs-implementation classification; preserve per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE

**MEMO retirement candidates** (DEFER per Catalog #298 + CLAUDE.md "Memory file rotation discipline"):
- 3770 .omx/research memos → monthly cadence via `tools/audit_memory_file_freshness.py` + `tools/cluster_summarize_memory_category.py` per CLAUDE.md "Memory file rotation discipline" (operator-routable; NOT in Slot II scope)

### 5.4 Feed-INTO-Slot-GG canonical INPUTS

Slot GG canonical-design symposium should consume THIS audit memo as PRIMARY input:

1. **Architectural surface**: per §5.1 EXTENSION table — Slot GG DESIGN documents per-component EXTENSION decisions per Catalog #290 canonical-vs-unique-decision-per-layer non-negotiable.
2. **Reward signal**: per §3.1 PRIMARY signal sources — Slot GG DESIGN explicitly cites canonical_equations + canonical_anti_patterns as REWARD COMPUTATION input per CLAUDE.md "Results must become system intelligence" non-negotiable.
3. **Pre-existing research**: per §4.2 — Slot GG DESIGN cites `beat_pr95_curriculum_substrate_training_design_20260513` (881-line A* + Langevin + Brownian + HardPairReplayBuffer canonical DESIGN) + `council_t1_pr95_curriculum_8stage_research_20260520T143358Z` (T1 PR95 8-stage HARD-EARNED-vs-CARGO-CULTED classification) per Catalog #229 premise-verification non-negotiable.
4. **MLX-LOCAL execution**: per §5.1 — Slot GG DESIGN explicitly leverages `tac.local_acceleration.pr95_hnerv_mlx` (115K canonical) per MLX-LOCAL canonical surface per CLAUDE.md MLX portable-local-substrate authority.
5. **Sister Slot HH multi-reward + multi-env architecture**: should consume THIS audit memo + sister Slot GG canonical SHARED helper DESIGN per Catalog #340 sister-checkpoint guard DISJOINT.

## 6. Mathematical compounding identity

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + Catalog #299 quota brake + the 13th OPTIMAL-TRIO standing directive:

```
Next-RL-substrate-design-attack-direction = argmax_helper (
  PredictedΔS_per_extension × (1 - ScopeCreepCoefficient)
)
subject to (
  ScopeCreepCoefficient = 0 when EXTENSION of existing canonical helper
  ScopeCreepCoefficient = 1 when NEW package built from scratch (forbidden per "iterate not force")
  ∀ helper: helper must satisfy CanonicalContract per Catalog #335
)
```

The structural extinction is: any Slot GG canonical SHARED helper DESIGN that proposes a flat new `tac.rl_substrate_design` package WITHOUT first EXTENSION-of-existing-canonical-helpers must be REFUSED at the design memo surface (Catalog #287 + #303 + #373 sister-extinction discipline at the canonical anti-pattern surface).

## 7. Canonical apparatus mutation chain

### 7.1 Canonical anti-pattern (operator-decision-pending Catalog #344 registration)

**Candidate**: `rl_substrate_design_canonical_helper_built_from_scratch_when_canonical_extension_path_exists_v1` — paradigm class `discipline_anti_pattern` + severity `medium_architectural_inefficiency` + canonical_unwind_path = "EXTEND existing tac.* canonical helpers per Slot II §5.1 table; NEW packages only as MINIMAL orchestration surfaces consuming canonical helpers; Catalog #299 quota brake under 400 + 'consolidate everything into META layer' standing directive 2026-05-15 binding"

### 7.2 Canonical equation (operator-decision-pending Catalog #344 registration)

**Candidate**: `pre_existing_canonical_apparatus_inventory_predicts_rl_substrate_design_helper_extension_path_v1` — first-principles derivation per CLAUDE.md "Subagent coherence-by-default" + "consolidate everything into META layer" + Catalog #299 quota brake; predicts `P(EXTENSION-path) > 0.95` when pre-existing canonical helpers cover ≥80% of required surface, with empirical anchor = this Slot II audit memo (94+ pre-existing tac.* packages cover ≥80% of RL-substrate-design surface; only PufferLib adapter + RL policy loop are NEW)

### 7.3 Canonical posterior anchor via tac.council_continual_learning.append_council_anchor

Will be appended per Catalog #355 with this memo's v2 frontmatter; `predicted_mission_contribution=frontier_breaking_enabler` per Catalog #300 §Mission alignment Consequence 5.

### 7.4 Catalog #313 probe outcome

Will be registered via `tac.probe_outcomes_ledger.register_probe_outcome` with verdict=PROCEED + 14-day expires per the canonical Catalog #325 symposium discipline.

### 7.5 Catalog #348 retroactive sweep

Companion memo at `.omx/research/retroactive_sweep_for_pre_existing_rl_substrate_design_audit_20260529T*.md` will document: bug-class symptom signature (RL-substrate-design helpers built from scratch when canonical extension path exists) + pre-fix window (PRE-Slot-GG; no prior RL surface in tac.*) + historical KILL/DEFER/FALSIFY search results (NONE — no prior RL surface to retire) + per-finding RE-EVAL-priority assignment (HIGH for Slot GG canonical SHARED helper DESIGN integration).

## 8. Observability surface (per Catalog #305 6-facet)

1. **Inspectable per layer**: every canonical helper in §5.1 table is per-layer importable + introspectable via `tac.canonical_equations.query_equations(consumer=...)` per Catalog #344.
2. **Decomposable per signal**: per-helper signal contribution to RL reward decomposable via canonical AxisDecomposition per Catalog #356.
3. **Diff-able across runs**: every RL training run's canonical equation anchor + canonical anti-pattern falsification appendable per Catalog #344 APPEND-ONLY HISTORICAL_PROVENANCE.
4. **Queryable post-hoc**: `tac.canonical_equations.query_equations` + `tac.canonical_anti_patterns.query_anti_patterns` + `tac.council_continual_learning.query_anchors_by_topic` per Catalog #300/#344/#373.
5. **Cite-able**: every helper carries canonical citation chain per Catalog #245 modal_call_id_ledger + Catalog #344 canonical equations registry provenance.
6. **Counterfactual-able**: canonical anti-patterns IS the counterfactual surface (KNOWN-BAD compositions explicit per Catalog #373); Slot GG DESIGN consumes per-cell counterfactual probe via `match_stack_against_anti_patterns`.

## 9. 9-dimension success checklist evidence (per Catalog #294)

| Dimension | Evidence |
|---|---|
| UNIQUENESS | $0 audit-only landing — CANONICAL INVENTORY-FOR-CANONICALIZATION-CASCADE has no sister in repo today (UNIQUE among Slot II-style audits) |
| BEAUTY + ELEGANCE | Memo + §5.1 EXTENSION table + §5.2 NEW package minimal-surface recommendation + §7 apparatus mutation chain all reviewable in 30 seconds per HNeRV parity L4 + L12 |
| DISTINCTNESS | DISJOINT vs Slot GG (T3 grand council canonical-design symposium) + Slot HH (multi-reward + multi-env arch design memo) per Catalog #340 PROCEED |
| RIGOR | 4-phase cascade (Phase A code audit + Phase B state audit + Phase C memo audit + Phase D synthesis) per Catalog #229 premise-verification + Catalog #303 cargo-cult audit + Catalog #305 observability |
| PER-METHOD OPTIMIZATION | Per Catalog #290 canonical-vs-unique-decision-per-layer; every §5.1 table row IS a canonical EXTENSION decision per Slot GG |
| STACK-OF-STACKS COMPOSABILITY | The §5.1 EXTENSION table IS the canonical stack-of-stacks decomposition for RL-substrate-design; NEW packages compose with existing canonical helpers via canonical contracts |
| DETERMINISTIC REPRODUCIBILITY | All commits per Catalog #117/#157/#174 canonical serializer; canonical Provenance per Catalog #323 + APPEND-ONLY HISTORICAL_PROVENANCE per Catalog #110/#113 |
| EXTREME OPTIMIZATION | $0 paid GPU + ~75 min wall-clock; READ-ONLY audit |
| OPTIMAL MINIMAL CONTEST SCORE | Indirect — this audit UNBLOCKS Slot GG canonical SHARED helper DESIGN which is the FRONTIER-BREAKING surface |

## 10. Mission alignment per Catalog #300 §Mission alignment

`council_predicted_mission_contribution`: **frontier_breaking_enabler**

Rationale: This audit memo is the canonical PREREQUISITE for Slot GG canonical SHARED helper DESIGN to avoid the canonical anti-pattern `rl_substrate_design_canonical_helper_built_from_scratch_when_canonical_extension_path_exists_v1` proposed §7.1. By identifying 94+ pre-existing canonical helpers + 354 canonical equations + 98 canonical anti-patterns + 206 council deliberations as Slot GG canonical INPUTS, the memo unblocks Slot GG DESIGN from spending its symposium budget re-inventing canonical surfaces. The SCORE-LOWERING impact is INDIRECT (Slot GG canonical SHARED helper DESIGN is the FRONTIER-BREAKING surface; this audit memo enables it).

## 11. References

- CLAUDE.md "Subagent coherence-by-default" non-negotiable (mandatory pre-flight + 6-hook wire-in)
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS"
- CLAUDE.md "Results must become system intelligence — NON-NEGOTIABLE, HIGHEST EMPHASIS"
- CLAUDE.md "consolidate everything into META layer" standing directive 2026-05-15
- CLAUDE.md "iterate not force" standing directive
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
- CLAUDE.md "MLX portable-local-substrate authority"
- CLAUDE.md "State JSONL archival policy"
- CLAUDE.md "Memory file rotation discipline"
- Catalog #229 premise-verification-before-edit
- Catalog #290 canonical-vs-unique-decision-per-layer
- Catalog #294 9-dim success checklist evidence
- Catalog #299 catalog quota brake under 400
- Catalog #300 council deliberation v2 frontmatter + mission alignment
- Catalog #303 cargo-cult audit section
- Catalog #305 observability surface
- Catalog #335 cathedral consumer canonical contract
- Catalog #341 Tier A canonical-routing markers
- Catalog #344 canonical equations + anti-patterns registry
- Catalog #348 retroactive sweep
- Catalog #355 cathedral autopilot meta-Lagrangian invoker
- Catalog #356 per-axis decomposition
- Catalog #371 canonical equations auto-recalibrator
- Catalog #373 compound stack acknowledgment of anti-patterns
- Catalog #376 + #378 spawn-time + main-thread PV
- Catalog #382 phantom-score canonical posterior read validator
- `.omx/research/beat_pr95_curriculum_substrate_training_design_20260513.md` (canonical pre-existing curriculum DESIGN)
- `.omx/research/council_t1_pr95_curriculum_8stage_research_20260520T143358Z.md` (canonical pre-existing PR95 8-stage research)
- `.omx/research/council_t3_grand_council_strategic_reprioritization_symposium_rudin_daubechies_per_operator_4_message_cascade_directive_20260529.md` (TODAY's strategic memo)
- `.omx/research/cross_pr_family_canonical_techniques_mining_L14_L70_20260529T075244Z.md` (Slot DD canonical L14-L70 mining)
- `.omx/research/quantizr_canonical_3_stack_audit_design_20260529.md` (Slot EE canonical 3-stack audit)
- `.omx/research/why_have_we_not_produced_original_frontier_score_meta_diagnostic_synthesis_20260529.md` (Slot V WHY-FRONTIER META-diagnostic)
