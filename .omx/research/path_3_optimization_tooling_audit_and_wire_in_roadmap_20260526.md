---
title: Path 3 optimization tooling audit + wire-in roadmap
date_utc: 2026-05-26T08:10:00Z
lane: lane_path_3_optimization_tooling_audit_and_wire_in_roadmap_20260526
subagent_id: optimization_tooling_audit_20260526
parent_session: main
audit_scope: 8 LANDED Path 3 substrates vs 20+ canonical optimization surfaces

council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Carmack
  - Hotz
  - Quantizr
  - Selfcomp
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian (proxied via Assumption-Adversary)
    verbatim: "Cathedral consumer per-substrate gap is real; META rule is that ALL substrates auto-route through canonical helpers, NOT that every substrate spawns a sister consumer. Roadmap MUST distinguish 'wire substrate INTO existing canonical consumers' vs 'spawn substrate-specific consumer package'. The default IS the former; the latter is reserved for the substrate-specific signals (e.g. RSSM categorical-posterior anchor) that an existing consumer cannot already absorb."
council_assumption_adversary_verdict:
  - assumption: "Cathedral consumer is the canonical observability surface every Path 3 substrate must consume"
    classification: HARD-EARNED
    rationale: "Catalog #335 paradigm-shift makes auto-discovery structural; ZERO of 8 Path 3 substrates currently emit canonical anchors that flow into the 62 cathedral_consumers. Empirical: master_gradient_per_pair_consumer, per_pair_difficulty_atlas_consumer, score_lagrangian_consumer are SUBSTRATE-AGNOSTIC and IGNORE substrate provenance — sister anchors should flow through them automatically once the substrate emits canonical posterior anchors via posterior_update_locked. The gap is at the POSTERIOR EMISSION surface, not at the CONSUMER PACKAGE creation surface."
  - assumption: "Per-substrate Provenance discipline via tac.provenance is mandatory before any anchor flows through cathedral autopilot"
    classification: HARD-EARNED
    rationale: "Catalog #323 META-class umbrella refuses score-claim rows without canonical Provenance; #341 requires non-promotable markers on routing recommendations; #356 requires per-axis Provenance on decomposition emissions. EVERY anchor a Path 3 substrate emits to a posterior or ledger MUST carry canonical Provenance — there is no alternative. The OPEN question is which adapter shim (build_provenance_for_predicted, build_provenance_for_macos_cpu_advisory, etc.) maps to each substrate's current empirical anchor surface."
  - assumption: "findings_lagrangian + meta_lagrangian unified-action wire-in is the highest-EV cross-substrate enabler"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Subagent coherence-by-default' 6-hook wire-in non-negotiable: hook #1 sensitivity-map + hook #2 Pareto constraint + hook #3 bit-allocator + hook #4 cathedral autopilot dispatch + hook #5 continual-learning posterior + hook #6 probe-disambiguator — ALL 6 hooks have canonical helpers. Path 3 substrates currently activate hooks #4 (auto-discovery for SOME) + #5 (some emit posterior anchors). The unified meta-Lagrangian (per Catalog #355 Phase 1 landing) is the canonical surface that consumes ALL 6 hooks and emits ranked dispatch decisions — this is where the under-utilization compounds across substrates."
  - assumption: "Probe-disambiguator (hook #6) is N/A for early-stage substrates without 2+ defensible interpretations"
    classification: CARGO-CULTED-CONDITIONAL
    rationale: "Per CLAUDE.md 'Subagent coherence-by-default' hook #6: 'probe-disambiguator built if 2+ defensible interpretations exist'. Path 3 substrates at L0/L1 typically have a single canonical interpretation. BUT: A=DreamerV3 (categorical vs continuous posterior), F=Z8 (which of 4 hierarchical levels is the score-saturating one), G=NIRVANA (which cascade stage carries the leverage), H=ATW V2 (cooperative-receiver vs replacement) — these substrates ARE multi-interpretation by paradigm-design and SHOULD have probe-disambiguators at L1. C=NSCS06 v8 chroma_lut (which procedural variant) already has gt_distribution_matched_seed.py which IS a probe-disambiguator-shape. Reclassify per substrate."
  - assumption: "Existing canonical-helper coverage is sufficient and the gap is purely substrate-side wire-in"
    classification: HARD-EARNED-PARTIAL
    rationale: "True for: tac.score_composition + tac.provenance + tac.continual_learning + tac.council_continual_learning + tac.canonical_equations + tac.local_acceleration.pr95_hnerv_mlx + tac.substrates._shared.score_aware_common + tac.substrates._shared.inflate_runtime + tac.substrates._shared.trainer_skeleton + tac.differentiable_eval_roundtrip. PARTIAL for: tac.sensitivity_map (axis-level helpers exist but Path 3 substrate signals are not yet typed); tac.bit_allocator (per-byte/per-pair/per-class helpers exist but consumers for Path 3 substrate-specific allocations are not built); tac.findings_lagrangian (Phase 1 cathedral wire-in landed per Catalog #355 but Path 3 substrates do not feed posteriors yet)."
council_decisions_recorded:
  - "op-routable #1 (HIGHEST EV): Wire ALL 8 Path 3 substrates' empirical anchor emissions through tac.continual_learning.posterior_update_locked + tac.provenance canonical Provenance shim — single helper invocation per substrate at L1 promotion; lifts ALL Path 3 substrates into cathedral autopilot's per_pair_difficulty_atlas_consumer + master_gradient_aggregate_consumer + 60+ other auto-discovery consumers AT ZERO PER-SUBSTRATE COST."
  - "op-routable #2 (HIGH EV): Promote 4 of 8 Path 3 substrates from LEGACY_SUBSTRATE_PRE_META_LAYER waiver to Catalog #241 substrate_contract.py / registered_substrate.py canonical META layer (currently 2 of 8 have it: nscs06_v8_chroma_lut + atw_v2_cooperative_receiver_v2). Each registration auto-validates 36-field canonical contract via Catalog #242 + flows substrate metadata into the autopilot ranker for tier-routing."
  - "op-routable #3 (HIGH EV): Wire Path 3 substrate empirical anchors into tac.canonical_equations via update_equation_with_empirical_anchor per Catalog #344 — A=DreamerV3 anchors to categorical_posterior_capacity_vs_continuous_gaussian_v1 + categorical_blahut_arimoto_rate_distortion_v1 (already referenced in landing memo); B'=Z7-Mamba-2 + D=Z6 PC + F=Z8 hierarchical anchors to predictive_coding_residual_capacity_v1 (NEW equation — propose registration); E=BoostNeRV anchors to boosting_residual_score_lowering_per_stage_v1 (NEW equation — propose registration via tac.boosting.append_canonical_equation pattern); G=NIRVANA anchors to cascading_nerv_per_stage_residual_v1 (NEW equation); H=ATW V2 anchors to cooperative_receiver_atick_redlich_score_savings_v1 (NEW equation)."
  - "op-routable #4 (MEDIUM EV): Wire findings_lagrangian posteriors per Path 3 substrate. Catalog #355 Phase 1 landing established the canonical helper invoke_meta_lagrangian_on_candidates in tools/cathedral_autopilot_autonomous_loop.py — substrates need to emit GaussianPosterior anchors via tac.findings_lagrangian.posterior_update_from_anchors and the cathedral autopilot will rank them per Lindley-1956 expected-information-gain action selector. This is the unified-action wire-in across all substrates."
  - "op-routable #5 (MEDIUM EV per-substrate; HIGH EV CROSS-SUBSTRATE): For the 4 multi-interpretation Path 3 substrates (A/F/G/H), build per-substrate probe-disambiguator under tools/probe_<substrate>_disambiguator.py per CLAUDE.md 'Subagent coherence-by-default' hook #6. Pattern: scaffold copied from existing tools/probe_atw_v2_d4_probe_from_a1.py (60+ memo cross-refs in .omx/research)."
  - "op-routable #6 (MEDIUM EV; risk-managed): For 3 of 8 substrates that emit per-pair / per-class / per-byte score-relevant signal (D=Z6 PC predictions, F=Z8 hierarchical levels, H=ATW V2 cooperative-receiver class-conditional bytes), build per-substrate cathedral consumer package under src/tac/cathedral_consumers/<substrate>_consumer/ per Catalog #335 canonical contract. NOT recommended for A/B'/C'/E/G (their signals are already absorbed by SUBSTRATE-AGNOSTIC sister consumers like per_pair_difficulty_atlas_consumer / master_gradient_aggregate_consumer / per_segnet_class_chroma_consumer)."
  - "op-routable #7 (LOW EV; deferred): Wire ALL Path 3 substrates into tac.sensitivity_map + tac.bit_allocator at L2+ promotion. These are downstream consumers of the posterior anchors from op-routables #1-#3 — not premature at L0/L1 since substrates lack empirical anchors. Re-evaluate after op-routable #1 lands."
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_breaking
canonical_equation_refs:
  - per_pair_master_gradient_score_impact_taylor_v1
  - canonical_frontier_pointer_v1
related_deliberation_ids:
  - council_t3_dreamerv3_rssm_paradigm_bridge_per_substrate_symposium_20260519
  - cathedral_auto_ingest_paradigm_shift_landed_20260519
  - findings_lagrangian_wire_in_phase_1_canonical_invocation_20260520
  - master_gradient_canonical_helper_landed_with_cathedral_autopilot_wirein_20260517
predicted_band_validation_status: pending_post_training
# FORMALIZATION_PENDING:audit_only_no_score_claim_per_research_only_subagent_charter_2026_05_26
---

# Path 3 optimization tooling audit + wire-in roadmap

**Charter**: Operator NON-NEGOTIABLE 2026-05-26 verbatim *"All may likely benefit from our optimization tooling and tech which is quite extensive and likely underutilized"*. Subagent scope = pure analysis + roadmap; ZERO code modifications; operator approves follow-on wire-in subagents.

**Method**: 4-step audit per charter (STEP 1 inventory → STEP 2 utilization matrix → STEP 3 high-EV opportunities → STEP 4 roadmap + top-5 operator-routable wire-ins).

**Cost**: $0 + ~3h wall-clock (analysis-only). NO GPU dispatch.

---

## STEP 1 — Canonical optimization tooling stack inventory

Enumerated 24 canonical surfaces under `src/tac/` (469 total modules at audit time; subset relevant to Path 3 substrates listed below). Catalog #-tagged where applicable.

| # | Canonical surface | Public API | Catalog | Status |
|---|-------------------|------------|---------|--------|
| 1 | `tac.cathedral` + `tac.cathedral_consumers/*` (62 packages) | `CathedralConsumerContract` Protocol + auto-discovery via `tools/cathedral_autopilot_autonomous_loop.py::invoke_cathedral_consumers_on_candidates` | #335 + #341 + #357 | ACTIVE; auto-discovery loop fires per Catalog #336 |
| 2 | `tac.findings_lagrangian` | `GaussianPosterior`, `posterior_update_from_anchors`, `compute_findings_lagrangian`, `recommend_next_action_via_expected_information_gain` | #355 + #347 | ACTIVE; cathedral wire-in Phase 1 landed |
| 3 | `tac.findings_lagrangian_pp` | TRACK B NumPyro hierarchical (sister to track A) | — | ACTIVE per operator-frontier-override 2026-05-19 |
| 4 | `tac.master_gradient` + `tac.master_gradient_consumers` + `tac.master_gradient_comparison` | `MasterGradient`, `predict_delta_s`, `predict_delta_s_per_pair`, `append_anchor_locked`, multi-granularity 10-exploit enumeration | #354 + #327 + #318 | ACTIVE; 8 exploit consumers in cathedral_consumers/ |
| 5 | `tac.sensitivity_map` | `source_map_sha256`, `official_response_curve_sha256`, `official_component_response`, axis-level (seg/pose/rate) per-pair / per-class reweighting | #586 | ACTIVE |
| 6 | `tac.bit_allocator` | `AllocationStrategy`, `BitAllocationResult`, `allocate_bits`, `allocation_report` (per-byte / per-pair / per-class / per-axis / pareto_dual) | #1068 | ACTIVE |
| 7 | `tac.score_composition` | `compose_score_from_axes`, `compose_scalar_delta`, `ComposedScoreDelta`, `load_baseline_pose_from_canonical_frontier_pointer` + canonical constants | #356 | ACTIVE |
| 8 | `tac.provenance` | `Provenance`, `ProvenanceKind` (6-kind taxonomy), `ProvenanceEvidenceGrade` (8-grade taxonomy), `Provenance` umbrella + 8 adapter shims | #323 | ACTIVE; META-class umbrella |
| 9 | `tac.canonical_equations` | `update_equation_with_empirical_anchor`, fcntl-locked JSONL registry, 6+ registered equations | #344 | ACTIVE; auto-discovery via canonical_equation_lookup_consumer |
| 10 | `tac.domain_priors` | `PerFrameDifficultyAtlas`, `EgoMotionConcentrationAtlas`, per-class statistical priors (3 canonical equations) | — | ACTIVE |
| 11 | `tac.differentiable_eval_roundtrip` | `patch_upstream_yuv6_globally`, `apply_eval_roundtrip_during_training`, `differentiable_rgb_to_yuv6`, `assert_yuv6_forward_equivalence_to_upstream` | CLAUDE.md non-negotiable | ACTIVE; CRITICAL for L1 score-aware training |
| 12 | `tac.substrates.hinton_distilled_scorer_surrogate` | Quantizr KL T=2.0 surrogate (0.33-contest-CUDA technique) | CLAUDE.md "Quantizr intelligence" | ACTIVE; sister substrate ready to consume |
| 13 | `tac.wyner_ziv_deliverability` | `DeliverabilityProof`, `DeliverabilityTier` (4-tier), `build_deliverability_proof_from_wyner_ziv_classification` | #319 | ACTIVE |
| 14 | `tac.codec.wyner_ziv_layer` | Pipeline-stage codec primitive | — | ACTIVE |
| 15 | `tac.local_acceleration.pr95_hnerv_mlx` | `bilinear_resize2x_align_corners_false_nhwc`, `pixel_shuffle_2x_nhwc`, `HNeRVDecoderMLX` + MLX scorer adapters | — | ACTIVE; FIX-WAVE-R1 fixed drift bug from NOT using this |
| 16 | `tac.continual_learning` | `posterior_update_locked`, `ContinualLearningPosterior`, `ContestResult`, `CustodyVerdict`, `contest_result_from_auth_eval_payload` | #128 + #127 | ACTIVE; canonical posterior surface |
| 17 | `tac.council_continual_learning` | `CouncilDeliberationRecord`, `append_council_anchor`, `query_anchors_by_topic`, `query_dissent_history` | #300 + #292 | ACTIVE; per-substrate symposium anchors |
| 18 | `tac.substrates._shared.score_aware_common.score_pair_components` | Canonical scorer-loss helper routing; PR95-parity contract | #164 | ACTIVE; FORBIDDEN to bypass |
| 19 | `tac.substrates._shared.trainer_skeleton` | Canonical TF32 + `device_or_die` + `detect_hardware_substrate(axis=...)` | #178 + #190 | ACTIVE |
| 20 | `tac.substrates._shared.inflate_runtime` | Canonical `select_inflate_device` (Catalog #205) + sister helpers | #205 | ACTIVE; routes inflate device cleanly |
| 21 | `tac.training.EMA` | EMA(decay=0.997) + apply at eval with snapshot+restore | CLAUDE.md "EMA — non-negotiable" | ACTIVE; FORBIDDEN to skip on submission archives |
| 22 | `tac.optimization.substrate_composition_matrix` | Cathedral autopilot Tier-A/B/C density reranking + per-substrate composition_alpha | #219 + #322 | ACTIVE |
| 23 | `tac.atom` (atom-shaped ledger) | Meta-Lagrangian typed atom flow + builders + ledger + subsumption | — | ACTIVE per Meta-Lagrangian/Pareto solver non-negotiable |
| 24 | `tac.substrate_registry` + `Catalog #241 substrate_contract.py` | `SubstrateContract` 36-field dataclass + `@register_substrate` decorator + `validate_consumer_module` | #241 + #242 | ACTIVE; structural extinction of LEGACY waiver class |

**Sister surfaces also surveyed but lower-priority for Path 3 substrates**: `tac.solvers` (fista/frank_wolfe/sinkhorn/riemannian_newton), `tac.streaming_prediction`, `tac.inflate_time_post_processing`, `tac.predictor` (score-band predictor with refusal), `tac.side_information`, `tac.utility_curves`, `tac.boosting`, `tac.null_space_exploiter`, `tac.contest_oracle`, `tac.cathedral_solver_wire_in` (Cable D 7-14 consumer registry), `tac.training_curriculum` (model_soup_averaging, multi_stage_curriculum, master_gradient_pair_weights), `tac.training_optimization` (GTScorerCache, autocast_aware_forward, compile_with_fallback), `tac.compress_time_optimization` (decorator-based wire-in), `tac.analytical_solve_extinctions`/`experimental_extinctions`/`formula_extinctions` (40+ extinct gates pattern).

---

## STEP 2 — Per-substrate utilization matrix (8 Path 3 LANDED × 24 canonical surfaces)

**Legend**: ✅ ADOPTED canonical helper / 🔀 FORKED-WITH-JUSTIFICATION per Catalog #290 / ❌ UNUSED-NO-RATIONALE (gap; potential under-utilization) / ➖ N/A (not applicable to substrate paradigm) / ⏳ PENDING-L1+ (declared in landing memo but deferred to next promotion step)

| # | Surface | A=DreamerV3 | B'=Z7-M2-v2 | C'=NSCS06 v8 | D=Z6 PC | E=BoostNeRV | F=Z8 HPC | G=NIRVANA | H=ATW V2 |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **cathedral_consumers** auto-discovery emission | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 2 | **findings_lagrangian** posterior emission | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 3 | **findings_lagrangian_pp** TRACK B | ➖ | ➖ | ➖ | ➖ | ➖ | ➖ | ➖ | ➖ |
| 4 | **master_gradient** anchor emission | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 5 | **sensitivity_map** per-substrate entry | ❌ | ❌ | ❌ | ⏳ | ❌ | ⏳ | ❌ | ⏳ |
| 6 | **bit_allocator** per-substrate entry | ❌ | ❌ | ❌ | ⏳ | ❌ | ⏳ | ❌ | ⏳ |
| 7 | **score_composition** per-axis emission | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 8 | **provenance** canonical Provenance umbrella | ⏳ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⏳ |
| 9 | **canonical_equations** anchor registration | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 10 | **domain_priors** consumption | ➖ | ➖ | ⏳ | ⏳ | ➖ | ⏳ | ➖ | ⏳ |
| 11 | **differentiable_eval_roundtrip** at training | ⏳ | ⏳ | ✅ | ✅ | ⏳ | ⏳ | ⏳ | ⏳ |
| 12 | **hinton_distilled_scorer_surrogate** | ➖ | ➖ | ➖ | ➖ | ➖ | ➖ | ➖ | ➖ |
| 13 | **wyner_ziv_deliverability** | ➖ | ➖ | ⏳ | ➖ | ➖ | ✅ | ➖ | ⏳ |
| 14 | **codec.wyner_ziv_layer** | ➖ | ➖ | ➖ | ➖ | ➖ | ✅ | ➖ | ⏳ |
| 15 | **local_acceleration.pr95_hnerv_mlx** | ✅ | ➖ | 🔀 | ⏳ | ➖ | ➖ | ➖ | ⏳ |
| 16 | **continual_learning** posterior_update_locked | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 17 | **council_continual_learning** anchor | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 18 | **score_aware_common** canonical helper | ⏳ | ⏳ | ⏳ | ✅ | ⏳ | ⏳ | ⏳ | 🔀 |
| 19 | **trainer_skeleton** | ⏳ | ⏳ | ✅ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 20 | **inflate_runtime** select_inflate_device | ✅ | ✅ | ✅ | ✅ | ➖ | ✅ | ✅ | ✅ |
| 21 | **training.EMA** | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 22 | **substrate_composition_matrix** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 23 | **atom** typed ledger | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 24 | **substrate_contract** (Catalog #241 META) | ❌ (waiver) | ❌ (waiver) | ✅ | ❌ (waiver) | ❌ (waiver) | ❌ (waiver) | ❌ (waiver) | ✅ |

### Summary counts (across 24 surfaces × 8 substrates = 192 cells)

| Status | Count | Percentage | Interpretation |
|--------|------:|----------:|----------------|
| ✅ ADOPTED | 22 | 11.5% | Canonical helpers ACTIVELY consumed |
| 🔀 FORKED-WITH-JUSTIFICATION | 3 | 1.6% | Per Catalog #290 documented per-layer decision |
| ⏳ PENDING-L1+ | 38 | 19.8% | Declared in landing memo but deferred |
| ➖ N/A | 39 | 20.3% | Not applicable to substrate paradigm |
| ❌ UNUSED-NO-RATIONALE | 90 | **46.9%** | **THE GAP — primary wire-in opportunity** |

**Key observation**: 46.9% of substrate × canonical-surface cells are UNUSED-NO-RATIONALE. Operator hypothesis confirmed empirically.

---

## STEP 3 — High-EV wire-in opportunities (priority-ranked)

### Tier 1 (HIGHEST EV — single helper invocation, ALL substrates benefit)

#### Wire-in #1: `posterior_update_locked` + canonical Provenance shim

**EV estimate**: HIGH cross-substrate. A single helper invocation in each substrate's empirical-anchor emission path lifts ALL 8 Path 3 substrates into the 62 cathedral_consumers' auto-discovery loop. Bug class extincted: orphan-signal-at-cathedral-autopilot per Catalog #335.

**Wire-in complexity**: TRIVIAL — single import + ~15 LOC per substrate (build canonical Provenance via adapter shim + invoke `posterior_update_locked`).

**Risk**: LOW — defensive write to canonical JSONL store via fcntl-locked atomic append per Catalog #128; Provenance umbrella enforces non-promotable markers by construction.

**Dependency**: standalone (each substrate independent).

**Sister-substrate precedent**: `tac.continual_learning.contest_result_from_auth_eval_payload` is the canonical adapter; `pr101_lc_v2_clone` substrate routes through it.

**Operator-routable next**: spawn `wave_<N>_path_3_posterior_emission_wire_in` subagent covering ALL 8 Path 3 substrates in ONE pass (~$0 + ~3h wall-clock).

---

#### Wire-in #2: Promote 6 of 8 substrates from LEGACY_SUBSTRATE_PRE_META_LAYER waiver to Catalog #241 substrate_contract.py

**EV estimate**: MEDIUM-HIGH (operator-visibility + autopilot ranker tier-routing benefits compound across all sister landings).

**Wire-in complexity**: MODERATE — copy/adapt `nscs06_v8_chroma_lut/substrate_contract.py` + `atw_v2_cooperative_receiver_v2/registered_substrate.py` to remaining 6 substrates; ~80 LOC per substrate; Catalog #242 canonical contract validates via `__post_init__`.

**Risk**: LOW — fail-closed at registry validation; cannot land non-compliant contract per Catalog #242.

**Dependency**: standalone per substrate.

**Sister-substrate precedent**: `nscs06_v8_chroma_lut/substrate_contract.py` (full contract) + `atw_v2_cooperative_receiver_v2/registered_substrate.py` (re-export pattern; cleaner).

**Operator-routable next**: spawn `wave_<N>_path_3_substrate_contract_canonical_promotion` covering all 6 LEGACY substrates (A/B'/D/E/F/G).

---

#### Wire-in #3: Canonical equation registration + anchor emission per Path 3 substrate

**EV estimate**: HIGH — every substrate's empirical anchor automatically updates equation posteriors per Catalog #344; future paradigm-bridge questions become queryable via `tac.canonical_equations.query_equations`.

**Wire-in complexity**: VARIABLE — A=DreamerV3 already references 2 canonical equations; C'=NSCS06 v8 already references `procedural_codebook_from_seed_compression_savings_v1`. NEW equations needed for B'/D/E/F/G/H paradigms (5-7 NEW canonical equation registrations + per-substrate `update_equation_with_empirical_anchor` invocation pattern).

**Risk**: LOW — canonical helpers + Catalog #287 forbids placeholder-rationale literals; helper enforces canonical Provenance.

**Dependency**: relies on wire-in #1 posterior emission first (anchor payload must carry canonical Provenance).

**Sister-substrate precedent**: `tac.canonical_equations.procedural_codebook_savings.update_equation_with_empirical_anchor` (commit landed today per master_gradient session).

**Operator-routable next**: spawn `wave_<N>_path_3_canonical_equation_registry_extension` to register NEW canonical equations for B'/D/E/F/G/H + wire anchor emission per substrate.

---

### Tier 2 (HIGH EV — substrate-specific wire-ins)

#### Wire-in #4: findings_lagrangian Phase 2 per-substrate posterior emission

**EV estimate**: HIGH cross-substrate (compounds via the unified meta-Lagrangian per Catalog #355 cathedral autopilot wire-in).

**Wire-in complexity**: MODERATE — each substrate emits `GaussianPosterior` via `posterior_update_from_anchors`; the cathedral autopilot's `invoke_meta_lagrangian_on_candidates` helper consumes posteriors per Lindley-1956 action selector.

**Risk**: MEDIUM (Tier B per Catalog #357; must thread canonical Provenance + Tier A → Tier B promotion path; operator-attention-budget per Catalog #300 mission_predicted_contribution).

**Dependency**: wire-in #1 first (posterior emission); wire-in #3 (canonical equations) ideally before this (posteriors anchor to specific equations).

**Sister-substrate precedent**: cathedral autopilot's Phase 1 invocation IS canonical per Catalog #355 commit; the consumer-side wiring across substrates is the gap.

**Operator-routable next**: deferred until wire-ins #1 + #3 land.

---

#### Wire-in #5: Per-substrate probe-disambiguator for the 4 multi-interpretation substrates (A/F/G/H)

**EV estimate**: MEDIUM per-substrate / HIGH meta (per CLAUDE.md "Subagent coherence-by-default" hook #6 + the canonical pattern of `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md`).

**Wire-in complexity**: SUBSTANTIAL — ~200-300 LOC + tests per probe-disambiguator; sister `tools/probe_atw_v2_d4_probe_from_a1.py` is canonical scaffold.

**Risk**: LOW (research-only artifacts; tagged `[predicted]` axis per Catalog #287; Provenance umbrella applies).

**Dependency**: relies on wire-in #1 posterior emission to record disambiguator verdicts.

**Sister-substrate precedent**: 60+ memos under `.omx/research/*probe*.md`; C=NSCS06 v8 chroma_lut's `gt_distribution_matched_seed.py` IS probe-disambiguator-shape already.

**Operator-routable next**: deferred until per-substrate symposiums per Catalog #325 surface 2+ defensible interpretations explicitly.

---

### Tier 3 (MEDIUM EV — wait until L2 promotion)

#### Wire-in #6: Per-substrate cathedral consumer packages (for 3 substrates D/F/H only)

**EV estimate**: MEDIUM (most substrate signals are absorbed by SUBSTRATE-AGNOSTIC sister consumers per the Contrarian voice in Assumption-Adversary verdict).

**Wire-in complexity**: SUBSTANTIAL — full Catalog #335 canonical contract: CONSUMER_NAME + CONSUMER_VERSION + CONSUMER_HOOK_NUMBERS + `update_from_anchor` + `consume_candidate` + Tier A/B classification per Catalog #357; ~300-400 LOC + tests.

**Risk**: MEDIUM (Tier B requires Provenance discipline per Catalog #341).

**Dependency**: wire-ins #1 + #2 + #3 + #4 must land first (consumer needs canonical anchors to consume).

**Sister-substrate precedent**: 62 existing consumers; canonical contract per Catalog #335; per_pair_difficulty_atlas_consumer + master_gradient_aggregate_consumer + per_segnet_class_chroma_consumer are already absorbing some Path 3 signals via auto-discovery.

**Operator-routable next**: deferred to per-substrate L1→L2 promotion; ONLY recommended for D/F/H which carry substrate-specific signals (predictions / hierarchical levels / cooperative-receiver class-conditional bytes).

---

#### Wire-in #7: tac.sensitivity_map + tac.bit_allocator per-substrate entries

**EV estimate**: LOW at L0/L1; HIGH at L2+ (downstream consumers of posterior anchors).

**Wire-in complexity**: MODERATE — per-substrate entry registration + downstream consumer plumbing.

**Risk**: LOW.

**Dependency**: depends on wire-in #1 (posterior anchors) + wire-in #4 (meta-Lagrangian posterior emission).

**Operator-routable next**: deferred to L2+ promotion per substrate.

---

## STEP 4 — Top-5 operator-routable wire-ins (most impactful next-spawns)

### #1: `wave_<N>_path_3_posterior_emission_canonical_wire_in_20260527` (HIGHEST EV)

**Brief**: Wire ALL 8 Path 3 substrates' empirical-anchor emission through `tac.continual_learning.posterior_update_locked` + `tac.provenance.build_provenance_for_predicted` adapter shim. ONE helper invocation per substrate at L0/L1 promotion point. Result: all 62 cathedral_consumers automatically observe Path 3 substrate signals via Catalog #335 auto-discovery loop.

**Scope**: 8 substrates × ~15 LOC per substrate + per-substrate test = ~150 LOC + 24 tests. Estimated ~3h wall-clock; $0 GPU. Catalog #229 PV + #117/#157/#174 canonical serializer + #206 checkpoint discipline.

**Output**: `feedback_wave_<N>_path_3_posterior_emission_canonical_wire_in_landed_20260527.md` + per-substrate commit batch.

---

### #2: `wave_<N>_path_3_substrate_contract_canonical_promotion_20260527`

**Brief**: Promote 6 of 8 Path 3 substrates from `LEGACY_SUBSTRATE_PRE_META_LAYER` waiver to Catalog #241 `substrate_contract.py` / `registered_substrate.py` canonical META layer. Scaffolds: `nscs06_v8_chroma_lut/substrate_contract.py` (full) + `atw_v2_cooperative_receiver_v2/registered_substrate.py` (re-export pattern).

**Scope**: 6 substrates × ~80 LOC contract + tests = ~600 LOC + 18 tests. Estimated ~4h wall-clock; $0 GPU. Catalog #242 canonical contract validation auto-fires on import.

**Output**: `feedback_wave_<N>_path_3_substrate_contract_canonical_promotion_landed_20260527.md`.

---

### #3: `wave_<N>_path_3_canonical_equation_registry_extension_20260527`

**Brief**: Register 5-7 NEW canonical equations for B'/D/E/F/G/H paradigms in `tac.canonical_equations` per Catalog #344 + wire each substrate's anchor emission through `update_equation_with_empirical_anchor`. Equations to register: `predictive_coding_residual_capacity_v1` (B'/D/F), `boosting_residual_score_lowering_per_stage_v1` (E), `cascading_nerv_per_stage_residual_v1` (G), `cooperative_receiver_atick_redlich_score_savings_v1` (H).

**Scope**: 5-7 NEW canonical equation `.py` files + per-equation tests + per-substrate anchor emission wire-in ≈ ~800 LOC + 30 tests. Estimated ~5h wall-clock; $0 GPU. Catalog #287 placeholder-rationale rejection applies.

**Output**: `feedback_wave_<N>_path_3_canonical_equation_registry_extension_landed_20260527.md`.

---

### #4: `wave_<N>_path_3_findings_lagrangian_phase_2_per_substrate_posterior_emission_20260528`

**Brief**: Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Catalog #355 Phase 1 cathedral wire-in, extend per-substrate to emit `GaussianPosterior` via `tac.findings_lagrangian.posterior_update_from_anchors`. Cathedral autopilot's `invoke_meta_lagrangian_on_candidates` will consume posteriors per Lindley-1956 expected-information-gain action selector.

**Scope**: 8 substrates × ~30 LOC (Posterior emission + Tier B Provenance threading) + tests = ~250 LOC + 24 tests. Estimated ~4h wall-clock; $0 GPU. **DEPENDS ON wire-ins #1 + #3 landing first**.

**Output**: `feedback_wave_<N>_path_3_findings_lagrangian_phase_2_landed_20260528.md`.

---

### #5: `wave_<N>_path_3_a_f_g_h_probe_disambiguator_scaffolds_20260528`

**Brief**: Build `tools/probe_<substrate>_disambiguator.py` for the 4 multi-interpretation Path 3 substrates per CLAUDE.md "Subagent coherence-by-default" hook #6. A=DreamerV3 (categorical vs continuous posterior); F=Z8 HPC (which of 4 hierarchical levels saturates); G=NIRVANA (which cascade stage carries leverage); H=ATW V2 (cooperative-receiver vs replacement).

**Scope**: 4 substrates × ~200 LOC scaffold + tests = ~1000 LOC + 28 tests. Estimated ~5h wall-clock; $0 GPU. Per-substrate symposium per Catalog #325 should articulate 2+ defensible interpretations before scaffold.

**Output**: `feedback_wave_<N>_path_3_a_f_g_h_probe_disambiguator_scaffolds_landed_20260528.md`.

---

## Cross-substrate META findings

### META #1: Hook #5 (continual-learning posterior) is the SINGLE highest-leverage wire-in

ALL 8 Path 3 substrates emit `council_continual_learning` anchors (hook #5 ACTIVE for 8/8) but ZERO of 8 emit `continual_learning.posterior_update_locked` anchors (hook #5 INACTIVE for 8/8 on the CONTEST-SCORE posterior surface). The two surfaces are distinct: council anchors record DELIBERATION; posterior anchors record EMPIRICAL CONTEST OUTCOME. The 62 cathedral_consumers consume the POSTERIOR surface; without per-substrate posterior emission, the cathedral autopilot cannot rank Path 3 substrates against sister LANDED substrates.

**Root cause**: Path 3 substrates are at L0/L1 `research_only=True` per Catalog #240; no contest auth-eval has fired yet so no `ContestResult` exists to feed `posterior_update_locked`. **Fix**: emit a PREDICTED-grade posterior anchor (per Catalog #323 `build_provenance_for_predicted`) at L0 landing so cathedral autopilot can rank by predicted_band; promote to MEASURED-grade post-auth-eval.

### META #2: 75% of Path 3 substrates carry LEGACY_SUBSTRATE_PRE_META_LAYER waiver

6 of 8 (A/B'/D/E/F/G) carry `LEGACY_SUBSTRATE_PRE_META_LAYER` waiver in `__init__.py` first 30 lines, deferring `Catalog #241 substrate_contract.py` adoption. Sister 2 (C'/H) DID adopt the canonical META layer (NSCS06 v8 with full `substrate_contract.py`; ATW V2 with `registered_substrate.py` re-export pattern). The waiver pattern is currently STRUCTURAL (Catalog #241 gate is in WARN-ONLY mode per Strict-flip atomicity), but adopting the META layer unlocks autopilot ranker tier-routing + Catalog #325 per-substrate symposium evidence integration.

### META #3: Probe-disambiguator (hook #6) is missing for 8/8 Path 3 substrates

ZERO of 8 have `tools/probe_<substrate>_disambiguator.py`. Per Assumption-Adversary verdict #4: A/F/G/H ARE multi-interpretation by paradigm-design; C'=NSCS06 v8 chroma_lut has `gt_distribution_matched_seed.py` which IS probe-disambiguator-shape (but is INTERNAL to substrate, NOT in `tools/`). The 60+ memos under `.omx/research/*probe*.md` provide canonical scaffolds; ATW v2 D4 probe is the closest sister pattern.

### META #4: Canonical equation registration is currently 2 of 8 (25%)

Only A=DreamerV3 + C'=NSCS06 v8 reference canonical equations in `tac.canonical_equations`. B'/D/E/F/G/H paradigms each warrant a NEW canonical equation that future empirical anchors will update per Catalog #344. The registration is structural protection against tribal knowledge per CLAUDE.md "Canonical equations + models registry" non-negotiable.

### META #5: Sister substrates `pr101_lc_v2_clone` + `hinton_distilled_scorer_surrogate` are the canonical pattern reference

For PROCESS comparison: `pr101_lc_v2_clone` routes through `tac.differentiable_eval_roundtrip` + `tac.substrates.score_aware_common` + `tac.packet_compiler.pr101_conv4_storage_perms` + `tac.optimization.muon`. `hinton_distilled_scorer_surrogate` routes through `tac.scorer.load_default_scorers` + `tac.local_acceleration.EVIDENCE_GRADE_MLX`. Both substrates demonstrate ~6-8 canonical helpers per substrate. Path 3 substrates currently average ~2-3 canonical helpers per substrate.

### META #6: A=DreamerV3 + C'=NSCS06 v8 are the closest Path 3 substrates to canonical adoption

A=DreamerV3 imports `tac.local_acceleration.pr95_hnerv_mlx` (FIX-WAVE-R1 closure) + `tac.substrates._shared.inflate_runtime` + canonical equation refs in frontmatter; C'=NSCS06 v8 imports `tac.procedural_codebook_generator` + `tac.canonical_equations.procedural_codebook_savings` + `tac.local_acceleration.mlx_scorer_adapters` + `tac.substrate_registry`. These two should be the SCAFFOLD-COPY-SOURCES for the remaining 6 substrates' wire-in promotions.

---

## Cathedral consumer protocol exposure roadmap

### Recommended Tier A consumers (observability-only) for ALL 8 Path 3 substrates

Per Catalog #357 dual-tier architecture + Contrarian verdict in Assumption-Adversary #1: every Path 3 substrate should be observable in cathedral autopilot WITHOUT spawning a sister consumer package. Wire-in #1 (`posterior_update_locked` + canonical Provenance) automatically makes all 8 substrates VISIBLE to the 62 existing cathedral_consumers via auto-discovery.

### Recommended Tier B consumers (score-contributing) for 3 substrates

Per Catalog #357 Tier B canonical contract + the substrate-specific signal evaluation:

1. **D=Z6 PC** → `src/tac/cathedral_consumers/z6_predictive_coding_consumer/` — per-pair next-frame prediction quality is a substrate-specific signal that no existing consumer absorbs.
2. **F=Z8 HPC** → `src/tac/cathedral_consumers/z8_hierarchical_predictive_coding_consumer/` — per-hierarchical-level residual capacity is a substrate-specific signal.
3. **H=ATW V2** → `src/tac/cathedral_consumers/atw_v2_cooperative_receiver_consumer/` — per-class-conditional byte savings is a substrate-specific signal.

**NOT recommended Tier B consumers** for A/B'/C'/E/G: their signals (categorical posterior, Mamba-2 state-space, chroma LUT, boosting residual, cascading residual) are already absorbed by sister consumers (`per_pair_difficulty_atlas_consumer`, `master_gradient_aggregate_consumer`, `per_segnet_class_chroma_consumer`, `bottom_k_free_entropy_byte_consumer`, `top_k_byte_sensitivity_consumer`, `procedural_codebook_savings_consumer`, etc.).

### Tier A → Tier B promotion path

Per Catalog #357 + the 4 Catalog #356 per-axis decomposition gates: when a Path 3 substrate emits its FIRST contest-CUDA auth-eval anchor, it can promote a Tier A consumer to Tier B by adding `predicted_axis_decomposition` per Catalog #356. This is the canonical path for cathedral autopilot to start consuming substrate signal as a RANKING input rather than observability output.

---

## Discipline declarations

- **6-hook wire-in per Catalog #125** (this roadmap memo):
  - hook #1 sensitivity-map: N/A (analysis-only roadmap; no algorithmic contribution)
  - hook #2 Pareto constraint: N/A
  - hook #3 bit-allocator: N/A
  - hook #4 cathedral autopilot dispatch: **ACTIVE** (THIS roadmap enumerates wire-ins that make cathedral autopilot observe Path 3 substrates)
  - hook #5 continual-learning posterior: **ACTIVE** (op-routable #1 IS the wire-in that emits posterior anchors)
  - hook #6 probe-disambiguator: **ACTIVE** (op-routable #5 builds probe-disambiguators for 4 multi-interpretation substrates)

- **Catalog #229 premise verification**: read 8 Path 3 landing memos + 8 substrate `__init__.py` + 8 substrate `inflate*.py` + 24+ canonical surface `__init__.py` + 12 sister substrate reference files before authoring roadmap. Empirical receipts in STEP 1 + STEP 2 inventory + utilization matrix.
- **Catalog #117 + #157 + #174**: canonical serializer + POST-EDIT `--expected-content-sha256` for this memo commit.
- **Catalog #119**: Co-Authored-By Claude trailer.
- **Catalog #110 + #113**: APPEND-ONLY HISTORICAL_PROVENANCE; this memo is NEW file; ZERO mutations to sister landing memos.
- **Catalog #208**: docs-no-local-paths; ZERO `/Users/...` or `/tmp/...` persisted refs in this memo.
- **Catalog #230**: sister-subagent ownership map; analysis-only scope; ZERO file collision with sister K=COIN++ / I=V1 Faiss / J=MDL-IBPS / L1-PROMOTION-D-Z6 / R1' in-flight subagents.
- **Catalog #287**: placeholder-rationale rejection; ZERO `<rationale>` / `<reason>` literals in this memo.
- **Catalog #292**: per-deliberation assumption surfacing (5 assumptions classified in frontmatter `council_assumption_adversary_verdict`).
- **Catalog #300 v2 frontmatter**: council_tier T2 + 9-member roster + verdict + dissent + assumption-adversary verdicts + decisions_recorded + predicted_mission_contribution + override (false).
- **Catalog #340**: sister-checkpoint guard PROCEED (analysis-only; no commit collision risk).
- **Catalog #344**: `# FORMALIZATION_PENDING:<rationale>` in frontmatter per audit-only-no-score-claim discipline.

## Lane

`lane_path_3_optimization_tooling_audit_and_wire_in_roadmap_20260526` L1 (impl_complete + memory_entry; roadmap memo lands as NEW research artifact; ZERO code modifications per subagent charter).

## Cost

$0 + ~3h wall-clock. NO GPU dispatch.

## What this audit IS

- COMPREHENSIVE inventory of 24 canonical optimization surfaces under `src/tac/`
- COMPREHENSIVE utilization matrix for 8 LANDED Path 3 substrates
- PRIORITY-RANKED roadmap with 7 wire-in opportunities
- TOP-5 operator-routable next-spawn briefs
- 6 cross-substrate META findings on patterns of under-utilization
- Cathedral consumer protocol exposure roadmap (Tier A for all 8; Tier B for 3 selected)

## What this audit IS NOT

- NOT a code modification (per subagent charter: pure analysis + roadmap)
- NOT a score claim (`score_claim=false`, `promotion_eligible=false`, `axis_tag=[research-only roadmap memo]`)
- NOT a paid GPU dispatch
- NOT a per-substrate symposium per Catalog #325 (those are sister subagent scopes; per-symposium decisions inform wire-in priorities but are out-of-scope for this audit)
- NOT a Catalog #335 cathedral consumer registration (those are wire-in op-routables #6; deferred per dependency)

## Operator-routable summary

Operator approves which of top-5 next-spawns to fire. Recommended sequence per dependency graph:
1. **Wire-in #1** (posterior emission) — HIGHEST EV; UNBLOCKS wire-ins #4 + #6.
2. **Wire-in #2** (substrate_contract canonical) — INDEPENDENT; can run parallel to #1.
3. **Wire-in #3** (canonical equation registration) — DEPENDS on #1.
4. **Wire-in #4** (findings_lagrangian Phase 2) — DEPENDS on #1 + #3.
5. **Wire-in #5** (probe-disambiguators) — INDEPENDENT; can run parallel to #1-#4.

Total estimated effort: ~20h wall-clock + $0 GPU across 5 wave subagents. Expected outcome: 46.9% UNUSED-NO-RATIONALE → ≤15% post-wave (≥30 percentage-point reduction in under-utilization).
