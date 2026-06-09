# Evaluator-Inverse / Scorer-Exploit Orphan Inventory — 2026-06-09

**Subagent:** `evaluator_inverse_orphan_inventory_20260609` (READ-ONLY)
**Evidence grade:** `[macOS-CPU advisory]` / mechanism-only. No score claims; no dispatch.
**Audit-provenance:** every value below cites `{file, line, observed_value, reproduce_command}`.
**Operator directive (2026-06-09):** before tasks #35 (frame1 joint-safe-cone), #36 (atlas engine), #46 (LF rate-distortion waterfiller), #47 (null-space/invisibility basis) are built FRESH, inventory the EXTENSIVE existing partial/orphaned work. **Do not rebuild what exists** (no-duplicative-code directive).

**Canonical frontier at audit time** (reproduce: `cat .omx/state/canonical_frontier_pointer.json`):
- contest-CPU: **0.19198533** — archive `b7106c9b…`, 178,493 B, arch class `fp11_source_brotli_recode_b7106c9bdbb8_cpu_exact`, measured 2026-05-28.
- contest-CUDA: 0.20533 — archive `9cb989ce…`, 186,876 B, `lane_pr106_format0d_latent_score_table`, 2026-05-16.

**Headline:** 103 task-relevant module surfaces in `src/tac/` (non-test) + 19 `contest_exploits/*.py`.
Reproduce count: `find src/tac -type f -name '*.py' \( -iname '*sensitivity*' -o -iname '*saliency*' -o … \) | grep -v pycache | grep -v test_ | wc -l` → `103`.

> **CRITICAL FINDING:** tasks #46 and #36's canonical admission-law surfaces were committed in the **last 48 hours** by a sister agent and are directly on-point. Building them fresh would duplicate live code. See the REUSE PLAN.

---

## A. Mapping table — surface → what it computes → status → task → verdict

Status legend: **CONSUMED** = has non-test production importers; **ORPHAN** = no non-test importer; **PKG** = directory package.
Five-link = the AGENTS.md 6-hook wire-in posture (sensitivity / Pareto / bit-allocator / cathedral / continual-learning / probe). Inferred from consumer set + module docstrings; "advisory" where not directly verified.

### Task #46 — LF rate-distortion waterfiller (evaluator-conditioned)

| Surface | Computes (file:line) | Last commit | Consumers | Task verdict |
|---|---|---|---|---|
| `src/tac/optimization/lf_payload_rate_distortion.py` | **THE LAW**: `keep c iff -ΔS_distortion(c) > 25·Δbytes(c)/37,545,489`, ΔS_distortion=`100·Δd_seg+Δsqrt(10·d_pose)`; scorer-response (NOT pixel-variance) reverse-waterfill over SNeRV LF/source-state payload; sensitivity from `scorer_spectral_sensitivity.v2` atlas (l.1–16) | **2026-06-09 (TODAY)** | (none found yet — fresh) | **REUSE-AS-IS.** This *is* task #46. `keep_component` (l.292), `delta_distortion_score` (l.271), `estimate_section_sensitivity` (l.459), `CandidateActionEvaluation` (l.501). Import it; do not rebuild. |
| `src/tac/optimization/joint_p18_p19_waterfill.py` | Coupled scorer water-fill weight `w_i = 100·|dL_seg/dx_i| + 5/sqrt(10·d_pose)·‖J_pose,i‖_{Σ⁻¹}`; pose-null subset = low-Mahalanobis atoms; LOCAL TANGENT PLANE of S_full at `linearization_archive_sha` (l.1–22) | 2026-06-01 | `analysis/hprc_saliency_rd_allocation.py`, `hprc_synthesis_adjoint.py`, `score_exact_saliency.py`, `nerv_control_inventory.py` | **REUSE-AS-IS** for the bit-allocator weight inside #46/#47. Already the canonical P18(seg-flip)+P19(pose-Jacobian) coupling. |
| `src/tac/optimization/scorer_region_waterfill.py` | P19/P18 scorer-region artifacts for queue-owned RD cascades (l.1) | 2026-05-31 | `substrates/hprc/native_rate_surface.py` + 3 tools incl `build_p19_posenet_null_pairs.py` | **EXTEND** — region-granular companion to the per-section LF law. |
| `src/tac/analysis/hprc_saliency_rd_allocation.py` | HPRC saliency→RD allocation | 2026-05-x | consumes joint_p18_p19 | EXTEND/reference. |
| `src/tac/substrates/z8_hierarchical_predictive_coding/per_subband_rd_waterfill_solver.py` + `joint_coefficient_waterfill.py` | per-subband RD water-fill (Z8 wavelet) | (Z8 lane) | REUSE-the-solver-kernel for subband allocation if LF payload is wavelet-banded. |
| `src/tac/water_filling_codec.py` / `_v2.py`, `joint_admm_proximal_water_filling_v2.py`, `solvers/numba_jit_water_filling.py` | generic water-fill kernels (ADMM/proximal, numba JIT) | older | REUSE the numerical kernel; do not reinvent the projection. |

### Task #36 — Atlas engine (score-effect codebook / ActionEffect IR)

| Surface | Computes (file:line) | Last commit | Consumers | Task verdict |
|---|---|---|---|---|
| `src/tac/analysis/action_effect.py` | **The ActionEffect IR named in AGENTS.md.** TWO surfaces: `EvaluatorActionEffect` (schema `nerv_action_effect.v1`, receiver-surface admission + custody-hash gating + commutator/ledger) and thin `ActionEffect` (schema `tac.action_effect.v1`, ONE typed ledger row across HiNeRV-birth + pair-servo + PR110-selector; `compute_delta_scores` delegates to `tac.score_geometry.contest_score`) (l.1–35) | **2026-06-07** | 8+ incl `inverse_scorer_actions.py`, `support_codec_router.py`, `evaluator_action_lowering_race.py`, `action_commutator.py`, `nerv_long_run_launch_gate.py`, `hinerv_scorer_bootstrap_action_effect.py` | **REUSE-AS-IS** as the atlas's per-atom score-unit record. Heavily CONSUMED, very fresh. The atlas engine is a *producer/index over* these rows — build the index, reuse the row. |
| `src/tac/optimization/evaluator_action_waterfill.py` | **Canonical atom admission law** `admit σ vs base P iff S(P+σ)<S(P)`; `waterfill_select_actions` (l.188), `action_commutator` (l.176); extincts the 2026-06-08 sidecar phantom-base incident (l.1–22) | **2026-06-08** | `evaluator_action_lowering_race.py`, `snerv_official_source_forward_harness.py`, `snerv_lf_hf_replacement_queue.py` | **REUSE-AS-IS** — the atlas's atom-selection law. Brand new, on-point. |
| `src/tac/cathedral_consumers/per_pair_difficulty_atlas_consumer/` (PKG) | per-pair difficulty atlas (cathedral consumer) | (cathedral) | **EXTEND** — existing atlas consumer surface; the engine should emit into this. |
| `src/tac/analysis/scorer_spectral_sensitivity_v2.py` | `scorer_spectral_sensitivity.v2` atlas: H_seg/H_pose per band×orientation×amplitude×channel×frame-incidence cell (the sensitivity source the LF law reads) | 2026-05-x | `lf_payload_rate_distortion.py`, `constants_provenance_manifest.py` | **REUSE-AS-IS** as the atlas engine's measured-sensitivity backbone. |
| `src/tac/analysis/inverse_scorer_actions.py`, `action_commutator.py`, `evaluator_action_lowering_race.py`, `support_codec_router.py` | inverse-action enumeration / noncommutative composition / lowering race / support-codec routing | recent | EXTEND — these are the atlas's action-vocabulary surfaces. |
| `src/tac/contest_exploits/` (19 modules) | problem-space exploit registry: `a1_specialized_inverter.py` (VQ packet), `per_class_chroma_anchor.py`, `tropical_argmax_boundary_grammar.py`, `stable_orbit_packet_diet.py`, `decoy_mosaic_residual_basis.py`, `precomputed_inference_outputs.py`, `pair_index_lookup_table.py`, `deterministic_scorer_exploit.py`, … | 2026-05-17/18 | mixed (a1_specialized_inverter CONSUMED by 4 tools; per_class_chroma_anchor ORPHAN) | **EXTEND/HARVEST** — this is the existing "score-effect codebook" vocabulary. The atlas engine should index these as canonical atom families rather than inventing a parallel registry. |

### Task #47 — Null-space / invisibility basis

| Surface | Computes (file:line) | Last commit | Consumers | Task verdict |
|---|---|---|---|---|
| `src/tac/null_space_exploiter/` (PKG: `core.py` 16.4K) | byte-space null basis: per-pair master-gradient `(n_bytes,n_pairs,n_axes)` → constraint matrix (scorer-axis sensitivities × byte offsets) → orthonormal null basis of byte perturbations with small first-order score response. `build_null_space_basis` (l.235), `compute_null_space_basis` (l.219), `plan_null_space_byte_reduction` (l.334), `apply_raw_byte_splice_reduction` (l.384), `project_modifications_onto_null_space` (l.285) (core.py l.1–15) | 2026-05-18 | **`src/tac/unified_action.py`** (l.533 `from tac.null_space_exploiter import build_null_space_basis`; l.574 builds; l.654 stores `null_space_basis`) | **REUSE-AS-IS.** This IS task #47's byte-space invisibility basis, already wired into the unified action. Build #47 as a thin extension (e.g. higher-order / pixel-domain basis) on top; do not re-derive byte-space null. |
| `src/tac/optimization/scorer_inverse_decision_surface.py` | inverse decision surface: scorer-response rows → "what SegNet/PoseNet/rate infer for free, sufficient statistics, fragile boundaries"; `build_inverse_scorer_decision_surface` (l.41); null/fragile delta thresholds (l.38–39) | 2026-05-23 | `byte_shaving_signal_surface_builder.py`, `tools/build_inverse_steganalysis_action_functional.py` | **REUSE-AS-IS** — the decision-surface complement to the null basis (fragile-boundary identification). |
| `src/tac/canonical_equations/null_space_byte_fraction.py` | canonical equation: null-space byte fraction | — | REUSE the equation anchor. |
| `src/tac/substrates/nscs01_nullspace_split_renderer/` (PKG) | renderer that splits null vs informative subspace | (substrate) | reference/EXTEND. |
| `src/tac/xray/bilinear_resize_nullspace.py` | bilinear-resize null space (eval preprocess invisibility) | — | REUSE — invisibility under the eval resize is exactly the contest's uint8/resize survival lens. |

### Task #35 — Frame1 joint-safe-cone (perturb frame1 without harming seg AND pose)

| Surface | Computes (file:line) | Last commit | Consumers | Task verdict |
|---|---|---|---|---|
| `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/` (PKG: architecture/archive/inflate/trainer/tier_c_hook + mlx_to_numpy_bridge) | **Frame-1 SegNet water-fill cascade** — a full substrate already targeting the frame1 axis | 2026-05-30 | (substrate) | **EXTEND** — task #35 adds the *joint* (seg AND pose) safe-cone; this surface already owns the frame1+SegNet half. Add the P19 pose-null constraint from `joint_p18_p19_waterfill`. |
| `tools/build_p19_posenet_null_pairs.py` + `.omx/research/cascade_c_posenet_null_segnet_region_waterfill_*_landed_20260526.md` | P19 PoseNet-null pair selection + landed cascade-C posenet-null × segnet-region per-region codec | 2026-05-26 | research/tool | **REUSE-AS-IS** — the pose-null half of the joint cone already exists and landed. Task #35 = intersect this (pose-null) with `cascade_c_prime_frame_1_segnet_waterfill` (seg-safe). |
| `src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/` (PKG) | pose-axis null projection applied on SegNet for motion-pair repair | (composition) | **REUSE** — exactly "pose-null perturbation that stays seg-safe", the joint-cone primitive. |
| `src/tac/analysis/segnet_boundary_marginals.py` | per-pixel SegNet argmax-flip boundary marginals (the seg-safe constraint surface) | 2026-05-14 | `segnet_semantic_bridge.py` + 2 tools | REUSE — seg-safe boundary half of the cone. |
| `src/tac/optimization/jacobian_fisher_importance_allocator.py` | Jacobian/Fisher importance allocator | 2026-05-17 | `optimization/cooperative_receiver_integration.py` | REUSE — pose-Jacobian importance for the pose-safe half. |

### Cross-cutting: the frontier driver (master-gradient family + fec6 89%/90% seg-share finding)

| Surface | Computes | Observed value | Reproduce |
|---|---|---|---|
| `.omx/state/master_gradient_anchors.jsonl` (11 rows) | per-pair fp64 score-axis dominance backfill, formula `abs(grad_axis)·marginal / Σ` | seg **threshold-dominant byte count 161,779 / 178,158 = 90.8%** (the "~89%/90% seg-share" finding); mean_axis_share seg≈0.708, pose≈0.202, rate=0.0; marginal coeffs seg=100.0, pose≈37.8, rate≈6.66e-7; pose-dominant bytes only 346 | `wc -l .omx/state/master_gradient_anchors.jsonl` → 11; ledger row `master_gradient_score_axis_dominance_v1` |
| `src/tac/master_gradient_consumers.py` (227 KB) + `master_gradient_*.py` (~25 modules) + `tools/extract_master_gradient.py` (138.8K), `master_gradient_xray.py` (99.8K) | the fp64 per-pair sensitivity producer/consumer stack that drives the 0.19199 frontier | 25+ modules, ~20 dedicated tests, last commit 2026-05-19 | `find src/tac -name 'master_gradient_*.py' \| grep -v test \| wc -l` |
| `src/tac/contest_exploits/rate_attack_autopilot_features.py` + `tools/build_frontier_rate_attack_feedback_refresh.py` (104K), `run_frontier_rate_attack_feedback_cycle.py` (64K), `run_repair_campaign_autonomous_floor_loop.py` (92.8K) | the offline-sweep / final-rate-attack / repair-campaign loop that produced/refines the frontier | huge live tooling | `ls tools/ \| grep -i 'rate_attack\|repair'` |

**Note on the 89% figure:** MEMORY.md cites "fec6 frontier 6bae0201 = 89% seg share gini 0.60". The live ledger (`b7106c9b` lineage) shows **90.8% threshold-dominant seg bytes / 70.8% mean seg-share**. Both are the same finding (seg dominates the byte-sensitivity budget). The bit-allocator priors for ALL four tasks should start from this: **seg-flip protection is the binding constraint; pose-null/rate-shave is where free bytes live.**

---

## B. The `repair_*` family (task: rate-attack-repair) — already a complete campaign

`src/tac/optimization/repair_*.py` = ~18 modules + `tools/*repair*` = ~30 tools. This is a **full landed campaign**, not orphaned: `repair_campaign_posterior.py`, `repair_campaign_scorer.py`, `repair_family_byte_transform_executor.py`, `repair_family_stack_search.py`, `boundary_repair_runtime_materializer.py`, `repair_entropy_coder_runtime_adapters.py`, `exact_ready_axis_repair.py`. Verdict for any "rate-attack-repair" task: **REUSE-AS-IS / EXTEND the existing campaign** — do not start a new repair lane. Entry points: `tools/run_repair_campaign_autonomous_floor_loop.py` (92.8K), `score_repair_campaign.py`.

---

## C. REUSE PLAN per task (so build subagents IMPORT, not duplicate)

### Task #35 — frame1 joint-safe-cone
1. **Base:** extend `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/` (seg-safe frame1 half already exists).
2. **Pose-null half:** import `tools/build_p19_posenet_null_pairs.py` output + `src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/` (pose-axis null projection ON segnet = the joint primitive).
3. **Coupled weight:** import `joint_p18_p19_waterfill.py` (`w_i = 100·|dL_seg| + 5/sqrt(10·d_pose)·‖J_pose,i‖_{Σ⁻¹}`) — it already computes the joint seg+pose constraint. The "joint-safe cone" = atoms with low w_i (both seg-flip-safe AND pose-null).
4. **Seg-safe surface:** `segnet_boundary_marginals.py`. **Pose-safe surface:** `jacobian_fisher_importance_allocator.py`.
   → **New code is the intersection logic + frame1-specific archive grammar only.** The two half-cones already exist.

### Task #36 — atlas engine
1. **Atom score-unit row:** import `tac.analysis.action_effect.ActionEffect` / `EvaluatorActionEffect` (the IR; `compute_delta_scores` → `score_geometry.contest_score`). Do NOT create a new score row.
2. **Atom admission law:** import `tac.optimization.evaluator_action_waterfill.waterfill_select_actions` + `action_commutator` (canonical `S(P+σ)<S(P)` law; extincts the 2026-06-08 phantom-base bug).
3. **Atom vocabulary:** index the 19 `contest_exploits/*.py` families + `analysis/inverse_scorer_actions.py` as the atlas's canonical atoms; emit into `cathedral_consumers/per_pair_difficulty_atlas_consumer/`.
4. **Sensitivity backbone:** `scorer_spectral_sensitivity_v2.py` (the measured H_seg/H_pose atlas).
   → **New code is the engine/index that materializes + ranks atoms; the row, the law, the vocabulary, and the sensitivity source all exist.**

### Task #46 — LF rate-distortion waterfiller
1. **REUSE-AS-IS:** `src/tac/optimization/lf_payload_rate_distortion.py` (committed 2026-06-09) IS this task — the evaluator-conditioned reverse-waterfill with the exact admission law. Import `keep_component`, `estimate_section_sensitivity`, `delta_distortion_score`, `CandidateActionEvaluation`.
2. **Coupled weight / region granularity:** `joint_p18_p19_waterfill.py` + `scorer_region_waterfill.py`.
3. **Subband kernel (if wavelet-banded):** `z8_hierarchical_predictive_coding/per_subband_rd_waterfill_solver.py`.
4. **Numerical projection kernel:** `solvers/numba_jit_water_filling.py` / `water_filling_codec_v2.py`.
   → **Building this fresh would duplicate a 2-day-old live module. Wire the existing planner to the LF payload producer + exact re-measure; do not re-author the law.**

### Task #47 — null-space / invisibility basis
1. **REUSE-AS-IS:** `src/tac/null_space_exploiter/` (`build_null_space_basis`, `plan_null_space_byte_reduction`, `apply_raw_byte_splice_reduction`) — byte-space invisibility basis, already wired into `unified_action.py`.
2. **Fragile-boundary complement:** `scorer_inverse_decision_surface.py` (`build_inverse_scorer_decision_surface`).
3. **Eval-preprocess invisibility:** `xray/bilinear_resize_nullspace.py` (null under the contest resize).
4. **Equation anchor:** `canonical_equations/null_space_byte_fraction.py`.
   → **New code (if any) is a higher-order or pixel-domain basis extension; byte-space null + its unified-action wire-in already exist.**

---

## D. Orphan status summary

- **CONSUMED (do not rebuild):** `lf_payload_rate_distortion`, `joint_p18_p19_waterfill`, `action_effect`, `evaluator_action_waterfill`, `null_space_exploiter` (→unified_action), `scorer_inverse_decision_surface`, `segnet_boundary_marginals`, `jacobian_fisher_importance_allocator`, `scorer_region_waterfill`, `a1_specialized_inverter`, the entire `master_gradient_*` + `repair_*` families.
- **ORPHAN (no non-test importer — candidates for revival, NOT deletion):** `contest_exploits/per_class_chroma_anchor.py` (2026-05-17); several older `contest_exploits` atoms (`tropical_argmax_boundary_grammar`, `decoy_mosaic_residual_basis`, `stable_orbit_packet_diet`) have tools but no module importer; `scorer_exploits.py` / `saliency_inversion.py` / `score_gradient_param_saliency.py` (2026-05-14, test-only importers).
- **PHANTOM (does not exist):** `src/tac/sensitivity_map.py` — it is a **package** `src/tac/sensitivity_map/` (42 files), not a file. (MEMORY.md previously flagged the file path as DNE; the package is the real surface.)
- **GENUINELY NEW (no existing surface):** frame1 *joint* safe-cone *intersection* logic (the two half-cones exist separately); the atlas *engine/index* (the row+law+vocabulary exist). No `safe_cone`/`frame1`-named module exists yet.

---

## E. Audit-provenance index (reproduce commands)

- File existence: `ls -la src/tac/{scorer_exploits,saliency_inversion,…}.py`
- Surface enumeration: `find src/tac -maxdepth 2 \( -name '*sensitivity*' -o -name '*saliency*' -o … \) -type f | grep -v pycache`
- Consumers: `grep -rl "<module> import\|import.*<module>" src/tac tools experiments --include='*.py' | grep -v test_ | grep -v pycache`
- Last commit: `git log -1 --format='%ai' -- <file>`
- Frontier: `cat .omx/state/canonical_frontier_pointer.json`
- Seg-share: `.omx/state/master_gradient_anchors.jsonl` row `master_gradient_score_axis_dominance_v1`, field `threshold_dominant_axis_counts`
- Waterfill family: `find src/tac -iname '*waterfill*' -o -iname '*water_fill*' | grep -v pycache | grep -v test_`
