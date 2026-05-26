# Cascade C' subagent A — MLX-FIRST TRAINER + BRIDGE + TIER-C HOOK LANDED

- **subagent_id**: `cascade-c-prime-frame-1-segnet-waterfill-substrate-A-mlx-first-trainer-substrate-engineering-sister-nscs06-v8-20260526`
- **lane_id**: `lane_cascade_c_prime_option_a_build_scaffold_20260526` (extends)
- **date_utc**: 2026-05-26T21:53:00Z
- **scope**: per-substrate symposium PROCEED_WITH_REVISIONS revision #1 (MLX-first trainer landing) + sister deliverables (MLX→numpy bridge, Tier-C MDL ablation hook) per CLAUDE.md standing directive 2026-05-26 *"MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL"*
- **execution_substrate**: macOS M5 Max MLX-LOCAL (NO PAID DISPATCH per CLAUDE.md "Executing actions with care")
- **horizon_class** per Catalog #309: `frontier_pursuit` (sub-PR111 PLATEAU-BREAK candidate pending paired-CUDA validation)
- **mission_contribution** per Catalog #300: `frontier_breaking_enabler` (unblocks revision #1; sister subagent B + C now structurally able to fire inflate.sh + paired-CUDA Modal smoke)
- **cost**: $0 GPU + ~60 min wall-clock + 0 paid dispatches

## Landing summary

| # | Module | LOC | Status | Verification |
|---|---|---|---|---|
| 1 | `trainer.py` MLX-first compress-pass | 585 | LANDED | MLX-native primitives (mlx.core.random.normal + mlx.eval); ZERO torch imports; non-promotable provenance per Catalog #127/#192/#317/#341 |
| 2 | `mlx_to_numpy_bridge.py` state_dict bridge | 243 | LANDED | npz export/load roundtrip + archive byte-identity per Catalog #139 |
| 3 | `tier_c_hook.py` Tier-C ablation probe request builder | 232 | LANDED | canonical `tools/mdl_scorer_conditional_ablation.py --tier c` CLI emission; FORMALIZATION_PENDING per Catalog #344 |
| 4 | `tests/test_trainer.py` | 396 | LANDED | 39/39 PASS (5 no-torch-imports + 6 config invariants + 7 non-promotable provenance + 5 MLX-local smoke + 5 bridge roundtrip + 8 tier_c_hook + 3 contract preservation) |
| 5 | `__init__.py` re-exports updated | 169 | UPDATED | 21 symbols re-exported; backwards compat preserved for existing 1 export |

**Total package**: 11 files / 2511 LOC (up from ~1495 LOC pre-landing).

## MLX-LOCAL smoke verdict — PASS

- **MLX availability**: `mlx.core` import successful on M5 Max
- **Per-pair Lagrangian dual routing**: converges in single argmin pass (sister of `architecture.py::compute_per_pair_lagrangian_dual_routing`)
- **Atick-Redlich invariant** (frame-0 SegNet penalty STRUCTURALLY 0.0): verified in `test_smoke_atick_redlich_frame_0_seg_invariant`
- **Determinism given seed**: verified in `test_smoke_deterministic_given_seed`
- **Per-pair improvement invariant** (joint Lagrangian never worse than frame-0-only): verified in `test_smoke_routing_invariants`
- **MLX→numpy bridge round-trip byte-identical**: verified via `test_bridge_archive_roundtrip_byte_identical` (sha256 deterministic; routing_decision array-equal)
- **Tier-C ablation probe request canonical CLI**: emits `tools/mdl_scorer_conditional_ablation.py --tier c` per Catalog #324
- **No-torch-import structural invariant**: all 3 new modules (trainer, bridge, tier_c_hook) verified text-grep-clean

Sample 600-pair MLX-local smoke at synthesis-realistic scales (seed=20260526, perturbation_scale_seg=1e-6, perturbation_scale_pose=3e-5): `frame_1_pct=1.00%` (6 of 600 pairs route to frame-1 menu); `total_score_delta=-2.94e-02` [macOS-MLX research-signal]; archive_bytes_delta=85 bytes.

**This smoke is NOT a contest score claim.** Per CLAUDE.md "MLX portable-local-substrate authority" + the per-substrate symposium PROCEED_WITH_REVISIONS verdict + Contrarian + Atick dissent: the synthesis -0.058820 prediction remains CARGO-CULTED until sister subagent C lands paired-CUDA validation per revision #3. Even 10× literature overestimate (yielding -0.006) remains PR111-PLAUSIBLE.

## Canonical-vs-unique decision per layer (per Catalog #290; substrate-engineering exception per HNeRV parity L7)

| Layer | Decision | Rationale |
|---|---|---|
| Perturbation enumeration | UNIQUE (MLX-native) | Atick-Redlich joint frame-0 + frame-1 mode menu has no canonical sister |
| Per-pair Lagrangian dual | CANONICAL (`architecture.py`) | already landed in scaffold; closed-form O(n × m) argmin |
| Score-aware axis decomposition | CANONICAL stencil (`tac.substrates.score_aware_common.score_pair_components`) | per-axis seg + pose + bytes mirror per Catalog #356 |
| Archive emission | CANONICAL (`archive.py`) | CH-CCP-FRAME1-WATERFILL grammar already landed |
| Bridge contract | UNIQUE (`mlx_to_numpy_bridge.py`) | per-substrate state_dict shape; no canonical sister |
| Hardware substrate detection | CANONICAL (`tac.substrates._shared.trainer_skeleton.detect_hardware_substrate`) | per Catalog #190; macos_arm64 axis="cpu" token |
| Non-promotable provenance | CANONICAL (sister to NSCS06 v8 `MLX_NON_PROMOTABLE_PROVENANCE`) | per Catalog #127/#192/#317/#341 |
| Tier-C MDL ablation tool routing | CANONICAL (`tools/mdl_scorer_conditional_ablation.py`) | per Catalog #324 |

## 9-dimension success checklist evidence (per Catalog #294)

| Dim | Evidence |
|---|---|
| 1 UNIQUENESS | MLX-first per-pair Lagrangian dual routing — Atick-Redlich asymmetric channel; class-shift via frame-1 menu |
| 2 BEAUTY+ELEGANCE | trainer 585 LOC reviewable in segments; per-pair routing returned as canonical frozen dataclass |
| 3 DISTINCTNESS | DISTINCT from NSCS06 v8 (chroma LUT) + PR110 (K=16 frame-0-only) + Z6 (predictive coding) |
| 4 RIGOR | PV against scaffold + symposium memo + canonical equation #344 proposal; 39 dedicated tests |
| 5 OPTIMIZATION-PER-TECHNIQUE | per-pair Lagrangian dual + Atick-Redlich frame-0 invariant + waterfill bit allocator IS substrate-optimal |
| 6 STACK-OF-STACKS-COMPOSABILITY | composable as PR111-sub-frontier candidate atop PR110; sister-compatible with NSCS06 v8 chroma_lut |
| 7 DETERMINISTIC-REPRODUCIBILITY | MLX seed pinned (mx.random.seed) + numpy seed pinned; closed-form argmin deterministic |
| 8 EXTREME-OPTIMIZATION-PERFORMANCE | closed-form O(n_pairs × n_modes_joint); MLX-vectorized; ~31ms for 600-pair smoke |
| 9 OPTIMAL-MINIMAL-CONTEST-SCORE | PROVISIONAL-PENDING-VERIFICATION per Catalog #363 (paired-CUDA gate per revision #3) |

## Cargo-cult audit per assumption (per Catalog #303)

| Assumption | Classification | Unwind plan |
|---|---|---|
| MLX SegNet/PoseNet forward parity with PyTorch ≤2% argmax flip per #1258 | HARD-EARNED | sister `nscs06_v8_chroma_lut.mlx_iteration.verify_mlx_segnet_argmax_parity_with_torch` already exists |
| Random Gaussian perturbation menu approximates Pareto frontier of per-pair codec modes | CARGO-CULTED | 7th-order subagent: PR110-K=16-Huffman-codebook-aligned perturbation menu; PV-sufficient for MLX-local smoke per symposium PROCEED |
| Per-pair Lagrangian dual converges in single argmin pass | HARD-EARNED | architecture.py invariant + Catalog #355 wire-in unit tests |
| MLX state_dict → numpy bridge round-trip preserves byte-identity | HARD-EARNED via round-trip smoke | `test_bridge_archive_roundtrip_byte_identical` PASS |
| Tier-C MDL ablation hook predicts contest-CUDA across-class verdict at MLX-local | CARGO-CULTED | Catalog #324 post-training Tier-C validation defers empirical to sister subagent C paired-CUDA |
| `MLXFirstTrainerConfig.perturbation_scale_seg=1e-5` is sufficient for synthesis-realistic frame_1 routing | CARGO-CULTED | 7th-order: replace with PR110-empirical-measured per-mode scales; current scale is PR106-frontier-tuned |

## Observability surface (per Catalog #305)

- **inspectable per layer**: trainer emits PerPairRoutingDecision + JSON sidecar; bridge emits BridgeRoundtripVerdict; tier_c hook emits TierCAblationHookVerdict
- **decomposable per signal**: MLXFirstTrainerVerdict.routing surfaces selected_seg_delta + selected_pose_delta + predicted_archive_bytes_delta per Catalog #356
- **diff-able across runs**: deterministic given (seed, n_pairs, n_modes_per_frame, perturbation_scale); test_smoke_deterministic_given_seed verifies
- **queryable post-hoc**: JSON sidecar at `<output_dir>/mlx_first_compress_pass_verdict.json`; bridge npz files
- **cite-able**: provenance carries (subagent_id, lane_id, substrate_id, run_utc, canonical_equation_proposal)
- **counterfactual-able**: byte-mutation smoke per archive.py (existing) + bridge round-trip invariant (this landing)

## Predicted ΔS band (per Catalog #296)

| Status | Band | Validation |
|---|---|---|
| MLX-LOCAL synthesis (Cascade C') | -0.058820 [macOS-MLX research-signal] | Dykstra-feasibility per Cascade C' synthesis 48-cell sweep |
| Paired-CUDA expected | PROVISIONAL-PENDING-VERIFICATION | Contrarian + Atick dissent: 10-30× literature overestimate common |

Per Catalog #324: ``predicted_band_validation_status: pending_post_training``. Reactivation criterion: post-training Tier-C re-measurement on landed paired-CUDA smoke archive sha via `tools/mdl_scorer_conditional_ablation.py --tier c`.

## Canonical equation #344 anchor status

`atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1` remains **FORMALIZATION_PENDING** per Catalog #344 sister discipline. Registry count: 52 (unchanged this landing; sister subagent C registers the anchor via canonical helper post-paired-CUDA empirical per registration tool pattern commits `7ab5f58ae` + `04f34ea40`).

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: ACTIVE-research-signal (per-pair perturbation matrices IS the sensitivity surface; consumable by `tac.sensitivity_map.*`)
- **hook #2 Pareto constraint**: ACTIVE-research-signal (per Catalog #356 per-axis decomposition surfaced via MLXFirstTrainerVerdict)
- **hook #3 bit-allocator**: ACTIVE (per-pair Lagrangian dual routing decision IS bit allocator; sister of waterfill primitive in `architecture.py`)
- **hook #4 cathedral autopilot dispatch**: PROPOSED-pending-paired-CUDA per Catalog #335 contract (sister subagent C adds canonical_equation_lookup_consumer registration post-validation)
- **hook #5 continual-learning posterior**: ACTIVE (MLX-local smoke results carry MLX_NON_PROMOTABLE_PROVENANCE; sister consumers append via `tac.council_continual_learning.append_council_anchor`)
- **hook #6 probe-disambiguator**: ACTIVE (Tier-C MDL ablation hook IS the canonical disambiguator between within-class refinement vs class-shift; CLI emission ready for sister subagent C)

## Per-substrate symposium revision status

| # | Revision | Status |
|---|---|---|
| 1 | MLX-first trainer landing | **LANDED** (this subagent A) |
| 2 | inflate.sh 3-arg signature wrapper | DEFERRED to sister subagent B |
| 3 | Paired-CUDA Modal T4 smoke | DEFERRED to sister subagent C (PAID; operator-decision-required) |
| 4 | Paired-CPU Linux x86_64 anchor | DEFERRED to sister subagent C (PAID) |
| 5 | Canonical equation #344 anchor registration | DEFERRED to sister subagent C (post paired-CUDA empirical) |

**1 of 5 revisions LANDED + 4 of 5 DEFERRED** per operator-routable next step.

## Discipline closure

- **Catalog #229 PV**: read substrate scaffold (6 files) + symposium memo + 5 standing-directive memos + sister NSCS06 v8 mlx_iteration.py + canonical scorer-loss helper + trainer_skeleton.detect_hardware_substrate
- **Catalog #117/#157/#174 canonical serializer + POST-EDIT --expected-content-sha256**: used for final commit
- **Catalog #287 placeholder-rationale rejection**: every waiver ≥4 chars + substantive
- **Catalog #119 Co-Authored-By Claude trailer**: emitted via canonical serializer (auto-appends per Catalog #119)
- **Catalog #206 checkpoints**: 2 in_progress + 1 complete emitted via `tools/subagent_checkpoint.py`
- **Catalog #230 sister-subagent ownership map**: scope DISJOINT from any active sister (per pre-flight check)
- **Catalog #340 sister-checkpoint guard PROCEED**: pre-commit
- **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: NEW modules + tests only; ZERO mutation of existing scaffold files (substrate_contract / architecture / archive / inflate / test_scaffold_smoke)
- **Catalog #208 no local-paths**: all paths repo-relative
- **Catalog #343 no hardcoded score literals**: synthesis predictions tagged `[macOS-MLX research-signal]` per Catalog #323
- **Catalog #344 canonical equation registry**: remains 52 (FORMALIZATION_PENDING; sister subagent C registers post-empirical)
- **Catalog #346 canonical roster complete=True**: N/A this landing (no NEW T2+ deliberation invoked; reuses per-substrate symposium PROCEED_WITH_REVISIONS)

## Operator-routable next step

1. **Sister subagent B**: build `scripts/remote_lane_substrate_cascade_c_prime_frame_1_segnet_waterfill.sh` + `scripts/inflate_cascade_c_prime_frame_1_segnet_waterfill.sh` (~50 LOC each sister to NSCS06 v8 driver template); honor Catalog #244 canonical NVML block + Catalog #163 canonical sentinel
2. **Sister subagent C**: paired-CUDA Modal T4 smoke via `tools/operator_authorize.py --recipe substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch --target modal` per CLAUDE.md "Executing actions with care"; smoke-before-full per Catalog #167 + paired-CUDA per Catalog #246; ~$0.30-0.50; **OPERATOR-DECISION-REQUIRED**
3. **Post-empirical**: sister subagent C registers canonical equation #344 anchor via sister registration tool pattern
4. **Post-Tier-C**: per-substrate symposium re-convenes for PROCEED-unconditional verdict per Catalog #325 + #324

## Cross-references

- Predecessor landing memo (scaffold): `.omx/research/cascade_c_prime_option_a_build_scaffold_landed_20260526.md` (commit `aaf0b1eb6`)
- Per-substrate symposium: `.omx/research/council_t2_cascade_c_prime_frame_1_segnet_waterfill_per_substrate_symposium_20260526.md`
- Canonical equation #344 anchor proposal: `.omx/research/cascade_c_prime_canonical_equation_344_anchor_proposal_20260526.md`
- Sister NSCS06 v8 MLX iteration reference: `src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py`
- Substrate package: `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/`
- Operator-authorize recipe: `.omx/operator_authorize_recipes/substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch.yaml`
- CLAUDE.md "MLX portable-local-substrate authority" + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "Meta-Lagrangian/Pareto solver" non-negotiables
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons L1-L13 + L7 substrate-engineering exception
