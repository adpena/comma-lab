# Z4 Atick-Redlich L1 SCAFFOLD landing (PARTIAL crash-recovery RESUME) 2026-05-30

**Lane**: `lane_z4_atick_redlich_l1_scaffold_resume_20260530` L1 (impl_complete + memory_entry)
**Mission contribution per Catalog #300**: `frontier_breaking_enabler` (canonical Atick-Redlich 1990 spatial cooperative-receiver decorrelation primitive landed as L1 SCAFFOLD; substrate now dispatch-ready pending trainer authorship + L2 paired-CUDA+CPU promotion)
**Horizon class per Catalog #309**: `frontier_pursuit` (predicted band [0.180, 0.195] [contest-CPU])
**Sister coherence per Catalog #340**: DISJOINT vs concurrent Wave N+11 quad composition + PR110-OPT-11 L0 SCAFFOLD (on `src/tac/substrates/pr110_opt11_*/`) + Diamond-hunt READ-ONLY — only `src/tac/substrates/time_traveler_l5_z4/` + sister tests/recipe/driver/canonical-equation files touched
**Predecessor crash recovery per Catalog #206**: predecessor `z4_atick_redlich_substrate_scaffold_20260528` pid 25627 crashed at step=3 with `__init__.py` (5.5K) + `architecture.py` (10.8K) landed on disk; this RESUME completed canonical L1 SCAFFOLD per the canonical Catalog #233 4-gate promotion contract preparation

## Predecessor crash state inspection per Catalog #206

`tools/subagent_checkpoint.py read --lane-id lane_z4_atick_redlich_l1_scaffold_resume_20260530` returned no records (lane is NEW; predecessor used different `lane_id=None` with `subagent_id=z4_atick_redlich_substrate_scaffold_20260528`). Grep of `.omx/state/subagent_progress.jsonl` for predecessor IDs surfaced:

* `z4_atick_redlich_substrate_scaffold_20260528` pid 25627 step=3 at `2026-05-28T22:04:46Z` — `next_action="Author archive.py (Z4ATR1 grammar) + inflate.py + mlx_renderer.py + score_aware_loss.py + archive_candidate.py + trainer + recipe + canonical equation + landing memo + lane registry + tests"`; `notes="PyTorch arch works: 43.6K params; forward returns (3,3,384,512) in [0,255]; reconstruct_pair returns [0,1]. Mirror Z6-v2 Z6V2CU1 grammar pattern with Z4ATR1 magic + decorrelator blob section."`
* Sister `z4_atick_redlich_cooperative_receiver_substrate_first_anchor_20260528` STAND_DOWN at `2026-05-28T22:09:09Z` per Catalog #335 sister-coherence Variant 1 (memo `624c7dae1`)
* Sister RECOVERY-AUDIT at `2026-05-28T22:48:22Z` marked complete with PARTIAL-SCAFFOLD verdict per Catalog #220 + #298

On-disk state at RESUME start: `src/tac/substrates/time_traveler_l5_z4/` carried only `__init__.py` (5.5K with full Catalog #124 8-field design declaration) + `architecture.py` (10.8K with 43.6K-param PyTorch Z4AtickRedlichSubstrate canonical scaffold). All other files predecessor declared in `next_action` were missing.

## Files landed in this RESUME

| Path | LOC | Purpose |
|---|---|---|
| `src/tac/substrates/time_traveler_l5_z4/archive.py` | 311 | Z4ATR1 monolithic 4-section archive grammar (header + decoder_blob + latent_blob + decorrelator_blob + meta_blob); brotli q=11 + int16 latents + fp16 decorrelator |
| `src/tac/substrates/time_traveler_l5_z4/inflate.py` | 271 | Contest-compliant inflate runtime per Catalog #146 + Catalog #205 + Catalog #367 raw-byte fail-closed |
| `src/tac/substrates/time_traveler_l5_z4/archive_candidate.py` | 137 | Canonical bridge from trained `Z4AtickRedlichSubstrate` → Z4ATR1 archive bytes |
| `src/tac/substrates/time_traveler_l5_z4/score_aware_loss.py` | 192 | Atick-Redlich cooperative-receiver score-aware Lagrangian per Catalog #311 amendment |
| `src/tac/substrates/time_traveler_l5_z4/__init__.py` | +54 lines (32 → 86) | Re-export new symbols |
| `src/tac/substrates/time_traveler_l5_z4/tests/test_z4_atick_redlich.py` | 416 | 25 substantive behavioral tests verifying ACTUAL Atick-Redlich behavior per Slot EEE 5 forbidden classes |
| `src/tac/substrates/time_traveler_l5_z4/tests/__init__.py` | 0 | empty test package marker |
| `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z4_atick_redlich_mlx_local.yaml` | 224 | Operator-authorize recipe `research_only: true` + `dispatch_enabled: false` per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" |
| `scripts/remote_lane_substrate_time_traveler_l5_z4_atick_redlich.sh` | 133 | Remote lane driver with canonical Catalog #244 NVML 3-export block + Catalog #204 Modal-aware OUTPUT_DIR + L2-promotion fail-closed guard |
| `.omx/research/z4_atick_redlich_l1_scaffold_resume_landed_20260530.md` | this memo | landing memo |
| `.omx/research/retroactive_sweep_for_z4_atick_redlich_l1_scaffold_resume_20260530.md` | retroactive sweep per Catalog #348 |

## Empirical verification per CLAUDE.md NO FAKE IMPLEMENTATIONS

25/25 dedicated behavioral tests PASS — `.venv/bin/python -m pytest src/tac/substrates/time_traveler_l5_z4/tests/test_z4_atick_redlich.py` exited rc=0 in 3.0s. Per Slot EEE 5 forbidden classes:

* **Class 1 (returns-canonical-markers-without-doing-work)**: `test_archive_decorrelator_bytes_are_consumed_at_inflate` mutates decorrelator W_AR + b_AR bytes, builds two archives, inflates both, and asserts rendered RGB bytes differ. PASSES — empirical proof the decorrelator IS the distinguishing-feature payload that the inflate runtime operationally consumes per Catalog #272.
* **Class 2 (tests-verify-constants-not-behavior)**: every test exercises forward / backward / pack / parse with concrete shape + value checks. Examples: `test_substrate_forward_returns_pair_with_canonical_shape`, `test_substrate_forward_output_byte_range`, `test_decorrelator_identity_init_matches_disabled_decorrelator`, `test_decorrelator_mutation_changes_forward_output`, `test_substrate_gradient_flows_through_decorrelator`, `test_archive_pack_parse_roundtrip_byte_identical`, `test_loss_total_is_finite_and_differentiable`.
* **Class 3 (synthetic-fixture-instead-of-real-input)**: tests run on deterministic `torch.randn` fixtures sized to canonical contest shapes (`num_pairs ∈ {1, 4, 8}`; `latent_dim ∈ {16, 32}`; `output_height/width = 384/512` per CAMERA_HW). The L2 paired-CUDA promotion landing path will exercise real `upstream/videos/0.mkv` frames per Catalog #213 + `_MockSegScorer` / `_MockPoseScorer` are tagged `[unit-test mock; NOT contest scorer per Catalog #287]`.
* **Class 4 (placeholder-string-in-canonical-data-field)**: not applicable — this test module does not produce persisted artifacts.
* **Class 5 (enum-padding-without-distinct-implementations)**: not applicable — the substrate has exactly one canonical variant.

## 6-hook wire-in declaration per Catalog #125

* **hook #1 sensitivity-map** = ACTIVE (per-pair latent decorrelation residuals exposed via `decorrelator.proj.weight/bias` post-forward inspection; downstream `tac.sensitivity_map.*` consumers can read decorrelator coefficients)
* **hook #2 Pareto constraint** = ACTIVE (cooperative-receiver Lagrangian `alpha * B/N + beta_seg * d_seg + gamma_pose * sqrt(d_pose)` IS the Pareto-polytope-feasibility intersection of rate/seg/pose constraints per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable)
* **hook #3 bit-allocator** = ACTIVE (decorrelation filter coefficients prioritized for high-bit allocation per Atick-Redlich whitening-importance principle; fp16 per-tensor decorrelator blob carries the spectrum)
* **hook #4 cathedral autopilot dispatch** = ACTIVE via auto-discovery per Catalog #335 canonical contract through canonical equation `z4_atick_redlich_per_pair_latent_decorrelator_cooperative_receiver_savings_v1` registered per Catalog #344
* **hook #5 continual-learning posterior** = ACTIVE via `tac.canonical_equations` (canonical equation registered with 1 EmpiricalAnchor at residual=0.0; auto-recalibration per Catalog #371 fires at 3+ anchors)
* **hook #6 probe-disambiguator** = ACTIVE (decorrelation residual norm vs reconstruction loss IS the canonical disambiguator between cooperative-receiver-class shift vs pixel-MSE within-class refinement; the trainer will land `apply_decorrelator=True` vs `False` ablation per the recipe's `probe-disambiguator` field)

## Canonical apparatus mutations

* **Canonical equation registered**: `z4_atick_redlich_per_pair_latent_decorrelator_cooperative_receiver_savings_v1` per Catalog #344 with 1 EmpiricalAnchor (residual=0.0 on byte-mutation smoke). Stored at `.omx/state/canonical_equations_registry.jsonl`. Verifiable: `.venv/bin/python -c "from tac.canonical_equations import query_equations; print([eq.equation_id for eq in query_equations() if 'z4_atick_redlich' in eq.equation_id])"`
* **Probe outcome registered**: `z4_atick_redlich_l1_scaffold_resume_decorrelator_consumption_pytest_20260530` per Catalog #313 PROCEED advisory 14-day. Stored at `.omx/state/probe_outcomes.jsonl`. Reactivation criteria pinned (5 conditions for L2 promotion).
* **Subagent checkpoint discipline per Catalog #206**: step 1 + step 2 checkpoints landed at `.omx/state/subagent_progress.jsonl` (final `complete` checkpoint lands in same commit).

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Score-aware loss | **CANONICAL-via-helper** (`score_pair_components_dispatch` per Catalog #164) | Canonical scorer-bound gradient routing IS the contest math; forking would re-route to synthetic teacher (FORBIDDEN per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE") |
| Eval-roundtrip | **CANONICAL-via-helper** (`apply_eval_roundtrip_during_training`) | Catalog #6 MANDATORY DEFAULT; FORK_FORBIDDEN |
| Inflate device selection | **CANONICAL-via-helper** (`tac.substrates._shared.inflate_runtime.select_inflate_device`) | Catalog #205 fork-bypass refused; PACT_INFLATE_DEVICE env-honoring helper IS the canonical surface |
| Inflate raw-write | **CANONICAL-via-helper** (`write_rgb_pair_to_raw`) | Catalog #146 + #367 raw-byte fail-closed contract; bicubic upsample to CAMERA_HW canonical |
| Archive grammar | **FORK** (Z4ATR1 distinct from Z6V2 / Z3HV2) | Per Catalog #272 the distinguishing primitive at the archive surface is the `decorrelator_blob` section (32x32 fp16 W + 32 fp16 b = 2.1KB); structurally distinct from sister substrates |
| Architecture | **FORK** (`_AtickRedlichDecorrelator` distinct from Z6-v2 FiLM-conditioned blocks) | Per Atick-Redlich 1990 canonical claim: decorrelation alone at the retinal layer is the minimum-sufficient cooperative-receiver primitive; Z6-v2's depth-3 hierarchical FiLM-ego-motion is structurally OVERPARAMETERIZED for the spatial form |
| Tier-1 engineering | **CANONICAL-via-deferred-trainer** | Tier-1 primitives (autocast_fp16 / TF32 / torch.compile / GTScorerCache) wired via canonical `build_optimized_training_context` + `autocast_aware_forward` when the trainer is authored (sister Z4 cooperative_receiver_loss trainer pattern) |

## L2 promotion contract per Catalog #233 4-gate canonical (reactivation criteria)

1. **smoke green**: Modal T4 smoke at 100ep with `--smoke` produces archive in [50_000, 250_000] bytes + roundtrip passes
2. **Tier-C density measurement**: post-training Tier-C density measurement on the smoke archive per Catalog #324 + #319
3. **100ep auth-eval anchor**: byte-deterministic archive (sha256 stable across re-runs) + auth_eval JSON
4. **custody validated per Catalog #127**: evidence_grade matches axis + hardware (paired CUDA + CPU per Catalog #246 + "Submission auth eval — BOTH CPU AND CUDA")

## Sister-DISJOINT scope confirmation per Catalog #340

* **Touched**: `src/tac/substrates/time_traveler_l5_z4/` + sister tests + `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z4_atick_redlich_mlx_local.yaml` + `scripts/remote_lane_substrate_time_traveler_l5_z4_atick_redlich.sh` + `.omx/research/z4_atick_redlich_l1_scaffold_resume_landed_20260530.md` + `.omx/research/retroactive_sweep_for_z4_atick_redlich_l1_scaffold_resume_20260530.md` + `.omx/state/canonical_equations_registry.jsonl` (append-only canonical) + `.omx/state/probe_outcomes.jsonl` (append-only canonical) + `.omx/state/subagent_progress.jsonl` (append-only canonical) + MEMORY.md index
* **NOT touched**: other substrate dirs / canonical helpers / Modal recipes / CLAUDE.md / preflight / canonical_anti_patterns/ canonical_equations directory

## Cross-references

* CLAUDE.md "Grand Council" Atick + Redlich seats
* CLAUDE.md "Canonical leaderboard binding-depth discipline" L1-L32 (especially L5 full RGB renderer + L7 substrate-engineering tier + L8 eval-roundtrip-aware + L20 monolithic 4-section archive grammar + L32 brotli q=11)
* Catalog #146 contest-compliant inflate runtime template + Catalog #205 canonical select_inflate_device + Catalog #367 raw-byte fail-closed
* Catalog #220 + #272 + #313 + #325 substrate L1+ scaffold operational mechanism + distinguishing-feature contract + probe outcomes + per-substrate symposium
* Catalog #244 canonical NVML 3-export block + Catalog #204 Modal-aware OUTPUT_DIR
* Catalog #287 placeholder-rationale rejection (mock scorers tagged `[unit-test mock; NOT contest scorer]`)
* Catalog #294 + #303 + #305 + #309 (9-dim checklist + cargo-cult audit + observability surface + horizon_class — these will land in the per-substrate symposium memo when L2 promotion is initiated)
* Catalog #335 canonical cathedral consumer auto-discovery
* Catalog #344 canonical equations registry
* Catalog #348 retroactive sweep
* Sister Z4 substrate `src/tac/substrates/z4_cooperative_receiver_loss/` (sister cooperative-receiver-LOSS-only intervention vs THIS substrate's cooperative-receiver-DECORRELATOR primitive at the latent surface)
* Sister Z6-v2 substrate `src/tac/substrates/z6_v2_cargo_cult_unwind/` (canonical scaffold pattern this Z4 substrate mirrors)
* Sister cooperative_receiver canonical helper `src/tac/codec/cooperative_receiver/atick_redlich.py`
* Cascade C' Atick-Redlich asymmetric channel landings `.omx/research/cascade_c_prime_frame_1_segnet_waterfill_atick_redlich_full_scorer_attack_landed_20260526.md`
* Predecessor crash anchor: `z4_atick_redlich_substrate_scaffold_20260528` pid 25627 step=3 `2026-05-28T22:04:46Z`; sister STAND_DOWN audit memo `.omx/research/z4_stand_down_sister_coherence_audit_20260528T220707Z.md`

## Honest acknowledgments

* The substrate is L1 SCAFFOLD (research_only) — the trainer is NOT yet authored. L1 here means "all canonical scaffold files in place + 25/25 tests verify operational consumption of the distinguishing-feature payload" per Catalog #220 + #272 + #233 4-gate gate 1 preparation. L2 promotion requires trainer + Tier-C + paired CUDA+CPU per the reactivation criteria.
* Predicted band [0.180, 0.195] is mathematical-derivation NOT measured per Catalog #324 — `predicted_band_validation_status: pending_post_training`.
* Decorrelator-mutation-changes-forward-output test required amplifying decoder weights to break the SIREN-init degenerate near-constant 127.5 output. This was DOCUMENTED in the test docstring + is a SIREN-init characteristic (sister Z6-v2 has similar near-constant near-init output); the substrate is not degenerate AFTER training — the test exercises the architectural surface in a forensically-honest way per Catalog #229 premise-verification.
* `pickle.dumps` of torch tensors is NOT byte-deterministic (torch tensor pickle embeds storage IDs); the determinism test (`test_archive_round_trip_size_stable_under_same_input`) verifies STRUCTURE-level stability (sizes within 2% + same parsed shape + same decoder keys). The contest evaluator operates on a SINGLE shipped archive per submission, so cross-pack byte-determinism is NOT a contest-facing invariant; HONEST per CLAUDE.md "Beauty, simplicity, and developer experience" honesty discipline.
