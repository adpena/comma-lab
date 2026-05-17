# Phase 1b Z6 _full_main lift landed 2026-05-16

**Lane:** `lane_phase_1b_z6_lift_20260516`
**Status:** L1 (impl_complete=true / strict_preflight=false after 2026-05-17 dispatchability correction / deploy_runbook=true)
**Subagent:** `z6_lift_subagent_20260516`
**Predecessor:** none
**Parent directive:** Phase 1b Z6 implementation-lift per the operator's PR95-paradigm directive (Phase 1a sister owns dispatch on STC v2 + NSCS06 v8; LIFT-Rudin sister owns Rudin trainer lift; LIFT-ATW-v2 sister owns ATW v2 lift; this lane owns ONLY Z6).

## What changed

The Z6 trainer at `experiments/train_substrate_time_traveler_l5_z6.py::_full_main` was lifted
from `raise NotImplementedError("Phase 2 council approval required...")` (legacy gate at
commit `97dff03fc`) to a working PR95-paradigm-compliant implementation that binds ALL
ingredients per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" + Catalogs
#310/#311/#312 + the Z6/Z7/Z8 design memo §4.1.

## Premise verification per Catalog #229 (6 pre-edit anchors)

1. **PV-1**: `_full_main` (line 686-737 pre-edit) raised NotImplementedError — verified via `inspect.getsource`.
2. **PV-2**: `Z6PredictiveCodingSubstrate.reconstruct_pair(pair_indices)` returns 3-tuple `(rgb_0, rgb_1, z_t)` — verified at `src/tac/substrates/time_traveler_l5_z6/architecture.py:399-438`.
3. **PV-3**: `Z6PredictiveCodingScoreAwareLoss.forward()` requires `residuals` kwarg — verified at `src/tac/substrates/time_traveler_l5_z6/score_aware_loss.py:107-216`.
4. **PV-4**: `ego_motion_buffer` is registered as `(num_pairs, ego_motion_dim)` zeros — verified at architecture.py:385-389.
5. **PV-5**: PoseNet returns dict with `pose` key (12-dim Hydra head; first 6 are pose params per CLAUDE.md "Exact scorer architectures") — verified at `upstream/modules.py:130-152`.
6. **PV-6**: `decode_real_pairs` returns `(N, 2, 3, 384, 512)` float32 `[0, 255]` — verified at `src/tac/substrates/_shared/trainer_skeleton.py:629-692`.

## 9-dimension success checklist evidence

| Dimension | Evidence |
|---|---|
| **(1) Uniqueness** | Catalog #310 PRIMARY class-shift — Z6 is the architectural CORE substrate; NOT a bolt-on. The FiLM-conditioned next-frame predictor + autoregressive latent unroll + archived residuals form the primary architecture (`Z6PredictiveCodingSubstrate(cfg)` constructed inside `_full_main`; `substrate.reconstruct_pair(...)` IS the per-step forward; `substrate.parameters()` IS the trained surface). |
| **(2) Beauty / elegance** | ~600 LOC `_full_main` body; ~250 LOC helpers (ego-motion derivation + runtime emit + val loop). Single canonical insertion point per layer. Canonical-vs-unique decisions per layer documented inline. PR95-style reviewable in 30 sec. |
| **(3) Distinctness** | UNIQUE-AND-COMPLETE-PER-METHOD: Z6 score-aware loss FORKS canonical via Z6PredictiveCodingScoreAwareLoss + Rao-Ballard residual-entropy term; ego-motion-conditioning is substrate-distinguishing surface (no canonical helper exists for FOE-prior derivation — that IS the substrate engineering). |
| **(4) Rigor** | 6 premise verifications BEFORE edit (Catalog #229); 74 dedicated tests pass (35 scaffold + 24 lift + 15 probe/driver); Catalog #270 Tier 1/2/3 all complete; 2026-05-17 correction: local pre-deploy now correctly fails closed while the recipe remains `research_only=true` / `dispatch_enabled=false`; checkpoint discipline every milestone per Catalog #206; commit-serializer + --expected-content-sha256 per Catalog #117/#157/#174. |
| **(5) Optimization per technique** | Tier 1: autocast_fp16/TF32/torch.compile/no_grad/canonical scorer-loss helper ALL declared (5/5). Tier 2: NVML env block + min_vram_gb + min_smoke_gpu + video_input_strategy + pyav_decode_strategy + target_modes ALL declared (8/8). Tier 3: canonical auth_eval helper + inflate device + scorer loader order + recipe consistency + no phantom filename (5/5). |
| **(6) Stack-of-stacks composability** | Z6 is the FIRST Z-variant of the F-asymptote staircase; sister Z7/Z8 expand to multi-step + hierarchical (canonical quadruple per Catalog #312) on the SAME substrate basis. Composition orthogonality via the cooperative-receiver theorem ensures additive ΔS contributions. |
| **(7) Deterministic reproducibility** | Seed-pinned via `_pin_seeds` (random + numpy + torch + torch.cuda); Z6PCWM1 archive byte-stable via sorted-keys JSON + fp16 cast on CPU + fixed brotli quality=9 + deterministic ZIP per Catalog #19. Round-trip stability proven by 35 scaffold tests + dedicated archive_roundtrip_stability_post_pack lift test. |
| **(8) Extreme optimization** | Mini-batch reconstruct_pair (Catalog #218) supports per-pair autoregressive unroll within T4 budget. EMA shadow inference per CLAUDE.md non-negotiable. Cosine annealing schedule + AdamW + grad clip 1.0 + NaN watchdog. Archive byte proxy ~97 KB target (predictor + residuals + ego-motion). |
| **(9) Optimal minimal contest score** | Predicted ΔS band [0.13, 0.16] per design memo §18 Dykstra-feasibility convex-intersection — PLANNING PRIOR ONLY (NOT a score claim) per CLAUDE.md "Apples-to-apples evidence discipline". Awaits paired CPU/CUDA empirical anchor + Phase 2 council CONSENSUS per design memo §19 reactivation criteria #4 + #5. |

## 6-hook wire-in per Catalog #125

1. **Sensitivity-map** — `not_applicable_with_rationale` (FiLM predictor gradient norm IS the per-tensor sensitivity signal; registration happens post Phase 2 council approval).
2. **Pareto constraint** — `rate_distortion_v1` (canonical convex-feasibility region; predictor_residual_entropy ≤ ε_residual to be registered post-smoke).
3. **Bit-allocator hook** — `not_applicable_with_rationale` (int8 per-pair residuals + fp16 brotli weights at L1 SCAFFOLD; per-tensor bit allocator deferred).
4. **Cathedral autopilot dispatch hook** — recipe at `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml`; gated by Catalog #167 smoke-before-full; ranker v2 receives `literature_anchor=Rao-Ballard1999+Atick-Redlich1990` as source-basis metadata only. ACTIVE.
5. **Continual-learning posterior** — `not_applicable_with_rationale` (L1 SCAFFOLD has no contest-CUDA anchor yet; `posterior_update_locked_from_auth_eval_json` fires after Phase 2 dispatch lands paired CPU/CUDA evidence).
6. **Probe-disambiguator** — `tools/probe_z6_predictive_coding_vs_identity_disambiguator.py` (canonical identity-predictor ablation; if full FiLM beats identity by ΔS ≥ 0.005, Rao-Ballard hypothesis confirmed). ACTIVE.

## Catalog #310/#311/#312 evidence

- **Catalog #310 PRIMARY class-shift NOT bolt-on**: `Z6PredictiveCodingSubstrate(cfg)` is the primary architecture; `substrate.parameters()` is the trained surface; `substrate.reconstruct_pair(...)` is the per-step forward.
- **Catalog #311 ego-motion-conditioned next-frame prediction**: `_derive_ego_motion_from_posenet(...)` derives per-pair ego-motion vectors from PoseNet head output (first `ego_motion_dim` coords; standardized per-column for FOE-prior conditioning per Gibson 1950); `substrate.ego_motion_buffer.copy_(...)` wires it into the FiLM predictor; the autoregressive `predictor(z_{t-1}, ego_motion[t])` IS the canonical pose-conditioned next-frame predictor.
- **Catalog #312 hierarchical canonical quadruple**: NOT REQUIRED for Z6 per the design memo §4.1 + §13 — Z6 is the SIMPLEST viable predictive-coding variant (single-layer FiLM predictor); the hierarchical quadruple (Rao-Ballard + Mallat wavelet + DreamerV3 + Wyner-Ziv) is the Z8 scope. Z6 satisfies the Rao-Ballard + Atick-Redlich pair which IS the substrate's class-shift surface.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| 1. pyav decode | ADOPT canonical | `decode_real_pairs` helper (Catalog #114). |
| 2. seed pinning | ADOPT canonical | `_pin_seeds` mirrors trainer-skeleton pattern. |
| 3. device gate | ADOPT canonical | `device_or_die` (CLAUDE.md MPS-NOISE rule). |
| 4. YUV6 patch | ADOPT canonical | `patch_upstream_yuv6_globally` BEFORE scorer load (Catalog #187). |
| 5. scorer load | ADOPT canonical | `load_differentiable_scorers` order `(posenet, segnet)` (Catalog #222). |
| 6. EMA shadow | ADOPT canonical | `tac.training.EMA(decay=0.997)` per CLAUDE.md non-negotiable. |
| 7. score-aware loss | **FORK** | `Z6PredictiveCodingScoreAwareLoss` with Rao-Ballard residual-entropy Lagrangian (the canonical `score_pair_components_dispatch` homogenizes scorer-preprocess routing; the FORK is the residual term + the ego-motion sidecar requirement). |
| 8. ego-motion conditioning | **FORK** | `_derive_ego_motion_from_posenet` with FOE-prior standardization — no canonical helper exists; this IS the distinguishing substrate primitive per Catalog #311. |
| 9. mini-batch reconstruct | ADOPT canonical | `reconstruct_pair` indexed per Catalog #218 OOM discipline. |
| 10. archive pack/runtime | **FORK** | Z6PCWM1 grammar (predictor + residuals + ego-motion sidecar; canonical pack_archive helper validates the 6-section contract). |
| 11. auth-eval gate | ADOPT canonical | `gate_auth_eval_call` (Catalog #226). |
| 12. posterior update | ADOPT canonical | `posterior_update_locked_from_auth_eval_json` (Catalog #128). |
| 13. provenance + manifest | ADOPT canonical | Sister-substrate pattern (NSCS01 template). |
| 14. hardware detect | ADOPT canonical | `detect_hardware_substrate` (Catalog #190). |

## Local pre-deploy + Catalog #270 verdicts

```
[local-pre-deploy] validating: train_substrate_time_traveler_l5_z6.py  recipe=substrate_time_traveler_l5_z6_modal_t4_dispatch
[local-pre-deploy] mode: STRICT (exit 1 on fail)
  ✓ [py_compile] PASS
  ✓ [trainer_importable] PASS
  ✓ [full_main_implemented] PASS — _full_main appears implemented
  ✓ [archive_grammar] PASS
  ✓ [auth_eval_reachability] PASS — canonical helper
  ✓ [canonical_inflate_device] PASS — no inline torch.device cuda-fallback
  ✓ [deterministic_zip] PASS
  ✗ [recipe_status_consistent_with_trainer_state] FAIL — trainer _full_main is implemented but recipe is still non-dispatchable (research_only=true, dispatch_enabled=false, dispatch_blockers)
  ✓ [dispatch_optimization_protocol] PASS — Tier 1/2/3 all complete (tier1=5/5 / tier2=8/8 / tier3=5/5)
[local-pre-deploy] 1 CHECK(S) FAILED: recipe_status_consistent_with_trainer_state
```

2026-05-17 correction: the earlier "ALL 9 CHECKS PASSED. Safe to dispatch"
receipt was false authority. `operator_authorize.py --dry-run` correctly
refuses this recipe with `dispatch_enabled=false` and the declared
`dispatch_blockers`. `local_pre_deploy_check.py` has been corrected to match
that operator-authorize decision. The Z6 implementation remains live, but the
recipe is intentionally non-dispatchable until Phase 2 and smoke-before-full
gates clear.

## Lane gates marked

- `impl_complete=true` (`_full_main` lifted; 24 dedicated lift tests pass)
- `strict_preflight=false` (Catalog #270 remains green, but local pre-deploy now fails closed until recipe dispatch blockers clear)
- `deploy_runbook=true` (pre-existing `scripts/remote_lane_substrate_time_traveler_l5_z6.sh`)
- `real_archive_empirical=false` — pending Phase 2 council CONSENSUS + smoke-before-full at $1 Modal T4
- `contest_cuda=false` — pending paired CPU/CUDA empirical anchor
- `contest_cpu=false` — pending Linux x86_64 paired CPU eval
- `three_clean_review=false` — pending council review
- `memory_entry=true` (THIS file)

## Recipe state

`.omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml`:
- `research_only: true` (preserved per Catalog #220 substrate-level promotion gate; paired evidence required)
- `dispatch_enabled: false` (preserved per design memo §19 reactivation criteria #4 + #5; Phase 2 council CONSENSUS + smoke-green required)
- `dispatch_blockers:` now explicitly enumerate the REMAINING gates (council consensus + smoke-before-full + paired empirical anchor) — Phase 1b satisfied criterion #3 (sister L1 SCAFFOLD lands trainer + archive + inflate + tests) per memo §19.

## Sister-subagent ownership map honored (Catalog #230)

- **Phase 1a (sister)**: owns dispatch on STC v2 + NSCS06 v8 — NOT touched.
- **LIFT-Rudin (sister)**: owns `experiments/train_substrate_rudin_floor_interpretable_ml.py` + sisters — NOT touched.
- **LIFT-ATW-v2 (sister)**: owns `experiments/train_substrate_atw_codec_v2.py` + sisters — NOT touched.
- **This lane** owns ONLY: `experiments/train_substrate_time_traveler_l5_z6.py`, `src/tac/substrates/time_traveler_l5_z6/tests/test_z6.py`, `src/tac/tests/test_train_time_traveler_l5_z6_full_main_lift.py`, `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml`.

## Cost analysis

- **GPU spend**: $0 (CPU-only edits + tests).
- **Wall-clock**: ~3.5h editor + verification.
- **Phase 2 forward path**: per design memo §19 criteria #4 + #5 — sextet-pact council CONSENSUS (Shannon + Dykstra + Rao + Ballard + Tishby + Contrarian + Assumption-Adversary) + Catalog #167 smoke-before-full at $1 Modal T4 confirming smoke green (rc=0 + archive bytes in [80, 250] KB) before contest-CUDA dispatch can fire.

## Cross-references

- Parent design memo: `.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md` (commit `aa412d2db`)
- Sister Z5 L1 scaffold: `lane_z5_predictive_coding_world_model_step3_20260514`
- Sister Z4 L1 scaffold: `lane_z4_cooperative_receiver_loss_step2_20260514`
- Reference patterns: `experiments/train_substrate_nscs01_nullspace_split_renderer.py` + `experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py`
- HORIZON-CLASS standing directive 2026-05-16
