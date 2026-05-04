---
name: 2026-04-28 signal loss audit — what dead codex sessions left orphaned
description: Comprehensive audit found 55 uncommitted ?? files representing real codex deliverables that never got committed. Classified by type: 6 research docs, 16 modules + 17 matching tests (valid pairs), 4 lane scripts, 3 experiment scripts. Plus 4-6 incomplete recursive review chains and 1 Round 11 Finding still failing (Wave 2 timeout).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Why this matters

When 9 codex sessions died (per `project_dead_codex_recovery_inventory_20260428`),
they left 55+ uncommitted files containing real engineering work. The git index
treats them as `??` (untracked) so they SURVIVE the session death — but they
are invisible to:
- Preflight (untracked files aren't scanned by some checks)
- Test collection (uncommitted test files run only via explicit pytest path)
- Future sessions (a fresh agent doing `git status` sees the noise but won't auto-commit)

## Inventory of orphaned work (55 ?? files as of 2026-04-28 14:00)

### Research/synthesis docs (6, ALL valid — should commit)
- `.omx/research/arxiv_2604_24763_synthesis.md` — Tuna-2 deep read (993 lines)
- `.omx/research/codex_adversarial_review_round_11_20260428.md` — 4 findings
- `.omx/research/cosmos_deep_dive_addendum_20260428.md` — Cosmos corrective pass
- `.omx/research/cosmos_mae_2604_telescope_synthesis.md` — 4-resource synthesis
- `.omx/research/lane_g_v3_stacking_skunkworks_20260428.md` — wedge attribution + 3 lanes
- `.omx/research/lane_m_v2_audit_council_20260428.md` — BUG-1 audit
- `docs/lane_methodology.md` — controlled-variant pattern doc

### Modules + tests (paired, valid)
| Module | Test | Lane | Status |
|--------|------|------|--------|
| `src/tac/curator_outlier.py` | `test_curator_outlier.py` | Lane WC Cosmos | Subagent landed (commit 9cdd052b sibling?) |
| `src/tac/entropy_bottleneck.py` | `test_entropy_bottleneck.py` | Lane EBR Ballé-2018 | Untouched — needs Wave 3 |
| `src/tac/feature_masking.py` | `test_feature_masking.py` | Lane T2-DROP Tuna-2 | Untouched — needs Wave 3 |
| `src/tac/loss_t2_xpred.py` | `test_loss_t2_xpred.py` | Lane T2-XPRED Tuna-2 | Untouched — needs Wave 3 |
| `src/tac/contrib/multi_control_hint_encoder.py` | `test_multi_control_hint_encoder.py` | Lane TFR Cosmos | Untouched — needs Wave 3 |
| `src/tac/contrib/calibrated_positional_encoding.py` | `test_calibrated_positional_encoding.py` | Lane CPE | Untouched |
| `src/tac/contrib/homography_motion.py` | `test_homography_motion.py` | Lane HM | Already covered by HM+CG SHA b3b4a978 |
| `src/tac/pose_gaussian_process.py` | `test_pose_gaussian_process.py` | Lane GP task #149 | Module exists, deploy script missing |
| `src/tac/raft_pose.py` | `test_raft_pose.py` | Lane FL task #149 | Module exists, deploy script missing |
| `src/tac/uniward_texture.py` | `test_uniward_texture.py` | Lane SI-V3 | Already covered by Wave 1 HM+CG inventory |
| `src/tac/geodesic_pose.py` | `test_geodesic_pose.py` | Lane GE | Already covered by Wave 1 HM+CG |
| `src/tac/openpilot_features.py` | `test_openpilot_features.py` | Lane DI | Module exists; tests pass |
| `src/tac/self_augmentation_v2.py` | `test_self_augmentation_v2.py` | Lane SAUG-V2 | Wave 2 verified (487f94eb) |
| `src/tac/parametrize_strip.py` | `test_parametrize_strip.py` | helper | Round 11 cleanup — already factored |
| `src/tac/mask_prior.py` | (mask_class_prior) | Lane MOS | Wave 1 landed (9cdd052b) |
| `experiments/build_mask_class_prior.py` | (mask_class_prior) | Lane MOS | Wave 1 landed |
| `experiments/derive_poses_from_raft.py` | — | Lane FL helper | Untouched |
| `experiments/fit_curator_outlier_weights.py` | — | Lane WC helper | Wave 1 indirectly |
| `experiments/fit_pose_gp.py` | — | Lane GP helper | Untouched |
| `experiments/sweep_seg_weight.py` | `test_sweep_seg_weight.py` | Lane T2-RATIO | Untouched |

### Lane scripts (4, untouched)
- `scripts/launch_lane_on_vastai.py` — V6 launcher (committed via 58e55890 path)
- `scripts/remote_lane_t2drop_bootstrap.sh` — Lane T2-DROP deploy
- `scripts/remote_lane_t2ratio_bootstrap.sh` — Lane T2-RATIO deploy
- `scripts/verify_vast_instances.py` — Per-instance verify script (memory entry exists)

### Tests for landed work (cleanup)
- `test_lane_methodology_preflight.py` — methodology doc + preflight
- `test_lane_sg_self_compress.py` — Lane SG (Wave 2 commit 3aa7c2ef)
- `test_pose_projection_parity.py` — BUG-1 parity (Wave 2 commit 01804a4d)
- `test_undeployed_artifact_producers.py` — Check 39 (committed in 0380a869)
- `test_remote_lane_heartbeat_check.py` — Check 41 (committed)
- `test_remote_lane_gp_script.py` — Round 11 Finding 4 (transient; auto-resolves with GP script land)
- `test_fp4_hardware_disclosure.py` — Check 40 (committed)

### Result artifacts (large dirs, intentionally local-only)
- `experiments/results/lane_g_v3_landed/` — KEEP local (committing too large)
- `experiments/results/lane_m_v2_landed/` — KEEP local

## Incomplete recursive bug review chains

### Round 11 (latest) — 4 findings
- ✅ Finding 1 (Lane WC resume-from): FIXED via linter post-Wave-2 (visible in scripts/remote_lane_wc_curator_outlier.sh diff)
- ❌ Finding 2 (learnable pair/class weights don't learn): Wave 2 subagent timed out — STILL PENDING
- ✅ Finding 3 (Lane F-V5 imports): RESOLVED via Wave 1 commit 54d29cec
- ✅ Finding 4 (Lane GP test points at missing script): TRANSIENT, will auto-resolve when script lands

### Round 12 (next round, NOT YET RUN)
After Wave 1 + Wave 2 landings (5 lanes + various), a fresh adversarial review
should be run to catch any introduced regressions or missed integration issues.

## Comprehensive council eval — 6 architectures REVIVED but never deployed
Per `.omx/research/comprehensive_council_eval.md`:
1. DP-SIMS (SPADE Progressive Generator) — code complete in `src/tac/dp_sims_renderer.py` (778 lines), profile defined, never trained
2. VQ-VAE Latent Codec — code complete in `src/tac/vqvae_codec.py` (880 lines), never trained
3. Diffusion Teacher + Distillation — code complete in `src/tac/diffusion_renderer.py` (1192 lines), never trained
4. Wavelet-Domain Renderer — code complete in `src/tac/wavelet_renderer.py` (376 lines), never trained
5. Test-Time Optimization (TTO) — exists, never deployed at full scale
6. MLX Renderer (Phase 1 Acceleration) — exists for Apple Silicon dev

These represent ~3300 LOC of validated implementations that have never been pushed to a CUDA-CUDA auth eval.

## Key memory entries that recorded "TIER-1 next session" deferred work
- feedback_canonical_parent_shell_launcher_20260428 — launcher V2-V6 evolution; mostly done now
- feedback_bash_harness_kills_long_running_tasks_20260428 — split harness; partially fixed via subshell-detach
- feedback_cycle_1_launch_postmortem_20260428 — 5 metabugs identified; 3 fixed (NVDEC light probe, &&-gating, wrapper)
- feedback_launcher_v3_phase2_split_needed_20260428 — split phase2-deploy into sub-phases; superseded by V6
- feedback_metabugs_round_3_20260428 — 3 metabugs; A (frame_utils) + B (NVDEC) FIXED, C (env.sh hard-source) FIXED
- project_lane_i_crashed_parametrize_strip_20260428 — Lane I needs parametrize-strip fix in inline Stage 3 Python
- project_lane_v_crashed_channel_mismatch_20260428 — Lane V channel mismatch needs V2

## What Wave 3 must dispatch

1. **Audit-and-commit-orphans** subagent: walk all 55 ?? files, classify, commit valid pairs, document orphans
2. **Round 11 Finding 2 RETRY** subagent: learnable pair/class weights actually learn (anti-arbitrariness)
3. **Tuna-2 lanes batch** subagent: T2-DROP + T2-XPRED + T2-RATIO from existing modules (ship coherently)
4. **Lane GP + Lane FL deploy scripts** subagent: task #149 — write the deploy scripts so the modules become deployable
5. **Lane EBR Ballé entropy bottleneck** subagent: ship the orphaned src/tac/entropy_bottleneck.py + tests properly

## Cross-references
- `project_dead_codex_recovery_inventory_20260428` — original Wave 1+2 dispatch
- `project_outstanding_work_and_stacks_20260428` — full lane taxonomy
- `.omx/research/comprehensive_council_eval.md` — the 6 REVIVED architectures
- `.omx/research/codex_adversarial_review_round_11_20260428.md` — Round 11 findings
