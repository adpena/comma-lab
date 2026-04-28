# Orphan Implementations Audit — 2026-04-28

**Context.** Per CLAUDE.md NON-NEGOTIABLE Auth-Eval-Everywhere + the 35-Vast.ai-lane parallel cycle: every committed module that has a tests/ counterpart but no profile/CLI/deploy wiring is **wasted research**. This audit enumerates such orphans.

**Scope.** All public modules under `src/tac/**/*.py` (excluding `tests/`) that:
1. Were committed in the last ~2 days (since 2026-04-26 00:00), OR have a recent feature commit indicating active research.
2. Have a corresponding test file `src/tac/tests/test_<module>.py`.
3. Are NOT referenced from any `scripts/remote_lane_*.sh`, `experiments/*.py`, `src/tac/profiles.py`, or `src/tac/experiments/*.py` outside their own test files.

**Method.** Cross-reference `from tac.<module>` / `import tac.<module>` references in production code (deploy scripts + experiments + profiles + train_renderer.py). Modules with at least one production reference are NOT orphans (they may still need profile/CLI wiring but are partially integrated).

---

## Orphan modules (TRUE — zero production references)

### 1. `src/tac/uniward_texture.py` (164 LOC)
- **Test file:** `src/tac/tests/test_uniward_texture.py` (exists)
- **Last commit touching it:** `b0a2e45f` (2026-04-28) — "Fix 2 F821 false-confidence bugs: V6 launcher (production blocker) + uniward_texture (test gap)"
- **Missing integration:** profile entry, CLI flag, deploy script — none exist.
- **Predicted band per memory:** `feedback_scorer_alignment_audit` notes UNIWARD/L∞/Markov losses are PARTIALLY aligned with our scorers. CLAUDE.md "Fridrich inverse steganalysis" §1: "UNIWARD: errors in textured regions are undetectable. Weight loss by inverse local variance." Predicted gain band: speculative ~[0.05, 0.15] off SegNet via texture-weighted penalty. NOT YET independently estimated.
- **Recovery effort:** Medium — needs (a) profile inheriting Lane G v3 + `use_uniward_loss=True` + weight knob, (b) CLI flag in `train_renderer.py`, (c) loss-dispatch hook into the per-step loss block, (d) `scripts/remote_lane_uniward.sh`. Estimated 1.5–2 hours.

### 2. `src/tac/semantic_quantization.py` (233 LOC)
- **Test file:** `src/tac/tests/test_semantic_quantization.py` (exists)
- **Last commit touching it:** `064c9866` (2026-04-11) — "Implement Yousfi tricks 13-21: 5 new tac modules (Karpathy+Tao)"
- **Missing integration:** No profile, no CLI, no deploy script.
- **Predicted band per memory:** No specific memory entry. Likely one of the "Yousfi tricks 13-21" that never got plumbed through. Speculative.
- **Recovery effort:** Medium — needs same 4-piece wiring as uniward. Older module (April 11) so may need API-drift verification first. Estimated 2–3 hours.

### 3. `src/tac/archive_codec.py` (789 LOC)
- **Test file:** `src/tac/tests/test_archive_codec.py` (exists)
- **Last commit touching it:** `2bac5927` (2026-04-12) — "reorg: move dead code to archive/ and contrib/ directories"
- **Missing integration:** No production reference. Largest orphan by LOC.
- **Predicted band per memory:** Unknown. Possibly a precursor to `entropy_archive.py`.
- **Recovery effort:** Large — 789 LOC suggests substantial logic. May overlap with `entropy_archive` or `mask_entropy_coder`. Triage first: is this superseded? If yes, delete. Estimated 3–4 hours to triage + integrate, OR 30 min to delete.

### 4. `src/tac/entropy_archive.py` (1012 LOC)
- **Test file:** `src/tac/tests/test_entropy_archive.py` (exists, recently touched in commit `76c797d2`)
- **Last commit touching it:** `2f5ab137` (2026-04-12) — "fix: all 5 entropy archive issues + compress_byte_stream API"
- **Missing integration:** No production reference. Largest orphan by LOC.
- **Predicted band per memory:** Likely an early entropy-coding lane that was supplanted by Lane EBR (`ebr_dilated_h64` profile + `scripts/remote_lane_ebr_entropy_bottleneck.sh`).
- **Recovery effort:** Triage first — verify whether EBR makes this obsolete. If yes: delete. If no: ~3 hours to wire as a postfilter / archive packer. Estimated 3–4 hours.

### 5. `src/tac/geodesic_pose.py` (146 LOC)
- **Test file:** `src/tac/tests/test_geodesic_pose.py` (exists, in `?? src/tac/tests/test_geodesic_pose.py` per git status)
- **Last commit touching it:** `57506769` (2026-04-28) — "Round 11 Finding 2 fix: learnable pair/class weights actually adapt during training (anti-arbitrariness)"
- **Missing integration:** No profile, no CLI, no deploy. Tightly related to `se3.py` (which IS wired in `optimize_poses.py`, `riemannian_pose_optimizer.py`).
- **Predicted band per memory:** Lane RM (Riemannian pose TTO) is the production consumer of SE(3) machinery. `geodesic_pose` may be a parallel/alternative geodesic step that never got promoted to Lane RM's deploy script.
- **Recovery effort:** Small — could either (a) be merged into Lane RM as an alternative `--pose-update-mode geodesic` knob, or (b) get its own Lane GD deploy. Estimated 1 hour.

### 6. `src/tac/contrib/calibrated_positional_encoding.py` (120 LOC)
- **Test file:** `src/tac/tests/test_calibrated_positional_encoding.py` (exists, in `??` git status)
- **Last commit touching it:** `b7588336` (2026-04-28) — "contrib: 3 paired modules + tests for calibrated positional encoding, homography motion, multi-control hints"
- **Missing integration:** No reference outside own tests. Lives in `contrib/` (suggesting "research / not promoted").
- **Predicted band per memory:** No specific entry. Designed as a positional-encoding upgrade for the renderer; would complement architectures like the dilated-h64 baseline.
- **Recovery effort:** Medium — needs renderer-architecture integration (probably a `use_calibrated_pe=True` flag + builder branch in `build_renderer`), plus profile + deploy. Estimated 2–3 hours.

### 7. `src/tac/contrib/homography_motion.py` (170 LOC)
- **Test file:** `src/tac/tests/test_homography_motion.py` (exists, in `??` git status)
- **Last commit touching it:** `b7588336` (2026-04-28)
- **Missing integration:** No reference outside own tests.
- **Predicted band per memory:** Could replace or augment the `RadialZoomWarp` motion model (which is the half-frame paradigm anchor). Speculative ~[0.95, 1.10] if wired into Lane V/Lane K-style joint training.
- **Recovery effort:** Medium-Large — needs motion-module integration in `build_renderer` + flag + half-frame test compatibility. Estimated 3–4 hours.

### 8. `src/tac/contrib/multi_control_hint_encoder.py` (142 LOC)
- **Test file:** `src/tac/tests/test_multi_control_hint_encoder.py` (exists)
- **Last commit touching it:** `b7588336` (2026-04-28)
- **Missing integration:** No reference outside own tests.
- **Predicted band per memory:** No specific entry. Likely a ControlNet-style hint encoder; would need architectural integration.
- **Recovery effort:** Medium — same shape as #6/#7. Estimated 2–3 hours.

---

## Partially-integrated modules (have test + deploy, but profile/CLI gap)

These are NOT pure orphans (they have at least one production reference) but may still benefit from a sweep:

### A. `src/tac/iterative_magnitude_pruning.py`
- Production refs: `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh`, `experiments/train_imp_cycle.py`. Last commit `188eba16` (2026-04-28).
- Status: WIRED. No follow-up needed.

### B. `src/tac/losses_jbl.py`
- Production refs: `src/tac/profiles.py` (J_JBL_DILATED_H64), `src/tac/experiments/train_renderer.py`. Last commit `2b2dc9e6` (2026-04-28).
- Status: WIRED. (But added the broken loss_mode validator that this MAE-V commit fixed.)

### C. `src/tac/neural_weight_codec.py` + `neural_weight_codec_sensitivity.py` + `stack_compositions.py`
- Production refs: `scripts/remote_lane_j_nwc_*.sh`, `scripts/remote_lane_j_nwcs_*.sh`, `experiments/train_neural_weight_codec.py`. Last commits `a8315339`, `3775c9ce`, `8c80165b` (all 2026-04-28).
- Status: WIRED.

### D. `src/tac/pose_gaussian_process.py`
- Production refs: `scripts/remote_lane_gp_gaussian_process_pose.sh`, `experiments/fit_pose_gp.py`. Last commit `57506769` (2026-04-28).
- Status: WIRED.

### E. `src/tac/raft_pose.py`
- Production refs: `experiments/derive_poses_from_raft.py` + `scripts/remote_lane_fl_raft_derived_poses.sh`. Last commit `57506769` (2026-04-28).
- Status: WIRED.

### F. `src/tac/curator_outlier.py`
- Production refs: `experiments/fit_curator_outlier_weights.py` + `scripts/remote_lane_wc_curator_outlier.sh`.
- Status: WIRED.

### G. `src/tac/mae_mask_aug.py`
- Production refs (POST this commit): `src/tac/experiments/train_renderer.py` + `src/tac/profiles.py` + `scripts/remote_lane_mae_v.sh`.
- Status: WIRED (Lane MAE-V — this commit's deliverable).

---

## Recommended dispatch order

By predicted EV / recovery cost:

| # | Module | Effort | Predicted band | Notes |
|---|--------|--------|----------------|-------|
| 1 | `uniward_texture.py` | Medium (2h) | speculative [0.05, 0.15] off SegNet | Fridrich-aligned, CLAUDE.md flagged |
| 2 | `geodesic_pose.py` | Small (1h) | speculative — Lane RM ablation | Lowest-cost orphan |
| 3 | `contrib/calibrated_positional_encoding.py` | Medium (2-3h) | unknown | Architecture-integration risk |
| 4 | `contrib/homography_motion.py` | Medium-Large (3-4h) | [0.95, 1.10] speculative | Half-frame compat needed |
| 5 | `contrib/multi_control_hint_encoder.py` | Medium (2-3h) | unknown | ControlNet-class |
| 6 | `semantic_quantization.py` | Medium (2-3h) | unknown | API-drift risk (Apr 11) |
| 7 | `archive_codec.py` | Large (3-4h) OR delete | likely superseded | Triage vs `entropy_archive` first |
| 8 | `entropy_archive.py` | Large (3-4h) OR delete | likely superseded by EBR | Triage vs Lane EBR first |

**Triage first (#7 + #8):** these large modules likely overlap with already-deployed lanes (EBR + mask_entropy_coder). Confirm supersession before either deleting or wiring.

**Highest EV without overlap risk:** `uniward_texture.py` — Fridrich-explicit, CLAUDE.md flagged, clean module surface. Recommend this as the next dispatchable orphan-recovery deliverable.

---

## Methodology notes

- This audit is module-resolution only (`from tac.X import` checks). It does NOT verify whether each "wired" module is actually exercised at runtime — that would require dispatch + telemetry.
- Test-only modules (e.g., `test_round11_finding2_learnable_weight_adaptation.py` covers `learnable_pair_weights` which IS wired into `train_renderer.py`) are correctly excluded as non-orphans.
- The `?? src/tac/tests/test_*.py` entries in `git status` (test files for `geodesic_pose`, `raft_pose`, `pose_gaussian_process`, `homography_motion`, `calibrated_positional_encoding`, `lane_sg_self_compress`, `openpilot_features`, `uniward_texture`, `lane_ec_engineered_corrections`, `remote_lane_gp_script`) indicate fresh tests not yet committed — should be `git add`ed in a follow-up commit alongside their orphan-recovery wiring (or just committed as test coverage).
