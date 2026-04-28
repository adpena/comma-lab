# Battle Plan: Beat Quantizr (0.33)

**Deadline: May 3, 2026. 7 days remaining.**
**Verified contest-CUDA baseline: 0.9001** (pinned dilated h64 + CRF=50 + matched poses, 2026-04-25).
**Target: < 0.33 (beat Quantizr). Verified gap: 0.57.**

> **What's true:** the 0.90 is the *only* number this lab has ever measured on the real CUDA contest scorer that is not a measurement artifact. SHIRAZ (181K) and DEN (123K) both lost to it. Bigger architecture (or better training of the dilated-h64 family) is the path forward, not smaller experimental renderers.

## Score decomposition vs Quantizr

| Component | Baseline 0.90 | Quantizr 0.33 (est.) | Required for us |
|---|---|---|---|
| 100·seg_d | 0.24 (seg=0.0024) | ~0.07 | **0.05** |
| √(10·pose_d) | 0.327 (pose=0.0107) | ~0.06 | **0.05** |
| 25·rate | ~0.32 (~480KB archive) | 0.20 (~300KB) | **0.20** |
| **Sum** | **~0.90** | **~0.33** | **~0.30** |

Pose is the single biggest component (0.327 = 36% of score). Rate is the second-largest fixed cost. SegNet third. Cutting all three by ~3× hits target.

## Three pillars to attack the gap

### Pillar A — Reduce rate (target: 0.32 → 0.20, win 0.12)

| # | Action | Cost | Expected delta | Status |
|---|--------|------|----------------|--------|
| A1 | **Mask CRF sweep on baseline archive** (CRF 50→56→63 on the dilated-h64 baseline). Build archive at each, e2e auth eval. Pick Pareto best. | CPU + 3× CUDA eval = ~30min, $0.50 | rate −0.05 to −0.10 (potential SegNet trade) | NEW |
| A2 | **Half-frame mask encoding** (Quantizr trick — store only 600 odd-frame masks; reconstruct even from odd via warp at inflate). Already implemented locally. Apply to baseline archive. | CPU local, eval ~10min | rate −0.05 to −0.10, distortion neutral if warp clean | code ready, untested on baseline |
| A3 | **Lane-mark-zoom in archive** (Hotz analytical: zero archive cost, infers per-frame zoom from masks at inflate). | local code merge + eval | rate 0 (free), pose −0.001 to −0.005 | code ready, untested |

### Pillar B — Reduce PoseNet distortion (target: 0.327 → 0.05, win 0.28)

PoseNet is 36% of total score. The biggest single lever.

| # | Action | Cost | Expected delta | Status |
|---|--------|------|----------------|--------|
| B1 | **Pose TTO on the dilated-h64 baseline** (we ran SHIRAZ TTO but never on h64; the 0.90 used the original training poses, not optimized). | $0.30 (1h on 4090) | pose −50% to −70% (proxy) → CUDA pose 0.005-0.008 | UNTESTED on baseline |
| B2 | **eval_roundtrip + noise_std=0.5 in pose TTO** (already canonicalized — but verify it's wired into the dilated-h64 path) | trivial | proxy-CUDA gap < 5% (vs 23× MPS noise) | enforced |
| B3 | **Larger pose TTO budget** (2000 steps vs default 600). Diminishing returns but still positive. | +30 min | pose −10-20% additional | speculative |

### Pillar C — Reduce SegNet distortion (target: 0.24 → 0.05, win 0.19)

| # | Action | Cost | Expected delta | Status |
|---|--------|------|----------------|--------|
| C1 | **Re-train dilated-h64 with new techniques** (DCT-quant loss, KL distill T=2.0 on SegNet, UNIWARD/L∞/Markov, hinge loss, mixed-CRF mask aug). All techniques shipped today; never applied to the h64 path. | $0.80 (2-3h on 4090) | seg −30% to −50% (~0.0012-0.0017) | UNTESTED on h64 |
| C2 | **Engineered SegNet corrections** (gradient-directed pixel patches at compress time). | local CPU | seg −0.05 to −0.10 (proxy; auth gap unknown) | code exists, FP4 load broken — needs fix |
| C3 | **Bigger renderer** (scale h64 up — the user's insight that bigger = better for this arch family). E.g., dilated h96 or h128 with more channels. | $1-2 (3-4h on 4090) | seg + pose better, rate worse | ⚠ counter to rate goal — only if A pillar wins enough headroom |

## Stacked projection

| Path | Score | Math | Confidence |
|---|---|---|---|
| Verified today | **0.90** | dilated-h64 baseline | HIGH |
| + A1 (CRF tune) | **0.85** | rate −0.05 | HIGH |
| + A2 (half-frame masks) | **0.78** | rate −0.07 (−0.10 raw, +0.03 SegNet bleed) | MEDIUM |
| + B1 (pose TTO on h64) | **0.55** | pose 0.327 → 0.10 (sqrt(10·0.001) ≈ 0.10) | MEDIUM-HIGH |
| + C1 (h64 retrain new techniques) | **0.40** | seg 0.24 → 0.10 + further pose tightening | MEDIUM |
| + C3 (bigger arch, paid for by A wins) | **0.30** | breakthrough — needs measurement | SPECULATIVE |
| **Quantizr** | **0.33** | their published score | reference |

The path to sub-0.33 is **A1 + A2 + B1 + C1** stacked. None require fundamentally new techniques — every piece is already implemented, just never applied to the dilated-h64 baseline.

## Today's order of execution (zero waste, score-driven)

1. **A1 mask CRF sweep on existing 0.90 archive** — local CPU, no GPU, ~30 min. Picks the right CRF for the next steps.
2. **A2 half-frame masks on the same archive** — local CPU + 1 CUDA eval, ~20 min. Confirms the rate win.
3. **B1 pose TTO on dilated-h64 baseline** — fresh 4090, ~1h, $0.30. This is the biggest single lever (PoseNet 36% of score). Use the canonical bootstraps.
4. **C1 h64 retrain with new techniques** — fresh 4090, ~3h, $0.80. Only after A+B confirm the rate+pose lane is solid.
5. Eval after EACH step. Verify on real CUDA. Don't stack 4 changes and then measure.

## What we are explicitly NOT doing

- **DEN-V2 deploy patching** — sunk cost. Trained checkpoint preserved in `experiments/results/den_v2_archive/` for future. The ~$1 burn taught us PairGenerator deployment needs a proper interface refactor (in working tree, blocked by review gate). The score of a smaller renderer wouldn't beat 0.90 anyway.
- **SHIRAZ further iteration** — same architecture lesson. 109-181K is undermatched. SHIRAZ best is 2.70 (re-eval today with --poses fix); not on the path to 0.30.
- **Cool-Chic / C3 / DP-SIMS / VQ-VAE / diffusion teacher / cross_disc_*** — none have CUDA-verified scores; not deployable in 7 days.
- **Score chasing on MPS/proxy** — every score must be CUDA real. MPS has 23× drift on PoseNet alone.

## Active hardware budget

- **0.90 archive**: pinned, immutable, on local disk. No GPU needed for A1+A2.
- **One 4090 needed for B1** (1h, $0.30). Use any cheap US/CA/EU instance via canonical bootstrap.
- **Second 4090 for C1** when ready (3h, $0.80).
- Total tonight projected: $1.10 to push the score from 0.90 to ~0.55. $2.00 total to attempt sub-0.40.
- Hard cap remains $24. Burned so far ~$2.

## Hardening that must land before next deploy

| Item | Why | ETA |
|---|---|---|
| **Preflight: profile-pipeline interface check** | Instantiate `build_renderer(profile)`; verify all attrs the deploy chain reads exist. DEN burned $1 layer-by-layer because we lacked this. | 30 min |
| **PairGenerator forwarding shim** | Already in working tree; needs proper greenup (2 distinct approvers + 1 human) before merge. Unblocks future PairGenerator-class profiles. | depends on greenup |
| **step_pose_tto soft-skip on pose_dim=0** | LANDED commit 9436fb15 | done |
| **pipeline.step_eval auto-pass --poses** | LANDED commit 63854f31 | done |
| **mask_codec ffmpeg-new + LD_LIBRARY_PATH** | LANDED commit 2acbc25b | done |
| **Bootstrap profile case-normalization** | LANDED commit 94566150 | done |

## Non-negotiables (carried forward)

- All experiments through `pipeline.py` + canonical bootstraps. No ad-hoc.
- `eval_roundtrip=True` everywhere.
- All scores labeled `[contest-CUDA]` or `[proxy]`. MPS scores are noise.
- Verified CUDA gate before any submission.
- Destroy Vast.ai instances IMMEDIATELY after download — $24 hard cap.
- 5 consecutive clean adversarial review passes before final PR.

## 2026-04-27 Council verdict: Lane G KL-distill at pose TTO — ABANDONED

Council vote 5/5 against Lane G as a score-optimization technique (full deliberation in `.omx/research/findings.md` under "2026-04-27 Council on Lane G KL-distill at pose TTO"). Summary:

- **Yousfi**: KL on SegNet logits is orthogonal to FiLM-pose conditioning manifold. Low-leverage. Adds fragility tax against scorer drift.
- **Fridrich**: KL-distill is teacher→student compression objective; we have no teacher/student at pose TTO. Anti-aligned with 2 of 4 inverse-steganalysis principles (UNIWARD, L∞ spreading). Suppresses CNN-blind-spot exploitation.
- **Hotz**: $0.85 for V3 vs $0.30 for Lane M. EV per dollar 10×+ worse. Lane M radial-zoom is a measured-physics prediction; Lane G is a hyperparameter search with bounded-near-zero ceiling.
- **Quantizr**: Their KL is training-time, alongside MSE+adversarial+soft-argmax — never inference-time. Lifting it into our pose TTO is a cargo-cult category error. KL-confidence-pressure on argmax-correct cells is wasted work.
- **Contrarian**: V3 weight 5e-6 is post-hoc normalization, not a hypothesis. Even successful V3 outcomes do not justify cost vs Lane M alternative. Pre-registered prediction band [1.12, 1.18] kills Lane G in all three outcomes.

**Falsifiable prediction (only if V3 ever runs as null-measurement):** at `--kl-distill-weight 5e-6`, `--kl-distill-temperature 2.0`, single seed, contest-CUDA auth ∈ [Lane A − 0.03, Lane A + 0.03] = [1.12, 1.18]. Inside band → KL is a no-op (most likely outcome). Below 1.12 → marginally helpful but still cost-dominated by Lane M. Above 1.18 → actively harmful.

**Kill criteria during V3 (if ever launched):**
1. Proxy KL term magnitude diverges by >100× from pre-launch estimate within first 50 steps.
2. Proxy total scorer-hinge term *increases* between step 0 and step 100.
3. Vast.ai instance loading exceeds 10 min (per `feedback_vastai_correct_launch_pattern`).
4. Hard cost cap $1.00.

**Recommended pivot order (replaces all Lane G investment):**

| # | Lane | Cost | Predicted Δ vs Lane A 1.15 | Status |
|---|------|------|----------------------------|--------|
| 1 | **Lane M** (radial-zoom 1-DOF, FoE (256,174)) | $0.30 | −0.15 to −0.30 → 0.85–1.00 | UNTESTED, code on disk, COUNCIL #1 PRIORITY |
| 2 | **Lane M+** (zero-cost poses, archive bytes win) | ~$0 (local) | −0.01 to −0.02 (rate only) | UNTESTED, code on disk, run after Lane M |
| 3 | **Lane N** (L∞ pose penalty per Fridrich principle 3) | $0.10 | small positive | one-line code change pending |
| 4 | Lane G V3 (deferred null-measurement) | $0.85 | band-inside most likely | DEFER — only run after Lane M lands AND budget surplus exists |

**Why this matters:** The "lift competitor's training-stage trick into our inference-stage pipeline without re-deriving why it would work in our setting" pattern is a recurring failure mode (Hinton T² adaptive weights, eval_roundtrip default, KL distill as primary loss). Adding it to memory as a binding pattern via the council's deliberation. See findings.md for the full 5-position record.

**2026-04-27 Round-2 forensic verdict (UPDATES Round-1 ABANDON):** **VERDICT: (b) BUGGED — fix the bug + retry**. `kl_distill_segnet_only` (`src/tac/losses.py:688`) uses `F.kl_div(reduction="batchmean")` on a `(B, 5, 384, 512)` tensor, which divides by **B alone** instead of per-pixel-mean. This silently over-weights the KL term by exactly H × W = 196,608×. The empirically-discovered "safe" Lane G v3 weight `5e-6 ≈ 1/196608` is the operator implicitly compensating for the bug via the loss weight — a unit-error masquerading as a hyperparameter. Round-1 reasoned over a 196,608× measurement artifact and reached the right *cost* verdict (Lane M still better EV) but the wrong *structural* verdict (KL was never properly tested at pose TTO; v1/v2 were optimizing a 5000×-overweighted aux term). Council vote 5/5 to (1) FIX THE BUG IMMEDIATELY (one-line change at losses.py:688, mirror the canonical pattern at lines 622+646) — it also affects every training profile that sets `kl_distill_weight > 0` (DEN/SHIRAZ/WILDE: post-fix, weight=1.0 is approximately the old buggy weight=5e-6 regime, so profiles need re-tuning before any retraining lane launches); (2) RE-LAUNCH Lane G V3 ONCE post-fix at weight 1.0 with the same pre-registered prediction band [1.12, 1.18] as a clean falsifiability test. **Next action:** commit reduction fix + add regression test (KL magnitude < 10× scorer hinge for typical inputs); separately, run V3-post-fix as pre-registered measurement after Lane M lands. Full council deliberation in `.omx/research/findings.md` under "## 2026-04-27 Council forensics: Lane G — really dead, or bugged?".

## RE-TEST REQUIRED — invalidated by 2026-04-27 dead-resolver bug class

The R5 codex review + scanner (commit 040030df) confirmed 12 dead-resolver sites in `src/tac/experiments/train_renderer.py` for profile fields the bootstrap was supposed to read but never did. Already-fixed: `pose_dim` (FiLM disabled across SHIRAZ/DEN/WILDE/GREEN, fixed 0746a803/46e2ab6d), `segnet_uncertainty_weighted_loss` (undefined import, added 46e2ab6d), `args.uncertainty_loss_floor` (dead resolver, fixed 46e2ab6d). Still warn-only in preflight (resolvers NOT yet added): `blend_mode` (spatial→scalar fallback), `motion_type` (depth_aware→learned_cnn fallback), `noise_mode`, `beta_start`/`beta_end`. The findings below all measured a model that was missing one or more of these features at training time.

Order: biggest predicted score impact first.

### SHIRAZ v4 = 2.70 vs baseline 0.90 — "181K renderer is fundamentally undermatched"
- **Original conclusion** (`.ralph/run_log.md:67`): "SHIRAZ v4 lane is dead. Even with proper poses, the 181K-param renderer scores 3× worse than the dilated h64 + CRF50 baseline. PoseNet specifically is 24× worse — the architecture is fundamentally undermatched."
- **Why invalid**: SHIRAZ profile declares `pose_dim=6` (FiLM), `blend_mode="spatial"`, `motion_type="depth_aware"`, `use_uncertainty_loss=True`, `uncertainty_loss_floor=0.1`. ALL FIVE were silently masked at train time. The model that scored 2.70 had no FiLM, scalar (not spatial) blending, learned_cnn (not depth_aware) motion, and no uncertainty-weighted loss. The "architecture ceiling" verdict measured a misconfigured stub of SHIRAZ, not the actual design.
- **Re-test config**: `pipeline.py compress --profile shiraz` once `blend_mode/motion_type` resolvers land in `parse_args`. With FiLM already wired (46e2ab6d), partial re-test possible now if the spatial-blend / depth-aware-motion paths are not load-bearing for the conclusion — but full re-test requires the warn-only fields to be live.
- **Predicted outcome**: PoseNet drops 5-15× (FiLM is the canonical PoseNet conditioning trick, validated by Quantizr at 0.33). SegNet drops 1.5-2.5× (uncertainty + spatial blend both target boundary pixels). New score: 0.6-1.2 range [advisory only]. Still likely loses to 0.90 baseline on rate, but no longer "fundamentally undermatched."
- **Cost / lane priority**: $0.80 4090 (3h full pipeline). Lane B+ (after Lane A rate attack lands). Run after `blend_mode/motion_type` resolvers are added.

### DEN-V2 (88K Quantizr-class capacity) — declared dead before measurement
- **Original conclusion** (`.ralph/run_log.md:80-83`, `docs/BATTLE_PLAN.md:72`): "DEN-V2 deploy patching — sunk cost. The score of a smaller renderer wouldn't beat 0.90 anyway." DEN-V2 launched but never produced a verified contest-CUDA score before being shelved as a "PairGenerator interface refactor" issue.
- **Why invalid**: DEN profile declares `pose_dim=6`, `blend_mode="spatial"`, `motion_type="depth_aware"`, `use_uncertainty_loss=True`, `uncertainty_loss_weight=0.02`, `uncertainty_loss_floor=0.1`. All silently masked. Quantizr's 0.33 at 88K params PROVES this capacity class is enough — but only with FiLM + the right loss shaping. Killing DEN on "smaller arch can't win" while FiLM was off was a category error.
- **Re-test config**: `pipeline.py compress --profile den` once `blend_mode/motion_type` resolvers land. Use `use_zoom_flow=False` (current value) to keep loadable; `DEN-V2` (with use_zoom_flow=True) is a separate test once ego_flow plumbing is built.
- **Predicted outcome**: 88K params + FiLM + uncertainty loss + KL distill + DCT-quant + Fridrich aux losses is the closest published recipe to Quantizr's 0.33. Plausible 0.45-0.75 first try.
- **Cost / lane priority**: $1.25 4090 (5h full pipeline). Highest expected delta per dollar — moves us from "scaling up dilated-h64" to "matching Quantizr's compact architecture." Lane B+ priority.

### WILDE/WILDE_V2 — never produced a verified score, training conclusions all suspect
- **Original conclusion** (`.ralph/run_log.md:402, 415`): "WILDE: A100 SXM4, hinge + freeze/unfreeze + Fridrich, GT targets (unplanned A/B)" — used as the "empirical" arm of the WILDE-vs-SHIRAZ A/B. FLAG 1 (GT vs TTO targets A/B) and FLAG 2 (Phase 1 plateau at 181K params) were both raised against WILDE outcomes that ran without FiLM/uncertainty/spatial-blend/depth-aware-motion.
- **Why invalid**: Same as SHIRAZ — WILDE declares all five masked fields. Plus `use_variance_noise=True`, `kl_distill_weight=1.0`, `kl_distill_temperature=2.0` (these went through their own resolver fix on 38a250b8; verify whether the variance_noise/KL paths were actually live during the WILDE A100 run). The "Phase 1 plateau at 181K" FLAG 2 cannot diagnose architecture ceiling vs disabled-feature ceiling.
- **Re-test config**: `pipeline.py compress --profile wilde` after resolver fixes. The WILDE-vs-SHIRAZ A/B (PCGrad+focal_ste vs freeze/unfreeze+xent) becomes meaningful only when both arms have FiLM + uncertainty active.
- **Predicted outcome**: Similar to SHIRAZ (same architecture). 0.6-1.2. The A/B between WILDE and SHIRAZ likely needs to be re-run wholesale; previous "WILDE GT targets vs SHIRAZ TTO targets" comparison is meaningless when both runs were missing the same five features.
- **Cost / lane priority**: $0.80 4090 (3h). Run paired with SHIRAZ re-test for the A/B to be useful. Lane C priority (cheaper to wait for the Lane B SHIRAZ result first; if SHIRAZ alone closes the gap, WILDE arm may be redundant).

### GREEN — zoom-flow advantage hypothesis was uninterpretable
- **Original conclusion** (`.ralph/run_log.md:402`): "GREEN: A100 SXM4, WILDE + use_zoom_flow, TTO targets" — the zoom-flow-vs-not A/B against WILDE. No published verified score; results were "Phase 1 plateau" inconclusive.
- **Why invalid**: GREEN inherits WILDE via `**WILDE` spread. ALL five masked fields apply. The zoom-flow hypothesis (rank-1 radial advantage) cannot be tested when the FiLM that's supposed to gate the residual correction was disabled. Hotz's analytical zoom prediction (project_posenet_rank1_discovery) was specifically meant to be tested against a FiLM-conditioned baseline.
- **Re-test config**: `pipeline.py compress --profile green` after resolver fixes. Pair with WILDE re-test for the zoom-flow A/B.
- **Predicted outcome**: With FiLM active, zoom-flow either wins (Hotz's rank-1 hypothesis correct) or loses (FiLM already captures the rank-1 signal and zoom-flow is redundant). Either result is informative; current data is neither.
- **Cost / lane priority**: $0.80 4090. Lane C (paired with WILDE re-test).

### "FiLM didn't help" implicit findings across all 5-phase QAT lanes
- **Original conclusion** (multiple `.ralph/run_log.md` entries 2026-04-25→04-26): every 5-phase QAT lane (SHIRAZ/WILDE/DEN/GREEN/various V2 variants) underperformed the 0.90 dilated-h64 baseline. The implicit pattern conclusion was "FiLM-class architectures don't beat well-trained vanilla on this task."
- **Why invalid**: Every one of those lanes declared `pose_dim=6` (FiLM) but ran with FiLM disabled. The pattern measured "everything except FiLM" lanes losing to a baseline that itself was trained from PROVEN_BASELINE (no FiLM declared). It is now an open question whether FiLM-class architectures can beat 0.90 — we have never actually run the test.
- **Re-test config**: any one of the above profiles after `pose_dim` resolver fix landed (already done in 0746a803/46e2ab6d). The first FiLM-active retrain that completes is the first real data point on this question.
- **Predicted outcome**: Bayesian update — Quantizr's 0.33 at 88K + FiLM is strong evidence FiLM-class wins. Predicted score from any FiLM-active retrain: 0.4-0.8 range, materially better than the misconfigured 2.70-2.80 results.
- **Cost / lane priority**: Folded into the SHIRAZ / DEN / WILDE re-tests above — no separate spend needed.

### Uncertainty-weighted loss "doesn't help" — never actually executed
- **Original conclusion** (implicit across SHIRAZ/WILDE/DEN profile comments and council notes): the `use_uncertainty_loss=True` + `uncertainty_loss_weight=0.05` + `uncertainty_loss_floor=0.1` configuration was kept "light to avoid amplifying KL distill signal" — but no published result actually compared on-vs-off because the resolver was dead until 46e2ab6d.
- **Why invalid**: `segnet_uncertainty_weighted_loss` was an undefined import (silent ImportError → silent skip) AND `args.uncertainty_loss_floor` was a dead resolver. Profiles declared the loss but it was never applied during training. ScanNet-style spatial uncertainty weighting (Yousfi #5) is empirically validated in the steganalysis literature; we have zero data on whether it helps our renderer.
- **Re-test config**: A/B inside the SHIRAZ or DEN re-test — one run with `use_uncertainty_loss=True`, one with False. Cheaper than a fresh standalone test.
- **Predicted outcome**: Small but positive. Council quote ("with KL on, uncertainty loss is 80% redundant; keep it ≤0.05 or drop entirely") was a prediction, not a measurement. Actual delta: ±0.02-0.05 score.
- **Cost / lane priority**: Free if folded into SHIRAZ/DEN re-test as an inner A/B (same training budget). Lane C.

### Why this matters

Several months of "X didn't help" / "Y didn't help" / "architecture is undermatched" findings may have been measuring misconfigured runs rather than the actual feature being attributed the failure. The pattern: profile declares feature → resolver missing → train_renderer silently uses default → run produces bad score → bad score gets attributed to the feature that was never actually active. Until every one of the 12 dead-resolver sites is closed and a regression test introspects the target argparse, every new "X profile underperformed" conclusion is suspect.

Full audit trail: `feedback_dead_resolver_violations_20260427.md` (memory) and the scanner output in commit 040030df.

**Blocking constraint**: the `blend_mode` / `motion_type` / `beta_start` / `beta_end` resolver gaps are still warn-only in preflight. They MUST have resolvers added to `parse_args` before any of the re-tests above are launched, or the same silent-default bug recurs and we waste another $5-10 of GPU time measuring the same misconfigured stub. The `pose_dim` and `uncertainty_loss_floor` fixes (already landed) are necessary but not sufficient.

## 2026-04-27 Council verdict: Lane F (FP4 QAT) — BUGGED, not dead

**Vote 5/5: (b) BUGGED — fix the bug + retry.** Full deliberation in `.omx/research/findings.md` under "## 2026-04-27 Council audit: Lane F regression — bugged or dead?". Summary:

Lane F's first-round verdict ("regression vs baseline 2.29, abandon") was conditioned on a measurement artifact, not the actual FP4 effect. Three concrete bugs:

1. **CRITICAL — pose threading missing.** `experiments/qat_finetune.py` has no `--poses` CLI arg. The trainer auto-discovers `experiments/results/gt_poses.pt` / `upstream/gt_poses.pt`, neither of which is the load-bearing `optimized_poses.pt` artifact. Lane F's QAT trained against **zero poses** (preflight WARNed but didn't raise), then bundled `optimized_poses.pt` at archive time. Per memory `project_baseline_poses_load_bearing`, renderer + poses are JOINT artifact; pose-init mismatch → 23× PoseNet degrade. Lane F's PoseNet 0.247 → 0.391 (+58%) is the pose-bug fingerprint; SegNet stayed at floor (0.00365 ≈ baseline). FP4 weight quantization of 290K params can account for at most +0.020 PoseNet noise — observed +0.144 is two orders of magnitude over budget.
2. **Significant — under-trained QAT.** 50 epochs × batch_size=4 = 200 pair-samples vs 600 in the dataset (67% of pairs unseen). 47s wall time confirms 1 batch/epoch; consistent with smoke-run not real-experiment compute. Recommended re-launch: `--fp4-epochs 500` (10×, ~10 min wall, $0.10).
3. **Strategic — wrong baseline anchor.** Lane F was anchored to baseline 2.29; our verified best is Lane A 1.15. Even a perfectly-fixed Lane F-on-baseline lands at predicted 2.22 (still 0.87 worse than Lane A). The interesting experiment is **FP4 on Lane A** (predicted 1.05-1.30 [contest-CUDA], a measured rate win or marginal regression worth knowing).

**Recommended next action:**

| # | Lane | Cost | Predicted Δ vs Lane A 1.15 | Status |
|---|------|------|----------------------------|--------|
| 1 | **Lane F-V2** (FP4 QAT on Lane A renderer + Lane A poses, after `--poses` CLI fix lands) | $0.30 | predicted [1.05, 1.30] | BLOCKED on `qat_finetune.py --poses` arg + regression test |

**Required code changes BEFORE re-launch (estimated ~30 min):**
- Add `--poses` CLI arg to `experiments/qat_finetune.py` after line 633.
- Replace lines 706-718 (auto-discovery block) so explicit `--poses` is checked first; if `pose_dim > 0` and no poses found anywhere, **raise** rather than warn.
- New regression test `src/tac/tests/test_qat_finetune_pose_wiring.py` introspecting the argparse and asserting one-step training-loss differs between `--poses` set vs unset.
- New bootstrap script `scripts/remote_lane_f_fp4_qat_on_lane_a.sh` (copy of `remote_lane_b_fp4_qat.sh` with checkpoint, poses, archive paths repointed to `experiments/results/lane_a_landed/`).

**Falsifiability prediction (pre-registered):** Lane F-V2 contest-CUDA score ∈ [1.05, 1.30]. Inside band → FP4-on-Lane-A is a measured rate win/marginal-regression. Below 1.05 → outright win; investigate. Above 1.30 → FP4 noise steeper than expected at low PoseNet baselines; revisit codebook + epoch budget.

**Process lesson added to memory perimeter:** this is the *second* "declared dead, found bugged" in 2 days (Lane G yesterday, Lane F today). The forbidden pattern: voting "abandon" on a single subagent's regression result without auditing every `--*` flag against the target tool's argparse AND every config-derived auto-discovery path for "does that file actually exist on the target host." Future "abandon" verdicts that skip this audit must be marked `SUSPENDED PENDING AUDIT`, not `ABANDON`. Filed as `feedback_silent_default_masquerading_as_negative_result`.

## 2026-04-27 Lane S landed: Self-Compressing renderer codec ready to launch

**Status: ENGINEERING COMPLETE — ready for first Vast.ai dispatch.**

Lane S adapts Szabolcs Csefalvay's self-compression idea (arXiv 2301.13142, *"Self-Compressing Neural Networks"*) — already proven on the standalone postfilter in `src/tac/self_compress.py` — to the dilated-h64 baseline renderer. Per-channel learnable bit-depth via STE + Lagrangian rate penalty drives the average bit-depth from 8 (init) toward 2.5 (target) during training. Result: per-channel bit allocation rather than the uniform-4-bits FP4 scheme.

### What landed (working tree, uncommitted):

- `src/tac/self_compress.py` — `SelfCompressingConv2d` extended with stride/groups/padding_mode kwargs; new helpers `swap_renderer_convs_with_self_compress`, `list_self_compress_layers`, `renderer_total_weight_bits`, `compute_renderer_rate_penalty`, plus the protected-name pattern list (`SC_PROTECTED_NAME_PATTERNS`).
- `src/tac/renderer_export.py` — new `SCv1` magic format: header JSON (per-channel bit-depths + arch) + LZMA-compressed body. `export_self_compressed_renderer` / `load_self_compressed_renderer` round-trip is byte-exact (verified). `detect_checkpoint_type` and `load_any_renderer_checkpoint` recognize the new magic.
- `submissions/robust_current/inflate_renderer.py` — new `b"SCv1"` dispatch branch (CRITICAL tier). Uses the canonical `tac.renderer_export.load_self_compressed_renderer`; raises a clear RuntimeError if tac is missing (no silent fallback).
- `src/tac/profiles.py` — two new profiles:
  - `self_compress_renderer_smoke` (100 epochs, target_bits=4.0, no auth-eval — code-path validation only)
  - `self_compress_renderer_full` (1980 epochs, target_bits=2.5, full Quantizr-style 5-phase schedule, mirror of `dilated_h64_half_frame` for direct A/B)
- `src/tac/experiments/train_renderer.py` — argparse + resolver wiring for `--use-self-compress-codec` and the four tuning knobs (`--self-compress-init-bits`, `--self-compress-target-bits`, `--self-compress-lambda-start`, `--self-compress-lambda-end`, `--self-compress-lambda-ramp-start-frac`); post-build SC swap; in-loop Lagrangian rate penalty; bit_depth clamping after each opt step; **automatic disable of `--auth-eval-on-best` when SC is on** (FP4A export would lose all SC gain).
- `src/tac/tests/test_self_compress_renderer.py` — 24 tests covering SC primitives, swap helper, SCv1 export/load round-trip, byte-stability, FP4-vs-SC byte comparison, inflate dispatch source + e2e load, Lagrangian rate-penalty correctness, and profile sanity.

### Architecture decision: which layers stay FP32?

Per Lane F's PoseNet regression finding (FP4-QAT on dilated-h64 caused +0.144 PoseNet vs floor) and the standalone postfilter experiments showing FiLM is "3rd most scorer-sensitive," the swap helper PROTECTS these layers (kept FP32):

- `renderer.head` (final RGB conv before sigmoid output) — PoseNet sensitive
- `motion.head` (flow/gate/residual head) — sub-pixel warp coordinates
- `*.fuse_conv` / `*.fuse2_conv` (1×1 skip-mixing convs) — per-pixel sensitive
- `film_*.scale` / `film_*.shift` (FiLM linears) — tiny + scorer-sensitive
- All `nn.ConvTranspose2d` (up_conv, up2_conv) — STE backward through stride-2 transposed conv is ill-behaved + small param count

For the dilated-h64 baseline (288K params): **16 layers swapped (243K params, 84%), 3 layers protected (FP32), 1 ConvTranspose2d skipped (FP32).**

### Predicted byte counts (verified by smoke-test export of the same arch):

| Variant | Bytes | bits/param | Note |
|---|---|---|---|
| Float baseline (FP32) | ~1.15 MB | 32.0 | not shippable |
| FP4-QAT (uniform 4 bits) | ~144 KB | 4.00 | current frontier |
| **SCv1 @ 2.5 mean SC bits** | **~115 KB** | **3.20** | -20% vs FP4 |
| SCv1 @ 2.0 mean SC bits + 25% pruning | ~92 KB | 2.55 | aggressive |

(SCv1 byte count exceeds raw `bits/param × n_params / 8` because protected FP16 layers + ConvTranspose contribute ~50KB regardless of SC bit budget — that's ceiling unless those layers are also redesigned.)

### Recommended Lane S launch config (first dispatch):

| Param | Value | Rationale |
|---|---|---|
| Profile | `self_compress_renderer_full` | 1980 epochs Quantizr-style 5-phase; mirror of dilated_h64_half_frame |
| GPU | RTX 4090 on Vast.ai | $0.25/hr; ~5h wall = ~$1.25 |
| `target_bits` | 2.5 | matches Szabolcs's 2301.13142 §4.3 image-compression target |
| `init_bits` | 8.0 | full precision at init; rate penalty drives down |
| `lambda_start` | 0.0 | no rate pressure during phase-1 anchor |
| `lambda_end` | 1.0 | full rate pressure by phase-3 joint |
| `lambda_ramp_start_frac` | 0.30 | 30% of training at init_bits before pressure |
| Mask format | half-frame (Quantizr trick) | ~125 KB AV1 vs 250 KB full-frame |
| Pose artifact | reuse Lane A's optimized_poses.bin | ~15 KB |
| **Predicted archive** | **~255 KB** | renderer 115 + masks 125 + poses 15 |
| **Predicted score band** | **0.85-1.10 contest-CUDA** | best-case beats baseline by 0.05; worst-case +0.20 |

### Falsifiability checks for the first launch (kill criteria):

1. **Phase 1 end (~1h)**: pixel L1 < 12 AND avg SC bits == 8.0 (lambda not active yet).
2. **Phase 2 mid (~3h)**: avg SC bits monotonically decreasing AND scorer < 8.0.
3. **Phase 4 end (~5h)**: avg SC bits ≈ target_bits ± 0.5 AND fp4_scorer < 1.5 AND total renderer.bin < 130KB.
4. **Auth eval (separate launch using SCv1 inflate path)**: contest-CUDA score < 1.20.

If any of #1-#3 fail, kill the run; the rate penalty + SC machinery has a configuration bug.
If #4 fails (contest > 1.20), Lane S is BUGGED, not DEAD — audit pose threading + protected-layer set first per the Lane F-V2 protocol.

### Engineering risks (rank-ordered):

1. **The protected-layer set may be insufficient.** If PoseNet regresses more than expected (e.g. > 0.10 vs baseline 0.011), the candidate next-protections are: `*.bottleneck.conv2` (the deepest residual block), `motion.up_conv`. Add the next layer to the protected list, retrain, re-measure.
2. **The Lagrangian schedule may need re-tuning.** If avg SC bits doesn't reach target by phase-4 end, increase `lambda_end` to 2.0. If it crashes too fast (avg < 1.5 by phase-2), reduce `lambda_end` to 0.5.
3. **The SC pack/unpack assumes inner Conv2d state.** This works because `swap_renderer_convs_with_self_compress` always creates the SC layer with an inner `nn.Conv2d`. If a future arch path replaces inner conv with a different module, the swap and the export must be jointly updated.
4. **The `bit_depth.bits` tensor is in the optimizer's param list (it's an `nn.Parameter`).** When `lr_bits` differs from `lr_weights` (as in the standalone postfilter trainer), separate param groups are needed. The current train_renderer.py uses ONE optimizer for all params — this means SC bit-depth uses the same LR as the conv weights (typically 1e-3 → 5e-5 across phases). That is FINE for the rate penalty schedule (the penalty's gradient magnitude is much smaller than scorer loss, so a smaller `lr_bits` is automatic via the gradient itself). Future work: split into two param groups for finer control.

### Cost & ETA for first dispatch:

- **Setup**: bootstrap script + smoke probe ~10 min
- **Training**: 1980 epochs on RTX 4090 ≈ 5h
- **Auth eval**: separate dispatch using `inflate_renderer.py SCv1 dispatch` + `evaluate.py` ~30 min
- **Total**: ~6h wall, ~$1.50 on Vast.ai

### Concurrency-safe to ship:

- Lane S engineering work is in NON-OVERLAPPING files with the in-flight Lane F audit (`experiments/qat_finetune.py`, `src/tac/quantization.py`, `scripts/remote_lane_b_fp4_qat.sh` — all read-only here).
- All changes uncommitted in working tree per task instructions.
- 24 new tests passing; 7 existing Lane I dispatch tests still passing; original `self_compress.py` smoke still passing; original `renderer_export.py` smoke still passing.
