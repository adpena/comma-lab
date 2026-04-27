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

## RE-TEST REQUIRED — invalidated by 2026-04-27 dead-resolver bug class

The R5 codex review + scanner (commit 040030df) confirmed 12 dead-resolver sites in `src/tac/experiments/train_renderer.py` for profile fields the bootstrap was supposed to read but never did. Already-fixed: `pose_dim` (FiLM disabled across SHIRAZ/DEN/WILDE/GREEN, fixed 0746a803/46e2ab6d), `segnet_uncertainty_weighted_loss` (undefined import, added 46e2ab6d), `args.uncertainty_loss_floor` (dead resolver, fixed 46e2ab6d). Still warn-only in preflight (resolvers NOT yet added): `blend_mode` (spatial→scalar fallback), `motion_type` (depth_aware→learned_cnn fallback), `noise_mode`, `beta_start`/`beta_end`. The findings below all measured a model that was missing one or more of these features at training time.

Order: biggest predicted score impact first.

### SHIRAZ v4 = 2.70 vs baseline 0.90 — "181K renderer is fundamentally undermatched"
- **Original conclusion** (`.ralph/run_log.md:67`): "SHIRAZ v4 lane is dead. Even with proper poses, the 181K-param renderer scores 3× worse than the dilated h64 + CRF50 baseline. PoseNet specifically is 24× worse — the architecture is fundamentally undermatched."
- **Why invalid**: SHIRAZ profile declares `pose_dim=6` (FiLM), `blend_mode="spatial"`, `motion_type="depth_aware"`, `use_uncertainty_loss=True`, `uncertainty_loss_floor=0.1`. ALL FIVE were silently masked at train time. The model that scored 2.70 had no FiLM, scalar (not spatial) blending, learned_cnn (not depth_aware) motion, and no uncertainty-weighted loss. The "architecture ceiling" verdict measured a misconfigured stub of SHIRAZ, not the actual design.
- **Re-test config**: `pipeline.py compress --profile shiraz` once `blend_mode/motion_type` resolvers land in `parse_args`. With FiLM already wired (46e2ab6d), partial re-test possible now if the spatial-blend / depth-aware-motion paths are not load-bearing for the conclusion — but full re-test requires the warn-only fields to be live.
- **Predicted outcome**: PoseNet drops 5-15× (FiLM is the canonical PoseNet conditioning trick, validated by Quantizr at 0.33). SegNet drops 1.5-2.5× (uncertainty + spatial blend both target boundary pixels). New score: 0.6-1.2 range. Still likely loses to 0.90 baseline on rate, but no longer "fundamentally undermatched."
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
