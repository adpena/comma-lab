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
