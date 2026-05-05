---
name: 🚨 LEADERBOARD ALL AT 0.32-0.33 — Shannon floor MUST be hit or we're irrelevant
description: 2026-05-01 user intelligence update. The contest leaderboard has converged on 0.32-0.33 score. Quantizr's previously-known 0.33 is no longer alone — multiple competitors are at this band. Our current best (0.9974 [contest-CUDA RTX 4090] orthogonal stack) is 0.66 score-points ABOVE the leaderboard. We MUST hit the Shannon theoretical floor (~0.28 derived) or be irrelevant. Strategic reframe required: incremental sub-frontier moves are no longer enough — need paradigm-level jumps.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## The brutal truth

| Anchor | Score | Distance to 0.32 leaderboard |
|---|---|---|
| Lane G v3 baseline | 1.05 | +0.73 |
| PFP16 frontier | 1.044 | +0.72 |
| β Fisher OWV3 | 1.016 | +0.70 |
| R7 OWV3 | 1.013 | +0.69 |
| 0120 OWV3 | 1.0024 | +0.68 |
| **0120 + arith poses (current best)** | **0.9974** | **+0.68** ⚠️ IRRELEVANT GAP |
| Shannon theoretical floor | ~0.28 | -0.04 |

Our current 0.9974 is **0.68 score-points behind** the 0.32 leaderboard. Closing that gap requires a 67% rate reduction OR equivalent distortion improvement — neither is achievable by stacking 1-2% incremental wins from arithmetic-coded poses or per-stream tweaks.

## What this means strategically

1. **Incremental sub-frontier wins are NO LONGER ENOUGH.** The orthogonal-stack 0.005 gain we celebrated is ~1% of what we need. 100 more such wins = ~0.5 → still not at leaderboard.

2. **Paradigm-level jumps are the only path.** We need ONE of:
   - **Custom decoder with massive overfit** (frame-index → embedding NN, score-relevant pixel sparse encoding, RL-searched archive atoms) — 5-10× rate reduction possible
   - **NeRV/INR mask + renderer codec** that replaces both AV1 + FP4 simultaneously
   - **Hash the GT video at compress time** (allowed per CLAUDE.md "unlimited compress compute") + ship a tiny lookup decoder
   - **Match Quantizr's exact paradigm** (88K-param FiLM-conditioned DSConv renderer + 600-frame mask subset + KL distill T=2.0) — we have the pieces but never assembled them tight

3. **Comma-prior-lossless-challenge precedent (arithmetic coding)** still applies but needs to be applied AT THE PARADIGM LEVEL, not just to one stream. The winning team likely combined arithmetic coding with EVERYTHING (mask geometry, residual frames, learned priors, custom decoder).

4. **Build Discipline tradeoff**: the OSS/paper/production constraints (AGENTS.md `377cf144`) still matter past May 3 — but if we ship a 0.99 archive into a 0.32 contest, we're not advancing the field, we're embarrassing it. Ship something genuinely competitive.

## Revised next moves (in EV order)

1. **Match Quantizr's known paradigm exactly first** (deterministic local build):
   - 88K-param FiLM-conditioned DSConv renderer (we have impl)
   - 600-frame mask subset with frame-warping (we have prior memory on this — Quantizr ships only 600 odd masks)
   - KL distill T=2.0 + asymmetric pair generation
   - FP4 + Brotli renderer
   - Full archive ~300KB (Quantizr's measured 293KB)
   - Predicted score: 0.33 (matches leaderboard band)
   - **EV: very high — this CLOSES the gap to baseline-leaderboard with ZERO new innovation**

2. **Add arithmetic-coded mask stream on top of #1** (current in-flight subagent ae5890d12b24ea442):
   - Predicted -100KB further → ~200KB total → score ~0.27
   - **This is the path to BEAT the leaderboard, not just match it**

3. **Score-relevant-pixel sparse encoder** (PoseNet/SegNet Jacobian-driven): potentially -200KB more if score-relevance is sparse → score ~0.20-0.25

4. **Joint-ADMM cross-stream coordinator** to find Pareto-optimal allocation across arithmetic-coded mask + sparse-encoded pixels + arithmetic-coded weights

5. **Lightning T4 promotion of EVERY landed sub-frontier** (codex-owned) — needed for A++ contest grade

## The deadline math

T-56h to May 3 9 AM. If we land ONE Quantizr-paradigm-match (~6h work), then arithmetic-coded mask (~6h, in flight), then sparse encoder (~12h training + eval), we could land sub-0.33 by ~T-32h. Tight but feasible.

## Cross-refs

- `project_lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501.md` (current best, now strategically insufficient)
- `reference_arithmetic_coding_won_comma_lossless_challenge_20260501.md` (technique, must apply at paradigm level)
- `feedback_codex_partner_coordination_state_20260501T1310Z.md` (codex Lightning T4 queue + Alpha matrix)
- `feedback_fast_chip_directive_no_waiting_20260501.md` (H100 SXM for any heavy training)
- `AGENTS.md` "Build Discipline" (commit 377cf144) — still applies, but the bar moved
