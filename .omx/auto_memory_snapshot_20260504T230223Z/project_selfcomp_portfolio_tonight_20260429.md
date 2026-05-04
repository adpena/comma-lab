---
name: Selfcomp portfolio tonight — 14 lanes ~$50 / 50h to sub-0.3
description: 2026-04-29 PM portfolio after Selfcomp 0.38 RE + grand council. 14 lanes total in implementation/dispatch. Goal sub-0.3 NON-NEGOTIABLE. Selfcomp's grayscale-LUT mask + analytical-pose affine + block-FP 1.017bpw + 94K SegMap = paradigm shift; we're forking + stacking on top.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Score frontier**:
- Quantizr 0.33 (leader)
- Selfcomp 0.38 (NEW #2, our paradigm template)
- Mask2mask 0.60 (#3)
- Lane G v3 1.04 (our best)
- **GOAL: sub-0.3, NON-NEGOTIABLE per user 2026-04-29**

**14-lane Selfcomp paradigm portfolio**:

Tier 1 (encoder-only quick wins):
- Lane MM grayscale-LUT — RUNNING fc-01KQCZV4 — predicted 0.65-0.85
- Lane FR-MM variable-σ Gaussian-LUT — pending Subagent C — $1/3h
- Lane SH Shannon arithmetic on qints — pending Subagent D — $1/3h
- Lane TR temporal residual coding — pending Subagent D — $2/4h
- Lane PD pose deltas — pending Subagent D — $2/4h

Tier 2 (SegMap clones + KL distill):
- Lane SA SegMap clone — RUNNING fc-01KQD090 — predicted 0.40-0.55
- Lane SC++ SA + KL distill T=2.0 — RUNNING fc-01KQD092 — predicted 0.30-0.40 (sub-Quantizr)
- Lane PA Pose-as-Affine init — pending Subagent D — $0.50/2h, predicted 0.40

Tier 3 (sub-0.3 frontier moonshots):
- Lane SO SC + Hessian block-FP — RUNNING fc-01KQD093 — predicted 0.27-0.35
- Lane FR-Ω Fridrich-cost block-FP — pending Subagent C — $4/10h, predicted 0.28
- Lane HM-S 8-DOF homography (replaces 6-DOF affine) — pending Subagent C — $4/10h, predicted 0.32
- Lane DARTS-S arch search Selfcomp skipped — pending Subagent C — $6/18h, predicted 0.27
- Lane WC-S Curator outlier weighting — pending Subagent C — $3/8h, predicted 0.28
- Lane FC FiLM-Canvas (Quantizr+Selfcomp combo) — pending Subagent D — $5/12h, predicted 0.32

**Orthogonal in flight (yesterday's dispatches, sub-0.5 path)**:
- q_faithful_v3 (Quantizr 1:1 replica)
- sz_phase2_v2 (moonshot dilated)
- mae_v_v2 (mask-augment)
- lane_w_v2 (hard-pair self-compress)

**Critical paradigm shifts** (from Selfcomp RE):
1. Single-channel grayscale mask via Gaussian-LUT (vs 3-ch discrete)
2. Single-mask-per-pair + 6-DOF affine duality (vs 2 masks per pair)
3. Analytical pose via affine_delta (vs PoseNet predictor)
4. Block-FP weight self-compression at 1.017 bpw (vs FP4 4 bpw)
5. 94K-param SegMap (vs our 287K-param ASYM)

**Round 1 council review findings** (2026-04-29 ~11am):
- 3 CRITICAL bugs caught BEFORE Modal training wasted $20:
  - inflate_segmap.py clamp(0,1)*255 → all-zero frames (FIXED c78995f0)
  - remote_lane_so non-existent pack_payload kwarg (FIXED c78995f0)
  - noise_std dead-code in eval_roundtrip_chain (FIXED c78995f0)
- 3 lanes cancelled mid-flight, re-dispatched as v2 with fixes
- 4 Medium issues queued for Round 2

**How to apply**:
- Continue tight monitoring on all 11+ in-flight Modal lanes
- 3-clean-pass adversarial review gate per CLAUDE.md before any lane promoted to submission
- Stacking goal: SC++ (0.33) + SO Hessian (-0.03) + FR-Ω (-0.04) + DARTS-S (-0.05) + WC-S (-0.02) → 0.19 territory if all stack additively (likely overlap, realistic 0.25-0.30)
- Watch PR #56 selfcomp for any compress-side code release
- Re-fetch leaderboard before submission to detect competitor moves
