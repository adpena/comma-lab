---
name: Battle plan 2026-04-23 — Hotz is right, post-training stack is the lever
description: Two 4090s racing (Fridrich vs no-Fridrich). After training: pose TTO + postfilter = 0.874 point net benefit. Ship the winner. Next iteration adds phase-specific Fridrich + KL distillation + half-frame masks.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Current State (2026-04-23)

Two 4090s running head-to-head:
- #2 (ssh6:10476): Float + EMA, no Fridrich — Phase 2 started, proxy ~1.15
- #3 (ssh5:17296): Float + EMA + Fridrich — Phase 2 started, proxy ~1.00

Both: 103K params, DSConv, FiLM, CLADE, CRF50 masks, eval_roundtrip, hinge SegNet.
~8h remaining. Combined cost ~$0.48/hr.

## After Training Completes (the plan)

1. Take the winner (lower Phase 2 proxy)
2. Export ASYM (130KB) — NOT FP4 (QAT proved 8x worse than float)
3. Pose TTO with `--masks submissions/robust_current/masks_crf50.mkv` (~1h on 4090)
4. Postfilter training against hard pairs using difficulty.pt (~2h on 4090)
   - Curriculum: all pairs → top 30% → top 10%
   - Expected: 0.874 net point improvement (0.932 PoseNet - 0.030 rate)
5. Build archive: renderer(130KB) + postfilter(46KB) + masks(421KB) + poses(16KB) = 613KB
6. Full e2e eval: inflate_renderer.py → upstream evaluate.py on Modal T4
7. Ship it

## Why Post-Training Stack > Training Loss Tweaks

The difficulty map shows top 20% of pairs are 227x worse than bottom 80%.
These hard pairs dominate the PoseNet average, which gets sqrt'd in the formula.
Fixing hard pairs via postfilter: PoseNet 1.115 → 0.153 = 0.962 point improvement.
Rate cost of postfilter: 0.030. Net: +0.932 points.

No training loss tweak can match this. The renderer is already good on 80% of pairs.
The bottleneck is the 20% heavy tail, and the postfilter specifically targets it.

## Why Fridrich Losses Work in Phase 1 But Not Phase 2

Phase 1 (pixel regression): L1 loss treats all pixels equally. Fridrich losses add:
- Texture-aware weighting: concentrate L1 on smooth regions (where scorer is sensitive)
- L∞ penalty: spread errors evenly (sqrt law, harder to detect)
This gives the renderer a BETTER pixel-level learning signal than raw L1.

Phase 2 (scorer loss): The scorer ITSELF provides texture-aware gradient signal.
PoseNet gradient is naturally larger on pixels that matter for ego-motion.
SegNet gradient is naturally larger on class boundary pixels.
Fridrich losses are REDUNDANT with scorer feedback in Phase 2 — they compete
for gradient capacity without adding new information.

Phase 3 (hard pairs): Fridrich may help because hard pairs often involve
complex texture boundaries where the scorer signal is noisy. The Fridrich
texture map provides a complementary prior.

## Next Training Iteration (if time allows, 9 days remaining)

1. Fridrich losses Phase 1 + Phase 3 ONLY (off in Phase 2)
2. KL distillation (T=2.0) as AUXILIARY SegNet loss in Phase 2 (Quantizr uses this)
3. Half-frame masks (600 odd frames only → ~210KB instead of 421KB)
4. Smaller model (base_ch=18, mid_ch=24, ~75K params → ~100KB ASYM)
5. Post-hoc FP4 comparison (ASYM vs FP4 on the CRF50-trained model)

## Score Projection

| Config | Distortion | Rate | Total |
|--------|-----------|------|-------|
| Winner + CRF50 poses (no postfilter) | ~0.8-1.2 | 0.37 | ~1.2-1.6 |
| Winner + CRF50 poses + postfilter | ~0.3-0.6 | 0.41 | ~0.7-1.0 |
| Next iteration + half-frame + postfilter | ~0.3-0.5 | 0.28 | ~0.6-0.8 |
| Quantizr | 0.13 | 0.20 | 0.33 |

#2-3 on the leaderboard is realistic. #1 requires more training iterations
and the half-frame mask optimization.
