---
name: Post-0.43 Battle Plan — Push to Sub-0.2 [UNLIMITED-COMPUTE LANE ONLY]
description: Council battle plan after auth 0.43 victory. Three levers. Target sub-0.2. NOTE: This is unlimited-compute lane, not contest-compliant.
type: project
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
## IMPORTANT: This is UNLIMITED-COMPUTE lane (Lane 2)

Auth 0.43/0.41 was achieved with TTO (3+ hours of gradient descent). This CANNOT be submitted
to the contest (30-min inflate budget). The contest-compliant score remains 0.87.

## Starting Point: Auth 0.41 [unlimited-compute]
- SegNet: 0.148 (seg_dist 0.00148)
- PoseNet: 0.162 (pose_dist 0.00263)
- Rate: 0.100 (archive 150KB)

## Priority Experiments (all unlimited-compute lane)
1. **TTO v7 — hinge + eval roundtrip + embedding** (highest expected gain)
2. **TTO v5c — Aggressive PoseNet** (100:1 pose:seg ratio, find ceiling)
3. **Archive compression** (local, free — prune + entropy code to 75KB)
4. **LR schedule** (cosine annealing, after v7 results)
5. **Per-pair adaptive weights** (if variance is high)

## CONTEST-COMPLIANT PATH (separate)
To improve Lane 1 (0.87 -> sub-0.40), need:
- FiLM conditioning on pose in renderer (like Quantizr PR#55)
- Eval-matched resize in renderer training
- Latent codes optimized at compress time (deterministic inflate)
- See project_three_breakthroughs.md for full plan

## Resources
- Vast.ai 4090: $25 (PRIMARY)
- Modal T4: low credits (auth eval only)
- AWS T4: $100 reserve
- Local M5 Max: unlimited (archive, analysis, paper)

## Submit PR: last 5-7 days (April 27-May 3)
