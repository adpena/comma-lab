---
name: Competition State (2026-04-15, UPDATED)
description: Contest-compliant auth=0.61. Unlimited-compute auth=0.37. Quantizr leads at 0.33. 12 days left.
type: project
originSessionId: docs-agent-20260415
---

## Score Baselines (2026-04-15)

**Lane 1: Contest-Compliant (priority)**
- **Our best auth: 0.61** (distillation ep300 AND pose-space TTO — both achieved 0.61)
- Distillation ep900 (running): proxy 0.338, auth eval pending (~0.47 projected)
- Pipeline: masks.mkv -> distilled renderer + optimized poses -> raw RGB
- FP4 export done: 170 KB renderer (saves 0.085 rate points)
- Archive with FP4 + CRF30 masks: 215 KB total

**Lane 2: Unlimited-Compute (research/paper)**
- **TTO v7 auth: 0.37** (hinge loss, 500 steps, all 600 pairs)
- PoseNet=0.00250, SegNet=0.00094, Rate=0.00489
- Cannot be submitted (TTO takes ~40 min on 4090, exceeds 30-min budget)

**Leaderboard:**
- Quantizr: **0.33** (PR#55, April 19) — FiLM on pose, depthwise sep, eval-matched resize
- Us: **0.61** [contest-compliant, submitted informally via auth eval]
- neural_inflate (PR#49): 1.89

## Key Results This Session
- Pose-space TTO: 94.7% PoseNet reduction via 6D FiLM conditioning. Auth 0.61.
- Distillation v2: proxy 0.807→0.338 (ep900). Auth 0.61 at ep300.
- FP4 quantization: 297 KB → 170 KB. -0.085 rate points. Free improvement.
- Gradient corrections: DEAD. 743 KB for 20 frames → ~44 MB full scale.
- MiniSegNet (h=32): PASSES 98.7%. MiniPoseNet: FAILS (R²=0.002).
- Insight: FiLM conditioning space has PoseNet-SegNet orthogonality.

## Infrastructure
- Vast.ai: ~$9 spent, ~$15 remaining. Primary compute.
- Modal: deprioritized. Use for auth eval if needed.
- Local M5 Max: development, smoke tests.

## 12 days to May 3 deadline
