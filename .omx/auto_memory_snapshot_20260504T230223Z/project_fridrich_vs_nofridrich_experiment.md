---
name: Fridrich vs no-Fridrich head-to-head experiment (2026-04-23)
description: Two identical 4090 training runs, only difference is Fridrich losses (texture_weight=0.1, linf_weight=0.001). Fridrich wins Phase 1 by 5-7% but may compete with scorer loss in Phase 2. Phase-specific application (Phase 1+3 only) recommended.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Experiment Setup

Both runs: Float + EMA(0.997) + CRF50 masks + 103K params (base_ch=20, mid_ch=28)
+ DSConv + FiLM + CLADE + eval_roundtrip + hinge SegNet + pose_weight=10

| Run | Fridrich | Instance |
|-----|----------|----------|
| #2 | OFF | ssh6:10476 |
| #3 | texture_weight=0.1, linf_weight=0.001 | ssh5:17296 |

## Phase 1 Results (pixel regression)

| Epoch | No-Fridrich best | Fridrich best | Fridrich advantage |
|-------|-----------------|---------------|-------------------|
| 125 | 0.786 | 0.775 | 1.4% |
| 275 | ~0.650 | 0.620 | ~5% |
| 425 | ~0.590 | 0.559 | ~5% |
| 575 | ~0.555 | 0.530 | ~5% |
| 725 | ~0.550 | 0.519 | ~6% |
| 875 | 0.540 | 0.513 | 5% |

**Fridrich consistently 5-7% better in Phase 1.**

## Phase 2 Results (scorer-guided) — EARLY

| Epoch | No-Fridrich best | Fridrich best | Notes |
|-------|-----------------|---------------|-------|
| 1050 | 1.155 | 1.249 | No-Fridrich ahead initially |
| 1100 | ~1.10 | 1.006 | Fridrich catches up |

Phase 2 early results are mixed. No-Fridrich started ahead (lower total loss = more
gradient for scorer) but Fridrich caught up by epoch 1100.

## Council Analysis

- **Fridrich**: Phase 1 + Phase 3 only (pixel regression + hard pairs). OFF in Phase 2.
  Texture-aware loss helps the renderer learn WHERE errors matter.
  But in Phase 2, the scorer already provides that signal — Fridrich is redundant.
- **Hotz**: Wait for final results. Don't change running experiments.
- **Karpathy**: KL distillation as auxiliary Phase 2 loss is the bigger miss.

## Recommendation for Next Training Run
- Fridrich losses in Phase 1 and Phase 3 ONLY
- OFF in Phase 2 (scorer provides the texture signal already)
- Add KL distillation (T=2.0) as auxiliary SegNet loss in Phase 2
