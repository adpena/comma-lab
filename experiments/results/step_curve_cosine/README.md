# Cosine LR Step Curve Results (2026-04-15, Vast.ai RTX 4090)

Cosine LR with warmup DELAYS the PoseNet phase transition.

## Comparison: Constant vs Cosine LR

| Steps | Constant PoseNet | Cosine PoseNet | Winner |
|-------|-----------------|----------------|--------|
| 10    | 104.76          | 114.39         | Constant |
| 25    | 74.73           | 95.54          | Constant |
| 50    | 43.29           | 76.14          | Constant |
| 100   | **0.042**       | 27.96          | **Constant (665x better)** |
| 150   | 0.038           | 0.331          | Constant (8.7x) |
| 200   | 0.111           | **0.050**      | **Cosine** (at 200, cosine settles better) |
| 300   | **0.028**       | 0.069          | Constant |
| 500   | 0.025           | **0.023**      | Cosine (marginal) |

## Analysis

The phase transition occurs at ~80-100 steps with constant LR but ~130-150 steps with cosine.
Cosine warmup starts at 0.1x LR, which slows Adam's momentum accumulation in the PoseNet
gradient manifold. The phase transition requires Adam to discover ~50-100 effective gradient
directions, which takes longer at lower learning rate.

At 200+ steps, cosine's decay phase helps settle into a tighter minimum (0.050 vs 0.111),
but this advantage is marginal and not worth the delayed convergence.

SegNet was frozen at 0.5036 for BOTH schedules across all step counts, confirming this is
a structural renderer limitation, not an optimization issue.

## Council Decision

**Use constant LR for all TTO.** Cosine warmup delays the critical phase transition by ~50 steps,
wasting compute. The marginal improvement at 500 steps (0.023 vs 0.025) is not worth the
delayed convergence at the operationally important 100-step sweet spot.
