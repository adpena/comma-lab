# Current Focus -- 2026-04-15T18:00:00Z

## Paradigm Shift: SegNet is 77x More Important Than PoseNet

The step curve experiment (Vast.ai RTX 4090, 30 pairs) revealed a fundamental
reordering of priorities:

- PoseNet saturates at 100 TTO steps (165.27 -> 0.042, 3970x reduction)
- SegNet contributes 98.7% of remaining score after PoseNet convergence
- Leverage ratio: 77:1 (every 0.001 SegNet reduction = 77x more valuable than same PoseNet reduction)
- 500-step breakthrough: SegNet finally moves at high step counts (0.5036 -> 0.3435)

## Scores
- **Renderer baseline**: auth=0.87 (seg=0.21, pose=0.56, rate=0.10)
- **TTO v5a (gradient fix)**: auth=0.43 (first valid TTO with PoseNet gradients)
- **Target**: sub-0.20 auth

## Score Decomposition (auth=0.43, v5a)
```
Score = 100*seg + sqrt(10*pose) + 25*rate
  SegNet dominates at our operating point (77:1 leverage over PoseNet)
  The binding constraint is SegNet, not PoseNet
```

## Step Curve Results (Complete)
```
Steps    PoseNet      SegNet      Score    s/frame
    0    165.268      0.5036      91.02     0.000
   10    104.759      0.5036      82.73     0.056
   25     74.732      0.5036      77.70     0.101
   50     43.290      0.5036      71.17     0.185
  100      0.042      0.5036      51.01     0.356  <-- PoseNet phase transition
  150      0.038      0.5036      50.98     0.525
  200      0.111      0.5036      51.41     0.692
  300      0.028      0.5036      50.89     1.044
  500      0.025      0.3435      34.85     1.711  <-- SegNet breakthrough
```

## Active Work
1. **Step curve**: COMPLETE. Results saved to experiments/results/step_curve_v1/
2. **Distillation TTO targets**: Being generated on Vast.ai Instance B (12/60 batches)
3. **Per-pair difficulty map**: Script ready (experiments/pair_difficulty_map.py)
4. **Paper update**: Section 4.7 added with step curve analysis

## Key Strategic Insight
The path to sub-0.20 runs through SegNet, not PoseNet:
- PoseNet is already near-zero after 100 TTO steps
- SegNet improvement at 500 steps (32% reduction) proves it CAN be optimized
- Adaptive TTO budget: 100 steps for all pairs (PoseNet), then 500+ for hard SegNet pairs
- SegNet architectural improvements may yield bigger gains than any TTO strategy

## Deadline
- May 3, 2026 (~18 days remaining)
