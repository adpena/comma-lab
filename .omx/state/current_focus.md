# Current Focus -- 2026-04-15T23:30:00Z

## Session 35: DX Hardening + Data Permanence + Pair Difficulty Map

### Completed This Session
- **Hinge loss** (P0): Implemented in tac, registered as experiment
- **Two-phase TTO** (P1): Implemented (100 steps PoseNet, then SegNet-only)
- **simulate_resize default**: Fixed (now True by default, matching auth eval)
- **Cosine LR**: Empirically worse for TTO (step_curve_cosine experiment)
- **check_vastai.py**: Canonical Vast.ai interaction script with cost tracking,
  idle detection, SSH key verification, deploy bundles, destroy confirmation
- **download_modal_tto_frames.py**: Data permanence script for Modal TTO frames
- **PROVENANCE.md**: Full provenance documentation for all experiment results
- **Pair difficulty map**: COMPLETED on MPS (600 pairs, 41.1s)
  - Output: `experiments/results/pair_difficulty/pair_difficulty_map.json`
  - Visualization: `experiments/results/pair_difficulty/pair_difficulty_distribution.png`
- **tto_v6_hinge_phase2**: Registered in Vast.ai experiment registry
- **All experiment scripts verified**: --help passes for all 5 deployment scripts

### URGENT: Download TTO Frames from Modal
The highest-quality TTO frames (500-step) are on Modal volume:
- `asym_v5_lagrangian_fixed/tto_v5a_output_mse/tto_frames.pt` (auth 0.43)
- `asym_v5_lagrangian_fixed/tto_v5b_embedding/tto_frames.pt` (auth 0.41)

Run: `python scripts/download_modal_tto_frames.py`

These MUST be downloaded before Modal access expires.

## Paradigm Shift: SegNet is 77x More Important Than PoseNet

The step curve experiment (Vast.ai RTX 4090, 30 pairs) revealed:
- PoseNet saturates at 100 TTO steps (165.27 -> 0.042, 3970x reduction)
- SegNet contributes 98.7% of remaining score after PoseNet convergence
- Leverage ratio: 77:1
- 500-step breakthrough: SegNet finally moves (0.5036 -> 0.3435)

## Pair Difficulty Map Results (NEW)

600 pairs analyzed with simulate_resize=True:
- PoseNet MSE: mean=158.98, std=12.96, range=[85.75, 199.55]
- SegNet disagree: mean=0.505, std=0.006, range=[0.490, 0.519]
- Hard pairs (top 20%): 120 pairs with PoseNet MSE > 168.81
- Easiest pair: #514 (score=79.74), hardest: #523 (score=95.43)
- Data enables adaptive TTO budget: skip easy pairs, allocate more to hard

## Scores
- **Renderer baseline**: auth=0.87 (seg=0.21, pose=0.56, rate=0.10)
- **TTO v5a (gradient fix)**: auth=0.43 (first valid TTO with PoseNet gradients)
- **TTO v5b (embedding)**: auth=0.41 (10.8% improvement over v5a)
- **Target**: sub-0.20 auth

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

## Deadline
- May 3, 2026 (~18 days remaining)
