---
name: Theoretical Limits and Time Budget (Karpathy/Qwen/DeepSeek)
description: Absolute score floors, time budget breakdown, int8 bottleneck threshold, Pareto frontier, optimal temperature schedule
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Score Floors
- 3-layer h=64 CNN floor: **~1.37-1.42** (RF=15x15 is the bottleneck, not params)
- 45KB int8 optimal architecture: **~1.25-1.30** (need larger RF via dilation/PixelShuffle)
- Unconstrained model: **~0.875-0.975** (rate term 0.575 is irreducible)

## Time Budget (30 min GitHub Actions)
- Inflate: **~8-15 seconds** (trivial)
- Scorer: **~3-5 minutes** (PoseNet+SegNet on 600 pairs, batch=16)
- Total warm: **~10-14 min** (18+ min headroom)
- h=96 fits in TIME but NOT in 45KB size limit
- Multi-pass CNN (run twice): feasible (~20s total), doubles effective depth
- TTA (flip+average): NOT applicable (GT not available at inflate time)

## Training Bottleneck
- SegNet forward+backward: **~70-80%** of per-step training time
- Eval fraction at eval_every=5: **~35%** of wall time
- Error replay at every 200 epochs: negligible (~0.9s/epoch amortized)

## Critical Thresholds
- **Int8 becomes bottleneck below score ~1.40** — fine corrections (~1-2 pixel) hit quant noise
- PoseNet clamp (min=0.001) currently **NOT active** (pose=0.01229, threshold 12x away)
- EMA staleness: at ema_decay=0.997, half-life is 231 epochs — by ep 2500, EMA ≈ model copy
- LR at eta_min=1e-5: weight updates below int8 step size after ep ~2000

## Optimal Temperature Schedule
- Linear T=5→1 is suboptimal (hyperbolically increasing gradient)
- **Exponential decay is provably better**: T(t) = T_start * exp(-3*t/N)
- For T_start=5, N=2500: gives T_end ≈ 0.25
- Recommended: T_end=0.3-0.5 with exponential schedule

## Pareto Frontier
- At current operating point: 1 unit pose improvement = 0.1427 units seg improvement
- Both can improve simultaneously until CNN capacity saturates (~seg≈0.004, pose≈0.008)
- We are NOT on the Pareto boundary yet — both have room

## Key Insight: RF is the Real Bottleneck
- h=64 dilated RF=15x15 covers only ~2.5% of scorer input at 256x192
- PixelShuffle at half-res with dilation gives effective RF up to 64x64 (~25% coverage)
- This is why the competitor's PixelShuffle gets better SegNet (0.00434 vs our 0.00610)
