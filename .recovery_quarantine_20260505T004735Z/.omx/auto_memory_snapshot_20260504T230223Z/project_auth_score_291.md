---
name: Auth Score 2.91 Contest-Compliant
description: First honest contest-compliant auth score through full inflate→evaluate pipeline. PoseNet 75% of score. 835x proxy-auth gap on poses.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Auth score 2.91** [CONTEST-COMPLIANT] via upstream evaluate.py on 2026-04-24.

| Component | Value | Contribution | Quantizr | Gap |
|-----------|-------|-------------|----------|-----|
| SegNet | 0.00353 | 0.353 | ~0.0013 | 2.7x |
| PoseNet | 0.476 | 2.18 | ~0.0004 | 36x |
| Rate | 0.01503 | 0.376 | 0.200 | 1.9x |
| TOTAL | | 2.91 | 0.33 | 8.8x |

Archive: 564KB (renderer 170KB + masks_crf50 421KB + poses 7KB, all Brotli Q11)

**Why:** PoseNet proxy-auth gap is 835x (proxy 0.00057, auth 0.476). Optimized poses overfit to proxy distribution. The eval_roundtrip chain (384→874→uint8→384) destroys the pose signal. Need QAT + proper training to close this gap. 

**How to apply:** PoseNet is 75% of total score. Without fixing PoseNet, SegNet and rate optimizations are marginal. Need: (1) WILDE/SHIRAZ training with eval_roundtrip from epoch 0, (2) QAT to preserve quality through FP4, (3) fresh pose TTO on QAT model. Target: PoseNet < 0.05 (contrib < 0.71), SegNet < 0.003, rate < 0.25 → score < 1.2.
