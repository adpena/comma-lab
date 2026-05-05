---
name: Session state 2026-04-22 — overnight training + QAT pipeline + Fridrich losses
description: Two 4090s running. Instance 1: small renderer Phase 1 ep375 loss 0.57. Instance 2: loading for QAT of current renderer. 20+ commits, 13 bugs fixed, QAT crash-tested 4 times.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Running GPU Instances (2026-04-22 ~06:00 UTC)

1. **Instance 35404589** ($0.245/hr, ssh2.vast.ai:14588):
   - Phase 1 training: epoch 375/1000, loss 0.57, ~15s/epoch
   - Architecture: 122K params (base_ch=24, mid_ch=32, DSConv, FiLM pose_dim=6)
   - All fixes: eval_roundtrip=True, noise_std=0.5, hinge loss, pose_weight=10
   - Phase 2 is 3000 epochs — MUST reduce to 1000 when Phase 1 finishes
   - Post-training script ready: experiments/vastai_post_training.sh

2. **Instance 35409315** ($0.248/hr, ssh6.vast.ai:19314):
   - Loading — for QAT on CURRENT renderer (288K)
   - Script ready: experiments/qat_current_renderer.sh
   - Will produce: FP4 renderer (170KB) + optimized poses

## Key Data Points

- Current renderer baseline distortion: 0.394 (20 pairs via upstream DistortionNet)
- Mask CRF50: 421KB, 99.78% accuracy, pose_d=0.19 with stale poses
- Mask CRF56: 280KB, 99.66% accuracy, pose_d=0.26 with stale poses
- FP4 without QAT: 170KB but kills PoseNet (score 9.77 vs 2.01)
- QAT on MPS: BLOCKED by nn.utils.parametrize backward stride incompatibility
- FP4A inflate path: VERIFIED working

## Score Projections

| Path | Archive | Rate | Distortion | TOTAL |
|------|---------|------|-----------|-------|
| Path 1: QAT current + CRF50 | 607KB | 0.40 | ~0.40 | ~0.80 |
| Path 2: Small QAT + CRF56 | 348KB | 0.23 | ~0.20 | ~0.43 |
| Quantizr | 293KB | 0.20 | ~0.13 | 0.33 |

## Outstanding Work

1. Monitor instance 2, sync code, launch QAT for current renderer
2. When Phase 1 finishes on instance 1: kill, restart Phase 2 (1000 epochs)
3. Morning: download both sets of results, build archives, full e2e eval
4. If score > 0.40: add Fridrich texture losses, retrain
