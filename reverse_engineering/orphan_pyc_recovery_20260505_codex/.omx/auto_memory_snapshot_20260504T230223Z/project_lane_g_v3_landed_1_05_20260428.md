---
name: Lane G v3 LANDED at 1.05 [contest-CUDA] — NEW FRONTIER (beats Lane A 1.15 by 0.10)
description: 2026-04-28 Vast.ai instance 35733155 (ssh9:13154) completed Lane G v3 (KL-distill weight=0.002 + pose TTO retry on Lane A baseline anchor). Final score 1.05; PoseNet=0.003455 (Lane A 0.005), SegNet=0.004008 (Lane A 0.0046), Rate=0.0185 (same archive bytes as Lane A). Both distortion components improved while rate held flat — KL distill recipe transferred from Quantizr works. Archive harvested to experiments/results/lane_g_v3_landed/.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Score breakdown (verified [contest-CUDA])

```
Final score:    1.0500
PoseNet dist:   0.003455  (Lane A: 0.005000 → -31% better)
SegNet dist:    0.004008  (Lane A: 0.004600 → -13% better)
Rate (unscaled): 0.018486  (Lane A: 0.018486 → identical)
Archive bytes:  694,074
n_samples:      600

Score components:
  100 * seg     = 0.401
  sqrt(10*pose) = 0.186
  25 * rate     = 0.462
  TOTAL         = 1.049 → 1.05
```

## Provenance (from contest_auth_eval.json)

- **archive_sha256**: `9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b`
- **inflate_sha256**: `ddfa90816c96488aa14a2cd65e6485adff936184873d9271c0abf75c4d4ef4b2`
- **GPU**: NVIDIA GeForce RTX 4090, driver 580.126.09
- **torch**: 2.5.1+cu124, CUDA 12.4
- **archive path**: `/workspace/pact/lane_g_v3_results/archive_lane_g_v3.zip` (694,074 bytes)
- **Eval started**: 2026-04-28T11:36:55Z
- **Vast.ai instance**: 35733155 (ssh9.vast.ai:13154)
- **Cost**: $0.31/hr × 4.9h ≈ $1.50

## What worked

- **KL-distill weight=0.002 corrected** (post-bug-fix from earlier KL-bug-14000× incident in v1/v2)
- **Pose TTO retry** on Lane A's anchor archive
- Both PoseNet AND SegNet improved at same rate budget — KL distill is doing real work

## Stacking implications

Lane G v3 IS Lane A + KL-distill-aux + pose-TTO-retry. Composes orthogonally with:
- Lane W / Lane Ω-V2 (renderer rate attack — different bytes)
- Lane SAUG-V2 (proxy/auth-gap-closing during training, would compound)
- Lane WC (independent typicality weighting)
- Lane HF (Telescope foveation — orthogonal post-processing)
- Lane MAE-V (joint mask-aug from epoch 0 — different training)

Predicted Lane G v3 + Lane W stack: ~0.90 [contest-CUDA] (extrapolating Lane W's predicted 0.85-1.05 from the new 1.05 floor).

Predicted Lane G v3 + Lane SAUG-V2 stack: ~0.75 [contest-CUDA] if SAUG-V2 closes proxy/auth gap effectively (SAUG's predicted band [0.70, 1.00]).

## Local artifacts

- `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip` (694KB)
- `experiments/results/lane_g_v3_landed/contest_auth_eval.json` (provenance)

## Cross-references
- `project_verified_baseline_2_29` — old baseline before Lane A
- `project_council_eurekas_driving_geometry_20260428` — Lane G v3 was relaunch with KL-bug fix
- `project_lane_taxonomy_stacking_strategy_20260427` — composition matrix
- `feedback_proxy_auth_math_useless` — why we always do contest-CUDA auth eval
