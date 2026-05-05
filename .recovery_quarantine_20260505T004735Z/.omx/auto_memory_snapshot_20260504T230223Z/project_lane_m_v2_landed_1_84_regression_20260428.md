---
name: Lane M-V2 LANDED at 1.84 [contest-CUDA] — REGRESSION (rank-1 hypothesis invalidated again)
description: 2026-04-28 Lane M-V2 (radial-zoom 1-DOF properly engineered with frozen baseline dim 1-5 pads + Fridrich C1 noise/eval-roundtrip) landed contest-CUDA 1.84 vs Lane G v3 1.05 frontier. Improvement vs Lane M+N V1 2.35 but still architectural regression. PoseNet 0.076 (15× worse than Lane A 0.005). Confirms `project_lane_mn_radial_zoom_negative_20260427`: rank-1 PoseNet sensitivity ≠ rank-1 renderer input space. Lane HF (Telescope foveation) is the proper revival path.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Score breakdown (verified [contest-CUDA])

```
Final score:    1.84
PoseNet dist:   0.076027  (Lane A: 0.005000 → 15× worse)
SegNet dist:    0.005055  (Lane A floor, no regression)
Rate (unscaled): 0.018485  (matches Lane A baseline)
Archive bytes:  694,044   (matches G v3)

Score components:
  100 * seg     = 0.506
  sqrt(10*pose) = 0.872
  25 * rate     = 0.462
  TOTAL         = 1.840
```

archive_sha256: `88cde3c8e0bc8348...`

## Comparison to predecessor

| Lane | Score | PoseNet | SegNet | Rate | Note |
|------|-------|---------|--------|------|------|
| Lane M+N V1 | 2.35 | 0.20+ | 0.005 | 0.018 | Original radial-zoom + L∞ |
| **Lane M-V2** | **1.84** | **0.076** | 0.005 | 0.018 | Radial-zoom + frozen baseline pad + Fridrich C1 |
| Lane A baseline | 1.15 | 0.005 | 0.005 | 0.018 | Full 6-DOF pose TTO |
| Lane G v3 | 1.05 | 0.0035 | 0.004 | 0.018 | Lane A + KL distill |

Lane M-V2 improvement vs M+N V1 is meaningful (-0.51) — confirms the C1 fixes were real. But still 0.69 worse than Lane A's frontier and 0.79 worse than Lane G v3.

## Why it failed (architectural)

Per `project_posenet_rank1_discovery`: PoseNet's effective Jacobian rank ≈ 1.008 → only dim 0 carries signal IN PoseNet's OUTPUT. But the renderer is trained on full 6-DOF pose INPUTS (a different subspace). When you feed:
- 1-DOF pose dim 0 + zeros for dims 1-5
- OR 1-DOF dim 0 + frozen baseline dims 1-5

…the renderer's MotionPredictor is operating off-manifold from its training distribution. The motion module's `(e_t1 - e_t).abs()` diff feature was trained on real-world pose noise patterns; constant or near-constant dims 1-5 produce a feature distribution PoseNet's scorer doesn't recognize.

**Conclusion**: the rank-1 sensitivity finding is real for PoseNet's OUTPUT but cannot be exploited via 1-DOF pose INPUT — because the renderer trained on 6-DOF noise needs the noise to render scorer-faithful frames.

## What this implies for the portfolio

- **Lane M+N V1 / V2 / V3** (radial-zoom variants) are now empirically dead. Don't propose more 1-DOF pose attempts.
- **Lane HF (Telescope hyperbolic foveation)** is the proper revival path per `project_cosmos_mae_lyra_telescope_synthesis_20260428` — applies as POST-renderer wrap (not pose hack), so renderer stays on-manifold.
- **Lane GE (geodesic pose)** has the same risk profile — 1-DOF Chebyshev polynomial on dim 0 + zeros. Predicted band [1.05, 1.20] now looks optimistic given M-V2 result. Re-anchor expectation toward [1.30, 1.60].

## Local artifacts

- `experiments/results/lane_m_v2_landed/archive_lane_m_v2.zip` (694KB)
- `experiments/results/lane_m_v2_landed/contest_auth_eval.json`

## Cross-references
- `project_lane_mn_radial_zoom_negative_20260427` — Lane M+N V1 negative result
- `project_posenet_rank1_discovery` — the rank-1 finding
- `project_cosmos_mae_lyra_telescope_synthesis_20260428` — Lane HF revival
- `feedback_dont_abandon_high_score_lanes_for_stacking_20260428` — composability still possible
- `project_lane_g_v3_landed_1_05_20260428` — current frontier
