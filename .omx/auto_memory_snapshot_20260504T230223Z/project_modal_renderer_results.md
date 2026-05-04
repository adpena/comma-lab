---
name: Modal Renderer Training Results (2026-04-12/13)
description: First asymmetric warp training completed. Best 1.88 proxy at ep4000. Auto-killed at ep4418. Moonshot aborted after 1 epoch.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Run 1: asymmetric_warp_t4 (36,60) medium — COMPLETED + AUTO-KILLED

- **App ID**: ap-kaBjTn3TNabOJFkOgaYBID
- **Config**: base_ch=36, mid_ch=60, 287,019 params, FP4 ~140KB
- **GPU**: T4, 1.8 hours wall-clock
- **Epochs**: 4418 (auto-killed at ep4418, best at ep4000)
- **Phase transition**: Phase 1 → Phase 2 at epoch 4000 (40% of 10000)

### Eval scores (full evaluation, not training logs):
| Epoch | SegNet | PoseNet | Rate | Score |
|-------|--------|---------|------|-------|
| 3600 | 0.00601 | 0.14947 | 0.003822 | **1.9187** |
| 3800 | 0.00639 | 0.15357 | 0.003822 | 1.9740 |
| 4000 | 0.00624 | 0.13531 | 0.003822 | **1.8828** (BEST) |
| 4200 | 0.01252 | 0.18953 | 0.003822 | 2.7239 (regressed) |
| 4400 | 0.00849 | 0.16062 | 0.003822 | 2.2123 (regressed) |

### Key observations:
- **Rate is 6x better than postfilter** (0.004 vs 0.023) — worth 0.48 points
- **SegNet already at 0.006** — matches postfilter floor
- **PoseNet is the bottleneck** — 0.135 vs 0.057 (postfilter local)
- Phase 2 Lagrangian caused rho explosion → loss divergence → auto-kill at ep4418
- Gate mean dropped from 0.99 → 0.37 (learning to trust warp over residual)
- Flow magnitude stayed low (~0.007) — model prefers residual over flow

### Score decomposition at best (ep4000):
- 100 × 0.00624 = 0.624 (SegNet)
- sqrt(10 × 0.13531) = 1.163 (PoseNet) ← THIS is where we lose
- 25 × 0.003822 = 0.096 (Rate) ← THIS is our advantage
- **Total: 1.883**

### What went wrong:
- Phase 2 rho_growth=1.02 compounded too fast → rho hit 10000 cap → Lagrangian exploded
- Only 418 Phase 2 epochs before kill — not enough time for constraints to converge
- PoseNet was still high (0.135) because Phase 1 doesn't optimize PoseNet directly

## Run 2: asym_24_40_moonshot (24,40) small — ABORTED

- **App ID**: ap-IcZtit6gx53LdkU8BiYplC
- **Config**: base_ch=24, mid_ch=40, 155,167 params
- **Status**: Ran ONE epoch then client disconnected
- **Reason**: Forgot `--detach` flag
- **Artifacts**: Only config.json and replicability.json saved

## Volume: tac-asymmetric-results
- asymmetric_warp_t4/: 15 files (best checkpoint + periodic saves)
- asym_24_40_moonshot/: 2 files (config only)
