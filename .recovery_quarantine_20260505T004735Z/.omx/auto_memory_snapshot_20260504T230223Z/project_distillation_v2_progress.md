---
name: Distillation v2 Progress — Proxy 0.446 at Epoch 300, Still Converging
description: pose_weight=10 from warm checkpoint. PoseNet 0.007, SegNet 0.002. On track for sub-0.40 by epoch 500.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Distillation v2 (2026-04-20, Vast.ai RTX 4090)

**Config**: Resume from Phase 2 checkpoint (0.518), pose_weight=10.0, hinge, eval roundtrip, FiLM pose_dim=6
**Cost so far**: ~$1.70 (336 min at $0.302/hr)

### Trajectory
| Epoch | PoseNet | SegNet | Best Proxy |
|-------|---------|--------|------------|
| 0 | 0.032 | 0.0025 | 0.807 |
| 50 | 0.017 | 0.0024 | 0.596 |
| 100 | 0.012 | 0.0023 | 0.544 |
| 200 | 0.009 | 0.0021 | 0.492 |
| 300 | 0.007 | 0.0020 | 0.446 |

### Key findings
- pose_weight=10 was THE critical fix (4x faster early convergence vs pose_weight=1)
- Renderer SegNet (0.0020) now better than original auth eval (0.00217)
- PoseNet 76% reduced from start (0.032 → 0.007)
- Still converging — no plateau detected
- Proxy 0.446 captures ~60% of TTO quality (proxy 0.195)
- Projected: epoch 500 → ~0.40, epoch 1000 → ~0.36

### Comparison
- TTO proxy (unlimited): 0.195 — still 2.3x better
- Quantizr auth: 0.33 — we need proxy ~0.30 to match (auth is typically worse than proxy)
- The gap between renderer and TTO represents the capacity bottleneck (288K params vs 707M pixels)

### What closes the gap
- Per-pair latent codes (9.6KB) — per-pair adaptability
- Postfilter stacking (45KB) — residual correction
- FP4 quantization — rate savings
- Longer training — still converging at epoch 300
- Mini-scorer inflate TTO (SegNet-only, PoseNet via stored targets)
