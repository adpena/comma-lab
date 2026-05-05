---
name: TTO v3 Results — Embedding Loss "FAILED" (was gradient bug, VINDICATED by v5b)
description: v3 embedding loss appeared to fail but was caused by THE GREAT GRADIENT BUG. v5b with working gradients scored auth 0.41 (best ever).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## TTO v3 Results (2026-04-15) — INVALIDATED

v3 embedding loss appeared to fail (proxy 0.6219 vs v1's 0.5896). But this was caused by
THE GREAT GRADIENT BUG — `@torch.no_grad` on upstream `rgb_to_yuv6` killed ALL PoseNet
gradients. v3 had ZERO PoseNet signal. The "failure" was the bug, not the technique.

## v5b VINDICATION (2026-04-15)
With working gradients (make_scorers_differentiable), embedding loss is STRICTLY SUPERIOR:

| Metric | v5a (output MSE) | v5b (embedding 512D) |
|--------|-----------------|---------------------|
| Auth Score | 0.43 | **0.41** |
| PoseNet | 0.00295 | **0.00263** (-10.8%) |
| SegNet | 0.00160 | **0.00148** (-7.5%) |
| Proxy | 0.294 | **0.263** (-10.5%) |

**Embedding loss is the path forward.** 512D provides 85x more gradient directions than 6D output MSE.
