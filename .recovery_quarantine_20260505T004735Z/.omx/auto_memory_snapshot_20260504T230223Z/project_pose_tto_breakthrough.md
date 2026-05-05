---
name: BREAKTHROUGH — Pose-Space TTO with seg_weight=0 gives 94.7% PoseNet gain at zero SegNet cost
description: 6D FiLM pose space is naturally orthogonal to SegNet boundaries. Optimize ONLY PoseNet. Score -58% on smoke pairs. Contest-compliant.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Discovery (2026-04-20)

Optimizing 6D FiLM pose vectors with seg_weight=0 (PoseNet only):
- PoseNet: -94.7% (0.0057 → 0.0003)
- SegNet: +0.001% (unchanged!)
- Score: -58% (0.316 → 0.133 on 10 smoke pairs)

The key insight: the FiLM pose space is naturally orthogonal to SegNet argmax boundaries.
Including SegNet in the loss was COUNTERPRODUCTIVE — it steered the optimizer into
SegNet-sensitive directions that damaged both metrics.

## Winning config
```
--seg-weight 0 --pose-weight 1 --lr 0.001 --eval-roundtrip --steps 100
```

## Archive impact
Zero — same 14.4KB poses.pt, just with optimized values instead of GT.

## Projected compounding
- Distilled renderer proxy: 0.390 (ep480)
- Pose TTO 58% reduction: ~0.164 proxy
- Even at 30% efficiency at full scale: proxy ~0.273 → auth ~0.37
- At 50% efficiency: proxy ~0.195 → auth ~0.27

## Next step
Run on full 600 pairs on Vast.ai. This is the highest-leverage stacking technique.
