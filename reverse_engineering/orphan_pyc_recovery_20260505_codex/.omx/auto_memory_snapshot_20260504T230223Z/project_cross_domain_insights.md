---
name: Cross-Domain Insights — Robotics, NVS, 3DGS, Flow, Quantization
description: Council research on cross-domain techniques. Top picks: ego-motion precompute (TartanVO), learned entropy coding, depth conditioning, occlusion masks. 3DGS not viable as codec but useful for insights.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Highest-leverage unused techniques (ranked by impact/effort)

1. **Ego-motion pre-computation** — TartanVO/DPVO at compress time → store 6-DOF trajectory → PoseNet becomes nearly deterministic. 2 days, -0.3 to -0.5 PoseNet.
2. **Learned entropy coding on weights** — arithmetic coding with learned prior on model weights. 1 day, -0.1 rate (20-30% archive shrink).
3. **Depth as conditioning channel** — ZoeDepth at compress time → uint8 depth maps as side info. 1 day, -0.05 to -0.2 PoseNet.
4. **Occlusion mask** — computed from depth + ego-motion → tells renderer WHERE warping fails. 0.5 day, -0.05 both.

## Key frameworks to explore
- CompressAI (InterDigital) — entropy coding modules for weight compression
- RAFT-lite — correlation volume for better motion estimation
- AWQ — activation-aware weight quantization (from LLM quantization)

## 3DGS/NeRF verdict
- NOT viable as full codec in 21 days
- Useful INSIGHTS: depth conditioning, occlusion masks, explicit ego-motion
- Post-competition: Street Gaussians, 4D-GS, NeRV as video codec

## Key principle
Unlimited compute at COMPRESS time, constrained at INFLATE time.
Pre-compute everything possible: ego-motion, depth, occlusion masks, saliency.
Store as tiny side-information (few KB). Give the inflate-time renderer a massive head start.
