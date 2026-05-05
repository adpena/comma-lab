---
name: Shannon's Insights — Operationalized by Council
description: Entropy coding worth 0.027 for renderer path (pure weights archive), only 0.007-0.017 for codec path (97% video). Infrastructure exists. Polish phase priority.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Shannon's Key Insight
We're not coding at the entropy rate. FP4 weights have structure (near-zero values, per-layer statistics) that arithmetic coding could exploit.

## Council's Correction
- **Codec path (current 1.33):** Archive is 97% H.265 video (already entropy-coded). Weight compression saves 0.007-0.017 points. LOW priority.
- **Renderer path (training now):** Archive is 100% model weights (~140KB). Entropy coding saves ~0.027 points. MEANINGFUL when target is sub-0.50.

## Decisions
- Shannon added to council as advisory (information theorist, no veto)
- CompressAI: DO NOT ADD (our arithmetic coder in entropy_archive.py is sufficient)
- FP4 vs int8 A/B test: AFTER training converges
- Entropy coding: polish phase (last 3 days before deadline)
- Infrastructure ready: src/tac/entropy_archive.py tested and integrated

## The Deeper Insight (validated)
Our architecture IS a learned predictive codec:
- Warp = predictive component (high mutual information, near-zero bits)
- Gated residual = innovation signal (actual new information)
- This is theoretically optimal for task-aware video compression
