---
name: 835x Proxy-Auth Gap
description: Pose TTO produced 0.00057 proxy but 0.476 auth — 835x gap. NEVER celebrate proxy scores. Predictions based on proxy are unreliable.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Pose TTO proxy predicted score 0.140. Actual auth score: 2.91. PoseNet proxy 0.00057 → auth 0.476 = 835x gap.

**Why:** Proxy TTO optimizes at 384x512 internal resolution. Auth evaluates through the full resize chain (384→874→uint8→384). PoseNet (FastViT-T12, YUV6 input) is extremely sensitive to quantization artifacts in this chain. Our TTO-optimized poses overfit to the proxy distribution.

**How to apply:**
1. NEVER make predictions or celebrate based on proxy scores
2. ALWAYS run upstream/evaluate.py before reporting any score
3. When projecting outcomes, add 5-10x uncertainty on PoseNet proxy→auth gap
4. Pose TTO must be done WITH eval_roundtrip=True AND noise_std=0.5 to be meaningful
5. The only valid measurement is: archive.zip → inflate_renderer.py → upstream/evaluate.py → score
