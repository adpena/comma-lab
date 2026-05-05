---
name: Proxy MUST Always Simulate Resize — Non-Negotiable
description: compute_proxy_score without simulate_resize produces 5000x wrong PoseNet. Default must be True everywhere.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The proxy scorer WITHOUT `simulate_resize` produces PoseNet values ~5000x higher than auth eval. This is because the contest pipeline applies a resolution roundtrip (384→1164→uint8→384) that acts as a low-pass filter, smoothing high-frequency warp artifacts.

**Discovery**: Distillation TTO on Vast.ai 4090 showed proxy PoseNet=158.97 (baseline) while auth eval shows 0.031. Proxy SegNet=0.504 while auth=0.002. The proxy without resize simulation is directionally useful but quantitatively meaningless.

**Why:** `simulate_resize=False` evaluates at native 384×512 where warp artifacts are visible. `simulate_resize=True` replicates the contest's upscale→quantize→downscale roundtrip, making proxy scores match auth within ~10-20%.

**How to apply:**
- `simulate_resize` should default to `True` in compute_proxy_score (currently defaults to False)
- ALL experiment scripts should pass `--simulate-resize` by default
- ALL Vast.ai experiment configs should include `--simulate-resize`
- When comparing proxy to auth, ONLY use simulate_resize=True proxies
- Consider making simulate_resize=True the hardcoded default with no flag to disable
