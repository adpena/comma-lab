---
name: CRITICAL — Proxy-Auth PoseNet Drift Gets WORSE with Training (11x at ep3560)
description: Longer training overfits to proxy artifacts. PoseNet proxy-auth ratio went from 2.1x to 11.1x. ep1000+TTO (auth 0.36) may be better than ep3560 (auth 0.59).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The Problem (2026-04-21)

PoseNet proxy-auth ratio INCREASES with training:
- ep300: 2.1x (proxy 0.007 → auth 0.01454)
- ep3560: 11.2x (proxy 0.00146 → auth 0.01628)

The renderer is learning texture patterns that satisfy the PROXY scorer (bilinear
simulate_resize) but get DESTROYED by the AUTH scorer (DALI NVDEC hardware decode).

## Why This Happens
- simulate_resize uses bilinear F.interpolate (approximate)
- Auth eval uses DALI NVDEC (different rounding, different pixel values)
- At very low PoseNet values, even 0.5 pixel difference causes large MSE change
- The renderer finds proxy-specific textures that don't survive DALI's decode path

## Implication
- LONGER TRAINING IS COUNTERPRODUCTIVE after the proxy-auth gap widens
- ep1000 + pose TTO (auth 0.36) is our ACTUAL best, not ep3560 (auth 0.59)
- The optimal checkpoint is NOT the latest one — it's the one with best AUTH
- We need auth eval at ep500, ep1000, ep1500, ep2000, ep2500, ep3000 to find the sweet spot

## The "Best of All Worlds" Strategy
- Use the checkpoint with best AUTH SegNet (ep3560: 0.00060)
- Fix PoseNet with pose TTO (proven 90-99% improvement)
- The best COMBINED score comes from the right checkpoint + pose TTO
- May need to auth-eval 5-6 checkpoints with and without pose TTO
