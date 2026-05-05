---
name: Fridrich 2-Path Push — The Path to Beat Quantizr
description: Strategic priority as of 2026-04-12: deploy Fridrich constrained renderer on both CPU and GPU lanes to beat Quantizr's 0.60
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Priority (2026-04-12 evening)

The user's directive: "we must beat quantizr" with "absolute best" from the skunkworks team.

### The Plan
Train DP-SIMS (128,64,32,16) with Fridrich augmented Lagrangian:
- Phase 1: MSE warmup (memorize frames)
- Phase 2: Constrained optimization (seg < 0.005, pose < 0.02)  
- Phase 3: Self-compression (reduce ~489KB FP4 → 200-300KB)

### 2-Path Insight
One archive.zip, one inflate.py, works on BOTH CPU and GPU:
- SegNet extracts masks from GT video (SegNet provided by upstream environment)
- Renderer generates frames from masks (works on either device)
- Bilinear upscale 384x512 → 1164x874

### Score Projection
- seg ≈ 0.003: 100 * 0.003 = 0.30
- pose ≈ 0.002: sqrt(10 * 0.002) = 0.14
- rate ≈ 200KB/37.5MB = 0.005: 25 * 0.005 = 0.13
- **Total: 0.57 (beats Quantizr's 0.60)**

### Training Script Status
`experiments/train_renderer_fridrich.py` — APPROVED by tripartite pact
- All critical bugs fixed (fan_in, resume, flow_reg)
- Ready for deployment to Lightning T4

**Why:** Quantizr proved renderer lane works. Our architecture is comparable but with Fridrich hard constraints (novel optimization, not weighted sum).

**How to apply:** Every decision should be evaluated against "does this help beat Quantizr?" CPU postfilter training continues as insurance but GPU renderer is the primary attack vector.
