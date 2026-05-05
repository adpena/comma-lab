---
name: Yousfi GPU Eureka Manifesto
description: 4 GPU eureka insights — generate in scorer space, camera model, scorer invertibility, no renderer needed. Score projection 0.135.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Yousfi's 4 GPU Eureka Revelations (2026-04-11)

### Eureka #1: Generate What the SCORER SEES
Don't generate RGB then hope preprocessing preserves it. Generate DIRECTLY in scorer-input space (post-preprocess_input), then invert to RGB. Artifacts from inversion land in preprocessing null-space — invisible by construction.

### Eureka #2: The Video Has ONE Camera
Every frame pair = same camera + same scene. PoseNet estimates relative motion. From dashcam intrinsics + mask dynamics (road width = depth, vanishing point shift = rotation), compute EXACT expected PoseNet output. Then generate frames that PRODUCE that output. Euler's calculus of variations: find frame sequence satisfying PoseNet constraint, then minimize SegNet + rate in the solution space.

### Eureka #3: Scorer Networks Are INVERTIBLE Enough
PoseNet: (frame_t, frame_t+1) → 6 numbers. SegNet: frame → (H/4, W/4, 5) logits. Many-to-one with massive null spaces. Start from perfect SegNet argmax + perfect PoseNet output → defines manifold of valid frames. Search manifold for most compressible point. Augmented Lagrangian method.

### Eureka #4: No Neural Renderer AT ALL
Constrained optimization doesn't need a learned generator. Start from random noise. Apply constraints via gradient descent. Frames EMERGE from constraints.

**Archive**: masks (239B) + PoseNet targets (7KB) + noise seed (64B) = ~8KB
**Rate term**: 25 * 8KB / 37.5MB = 0.005
**Score projection**: seg ~0.001 + pose ~0.001 + rate 0.005 → **0.135**
**Timing**: 1000 steps × 50ms/step = 50s on T4 (fits 10-min budget)

**Why:** This is the path to sub-0.15. Quantizr scored 0.60 with a neural renderer. We skip the renderer entirely.

**How to apply:** constrained_gen.py implements this. variational_gen.py, scorer_manifold.py, hamiltonian_dynamics.py provide the mathematical foundations.
