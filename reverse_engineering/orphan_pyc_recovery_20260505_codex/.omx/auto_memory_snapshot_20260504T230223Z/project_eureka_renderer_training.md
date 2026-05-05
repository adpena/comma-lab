---
name: EUREKA — Fridrich Constrained Renderer Training (the path to sub-0.60)
description: Train DP-SIMS renderer with Fridrich hard constraints + coupled trajectory PoseNet loss + self-compression. The pieces are all built.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Breakthrough (2026-04-12 dinner conversation)

CPU lane ceiling is ~1.25 (codec artifacts irrecoverable). Must go renderer lane.

### Why GPU lane smoke tests failed:
- Tiny renderers (78KB) lack capacity for PoseNet temporal dynamics
- Constrained gen from NOISE can't satisfy PoseNet (no temporal structure)
- Modal DP-SIMS got SegNet 0.003 but PoseNet 0.482 (no coupled training)

### What Quantizr does:
- 386KB renderer in archive (masks → frames)
- Trained end-to-end against scorer
- Works on BOTH CPU and GPU
- PoseNet 0.00066 (excellent temporal coherence)

### The fix: Fridrich constrained RENDERER TRAINING

Train DP-SIMS (128,64,32,16) = 1M params = ~489KB FP4 with:
1. **Hard constraint: seg < 0.005** (DP-SIMS already achieves this)
2. **Hard constraint: pose < 0.01** (forces temporal coherence via Lagrangian)
3. **Coupled trajectory loss** (PoseNet evaluates consecutive PAIRS, gradient flows through both frames)
4. **Ego-motion flow constraint** (geometric prior on frame-to-frame motion)
5. **Self-compression** (reduce 489KB → 200-300KB for better rate)

### Score projection:
- Self-compress to 200KB: rate = 0.13
- seg ≈ 0.003: 100 × 0.003 = 0.30
- pose ≈ 0.002 (with ego-motion): sqrt(10 × 0.002) = 0.14
- **Total: 0.30 + 0.14 + 0.13 = 0.57 (beats Quantizr's 0.60)**

### What exists:
- `src/tac/dp_sims_renderer.py` — SPADE renderer with all channel configs
- `src/tac/fridrich.py` — constrained optimization with augmented Lagrangian
- `src/tac/constrained_gen.py` — coupled_trajectory_optimize
- `src/tac/domain_solvers.py` — EgoMotionFlowSolver
- `src/tac/self_compress.py` — learnable bit-depth per channel
- `src/tac/depth_motion.py` — DepthAwareMotionPredictor with learnable params

### What needs to be built:
A TRAINING SCRIPT that wires all of these together:
- DP-SIMS renderer generates frame pairs
- PoseNet evaluates pairs (coupled trajectory loss)
- SegNet evaluates individual frames
- Fridrich Lagrangian enforces hard constraints on both
- Ego-motion flow provides geometric prior
- Self-compression penalty on model size
- Train for 10K+ epochs on Lightning T4

### Auth results for context:
- Old checkpoint: auth 1.97 (PoseNet 0.057, SegNet 0.006)
- Lightning ep851 checkpoint: auth 1.93 (PoseNet 0.068, SegNet 0.005)
- Local PyAV ≈ DALI auth (difference negligible for CPU lane)
