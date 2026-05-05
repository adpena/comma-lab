---
name: MLX Full Strategy — Both Lanes, All Workloads
description: MLX 4.7x faster for training. Use for Phase 1 in GPU lane AND scorer-free components in CPU lane.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Where MLX wins (4.7x faster for fwd+bwd)
- ANY training loop that doesn't need frozen PyTorch scorer models
- Phase 1 pre-training (L1 loss) for renderer AND postfilter
- FP4/INT4 quantization experiments
- Architecture search / hyperparameter sweeps
- Data augmentation with gradient (differentiable augmentation)

## Where PyTorch MPS stays (scorer forward needed)
- Phase 2 scorer fine-tuning (frozen PoseNet/SegNet are PyTorch)
- Auth evaluation (upstream evaluate.py is PyTorch)
- Int8/FP4 checkpoint evaluation

## CPU Lane MLX opportunities
1. Pre-train postfilter with L1 reconstruction loss (Phase 1)
2. Architecture search: sweep h=16/32/48/64/96 in MLX (4.7x faster each)
3. SWA / weight averaging experiments
4. Saliency map computation (if ported to MLX)

## GPU Lane MLX opportunities
1. Phase 1 renderer pre-training (L1 + edge loss) — THE BIG WIN
2. Wavelet renderer training (all Haar ops are basic arithmetic)
3. VQ-VAE codebook learning
4. Diffusion teacher pre-training (no scorer in early epochs)

## MLX ↔ PyTorch weight bridge
- Conv2d: (O,I,H,W) PyTorch ↔ (O,H,W,I) MLX (transpose dims 1,2,3→2,3,1)
- Linear: same layout, no transpose needed
- BatchNorm/GroupNorm: same layout
- Embedding: same layout

## The Hybrid Pipeline
```
Phase 1 (MLX, 4.7x faster):
  GT frames → masks → MLX renderer → L1 loss → MLX optimizer
  Duration: ~40 min for 100 epochs (vs 3 hours in PyTorch)

Weight Bridge:
  MLX params → transpose conv weights → PyTorch state_dict → .pt file

Phase 2 (PyTorch MPS, scorer needed):
  masks → PyTorch renderer → scorer loss → PyTorch optimizer
  Loads Phase 1 weights from bridge
```

**Why:** 4.7x speedup on M5 Max for ALL training that doesn't need PyTorch scorers.
**How to apply:** Port renderer + postfilter to MLX. Use hybrid pipeline for all training.
