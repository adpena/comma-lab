---
name: Research bundle 2026-04-29 — Self-Compressing NN + C3 + dynamic-quantization water-filling
description: Three research threads added by user 2026-04-29 PM. (1) Self-Compressing NN (arXiv:2301.13142) jointly learns width+precision during training. (2) C3 single-image MLP overfit with quantization (arXiv:2312.02753) for residual codec. (3) Dynamic quantization with water-filling/Lagrangian bit-budget allocation (Lane Ω-Hessian).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
User mandate 2026-04-29 PM: pursue extreme stacking + multi-pass pre/during/post pipeline. Quantizr + Selfcomp left these doors open. Three research-thread integration targets:

## 1. Self-Compressing Neural Networks (arXiv:2301.13142)

**Headline contribution**: "removing redundant weights AND reducing the number of bits required" — simultaneously, joint width × precision optimization during training. Reported: "floating point accuracy with as few as 3% of the bits and 18% of the weights remaining."

**Why this matters for us**: replaces our train→QAT→export pipeline. Currently we train float (high precision), then post-hoc QAT to FP4. Self-Compressing learns per-layer bit-depth WHILE training, with a generalized size-minimization loss. Our 287K-param Lane G v3 might compress to ~40KB at learned mixed precision (vs uniform FP4 ~70KB).

**Integration target**: `experiments/train_distill.py` and/or `experiments/train_segmap.py` get a `--self-compress` flag. Add a learnable `bit_budget` parameter per layer; loss = task_loss + λ × total_bits. λ scheduled so the network converges to a Pareto-optimal (size, accuracy) point.

**Pre/during/post**: this is a DURING-COMPRESS technique.

## 2. C3 — High-Performance Low-Complexity Neural Compression (arXiv:2312.02753)

**Headline contribution**: per-image/video overfit small model (coordinate MLP). Decoding complexity 1 order of magnitude lower than neural baselines at same R-D. Image: matches VTM at <3k MACs/pixel. Video: matches Video Compression Transformer at <5k MACs/pixel. Builds on COOL-CHIC.

**Why this matters for us**: C3 is ORTHOGONAL to our renderer. Our renderer generates RGB from masks. C3 generates RGB from coordinates (NeRF-style). For our contest, the decoder IS the archive — so C3's MLP weights are the compressed representation.

**Integration target as RESIDUAL codec**: 
- Stage 1 (pre): renderer produces base frames at 384×512 from masks.mkv
- Stage 2 (residual): C3 MLP encodes the residual `delta = GT - renderer_output` per-pixel as compressed coordinates → tiny MLP weights.
- Stage 3 (decode): inflate runs renderer + C3, sums them.

Could attack the SegNet/PoseNet residual error directly.

**Pre/during/post**: Compress-time = train C3 MLP per-video. Inflate-time = MLP forward. Compatible with strict-scorer-rule (no scorer load at inflate).

**Tradeoff**: adds inflate time. C3's <5k MACs/pixel × 1.18 Gpixels = ~6 GOPS = under 10s on T4. Still within 30-min inflate budget.

## 3. Dynamic Quantization with Water-Filling/Lagrangian Bit-Budget

**Math**: given total bit budget B and per-channel quantization error ε_c(b_c), minimize Σ_c ε_c(b_c) subject to Σ_c b_c ≤ B. Lagrangian:
  L(b, λ) = Σ_c ε_c(b_c) + λ(Σ_c b_c - B)

Setting ∂L/∂b_c = 0 gives the water-filling solution: b_c = max(0, log(λ × Hessian_c) - log(σ_c²)/2). Allocate more bits to channels with higher Hessian curvature × signal-variance product. This is THE optimal solution for L2-quadratic quantization error.

**Why this matters for us**: Lane Ω-Hessian's premise (memory `project_lane_omega_bit_budget_hessian_aware_quantization`) but the implementation falls back to uniform allocation. Council kill-listed Lane SO for this self-deception.

**Integration target**: extend `src/tac/block_fp_codec.encode_conv_weight` per-channel-qint-max to use water-filling instead of fixed terciles. Use `src/tac/learnable_bit_quant` curvature pattern: compute Hessian via 1-step gradient approximation on training pairs; assign bit budget by water-fill.

**Pre/during/post**: POST-COMPRESS export-time technique. Run on best SC++ checkpoint; produce alternate archives at multiple bit budgets; pick best by auth eval.

## How to apply (extreme stacking pipeline)

The 3-pass compress pipeline becomes:

**Pass 1 — Pre-compress (data preparation)**:
- Lane PA: pose-as-affine init from frozen PoseNet (Lane PA in flight)
- Lane MAE-V: mask augmentation pretraining
- Lane SAUG: Cosmos self-augmentation (when applicable)
- Lane AL: optimize grayscale pixel values directly via SGD before encoding

**Pass 2 — During-compress (training)**:
- SC++: SegMap + KL distill T=2.0 + eval_roundtrip + EMA (running)
- **NEW**: Self-Compressing NN flag — learn per-layer bit-depth during SC++ training (replaces post-hoc QAT)
- Lane WC-S Curator outlier weighting (Lane WC-S coded)
- Lane FC FiLM-Canvas (Lane FC coded)
- Optional: train per-channel learnable quantization scales (LSQ)

**Pass 3 — Post-compress (encoder-side optimization)**:
- Lane FR-Ω Fridrich-cost block-FP exponents on best checkpoint
- **NEW**: Water-filling bit-budget allocation (Lane Ω-W water-filling)
- Lane SH Shannon arithmetic coder on qints (replaces tar.xz)
- Lane TR temporal residual coding (better AV1 inter-prediction settings)
- Lane PD pose deltas (cumsum encoding)
- **NEW**: C3 residual MLP encoding the renderer's per-pixel residual against GT
- Archive diet: zstd tune, manual tensor pack, split entropy streams (codex top-3)

This 3-pass stack potentially compounds. If each step saves 0.02-0.05 score, stacking 5-7 of them lands sub-0.30 territory. 

Compress-time compute budget: UNLIMITED per contest rules. Inflate budget: 30 min single GPU. Multi-pass compress is exactly the right shape for this asymmetry.

## Cross-refs
- project_codex_theoretical_floor_brutal_20260429 (codex floor verdict; cites archive-diet + Quantizr-family sweep)
- feedback_preserve_signal_extreme_stacking_strategy_20260429 (don't kill EUREKA lanes)
- project_council_stacking_lanes_to_03_20260429 (initial stacking design)
- project_lane_omega_bit_budget_hessian_aware_quantization (water-filling design)
- project_new_tools_inventory_20260429 (existing modules: arithmetic_qint_codec, pose_delta_codec, segmap_film_canvas_renderer)
