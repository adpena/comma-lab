---
name: Full Literature Survey — 30+ Papers for GPU Lane
description: Complete survey of SPADE, neural codecs, FP4, flow, task-aware compression with implementation priorities
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## MUST IMPLEMENT (council + literature consensus)

### 1. CLADE Per-Class Normalization (TPAMI 2021)
- Paper: arxiv.org/abs/2012.04644
- What: Per-class (gamma, beta) at every GroupNorm layer, replacing SPADE
- Why: 60x lighter than SPADE, comparable quality. Injects mask info at every layer.
- Cost: ~50 params per layer. Implementation: 30 minutes.
- Status: NOT YET IMPLEMENTED

### 2. Scorer-Aware Loss Weighting
- What: SegNet loss ONLY on last frame (scorer only evaluates last-frame argmax).
  PoseNet loss on all pairs. Earlier frames optimized purely for PoseNet.
- Why: Directly matches the scorer structure. Free improvement.
- Cost: 15 minutes to implement.
- Status: NOT YET IMPLEMENTED

### 3. INT4/Codebook Quantization (torchao)
- Paper: PyTorch torchao library (github.com/pytorch/ao)
- What: Replace INT8 with codebook-based INT4. Halves model size.
- Why: 300KB→150KB. Rate term is already cheap but this helps.
- Cost: 1 hour. Use CodebookWeightOnlyConfig.
- Status: FP4 implemented (our own), torchao INT4 NOT YET

### 4. FP4 QAT Fix (Critical Bug)
- What: Fix QATRendererFP4 gradient flow (parameter swapping breaks optimizer refs)
- Why: Current implementation detaches from optimizer. Must use STE pattern.
- Status: IN PROGRESS (fix agent working on it)

### 5. Vectorize FakeQuantFP4 (Performance)
- What: Replace Python loop over blocks with batched tensor ops
- Why: Current loop is 9375 iterations per param per step. Too slow for training.
- Status: IN PROGRESS (fix agent)

## SHOULD IMPLEMENT (medium priority)

### 6. Per-Class Flow in MotionPredictor
- What: Predict separate flow fields per semantic class (road=static, cars=moving)
- Why: Different scene elements have different motion. Global flow is suboptimal.
- Cost: 2 hours.

### 7. Knowledge Distillation (Teacher→Student)
- Papers: GAN Compression (arxiv.org/abs/1902.00159)
- What: Train 1-2M param "teacher" renderer, distill to 300K student
- Why: Teacher captures more detail, student preserves it at deployment size
- Cost: 1 day (need teacher training + distillation loop)

### 8. Shared Embedding Between Renderer and MotionPredictor
- What: Currently separate nn.Embedding(5, embed_dim) in each module
- Why: Reduces params, ensures consistent class representation
- Cost: 15 minutes.

### 9. Soft Sigmoid Output (Instead of Hard Clamp)
- What: Replace .clamp(0, 255) with 255*sigmoid(x) for renderer output
- Why: Hard clamp creates dead gradient zones. Sigmoid is smooth everywhere.
- Cost: 5 minutes.

### 10. Coord Grid Caching
- What: Register make_coord_grid as buffer instead of recreating per forward
- Why: Avoids GPU allocation per step
- Cost: 10 minutes.

## COULD IMPLEMENT (speculative / future)

### 11. Sandwiched Pre-Processor
- Paper: arxiv.org/abs/2402.05887
- What: Learnable mask pre-processing before AV1 to optimize representation
- Why: Current masks are raw SegNet argmax. Could be optimized for renderer.

### 12. Instance-Adaptive Weight Deltas
- Paper: arxiv.org/abs/2111.10302
- What: Pre-train general renderer, overfit per-clip with entropy-coded deltas
- Why: Reduces per-clip training time and weight storage.

### 13. SPADE Full (for Teacher Model Only)
- Paper: arxiv.org/abs/1903.07291
- What: Full spatially-adaptive normalization at every layer
- Why: Best quality but 39% parameter overhead. Use in teacher, distill to student.

### 14. VQ-VAE Latent Space (Replace Masks)
- What: Replace discrete masks with learned VQ codebook latents
- Why: More expressive than 5 discrete classes. But masks are already perfect for SegNet.

## KEY PAPERS BY CATEGORY

### Semantic Image Synthesis
| Paper | Year | Key Idea |
|-------|------|----------|
| SPADE (Park et al.) | CVPR 2019 | Spatially-adaptive normalization from masks |
| CLADE | TPAMI 2021 | Class-adaptive norm, 60x lighter than SPADE |
| OASIS | ICLR 2021 | Segmentation-based discriminator |
| pix2pixHD | CVPR 2018 | Multi-scale generation |

### Neural Codecs
| Paper | Year | Key Idea |
|-------|------|----------|
| DCVC-RT | CVPR 2025 | 100+ FPS neural video codec |
| Cool-Chic | 2024 | 800-param overfitted codec |
| HiFiC | NeurIPS 2020 | GAN decoder for compression |
| Sandwiched | 2024 | Neural pre/post around standard codec |
| Instance-Adaptive | ICLR 2023 | Per-content finetuning |

### Quantization
| Paper | Year | Key Idea |
|-------|------|----------|
| NVFP4 | 2025 | Native FP4 on NVIDIA Blackwell |
| torchao | 2024-25 | PyTorch codebook quantization |
| BCQ/LO-BCQ | 2024 | Learned codebooks, <1% loss at W4A4 |

### Task-Aware Compression
| Paper | Year | Key Idea |
|-------|------|----------|
| AccelIR | CVPR 2023 | Optimize for downstream task, not PSNR |
| PICM-Net | 2024 | Semantic ROI-aware compression |
| VCM MPEG | 2023-25 | ISO standard for machine-targeted compression |

### Flow / Motion
| Paper | Year | Key Idea |
|-------|------|----------|
| Forecast Warping | 2022 | Autoregressive flow for future segmentation |
| Segment-aware flow | 2023 | Per-segment flow estimation |

**Why:** This is the complete knowledge base for the GPU lane.
**How to apply:** Implement items 1-5 immediately. Items 6-10 this week. Items 11-14 only if time permits.
