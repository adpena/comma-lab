# Findings

## 2026-04-10 promoted floor

- Track B promoted floor is **1.33**.
- Variant: `dilated_h64`.
- Platform: `modal_a10g`.
- PoseNet `0.00218374`, SegNet `0.00609921`, rate `0.02301653`.
- Canonical score/report mirrors are generated from `.omx/state/promoted_result.json`.

## 2026-04-10 CRF 35 auth eval — distribution shift confirmed

- Auth eval of CRF 34-trained postfilter on CRF 35 video: **2.08**
- PoseNet regressed 7.3x (0.00218 -> 0.0898), SegNet unaffected
- Confirms postfilter is CRF-specific — corrections do NOT transfer across CRF values
- Implication: each CRF needs its own retrained filter

## 2026-04-10 PSD h=64 auth eval

- PSD (PixelShuffle-Downscale) h=64, epoch 809: auth score **1.49**
- SegNet: 0.00532 (8.3% better than unfiltered baseline — first architecture to improve SegNet)
- PoseNet: 0.01108 (still converging, 5x worse than dilated)
- Proxy-auth transfer is clean (<0.01 gap on distortion components)
- Demonstrates both metrics CAN improve simultaneously; architecture navigates balanced Pareto region

## 2026-04-10 mask2mask competitive intelligence (PR#53)

- Submitter: Quantizr, score **0.60**
- Paradigm: segment original frames -> compress semantic masks -> neural render at inflate time
- Architecture reverse-engineered:
  - TinyFrame2Renderer: U-Net with channels 36->60->36, skip connections
  - TinyMotionFromMasks: optical flow estimation from consecutive mask pairs, flow warping for temporal coherence
  - FP4 quantization with 8-value codebook (not standard int4/int8)
  - Total archive: 386KB
- Validates task-aware compression thesis from synthesis angle
- Near-perfect SegNet (masks are ground truth), competitive PoseNet via flow warping

## 2026-04-10 theoretical score floor under mask rendering

- With mask-conditioned neural rendering, theoretical score floor is **sub-0.10**
- Masks compress to near-zero rate, renderer can synthesize pixel-perfect scorer inputs
- GPU required at inflate time (CUDA/MPS)
- CPU postfilter theoretical floor remains ~1.10

## 2026-04-10 MPS manual grid_sample speedup

- Custom MPS implementation of grid_sample: **11.3x speedup** over CPU fallback
- Critical for GPU lane — PyTorch's grid_sample has no MPS backend, falls to CPU
- P0-P4 MPS optimizations identified for renderer training loop

## 2026-04-10 CLADE is obsolete, DP-SIMS is current SOTA

- CLADE (2021) was initially considered for semantic synthesis
- DP-SIMS (CVPR 2024) is the actual 2024 SOTA for semantic image synthesis
- More relevant architecture reference for mask renderer design

## 2026-04-10 GT scorer cache insight

- Caching ground-truth scorer outputs (PoseNet/SegNet on original frames) can save 40-50% training time
- GT frames don't change across epochs — no need to recompute scorer outputs each time
- Applicable to both CPU and GPU lane training
