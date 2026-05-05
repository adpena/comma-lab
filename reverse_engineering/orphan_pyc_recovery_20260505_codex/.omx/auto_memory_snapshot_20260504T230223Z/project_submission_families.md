---
name: Two Submission Families — CPU + GPU, 3 variants each
description: Strategy for multiple distinct submissions plus comprehensive writeup
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Submission Family 1: CPU Lane (postfilter, CPU-only inflation)

### Variant A: Extreme PoseNet (current best — 1.33 auth)
- Dilated h=64, standard loss, 905 epochs on CRF 34
- pose=0.00218, seg=0.00610
- 46KB int8, CPU inference <30s

### Variant B: CRF-optimized (projected ~1.27-1.30)
- Dilated h=64 retrained on CRF 35 or 36
- Same architecture, smaller video = better rate
- Training in progress (epoch ~80)

### Variant C: Balanced (PSD, projected ~1.3-1.4 if PoseNet converges)
- PSD h=64, improves BOTH SegNet and PoseNet
- First architecture to beat baseline SegNet
- 95KB model (rate penalty)

## Submission Family 2: GPU Lane (mask-conditioned renderer, GPU inflation)

### Variant A: Standard renderer (projected 0.3-0.6)
- MaskRenderer U-Net 40→64→40, 307K params
- Masks compressed as AV1 (~33KB)
- FP4 quantization (~200KB model)

### Variant B: CLADE-enhanced (projected 0.15-0.30)
- Per-class normalization at every layer
- Scorer-aware loss (SegNet on last frame only)
- Larger capacity (48→80→48, ~489K params)

### Variant C: Teacher-distilled (stretch, projected sub-0.10)
- Train 1-2M param teacher with SPADE normalization
- Distill to compact student
- Lossless mask encoding

## Writeup Strategy
- Document BOTH paradigms: postfilter vs conditional rendering
- Show Pareto frontier across ALL variants (CPU and GPU)
- Analyze CPU vs GPU hardware constraints as fundamentally different
  evaluation regimes (same score, different deployment implications)
- Include mask2mask reverse engineering as competitive validation
- Scoring formula critique applies to both paradigms
- Deployment analysis: CPU lane for on-device, GPU lane for cloud

**Why:** Multiple distinct submissions maximize prize chances.
**How to apply:** Each variant is independently trainable and submittable.
