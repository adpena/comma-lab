---
name: Eureka Self-Compression Session
description: 6 eureka moments from council session on self-compression findings. Quantization IS steganalysis. Mixed-precision + LZMA2 → 25-35KB. CLADE + quantization = near-zero SegNet.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Council Eureka Session (2026-04-25 overnight)

1. **Quantization IS inverse steganalysis** (Yousfi) — 2-bit quantization noise in bottleneck layers acts as steganographic cover that SegNet's stride-2 stem can't detect. Not a bug, a FEATURE. Inject during training, not just export.

2. **Per-layer bit-depth = learned embedding strategy** (Fridrich) — scorer sensitivity IS the distortion cost map. Optimal bit allocation IS the rate-distortion solution. Novel connection between neural compression and steganography. PUBLISHABLE.

3. **Sensitivity sweep on A100 with full dataset** (Hotz) — 5 pairs on MPS isn't enough. Run with 600 pairs on A100 after training. Add to auto-pipeline. ~$0.50 marginal cost.

4. **Mixed-precision + LZMA2 → 25-35KB** (Hotz) — 2-bit layers have fewer unique values → better entropy coding. Projected renderer: 25-35KB vs 170KB FP4. Rate savings: 0.09 points.

5. **181K × 2.2 bits ≈ 88K × 4 bits** (Quantizr) — param count doesn't matter, total bits do. Our larger architecture with mixed precision matches Quantizr's archive size with 2x capacity.

6. **CLADE + engineered quantization → near-zero SegNet** (Contrarian) — if quantization noise can be DIRECTED to flip SegNet predictions favorably (not randomly), SegNet distortion approaches zero. Combine with per-class CLADE normalization.

**Why:** These insights compound. The self-compression breakthrough isn't just about rate — it's about using quantization as a QUALITY tool. The arXiv paper should frame this as "Inverse Steganalysis via Learned Quantization."

**How to apply:** 
1. Add sensitivity sweep to auto-pipeline after training
2. Implement knapsack allocator targeting 80KB archive
3. Combine with Int4+LZMA2 for further compression
4. Engineer beneficial quantization noise during training (Phase 2 loss term)
5. Test CLADE with per-class bit allocation
