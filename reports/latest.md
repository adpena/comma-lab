# latest report

## current state - 2026-04-10

Track B's promoted honest floor is now **`1.33`** via `dilated_h64`.

## authoritative promoted floor

- Track: `robust_current`
- Variant: `dilated_h64`
- Platform: `modal_a10g`
- Current-workflow score: **`1.33`** at `864,167` bytes
- Distortions: PoseNet `0.00218374`, SegNet `0.00609921`
- Rate: `0.02301653`
- Evidence: `reports/raw/2026-04-10-dilated-h64-authoritative/robust_current-dilated-h64-authoritative-cpu-report.txt`

## additional auth evals today

| Variant | Auth Score | SegNet | PoseNet | Notes |
|---------|-----------|--------|---------|-------|
| PSD h=64 ep809 | 1.49 | 0.00532 | 0.01108 | First arch to improve both metrics vs baseline |
| CRF 35 (old filter) | 2.08 | 0.00581 | 0.08980 | Distribution shift — filter is CRF-specific |

## active experiments

| Experiment | Status | Current Score | Lane |
|-----------|--------|---------------|------|
| CRF 35 retrain | epoch ~110 | 1.54 | CPU |
| CRF 36 retrain | epoch ~110 | 1.95 | CPU |
| Mask renderer | epoch 0 | ~90 | GPU |

## competitive intel

- **PR#53 (mask2mask)** by Quantizr: score **0.60**
  - Paradigm: segment -> compress masks -> neural render
  - Architecture: TinyFrame2Renderer U-Net (36->60->36), FP4 quantization, 386KB
  - Not yet verified by organizers
  - If verified, displaces us from #1
  - Validates GPU-lane mask renderer strategy

## two-lane strategy

- **CPU lane**: postfilter retrains at CRF 35/36, targeting sub-1.33
- **GPU lane**: mask-conditioned renderer, targeting sub-0.50
- Priority: MPS optimizations -> renderer training -> Modal A10G deployment

## key findings

- Postfilter is CRF-specific: CRF 34 filter on CRF 35 video scores 2.08 (worse than no filter)
- PSD architecture improves both SegNet (8.3%) and PoseNet (9.8%) vs baseline simultaneously
- MPS manual grid_sample delivers 11.3x speedup over CPU fallback
- DP-SIMS (CVPR 2024) supersedes CLADE (2021) as SOTA for semantic synthesis
- GT scorer cache can save 40-50% training time
