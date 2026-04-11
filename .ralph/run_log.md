# run log

## 2026-04-10T23:59:00-05:00 - session state dump (massive session)

### PSD h=64 auth eval
- PSD (PixelShuffle-Downscale) h=64, epoch 809: auth score **1.49**
- SegNet 0.00532 (8.3% better than baseline), PoseNet 0.01108 (still converging)
- First architecture to improve BOTH metrics vs unfiltered baseline
- Proxy-auth transfer clean (<0.01 gap)
- Not promoted (1.49 > 1.33 floor)

### CRF 35 auth eval — distribution shift
- Auth eval of CRF 34-trained filter on CRF 35 video: **2.08**
- PoseNet regressed 7.3x (0.00218 -> 0.0898), SegNet unaffected
- Confirms postfilter corrections are CRF-specific, do NOT generalize
- CRF 35 and CRF 36 retrains started to address this

### mask2mask competitive intel (PR#53)
- Submitter: Quantizr, score **0.60**
- Paradigm: segment -> compress masks -> neural render
- Reverse-engineered: TinyFrame2Renderer U-Net (36->60->36), TinyMotionFromMasks (flow warping), FP4 8-value codebook, 386KB archive
- Not verified by organizers yet
- Threat: displaces us from #1 if verified

### GPU lane architecture built
- Mask-conditioned renderer pipeline implemented
- Takes compressed semantic masks, synthesizes scorer-compatible frames
- MPS support scaffolded (Apple Silicon M5 Max)

### Renderer first epoch
- Epoch 0 score ~90 (expected for random init)
- Training loop functional but slow (MPS fallbacks to CPU for grid_sample)

### MPS optimizations identified (P0-P4)
- P0: manual grid_sample — 11.3x speedup demonstrated
- P1: MPS batch normalization
- P2: MPS upsampling
- P3: training loop profiling
- P4: GT scorer cache (40-50% time savings)

### CRF retrains launched
- CRF 35 retrain: running, epoch ~110, score 1.54
- CRF 36 retrain: running, epoch ~110, score 1.95
- Both need 300+ epochs for meaningful eval

### Research notes
- CLADE (2021) is obsolete — DP-SIMS (CVPR 2024) is current SOTA for semantic synthesis
- Theoretical score floor under mask rendering: sub-0.10
- CPU postfilter theoretical floor: ~1.10

## 2026-04-10T21:30:00-05:00 - promoted floor synchronized

- authoritative promoted floor: **1.33**
- variant: `dilated_h64`
- platform: `modal_a10g`
- evidence: `reports/raw/2026-04-10-dilated-h64-authoritative/robust_current-dilated-h64-authoritative-cpu-report.txt`
- mirrors are now expected to be derived from canonical promoted_result.json
