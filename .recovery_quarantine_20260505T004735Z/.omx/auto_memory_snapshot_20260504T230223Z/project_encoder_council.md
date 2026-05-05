---
name: Encoder Council (Carmack/Bellard/Ohm/Collet)
description: Concrete encoder optimizations ranked by ROI. keyint=-1 applied. CRF 36 queued for retrain.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Applied (zero retrain)
- keyint=-1: ~9KB savings, zero quality change. DONE.
- -map_metadata -1: ~5KB savings. TODO (need to find the right ffmpeg line in compress.sh).

## Queued (needs retrain)
- CRF 36: ~57KB savings (+0.038 score). Must retrain post-filter on new artifacts.
  Risk: PoseNet may regress if CRF 36 artifacts are out of distribution.
- tune=3 (SSIM-optimized): saves 2-5% on dashcam. Bellard recommendation. Untested.

## Rejected
- SVT-AV1 v2.3.0 pinning: build dependency risk > 1-2% RD gain
- Resolution change (512x384 or 582x436): breaks post-filter training distribution
- Unsharp before post-filter: redundant, CNN already learns edge restoration
- h=48 checkpoint shrink: -0.043 SegNet vs +0.010 rate. Net negative unless trained 1.5x longer.

## Multi-pass inflate
- 14 min on M5 Max (under CPU pressure), est. 4-6 min on clean contest runner
- Total with scorer: 26-31 min. TOO RISKY for 30-min contest limit.
- Useful for local score measurement only, not deployable.

## Timing hardening needed
- Add wall-clock timing to inflate.sh (log total inflate time)
- Add a 25-min timeout guard in evaluate.sh wrapper
- Test on a clean 4-core machine (not under training pressure)
