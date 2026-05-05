---
name: 48-Hour Battle Plan — Council Binding (2026-04-12)
description: Three parallel tracks. CRF sweep is highest ROI. Renderer sweep after. h=96 auth eval quick win.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Track 1 (CPU, highest ROI): CRF sweep on postfilter
- CRF 36/38 → estimated 1.33 → 1.05-1.17
- MUST retrain postfilter on new CRF artifacts (M5 Max, ~2hr each)
- Script ready: experiments/crf_sweep_score.sh
- Rate is 43% of current score — CRF is the biggest lever for postfilter path

## Track 2 (GPU, concurrent): Renderer channel sweep
- (36,60) d=1 primary (currently training on Modal)
- (24,40) d=1 moonshot (68KB = tiny rate, breakeven if pose < 0.007)
- Skip (48,80) and (64,128) — rate penalty kills them
- Config: experiments/configs/h_sweep.json
- Runner: experiments/run_h_sweep.py

## Track 3 ($0.30): Auth eval h=96 postfilter
- reports/raw/modal_artifacts/postfilter_h96_standard_ep900_best_int8.pt
- Never scored despite $32 training cost
- Tells us if postfilter capacity has plateaued

## Decision gate at T+24h
- If CRF delivers 1.05-1.17: postfilter path is insurance floor
- If CRF disappoints: double down on renderer (24,40) for minimal rate
