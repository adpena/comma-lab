---
name: Contrarian+Karpathy+LeCun Deep Review (2026-04-10)
description: Comprehensive panel review — FakeQuant mismatch fix, T<1.0 argmax regime, SWA/ensemble, error replay
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Consensus (all three agree):
1. FakeQuant per-tensor vs eval per-channel was a train/eval mismatch — FIXED
2. Temperature floor 1.0 was too conservative — relaxed to 0.1 for argmax pressure — FIXED
3. Error replay should be turned on (error_replay_every=100) — IMPLEMENTED, not yet deployed
4. SWA class exists but unused — wider minima are more int8-robust — TODO

## Karpathy:
- EMA 0.997 too slow for fast runs (<100 ep). Use 0.99.
- eta_min=1e-6 kills LR before final-phase eval. Raised to 1e-5. — FIXED
- Error replay at every 50 epochs is the cheapest high-EV experiment.

## LeCun:
- PSD architecture has better RF alignment with scorer resolutions than dilated.
- alpha_seg=5000 may be 3x too aggressive (scoring leverage is only ~3:1 at current op point, not 10:1).
- YUV-space corrections would concentrate capacity on scorer-relevant dimensions.

## Contrarian:
- Checkpoint ensemble at inference: pixel-median of multiple checkpoints. Free diversity.
- Focal gamma=2 too mild for 95% easy pixels. Need gamma=4-5.
- The optimal post-filter is a *targeted adversarial perturbation generator*, not an image enhancer.
- Focus 100% of loss on pixels where filtered_argmax != GT_argmax, not global distribution matching.

## Unimplemented high-value ideas (priority order):
1. Error replay every 100 epochs (implemented, deploy with --error-replay-every 100)
2. Checkpoint ensemble at inference (pixel-median of top N checkpoints)
3. T=5→1→0.2 three-phase schedule (now possible with relaxed floor)
4. SWA over final 20% of training
5. Focal gamma=4-5 for aggressive hard-pixel focus
6. PSD architecture for better scorer-aligned RF
