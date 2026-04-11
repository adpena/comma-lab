# Current Focus — 2026-04-10

## Floor
- **Official score**: 1.33 authoritative (compliant ~1.356)
- **Variant**: dilated_h64, standard loss
- **Platform**: modal_a10g
- **Epoch**: 905
- **Archive**: 903KB (877KB video + 46KB compressed checkpoint)

## What works
- Standard loss with dilated h=64 CNN is the ONLY proven technique
- tac v1.0.0, 70 tests, pydantic validation, atomic saves
- QAT + EMA + best-checkpoint int8 selection

## What is dead
- **KL distillation**: two auth evals confirmed PoseNet collapse (1.85 and 2.05)
- **Adaptive weight formula**: T^2 cancels, formula vacuous, 125x mismatch with empirical optimum
- **All preprocessing**: blur, denoise, chroma subsampling all kill PoseNet
- **PoseNet gradient caps**: caused 26x regression

## Current experiments
- PSD (PixelShuffle-Downscale) architecture with standard loss on Kaggle
- 5 adaptive frontier items: boundary dispatch, sin^2 ramp, replay gate, 3-phase eval, plateau LR
- Dilated h=64 long training runs (2500+ epochs)

## Key dates
- Deadline: May 3, 2026 (22 days remaining)
- Current position: #1 by 0.53 margin
