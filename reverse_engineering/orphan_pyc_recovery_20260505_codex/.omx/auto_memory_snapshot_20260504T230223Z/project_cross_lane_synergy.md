---
name: Cross-Lane Synergy — Every Technique Benefits Both CPU and GPU
description: The tac library is becoming a cross-platform toolkit where improvements compound across both lanes
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Synergy

Every technique we build for one lane transfers to the other:

| Technique | CPU Lane Benefit | GPU Lane Benefit |
|-----------|-----------------|-----------------|
| GT scorer cache | fit_lazy 40-50% speedup | train_renderer 40-50% speedup |
| Entropy coder | Could encode saliency maps | Smaller mask archive = better rate |
| TTO | Adapt postfilter at inflate | Adapt renderer at inflate |
| Wavelet domain | Wavelet postfilter possible | Wavelet renderer (direct) |
| channels_last + FP16 | All MPS training faster | All MPS training faster |
| Signal handling | Already in Trainer | Now in train_renderer via utils.py |
| AV1 loop filter disable | Better analysis of mask quality | Better masks = better renderer |
| CLADE normalization | Could apply to PSD postfilter | Already in renderer |
| FP4 quantization | Halves CPU postfilter | Already in renderer |
| Scorer-aware loss | SegNet on last frame only | Already implemented |

## The tac Library Evolution

tac started as a postfilter training loop. It's now:
- Cross-paradigm: postfilter (CPU) + renderer (GPU) + wavelet (GPU)
- Cross-platform: CUDA + MPS + CPU with auto-detection
- Cross-quantization: INT8, FP4, INT4 with per-channel/codebook
- Shared infrastructure: EMA, QAT, telemetry, signal handling, scorer cache

## Paper Framing
This is a publishable contribution independent of the competition results:
"A unified toolkit for task-aware video compression spanning CPU postfilters
to GPU neural renderers, with shared training infrastructure."

**Why:** Understanding cross-lane synergy helps prioritize development.
**How to apply:** When building for one lane, always consider the other.
