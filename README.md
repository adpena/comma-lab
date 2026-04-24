# pact

Task-aware video compression for the [comma.ai video compression challenge](https://github.com/commaai/comma_video_compression_challenge). A 287K-parameter conditional generative model compresses a 60-second driving video so that two frozen neural networks (SegNet, PoseNet) produce nearly identical outputs on the reconstruction.

Scoring formula: `S = 100 * seg_distortion + sqrt(10 * pose_distortion) + 25 * rate`

Current best: **0.407 proxy** (training complete, auth eval pending). Leaderboard leader is 0.33.

## Architecture

The renderer is an asymmetric warp pair generator built on CLADE-conditioned U-Net blocks ([arxiv 2012.04644](https://arxiv.org/abs/2012.04644)):

```
frame2 = renderer(mask2)                        # Direct render from segmentation mask
flow, gate, residual = motion(mask1, mask2)     # Motion prediction from both masks
frame1 = warp(frame2, flow) + gate * residual   # Geometric warp + gated correction
```

Frame2 is rendered directly. Frame1 is derived by warping frame2 with learned optical flow and a gated residual, making temporal coherence architectural rather than learned through loss alone. PoseNet sees geometric ego-motion between frames, which is what real driving video produces.

Key components:

- **CLADENorm**: GroupNorm with per-class affine modulation from the 5-class segmentation mask. Spatially-varying conditioning at ~10 parameters per layer.
- **Radial zoom warp** (`src/tac/radial_zoom.py`): The PoseNet Jacobian has effective rank 1.008 (verified empirically). Only one degree of freedom matters: a scalar radial zoom from the Focus of Expansion. Replaces a 50K-param motion predictor with 600 learned scalars (1.2 KB at FP16).
- **Int4+LZMA2 export** (`src/tac/mixed_precision_export.py`): Per-tensor symmetric int4 quantization followed by LZMA2 compression. Achieves ~2.2 bits/weight vs 4.4 bits/weight for FP4 codebook approaches.
- **Fridrich inverse steganalysis losses** (`src/tac/fridrich.py`, `src/tac/fridrich_losses.py`): The challenge scorers are forensic detectors. Training losses derived from Fridrich's steganalysis framework push errors into the scorers' null space: UNIWARD texture masking, L-infinity spreading (square root law), and Markov transition statistics.
- **Forensic analysis tools** (`src/tac/forensics.py`): Boundary artifact detection, SegNet class-boundary error analysis, PoseNet per-pixel Jacobian sensitivity maps, and eval roundtrip distortion maps.

## Quick start

```bash
# Install
git clone https://github.com/adpena/pact.git && cd pact
uv venv && uv pip install -e ".[dev]"

# Compress a video
PYTHONPATH=src:upstream python experiments/pipeline.py compress \
    --video upstream/videos/0.mkv \
    --checkpoint path/to/checkpoint.pt \
    --device cuda --output-dir results/run_01

# Evaluate an archive
PYTHONPATH=src:upstream python experiments/pipeline.py eval \
    --archive results/run_01/archive.zip \
    --video upstream/videos/0.mkv --device cuda
```

## Training profiles

Three experiment profiles encode different training philosophies. All share the same architecture for fair comparison.

| Profile | Strategy | Key idea |
|---------|----------|----------|
| **WILDE** | Empirical 5-phase schedule | Freeze/unfreeze phases with hard-mined error boosting (9x/49x). Quantizr-adapted anchor training. |
| **SHIRAZ** | Principled adaptive training | PCGrad gradient surgery + focal STE loss. No freeze/unfreeze. Score-contribution-proportional weighting. |
| **GREEN** | WILDE + radial zoom warp | Same as WILDE but MotionPredictor outputs only gate+residual (4ch). Flow from RadialZoomWarp. 14K fewer params. |

```bash
# Train with a named profile
PYTHONPATH=src:upstream python experiments/train_tac.py --profile wilde --device cuda
```

Profiles are defined in `src/tac/profiles.py` with full provenance for every hyperparameter choice.

## Project structure

```
src/tac/                    Core library: renderer, losses, quantization, forensics
experiments/                Training scripts, pipeline, analysis tools
experiments/pipeline.py     Canonical compress + eval pipeline
docs/paper/                 Technical paper (in progress)
submissions/                Submission packaging
upstream/                   Pinned upstream challenge snapshot (read-only)
```

## Results timeline

| Stage | Score | Method |
|-------|-------|--------|
| H.265 re-encode | 1.97 | CRF 28, no postfilter |
| CPU postfilter | 1.33 | Dilated h=64 CNN trained against scorers |
| Asymmetric warp renderer | 0.87 | Lagrangian annealing, FP4 quantized |
| + TTO (blind PoseNet) | 0.70 | 500-step test-time optimization, SegNet spillover only |
| + TTO (gradient fix) | 0.43 | Differentiable BT.601 YUV — fixed upstream `@torch.no_grad` bug |
| + WILDE/SHIRAZ training | 0.407 | Proxy score, auth eval pending |

The single largest improvement (0.70 to 0.43, 38.6% reduction) came from discovering that the upstream scorer's RGB-to-YUV conversion was decorated with `@torch.no_grad`, silently zeroing all PoseNet gradients during test-time optimization. Every prior TTO experiment was optimizing PoseNet blind. See `docs/paper/03_gradient_bug.md`.

## Methodology

Design decisions are made by a 15-member skunkworks council with domain expertise in steganalysis, neural compression, and adversarial ML. The challenge creator (Yousfi) was Fridrich's PhD student at Binghamton; the SegNet scorer is a steganalysis detector. We frame the problem as inverse steganalysis and apply Fridrich's framework directly.

All training code passes a recursive adversarial review protocol: 3 consecutive clean passes from 5 independent reviewers before deployment. The review gate is enforced by pre-commit hooks.

Research state is maintained in durable files (`.omx/state/`, `.ralph/run_log.md`, `reports/`) so work can resume across sessions without relying on chat context.

## Paper

The technical paper is in `docs/paper/`. It covers the asymmetric warp architecture, the gradient obstruction bug discovery, Fridrich-informed loss design, and the rank-1 radial zoom warp derivation.

## Requirements

- Python 3.11+
- PyTorch 2.0+
- ffmpeg (video decode)
- CUDA GPU recommended (MPS and CPU supported for development)

## License

MIT
