# comma-lab / pact

Task-Aware Compression research and artifact tooling for the
[comma.ai video compression challenge](https://github.com/commaai/comma_video_compression_challenge).
This repository is maintained as a community and historical-record workspace:
public-archive intake, exact replay custody, writeup drafts, and OSS tooling.
It is not a live leaderboard page and it does not make an arXiv or preprint
commitment.

In this repository, `tac` means **Task-Aware Compression**: compression
optimized for a downstream task/scorer. A **codec** is a narrower concrete
encoder/decoder or wire format inside that stack. The `comma_lab` package is the
lab and operations layer around `tac`, not a second compression engine.
The package-level boundary docs are [src/tac/README.md](src/tac/README.md) and
[src/comma_lab/README.md](src/comma_lab/README.md). The canonical terminology
and contest-compliance boundary is
[docs/terminology_and_boundaries.md](docs/terminology_and_boundaries.md).

| Name | Canonical role |
|---|---|
| `comma-lab` | Public repository and lab workspace for the challenge research system |
| `tac` | Python package: Task-Aware Compression library and algorithmic engine |
| `comma_lab` | Python package: lab operations, custody, state projection, and reporting |
| `pact` | Internal workspace alias retained in historical docs and local paths |

Score-bearing claims must be read through the repository evidence grades. The
ranked public rows live in `docs/paper/04_results.md`; roadmap and planning
rows are not score claims until exact CUDA auth eval lands on exact archive
bytes.

Scoring formula: `S = 100 * seg_distortion + sqrt(10 * pose_distortion) + 25 * rate`

## Evidence Grades

| Grade | Public/writeup use | Minimum requirement |
|---|---|---|
| `A++` / `A` | Ranked score row | Exact archive bytes and SHA-256, CUDA auth-eval JSON, component recomputation, runtime custody, full sample count |
| `A-negative` | Scoped negative result | Same custody standard as a score row, but used only for the measured implementation/config |
| `empirical` | Roadmap or engineering signal | Byte, smoke, loss, round-trip, or component evidence without full score custody |
| `derivation` / `prediction` | Roadmap only | Formula or model-based hypothesis awaiting archive evidence |
| `external` | Community/historical context | Public PR text, leaderboard metadata, or outside papers before local exact replay |
| `invalid` | Compliance lesson | Proxy, CPU/MPS, stale, sidecar, exploit, malformed, or otherwise non-ranking evidence |

## CUDA vs CPU auth eval split (2026-05-08)

The contest scorer at `upstream/evaluate.py` produces two distinct authoritative
score axes for the same archive bytes — `--device cuda` and `--device cpu` —
and the public leaderboard ranks by the **CPU** score, not the CUDA score.
Across the medal-band HNeRV cluster (PR100/101/102/103/105) we measured a
remarkably tight `R_pose = pose_cuda / pose_cpu = 5.04 ± 0.10` and
`R_seg = seg_cuda / seg_cpu = 1.17 ± 0.01`, producing a near-constant
score-axis gap of `Δscore = 0.0330 ± 0.0004`. PR #102's third-prize 0.195 was
the CPU score; the CUDA bot comment for the same archive bytes was 0.228.

Operational consequence: every shippable archive now gets dual-eval —
authoritative `[contest-CUDA]` and `[contest-CPU]` axes on Linux x86_64
hardware that is 1:1 contest-compliant with the GitHub Actions CI runner.
Apple Silicon CPU eval is `[macOS-CPU advisory only]`, never `[contest-CPU]`.

Full write-up: [`docs/findings/cuda_cpu_auth_eval_split_20260508.md`](docs/findings/cuda_cpu_auth_eval_split_20260508.md).
Methodology long-form: [`docs/writeup/cuda_cpu_drift_methodology.md`](docs/writeup/cuda_cpu_drift_methodology.md).

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
git clone https://github.com/adpena/comma-lab.git && cd comma-lab
uv venv && uv pip install -e ".[dev]"

# Compress a video
PYTHONPATH=src:upstream python experiments/pipeline.py compress \
    --video upstream/videos/0.mkv \
    --checkpoint path/to/checkpoint.pt \
    --device cuda --output-dir results/run_01

# Evaluate an archive
PYTHONPATH=src:upstream python experiments/pipeline.py eval \
    --archive results/run_01/archive.zip \
    --checkpoint results/run_01/iter_01/renderer.bin \
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
PYTHONPATH=src:upstream python experiments/pipeline.py compress \
    --profile wilde \
    --video upstream/videos/0.mkv \
    --checkpoint path/to/checkpoint.pt \
    --device cuda --output-dir results/wilde
```

Profiles are defined in `src/tac/profiles.py` with full provenance for every hyperparameter choice.

## Project structure

```
src/tac/                    Task-Aware Compression library and reusable algorithms
src/comma_lab/              Lab operations, state projection, preflight adapters
experiments/                Training scripts, pipeline, analysis tools
experiments/pipeline.py     Canonical compress + eval pipeline
docs/paper/                 Technical paper (in progress)
submissions/                Submission packaging
upstream/                   Pinned upstream challenge snapshot (read-only)
```

## Historical Timeline

The early renderer/post-filter numbers in this repository are retained as
historical research context. They should not be copied into a public ranked
table unless the row has an `A++`/`A` evidence tag and a cited
`contest_auth_eval.json`.

| Thread | Public/writeup status |
|---|---|
| H.265 and CNN post-filter baselines | Historical context for the scorer-aware workflow |
| Asymmetric warp renderer and pose TTO | Methodology and negative-result context; only exact CUDA rows may rank |
| Gradient obstruction fix | Measurement-methodology contribution; see `docs/paper/03_gradient_bug.md` |
| Public PR replay/deconstruction | Community/historical-record corpus plus exact replay rows when CUDA custody exists |
| Post-deadline hidden-gem lanes | Roadmap until charged archives pass exact CUDA auth eval |

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
