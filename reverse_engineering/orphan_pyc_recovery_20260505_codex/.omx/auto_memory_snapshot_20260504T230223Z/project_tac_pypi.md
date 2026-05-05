---
name: tac as PyPI Package — Production-Grade Open Source Library
description: Package tac for publication to PyPI as a standalone task-aware compression toolkit
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Vision
tac (Task-Aware Codec) should be a publishable, pip-installable Python library
that anyone can use to build task-aware video compression systems.

`pip install tac`

## What tac already has (13,833 lines, 27 modules)
- 6 GPU renderer architectures (U-Net, wavelet, DP-SIMS, diffusion, VQ-VAE, coord)
- CPU postfilter family (standard, dilated, PSD, gated, depthwise, luma, FiLM, pair-aware)
- FP4 + INT8 + INT4 quantization with QAT
- Custom entropy coder for segmentation masks (140x better than AV1)
- MLX port for Apple Silicon (4.7x training speedup)
- GT scorer cache for 40-50% training speedup
- Manual grid_sample for MPS compatibility (11.3x vs CPU fallback)
- TTO (test-time optimization) with self-supervised and supervised modes
- Stored PoseNet targets for supervised TTO
- EMA, SWA, LSQ, error replay, hard-frame curriculum
- Signal handling, JSONL telemetry, wall-clock timeout
- Zero-pollution auth eval tool
- 9+ test files with 200+ tests

## What's needed for PyPI
1. Clean up `pyproject.toml` (version, description, classifiers, dependencies)
2. `src/tac/__init__.py` with proper public API
3. Documentation (README.md for tac specifically, not just the competition)
4. CLI entry points (`tac train`, `tac eval`, `tac compress`, `tac inflate`)
5. Type hints throughout (pydantic for configs, standard for functions)
6. Remove competition-specific hardcoding (upstream paths, etc.)
7. Make scorer models pluggable (not just PoseNet/SegNet)
8. License (MIT or Apache 2.0)
9. CI/CD (GitHub Actions for tests + publish)
10. Versioning (already at 0.8.0 from Kaggle hardening)

## The positioning
"tac: A toolkit for training neural video compression components that optimize
for downstream task performance rather than perceptual quality."

Use cases beyond comma:
- Medical imaging compression preserving diagnostic CNN features
- Satellite imagery compression preserving detection model features
- Surveillance compression preserving person/vehicle detection
- Robotics perception compression preserving planning model features
- Any domain where compressed media feeds into neural network inference

## Distribution channels
1. **PyPI**: `pip install tac` (Python library)
2. **Homebrew**: `brew install tac` (CLI tool for macOS)
3. **conda-forge**: for scientific computing users
4. **GitHub Releases**: binary wheels + source tarball
5. **Docker**: pre-built images with all dependencies

The Homebrew formula would provide the CLI tools:
```
brew install tac
tac train --profile proven_baseline --tag my_experiment
tac eval --checkpoint model.pt --video input.mkv
tac compress --config config.env --output archive.zip
tac inflate --archive archive.zip --output inflated/
tac monitor --telemetry experiment_telemetry.jsonl
```

**Why:** A published, well-documented library demonstrates production engineering skill.
Multi-channel distribution shows understanding of developer experience across platforms.
**How to apply:** Start packaging incrementally. Clean public API first, docs second.
