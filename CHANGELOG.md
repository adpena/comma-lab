# Changelog

All notable changes to the `tac` library are documented here.
Dates are approximate milestones from the research sprint.

## [1.0.5] - 2026-04-15

### Added
- FiLM conditioning layer (`FiLMPostFilter`) for pose-aware rendering
- Eval-matched resize roundtrip with Straight-Through Estimator (STE) for training
- Hinge loss for SegNet optimization (empirically better than cross-entropy)
- Two-phase TTO: joint PoseNet+SegNet optimization then SegNet-only refinement
- Checkpoint MD5 verification system in `tac.checkpoint`
- Vast.ai deployment tooling: `src/tac/deploy/vastai/` (client, budget tracker, launcher)
- Pipeline profiler (`tac.profiling.PipelineProfiler`) for stage-level timer analysis
- `constrained_gen.py`: asymmetric warp generation with learnable mask-to-RGB renderer
- `dp_sims_renderer.py`: SPADE-based progressive generator for mask-to-pair synthesis
- `multi_model_inflate.py`: parallel inflate with multiple renderer checkpoints
- `parallel_inflate.py`: concurrent frame rendering for T4-budget inflation
- `scorer_targets.py`: precomputed PoseNet target extraction for supervision
- `lossless/` module: entropy coding utilities for archive post-processing
- `cross_frame_attention.py`: temporal attention for multi-frame rendering
- `depth_motion.py`: ego-motion depth integration for flow prediction
- `entropy_archive.py`: entropy-coded archive builder with rate accounting
- `mask_entropy_coder.py`: mask-specific entropy coding with run-length stats
- `roi_analysis.py` / `roi_preprocessing.py`: region-of-interest saliency weighting
- Named training profiles system (`tac.profiles`): `proven_baseline`, `psd_standard_adaptive`, `smoke`, and more
- Per-channel int8 quantization with LSQ (Learned Step-size Quantization)
- FP4 extreme quantization with custom codebook (`tac.fp4_quantize`)
- `BudgetTracker` for cloud GPU cost enforcement (hard cap, per-hour accounting)

### Changed
- Adaptive weight formula retired — T² cancellation made it vacuous; use static weights
- `Trainer` now supports lazy pair loading and precomputed data pipelines
- Archive codec updated for contest-compliant rate accounting
- SegNet loss switched from cross-entropy to hinge loss by default
- `build_postfilter()` now accepts named architectures: `"standard"`, `"dilated"`, `"psd"`, `"film"`, `"pairaware"`, `"depthwise"`, `"luma"`, `"pixelshuffle"`

### Fixed
- Dead PoseNet gradients caused by upstream `@torch.no_grad` on `rgb_to_yuv6` — fixed via detach-aware wrapper
- Auto-kill at epoch 200 from phase-gating bug in Lagrangian annealing
- OOM fix in R1 Lagrangian patch being bypassed by callers passing the old default explicitly
- ffmpeg deadlock in `data.py` when stderr buffer fills on long videos
- `constrained_gen.py`: matplotlib, CLI viz, profiler integration, type hint issues

## [1.0.4] - 2026-04-10

### Added
- Lagrangian annealing for constrained optimization (rho growth schedule)
- Coupled PoseNet+SegNet optimizer with separate learning rates
- Self-compress pipeline: compress-then-optimize in a single pass
- `archive_optimizer.py`: rate-constrained archive byte budget enforcement
- `semantic_quantization.py`: task-aware quantization rounding

### Fixed
- Lagrangian cap values (alpha_seg corrected from 5000 to 200)

## [1.0.3] - 2026-04-07

### Added
- RAFT optical flow integration for motion-conditioned rendering
- `ego_flow.py`: ego-motion aware optical flow estimation
- Decorrelated batching for better gradient diversity (`decorrelated_batching.py`)
- `joint_pair_generator.py`: Y-shaped U-Net for mask-conditioned frame pair generation
- Wavelet renderer (`contrib/wavelet_renderer.py`)
- Diffusion-based decoder (`contrib/diffusion_renderer.py`)
- VQ-VAE codec (`contrib/vqvae_codec.py`)
- Scorer manifold geometry analysis (`contrib/scorer_manifold.py`)

### Changed
- KL distill loss mode disabled (failed authoritative eval — PoseNet collapse)
- `trick_stack.py` updated to include all Yousfi contest tricks

## [1.0.2] - 2026-04-03

### Added
- SWA (Stochastic Weight Averaging) in `Trainer`
- EMA checkpoint tracking with configurable decay
- Best-checkpoint selection based on proxy eval score
- `evaluate.py`: proxy scorer, top-K checkpoint averaging
- `proxy_eval.py`: lightweight evaluation without full scorer stack
- Precomputed data pipeline (skips 5-min video decode on startup)
- `ensemble.py`: multi-checkpoint ensemble inference

### Fixed
- Resume training from checkpoint restoring optimizer state correctly

## [1.0.1] - 2026-03-28

### Added
- Saliency-weighted loss (`roi_analysis.py`, `saliency.py`)
- `mask_generation.py`: SegNet mask extraction and caching
- `mask_codec.py`: AV1/VVC encoding of segmentation masks
- `camera.py`: comma.ai camera intrinsics and projection utilities
- `pose_extraction.py`: PoseNet output extraction utilities
- `versioned_output.py`: timestamped output directories for reproducibility

## [1.0.0] - 2026-03-20

### Added
- Initial release of `tac` (Task-Aware Codec)
- `tac.architectures`: `PostFilter`, `DilatedPostFilter`, `PixelShufflePostFilter` with 12 variant aliases
- `tac.training`: `Trainer` with QAT, EMA, best-checkpoint, lazy loading
- `tac.losses`: `scorer_loss`, `eval_scorer_loss`, `segnet_ste_loss`
- `tac.data`: video decoding, lazy pair construction via PyAV
- `tac.quantization`: FakeQuant STE, per-channel int8 save/load
- `tac.scorer`: scoring formula, sensitivity analysis, `load_scorers`, `detect_device`
- `tac.models`: Pydantic models (`ScoreResult`, `CheckpointMeta`, `TrainConfig`)
- `tac.profiles`: named training profiles for reproducible experiments
- CLI entry point: `tac lossy train`, `tac lossy eval`, `tac lossless compress/decompress`
- MIT license
