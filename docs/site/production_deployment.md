# Production Deployment

## From contest to fleet

The comma.ai video compression challenge models a real engineering problem: compressing dashcam video while preserving what the driving stack actually needs. Our system --- a 287K-parameter renderer that generates frames from semantic masks, trained end-to-end against comma's SegNet and PoseNet scorers --- achieves a score of 0.37 with test-time optimization [unlimited-compute] and 0.61 [contest-compliant] with distillation. The current leader (Quantizr) scores 0.33. The architecture maps directly to comma's production data pipeline.

## Where it fits

comma's fleet of 250,000+ devices captures petabytes of driving video, uploaded over cellular and stored on servers for training data. Our renderer compresses 60 seconds of video into a ~185 KB archive (masks + weights) --- a 14x reduction over H.265, 170x over raw video. Reconstruction takes 3 seconds on a T4 GPU with 0.094% SegNet cross-entropy disagreement and 0.25% PoseNet L2 error (with TTO). The archive includes everything needed to reconstruct: no external model weights, no scorer loading, no dependencies beyond a standard PyTorch runtime.

## Scalability

The system has a compute-quality knob. Renderer-only reconstruction (3 seconds, score 0.87) is cheap enough for real-time mobile playback on Snapdragon hardware. Distillation training teaches the renderer to reproduce TTO-quality frames in a single forward pass (score 0.61 [contest-compliant], still converging toward 0.45). Adding pose-space TTO (optimizing 6D FiLM conditioning vectors at compress time, 14.4 KB archive cost) reduces PoseNet distortion by 94.7% without any scorer at inflate time. The full stack --- distilled renderer + pose-space TTO + FP4 quantization (215 KB archive) + MiniSegNet inflate TTO --- projects to sub-0.25 [contest-compliant]. For fleet archival, the amortized cost is approximately $0.001 per route for renderer-only, or $0.17 per route for full distillation+TTO on a 4090.

## Details

For a full analysis --- including comma hardware specs, per-operation latency estimates, cost modeling for fleet-scale deployment, and a discussion of limitations (single-video evaluation, real-time encoding, fleet diversity) --- see [Section 5 of the paper](../paper/05_production.md).

## Evaluation Hardware Caveat

Contest scores are not a single hardware-independent scalar. The same archive
must be evaluated on both CUDA and CPU paths because `upstream/evaluate.py`
changes the ground-truth loader and scorer device. For production-facing
claims, report the archive SHA, runtime-tree SHA, `[contest-CUDA]` score, and
Linux x86_64 `[contest-CPU]` score. macOS CPU is useful for fast local sweeps
after PR107 showed a `6e-6` gap from GitHub Actions Linux CPU, but it remains
an advisory development signal until promoted by a Linux replay.
