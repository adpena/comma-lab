# HiNeRV QAT 32-Pair Sampled MLX Demotion

UTC: 2026-06-01T22:50:00Z
Author: Codex
Axis: [macOS-MLX research-signal], non-authority

## Run

HiNeRV was trained for 32 pairs with the real SegNet/PoseNet teachers, the
8-epoch PR95-faithful curriculum gate satisfied, and decoder coder-aware QAT
enabled.

Report:

`/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_qat_realteacher_32pair_20260601T224932Z/compact_renderer_mlx_spine_runner_report.json`

Archive:

- path: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_qat_realteacher_32pair_20260601T224932Z/hi_nerv_mlx_training/archive.zip`
- bytes: 41,488
- sha256: `1cac04e39cce36651ba5e9e856d42057f71b662ea55a599c7cb76d23e4873242`
- receiver proof emitted under the run directory
- score claim: false
- ready for exact eval dispatch: false

## Sampled MLX Scorer Response

Cache/profile root:

`/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_qat_realteacher_32pair_20260601T224932Z/mlx_prefilter_32pair`

Response:

`/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_qat_realteacher_32pair_20260601T224932Z/mlx_prefilter_32pair/mlx_scorer_response.json`

Result:

- pairs: 32
- batch pairs: 1
- hardware: MLX gpu
- canonical advisory score: 93.8341124272197
- avg SegNet distortion: 0.505702493712306
- avg PoseNet distortion: 186.93722677230835
- score rate contribution: 0.027625156247132646
- archive-size bytes: 41,488
- prefilter cache footprint: 288M on SSD
- retained raw/video artifacts: none found under the run directory

## Verdict

This is not a promotion or CPU replay candidate. It is a precise local-basin
demotion for the current tiny HiNeRV configuration: rate is cheap, but
distortion remains unusable even after real teachers, 8 epochs, and coder-aware
QAT. The next score-lowering step must change the representation/training
geometry rather than replay this archive harder.

The next viable arms are:

- protected high-resolution pose pathway / flow-conditioned pose channel
- SR-low-resolution carrier with scorer-downsample-aware output
- SNeRV/HiNeRV larger fit with native rate pressure throughout training
- full-video P18/P19 scorer surfaces as training weights, not post-hoc leaf edits
- PR95/HNeRV control-arm parity and bounded continuation as the acceptance floor
