# Codex Findings: MLX Video Reference Cache + Real-Pair Scorer Smoke

Timestamp: 2026-05-21T23:02:26Z
Agent: Codex
Lane: mlx_auth_scorer_port
Evidence grade: local MLX CPU scorer-input response smoke
Score claim: false
Promotion eligible: false

## Summary

Extended the scorer-input cache builder so the MLX path can build the missing
reference side from upstream-format video files, not only candidate inflated
`.raw` files. This closes a practical harness gap: the evaluator compares
ground-truth `upstream/videos/0.mkv` against candidate `inflated/0.raw`, so the
local MLX scorer-response path needs both cache types.

Also ran a real two-pair FEC6 smoke:

- reference cache from `upstream/videos/0.mkv`;
- candidate cache from the FEC6 macOS advisory inflated raw;
- MLX scorer-response CLI over the two caches;
- upstream PyTorch DistortionNet comparison on the same tensors.

## Landed surfaces

- `tac.local_acceleration.mlx_preprocess.write_scorer_input_cache_from_video_file(...)`
- `tac.local_acceleration.mlx_preprocess.count_video_pairs(...)`
- `tools/build_mlx_scorer_input_cache.py --video ...`
- Regression coverage for capped video-cache CLI construction.

The video path mirrors upstream CPU `AVVideoDataset`:

- PyAV decode;
- YUV420 planar extraction;
- bilinear chroma upsampling;
- BT.601 limited-range RGB conversion;
- `round().to(uint8)`;
- existing scorer-input preprocessing for SegNet/PoseNet.

## Empirical verification

Lint and preprocess tests:

```text
.venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_preprocess.py \
  tools/build_mlx_scorer_input_cache.py \
  src/tac/tests/test_mlx_preprocess.py
All checks passed!

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_preprocess.py -q
11 passed in 3.83s
```

Cheap video-cache smoke:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tools/build_mlx_scorer_input_cache.py \
  --video upstream/videos/0.mkv \
  --output-dir experiments/results/mlx_scorer_input_cache_reference_video_smoke_20260521T_local_pairs2_fast \
  --max-pairs 2 \
  --batch-pairs 1

pair_count=2
segnet_last_rgb_shape=[2, 3, 384, 512]
posenet_yuv6_pair_shape=[2, 12, 192, 256]
```

Real FEC6 two-pair MLX scorer-response smoke:

```text
avg_posenet_dist=2.2402296053769533e-05
avg_segnet_dist=0.0006078084406908602
canonical_score=0.19454879749345233
n_samples=2
score_claim=false
```

Upstream PyTorch comparison on the exact same two-pair caches:

```text
torch_avg_posenet_dist=2.240004232589854e-05
torch_avg_segnet_dist=0.0006078084406908602
torch_canonical_score=0.19454804459546382

mlx_minus_torch_pose_avg=2.253727870993316e-09
mlx_minus_torch_seg_avg=0.0
mlx_minus_torch_score=7.528979885096376e-07
```

## Modal CPU transfer blocker

The existing full FEC6 macOS-built cache is not a valid Modal CPU transfer
calibration anchor:

```text
scorer_input_array_sha256_mismatch:segnet_last_rgb
scorer_input_array_sha256_mismatch:posenet_yuv6_pair
```

The `pair_indices` hash matches, but the scorer-input tensor hashes do not.
Therefore local MLX responses from that full tensor cache must not be compared
as if they were Modal Linux contest-CPU responses.

## Recommended next action

Build a full 600-pair local advisory reference cache from
`upstream/videos/0.mkv`, then run the MLX response harness against the existing
macOS advisory FEC6 candidate cache. That will give a full local PyTorch-vs-MLX
training-signal parity anchor.

For Modal CPU parity, either recover/download the Modal inflated raw bytes or
run a new Modal CPU job that exports enough scorer-response artifacts to avoid
multi-GB tensor transfer. Until then, Modal transfer calibration remains
blocked by scorer-input identity mismatch.
