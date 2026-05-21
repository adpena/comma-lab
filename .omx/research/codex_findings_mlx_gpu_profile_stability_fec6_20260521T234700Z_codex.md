# Codex Findings: MLX GPU scorer-response profile stability on FEC6

## Summary

Ran the new MLX scorer-response profile stability gate against a small FEC6
pair window to classify MLX GPU as either CPU-transfer-stable or fast
prescreen-only.

Verdict: MLX GPU is fast and internally repeat-stable on this window, but it is
not CPU-transfer-stable against the MLX CPU baseline. Use GPU scorer-response
output as a fast prescreen only until a stricter CPU/GPU transfer calibration
passes.

## Profile 1: CPU vs GPU, same 4-pair window

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/profile_mlx_scorer_response_cache.py \
  --reference-cache-dir experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600 \
  --candidate-cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --archive-size-bytes 178517 \
  --repo-root . \
  --devices cpu,gpu \
  --batch-pairs 1,2 \
  --start-pair 16 \
  --max-pairs 4 \
  --output experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/profile.json
```

Best throughput row:

- device: `gpu`
- batch_pairs: `1`
- elapsed_seconds: `0.3426780700683594`
- pairs_per_second: `11.672763299974397`
- canonical_score: `0.1788319982014162`

CPU-baseline stability gate:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_mlx_scorer_response_profile_stability.py \
  --profile experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/profile.json \
  --output experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/stability_cpu_baseline.json \
  --run-id mlx_profile_fec6_pairs16_4_cpu_gpu_stability_20260521 \
  --baseline-device cpu \
  --baseline-batch-pairs 1
```

Observed verdict:

- `passed`: `false`
- `verdict`: `FAIL_MLX_PROFILE_STABILITY`
- CPU batch_pairs=2 vs CPU batch_pairs=1:
  - `score_delta`: `3.6323648794356345e-7`
  - `posenet_avg_delta`: `5.688605142495362e-10`
  - `segnet_avg_delta`: `0.0`
- GPU batch_pairs=1 vs CPU batch_pairs=1:
  - `score_delta`: `0.0006361803902723562`
  - `posenet_avg_delta`: `6.217817372089485e-10`
  - `segnet_avg_delta`: `6.357833626680076e-6`
- GPU batch_pairs=2 vs CPU batch_pairs=1:
  - `score_delta`: `0.006707996834292129`
  - `posenet_avg_delta`: `1.1133447245015304e-5`
  - `segnet_avg_delta`: `1.398719905409962e-5`

Interpretation: CPU batch sizing is locally stable on this window. GPU batch 1
is fast but has SegNet-driven CPU-transfer drift above the local threshold.
GPU batch 2 is not usable for scorer-response optimization without additional
calibration.

## Profile 2: GPU repeat stability, same 4-pair window

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/profile_mlx_scorer_response_cache.py \
  --reference-cache-dir experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600 \
  --candidate-cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --archive-size-bytes 178517 \
  --repo-root . \
  --devices gpu \
  --batch-pairs 1 \
  --repeat 3 \
  --start-pair 16 \
  --max-pairs 4 \
  --output experiments/results/mlx_scorer_response_profile_fec6_20260521T2345Z_pairs16_4_gpu_repeat/profile.json
```

GPU-baseline stability gate:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_mlx_scorer_response_profile_stability.py \
  --profile experiments/results/mlx_scorer_response_profile_fec6_20260521T2345Z_pairs16_4_gpu_repeat/profile.json \
  --output experiments/results/mlx_scorer_response_profile_fec6_20260521T2345Z_pairs16_4_gpu_repeat/stability_gpu_baseline.json \
  --run-id mlx_profile_fec6_pairs16_4_gpu_repeat_stability_20260521 \
  --baseline-device gpu \
  --baseline-batch-pairs 1
```

Observed verdict:

- `passed`: `true`
- `verdict`: `PASS_MLX_PROFILE_STABILITY`
- all repeat deltas: `0.0`

## Operational policy

- MLX CPU scorer-response remains the local calibration reference.
- MLX GPU scorer-response may be used as fast prescreen signal only when paired
  with CPU spot checks.
- MLX GPU scorer-response must not drive rank/kill, promotion, or paid-dispatch
  decisions until CPU-transfer stability passes on the relevant window and batch
  shape.
- For scorer-response training, prefer `gpu batch_pairs=1` for exploratory
  throughput and periodically re-score the same candidates on MLX CPU before
  selecting exact-eval dispatch targets.

## Authority status

This memo records local MLX evidence only. It is not an auth-eval score, not a
contest-axis result, and not sufficient for candidate promotion.
