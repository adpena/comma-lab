# Codex Findings: MLX Stage-0 SE Conv Variant Probe

## Scope

The corrected SE pool probe showed:

- MLX CPU: tuple-axis pooling is the immediate issue; `mean_w_then_h` is best on fec6 `[156,160]`.
- MLX GPU: pooling is tight, but `se.conv_expand` drifts by `0.0005261898040771484`.

This pass compares native MLX `Conv2d` against an explicit ordered 1x1 implementation for the two stage-0 SE convs:

- `conv_reduce`
- `conv_expand`

The probe uses exact PyTorch SE inputs for each conv so it isolates conv operator drift.

## Artifact

- module: `src/tac/local_acceleration/mlx_segnet_se_conv_variants.py`
- CLI: `tools/probe_mlx_segnet_stage0_se_conv_variants.py`
- test: `src/tac/tests/test_mlx_segnet_se_conv_variants.py`
- ignored CPU diagnostic JSON: `experiments/results/mlx_segnet_stage0_se_conv_variants_cpu_fec6_pr101_pair156_160_20260522T015135Z.json`
- ignored GPU diagnostic JSON: `experiments/results/mlx_segnet_stage0_se_conv_variants_gpu_fec6_pr101_pair156_160_20260522T015135Z.json`

CPU command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_stage0_se_conv_variants.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_stage0_se_conv_variants_cpu_20260522T015135Z \
  --output experiments/results/mlx_segnet_stage0_se_conv_variants_cpu_fec6_pr101_pair156_160_20260522T015135Z.json
```

GPU command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_stage0_se_conv_variants.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device gpu \
  --allow-gpu-research-signal \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_stage0_se_conv_variants_gpu_20260522T015135Z \
  --output experiments/results/mlx_segnet_stage0_se_conv_variants_gpu_fec6_pr101_pair156_160_20260522T015135Z.json
```

## CPU Result

Verdict: `EXPLICIT_1X1_DOES_NOT_IMPROVE`

| op | native MLX Conv2d max_abs_delta | explicit ordered 1x1 max_abs_delta |
| --- | ---: | ---: |
| `conv_reduce` | `0.0000002384185791015625` | `0.0000011920928955078125` |
| `conv_expand` | `0.0000002384185791015625` | `0.0000002384185791015625` |

Interpretation: do not use explicit 1x1 as a CPU repair for stage-0 SE. CPU should pursue the sequential pooling repair instead.

## GPU Result

Verdict: `EXPLICIT_1X1_IMPROVES:conv_expand`

| op | native MLX Conv2d max_abs_delta | explicit ordered 1x1 max_abs_delta |
| --- | ---: | ---: |
| `conv_reduce` | `0.000000476837158203125` | `0.0000011920928955078125` |
| `conv_expand` | `0.0005261898040771484` | `0.0000002384185791015625` |

Interpretation: the MLX GPU stage-0 SE issue is isolated to native `Conv2d` for `conv_expand`; explicit ordered 1x1 eliminates the observed drift on this window.

## Recommended Next Repair

Build a narrow experimental adapter mode for stage-0 SE only:

- CPU: `mean_w_then_h` pooling, native convs.
- GPU: native pooling, native `conv_reduce`, explicit ordered 1x1 for `conv_expand`.

Then run the known-failing `[156,160]` full SegNet parity window before broadening to a sweep. Do not generalize this to all Conv2d layers without profiling; CPU explicit 1x1 was not a win for `conv_reduce`.

## Authority

This is local MLX implementation evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval remains required before any score or promotion claim
