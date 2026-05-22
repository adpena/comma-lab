# Codex Findings: MLX Stage-0 SE Pool Variant Probe

## Scope

After correcting the in-place PyTorch SiLU trace bug, the stage-0 SE trace showed the first forced-input drift at `se.pool`. This pass compares MLX pooling variants against PyTorch `x.mean((2, 3), keepdim=True)` on the real fec6 `[156,160]` tensors.

The probe is intentionally SE-local and does not depend on the dirty SegNet-head adapter WIP currently present in the worktree.

## Artifact

- module: `src/tac/local_acceleration/mlx_segnet_se_pool_variants.py`
- CLI: `tools/probe_mlx_segnet_stage0_se_pool_variants.py`
- test: `src/tac/tests/test_mlx_segnet_se_pool_variants.py`
- ignored CPU diagnostic JSON: `experiments/results/mlx_segnet_stage0_se_pool_variants_cpu_fec6_pr101_pair156_160_20260522T014359Z.json`
- ignored GPU diagnostic JSON: `experiments/results/mlx_segnet_stage0_se_pool_variants_gpu_fec6_pr101_pair156_160_20260522T014359Z.json`

CPU command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_stage0_se_pool_variants.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_stage0_se_pool_variants_cpu_20260522T014359Z \
  --output experiments/results/mlx_segnet_stage0_se_pool_variants_cpu_fec6_pr101_pair156_160_20260522T014359Z.json
```

GPU command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_stage0_se_pool_variants.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device gpu \
  --allow-gpu-research-signal \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_stage0_se_pool_variants_gpu_20260522T014359Z \
  --output experiments/results/mlx_segnet_stage0_se_pool_variants_gpu_fec6_pr101_pair156_160_20260522T014359Z.json
```

## CPU Result

Verdict: `BEST_SE_POOL_VARIANT:mean_w_then_h`

| variant | pool max_abs_delta | conv_reduce max_abs_delta | gate max_abs_delta | SE output max_abs_delta |
| --- | ---: | ---: | ---: | ---: |
| `mean_tuple` | `0.00009742379188537598` | `0.000058650970458984375` | `0.0000014901161193847656` | `0.0000362396240234375` |
| `mean_h_then_w` | `0.00000286102294921875` | `0.000002384185791015625` | `0.00000005960464477539063` | `0.000003337860107421875` |
| `mean_w_then_h` | `0.0000026226043701171875` | `0.000002384185791015625` | `0.00000005960464477539063` | `0.0000019073486328125` |
| `sum_tuple_div` | `0.00009745359420776367` | `0.0000591278076171875` | `0.0000014901161193847656` | `0.0000362396240234375` |

Interpretation: On MLX CPU, tuple-axis reduction is the wrong primitive for this numerically sensitive SE pool. Sequential reduction, especially width-then-height, cuts final SE output drift by about 19x on this window.

## GPU Result

Verdict: `BEST_SE_POOL_VARIANT:mean_tuple`

| variant | pool max_abs_delta | conv_reduce max_abs_delta | gate max_abs_delta | SE output max_abs_delta |
| --- | ---: | ---: | ---: | ---: |
| `mean_tuple` | `0.00000095367431640625` | `0.00000095367431640625` | `0.00009104609489440918` | `0.0027246475219726562` |
| `mean_h_then_w` | `0.00000095367431640625` | `0.00000095367431640625` | `0.00009104609489440918` | `0.0027246475219726562` |
| `mean_w_then_h` | `0.00000095367431640625` | `0.00000095367431640625` | `0.00009104609489440918` | `0.0027246475219726562` |
| `sum_tuple_div` | `0.0000007152557373046875` | `0.0000007152557373046875` | `0.00009104609489440918` | `0.0027246475219726562` |

Interpretation: On MLX GPU, pooling is not the bottleneck on this window. The SE gate/output drift is dominated by the activation/gate path, likely `mx.exp`/sigmoid behavior on this sensitive tensor. That needs a separate GPU gate probe and should not be mixed with the CPU pool repair.

## Recommended Next Repair

For CPU: test a local adapter patch that changes `MLXEfficientNetSqueezeExciteAdapter` pooling from tuple-axis mean to `mean_w_then_h`, then rerun:

- SE trace;
- stage-0 prefix reset;
- full MLX-vs-PyTorch parity sweep on the known failing `[156,160]` window.

For GPU: do not apply the CPU pool result as a claimed GPU fix. Build a separate SE gate/sigmoid variant probe.

## Authority

This remains local MLX implementation evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval remains required before any score or promotion claim
