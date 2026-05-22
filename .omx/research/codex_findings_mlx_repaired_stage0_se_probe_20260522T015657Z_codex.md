# Codex Findings: MLX Repaired Stage-0 SE Full-Logit Probe

## Scope

This pass validates whether the SE-local repair candidates improve final SegNet logits on the known failing fec6 `[156,160]` window.

The probe compares four full-logit paths:

- `native`: tuple-axis SE pool, native `conv_expand`
- `cpu_pool_repair`: `mean_w_then_h` SE pool, native `conv_expand`
- `gpu_conv_expand_repair`: tuple-axis SE pool, explicit ordered 1x1 `conv_expand`
- `combined_repair`: `mean_w_then_h` SE pool, explicit ordered 1x1 `conv_expand`

The implementation is isolated in a new probe module and does not stage or absorb the dirty adapter/prefix-reset WIP already present in the worktree.

## Artifact

- module: `src/tac/local_acceleration/mlx_segnet_repaired_se_probe.py`
- CLI: `tools/probe_mlx_segnet_repaired_stage0_se.py`
- test: `src/tac/tests/test_mlx_segnet_repaired_se_probe.py`
- ignored CPU diagnostic JSON: `experiments/results/mlx_segnet_repaired_stage0_se_cpu_fec6_pr101_pair156_160_20260522T015657Z.json`
- ignored GPU diagnostic JSON: `experiments/results/mlx_segnet_repaired_stage0_se_gpu_fec6_pr101_pair156_160_20260522T015657Z.json`

CPU command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_repaired_stage0_se.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_repaired_stage0_se_cpu_20260522T015657Z \
  --output experiments/results/mlx_segnet_repaired_stage0_se_cpu_fec6_pr101_pair156_160_20260522T015657Z.json
```

GPU command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_repaired_stage0_se.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device gpu \
  --allow-gpu-research-signal \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_repaired_stage0_se_gpu_20260522T015657Z \
  --output experiments/results/mlx_segnet_repaired_stage0_se_gpu_fec6_pr101_pair156_160_20260522T015657Z.json
```

## CPU Result

Verdict: `REPAIRED_SE_IMPROVES_ARGMAX:cpu_pool_repair`

| variant | pool | conv_expand | argmax diff pixels | final logit max_abs_delta |
| --- | --- | --- | ---: | ---: |
| `native` | `mean_tuple` | `native` | `1` | `0.00037860870361328125` |
| `cpu_pool_repair` | `mean_w_then_h` | `native` | `0` | `0.00023031234741210938` |
| `gpu_conv_expand_repair` | `mean_tuple` | `explicit_ordered_1x1` | `1` | `0.0003960132598876953` |
| `combined_repair` | `mean_w_then_h` | `explicit_ordered_1x1` | `0` | `0.0002422332763671875` |

Interpretation: the CPU repair candidate is validated on the known failing window. `mean_w_then_h` removes the final SegNet argmax mismatch. Explicit `conv_expand` is not needed for CPU and slightly worsens logit delta versus pool-only repair.

## GPU Result

Verdict: `REPAIRED_SE_NO_IMPROVEMENT`

| variant | pool | conv_expand | argmax diff pixels | final logit max_abs_delta |
| --- | --- | --- | ---: | ---: |
| `native` | `mean_tuple` | `native` | `7` | `0.1348133087158203` |
| `cpu_pool_repair` | `mean_w_then_h` | `native` | `7` | `0.1348133087158203` |
| `gpu_conv_expand_repair` | `mean_tuple` | `explicit_ordered_1x1` | `10` | `0.11714458465576172` |
| `combined_repair` | `mean_w_then_h` | `explicit_ordered_1x1` | `10` | `0.11622905731201172` |

Interpretation: the GPU explicit-1x1 `conv_expand` repair reduces final logit drift but worsens argmax count on this window. It is not a candidate for broad adapter replacement yet. GPU needs a deeper full-path diagnosis before using local MLX GPU output as training authority.

## Recommended Next Step

CPU: patch the real stage-0 EfficientNet SE adapter to use `mean_w_then_h`, then rerun:

- `[156,160]` full parity;
- a broader CPU parity sweep;
- production-contract check.

GPU: keep `explicit_ordered_1x1` as diagnostic only. Do not route GPU MLX scorer outputs into optimization until a broader GPU-local drift trace explains the remaining `0.13` logit drift.

## Authority

This is local MLX implementation evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval remains required before any score or promotion claim
