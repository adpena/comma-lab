# Codex Findings: MLX EfficientNet SE Pool Repair Landing

## Scope

The repaired stage-0 SE full-logit probe showed that the CPU repair candidate `mean_w_then_h` removes the known fec6 `[156,160]` one-pixel SegNet argmax mismatch.

This pass promotes that CPU repair into the real `MLXEfficientNetSqueezeExciteAdapter` pooling path.

## Patch

`src/tac/local_acceleration/mlx_scorer_adapters.py`:

```python
x_se = mx.mean(mx.mean(x_nhwc, axis=2, keepdims=True), axis=1, keepdims=True)
```

This replaces tuple-axis spatial pooling inside `MLXEfficientNetSqueezeExciteAdapter`.

## Regression Test

`src/tac/tests/test_mlx_efficientnet_se_pool_repair.py` verifies the adapter uses sequential width-then-height pooling, not tuple-axis pooling.

## Empirical Check

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_mlx_scorer_torch_parity.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_adapter_se_pool_repair_cpu_20260522T015934Z \
  --output experiments/results/mlx_torch_parity_adapter_se_pool_repair_cpu_fec6_pr101_pair156_160_20260522T015934Z.json
```

Result:

- verdict: `PASS_MLX_TORCH_SCORER_PARITY`
- `segnet_argmax_diff_pixels`: `0`
- `segnet_logit_abs_max`: `0.0000858306884765625`
- `posenet_output_abs_max`: `0.000003814697265625`
- `posenet_component_abs_max`: `2.4303614676313146e-12`

Important caveat: the worktree still contains unrelated dirty MLX adapter/prefix-reset WIP. The committed patch is only the EfficientNet SE pooling hunk plus its test. The empirical check above was run in the live dirty worktree and should be treated as a positive local smoke, not as isolated proof for the single hunk.

## Authority

This remains local MLX implementation evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval remains required before any score or promotion claim

## Next Action

Once the unrelated MLX WIP is either landed or cleared by its owner, rerun the CPU parity sweep from a clean worktree to confirm the single-hunk repair generalizes beyond the known `[156,160]` window.
