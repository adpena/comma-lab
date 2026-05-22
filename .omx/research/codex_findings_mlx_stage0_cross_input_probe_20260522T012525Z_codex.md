# Codex Findings: MLX SegNet Stage-0 Cross-Input Probe

## Scope

Follow-up to the MLX SegNet layer trace that found the first local drift cliff at:

- cache: `experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs`
- pair window: `[156, 160]`
- prior cliff: `encoder.stage_0.block_0.bn2`
- prior native max_abs_delta: `0.00026702880859375`
- prior symptom: 1 SegNet argmax pixel mismatch

This pass adds an op-local cross-input probe. Each `stage_0.block_0` MLX op is compared in two modes:

- native MLX path, where upstream MLX drift propagates normally;
- forced torch-input path, where the MLX op receives the exact PyTorch tensor consumed by the equivalent PyTorch op.

The forced-input path isolates operator-local drift from propagated input drift.

## Artifact

- module: `src/tac/local_acceleration/mlx_segnet_cross_input_probe.py`
- CLI: `tools/probe_mlx_segnet_stage0_cross_input.py`
- test: `src/tac/tests/test_mlx_segnet_cross_input_probe.py`
- ignored diagnostic JSON: `experiments/results/mlx_segnet_stage0_cross_input_fec6_pr101_pair156_160_20260522T012525Z.json`

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_stage0_cross_input.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_stage0_cross_input_20260522T012525Z \
  --output experiments/results/mlx_segnet_stage0_cross_input_fec6_pr101_pair156_160_20260522T012525Z.json
```

## Result

Verdict: `UPSTREAM_INPUT_DRIFT_DOMINATES`

The native path still cliffs at `encoder.stage_0.block_0.bn2`, but no forced-input row crosses the `1.0e-4` cliff threshold.

| row | native max_abs_delta | forced torch-input max_abs_delta |
| --- | ---: | ---: |
| `input` | `0.0` | `0.0` |
| `encoder.stem` | `0.0000133514404296875` | `0.0000133514404296875` |
| `encoder.stage_0.block_0.conv_dw` | `0.000023365020751953125` | `0.0000152587890625` |
| `encoder.stage_0.block_0.bn1` | `0.000026702880859375` | `0.00000762939453125` |
| `encoder.stage_0.block_0.se` | `0.0000362396240234375` | `0.0000362396240234375` |
| `encoder.stage_0.block_0.conv_pw` | `0.000036716461181640625` | `0.0` |
| `encoder.stage_0.block_0.bn2` | `0.00026702880859375` | `0.000011444091796875` |

## Interpretation

The deterministic affine BatchNorm repair was already falsified. This probe rules out a narrower replacement theory: the `[156,160]` mismatch is not explained by an isolated MLX `conv_pw` or `bn2` operator drift at the current cliff threshold. When those ops receive the exact PyTorch input tensors, their output deltas collapse below threshold.

The failing native `bn2` row is therefore a propagation/amplification problem: small earlier drift, especially through the stem and depthwise/se path, is amplified by the full native block enough to perturb one downstream SegNet argmax pixel.

## Authority

This is local MLX implementation evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval remains required before any score or promotion claim

## Next Action

Build a cumulative-prefix reset probe for `stage_0.block_0`: feed PyTorch-synchronized tensors at each boundary and continue the rest of SegNet natively in MLX through logits. That will identify the earliest boundary reset that eliminates the final one-pixel argmax mismatch and prioritize the smallest repair surface for MLX training fidelity.
