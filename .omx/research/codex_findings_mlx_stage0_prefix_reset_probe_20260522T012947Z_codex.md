# Codex Findings: MLX SegNet Stage-0 Prefix-Reset Probe

## Scope

This follows the stage-0 cross-input probe, which found:

- native MLX drift cliffs at `encoder.stage_0.block_0.bn2`;
- forced PyTorch-input MLX ops remain below the `1.0e-4` local cliff threshold;
- the mismatch is therefore propagated small drift rather than a standalone `conv_pw` or `bn2` operator-local failure.

This pass asks which cumulative PyTorch-synchronized boundary, followed by native MLX execution through the rest of SegNet logits, first removes the final SegNet argmax mismatch.

## Artifact

- module: `src/tac/local_acceleration/mlx_segnet_prefix_reset_probe.py`
- CLI: `tools/probe_mlx_segnet_stage0_prefix_reset.py`
- test: `src/tac/tests/test_mlx_segnet_prefix_reset_probe.py`
- ignored diagnostic JSON: `experiments/results/mlx_segnet_stage0_prefix_reset_fec6_pr101_pair156_160_20260522T012947Z.json`

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_stage0_prefix_reset.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_stage0_prefix_reset_20260522T012947Z \
  --output experiments/results/mlx_segnet_stage0_prefix_reset_fec6_pr101_pair156_160_20260522T012947Z.json
```

## Result

Verdict: `PREFIX_RESET_FIXES_ARGMAX_AT:encoder.stage_0.block_0.se`

| reset boundary | SegNet argmax diff pixels | final logit max_abs_delta |
| --- | ---: | ---: |
| `input` | `1` | `0.00039064884185791016` |
| `encoder.stem` | `1` | `0.0004239082336425781` |
| `encoder.stage_0.block_0.conv_dw` | `1` | `0.0003927946090698242` |
| `encoder.stage_0.block_0.bn1` | `1` | `0.00039947032928466797` |
| `encoder.stage_0.block_0.se` | `0` | `0.0002779960632324219` |
| `encoder.stage_0.block_0.conv_pw` | `0` | `0.0002779960632324219` |
| `encoder.stage_0.block_0.bn2` | `0` | `0.0002484321594238281` |

## Interpretation

The first boundary that fixes the final argmax mismatch is the SE output. Resetting only through stem, depthwise conv, or `bn1` is insufficient. Resetting at SE, `conv_pw`, or `bn2` removes the argmax mismatch.

This narrows the next repair/fidelity target to the SE subpath and its input sensitivity:

- inspect MLX-vs-PyTorch SE reduction, sigmoid gate, and multiply behavior on the real `[156,160]` tensors;
- test whether higher precision or deterministic PyTorch-equivalent SE math removes the mismatch without replacing broader layers;
- keep BatchNorm replacement deprioritized because forced-input `bn2` already stayed below the local cliff threshold.

## Authority

This remains local MLX implementation evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval remains required before any score or promotion claim
