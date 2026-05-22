# Codex Findings: MLX SegNet Stage-0 SE Trace

## Scope

The prefix-reset probe found the first final-argmax repair boundary at:

- `encoder.stage_0.block_0.se`

This pass traces the squeeze-excite subpath itself on the real fec6 `[156,160]` tensors, comparing:

- native MLX input path;
- forced PyTorch `encoder.stage_0.block_0.bn1` input path.

## Artifact

- module: `src/tac/local_acceleration/mlx_segnet_se_trace.py`
- CLI: `tools/trace_mlx_segnet_stage0_se.py`
- test: `src/tac/tests/test_mlx_segnet_se_trace.py`
- ignored diagnostic JSON: `experiments/results/mlx_segnet_stage0_se_trace_fec6_pr101_pair156_160_20260522T013352Z.json`

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/trace_mlx_segnet_stage0_se.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_stage0_se_trace_20260522T013352Z \
  --output experiments/results/mlx_segnet_stage0_se_trace_fec6_pr101_pair156_160_20260522T013352Z.json
```

## Result

Verdict: `SE_FORCED_OUTPUT_EXCEEDS_CLIFF` at the stricter `1.0e-5` SE cliff.

| SE row | native max_abs_delta | forced torch-input max_abs_delta |
| --- | ---: | ---: |
| `se.input` | `0.000026702880859375` | `0.0` |
| `se.pool` | `0.00009742379188537598` | `0.00009742379188537598` |
| `se.conv_reduce` | `5.73841667175293` | `5.73841667175293` |
| `se.act1_silu` | `0.000003509223461151123` | `0.000003509223461151123` |
| `se.conv_expand` | `0.000008106231689453125` | `0.000007867813110351562` |
| `se.gate_sigmoid` | `0.0000014901161193847656` | `0.0000014901161193847656` |
| `se.output_multiply` | `0.0000362396240234375` | `0.0000362396240234375` |

## Interpretation

The forced-input trace makes the current repair target concrete:

- the SE input itself is exact in forced mode;
- the first forced drift appears at `se.pool`, so the reduction path is suspect;
- `se.conv_reduce` has a large intermediate delta, but the subsequent SiLU saturates it down to a small delta;
- the final SE output still has `3.62396240234375e-05` max_abs_delta, enough for the previously observed one-pixel downstream argmax sensitivity.

This points to testing a PyTorch-equivalent SE pooling/reduction implementation or a deterministic higher-precision pool before touching broader SegNet layers.

## Authority

This is local MLX implementation evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval remains required before any score or promotion claim
