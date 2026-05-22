# Codex Premise Falsification: MLX Stage-0 SE Trace In-Place SiLU Label Bug

## Scope

The prior SE trace memo `codex_findings_mlx_stage0_se_trace_20260522T013352Z_codex.md` reported a large `se.conv_reduce` forced-input delta of `5.73841667175293` and treated it as a raw `conv_reduce` intermediate.

That premise was false.

## Root Cause

PyTorch stage-0 `SqueezeExcite.act1` is `SiLU(inplace=True)`. The tracer stored the `conv_reduce` tensor after calling the in-place activation, so the row labeled `se.conv_reduce` actually contained post-SiLU values. MLX `mlx_silu` is not in-place, so the trace compared:

- MLX raw `conv_reduce`
- PyTorch post-SiLU tensor mislabeled as raw `conv_reduce`

This produced a false large delta.

## Fix

`src/tac/local_acceleration/mlx_segnet_se_trace.py` now clones `conv_reduce` before the in-place PyTorch activation:

```python
conv_reduce = se.conv_reduce(pooled)
conv_reduce_raw = conv_reduce.clone()
act1 = se.act1(conv_reduce)
```

`src/tac/tests/test_mlx_segnet_se_trace.py` now asserts forced-input `se.conv_reduce` drift stays below `1.0e-4`, which catches this label/mutation bug class.

## Corrected Result

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/trace_mlx_segnet_stage0_se.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_stage0_se_trace_corrected_20260522T013917Z \
  --output experiments/results/mlx_segnet_stage0_se_trace_corrected_fec6_pr101_pair156_160_20260522T013917Z.json
```

Corrected verdict: `SE_FORCED_OUTPUT_EXCEEDS_CLIFF`

Corrected dominant forced row: `se.pool`

| SE row | native max_abs_delta | forced torch-input max_abs_delta |
| --- | ---: | ---: |
| `se.input` | `0.000026702880859375` | `0.0` |
| `se.pool` | `0.00009742379188537598` | `0.00009742379188537598` |
| `se.conv_reduce` | `0.000058650970458984375` | `0.000058650970458984375` |
| `se.act1_silu` | `0.000003509223461151123` | `0.000003509223461151123` |
| `se.conv_expand` | `0.000008106231689453125` | `0.000007867813110351562` |
| `se.gate_sigmoid` | `0.0000014901161193847656` | `0.0000014901161193847656` |
| `se.output_multiply` | `0.0000362396240234375` | `0.0000362396240234375` |

## Updated Interpretation

The stage-0 SE target remains valid, but the prior `conv_reduce` magnitude claim is invalid. The corrected first forced-input drift remains `se.pool`, and the next repair probe should compare MLX reduction variants against PyTorch `x.mean((2, 3), keepdim=True)`.

## Authority

This is local MLX implementation evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval remains required before any score or promotion claim
