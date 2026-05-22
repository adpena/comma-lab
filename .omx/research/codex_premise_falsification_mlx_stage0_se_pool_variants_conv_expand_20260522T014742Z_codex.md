# Codex Premise Falsification: MLX SE Pool Probe Missing `conv_expand_delta`

## Scope

The prior pool-variant memo `codex_findings_mlx_stage0_se_pool_variants_20260522T014359Z_codex.md` correctly identified the CPU pooling repair candidate, but its GPU interpretation was under-instrumented.

It reported that MLX GPU pooling was tight and that the visible drift emerged at gate/output. The probe did not emit `conv_expand_delta`, so that conclusion was incomplete.

## Fix

`src/tac/local_acceleration/mlx_segnet_se_pool_variants.py` now emits `conv_expand_delta` for every variant.

`src/tac/tests/test_mlx_segnet_se_pool_variants.py` asserts the field is present and shape-matched.

## Corrected CPU Result

The CPU conclusion is unchanged. `mean_w_then_h` remains the best local pool variant on fec6 `[156,160]`.

| variant | pool max_abs_delta | conv_expand max_abs_delta | gate max_abs_delta | SE output max_abs_delta |
| --- | ---: | ---: | ---: | ---: |
| `mean_tuple` | `0.00009742379188537598` | `0.000007867813110351562` | `0.0000014901161193847656` | `0.0000362396240234375` |
| `mean_h_then_w` | `0.00000286102294921875` | `0.000000476837158203125` | `0.00000005960464477539063` | `0.000003337860107421875` |
| `mean_w_then_h` | `0.0000026226043701171875` | `0.0000002384185791015625` | `0.00000005960464477539063` | `0.0000019073486328125` |
| `sum_tuple_div` | `0.00009745359420776367` | `0.000007867813110351562` | `0.0000014901161193847656` | `0.0000362396240234375` |

## Corrected GPU Result

The GPU bottleneck is not pooling and should not be attributed first to sigmoid. The missing row shows the larger upstream GPU drift occurs at `se.conv_expand`.

| variant | pool max_abs_delta | conv_expand max_abs_delta | gate max_abs_delta | SE output max_abs_delta |
| --- | ---: | ---: | ---: | ---: |
| `mean_tuple` | `0.00000095367431640625` | `0.0005261898040771484` | `0.00009104609489440918` | `0.0027246475219726562` |
| `mean_h_then_w` | `0.00000095367431640625` | `0.0005261898040771484` | `0.00009104609489440918` | `0.0027246475219726562` |
| `mean_w_then_h` | `0.00000095367431640625` | `0.0005261898040771484` | `0.00009104609489440918` | `0.0027246475219726562` |
| `sum_tuple_div` | `0.0000007152557373046875` | `0.0005261898040771484` | `0.00009104609489440918` | `0.0027246475219726562` |

## Updated Interpretation

There are two separate repair lanes:

- MLX CPU: tuple-axis SE pooling is the immediate repair candidate; `mean_w_then_h` is best on this window.
- MLX GPU: SE `conv_expand` is the immediate repair candidate; pooling variants do not change the output error on this window.

The next GPU probe should compare native MLX `Conv2d` against an explicit 1x1 conv implementation for stage-0 SE `conv_expand`.

## Authority

This is local MLX implementation evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval remains required before any score or promotion claim
