# HDM8 Selector CUDA Calibration Table

- score_claim: `false`
- calibration_status: `blocked`
- ready_for_broad_waterfill_dispatch: `false`

## Exact-CUDA Rows

| technique | kind | mode | score delta | outcome |
|---|---|---|---:|---|
| `hdm8_cuda_selector_sparse_top001_exact_cuda_review` | `selector_sparse` | `` | 0.00011799785677868435 | `exact_cuda_regression` |
| `hdm8_cuda_selector_sparse_budget128_exact_cuda_review` | `selector_sparse` | `` | 0.0015155133447327107 | `exact_cuda_regression` |
| `hdm8_fixed_even_rgb_bias_m1_p05_p05_positive_control_exact_cuda_review` | `fixed_mode` | `even_rgb_bias:-1,0.5,0.5` | 0.0004230671821068843 | `exact_cuda_regression` |
| `hdm8_fixed_even_rgb_bias_0_p05_m05_positive_control_exact_cuda_review` | `fixed_mode` | `even_rgb_bias:0,0.5,-0.5` | 0.000633681497760552 | `exact_cuda_regression` |
| `hdm8_fixed_even_rgb_bias_m05_p05_0_positive_control_exact_cuda_review` | `fixed_mode` | `even_rgb_bias:-0.5,0.5,0` | 0.0005124658445844199 | `exact_cuda_regression` |
| `hdm8_fixed_even_grain_chroma_1_positive_control_exact_cuda_review` | `fixed_mode` | `even_grain_chroma:1` | 0.000466466069987459 | `exact_cuda_regression` |

## Blockers

- `exact_cuda_positive_or_neutral_control_missing`
- `proxy_positive_calibration_rows_transferred_negative`
- `broad_waterfill_selector_blocked_until_transfer_model`

## Policy

Broad proxy-ranked selector and waterfill dispatch stays blocked until an exact-CUDA positive/neutral control exists or a calibrated transfer model explains why the current proxy-positive rows regress.
