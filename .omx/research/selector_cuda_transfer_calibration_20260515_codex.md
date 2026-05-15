# Selector CUDA Transfer Calibration

- score_claim: `false`
- dispatch_attempted: `false`
- calibration_status: `blocked`
- ready_for_broad_waterfill_dispatch: `false`
- ready_for_exact_eval_dispatch: `false`

## Decision

Do not route broad proxy-ranked film-grain/selector/water-fill dispatch until a CUDA-in-loop selector control is positive or neutral, or a transfer model explains the current regressions.

## Blockers

- `exact_cuda_positive_or_neutral_selector_control_missing`
- `measured_selector_controls_transfer_negative_on_cuda`
- `pr101_cpu_cuda_inflated_output_aggregate_mismatch`
- `cpu_cuda_gap_component_dominated_not_rate_limited`

## PR101 FEC6 CPU/CUDA Drift

- classification: `cpu_positive_cuda_miss_due_to_component_drift`
- dominant score-delta component: `pose`
- score-delta byte equivalent: `51300.21103151698`
- raw aggregate match: `false`

## Exact-CUDA Selector Rows

| technique | score delta | byte equivalent | outcome |
|---|---:|---:|---|
| `hdm8_cuda_selector_sparse_top001_exact_cuda_review` | 0.000117997857 | 177 | `cuda_regression` |
| `hdm8_fixed_even_rgb_bias_m1_p05_p05_positive_control_exact_cuda_review` | 0.000423067182 | 635 | `cuda_regression` |
| `hdm8_fixed_even_grain_chroma_1_positive_control_exact_cuda_review` | 0.000466466070 | 701 | `cuda_regression` |
| `hdm8_fixed_even_rgb_bias_m05_p05_0_positive_control_exact_cuda_review` | 0.000512465845 | 770 | `cuda_regression` |
| `hdm8_fixed_even_rgb_bias_0_p05_m05_positive_control_exact_cuda_review` | 0.000633681498 | 952 | `cuda_regression` |
| `hdm8_cuda_selector_sparse_budget128_exact_cuda_review` | 0.001515513345 | 2276 | `cuda_regression` |
| `hdm8_film_grain_selector_charged_mps_aggressive_v2_cuda_review` | 0.009748252358 | 14640 | `cuda_regression` |
| `hdm8_even_frame_selector_cuda_review` | 0.021803620925 | 32745 | `cuda_regression` |

## Next Actions

- Build a CUDA-in-loop selector objective before broad water-fill dispatch.
- Continue PR101 CPU-only selector work only when the candidate changes components by more than the charged byte cost.
- Treat rate-only PR101 FEC6 polishing as insufficient for CUDA while the paired aggregate hashes differ.
- Use PR106 format0B as the closed PacketIR reference; do not redispatch identical archives.
