# HDM8 Fixed-Mode Exact-CUDA Positive-Control Sweep - 2026-05-15

score_claim: `false`
promotion_eligible: `false`
axis: `[contest-CUDA]`

## Purpose

After the broad HDM8 selector and sparse water-fill probes regressed on exact
CUDA, this sweep tested whether the top fixed postdecode modes from the
Modal T4 CUDA-prefix proxy table transfer when there is no selector sidecar
payload and no per-pair selection overhead.

## Source

- CUDA-prefix source sweep:
  `experiments/results/modal_hdm8_postfilter_sweep/hdm8_cuda_full_aggressive_v1_fix1_20260515T023053Z/hdm8_postfilter_sweep.json`
- Base archive:
  `experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/exact_eval_static_release_surface/archive.zip`
- Base archive SHA-256:
  `8a30730e863a2f846d7ca3a707b3191ad64312f5270976dc5f9322ba4228e8c2`
- Base archive bytes: `186395`
- Matched exact-CUDA baseline:
  `experiments/results/modal_auth_eval/hnerv_hdm8_fixed_lengths_modal_t4_retry2_20260514T095000Z/contest_auth_eval.json`
- Baseline score: `0.20636166502462222`
- Baseline avg PoseNet distance: `0.00003236`
- Baseline avg SegNet distance: `0.0006426`

## Exact-CUDA Results

| mode | exact-CUDA score | delta vs HDM8 baseline | avg PoseNet | avg SegNet | bytes | review |
|---|---:|---:|---:|---:|---:|---|
| `even_rgb_bias:-1,0.5,0.5` | `0.2067847322067291` | `+0.0004230671821068843` | `0.00003390` | `0.0006426` | `186395` | `.omx/research/hdm8_fixed_even_rgb_bias_m1_p05_p05_exact_cuda_result_review_20260515_codex.json` |
| `even_grain_chroma:1` | `0.20682813109460968` | `+0.000466466069987459` | `0.00003406` | `0.0006426` | `186395` | `.omx/research/hdm8_fixed_even_grain_chroma_1_exact_cuda_result_review_20260515_codex.json` |
| `even_rgb_bias:-0.5,0.5,0` | `0.20687413086920664` | `+0.0005124658445844199` | `0.00003423` | `0.0006426` | `186395` | `.omx/research/hdm8_fixed_even_rgb_bias_m05_p05_0_exact_cuda_result_review_20260515_codex.json` |
| `even_rgb_bias:0,0.5,-0.5` | `0.20699534652238277` | `+0.000633681497760552` | `0.00003468` | `0.0006426` | `186395` | `.omx/research/hdm8_fixed_even_rgb_bias_0_p05_m05_exact_cuda_result_review_20260515_codex.json` |

All four fixed-mode packets used the base archive bytes and changed only the
runtime postdecode mode. SegNet stayed flat at reported precision; the exact
CUDA regressions are from PoseNet movement, not rate.

## Calibration Gate

The exact-CUDA negative rows were folded into a fail-closed selector calibration
artifact:

- `.omx/research/hdm8_selector_cuda_calibration_table_20260515_codex.json`
- `.omx/research/hdm8_selector_cuda_calibration_table_20260515_codex.md`

Calibration summary:

- exact-CUDA review rows: `6`
- exact-CUDA regressions: `6`
- exact-CUDA positive or neutral rows: `0`
- best delta vs baseline: `+0.00011799785677868435`
- blockers:
  - `exact_cuda_positive_or_neutral_control_missing`
  - `proxy_positive_calibration_rows_transferred_negative`
  - `broad_waterfill_selector_blocked_until_transfer_model`

## Classification

`measured_config_regression`

The current proxy-ranked postdecode film-grain/RGB-bias selector basin is not
promotion-eligible. The engineering stack is real: CUDA-prefix sweep, selector
packing, archive byte charging, runtime consumption, lane claims, and exact
CUDA recovery all exist. The falsified part is the objective: CPU/MPS/CUDA-prefix
local water-fill gains did not transfer to byte-closed exact CUDA.

This does not globally retire film grain, postdecode transforms, or learned
selector ideas. It retires the measured fixed modes and blocks broad promotion
from the current proxy table until one of these reactivation criteria is met:

1. An exact-CUDA positive or neutral fixed-mode control beats the HDM8 baseline
   after charged bytes.
2. A calibrated transfer model explains the current proxy-positive/exact-CUDA
   negative rows and predicts held-out exact-CUDA positives.
3. The postfilter becomes part of a trained substrate where PoseNet-safe effects
   are learned against the scorer instead of selected from a static proxy table.

## Next Action

Stop spending broad-dispatch wall-clock on this local selector objective. Keep
the machinery as a reusable probe/custody surface, but move frontier effort to
byte-closed PacketIR/PR106 deconstruction and higher-upside substrate-class
work (D4/C6/PR95/T1/T10) unless a new exact-CUDA-calibrated transfer model is
landed first.
