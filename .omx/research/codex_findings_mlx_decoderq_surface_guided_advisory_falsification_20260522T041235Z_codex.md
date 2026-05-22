# Codex Findings: MLX Decoder-Q Surface-Guided Advisory Falsification

Generated: 2026-05-22T04:12:35Z

## Verdict

DO_NOT_DISPATCH_EXACT_EVAL for the three surface-guided decoder-q candidates.

The response-surface-guided waterbucket objective produced valid fixed-length archives and official-inflate-visible raw changes, but local macOS CPU advisory evaluation shows all three candidates regress against the baseline. This falsifies promotion of the current suppress-first surface proxy as a standalone exact-eval selector.

## Artifact

- Advisory summary: `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/decoder_q_surface_guided_advisory_batch/summary.json`
- Input candidate root: `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/decoder_q_surface_guided_waterbucket_candidates`
- Axis: `[macOS-CPU advisory decoder-q]`
- Device: `cpu`
- Batch size: 16
- `num_threads`: 2
- Baseline raw SHA-256: `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`
- Baseline score used for advisory deltas: 0.19206142414659494

Authority remains false:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## Results

| candidate_id | edits | surface proxy priority | advisory score | delta vs baseline | avg SegNet dist | avg PoseNet dist | changed bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `a2f90a216aac4184` | 1 | 0.47923087385670265 | 0.19249142414659495 | +0.0004300000000000137 | 0.00056469 | 0.00002943 | 35132514 |
| `a9b04920db67ec71` | 2 | 0.9584617477134053 | 0.19254442414659495 | +0.0004830000000000112 | 0.00056522 | 0.00002943 | 46453851 |
| `8f3a33e49b9b7906` | 4 | 1.9169234954268106 | 0.19259533847162177 | +0.0005339143250268352 | 0.00056570 | 0.00002944 | 62258822 |

All candidates:

- Passed official inflate.
- Changed all 600 frames.
- Preserved archive byte count at 178,517 bytes.
- Regressed advisory score.
- Remain blocked by `raw_eval_advisory_not_full_archive_inflate_custody`, `not_contest_cuda`, and `exact_cuda_auth_eval_missing`.

## Interpretation

The failure mode is useful: response-surface proxy priority increases with edit budget, while advisory score also worsens. The current objective successfully finds high-leverage decoder-q atoms, but the sign/direction proxy is not sufficient to turn those atoms into score-lowering edits.

This should be treated as a selector falsification, not as evidence against the broader MLX response-surface program. The valid next move is to use this negative batch as labeled training data for the planner:

- Downweight standalone `suppress_or_invert_regressions_first` candidate selection.
- Require advisory-measured sign calibration before exact-eval dispatch.
- Prefer per-window or per-axis signed response targets that can distinguish "large visible change" from "score-lowering change".
- Keep the three candidates out of exact CUDA dispatch unless a later calibrated planner can explain why this advisory negative is a false negative.
