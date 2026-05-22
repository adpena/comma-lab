# Codex Findings: MLX Scorer-Response Out-of-Fold Prediction Gate

generated_at_utc: 2026-05-22T03:33:30Z
lane_id: lane_mlx_oof_response_prediction_gate_20260522_codex
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
rank_or_kill_eligible: false
axis: [macOS-MLX research-signal]

## Verdict

PROCEED as a local LL/surrogate gating tool.

The previous same-axis 600-row MLX dataset was blocked only because it had no prediction field. This pass adds an out-of-fold linear predictor that uses only pre-score row metadata: scorer-pair position, archive byte count, and response-family labels. It does not use observed score, PoseNet distortion, SegNet distortion, scorer deltas, or target fields as features.

## Implementation

Added:

- `src/tac/optimization/scorer_response_prediction.py`
- `tools/fit_scorer_response_oof_predictions.py`

The helper writes `ll_predicted_delta_vs_baseline_score` by holding out each row's declared `holdout_fold`, fitting ridge regression on the remaining folds, and predicting only the held-out rows. Authority fields remain explicit false.

Feature set:

`pair_family_archive_linear_v1`

Features:

- intercept
- normalized pair start
- normalized pair start squared
- one sine/cosine seasonal pair-position term
- normalized archive bytes
- response-family one-hot labels
- response-family by pair-position interactions

## Empirical Result

Input dataset:

`experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_fec6_decoderq_same_axis_600row_dataset.json`

Predicted dataset:

`experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_fec6_decoderq_same_axis_600row_oof_predicted_dataset.json`

Validation gate:

`experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_fec6_decoderq_same_axis_600row_oof_validation_gate.json`

Gate result:

- status: `passed`
- blockers: `[]`
- row_count: `600`
- axis_counts: `{ "[macOS-MLX research-signal]": 600 }`
- family_counts: `{ "mlx_decoder_q": 300, "mlx_scorer_response": 300 }`
- passing_prediction_fields: `["ll_predicted_delta_vs_baseline_score"]`
- overall Pearson r: `0.6038362198259184`
- fold Pearson r: `0.5496121677794259`, `0.6025760779531153`, `0.6586948491831607`, `0.6976638814853103`, `0.5644049600702667`

Planner output:

`experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/ll_next_probe_plan_same_axis_600rows_oof_validated.json`

The prior `do_not_use_response_dataset_for_exact_eval_selection` prohibition is gone after the validation pass. The remaining prohibition is `do_not_widen_coordinate_sparse_residual_sidecar`, because observed scorer gains still do not pay for residual payload bytes.

## Authority Boundary

This is still local MLX research signal. It is eligible as LL/surrogate planner input under the attached strict-pass MLX/PyTorch parity sweep, but it is not a contest score, not promotion evidence, not rank/kill evidence, and not exact-eval dispatch authority by itself.

## Next Action

Use this validated dataset to drive the next cheap local probe: add response features with actual decoder-q mutation metadata and master-gradient/frame-sensitivity priors, then compare whether that richer predictor beats the metadata-only OOF baseline before any paid exact-eval spend.
