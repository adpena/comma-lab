# Codex Findings: MLX Structural OOF Feature Gate

## Scope

Test whether contest-structured, non-authoritative row features improve the
same-axis 600-row MLX scorer-response held-out predictor for FEC6 vs decoder-q
candidate `d1f1e56e042692f2`.

All artifacts remain `[macOS-MLX research-signal]` and carry `score_claim=false`,
`promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`,
`rank_or_kill_eligible=false`, and `promotable=false`.

## Inputs

- Baseline same-axis OOF gate:
  `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_fec6_decoderq_same_axis_600row_oof_validation_gate.json`
- Diagnostic frame-axis prior:
  `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/master_gradient_frame_axis_l1_20260520_codex.npy`
- Frame-decomposition metadata:
  `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/master_gradient_frame_decomposition_20260520_codex.json`
- Decoder-q mutation manifest:
  `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/op3v3_decoder_q_micro_candidates_20260520_codex/d1f1e56e042692f2/mutation_manifest.json`

## Landed reusable surfaces

- `src/tac/optimization/scorer_response_structural_features.py`
  attaches diagnostic frame-cycle and decoder-q mutation metadata while
  preserving false-authority fields.
- `tools/attach_scorer_response_structural_features.py`
  is the operator CLI for applying those features.
- `src/tac/optimization/scorer_response_dataset.py`
  now records same-window baseline difficulty fields when building MLX window
  response datasets.
- `src/tac/optimization/scorer_response_prediction.py`
  consumes only an explicit allowlist of extra numeric features. Candidate
  outcome fields remain excluded.

## OOF comparison

| Feature set | Gate | Overall Pearson r | Verdict |
|---|---:|---:|---|
| metadata-only baseline | passed | 0.6038362198259184 | current best |
| diagnostic 16-frame cycle + decoder-q mutation metadata | passed | 0.6016238335302896 | falsified as improvement |
| same-window baseline difficulty | passed | 0.6038362198259173 | neutral |
| baseline difficulty + diagnostic cycle + decoder-q metadata | passed | 0.6016238335302895 | falsified as improvement |

## Finding

The 16-frame master-gradient frame decomposition is useful as a structural
diagnostic, but treating it as an 8-pair cyclic prior does not improve this
600-row MLX response predictor. Same-window baseline difficulty is valid to
record and reuse, but it is neutral here.

The canonical LL planner input should therefore stay on the metadata-only OOF
dataset for this artifact family unless a future full-window structural prior
beats the baseline gate out-of-fold.

## Next action

Build the next structural feature class from full 300-window evidence rather
than the 16-frame diagnostic cycle: per-window MLX/Torch parity-stable
PoseNet/SegNet response summaries, full-window saliency, or exact per-window
master-gradient projections. Treat any cyclic diagnostic feature as advisory
only until it improves held-out correlation.
