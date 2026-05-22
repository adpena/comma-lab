# Codex Findings: MLX Same-Axis Decoder-Q Response Family

generated_at_utc: 2026-05-22T03:25:20Z
lane_id: lane_mlx_same_axis_decoderq_response_family_20260522_codex
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
rank_or_kill_eligible: false
axis: [macOS-MLX research-signal]

## Verdict

PROCEED as a local MLX response-dataset hardening artifact.

This pass closes the prior single-family blocker for the local MLX scorer-response dataset without creating score authority. The merged dataset now has 600 rows, two candidate families, one score axis, all required holdout folds, and zero skipped rows. The remaining validation blocker is honest and correct: `no_prediction_fields_present`.

## Code Change

Added an explicit `response_family` field to MLX scorer-response payloads and taught the scorer-response dataset normalizer to use it as the row family when present.

This separates two concepts that were previously conflated:

- `score_axis`: transport/evidence axis, here `[macOS-MLX research-signal]`.
- `response_family`: candidate family, here `mlx_decoder_q`.

The tag is validated as a conservative slug: lowercase letters, digits, underscore, dash, or dot only.

## Empirical Artifact

Root:

`experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/`

Decoder-q candidate:

- candidate id: `d1f1e56e042692f2`
- archive bytes: `178517`
- archive SHA-256: `022ac0f391bc9408c357575496c3b680fc5cf9da6ca85d23c3ff994c370a1347`
- inflated raw SHA-256: `5cae18b3430b733309306d658568c038d1a30c63fbb56e735124068ddf7520d2`
- inflated output aggregate SHA-256: `3d00b08c5969e1f42061c58b8b3e270726b3c35ceed762e67322206be8aa1280`
- official `inflate.sh` raw visibility: `visible_change_count=1`, `no_visible_change_count=0`
- changed frames: `600`
- changed raw bytes: `33398182`

Local MLX singleton CPU response over scorer pairs `[0,300]`:

- canonical_score: `0.19267559018106015`
- avg_posenet_dist: `0.00003968536578364971`
- avg_segnet_dist: `0.0005388726129118974`
- elapsed_seconds: `249.21563482284546`
- throughput: approximately `1.2038` pairs/sec at completion
- response_family: `mlx_decoder_q`

For comparison, the prior FEC6 singleton MLX response over the same window recorded:

- canonical_score: `0.1922605584791145`
- avg_posenet_dist: `0.000039679991221153914`
- avg_segnet_dist: `0.0005347357859136537`

This decoder-q candidate is therefore worse on the local MLX research axis by about `+0.0004150317`. This is not a contest score and must not be used for ranking, killing, promotion, or exact-eval dispatch selection without the separate authority gates.

## Dataset Gate

Merged dataset:

`experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_fec6_decoderq_same_axis_600row_dataset.json`

Validation gate:

`experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_fec6_decoderq_same_axis_600row_validation_gate.json`

Coverage:

- row_count: `600`
- axis_counts: `{ "[macOS-MLX research-signal]": 600 }`
- family_counts: `{ "mlx_decoder_q": 300, "mlx_scorer_response": 300 }`
- families_with_required_folds: `["mlx_decoder_q", "mlx_scorer_response"]`
- missing_global_folds: `[]`
- skipped_count: `0`

Gate status:

- status: `blocked`
- blockers: `["no_prediction_fields_present"]`

This is the desired fail-closed state. We now have the same-axis and family-diverse response observations, but no held-out predictor has earned permission to consume them for LL training selection.

LL planner artifact:

`experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/ll_next_probe_plan_same_axis_600rows.json`

The planner was run with the strict-pass CPU singleton MLX/PyTorch parity sweep:

`experiments/results/mlx_torch_parity_sweep_clean_head_52093e425_cpu_fec6_pr101_singleton_full300pairs_20260522T021600Z.json`

Planner prohibitions include `do_not_use_response_dataset_for_exact_eval_selection` with blocker `no_prediction_fields_present`, which is the correct authority boundary.

## Next Action

Build the first non-leaking prediction-field attach pass for this dataset. Recommended source features:

- pre-score pixel/raw deltas against the FEC6 baseline by scorer pair;
- master-gradient/frame-sensitivity priors already available in canonical anchors;
- decoder-q mutation metadata such as tensor name, q offset, sign, and targeted byte-span sensitivity.

Then rerun `tools/validate_scorer_response_dataset.py` and require held-out correlation before the LL planner uses the rows for training or candidate routing.
