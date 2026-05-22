# Codex Findings: MLX Parent Production Contract Plan

Date: 2026-05-22T10:01:01Z

## Authority

- Score claim: `False`
- Promotion eligible: `False`
- Ready for exact-eval dispatch: `False`
- Rank/kill eligible: `False`
- Spend triage authority: `False`

## Change

Landed a fail-closed parent-window production-contract planner for MLX
scorer-response datasets:

- `tac.optimization.scorer_response_dataset.build_mlx_parent_production_contract_plan`
- `tac.optimization.scorer_response_dataset.render_mlx_parent_production_contract_plan_markdown`
- `tools/plan_mlx_parent_production_contracts.py`

The planner groups singleton MLX response rows by the parent identity that a
strict production contract must cover:

- archive SHA-256
- inflated output aggregate SHA-256
- `source_batch_pairs`
- candidate scorer-input cache array SHA-256s
- reference scorer-input cache array SHA-256s
- containing pair window

It does not treat a matching parent response artifact as sufficient. A group is
covered only by a supplied strict MLX production contract or strict contract
bundle. The output remains an inventory and never grants score, promotion,
rank/kill, dispatch, or spend-triage authority by itself.

## Live 600-row dataset result

Command:

```bash
.venv/bin/python tools/plan_mlx_parent_production_contracts.py \
  --dataset experiments/results/mlx_source_identity_refresh_20260522T0945Z/mlx_fec6_decoderq_same_axis_600row_baseline_structural_oof_predicted_dataset_identity_refreshed.json \
  --production-contract experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_production_contract_bundle_v2_dataset_verified_rich_identity.json \
  --json-out experiments/results/mlx_parent_contract_plan_20260522T100028Z/parent_production_contract_plan.json \
  --md-out experiments/results/mlx_parent_contract_plan_20260522T100028Z/parent_production_contract_plan.md \
  --allow-blocked-output
```

Result:

- Status: `blocked`
- MLX rows: `600`
- Required parent contracts: `2`
- Covered parent groups: `0`
- Missing parent groups: `2`
- Strict supplied contracts: `1`
- Blocker count: `2`

The existing supplied strict bundle is valid for its own PR101 pose-axis
identity, but it covers `0/600` rows in this dataset because the archive and
scorer-input cache identities differ. This is the correct fail-closed behavior.

## Missing strict parent contracts

FEC6 group:

- Rows: `300`
- Family: `mlx_scorer_response`
- Required parent window: `[0, 300]`
- Archive SHA-256:
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Inflated aggregate SHA-256:
  `dbc67c898ecb158912f86c920f09bf2c68307b77c1cec3c1baa27a845d3850f1`
- Parent response candidate:
  `experiments/results/mlx_singleton_window_harvest_fec6_20260522T0250Z_full300/candidate_parent_0000_0300.json`

Decoder-q group:

- Rows: `300`
- Family: `mlx_decoder_q`
- Required parent window: `[0, 300]`
- Archive SHA-256:
  `022ac0f391bc9408c357575496c3b680fc5cf9da6ca85d23c3ff994c370a1347`
- Inflated aggregate SHA-256:
  `3d00b08c5969e1f42061c58b8b3e270726b3c35ceed762e67322206be8aa1280`
- Parent response candidate:
  `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/candidate_parent_0000_0300.json`

## Adversarial interpretation

This does not mean MLX is production-ready for this dataset. It means the
remaining blocker is now precise: build two strict parent-window production
contracts against the two listed parent responses, then bundle them with
dataset coverage and re-run the effective MLX spend-triage gate.

The risky shortcut remains forbidden: do not use the old PR101 pose-axis bundle
or local/macOS advisory cache identity as a proxy for the FEC6/decoder-q
dataset. Exact-eval spend triage requires the effective gate to pass after
dataset-specific production-contract coverage.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_scorer_response_dataset.py tools/plan_mlx_parent_production_contracts.py`
- `.venv/bin/python -m pytest src/tac/tests/test_scorer_response_dataset.py -q`

Both passed.
