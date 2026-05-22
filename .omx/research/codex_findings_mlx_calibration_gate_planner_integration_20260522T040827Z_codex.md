# Codex Findings: MLX Calibration Gate Planner Integration

UTC: 2026-05-22T04:08:27Z

## Verdict

PROCEED_WITH_OPERATIONAL_GUARD. The MLX score-calibration decision band is now wired into the LL scorer-response planner. MLX rows can still train local response models under the existing parity gate, but MLX-based exact-eval spend filtering is explicitly prohibited unless a score-calibration manifest is attached and all pairwise calibration decisions are certified.

This remains non-authoritative local signal.

## Code Changes

- Added `build_mlx_score_calibration_gate(...)` to `tac.optimization.scorer_response_dataset`.
- Added optional `mlx_score_calibration` input to `build_next_probe_plan(...)`.
- Added `--mlx-score-calibration` to `tools/plan_ll_scorer_response_next.py`.
- Added planner prohibitions:
  - `do_not_use_mlx_rows_for_exact_eval_spend_triage_without_score_calibration`
  - `do_not_use_mlx_rows_for_exact_eval_spend_triage_after_uncertain_calibration`
- Extended Markdown rendering with an `MLX Score Calibration Gate` section.
- Added tests for missing calibration, passing calibration, and uncertain calibration.

Authority remains explicit false:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## Real Planner Re-run

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tools/plan_ll_scorer_response_next.py \
  --dataset experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_fec6_decoderq_same_axis_600row_structural_oof_predicted_dataset.json \
  --mlx-torch-parity-sweep experiments/results/mlx_torch_parity_sweep_clean_head_52093e425_cpu_fec6_pr101_singleton_full300pairs_20260522T021600Z.json \
  --mlx-score-calibration experiments/results/mlx_public_frontier_calibration_20260522T023053Z/calibration_manifest_decision_band.json \
  --decoder-q-response-surface experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_decoderq_response_surface_plan.json \
  --json-out experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/ll_next_probe_plan_same_axis_600rows_oof_validated_decoderq_surface_calibrated.json \
  --md-out experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/ll_next_probe_plan_same_axis_600rows_oof_validated_decoderq_surface_calibrated.md
```

Result summary:

- Response validation gate: `passed`
- MLX/Torch parity gate: `strict_pass`
- MLX score-calibration gate: `strict_pass`
- Certified pairwise calibration decisions: `6`
- Uncertain pairwise calibration decisions: `0`
- Recommended minimum MLX gap for spend triage: `8.801772121230789e-05`
- Remaining prohibition: `do_not_widen_coordinate_sparse_residual_sidecar`
- First probe: `ll_decoder_q_window_signed_response_surface`
- Second probe: `ll_mlx_cpu_stable_response_harvest`

Generated real-plan artifacts are ignored under `.gitignore:55:experiments/results/*`.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/optimization/scorer_response_dataset.py \
  tools/plan_ll_scorer_response_next.py \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/local_acceleration/mlx_score_calibration.py \
  tools/calibrate_mlx_scorer_response_scores.py \
  src/tac/tests/test_mlx_score_calibration.py
```

Result: `All checks passed!`

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_scorer_response_dataset.py
```

Result: `52 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_mlx_score_calibration.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_production_contract.py
```

Result: `95 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
  $(rg --files src/tac/tests | rg 'mlx')
```

Result: `202 passed`.

## Next Action

Use the calibrated plan to choose local decoder-q follow-up probes, but keep exact-eval spend gated by the lane-claim lifecycle and exact CUDA auth eval. For close MLX deltas below the calibration threshold, the planner must classify the decision as uncertain instead of spending based on MLX order.
