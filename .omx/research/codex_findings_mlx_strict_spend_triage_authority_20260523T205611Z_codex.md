# Codex Findings: MLX Strict Spend-Triage Authority

UTC: 2026-05-23T20:56:11Z
Agent: Codex
Lane: codex_mlx_strict_spend_triage_authority_20260523

## Summary

MLX speed/quality observations are useful for local candidate generation and
spend triage, but only after strict calibration against auth-axis evidence.
The previous path could fall back to weaker public/local calibration fields or
silently treat missing source authority as acceptable.

## Change

- MLX quality-speed deltas now require the canonical
  `mlx_score_calibration.v1` policy and strict auth-axis spend-triage allowed
  use before setting a decision band.
- Weak or legacy calibration payloads add blockers and leave
  `decision_band=None`.
- Local CPU advisory cache identities are preserved in MLX response payloads
  but labeled as non-auth-axis for spend triage.
- Empty MobileOne blocks fail at adapter construction instead of producing a
  silent zero tensor.
- Scorer-response dataset normalization now requires complete explicit-false
  source authority fields.
- `tools/build_scorer_response_dataset.py` fails closed on skipped requested
  inputs unless `--allow-skipped` is explicit.
- Strict contest auth-axis payload blockers now require
  `promotion_eligible=false` and `rank_or_kill_eligible=false`, not merely
  "not true."

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_mlx_quality_speed_delta.py src/tac/tests/test_mlx_scorer_adapters.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_auth_eval_schema.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_build_scorer_response_dataset_cli.py -q`
- `.venv/bin/python -m ruff check src/tac/auth_eval_schema.py src/tac/local_acceleration/mlx_quality_speed_delta.py src/tac/local_acceleration/mlx_scorer_adapters.py src/tac/local_acceleration/mlx_scorer_response.py src/tac/optimization/scorer_response_dataset.py tools/build_scorer_response_dataset.py src/tac/tests/test_auth_eval_schema.py src/tac/tests/test_mlx_quality_speed_delta.py src/tac/tests/test_mlx_scorer_adapters.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_build_scorer_response_dataset_cli.py`
