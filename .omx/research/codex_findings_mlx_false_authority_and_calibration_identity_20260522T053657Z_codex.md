# Codex Findings: MLX False-Authority + Calibration Identity Hardening

utc: 2026-05-22T05:36:57Z
lane: lane_mlx_false_authority_and_calibration_identity_20260522
status: LANDED
score_claim: false
promotable: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Finding

An xhigh adversarial review found two production-readiness gaps in the MLX
local-acceleration stack:

1. MLX-derived rows enforced and emitted `score_claim=false`,
   `promotion_eligible=false`, `rank_or_kill_eligible=false`, and
   `ready_for_exact_eval_dispatch=false`, but not the CLAUDE.md-required
   `promotable=false` field.
2. `mlx_score_calibration` validated strict auth-axis payloads by score axis,
   sample count, and archive byte count, but did not bind the calibration row to
   the MLX response archive SHA-256 and inflated-output aggregate SHA-256. A
   same-size wrong archive could therefore be accepted as calibration evidence.

## Landing

Patched the MLX scorer-response, cache-audit, production-contract,
score-calibration, materialization-plan, profile, batch-invariance, parity,
trace, window, and execution-plan surfaces so generated MLX artifacts carry
`promotable=false` and strict checkers require it where they already require
the other false-authority fields.

`mlx_score_calibration` now also requires:

- MLX response `evidence_grade`, `evidence_tag`, and `score_axis` to match the
  local MLX research-signal contract exactly;
- `candidate_generation_only=true`;
- `requires_exact_eval_before_promotion=true`;
- auth-axis payload archive SHA-256 identity with the MLX response;
- auth-axis payload inflated-output aggregate SHA-256 identity with the MLX
  response.

The live FEC6 materialization plan was regenerated and still fails closed:

- `passed=false`
- `promotable=false`
- `verdict=AUTH_CACHE_MATERIALIZATION_REQUIRED`
- `next_materialization_action=materialize_auth_axis_tensor_cache_from_modal_linux_raw_or_export_linux_tensor_cache`

## Verification

- `.venv/bin/ruff check ...`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_score_calibration.py src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_mlx_auth_cache_materialization.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_scorer_torch_parity.py src/tac/tests/test_mlx_profile_stability.py src/tac/tests/test_mlx_batch_invariance.py src/tac/tests/test_mlx_response_windows.py src/tac/tests/test_mlx_execution_plan.py src/tac/tests/test_mlx_scorer_fidelity.py src/tac/tests/test_mlx_segnet_trace_compare.py -q`

Result: 121 passed.

## Next Action

The MLX stack is now stricter, but the production-enabling blocker is unchanged:
materialize auth-side scorer-input tensors from the Modal/Linux raw surface, or
export those tensors from Linux directly, then require
`PASS_CACHE_AUTH_EVAL_IDENTITY` before using local MLX outputs as training
targets.
