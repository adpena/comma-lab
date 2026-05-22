# Codex Findings: MLX Production Contract V2 Strict Gate

## Verdict

`NEEDS_REBUILD_BEFORE_LOCAL_MLX_PRODUCTION_USE`.

The MLX scorer-response path remains useful as local research signal, but the current FEC6 cache packet is not yet eligible for production local training/selection against the auth-eval scorer surface.

## What changed

- Bumped the production contract to `mlx_scorer_production_contract.v2` with `mlx_scorer_production_gate_set.v2.cache_auth_torch_profile`.
- Added top-level false-authority fields to production-contract manifests: `score_claim=false`, `score_claim_valid=false`, `candidate_generation_only=true`, `requires_exact_eval_before_promotion=true`, and `score_axis=[macOS-MLX research-signal]`.
- Split advisory bypasses from production PASS: disabling required gates now yields `ADVISORY_MLX_SCORER_DEV_CONTRACT`, not `PASS_MLX_SCORER_PRODUCTION_CONTRACT`.
- Tightened cache/auth audit validation to require canonical-equation transfer eligibility, zero identity residual, auth/cache hash-domain custody, array hashes, shapes, pair count, and explicit false-authority fields.
- Tightened PyTorch-vs-MLX parity validation to bind cache path, pair count, raw/inflated hashes, covered pair window, array hashes, shapes, and hash domain.
- Shared the strict MLX score-calibration `allowed_use` string with the scorer-response dataset gate to remove split semantics.

## Empirical packet attempt

Generated a strict singleton-window packet under:

`experiments/results/mlx_production_contract_fec6_pr101_singleton_window_20260522T051339Z/`

The final contract artifact:

`production_contract_failclosed_v5.json`

fails closed with blockers:

- `cache_auth_audit_blocker:inflated_outputs_aggregate_sha256_mismatch_or_missing`
- `cache_auth_audit_blocker:raw_sha256_mismatch_or_missing`
- `cache_auth_audit_blocker:scorer_input_array_sha256_mismatch:segnet_last_rgb`
- `cache_auth_audit_blocker:scorer_input_array_sha256_mismatch:posenet_yuv6_pair`
- cache audit canonical equation not transfer-eligible / identity residual nonzero
- `torch_parity_cache_identity_hash_domain_missing`
- stale score calibration missing `decision_policy`

## Interpretation

The exact missing artifact is a regenerated FEC6 scorer-input cache and strict auth-cache audit where archive SHA, inflated-output aggregate SHA, raw SHA, scorer-input hashes, hash domain, shapes, and pair count all match the Modal/Linux contest-CPU auth surface.

Until that exists, MLX output must remain non-promotional research signal only.

## Verification

- `ruff check` on changed MLX contract/calibration/cache-audit/scorer-response files: PASS
- focused contract/calibration/cache/dataset tests: `105 passed`
- broader MLX/scorer-response slice: `241 passed`
