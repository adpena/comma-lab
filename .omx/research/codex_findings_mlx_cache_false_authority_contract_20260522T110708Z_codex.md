# Codex Findings: MLX Cache False-Authority Contract Tightening

UTC: 2026-05-22T11:07:08Z

## Verdict

PROCEED. A residual false-authority gap in MLX scorer-input cache custody was
closed at the cache/audit boundary.

## Finding

The MLX scorer-input cache audit rejected `score_claim=true` and
`promotion_eligible=true`, but did not require the complete false-authority
tuple to be explicitly false on cache manifests. Some cache writers also omitted
`score_claim_valid=false`.

That is too weak for production MLX transfer calibration: a stale, hand-written,
or schema-drifted cache manifest could be accepted as local training/calibration
evidence without carrying the same false-authority contract enforced on scorer
responses and production contracts.

## Patch

- `src/tac/local_acceleration/mlx_preprocess.py`: all full and hash-only cache
  manifest writers now emit `score_claim_valid=false`.
- `src/tac/local_acceleration/mlx_cache_audit.py`: cache audits now require the
  full false-authority tuple to be exactly false:
  `score_claim`, `score_claim_valid`, `promotion_eligible`, `promotable`,
  `rank_or_kill_eligible`, and `ready_for_exact_eval_dispatch`.
- `src/tac/tests/test_mlx_cache_audit.py`: added a regression test proving that
  missing `score_claim_valid` and true `rank_or_kill_eligible` block transfer
  calibration.

## Verification

- `ruff check src/tac/local_acceleration/mlx_cache_audit.py src/tac/local_acceleration/mlx_preprocess.py src/tac/tests/test_mlx_cache_audit.py`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_production_contract.py -q`

Result: 77 passed.

## Residual MLX Contract Work

Current `main` already has the adjacent high-risk guardrails from the concurrent
review pass: cache tensor integrity is recomputed before scoring, strict
production contracts require reference and candidate torch parity, score
calibration is response-bound, profile stability is value-bound, and failed
contract bundle verdicts fail closed.

Remaining frontier-critical work is to complete the decoder-q parent production
contract or record its exact non-promotional blocker, then use only fully
covered MLX response rows for local spend triage and training acceleration.
