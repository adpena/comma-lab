# Codex Findings: MLX Response Promotable Stdout Guard

Date: 2026-05-22T10:05:58Z

## Authority

- Score claim: `False`
- Promotion eligible: `False`
- Ready for exact-eval dispatch: `False`
- Rank/kill eligible: `False`
- Spend triage authority: `False`

## Finding

The two parent response artifacts currently needed for the 600-row FEC6 +
decoder-q dataset are stale with respect to the current MLX response writer:
they lack top-level `promotable: false`, causing strict parent-contract probes
to block on `response_promotable_not_false`.

The live writer in `tac.local_acceleration.mlx_scorer_response` already emits
`promotable: false`, so this is not a current writer-code bug. It is stale
artifact state. Regenerating the parent responses with the current writer will
remove that specific blocker, while leaving the substantive strict-contract
blockers in place until auth-axis cache audit, parity, profile stability, and
score calibration are supplied.

## Guard

Updated `tools/run_mlx_scorer_response_cache.py` so normal CLI stdout includes
`promotable: false` next to `score_claim: false`. This makes the false-authority
state visible in ordinary operator flows and locks the field with a regression
test.

## Verification

- `.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_scorer_response.py src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_scorer_response_dataset.py tools/run_mlx_scorer_response_cache.py tools/plan_mlx_parent_production_contracts.py`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_mlx_production_contract.py -q`

Both passed. The focused test set reported `126 passed`.
