# Codex Findings: MLX Production Contract Cache + Parity Gate

timestamp_utc: 2026-05-22T05:05:48Z
lane_id: lane_codex_mlx_auth_scorer_hardening_20260522
author: codex
verdict: PROCEED

## Scope

Raised the production-readiness bar for local MLX scorer-response artifacts. A local MLX scorer response can no longer pass the production contract from profile stability and batch invariance alone.

## Change

`src/tac/local_acceleration/mlx_production_contract.py` now requires, by default:

- a passing `mlx_scorer_input_cache_auth_eval_audit.v1` manifest
- a passing PyTorch-vs-MLX scorer parity manifest, either single-window or sweep
- the existing profile-stability gate
- the existing batch-invariance gate

The cache/auth audit must match the response archive and inflated-output identity. The PyTorch-vs-MLX parity manifest must be non-authoritative local-MLX evidence and, when it carries cache identity, its archive identity must match the response/reference/candidate cache surface.

The contract also supports an explicit score-calibration gate via `require_score_calibration=True`; this requires strict auth-axis calibration with no uncertain pairwise spend-triage decisions.

`tools/check_mlx_scorer_production_contract.py` now exposes:

- `--cache-auth-audit`
- `--torch-parity`
- `--score-calibration`
- `--no-require-cache-auth-audit`
- `--no-require-torch-parity`
- `--require-score-calibration`

## Verification

Commands:

```bash
.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_production_contract.py src/tac/tests/test_mlx_production_contract.py tools/check_mlx_scorer_production_contract.py
.venv/bin/python -m pytest src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_scorer_torch_parity.py src/tac/tests/test_mlx_score_calibration.py src/tac/tests/test_mlx_scorer_response.py -q
.venv/bin/python -m pytest $(rg --files src/tac/tests | rg '/test_mlx.*\.py$' | tr '\n' ' ') -q
```

Observed:

- `ruff`: pass
- focused production/cache/parity/calibration/response tests: `61 passed`
- full MLX test sweep: `176 passed`

## Residual Risk

This contract still certifies only local MLX acceleration signal. It is a stronger gate for local training, profiling, and candidate-generation velocity, but contest CPU/CUDA auth eval remains required for score claims, promotion, and rank/kill decisions.
