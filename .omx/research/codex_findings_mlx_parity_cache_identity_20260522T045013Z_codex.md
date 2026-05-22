# Codex Findings: MLX Parity Cache Identity

timestamp_utc: 2026-05-22T04:50:13Z
lane_id: lane_codex_mlx_auth_scorer_hardening_20260522
author: codex
verdict: PROCEED

## Scope

Hardened PyTorch-vs-MLX scorer parity manifests so a passing local parity result is bound to the exact scorer-input cache identity used for the comparison.

## Change

`src/tac/local_acceleration/mlx_scorer_torch_parity.py` now records `cache_identity` in:

- single-window PyTorch-vs-MLX parity manifests
- parity sweep manifests
- SegNet layer-trace manifests
- PyTorch backend batch-invariance manifests

The identity includes:

- cache path
- cache schema/source/hash domain
- archive SHA-256
- inflated output aggregate SHA-256
- raw SHA-256
- source-video SHA-256 when present
- array SHA-256 records
- pair count and tensor shapes

## Rationale

MLX parity is only useful if it is tied to the exact byte/cache surface being compared. A path-only parity manifest can drift if a cache directory is rebuilt. The new identity block makes later calibration, dispatch filtering, and auth-axis bridge audits traceable to concrete cache hashes.

## Verification

Commands:

```bash
.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_scorer_torch_parity.py src/tac/tests/test_mlx_scorer_torch_parity.py
.venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_torch_parity.py -q
```

Observed:

- `ruff`: pass
- scorer parity tests: `12 passed`

## Residual Risk

This remains local implementation evidence only. `cache_identity` improves provenance of a PyTorch-vs-MLX parity pass; it does not make MLX output an auth-eval score or a promotion axis.
