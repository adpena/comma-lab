# Codex Findings: MLX Auth-Axis Fidelity Gate Hardening

## Summary

Codex tightened the MLX scorer-training-signal fidelity gate so a local MLX
payload can pass transfer calibration only when the comparison target is a
contest auth-eval axis payload:

- accepted auth evidence pairs are `contest-CPU` / `contest_cpu` and
  `contest-CUDA` / `contest_cuda`
- diagnostic/advisory/proxy auth-side payloads now fail closed even if their
  numeric components match
- the output manifest records the auth-eval contract used for the decision
- MLX-side outputs remain non-promotable and candidate-generation-only

This closes a false-authority path where a local advisory payload could be
used as the right-hand "auth" comparison input for
`mlx_scorer_training_signal_fidelity.v1`.

## Files Changed

- `src/tac/local_acceleration/mlx_scorer_fidelity.py`
- `src/tac/tests/test_mlx_scorer_fidelity.py`

## Verification

Commands:

```bash
.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_scorer_fidelity.py src/tac/tests/test_mlx_scorer_fidelity.py
.venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_fidelity.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_to_pytorch_export.py -q
```

Results:

- Ruff: passed
- Pytest: `32 passed`

## Review Notes

Earlier review findings for dtype-preserving MLX-to-PyTorch export, hash-only
cache bounds, large eager-cache acknowledgement, and scorer-input tensor hash
identity were stale on current `main`; current tests already cover those
surfaces. The live issue fixed here was the missing auth-side contest-axis
requirement in the fidelity comparison gate.

## Authority Semantics

This memo is not a score claim and does not promote any MLX result. The
changed gate only decides whether a local MLX signal is eligible to guide
candidate generation or dispatch filtering after comparison against a real
contest auth-eval axis payload. Candidate archives still require paired
contest auth eval before any promotion, ranking, or submission claim.
