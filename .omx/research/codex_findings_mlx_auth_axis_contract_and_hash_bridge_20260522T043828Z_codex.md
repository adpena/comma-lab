# Codex Findings: MLX Auth-Axis Contract + Hash-Bridge Hardening

## Summary

Follow-up adversarial review found that the first MLX fidelity hardening pass
closed basic grade/axis label checks but still needed a stronger auth-eval
custody contract. Codex landed a shared strict helper in `tac.auth_eval_schema`
and wired it into both MLX transfer gates:

- `build_mlx_scorer_training_signal_fidelity_manifest`
- `audit_mlx_scorer_input_cache_against_auth_eval`

The gates now reject diagnostic/advisory/forged auth-side payloads even when
their numeric metrics and hashes match. They require strict `contest-CPU` or
`contest-CUDA` evidence semantics, score-claim validity, expected sample count,
device/provenance checks, and no diagnostic blockers.

## Additional Fixes

- Narrowed MLX fidelity inflated-output aggregate hash extraction to explicit
  top-level fields or `provenance.inflated_output_manifest`; unrelated nested
  `aggregate_sha256` values no longer satisfy byte closure.
- Required the strict auth-axis side of MLX fidelity/cache-audit checks to stay
  full-contest `600` samples even when a local debug cache passes a smaller
  `expected_pair_count` / threshold override.
- Added early fail-closed validation for
  `--scorer-input-cache-hash-batch-pairs >= 1` in direct auth eval and both
  Modal CUDA/CPU wrappers, before provider dispatch/claim construction.
- Kept MLX/cache outputs non-promotable; passing these gates only authorizes
  local transfer calibration or candidate-generation filtering.

## Verification

Commands:

```bash
.venv/bin/python -m ruff check src/tac/auth_eval_schema.py src/tac/local_acceleration/mlx_scorer_fidelity.py src/tac/local_acceleration/mlx_cache_audit.py src/tac/tests/test_auth_eval_schema.py src/tac/tests/test_mlx_scorer_fidelity.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_modal_auth_eval.py src/tac/tests/test_contest_auth_eval.py experiments/contest_auth_eval.py experiments/modal_auth_eval.py experiments/modal_auth_eval_cpu.py
.venv/bin/python -m pytest src/tac/tests/test_auth_eval_schema.py src/tac/tests/test_mlx_scorer_fidelity.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_to_pytorch_export.py -q
.venv/bin/python -m pytest src/tac/tests/test_modal_auth_eval.py -q
.venv/bin/python -m pytest src/tac/tests/test_contest_auth_eval.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_scorer_response_dataset.py $(rg --files src/tac/tests | rg 'mlx') src/tac/tests/test_auth_eval_schema.py -q
```

Results:

- Ruff: passed
- Auth/MLX focused pytest: `57 passed`
- Modal wrapper pytest: `42 passed`
- Direct contest auth-eval pytest: `49 passed`
- Broad scorer-response + MLX + auth-schema sweep: `237 passed`

## Authority Semantics

This is not a score claim. It hardens the contract for deciding whether local
MLX artifacts can be used for training-signal transfer calibration against a
real auth-eval axis. Any candidate selected with this local signal still needs
paired contest auth eval before ranking, promotion, or submission use.
