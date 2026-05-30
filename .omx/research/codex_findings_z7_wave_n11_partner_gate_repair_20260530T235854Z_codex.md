# Codex Findings: Z7 Wave N+11 Partner Gate Repair

- UTC: 2026-05-30T23:58:54Z
- Commit target: main
- Scope: partner commit `76fa51b87` (`Z7-Mamba-2 Wave N+11 stabilizer canonical adapter wire-in`).

## Finding

The Z7 Wave N+11 stabilizer landing passed its dedicated MLX adapter tests, but
the new test file was not `ruff` clean. The failures were two unused optimizer
locals and two constant `getattr` calls in weight-decay assertions.

## Fix

The repair is test-only:

- warmup and warmup-plus-cosine optimizer construction now assert the optimizer
  object exists before separately checking schedule boundary behavior;
- weight-decay checks use direct attribute access.

No trainer, adapter, harness, artifact, or authority semantics changed.

## Verification

- `.venv/bin/python -m ruff check experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py src/tac/substrates/_shared/mlx_score_aware/adapter.py src/tac/substrates/_shared/mlx_score_aware/harness.py src/tac/substrates/_shared/mlx_score_aware/tests/test_wave_n11_stabilizer.py tools/register_z7_mamba2_wave_n11_stabilizer_anchor.py`
- `.venv/bin/python -m pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_wave_n11_stabilizer.py -q`
- Result: 19 passed.
