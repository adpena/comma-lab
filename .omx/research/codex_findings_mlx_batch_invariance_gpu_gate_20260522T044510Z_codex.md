# Codex Findings: MLX Batch-Invariance GPU Research Gate

## Summary

Noether's read-only adversarial pass identified that MLX batch-invariance
audits accepted `--device gpu` without the explicit research-signal allowance
already required by neighboring MLX scorer-response and parity tools.

Codex closed that gap:

- `build_mlx_scorer_batch_invariance_manifest(...)` now refuses GPU execution
  unless `allow_gpu_research_signal=True`.
- Synthetic `build_batch_invariance_manifest_from_outputs(...)` manifests fail
  closed for `device_type="gpu"` unless the same allowance is recorded.
- `tools/audit_mlx_scorer_batch_invariance.py` exposes
  `--allow-gpu-research-signal`.
- The manifest records a `device_contract` with the GPU allowance state and the
  canonical GPU research-signal blocker.

## Authority Semantics

This is not a score claim and does not make MLX GPU/batch output promotable.
It only prevents GPU batch-invariance diagnostics from being mistaken for a
normal local scorer-training signal without an explicit research-only marker.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_batch_invariance.py \
  src/tac/tests/test_mlx_production_contract.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_batch_invariance.py \
  tools/audit_mlx_scorer_batch_invariance.py \
  src/tac/tests/test_mlx_batch_invariance.py
```

Results:

- Pytest: `17 passed`
- Ruff: passed

## Next

The remaining MLX drift work is sign-calibration and trace-first: keep
singleton CPU as the production local signal, use GPU/batched MLX only for
explicit research probes, and record first-divergence layer/window traces before
using any larger batch shape in candidate generation.
