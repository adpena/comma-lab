# Codex Findings: MLX score-aware harness refactor

UTC: 2026-05-27T17:17:46Z

## Context

The MLX score-aware harness existed as a large monolithic helper in `mlx_score_aware_full_main.py`. It was useful, but the surface was hard to review and hard to reuse across MLX-first substrates without carrying an oversized import path.

## Landing

- Split the harness into a typed package:
  - `device_gate`: fail-closed MLX availability and shared harness error type.
  - `targets`: real-video target decode into MLX NHWC buffers.
  - `bundle`: substrate-specific renderer/target contract.
  - `loss`: reconstruction plus optional Hinton KL T=2 score-aware surrogate and variant-specific extra terms.
  - `adapter`: Style-B MLX adapter for canonical long-training.
  - `portability`: numpy-portable inflate verifier.
  - `harness`: canonical MLX `_full_main` orchestrator.
- Kept `mlx_score_aware_full_main.py` as a backward-compatible facade so existing Dreamer/Z8 users and tests retain their import path.
- Added package-level tests for device gating, bundle validation, portability, both renderer conventions, score-aware loss composition, and facade compatibility.

## Verification

- `.venv/bin/python -m ruff check src/tac/substrates/_shared/mlx_score_aware src/tac/substrates/_shared/mlx_score_aware_full_main.py src/tac/substrates/_shared/tests/test_mlx_score_aware_full_main.py`
- `.venv/bin/python -m py_compile src/tac/substrates/_shared/mlx_score_aware_full_main.py src/tac/substrates/_shared/tests/test_mlx_score_aware_full_main.py $(find src/tac/substrates/_shared/mlx_score_aware -name '*.py' -not -path '*/__pycache__/*')`
- `.venv/bin/python -m pytest src/tac/substrates/_shared/mlx_score_aware/tests src/tac/substrates/_shared/tests/test_mlx_score_aware_full_main.py -q`
  - 59 passed, 1 skipped on this MLX-local host.

## Authority

This is local MLX training infrastructure. Outputs remain `[macOS-MLX research-signal]`, not contest score claims. Promotion still requires numpy-portable archive/runtime custody and exact contest CPU/CUDA auth evaluation.
