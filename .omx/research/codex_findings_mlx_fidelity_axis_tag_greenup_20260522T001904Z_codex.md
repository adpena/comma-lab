# Codex Findings: MLX Fidelity Axis-Tag Greenup

utc: 2026-05-22T00:19:04Z
lane_id: lane_mlx_auth_scorer_training_signal_fidelity_20260521
evidence_axis: [macOS-MLX research-signal]
verdict: PROCEED_WITH_STRICT_FALSE_AUTHORITY_GUARD

## What changed

- Hardened `tac.local_acceleration.mlx_scorer_fidelity` so MLX transfer-fidelity manifests require the MLX payload to carry both:
  - `evidence_tag == "[macOS-MLX research-signal]"`
  - `score_axis == "[macOS-MLX research-signal]"`
- Added a regression test proving a payload labeled as `[contest-CUDA]` is rejected even if it carries the correct MLX evidence grade.

## Why

The MLX scorer-response path is intended to accelerate local training and candidate generation, not to become an auth-eval scorer. The existing gate already required `evidence_grade="macOS-MLX-research-signal"`, but did not require the visible axis labels to match. That left a false-authority path where a malformed payload could pass calibration while wearing contest-axis labels.

This patch closes that path without changing MLX numeric behavior or loosening any promotion gate.

## Verification

```bash
.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_scorer_fidelity.py src/tac/tests/test_mlx_scorer_fidelity.py
.venv/bin/python -m pytest -q src/tac/tests/test_mlx_scorer_fidelity.py
.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_mlx_execution_plan.py src/tac/tests/test_mlx_profile_stability.py src/tac/tests/test_profile_mlx_scorer_response_cache.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_batch_invariance.py src/tac/tests/test_mlx_scorer_fidelity.py
```

Results:

- Ruff: passed.
- MLX fidelity tests: 8 passed.
- Broader MLX/scorer-response suite: 66 passed.

## Remaining work

Next high-EV hardening step is a cache-window PyTorch-vs-MLX full DistortionNet parity audit that writes per-window PoseNet pose-output deltas, SegNet logit deltas, and component-distance deltas before any local MLX-trained candidate can be promoted to paid exact-eval priority.
