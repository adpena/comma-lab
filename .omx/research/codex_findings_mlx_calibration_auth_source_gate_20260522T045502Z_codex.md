# Codex Findings: MLX Calibration Auth-Source Gate

timestamp_utc: 2026-05-22T04:55:02Z
lane_id: lane_codex_mlx_auth_scorer_hardening_20260522
author: codex
verdict: PROCEED_WITH_GUARDS

## Scope

Closed two MLX false-authority paths found by adversarial review:

1. MLX score calibration accepted direct `cpu_score` / `cuda_score` scalars or loose score JSON.
2. MLX cache audit could use an under-custodied independent reference hash manifest.

## Fixes

### F-1: Calibration now requires strict contest auth-axis payloads

`src/tac/local_acceleration/mlx_score_calibration.py` now rejects direct `cpu_score` / `cuda_score` scalar anchors. Calibration rows must point to `cpu_auth_eval_path` / `cuda_auth_eval_path` JSON payloads that pass `required_contest_auth_axis_payload_blockers(...)`.

Additional axis checks require the row axis to match the payload axis:

- `cpu_auth_eval_path` must be `contest-CPU` / `contest_cpu`
- `cuda_auth_eval_path` must be `contest-CUDA` / `contest_cuda`

The spend-triage uncertainty basis now uses strict auth-axis calibration error only. Local CPU deltas may still be reported as diagnostics, but they no longer unlock the recommended MLX gap for spend triage by themselves.

### F-2: Reference cache manifests now require custody fields

`src/tac/local_acceleration/mlx_cache_audit.py` now fails closed when a standalone `reference_cache_manifest` is used as the scorer-input hash source without full custody:

- archive SHA-256
- inflated output aggregate SHA-256
- raw SHA-256
- hash domain
- pair count
- scorer-input tensor shapes
- `segnet_last_rgb`, `posenet_yuv6_pair`, and `pair_indices` array hashes

## Verification

Commands:

```bash
.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_score_calibration.py src/tac/tests/test_mlx_score_calibration.py src/tac/local_acceleration/mlx_cache_audit.py src/tac/tests/test_mlx_cache_audit.py
.venv/bin/python -m pytest src/tac/tests/test_mlx_score_calibration.py src/tac/tests/test_mlx_cache_audit.py -q
.venv/bin/python -m pytest src/tac/tests/test_mlx_score_calibration.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_scorer_torch_parity.py src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_scorer_fidelity.py src/tac/tests/test_auth_eval_schema.py -q
```

Observed:

- `ruff`: pass
- calibration/cache-audit focused tests: `21 passed`
- broader MLX/auth slice: `87 passed`

## Residual Risk

`score_claim_valid=True` remains the strict auth-axis marker for CPU/CUDA calibration payloads, while `promotion_eligible` and `rank_or_kill_eligible` remain false for CPU calibration rows. That is internally enforced by `required_contest_auth_axis_payload_blockers(...)`, but the field name is easy to misread as promotion authority. A future naming cleanup should separate `calibration_authoritative` from public score-claim wording.
