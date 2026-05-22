# Codex Findings Addendum: MLX Calibration Scalar Cross-Check

timestamp_utc: 2026-05-22T04:57:58Z
lane_id: lane_codex_mlx_auth_scorer_hardening_20260522
author: codex
verdict: PROCEED_WITH_GUARDS

## Scope

Follow-up to `codex_findings_mlx_calibration_auth_source_gate_20260522T045502Z_codex.md` after the wider MLX test sweep.

## Clarification

The calibration contract is:

- scalar-only `cpu_score` / `cuda_score` anchors are rejected;
- `cpu_auth_eval_path` / `cuda_auth_eval_path` must point to strict contest auth-axis payloads;
- if a scalar is also present next to a strict auth payload, it is treated only as a redundant consistency cross-check and must match the auth payload score within tolerance.

The strict auth payload remains the authority source. A matching scalar does not by itself unlock spend-triage calibration.

## Verification

Commands:

```bash
.venv/bin/python -m ruff check src/tac/tests/test_mlx_score_calibration.py
.venv/bin/python -m pytest src/tac/tests/test_mlx_score_calibration.py -q
```

Observed:

- `ruff`: pass
- calibration tests: `10 passed`
