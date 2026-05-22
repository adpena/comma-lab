# Codex Findings: MLX Generated Markdown Authority Blocks

timestamp_utc: 2026-05-22T05:01:51Z
lane_id: lane_codex_mlx_auth_scorer_hardening_20260522
author: codex
verdict: FIXED

## Scope

Markdown audit found that generated MLX scorer-response reports showed deltas
and `Score claim: False` but omitted the full false-authority block. The JSON
payloads were mostly fail-closed, but the Markdown summaries were too easy to
misread as score or ranking evidence.

## Fix

Added a shared generated-report renderer for authority fields and wired it into:

- `tac.optimization.scorer_response_dataset.render_markdown`
- `tac.optimization.scorer_response_dataset.render_validation_gate_markdown`
- `tac.optimization.scorer_response_dataset.render_next_probe_plan_markdown`
- `tac.optimization.scorer_response_family_delta.render_family_delta_markdown`
- `tac.optimization.decoder_q_response_surface.render_decoder_q_response_surface_markdown`

The rendered block now includes:

- evidence_grade
- evidence_tag
- score_axis
- score_claim
- score_claim_valid
- promotion_eligible
- rank_or_kill_eligible
- ready_for_exact_eval_dispatch
- promotable

MLX windowed datasets and derived family/surface reports now propagate
`[macOS-MLX research-signal]` where the row axis is uniformly MLX.

## Additional Contract Tightening

The MLX production contract was tightened so optional singleton
batch-invariance manifests still have their false-authority fields checked,
even when batch-invariance pass/fail is advisory for singleton responses.

## Verification

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check src/tac/local_acceleration/mlx_production_contract.py src/tac/local_acceleration/mlx_score_calibration.py src/tac/tests/test_mlx_production_contract.py src/tac/optimization/scorer_response_dataset.py src/tac/optimization/scorer_response_family_delta.py src/tac/optimization/decoder_q_response_surface.py src/tac/tests/test_scorer_response_dataset.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_score_calibration.py src/tac/tests/test_mlx_segnet_trace_compare.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_scorer_torch_parity.py src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_scorer_fidelity.py src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_auth_eval_schema.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_decoder_q_surface_objective.py -q
git diff --check
```

Observed:

- ruff: pass
- targeted MLX/auth/scorer-response pytest: `187 passed`
- diff whitespace check: pass

## Non-Authority

This is report-surface and local-production-contract hardening only. It does
not promote any MLX, decoder-q, HFV9, or grayscale LUT result.
