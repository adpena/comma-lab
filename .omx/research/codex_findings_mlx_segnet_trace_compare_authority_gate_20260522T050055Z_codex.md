# Codex Findings: MLX SegNet Trace-Compare Authority Gate

timestamp_utc: 2026-05-22T05:00:55Z
lane_id: lane_codex_mlx_auth_scorer_hardening_20260522
author: codex
verdict: PROCEED

## Scope

Hardened `mlx_segnet_trace_compare` so local PyTorch-vs-MLX SegNet layer-trace comparisons cannot accidentally inherit score authority from their inputs or outputs.

## Change

`src/tac/local_acceleration/mlx_segnet_trace_compare.py` now requires trace manifests to carry the local-MLX false-authority contract:

- `score_claim=false`
- `score_claim_valid=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `candidate_generation_only=true`
- `requires_exact_eval_before_promotion=true`
- `evidence_grade="macOS-MLX-research-signal"`
- `evidence_tag="[macOS-MLX]"`
- `score_axis="[macOS-MLX]"`

GPU trace inputs also require explicit `gpu_research_signal_allowed=true`; otherwise the comparison fails closed with the shared MLX GPU research-signal blocker.

Comparison outputs now repeat the non-authoritative evidence labels and include a `device_contract` block with forbidden uses.

## Verification

Commands:

```bash
.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_segnet_trace_compare.py src/tac/tests/test_mlx_segnet_trace_compare.py
.venv/bin/python -m pytest src/tac/tests/test_mlx_segnet_trace_compare.py -q
.venv/bin/python -m pytest $(rg --files src/tac/tests | rg '/test_mlx.*\.py$' | tr '\n' ' ') -q
```

Observed:

- `ruff`: pass
- trace-compare tests: `9 passed`
- full MLX test sweep: `170 passed`

## Residual Risk

This is still diagnostic implementation evidence only. It helps locate MLX/PyTorch drift and guide local port work; it does not replace contest CPU/CUDA auth eval.
