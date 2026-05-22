# Codex Findings: MLX Windowed Dataset Contract Greenup

**UTC:** 2026-05-22T00:21:19Z
**Lane:** `lane_mlx_auth_scorer_production_hardening_20260521`
**Scope:** follow-up adversarial review of `3bb2f029f` windowed MLX scorer-response dataset landing.

## Finding

`3bb2f029f` added a useful per-window baseline path for direct MLX scorer-response rows, but it weakened one production contract invariant: direct MLX response ingestion accepted payloads with missing `requires_exact_eval_before_promotion`. That field is part of the false-authority contract for MLX scorer-response artifacts and must stay mandatory.

Two quiet-failure risks were also present:

- duplicate baseline responses for the same scorer-pair window silently used the last baseline loaded;
- `tools/build_mlx_window_response_dataset.py` exited success even when every candidate was skipped.

## Fix

- Restored mandatory `requires_exact_eval_before_promotion=true` for direct MLX scorer-response ingestion.
- Added duplicate-baseline-window detection; ambiguous windows now produce no rows for that window.
- Added CLI fail-closed behavior when zero usable rows survive, with `--allow-empty` as the explicit advisory escape hatch.
- Added regression tests for missing exact-eval flag, duplicate baselines, and empty CLI output.

## Real FEC6 Check

The previously generated candidate response lacked the now-mandatory exact-eval flag and correctly fails the stricter windowed dataset builder. Regenerating the same FEC6 CPU b2 window with current `tools/run_mlx_scorer_response_cache.py` produces a strict payload that passes the windowed dataset path:

```bash
tmpdir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_mlx_scorer_response_cache.py \
  --reference-cache-dir experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600 \
  --candidate-cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --archive-size-bytes 178517 \
  --repo-root . \
  --batch-pairs 2 \
  --start-pair 16 \
  --max-pairs 4 \
  --device cpu \
  --output "$tmpdir/next_response_cpu_b2_pairs16_20_strict.json"

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_mlx_window_response_dataset.py \
  --candidate-response "$tmpdir/next_response_cpu_b2_pairs16_20_strict.json" \
  --baseline-response experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/baseline_response_cpu_b1_pairs16_20.json \
  --json-out "$tmpdir/windowed.json" \
  --md-out "$tmpdir/windowed.md"
```

Observed strict windowed dataset:

- `row_count=1`
- `family_counts.mlx_scorer_response=1`
- `score_claim=false`
- `window_baseline_key=start=16:max=4:window=16-20`
- `delta_vs_baseline_score=3.6323648794356345e-07`

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_plan_ll_scorer_response_next_cli.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_profile_stability.py
# 42 passed in 0.74s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/optimization/scorer_response_dataset.py \
  src/tac/tests/test_scorer_response_dataset.py \
  tools/build_mlx_window_response_dataset.py \
  tools/build_scorer_response_dataset.py \
  tools/plan_ll_scorer_response_next.py
# All checks passed!

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_plan_ll_scorer_response_next_cli.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_batch_invariance.py \
  src/tac/tests/test_mlx_production_contract.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py \
  src/tac/tests/test_mlx_to_pytorch_export.py
# 138 passed in 17.27s
```
