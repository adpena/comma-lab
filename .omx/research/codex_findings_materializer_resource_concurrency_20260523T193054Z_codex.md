# Codex Findings - Materializer Resource Concurrency Controls

UTC: 2026-05-23T19:30:54Z
Lane: `lane_materializer_resource_concurrency_20260523`

## Finding

The inverse-scorer byte-shaving loop had DAG/work-queue ownership and the
materializer execution queue schema already supported per-resource concurrency,
but the operator-facing queue builder still made resource-specific saturation
too manual. Local CPU and MLX capacity should be tunable from the campaign
queue command line so broad candidate batches can run concurrently without hand
editing generated queue JSON.

## Fix

- `tools/build_byte_shaving_campaign_queue.py` now accepts repeatable
  `--materializer-resource-concurrency KIND=LIMIT` overrides.
- The CLI validates missing `=`, empty resource kinds, non-integer limits, and
  limits below one.
- Parsed limits flow into `build_materializer_execution_queue(...)`, preserving
  existing queue controls while allowing resource-specific settings such as
  `local_cpu=8` and `local_mlx=3`.

## Verification

- `src/tac/tests/test_byte_shaving_campaign_queue.py`: 29 passed.
- Integrated focused slice
  `src/tac/tests/test_inverse_scorer_cell_materializer.py`
  `src/tac/tests/test_artifact_retention.py`
  `src/tac/tests/test_optimizer_exact_readiness.py`
  `src/tac/tests/test_exact_dispatch_authority.py`
  `src/tac/tests/test_byte_shaving_campaign_queue.py`: 127 passed.
- `ruff check` on changed source/tests/tools: passed.
- `compileall` on changed source/tests/tools: passed.
- `git diff --check`: passed.

## Authority

This is orchestration throughput wiring only. It does not claim score,
promotion, rank, kill, or exact-eval dispatch authority.
