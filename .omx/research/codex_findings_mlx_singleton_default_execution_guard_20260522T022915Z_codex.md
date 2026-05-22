# Codex Findings: MLX Singleton Default Execution Guard

UTC: 2026-05-22T02:29:15Z

## Verdict

PROCEED for singleton CPU MLX scorer-response execution as the default local
training/candidate-generation signal path.

Non-singleton MLX scorer batches are now explicitly research-only. They cannot
enter the response runner, profiler/stability selector, or execution-plan bridge
without an explicit `--allow-batch-shape-research-signal` / API equivalent.

## Trigger

The prior clean-HEAD FEC6 parity audit found:

- `batch_pairs=1`: 300/300 windows passed, zero SegNet argmax mismatches.
- `batch_pairs=4`: 74/75 windows passed, but `[208, 212]` failed with one
  SegNet argmax mismatch on a boundary pixel.

That means singleton is the proven local MLX signal shape. Multi-pair batching
is an optimization lane, not a production default.

## Guardrail Landed

- `tools/run_mlx_scorer_response_cache.py` defaults to singleton and fails
  cleanly for `batch_pairs > 1` unless the caller explicitly marks the run as
  batch-shape research.
- `tools/profile_mlx_scorer_response_cache.py` changed default
  `--batch-pairs` from `1,2,4` to `1`.
- `tac.local_acceleration.mlx_profile_stability` rejects non-singleton rows
  from default execution selection.
- `tac.local_acceleration.mlx_execution_plan` rejects non-singleton recommended
  rows unless explicit batch-shape research is requested.
- Both runner CLIs now fail with `FATAL:` messages instead of raw Python
  tracebacks on validation errors.

## Verification

Focused test command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_production_contract.py
```

Result: 36 passed.

CLI failure proof for non-singleton scorer response:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_mlx_scorer_response_cache.py \
  --reference-cache-dir /does/not/exist/reference \
  --candidate-cache-dir /does/not/exist/candidate \
  --archive-size-bytes 1 \
  --output /tmp/should_not_exist_mlx_response.json \
  --batch-pairs 2
```

Result: `rc=2`, clean `FATAL:
mlx_scorer_response_non_singleton_batches_require_explicit_batch_shape_research_signal_allowance`.

CLI failure proof for non-singleton profiling:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/profile_mlx_scorer_response_cache.py \
  --reference-cache-dir /tmp/ref \
  --candidate-cache-dir /tmp/cand \
  --archive-size-bytes 1 \
  --output /tmp/should_not_exist_profile.json \
  --batch-pairs 1,2
```

Result: `rc=2`, clean `FATAL: --batch-pairs values other than 1 require
--allow-batch-shape-research-signal`.

## Authority Contract

This is local MLX implementation-signal hardening only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact contest CPU/CUDA auth eval remains required for any score claim,
  promotion, rank/kill decision, or public frontier claim.

## Next Action

Use singleton CPU MLX for the first scorer-response dataset/training loop.
Treat SIMD/MLX/GPU/multi-batch acceleration as a measured optimization lane
behind batch-invariance gates, not as the default route for frontier decisions.
