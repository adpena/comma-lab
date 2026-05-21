# Codex Findings: MLX Scorer-Response Cache Harness

Timestamp: 2026-05-21T22:55:08Z
Agent: Codex
Lane: mlx_auth_scorer_port
Evidence grade: local MLX CPU scorer-input response harness
Score claim: false
Promotion eligible: false

## Summary

Built the first cache-driven MLX scorer-response harness:

- loads reference and candidate fixed scorer-input caches;
- validates shape and `pair_indices` alignment;
- runs the MLX PoseNet+SegNet scorer adapters in batches;
- computes upstream-compatible PoseNet and SegNet distortion components;
- writes a non-authoritative JSON payload with contest-formula components;
- exposes a CLI under `tools/run_mlx_scorer_response_cache.py`.

This is the practical bridge between byte-closed scorer-input cache custody and
local MLX training/search. It does not replace CPU/CUDA auth eval.

## Landed surfaces

- `src/tac/local_acceleration/mlx_scorer_response.py`
  - `load_scorer_input_cache(...)`
  - `build_mlx_scorer_response_payload(...)`
  - `write_mlx_scorer_response_payload(...)`
- `tools/run_mlx_scorer_response_cache.py`
- `src/tac/tests/test_mlx_scorer_response.py`

The CLI requires:

```text
--reference-cache-dir
--candidate-cache-dir
--archive-size-bytes
--output
```

Optional:

```text
--batch-pairs
--device cpu|gpu
--components-dir
```

## Empirical verification

New focused tests:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_scorer_response.py -q
2 passed in 1.90s
```

Broader MLX/local-acceleration suite:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_local_acceleration.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
94 passed in 14.92s
```

Ruff:

```text
.venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_scorer_adapters.py \
  src/tac/local_acceleration/mlx_scorer_response.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_response.py \
  tools/run_mlx_scorer_response_cache.py
All checks passed!
```

## Adversarial notes

- The harness rejects hash-only caches because it needs actual tensors. Hash-only
  manifests remain appropriate for auth-axis identity artifacts.
- The payload is explicitly non-promotable:
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.
- The JSON includes `canonical_score` only as a local MLX research signal from
  local component averages and caller-provided archive bytes. The field exists
  so `check_mlx_scorer_fidelity.py` can compare deltas against auth-eval
  payloads; it is not a leaderboard or promotion score.
- The CLI defaults to MLX CPU because current evidence shows MLX GPU drift is
  not yet an authority-grade parity surface for SegNet.

## Recommended next action

Use this harness on a real byte-closed cache pair:

1. Build full reference and candidate scorer-input caches from recovered raw
   outputs with `tools/build_mlx_scorer_input_cache.py`.
2. Audit cache hashes against the recovered auth-eval scorer-input hash
   manifest.
3. Run `tools/run_mlx_scorer_response_cache.py` on MLX CPU.
4. Gate the resulting payload through `tools/check_mlx_scorer_fidelity.py`
   against the matching Modal Linux contest-CPU artifact.

Only after that passes should MLX scorer responses steer larger local
optimization or candidate-search loops.
