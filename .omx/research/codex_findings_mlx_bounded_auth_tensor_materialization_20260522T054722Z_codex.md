# Codex Findings: MLX Bounded Auth Tensor Materialization

## Verdict

PROCEED_WITH_FAIL_CLOSED_AUTHORITY.

The MLX local-acceleration lane still must not be used as score authority, but
the full scorer-input cache materialization path no longer requires eager
all-pair raw tensor loading. Raw `.raw` scorer-input cache writes are now
chunked by `batch_pairs` and written through NumPy `.npy` memmaps while
preserving the same `ARRAY_HASH_DOMAIN` used by the compact auth hash bridge.

## Changes Landed

- `tac.local_acceleration.mlx_preprocess.write_scorer_input_cache_from_raw_file`
  now streams raw frame pairs in bounded chunks and records
  `source_kind=raw` plus `streaming_batch_pairs`.
- Full-cache manifest hashes for raw and video scorer tensors are now streamed
  from the same chunks written to `.npy` memmaps. They no longer copy GB-scale
  memmaps back into contiguous RAM for hashing.
- Raw/video full-cache materialization rejects unbounded `batch_pairs` working
  sets before writing payload files.
- `tools/build_mlx_scorer_input_cache.py` now passes `--batch-pairs` through
  for raw full-cache materialization instead of only for video/hash paths.
- `experiments/contest_auth_eval.py` now has an explicit full tensor export
  path:
  `--scorer-input-cache-tensors-out-dir`.
- Full tensor export remains fail-closed for large pair counts unless
  `--allow-large-scorer-input-cache-tensor-export` is set.
- Hash and tensor artifact paths are resolved under `contest_auth_eval`
  `work_dir`; path escapes fail closed.
- Tensor payloads are not returned through Modal artifacts; only the manifest
  is recorded in provenance. This avoids silently pushing multi-GB arrays
  through the Modal function result channel.

## Current Auth Blocker

The existing FEC6 local MLX cache remains non-transferable to auth score
training because its local inflated raw identity differs from the Modal/Linux
CPU auth eval raw identity. Regenerated materialization plan verdict:

- `AUTH_CACHE_MATERIALIZATION_REQUIRED`
- Next action:
  `materialize_auth_axis_tensor_cache_from_modal_linux_raw_or_export_linux_tensor_cache`

## Verification

- `.venv/bin/python -m ruff check experiments/contest_auth_eval.py src/tac/local_acceleration/mlx_preprocess.py tools/build_mlx_scorer_input_cache.py src/tac/tests/test_mlx_preprocess.py`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_score_calibration.py src/tac/tests/test_mlx_auth_cache_materialization.py src/tac/tests/test_contest_auth_eval.py::test_main_rejects_nonpositive_scorer_hash_batch_before_path_resolution src/tac/tests/test_modal_auth_eval.py::test_modal_auth_eval_rejects_nonpositive_scorer_hash_batch_before_claim src/tac/tests/test_modal_auth_eval.py::test_modal_cpu_auth_eval_rejects_nonpositive_scorer_hash_batch_before_claim src/tac/tests/test_modal_auth_eval.py::test_modal_cuda_scorer_input_hash_bridge_flows_to_remote_call -q`

Result: 37 passed.

Additional adversarial-review hardening in this landing:

- Full-cache float tensor hashes are guarded by a test that fails if
  `_array_sha256` is called on float scorer tensors.
- Path-custody escape is guarded by a test that refuses outputs outside
  `work_dir`.
- Oversized `batch_pairs` is guarded by a bounded working-set test.

## Recommended Next Action

Run an auth-axis materialization on the exact Linux/auth inflated raw surface,
then require `mlx_cache_audit` identity agreement before enabling any local MLX
training loop to use the cache for candidate generation.
