# Codex Findings: MLX Modal Tensor Volume Export

## Verdict

PROCEED_WITH_FAIL_CLOSED_VOLUME_CUSTODY.

The auth-side scorer-input tensor export path is now reachable from the Modal
CPU/CUDA wrappers without returning multi-GB `.npy` payloads through the Modal
function result cache. Full tensor payloads are written to the
`comma-auth-eval-cache-artifacts` Modal Volume, while the local result harvests
only the small tensor-volume manifest.

## Changes Landed

- `experiments/contest_auth_eval.py` keeps work-dir custody as the default for
  scorer-input hash/tensor artifacts, but accepts an explicit mounted-volume
  acknowledgement:
  `--allow-scorer-input-cache-artifact-output-outside-work-dir`.
- `experiments/modal_auth_eval_cpu.py` and `experiments/modal_auth_eval.py`
  mount `comma-auth-eval-cache-artifacts` at `/modal_auth_cache`.
- Modal CPU/CUDA wrappers now accept:
  `scorer_input_cache_tensors`,
  `scorer_input_cache_tensor_batch_pairs`,
  `scorer_input_cache_tensor_large_pair_threshold`,
  `allow_large_scorer_input_cache_tensor_export`, and
  `scorer_input_cache_tensor_volume_run_id`.
- Remote wrappers pass the guarded tensor export flags through to
  `contest_auth_eval.py`, commit the Modal Volume, and return only
  `scorer_input_cache_tensor_volume_manifest.json`.
- `mlx_auth_cache_materialization` recommended commands now include the Modal
  CPU tensor-volume export path and `modal volume get` recovery command.

## Authority Contract

This does not make MLX a scoring axis. The tensor-volume export is a custody
and local-training materialization path only. `PASS_CACHE_AUTH_EVAL_IDENTITY`
from the cache/auth audit is still required before using a materialized cache
as local MLX training target data.

## Verification

- `.venv/bin/python -m ruff check experiments/contest_auth_eval.py experiments/modal_auth_eval.py experiments/modal_auth_eval_cpu.py src/tac/local_acceleration/mlx_auth_cache_materialization.py src/tac/tests/test_modal_auth_eval.py src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_auth_cache_materialization.py`
- `.venv/bin/python -m pytest src/tac/tests/test_modal_auth_eval.py src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_auth_cache_materialization.py -q`

Result: 64 passed.

## Recommended Next Action

Dispatch a Modal CPU auth eval for the FEC6 PR101 archive with
`--scorer-input-cache-tensors`, download the resulting
`comma-auth-eval-cache-artifacts/<run_id>/` volume subtree, then rerun
`tools/audit_mlx_scorer_input_cache.py` against the downloaded manifest and
local MLX cache before any local MLX training loop consumes the tensors.
