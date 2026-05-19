# Codex findings — Modal no-spawn guard

**UTC:** 2026-05-19T07:32:02Z  
**Author:** codex  
**Evidence grade:** `[engineering-guard; no score claim]`  
**Source signal:** `.omx/research/cheap_signal_first_dispatch_wave_synthesis_20260519T063932Z.md`

## Finding

The cheap-signal wave documented two Modal runs that exited `rc=0` after app
initialization, mount creation, and function creation, but produced no spawned
function call id and no `modal_metadata.json`. `tools/operator_authorize.py`
treated `rc=0` as dispatch success before checking for `.spawn()` evidence.

That is a false-success dispatch bug: it can mark a paid or score-gating action
as launched while no Modal function executed.

## Fix

- `experiments/modal_train_lane.py` now emits a machine-readable line immediately
  after `fn.spawn(...)` returns:
  `[modal_train_lane] dispatch_completed call_id=<id>`.
- `src/tac/deploy/modal/mount_manifest.py` recognizes that line as an explicit
  spawned-call marker.
- `tools/operator_authorize.py` now fails closed when the Modal process exits
  `rc=0` without any spawned-call marker. Modal app initialization, object
  creation, and function creation are no longer dispatch success.

The guard still avoids duplicate paid work: if output proves a call id exists
but the wrapper later exits nonzero, `operator_authorize` returns the nonzero
code and does not retry after spawn.

## Verification

- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_operator_authorize_canonical_tool.py::test_operator_authorize_modal_dispatch_threads_recipe_lane_id src/tac/tests/test_operator_authorize_canonical_tool.py::test_operator_authorize_modal_dispatch_rejects_rc0_without_spawn_marker src/tac/tests/test_operator_authorize_canonical_tool.py::test_operator_authorize_modal_dispatch_retries_transient_mount_upload_race src/tac/tests/test_operator_authorize_canonical_tool.py::test_operator_authorize_modal_dispatch_fails_closed_after_mount_retry_budget src/tac/tests/test_mount_manifest.py::test_spawn_marker_accepts_machine_readable_dispatch_completed_line src/tac/tests/test_mount_manifest.py::test_spawn_marker_rejects_modal_initialization_without_call_id src/tac/tests/test_modal_train_lane_wave_3_trainer_module_path.py::test_modal_train_lane_main_derives_and_spawns_with_payload]` — 7 passed.
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/operator_authorize.py experiments/modal_train_lane.py src/tac/deploy/modal/mount_manifest.py src/tac/tests/test_operator_authorize_canonical_tool.py src/tac/tests/test_mount_manifest.py src/tac/tests/test_modal_train_lane_wave_3_trainer_module_path.py]` — passed.
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile tools/operator_authorize.py experiments/modal_train_lane.py src/tac/deploy/modal/mount_manifest.py]` — passed.
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/canonical_task_status.py --validate]` — 115 rows valid.

## Residual risk

The broader `test_operator_authorize_canonical_tool.py` file still has an
existing recipe-state failure unrelated to this patch: the live
`substrate_sane_hnerv_modal_a100_dispatch` recipe carries unresolved
`dispatch_blockers`, so its old smoke-override test now fails before dispatch.
This patch did not change recipe dispatchability semantics.

