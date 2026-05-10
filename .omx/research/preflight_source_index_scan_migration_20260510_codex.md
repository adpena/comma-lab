# Preflight SourceIndex scan migration (2026-05-10)

## Scope

This ledger records a small DX/performance tranche after the operator's
direction that preflight must stay fast, parallelizable, and free of repeated
per-check full-tree scans.

## Change

Six strict preflight scanners were migrated away from direct
`scan_dir.rglob("*.py")` loops and onto the canonical `_iter_python_files(...)`
path:

- `check_state_writers_strict_load_for_mutating_path`
- `check_state_writers_own_their_lock_end_to_end`
- `check_state_helper_paths_explicit`
- `check_paid_job_register_before_submit`
- `check_lightning_submit_cancel_only_before_network`
- `check_phase_b_auth_memo_in_repo`

Inside normal `preflight_developer()` / `preflight_all()` these now share the
process-local `SourceIndex` inventory instead of rebuilding independent file
walks. The no-index fallback remains pure Python and preserves the old
semantics.

## Verification

Commands:

```bash
.venv/bin/python -m py_compile src/tac/preflight.py
.venv/bin/python -m pytest \
  tests/test_preflight_source_index_equivalence.py \
  src/tac/tests/test_source_index.py \
  src/tac/tests/test_codex_round78_check_147_lightning_submit_cancel_pre_network.py \
  src/tac/tests/test_codex_round6_high2_paid_job_register_before_submit.py -q
.venv/bin/python -m tac.preflight --scope dev \
  --timings-json .omx/research/artifacts/preflight_dev_timing_after_source_index_migration_20260510_codex.json
```

Results:

- 59 focused tests passed.
- Developer preflight passed.
- Timing profile:
  - `wall_elapsed_s=7.662618`
  - `serial_elapsed_s=4.120633`
  - `step_count=23`

## Remaining DX work

- Convert the remaining direct recursive scanners one at a time, with
  SourceIndex/no-index equivalence tests.
- Parallelize `preflight_developer()` read-only checks after the remaining
  mutable/artifact checks are isolated.
- Lower only the stable file-inventory/substr-fact extraction layer to
  Rust/Zig later, after golden equivalence vectors exist. Keep Python AST and
  custody semantics as the oracle until native output is byte-for-byte
  identical.
