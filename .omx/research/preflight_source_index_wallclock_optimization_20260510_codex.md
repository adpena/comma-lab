# Preflight SourceIndex wall-clock optimization - 2026-05-10

research_only: true

## Scope

Ownership honored:

- `src/tac/preflight.py`
- `tests/test_preflight_source_index_equivalence.py`
- this dated ledger

No dispatch tools, score-lane files, archives, or provider surfaces were modified.

## Migrated checks

Another safe batch of broad source scans now uses `SourceIndex` candidate
selection and cached text reads when a source-index context is active:

- `check_state_writers_strict_load_for_mutating_path`
  - Migrated candidate discovery from direct `_iter_python_files(...)` over
    `src/tac` and `tools` to intersected SourceIndex substring candidate sets.
- `check_state_writers_own_their_lock_end_to_end`
  - Migrated direct scan over `src/tac`, `tools`, and `scripts` to SourceIndex
    doc-token plus save-token candidate intersection.
- `check_state_helper_paths_explicit`
  - Migrated direct scan over `src/tac`, `tools`, and `scripts` to SourceIndex
    helper-import candidate selection.
- `check_unsafe_test_only_restricted_to_test_paths`
  - Migrated direct recursive scan over `src/tac`, `tools`, `experiments`, and
    `scripts` to SourceIndex files containing both `Phase3DispatchGate` and
    `unsafe_test_only=True`.
- `check_phase_b_auth_memo_in_repo`
  - Migrated direct scan over `src/tac`, `tools`, `experiments`, and `scripts`
    to SourceIndex files containing both `phase_b_preconditions_status` and
    `auth_memo_path`.

The prewarm group list now includes the exact `("src/tac", "tools")` and
`("src/tac", "tools", "scripts")` Python groups used by the migrated state
checks.

## Equivalence tests

Added focused SourceIndex/no-index equivalence tests for:

- `check_state_writers_own_their_lock_end_to_end`
- `check_state_helper_paths_explicit`
- `check_unsafe_test_only_restricted_to_test_paths`

Existing equivalence coverage continues to cover the migrated
`check_state_writers_strict_load_for_mutating_path` and
`check_phase_b_auth_memo_in_repo` paths.

## Verification

- `.venv/bin/python -m py_compile src/tac/preflight.py tests/test_preflight_source_index_equivalence.py`
  - passed
- `.venv/bin/python -m pytest tests/test_preflight_source_index_equivalence.py -q`
  - `20 passed in 0.65s`
- `/usr/bin/time -p .venv/bin/python -m tac.preflight --scope dev --timings-json /tmp/pact_preflight_dev_timing_source_index_20260510.json`
  - `PREFLIGHT PASSED`
  - timing JSON: `wall_elapsed_s=8.316`, `serial_elapsed_s=4.205`, `step_count=23`, `failed_step_count=0`
  - `/usr/bin/time`: `real 8.79`, `user 6.77`, `sys 3.97`

Focused migrated-check micro-timing on this checkout:

| Check | No-index | SourceIndex after prewarm | Rows | Equal |
|---|---:|---:|---:|---|
| `check_state_writers_strict_load_for_mutating_path` | 0.104s | 0.250s | 0 | true |
| `check_state_writers_own_their_lock_end_to_end` | 0.080s | 0.162s | 0 | true |
| `check_state_helper_paths_explicit` | 0.094s | 0.059s | 0 | true |
| `check_unsafe_test_only_restricted_to_test_paths` | 0.909s | 0.066s | 0 | true |
| `check_phase_b_auth_memo_in_repo` | 0.056s | 0.054s | 0 | true |

The focused total includes SourceIndex prewarm cost in the indexed aggregate,
so the useful signal is the per-check post-prewarm timing and exact output
equivalence.

## Remaining hotspots

Current dev-preflight hot steps from the timing JSON:

- `check_no_mps_fallback_default`: 0.841s
- `check_public_pr_intake_clones_pristine`: 0.695s
- `check_dispatch_cli_shell_hazards`: 0.533s
- `check_no_bare_writes_to_shared_state`: 0.427s
- `check_custody_gate_accept_tokens_concrete_only`: 0.329s
- `preflight_shell_lane_arity`: 0.220s
- `check_state_writers_strict_load_for_mutating_path`: 0.216s

Next safe wall-clock targets are likely the remaining broad MPS/eval-roundtrip
family and the public-PR clone pristine check. The strict-load state-writer
check still has SourceIndex substring-index overhead and should be revisited
only if it remains hot after the broader source-scan family is exhausted.

## Unified-solver hook declaration

This landing is `research_only=true`: it changes static preflight scan
mechanics and tests, not scored archive behavior or empirical model results.

- Sensitivity map: N/A
- Pareto constraint: N/A
- Bit allocator: N/A
- Cathedral autopilot dispatch: N/A
- Continual-learning posterior: N/A
- Probe-disambiguator: N/A
