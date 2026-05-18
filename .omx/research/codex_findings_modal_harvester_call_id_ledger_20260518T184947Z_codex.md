# Codex Findings - Modal Harvester Call-ID Ledger Closure

Date: 2026-05-18 18:49:47 UTC
Author: Codex

## Finding

The Z6-v2 recipe-mode blocker and the primary `tools/harvest_modal_calls.py`
ledger terminalization path were already fixed, but a recurrence class remained:
other Modal harvester/recovery surfaces could call
`FunctionCall.from_id(...).get(...)`, observe terminal provider state, and leave
the canonical `.omx/state/modal_call_id_ledger.jsonl` lifecycle stuck at
`dispatched`.

## Patch

- Added `tac.deploy.modal.harvest_outcomes.append_terminal_call_id_ledger_event`
  as the reusable harvest-side sister of the Catalog #245 dispatch registration
  helper.
- Rewired `tools/harvest_modal_calls.py` to delegate its existing private
  terminal ledger behavior to the shared helper.
- Rewired `tools/parallel_harvest_actuator.py` to record terminal success,
  nonzero rc, Modal result-cache expiry, Modal function timeout, and
  already-harvested backfill; plain poll `TimeoutError` remains nonterminal.
- Added strict Catalog #330 preflight:
  `check_modal_harvesters_record_call_id_outcome`.
- Wired legacy lane-local Modal recoverers through the helper or an explicit
  read-only waiver for `tools/modal_function_status.py`.
- Added missing CLAUDE row for the already-strict
  `check_canonical_task_status_no_dangling_transitions` meta-gate so strict
  preflight row-presence remains clean.

## Verification

- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_parallel_harvest_actuator.py src/tac/tests/test_build_pr101_finetuned_archive_codec_dir.py::test_harvest_modal_calls_appends_call_id_ledger_terminal_event src/tac/tests/test_build_pr101_finetuned_archive_codec_dir.py::test_harvest_modal_calls_supplements_lossy_terminal_call_id_row src/tac/tests/test_check_245_modal_call_id_ledger_registration.py src/tac/tests/test_check_330_modal_harvester_call_id_ledger_outcome.py src/tac/tests/test_ci_ruff_scope.py]`
  - Result: `50 passed in 4.11s`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_check_176_strict_callsite_has_claude_md_row.py]`
  - Result: `14 passed in 1.38s`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check --force-exclude --select F821 src/ experiments/ submissions/robust_current/ scripts/ tools/]`
  - Result: `All checks passed!`
- `[empirical:.venv/bin/python tools/canonical_task_status.py --validate]`
  - Result: `{"rows": 45, "status": "valid"}`
- `[empirical:.venv/bin/python tools/check_canonical_task_status_no_dangling_transitions.py --strict --json]`
  - Result: `{"status": "pass", "violations": []}`
- `[empirical:check_modal_harvesters_record_call_id_outcome(strict=True)]`
  - Result: `0 violations`

## Boundary

This hardens provider-state custody only. It does not promote any harvested
result to a score claim; score authority still flows through axis custody,
runtime/archive custody, component recomputation, and exact-eval gates.
