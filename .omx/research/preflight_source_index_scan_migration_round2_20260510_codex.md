# Preflight SourceIndex scan migration round 2 (2026-05-10)

## Result

Four additional Python-file scan loops in `src/tac/preflight.py` now use the
active `SourceIndex` text cache when available instead of re-opening files
inside each strict check.

## Migrated checks

- `check_block_fp_exponents_alongside_qint`
- `check_segmap_export_calls_verify_roundtrip`
- `check_state_writers_strict_load_for_mutating_path`
- `check_phase_b_auth_memo_in_repo`

## Review note

The worker migration initially left two large loop bodies with visually
misleading indentation. `check_state_writers_strict_load_for_mutating_path` had
the dangerous variant: the scan body drifted under the negative text prefilter,
which would have skipped the intended analysis for files that actually matched
the prefilter. That was corrected before landing and covered by the existing
equivalence tests plus py_compile.

## Evidence

Regression coverage added to:

- `tests/test_preflight_source_index_equivalence.py`

Focused verification:

```text
.venv/bin/python -m py_compile src/tac/preflight.py
.venv/bin/python -m pytest -q tests/test_preflight_source_index_equivalence.py
.venv/bin/python -m pytest tests/test_parallel_dispatch_top_k_exact_ready_audit.py src/tac/tests/test_optimizer_exact_ready_audit.py tests/test_audit_exact_ready_queues_cli.py src/tac/tests/test_operator_briefing.py tests/test_promote_optimizer_candidate_for_exact_eval_cli.py
```
