# NEXT IF RESUMED: ddm_cu2

## If this unit is already committed

No continuation is required. Spot-check with:

```bash
.venv/bin/python -m pytest src/tac/tests/test_sister_checkpoint_guard.py src/tac/tests/test_subagent_commit_serializer.py src/tac/tests/test_serializer_file_attribution_reconcile.py src/tac/tests/test_subagent_commit_serializer_postcommit_clobber.py
.venv/bin/python -m ruff check tools/subagent_commit_serializer.py src/tac/commit_safety/sister_checkpoint_guard.py src/tac/tests/test_sister_checkpoint_guard.py src/tac/tests/test_serializer_file_attribution_reconcile.py
```

## If serializer commit was blocked by managed Git writes

Preserve these files exactly:

- `tools/subagent_commit_serializer.py`
- `src/tac/commit_safety/sister_checkpoint_guard.py`
- `src/tac/tests/test_sister_checkpoint_guard.py`
- `src/tac/tests/test_serializer_file_attribution_reconcile.py`
- `.omx/research/ddm_cu2_20260805/RECEIPT.md`
- `.omx/research/ddm_cu2_20260805/CORRECTION_06fa0ad37d_911.md`
- `.omx/research/ddm_cu2_20260805/NEXT_IF_RESUMED.md`

Run the same focused tests and lint above. Then commit via
`tools/subagent_commit_serializer.py` with `[no-triality] [p0-ledger-ok]`, label `ddm_cu2`,
and per-file `--expected-content-sha256` values computed after edits.

## Boundaries

- Do not mutate the staged index for #914 unless a live staged entry reappears and the operator
  explicitly approves the disposition.
- Do not run scorers or launches for this custody unit.
- Do not revert `06fa0ad37d` solely for attribution; use the correction memo and structural guard.
