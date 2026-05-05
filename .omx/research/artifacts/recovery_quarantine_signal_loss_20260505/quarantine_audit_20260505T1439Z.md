# Recovery Quarantine Audit

- quarantine: `/Users/adpena/Projects/pact/.recovery_quarantine_20260505T004735Z`
- total_files: `745`

## Category Counts

| category | files |
|---|---:|
| `memory_snapshot` | 562 |
| `recovery_spec` | 87 |
| `public_or_experiment_intake` | 32 |
| `test` | 21 |
| `experiment_tool` | 17 |
| `tac_module` | 11 |
| `script` | 6 |
| `provider_state` | 4 |
| `docs` | 3 |
| `submission_runtime` | 1 |
| `tool` | 1 |

## Disposition Counts

| disposition | files |
|---|---:|
| `duplicate_same_safe_to_delete_after_manifest_commit` | 566 |
| `do_not_promote_incomplete_recovery_preserve_for_manual_rehydration` | 88 |
| `preserve_until_matching_source_is_canonical_then_delete` | 87 |
| `blocked_recovery_input_needs_canonicalization_before_promotion` | 3 |
| `compare_by_hand_live_diff_before_merge_or_delete` | 1 |

## Promotion Queue

| relpath | category | bytes | sha256 |
|---|---|---:|---|

## Live Diff Queue

| relpath | category | bytes | quarantine sha | live sha |
|---|---|---:|---|---|
| `docs/paper/ara/trace/events.jsonl` | `docs` | 106050 | `72ed77a99fed6d7e` | `0cef0e0c1bf75ad2` |
