# Codex Findings - Artifact Retention Default Execution Journal

- **UTC:** 2026-05-23T06:41:47Z
- **Lane:** `lane_codex_retention_default_journal_20260523`
- **Scope:** artifact-retention custody for high-volume MLX/local CPU queue saturation
- **Authority:** non-score infrastructure hardening; no score claim

## Finding

`tools/compact_experiment_artifacts.py --execute` already supported execution
journaling when `--json-output` was provided, but stdout-only execution passed
`journal_path=None` into `execute_retention_plan(...)`. That left a valid
operator path where certified deletion/move mutations could occur without a
durable JSONL execution journal.

This is a signal-preservation bug class: even if every candidate is certified
and revalidated, the mutation itself must be reconstructable after crash,
terminal loss, or stdout truncation.

## Fix Landed

- Added `--journal-output` to `tools/compact_experiment_artifacts.py`.
- Added default execution journal routing to:
  `.omx/state/artifact_retention_journals/artifact_retention_execute_<utc>.jsonl`
  when `--execute` is used without `--json-output`.
- Kept the existing sibling journal behavior when `--json-output` is supplied.
- Added `journal_path` to the execution payload returned by
  `comma_lab.artifact_retention.execute_retention_plan(...)`.
- Added a regression test proving stdout-only execution writes the default
  journal and records start/candidate events while deleting only certified raw
  scratch.

## Verification

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_artifact_retention.py
```

Result: `8 passed`.

```bash
.venv/bin/python -m ruff check src/comma_lab/artifact_retention.py tools/compact_experiment_artifacts.py src/tac/tests/test_artifact_retention.py
```

Result: `All checks passed!`.

```bash
git diff --check
```

Result: passed.

## Remaining Work

- Add a cold-store volume contract for attached SSD moves: volume identity,
  free-space probe, write probe, copy digest, index row, and refusal on mismatch.
- Expand retention certifiers by manifest schema rather than only directory
  suffixes so more `auth_eval_work` / `eval_work` raw surfaces become either
  certifiable or explicitly blocked.
- Keep `.gitignore` aligned with every new rebuildable bulk-output namespace.
