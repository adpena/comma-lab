# Codex Findings - Artifact Retention Cold-Store Verification

- **UTC:** 2026-05-23T06:47:22Z
- **Lane:** `lane_codex_retention_cold_store_verify_20260523`
- **Scope:** cold-store safety for high-volume MLX/local CPU queue saturation
- **Authority:** non-score infrastructure hardening; no score claim

## Finding

Artifact retention already revalidated candidates before deletion and could
move certified scratch into a cold-store root, but the move path trusted the
destination root too weakly and copied directly to the final destination before
deleting the source. A destination typo, repo-local cold-store path, failed
write, or interrupted copy could reduce signal custody while trying to relieve
disk pressure.

## Fix Landed

- Added `validate_cold_store_root(...)` before any move execution:
  - root must exist, be a directory, and not be a symlink;
  - root must be outside `repo_root`;
  - free space must cover the reclaimable byte estimate;
  - a write/read/fsync probe must pass;
  - the execution payload and journal start event record device/free-space
    metadata and false-authority markers.
- Changed cold-store moves to copy into a `.partial-*` destination, digest
  verify, rename into the final destination, digest verify again, and only then
  delete the source.
- Added `candidate_error` journal rows on move/delete exceptions.
- Added `journal_path` / `cold_store_contract` / `cold_store_verification`
  payload fields so retention actions are reconstructable.
- Added `.omx/state/artifact_retention_journals/` to `.gitignore` because
  default execution journals are local state, while durable findings belong in
  dated `.omx/research/` memos.

## Verification

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_artifact_retention.py
```

Result: `10 passed`.

```bash
.venv/bin/python -m ruff check src/comma_lab/artifact_retention.py tools/compact_experiment_artifacts.py src/tac/tests/test_artifact_retention.py
```

Result: `All checks passed!`.

```bash
git diff --check
```

Result: passed.

## Remaining Work

- Extend MLX cache retention so scorer-input caches are movable only with
  dereferenced auth-audit stamps and are not deletable by default.
- Add a cold-store index for `/Volumes/APDataStore` once the mounted SSD path is
  chosen as the active store.
- Expand retention certifiers by manifest schema rather than only directory
  suffixes.
