# Codex greenup: preflight cache and source-index performance (2026-05-09)

## Scope

- `src/tac/preflight.py`
- `src/tac/tests/test_preflight_all_clean_cache.py`
- `src/tac/tests/test_undeployed_artifact_producers.py`

## Pass 1 review

### src/tac/preflight.py -- CLEAN

Verdict: clean.

Rationale: the generic codebase-clean cache helper preserves the existing
`preflight_all_clean` semantics while adding a separate
`preflight_developer_clean` namespace. The developer cache is only consulted
for codebase-only invocations with no artifact or training input paths, so
artifact validation and training-input validation still run live.

### src/tac/tests/test_preflight_all_clean_cache.py -- CLEAN

Verdict: clean.

Rationale: tests cover unchanged-tree hits, source-change misses, ignored
rebuildable research artifacts, and separation between developer and release
cache namespaces.

### src/tac/tests/test_undeployed_artifact_producers.py -- CLEAN

Verdict: clean.

Rationale: the source-index single-pass regression test validates both the
producer map semantics and the file/text miss counts for the optimized scan.

## Pass 2 review

### src/tac/preflight.py -- CLEAN

Verdict: clean.

Rationale: the CLI remains bounded by the existing 30-second timeout. The new
cache changes the no-input developer fast path only after a complete clean run
has already stored the current source/state fingerprint.

### src/tac/tests/test_preflight_all_clean_cache.py -- CLEAN

Verdict: clean.

Rationale: tests are deterministic, repo-local, and exercise cache identity by
metadata fingerprint rather than wall-clock timing.

### src/tac/tests/test_undeployed_artifact_producers.py -- CLEAN

Verdict: clean.

Rationale: the added test stays local to a temporary repo and does not weaken
the existing deployment, basename, strict-mode, or library-exemption coverage.
