# Codex greenup: Check 39 preflight source-index test (2026-05-09)

## Scope

- `src/tac/tests/test_undeployed_artifact_producers.py`
- Change class: performance regression test for Check 39 archive-artifact producer
  discovery under `tac.source_index.source_index_context`.

## Pass 1 review

### src/tac/tests/test_undeployed_artifact_producers.py -- CLEAN

Verdict: clean.

Rationale: the new test creates two concrete producer scripts and one irrelevant
library file, runs the new archive-producer scan through `SourceIndex`, and
asserts both semantic discovery and the intended single text-pass behavior.

## Pass 2 review

### src/tac/tests/test_undeployed_artifact_producers.py -- CLEAN

Verdict: clean.

Rationale: the test is deterministic, does not depend on repo-global state, and
keeps the performance invariant local to the helper that was changed. Existing
coverage for deployment references, basename matching, strict raising, and
library exemption remains intact.
