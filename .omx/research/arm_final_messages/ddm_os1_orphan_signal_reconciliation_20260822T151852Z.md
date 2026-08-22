Executed the OS1 charter in full.

Frozen census at `2026-08-22T15:07:14Z`:

- 26/9,119 ordinary-untracked research Markdown files: 11 final-message captures and 15 charters.
- 35/9,119 separately ignored artifact/extraction files.
- 4/9,047 tracked graph-eligible files absent from the frozen cache.
- Recall reached 28/32 seeded-random memos; all four misses were ambiguous-title collisions and were recovered at rank 1 with distinctive queries.
- Prior-law verdict: supported at the shared arm-lifecycle producer level, but refuted as literally one directory.

The producer cure is implemented in [codex_arm_queue.py](/Users/adpena/Projects/pact/tools/codex_arm_queue.py:242), tested in [test_codex_arm_queue.py](/Users/adpena/Projects/pact/src/tac/tests/test_codex_arm_queue.py:475), and documented in [the OS1 memo](/Users/adpena/Projects/pact/.omx/research/ddm_os1_orphan_signal_reconciliation_20260822.md). It blocks queue intake/saturation when managed captures or charters do not exactly match `HEAD`, including ignored, staged-only, nested, modified, deleted, and symlink cases.

Verification: 91 tests passed, compileall passed, Ruff fatal/static checks passed, and `git diff --check` passed. The retained machine-readable [receipt](/Volumes/APDataStore/pact/ddm_os1_orphan_signal_reconciliation/os1_census_reachability_receipt.json) is 26,597 bytes with SHA-256 `93bef57a9db16c8b94c4296e29bd826c167f4aa6b7586df290e2d92c6c70885f`.

Landing is blocked, not complete. The canonical serializer included the final memo, both Python files, and all 27 managed births visible at lock time. Git rejected staging with `Operation not permitted` and rc 128; the staged index remains empty. Nothing was bypassed or falsely claimed committed.

S = `0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`; OS1 moved no score or archive bytes.

## NEXT_IF_RESUMED

- **BLOCKED_PENDING_SERIALIZER** — owner: MAIN or the next Git-writable operator; consumer store: `git:main` plus the graph-memory research corpus; fire trigger: Git index/object writes become available. Re-run the canonical serializer with post-edit SHA-256 values for every path returned by `uncustodied_research_births()`, the two Python files, and the OS1 memo.

## LIVE-HYPOTHESES

- Distinctive retrieval-card headings or title-collision expansion should improve default recall because every sampled miss became rank 1 when filename tokens disambiguated it.
- Once landed, the fail-closed queue gate should stop this orphan class from accumulating because every measured ordinary-untracked birth passed through the guarded arm-lifecycle surfaces.

## DEAD-ENDS

- “Git-untracked means graph-absent” is closed: 16/26 untracked files were already indexed.
- Blanket-committing ignored raw artifact Markdown is closed: that class is deliberately routed to payload custody and compact promoted signal.
- Auto-committing from the asynchronous completion watcher is closed because it could absorb unrelated shared-index state.
- Rebuilding graph memory during baseline measurement is closed because it self-confounds the frozen comparison.

