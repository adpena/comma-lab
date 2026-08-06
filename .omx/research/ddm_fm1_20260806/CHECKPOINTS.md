# ddm_fm1 Checkpoints

| Checkpoint | Status | Evidence |
|---|---|---|
| Governing contract read | COMPLETE | `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, operating manual, main hot state, fm1 charter, and common contract were read before edits. |
| SDK research | COMPLETE | Current Apple Python/native Foundation Models surfaces researched online; external fmtools checkout inspected read-only and pinned by commit/file SHA-256. |
| Recall pass | COMPLETE | Memory, `.omx/research`, DAG/code surfaces, and canonical equations searches were consulted; no direct fmtools equation surface found. |
| Implementation | COMPLETE | Added advisory `charter_class` and `mechanism_reduction_language`; wired queue WARN-only enrichment. |
| Behavior guard | COMPLETE | Deterministic optimal-form lint remains the only strict refusal path; fmtools absence is silent. |
| Tests | COMPLETE | Targeted pytest: 54 passed. Ruff F checks passed. |
| Review tracker | COMPLETE | New/changed Python symbols marked through two review passes. |
| Live fmtools probe | PARTIAL | Availability true, but classifier generation returned `None` / unclassified RuntimeError rows; live advisory classification remains unverified. |
| Serializer commit | BLOCKED | Main checkout serializer reached Git staging and failed with `Operation not permitted` while creating Git object temp files; index remained clean after failure. |
| Scorer / exact eval | NOT RUN | Scorer slot is not owned by fm1; this is a no-score, no-pointer-move unit. |
