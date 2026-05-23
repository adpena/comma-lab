# Codex Findings: False-Green Materializer Contract Follow-up

utc: 2026-05-23T21:37:39Z
agent: codex
topic: false-green authority and materializer completion contracts
research_only: false

## Scope

Follow-up adversarial review after the fake-implementation hardening landing.
The prior tranche removed several false-authority paths, but two subagent
reviews found remaining fake-green edges in review gating, greenup import,
storage/cleanup preflight validation, decoder-q advisory ingestion, and
materializer postconditions.

## Fixed Bug Classes

- Contest-final compliance now accepts a supplied `--submission-score` only as
  a matching assertion against the selected-axis auth artifact. The auth
  artifact remains the score source; a matching explicit score no longer
  creates a false red, and a mismatching explicit score still fails.
- Review-gate tracked scope now includes `scripts/`, and the review tracker
  scans `scripts/` so production launch/compliance scripts are not invisible to
  normal review flow.
- Review-gate JSON/DB fallback now blocks critical/standard files when DuckDB
  exists but no readable JSON snapshot is available, instead of warning through
  missing policy evidence.
- `greenup-import` now treats only exact `CLEAN` verdict bullets as clean.
  `NOT CLEAN`, `ISSUES FOUND`, and similar strings no longer grant review.
- Staircase storage preflight accepts the real
  `tools/compact_experiment_artifacts.py` artifact schema
  (`{"plan": ..., "execution": ...}`) and still requires cleanup pressure
  accounting and false-authority fields.
- Decoder-q next-candidate selection accepts current raw advisory batch rows
  while preserving row-level false-authority and custody checks.
- Materializer work queues no longer unlock execution from schema-only JSON.
  They now emit typed completion contracts requiring byte-closed archive
  artifacts, SHA/byte custody, receiver proof readiness, false-authority
  fields, and successful chain steps.

## Verification

- `75 passed`: experiment queue + byte-shaving materializer queue.
- `36 passed`: review-gate fallback and greenup-import regressions.
- `35 passed`: staircase DAG, storage tiers, decoder-q next selector, MLX
  dynamic learned sweep.
- `212 passed`: pre-submission compliance, frontier score helper, Lightning
  batch jobs, comma-lab research-state boundary.
- `ruff check` passed on touched files excluding legacy full-file lint debt in
  `tools/review_tracker.py`; `compileall` passed on touched production modules.

## Outstanding

- `tools/review_tracker.py` still has legacy full-file ruff debt unrelated to
  this patch. This pass kept the review-tracker edits scoped to scan coverage
  and exact CLEAN parsing.
- The materializer completion contract is local/queue authority only. It does
  not grant score authority, promotion eligibility, rank/kill eligibility, or
  exact-eval readiness.
