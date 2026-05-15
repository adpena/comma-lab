# Subagent Checkpoint Discipline Backfill - 2026-05-15

Purpose: preserve Catalog #206 strictness without rewriting already-landed
history. These entries are SHA-scoped backfills for commits whose serializer log
classified them as subagent commits after the strict cutoff but whose commit body
does not carry a checkpoint token or waiver. The gate now accepts only explicit
lines with `# CHECKPOINT_DISCIPLINE_BACKFILLED:<reason>`.

Scope reviewed:

- `git show -s --format=fuller <sha>` for the listed commits.
- `.omx/state/commit-serializer.log` rows for the listed commits.
- Current worktree conflict recovery showed no unmerged entries after Catalog
  #248 resolution.

Backfilled commits:

- commit e9d1ced36aaecc38ee59a814b9c6d55a4d406985 - # CHECKPOINT_DISCIPLINE_BACKFILLED:already in origin/main; serializer log marks it as subagent work but body has only attribution; current recovery found no unreported WIP to recover and history rewrite is not acceptable
- commit 639e00fe0c26da8a0a7c35905909421c6db5c7d0 - # CHECKPOINT_DISCIPLINE_BACKFILLED:already on local main; commit body has extensive conflict-resolution custody and no unreported WIP remained, but it predates the exact Catalog #206 token wording now enforced by preflight
- commit c2d538f7d9ff38bf4502f929f6af7019d336d601 - # CHECKPOINT_DISCIPLINE_BACKFILLED:single-file transactional catalog claim; helper failed to emit the documented waiver before this hardening pass; no lane work or file-edit payload was hidden in this commit
- commit fb38f3919277f2435db27f08d3ea3dc9df2714b7 - # CHECKPOINT_DISCIPLINE_BACKFILLED:single-file transactional catalog claim; helper failed to emit the documented waiver before this hardening pass; no lane work or file-edit payload was hidden in this commit

Forward fix:

- Catalog #206 token matching is case-insensitive, so the documented phrase
  "Checkpoint discipline honored" is accepted.
- Historical backfill is limited to this committed research ledger pattern and
  exact 40-character commit SHAs.
