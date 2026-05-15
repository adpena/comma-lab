# 2026-05-15 Conflict Snapshot Manifest

This directory preserves the structured, production-safe custody surface for
the 2026-05-15 multi-sister stash-pop conflict recovery.

Tracked files:

- `20260515_conflict_status_after_22c711b00.txt`
- `20260515_unmerged_files_after_22c711b00.txt`
- `20260515_unmerged_stages_after_22c711b00.txt`

The raw conflict diff contains literal git conflict markers and is intentionally
not tracked as public OSS state. It was moved to the local scratch custody path:

`.omx/tmp/conflict_snapshots/20260515_worktree_conflict_diff_after_22c711b00.patch`

SHA-256:

`66ecdb2f5d90ee09fb058f9bf3e69de08ea857abf7cb789445d61f69b52a21bf`

Rationale: the tracked text files preserve file list, index stages, and status
needed for future analysis, while avoiding a committed `.patch` artifact full of
intentional conflict markers that would violate the production preflight surface.
