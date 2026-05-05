---
name: Subagents Must Git Commit
description: Always git add and commit after subagent file changes return — don't let subagent work sit uncommitted
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
When a subagent completes and has modified files (writeup, code, configs, etc.),
immediately `git add` and `git commit` those changes before doing anything else.

**Why:** Subagent work is invisible history if not committed. The user caught that
subagent writeup improvements were sitting uncommitted. Git history IS the research
timeline — every uncommitted change is lost context for future sessions.

**How to apply:** After every subagent completion notification that touched files:
1. `git diff --stat` to see what changed
2. `git add <specific files>` 
3. `git commit` with descriptive message crediting the subagent's work
4. THEN continue with other work

This applies to ALL subagents — writeup, council implementations, research, etc.
Do not batch subagent commits. Commit each one individually as it completes.
