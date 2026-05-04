---
name: Branch Discipline — main only, no worktrees that lose work
description: Always work on main branch. Worktree agents silently lose changes. All fixes must persist to main.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
RULE: All work happens on `main` branch in `/Users/adpena/Projects/pact`. No other branches, no worktrees.

**Why:** Multiple agent implementations were lost because they worked in git worktrees that auto-cleaned after the agent finished. The fixes appeared to be done but never persisted to the working tree. This caused repeated "fix → find the fix is gone → refix" cycles.

**How to apply:**
- NEVER use `isolation: "worktree"` for agents that write code
- Verify every agent's output persists in the working tree after completion: `git diff --stat HEAD`
- If an agent reports changes but git shows no diff, the changes were in a worktree that was cleaned
- The working directory is `/Users/adpena/Projects/pact`, branch `main`
- The upstream repo at `workspace/upstream/comma_video_compression_challenge` is a DIFFERENT repo on branch `master` — never commit pact code there
- After agent completion: always verify with `ls -la` and `grep` that files actually changed
