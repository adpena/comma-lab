---
name: Concurrent subagent commit-message swap when 4-5+ agents commit in parallel
description: 2026-04-29 PM. Spawning 5 BG subagents that all stage + commit work in parallel caused commit MESSAGES to be attached to the wrong commit objects. Code lands intact; attribution is shuffled. Discovered when Lane PD-V2 subagent and Lane Joint-ADMM subagent both reported their changes landed on commits with the OTHER lane's body.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The bug

Today (2026-04-29 PM) I spawned 5 background subagents to implement Phase 1.5 lanes in parallel:
- Lane 8 inner loop MVP (subagent a2bf40d0be817e4bb)
- Lane PD-V2 arithmetic-coded pose deltas (subagent a2a86198381fdb54a)
- Lane Ω-W-V2 water-fill+arithmetic (subagent a7244150d68877748)
- Lane Joint-ADMM Boyd coordinator (subagent a9be6df8735bed278)
- Lane J-NWC full neural weight codec (subagent a0bcb0b6882410dc7)

When the agents completed and reported back:

**Joint-ADMM agent's report** (commit `0e43d299`/`152ba503`):
> "my Joint-ADMM body landed on commit 152ba503 but the actual ADMM source files landed on 0e43d299. Per CLAUDE.md 'never destructive git commands' I did not amend/rebase. The code is canonical at HEAD; `git log --all -- src/tac/joint_admm_coordinator.py` confirms presence."

**PD-V2 agent's report** (commit `152ba503`):
> "Multiple agents are concurrently committing in this session. My Lane PD-V2 changes were captured by commit 152ba503 (titled 'Lane Joint-ADMM coordinator' by the parallel agent) instead of landing under my drafted Lane PD-V2 commit message. The CODE is correct and the tests pass; the commit message attribution is an artifact of the concurrent-agent execution, not a code issue."

**Lane 8 agent's report** (commit `0e43d299`):
> "Commit message arrived attached via amend after a parallel agent's commit hook collision swept the original commit message; the amended HEAD has the correct Lane 8 message but the diff includes some unrelated `joint_admm_*` files that another agent staged."

Net effect: code landed correctly, but `git log` shows commit BODIES that don't match the actual diffs. Forensic recovery requires `git show <commit> --stat` per commit + grep for the source file.

## Why it happened

Each subagent independently:
1. Implemented its work
2. Ran `git add <files>`
3. Ran `git commit -m "..."`

When 2+ subagents reach `git commit` near-simultaneously:
- The first to acquire git's commit lock wins.
- The losing subagent sees the "other" agent's staging area in HEAD already (because git index is shared).
- Re-staging + re-committing creates a commit with the LOSER's body but contains the WINNER's files (because they were staged first).
- Or: pre-commit hooks (review-gate, preflight) fire on the COMBINED staged set, blocking both, and they retry interleaved.

This is a `git index is process-shared, commit lock is per-process` race condition. NOT a Modal/codex thing — pure git plumbing.

## Permanent fixes (pick one)

**Option A — Per-subagent worktree** (RECOMMENDED):
Each subagent gets its own `git worktree add` so it has an isolated index + HEAD pointer. Commits land in the worktree's branch, then a coordinator merges them sequentially. Native git feature designed for this.

**Option B — Subagent commit serialization**:
Wrap subagent prompts with a "commit lock" instruction: claim the lock at start of commit, release after success. Fragile (relies on file lock semantics + cleanup on crash) but doesn't require worktree setup.

**Option C — Subagents stage but DON'T commit**:
Subagents emit `git add <files>` only, then return. The PARENT agent reviews each subagent's staged set + commits sequentially with the right message. Loses some autonomy but eliminates the race entirely.

**Option D — Each subagent uses `git stash push` after staging**:
Convert work to a stash entry per-subagent. Parent agent applies + commits each stash sequentially. Race-free but parent must manage stash queue.

## Forensic recovery for THIS incident

To verify what's where:
```bash
# Find which commit actually contains a given file modification:
git log --all --follow -- src/tac/joint_admm_coordinator.py
git log --all --follow -- src/tac/pose_delta_codec_v2.py
git log --all --follow -- src/tac/water_filling_codec_v2.py

# See the actual diffs per commit:
git show 0e43d299 --stat
git show 152ba503 --stat
git show 9987a5d9 --stat
```

The Lane Ω-W-V2 commit `9987a5d9` is the only one that didn't get swapped — likely because that subagent finished earlier than the other 4. The Joint-ADMM/PD-V2/Lane-8 commits all collided in a tight window.

## When to use this knowledge

- Before spawning 4+ BG subagents that will each commit: pick fix Option A (worktree) or Option C (parent commits).
- After the fact, if commit messages don't match diffs: don't rebase/amend; just document the actual file→commit mapping in a memory file or commit notes.
- The user pre-approval message "spawn a grand council subagent ... do all council's spec work, land all other work you can do, spawn subagents to implement all of these" implies parallel subagents are OK; but the implementation should use Option A worktrees to avoid this race.

## Cross-refs

- 5 affected commits (2026-04-29 ~5-6PM): 0e43d299, 152ba503, 9987a5d9 (clean), 45d808ae (mine, sequential, also clean)
- No memory existed for this bug before today; this file is the first.
- See also: feedback_modal_spawn_result_cache_pattern_20260429.md (companion incident from same parallel-dispatch session)
