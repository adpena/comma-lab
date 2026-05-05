---
name: Parallel codex-finding subagents on non-overlapping file scopes is the working pattern
description: 2026-04-27: Confirmed across two codex review rounds (R5 + R5-2). When codex returns N findings, dispatching one rigor subagent per finding-cluster (split by file ownership, not by severity) is consistently faster + cleaner than a single big subagent OR sequential fixes. User affirmed after the third round.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**The working recipe (validated 3x in 2026-04-27 session):**

1. Run `/codex:adversarial-review` in background.
2. When codex returns N high/medium findings, group them into clusters by FILE OWNERSHIP (not by severity). Each cluster = one subagent's exclusive write scope.
3. Dispatch one rigor subagent per cluster IN PARALLEL via `run_in_background:true`.
4. Each subagent prompt should include: the verbatim codex finding text, the exact files in scope, "you OWN <file> for this work" so subagents don't fight, and a "DO NOT commit, leave in working tree" instruction so the parent can review + commit.
5. When notifications arrive, verify tests, two-pass review-tracker mark, commit each cluster as its own logical commit. The codex reviewer is registered as an L3 approver principal so it can satisfy the second-approver requirement on CRITICAL files (the 2026-04-27 a21a3387 policy commit).

**Why this beats alternatives:**

- vs. single big subagent on all findings: parallel subagents have isolated context windows; one big subagent burns >100K tokens and risks losing detail by the end. Three parallel agents sharing the load is roughly the same wall time + better fidelity.
- vs. sequential codex → fix → codex → fix: parallel saves 2-3x wall time and lets the next codex round see all fixes at once.
- vs. doing fixes inline myself: subagents have fresh context + adversarial framing. They catch incidental bugs the inline version skips (the pose_dim incidental fix on 2026-04-27 was caught only because the subagent was paranoid about Issue 1's class).

**Critical constraint that prevents merge conflicts:**

- File-ownership split, not topic split. Two subagents touching the same .py file at different lines WILL cause merge friction at commit time.
- For ADDITIVE work (e.g., adding new functions), tell the subagent to leave a TODO comment for wiring instead of touching the central dispatcher (the 2026-04-27 109a85e8 meta-bug commit landed cleanly because subagent C was forbidden from touching `preflight_all()` — that wiring is a separate commit after parallel subagent A's flip lands).

**When NOT to parallelize:**

- Findings that share a single file at overlapping lines — serialize them OR put both in one subagent prompt.
- Findings that depend on each other's outputs (e.g., "fix A so B becomes testable") — sequential.

**Cost in this session:** 3 codex rounds × ~3 fix subagents each = ~9 parallel subagent dispatches over ~6h wall, all clean commits, zero merge conflicts. User affirmed pattern is working.
