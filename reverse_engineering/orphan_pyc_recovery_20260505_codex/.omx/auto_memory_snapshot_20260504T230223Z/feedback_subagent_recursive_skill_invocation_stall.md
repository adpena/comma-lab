---
name: Subagent recursive skill invocation stalls — never invoke /codex from inside a subagent
description: 2026-04-27 Lane S subagent (Self-Compression engineering) stalled at 600s while trying to invoke `/codex:adversarial-review` mid-task. Stream watchdog did not recover. The failure mode is independent from the one-shot Vast.ai-eval pattern. Subagents should NEVER invoke slash-skills — only parent does.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Rule: subagents MUST NOT invoke slash-skills (codex, plan, brainstorm, etc.).** Only the parent agent invokes skills.

**Why:**
- 2026-04-27 Lane S engineering subagent (`aee686786fb9a6317`) stalled at 600s. Failure summary: "Agent stalled: no progress for 600s (stream watchdog did not recover)". The result message captured: "There's a `codex` skill. The instructions say 'use codex 2nd-approver via review-tracker pattern'. Let me invoke the codex rescue skill for the CRITICAL inflate_renderer.py change:" — and then it hung indefinitely.
- This is a SECOND subagent failure-mode pattern, distinct from the one-shot Vast.ai-eval failure (`feedback_oneshot_vastai_subagent_failure_pattern`). Both have the same root: subagents handle linear well-bounded tasks well, but fail when they need to invoke nested asynchronous tooling (Vast.ai lifecycle in the first case, slash-skills in the second).
- Cost: Lane S +1299 lines of partial work in the working tree (314 self_compress.py + 645 renderer_export.py + 178 profiles.py + 134 train_renderer.py + 33 inflate_renderer.py) — uncommitted, possibly broken, requires triage before either commit or revert.

**How to apply:**
1. When dispatching a subagent, the prompt MUST forbid `/codex`, `/plan`, `/brainstorm`, and other slash-skill invocations. The subagent does the linear work; the parent invokes adversarial review against the diff afterward.
2. Subagent prompts that say "use codex 2nd-approver via review-tracker pattern" are now FORBIDDEN — the parent does the codex pass on the staged diff before commit.
3. Add the rule to the agent-creator template so future subagents inherit the constraint.
4. If a subagent legitimately needs adversarial review of a critical file (CLAUDE.md mandates 2-pass review for inflate_renderer.py and other safety-critical paths), the subagent should mark the file with the review-tracker `pending-2nd-approval` status and STOP, then return control to the parent.

**Triage protocol when a subagent stalls leaving partial state:**
1. Inspect the working tree diff size (`git diff --stat`) to estimate how much was completed.
2. If the changes are isolated to NEW functions (e.g., new `export_*` + `load_*`), they may be safely committable.
3. If the changes touch CRITICAL paths (inflate_renderer.py, scorer-load gates, archive builders), NEVER commit without manual codex review.
4. Default action: stash with a note (`git stash push -m "Lane S partial - subagent stall 2026-04-27"`) and pick up later when the parent has bandwidth.
5. Do NOT delete the partial work — 1299 lines of code is non-trivial; treat it as a recoverable WIP.

**Related failure-mode memories:**
- `feedback_oneshot_vastai_subagent_failure_pattern` — subagents bail on Vast.ai create+SSH+SCP+launch+monitor+destroy lifecycle (5/5 failure). Use launch-and-return-early instead.
- `feedback_parallel_codex_fix_subagents` — subagents handle WELL-BOUNDED linear edits in parallel when each owns a single file cluster.

**The unifying principle:** A subagent succeeds when its task is "edit these files, run tests, return". It fails when its task involves inversion-of-control (waiting for an external system, invoking a nested skill that itself spawns work). Keep subagents purely synchronous + file-bounded.
