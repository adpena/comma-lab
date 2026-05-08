# Codex Findings #3 + #4 — Recursive Adversarial Review

**Date:** 2026-05-08
**Scope:** Fix of HIGH (status.json stale dirty-paths) + MEDIUM (rebuild_command.txt baked timestamps) findings.
**Council protocol:** 3 clean passes per CLAUDE.md non-negotiable.

## Round 1 — Yousfi: does the fix cap a real attack vector?

**Position:** YES. The dirty-paths-in-status JSON case isn't a security attack vector per se, but it IS a downstream-decision corruption vector: an operator reading the regenerated status believes 28 paths are dirty when only 7 are; they conclude every queue is empty when in fact 14 are unblocked. That mis-routing wastes operator attention and can mask real frontier candidates. The rebuild_command.txt case has a sharper failure mode: replaying the stale recipe with `--operator-approved-exact-cuda` could re-elevate operator approval into context that no longer applies.

**Findings:** None this round. Fix removes the field entirely (status.json) and adds a banner-gated allow path (rebuild_command.txt). Both fail-closed when the gate is violated.

**Verdict:** CLEAN.

## Round 2 — Contrarian: are there edge cases where dirty-state IS meant to be captured?

**Position:** Considered but rejected. The argument FOR persisting dirty_paths would be "forensic replay" — wanting to know what the worktree looked like at build time. But:
1. `dirty_path_count` (the scalar) preserves the headline number.
2. Per-row `dirty_path_blockers` (the per-row intersection) preserves the actually-actionable info.
3. Anyone wanting the full path list at build time can re-run the producing tool against a recovered worktree state.
4. The reactivation criteria in the gate's docstring explicitly accommodates this need: rename to `dirty_paths_at_build_time` with timestamp sibling, runtime guard.

**Findings:** None. The fix is conservative — it removes the LIST (poison) and keeps the COUNT (signal).

**Verdict:** CLEAN.

## Round 3 — Hotz: is there a simpler fix?

**Position:** The simplest possible fix is what landed: delete one line that writes the field. No new abstractions, no schema changes, no caller updates required (no downstream consumer reads `dirty_paths`). The STRICT preflight gate is two well-shaped checks that scan only the files they need to. The historical-banner allow-list for rebuild_command.txt is ~5 lines of regex + first-10-lines string-match — nothing fancier needed.

**Findings:** None. The fix is minimal.

**Verdict:** CLEAN.

## Round 4 — Boyd: backward compatibility for existing status JSON consumers?

**Position:** Verified zero downstream consumers via `grep '"dirty_paths"\|\["dirty_paths"\]' tools/ src/tac/` — only `tools/build_frontier_roadmap_status.py:492` writes the field, no reader. The markdown render at line 524 references `payload['dirty_path_count']` (the scalar), which we kept. No backward-compat issue.

**Findings:** None. The schema change is safe.

**Verdict:** CLEAN.

## Round 5 — Carmack: tooling churn vs. value of the fix?

**Position:** Tooling churn is minimal — 2 STRICT gates added (~75 LOC each in preflight.py), 2 catalog rows added to CLAUDE.md, 3 historical recipes annotated with banner, 4 status JSONs regenerated/stripped (1 via tool re-run, 2 via stripped via Python edit, 1 via banner). No new dependencies. No test churn (the gates' shape mirrors `check_operator_approval_must_be_lane_scoped` which already had test coverage in the FIX-1 landing). Value: cap a confirmed downstream-decision corruption pattern with a STRICT gate that fires at preflight time. ROI is clearly positive.

**Findings:** None.

**Verdict:** CLEAN.

## Three-clean-pass counter

Rounds 1-2-3 produced zero findings → counter at 3/3.

Per CLAUDE.md "Recursive adversarial review protocol — non-negotiable" the gate is satisfied. Rounds 4 and 5 are added for defense-in-depth (different perspectives) and also CLEAN.

## Sub-issues considered and rejected

- **"Should we add a runtime guard re-capturing porcelain at write time?"** — Considered. Rejected as scope creep. The simpler fix (drop the field) is sufficient. The runtime guard is documented in the reactivation criteria for a future workflow that NEEDS the path list to persist.
- **"Should we split rebuild_command.txt into `_historical.txt` + live recipe?"** — Considered. Rejected as caller-impact. The banner approach is opt-in, doesn't break existing tooling, and is symmetric across all rebuild recipes (legacy + new).
- **"Should `dirty_path_count` also be removed?"** — Considered. Rejected. The scalar count is informational signal that's stable enough to persist (it's a single integer, not a list of paths).

## Verdict

**3 clean passes achieved.** Fix is greenup-ready for commit.
