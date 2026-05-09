# DEFER T9 directive for in-flight subagent a0be36e (2026-05-09)

<!-- generated_at: 2026-05-09T06:10:00Z, from_state_hash: operator_approved_defer_t9 -->

## Operator decision (2026-05-09): DEFER T9

The HNeRV retrospective (subagent a1a9359d) identified T9 (cross-archive substrate composition) as the **kitchen_sink anti-pattern under a new name** — violates 10 of 13 lessons in the new CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" section. Operator approved DEFER.

## What this means for a0be36e

You are in-flight building "T8+T9 implementation w/ scaffolds." Continue the T8 W₂ Sinkhorn surrogate work (with the codex HIGH 3 fix already landed in `src/tac/losses.py` — minimum blur 0.01, safe iteration floor, soft-distribution regression test).

**T9 work**: STOP current T9 implementation. Either:

(a) **Land T9 as DEFERRED scaffold only** — write the lane registry entry as `lane_t9_cross_archive_substrate_composition` at L0 with:
- `status: DEFERRED-pending-substrate-anchor`
- `reactivation_criteria`: ["A1 score-aware substrate verified at [contest-CUDA]", "≥1 second composable substrate verified at [contest-CUDA]", "single-axis branching decision approved by council"]
- `notes`: "Per CLAUDE.md HNeRV parity discipline lesson 5 + forbidden pattern #5: cross-archive composition without verified substrate is the kitchen_sink anti-pattern. PR105 (1776 LOC, 21 files) lost to rem2's 241 LOC silver. Re-scope to single-axis branching from A1 OR defer until composable substrates exist."
- DO NOT write any T9 implementation code; the lane is a pre-registered SKETCH only.

(b) **Re-scope T9 to single-axis A1 branching** (alternative per lesson 5) — if you can complete this within remaining budget, this is acceptable. The re-scope means:
- Pick ONE substrate axis (e.g., the latent stream from A1)
- Build a single experiment that branches from A1 along that axis
- Predicted save: -0.005 to -0.015 score [predicted]
- This is NOT cross-archive composition; it's substrate-engineering on the verified A1 anchor

Default to (a) unless you have time for (b) and operator-clear permission.

## Coordination

- The CLAUDE.md addition has landed (2026-05-09T06:05Z) — your work must honor the 13 lessons + 5 forbidden patterns
- Codex HIGH 3 fix on `src/tac/losses.py` Sinkhorn already landed; T8 implementation should USE the fixed loss (not re-implement it)
- Co-author trailer auto-appended via `tools/subagent_commit_serializer.py`
- 3-clean-pass adversarial greenup before declaring done

## References

- CLAUDE.md HNeRV parity discipline section (lines 64-119 after edit)
- HNeRV retrospective: `~/.claude/projects/.../feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`
- Operator decisions executed: `.omx/research/operator_decisions_executed_20260509.md`
- Codex review fixes (Sinkhorn HIGH 3): `.omx/research/codex_adversarial_review_findings_for_inflight_subagents_20260509.md`
