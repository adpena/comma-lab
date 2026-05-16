# L5 V2 Paired Dispatch Work-Unit Self-Description Hardening

score_claim=false
promotion_eligible=false
rank_or_kill_eligible=false
ready_for_exact_eval_dispatch=false
dispatch_attempted=false

## Purpose

The L5-v2 paired measurement dispatch plan was already fail-closed and visible
through `tools/operator_briefing.py`, but raw JSON consumers had to know the
internal `measurement_blockers_to_close` + `dispatch_blockers` split. A simple
consumer could ask for `blockers` or operator readiness on each work unit and
see nothing, even though the plan was not dispatchable.

This hardening makes each work unit self-describing:

- `ready_for_operator_dispatch=false`
- `ready_for_provider_dispatch=false`
- `readiness_blockers=[measurement blockers + dispatch blockers]`
- `blockers=readiness_blockers`

The top-level plan remains planning-only and non-promotional.

## Current Evidence

- Artifact updated:
  `.omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.json`
- Markdown updated:
  `.omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.md`
- Builder updated:
  `src/tac/optimization/l5_v2_paired_measurement_dispatch_plan.py`
- Operator surface updated:
  `tools/operator_briefing.py`
- Tests updated:
  `src/tac/tests/test_l5_v2_paired_measurement_dispatch_plan.py`
  and `src/tac/tests/test_operator_briefing.py`

## Dispatch Meaning

No dispatch authority is added. The three active work units still require real
byte-closed archives, archive SHA-256 values, submission runtime paths, and
operator execution intent before the paired Modal CPU/CUDA dispatcher can run.

The change only prevents false-negative blocker reads from downstream tools and
future agents. It is a no-signal-loss surface improvement, not a score claim.

## Next Concrete L5 Work

The highest remaining frontier blocker is not this dispatch-plan schema. It is
materializing at least one byte-closed C1/Z5/TT5L probe artifact whose paired
CPU/CUDA work unit can replace `FILL_ARCHIVE_ZIP`, `FILL_ARCHIVE_SHA256`, and
`FILL_SUBMISSION_DIR` in the current plan. Until then, the L5-v2 staircase
correctly stays non-dispatch-ready.
