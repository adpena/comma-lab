# Codex Findings: Feedback Refresh Portfolio Wiring

## Finding

The strict final-rate autoloop preflight found a real integration gap. The
post-execute feedback refresh emitted child queues, but the CLI path in
`tools/build_frontier_rate_attack_feedback_refresh.py` did not emit
`frontier_rate_attack_portfolio_coverage.json` into its artifact map. This meant
new strict child-queue execution correctly refused with
`frontier_rate_attack_portfolio_coverage_artifact_missing`.

## Landing

- Wired the feedback-refresh CLI to the existing
  `build_frontier_rate_attack_portfolio_coverage(...)` helper.
- Persisted `frontier_rate_attack_portfolio_coverage.json`.
- Added it to CLI stdout artifacts, persisted `feedback_refresh_report.json`, and
  operator commands.
- Added regression coverage to `test_frontier_feedback_cli_writes_valid_followup_queue`.

## Verification

- `.venv/bin/ruff check tools/build_frontier_rate_attack_feedback_refresh.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q -k 'frontier_feedback_cli_writes_valid_followup_queue or portfolio_coverage'`
- Synthetic strict end-to-end smoke:
  `.venv/bin/python tools/build_frontier_final_rate_attack_queue.py --no-current-frontier --archive smoke=/tmp/pact_portfolio_preflight_smoke2.iGp7O2/archive.zip --output-dir /tmp/pact_portfolio_preflight_smoke2.iGp7O2/out --execute --max-steps 2 --max-parallel 1 --post-feedback-child-queue-limit 1 --post-feedback-child-queue-max-steps 1 --post-feedback-child-queue-max-parallel 1 --poll-interval-seconds 0.01 --idle-sleep-seconds 0 --max-idle-cycles 1`

Smoke result: portfolio preflight valid, `preflight_blocked_execution=false`,
`selected_queue_count=1`, `executed_queue_count=1`,
`observer_revalidation_valid=true`, and all authority fields remained false.

## Authority

This is local scheduler and planner custody only.

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

