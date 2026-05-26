# Codex Findings: Materializer Skip Classification

UTC: 2026-05-26T23:51:49Z

## Finding

The exact-readiness bridge correctly refuses non-rate-positive materializer
rows, but higher-level consumers still treated those skipped rows as blocked
exact-readiness work. That polluted two operator surfaces:

1. `frontier_rate_attack_feedback` copied the skipped blocker into
   `top_blockers` and operation-row blockers.
2. `operator_briefing` fell back from an explicit `blocked_candidate_count: 0`
   to counting every non-ready row as blocked.

The result was false blocker telemetry after a zero-delta materializer was
properly classified as acquisition evidence.

## Fix Landed

`frontier_rate_attack_feedback` now separates actionable exact-readiness
candidates from skipped candidates. Skipped non-rate-positive rows are recorded
as `top_skip_reasons`, not `top_blockers`, and chain-level readiness blockers
only fire when there are actionable non-skipped candidates.

`operator_briefing` now preserves explicit zero counts and reports skipped
bridge rows separately from blocked rows.

## Verification

Commands executed:

```bash
.venv/bin/python -m ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py tools/operator_briefing.py src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_operator_briefing.py
.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py::test_exact_readiness_bridge_summary_keeps_skipped_materializers_out_of_blockers src/tac/tests/test_operator_briefing.py::test_materializer_exact_ready_handoff_summary_counts_skipped_not_blocked -q
.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q
.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py -q
git diff --check
```

Results:

- `ruff`: all checks passed
- targeted regression tests: `2 passed`
- `test_frontier_rate_attack_feedback.py`: `49 passed`
- `test_operator_briefing.py`: `51 passed`
- `git diff --check`: clean

## Integration Note

This keeps exact-readiness telemetry for dispatch authority clean while still
allowing materializer-negative outcomes to flow into acquisition/planner logic.
Rate-negative rows should remain planner evidence, not exact-eval work and not
phantom blocker debt.
