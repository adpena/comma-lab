# Codex Findings - Posterior Follow-Up Queue Wiring - 2026-05-27T20:42:25Z

## Scope

The repair stackability posterior already emitted acquisition follow-up routes,
but feedback refresh only embedded them in a summary. That left learned signal
visible to humans while the queue/DAG layer still needed manual interpretation.

## Landing

- Added a reusable scheduler surface that turns
  `repair_campaign_posterior_acquisition_followup.v1` routes into
  `repair_posterior_acquisition_followup_queue.json`.
- The queue maps posterior policies to executable child queues:
  targeted-response harvest routes run the targeted correction queue, receiver
  credit routes run receiver repair, MLX custody routes run the repair campaign
  score queue, and exact-auth routes remain fail-closed.
- Feedback refresh and feedback cycle now write the posterior follow-up queue,
  summary, and normal validate/init/run operator commands.
- Autonomous chain optimization now recognizes the posterior follow-up queue as
  a child action when the artifact is present.
- The final-rate autoloop priority list already had a posterior-followup slot;
  tests now pin that it is selected ahead of repair waterfill when both are
  ready.

## Authority

All new artifacts remain false-authority:

- `budget_spend_allowed=false`
- `ready_for_exact_eval_dispatch=false`
- no score, promotion, rank, or kill authority

The queue only converts learned local posterior signal into bounded local work.
Exact contest CPU/CUDA anchors remain mandatory before promotion or score claims.

## Validation

- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_repair_campaign_scorer.py -q`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_score_queue.py -q`

## Remaining Work

Next loop should execute a bounded feedback-refresh smoke and inspect the
generated posterior follow-up queue rows. If the route mix is dominated by
frozen exact-auth handoffs, the next integration should add a claim-aware exact
handoff queue instead of silently dropping those routes.
