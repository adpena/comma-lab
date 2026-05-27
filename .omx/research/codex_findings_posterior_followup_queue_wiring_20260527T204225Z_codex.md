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
- Smoke:
  `.venv/bin/python tools/build_frontier_rate_attack_feedback_refresh.py --output-dir .omx/research/frontier_rate_attack_feedback_posterior_followup_smoke_20260527T204506Z --results-root /Volumes/VertigoDataTier/pact/frontier_rate_attack_feedback_posterior_followup_smoke_20260527T204506Z --frontier-artifact-root .omx/research/frontier_final_rate_attack_activation_posterior_smoke_20260527T2021Z --candidate-limit 2 --max-files-per-root 64 --local-cpu-concurrency 1 --local-io-concurrency 1 --action-summary latest`
  produced `repair_posterior_acquisition_followup_queue.json`.
- Smoke queue validation:
  `.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_rate_attack_feedback_posterior_followup_smoke_20260527T204506Z/repair_posterior_acquisition_followup_queue.json validate`
  returned valid with 2 experiments and 5 steps.

## Smoke Signal

The generated queue had 2 ready posterior-followup experiments from 3 posterior
routes. The top route was
`increase_priority_for_targeted_component_response_harvest`, mapped to
`targeted_component_correction_queue`, with no missing prerequisite artifacts.
The second ready route was
`materialize_missing_local_mlx_custody_before_stackability`, mapped to
`repair_campaign_score_queue`, also with no actuation blockers. This proves the
previous route-only signal now advances into executable local queue work.

## Remaining Work

Next loop should execute a bounded feedback-refresh smoke and inspect the
generated posterior follow-up queue rows. If the route mix is dominated by
frozen exact-auth handoffs, the next integration should add a claim-aware exact
handoff queue instead of silently dropping those routes.
