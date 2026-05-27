# Codex Findings: Target Profile Chain Handoff Wiring

UTC: 2026-05-27T17:40:31Z

## Finding

The frontier target-optimization profile was queue-visible at the targeted
component correction root, but downstream generated artifacts could still shed
that metadata as the rate attack moved from acquisition into work orders,
response harvests, materialization requests, operation-chain work orders,
stage-plan inputs, and materializer handoff rows.

That created a false-authority and signal-loss class: a local queue artifact
could remain executable while no longer carrying the contest-video or corpus
target profile that explains what its measurements are allowed to train,
optimize, or spend against.

## Fix Landed

- Added one canonical extractor for target-profile metadata across queue roots,
  nested metadata, and direct profile payloads.
- Propagated the profile through targeted correction work orders, response
  harvest rows, materialization request rows, materialization queues,
  chain work orders, operation-chain stage plans, byte-range stage inputs,
  targeted drop-many stage inputs, and materializer handoff/backlog rows.
- Updated the work-order CLI and queue command emitter so queue-owned local
  execution receives the same false-authority target metadata as the queue root.
- Added regression assertions across the queue, harvest, request, stage-plan,
  and handoff boundaries.

## Verification

- `.venv/bin/python -m py_compile src/comma_lab/scheduler/frontier_rate_attack_feedback.py tools/build_frontier_targeted_component_correction_work_order.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py tools/build_frontier_targeted_component_correction_work_order.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py::test_frontier_feedback_compiler_discovers_materializers_and_refreshes_dqs1_queue src/tac/tests/test_frontier_rate_attack_feedback.py::test_rate_budget_preservation_keeps_rate_only_floor_for_distortion_regressions src/tac/tests/test_frontier_rate_attack_feedback.py::test_targeted_component_correction_materialization_requests_group_responses src/tac/tests/test_frontier_rate_attack_feedback.py::test_post_auxiliary_targeted_component_refresh_reharvests_into_chain -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`

## Next Integration Edge

The next no-signal-loss boundary is the observer and exact-readiness side:
when queue-owned materializer-chain results are harvested or refused, their
target profile should remain present in observer revalidation, exact-readiness
refusal rows, and any canonical frontier-rate attack feedback-cycle summaries.
