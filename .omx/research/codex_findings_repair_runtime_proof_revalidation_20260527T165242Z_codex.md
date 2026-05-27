# Codex Findings: Repair Runtime Proof Revalidation

Date: 2026-05-27T16:52:42Z

## Verdict

The repair materialization chain no longer treats `proof_present`,
`receiver_consumed`, or `receiver_closed` booleans as sufficient custody. Parent
rate-credit rows, direct materializer manifests, and child component replay
manifests now re-read the runtime proof JSON artifact and require the proof to
bind to the expected archive SHA before the candidate is considered locally
receiver-consumed.

## Landed Integration

- Added a local runtime-consumption proof revalidation record inside
  `frontier_rate_attack_feedback`.
- Rate-only parent materialization now revalidates both archive file custody and
  runtime proof JSON before setting `candidate_archive_materialized`,
  `runtime_consumption_proof_present`, or `receiver_consumed`.
- Materializer manifest binding now rejects proof files that only carry
  schema-level booleans or stale blockers, and it preserves the revalidation
  record in the binding row.
- Child component replay manifests now revalidate the inherited proof file
  instead of carrying the parent `runtime_consumption_proof_present` flag.
- The repair materialization CLIs pass `repo_root` through so relative proof and
  archive paths resolve deterministically from operator queue execution.
- Repair tests now use archive-bound proof fixtures, and a regression locks out
  the previous `{"receiver_consumed": true}` soft-proof pattern.

## Authority Boundary

All new revalidation records remain false-authority. They can unblock local
materialization and replay auditing, but they still cannot claim score, promote,
rank/kill, spend budget, or dispatch exact eval. Exact CPU/CUDA auth eval remains
the only score authority.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py tools/build_frontier_repair_budget_materialization_plan.py tools/build_frontier_repair_budget_child_component_replay_manifests.py src/tac/tests/test_repair_budget_materialization_execution.py src/tac/optimization/repair_campaign_posterior.py tools/append_repair_campaign_stackability_posterior.py src/comma_lab/scheduler/repair_campaign_stackability_queue.py src/tac/optimization/__init__.py src/tac/tests/test_repair_campaign_stackability_queue.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/optimization/repair_campaign_posterior.py tools/append_repair_campaign_stackability_posterior.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_budget_materialization_execution.py src/tac/tests/test_repair_campaign_stackability_queue.py -q`
  - `15 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
  - `55 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue_observer.py -q`
  - `27 passed`
- `.venv/bin/python tools/lane_maturity.py validate`
  - `1439 lane(s) validated cleanly`

## Remaining Work

The observer and repair materialization chain now reject soft receiver proof.
The next closure target is the exact-readiness handoff from materializer
execution queues into dispatch candidates: the queue should carry the same
revalidation records into the final byte-closed archive packet and make exact
dispatch selection consume them automatically.
