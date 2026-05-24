# Codex Findings: Materializer Runner Replan Artifacts

- UTC: 2026-05-24T18:11:35Z
- Lane: `codex_materializer_runner_replan_artifacts_20260524`
- Scope: close the local materializer campaign feedback loop by emitting first-class queue performance and replan/update artifacts.

## Findings

`tools/run_byte_shaving_materializer_campaign.py` already built the action functional, campaign plan, materializer execution queue, optional runtime policy, worker run, queue observation, and performance summary. The missing loop closure was artifact custody: the queue performance summary existed only as an embedded field inside `materializer_campaign_run.json`, so the next action-functional build did not have a stable `--queue-performance-summary` artifact path to consume.

The runner now writes two durable files in every campaign run directory:

- `queue_performance_summary.json`: the standalone `experiment_queue_performance_summary.v1` payload, forced through false-authority fields and tagged for inverse-action acquisition denominator updates.
- `canonical_response_update_placeholder.json`: a planning-only response/replan handoff with source run path, queue path, queue state path, runtime policy paths, exact-readiness handoff paths, and the next-run hint `["--queue-performance-summary", "<path>"]`.

The run summary also records both artifact paths, `response_update_applied=false`, `replan_required=true`, `next_run_hint`, and discovered exact-readiness handoff paths.

## Authority Boundary

The new artifacts are telemetry/replan inputs only. They do not claim score, promotion eligibility, rank/kill authority, exact-eval readiness, paid dispatch authority, or byte-closed archive validity. Exact-readiness handoff paths are evidence pointers; they do not authorize dispatch.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_executes_no_paid_inverse_scorer_chain_and_handoff src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_executes_no_paid_packet_member_handoff -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_inverse_steganalysis_acquisition.py -q`
- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_inverse_steganalysis_acquisition.py`
- `git diff --check -- tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`

## Remaining Gap

This closes the runner -> replan artifact custody gap. The next highest-EV gap is to make operation-set compiler lowering a reusable service rather than a narrow inline bridge, and to broaden executable materializers so more inverse-action cells can become byte-closed operation sets automatically.
