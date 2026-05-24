# Codex Findings: Materializer Runner Feedback Replan Request

- UTC: 2026-05-24T18:16:35Z
- Lane: `codex_materializer_runner_replan_artifacts_20260524`
- Scope: harden the runner feedback artifact so queue telemetry cannot masquerade as score evidence or as a complete runnable replan without required identities.

## Findings

The runner replan artifact needed one more guard layer after the standalone `queue_performance_summary.json` and placeholder were added. The action-functional consumer requires runtime identity, cache identity, output paths, and a real action source; queue telemetry alone is only a denominator update.

The runner now emits `queue_feedback_replan_request.json` with:

- `schema=byte_shaving_materializer_campaign_feedback_replan_request.v1`;
- source run, plan, queue, state, and queue performance summary paths;
- `ready_for_action_functional_feedback=false` until runtime/cache identities and completed queue telemetry are present;
- explicit blockers for missing runtime/cache identity or unavailable performance summaries;
- a full command template, while `suggested_action_functional_command` stays null when blockers remain.

The exact-readiness handoff pointer now uses the queue's canonical `exact_eval_handoff/exact_readiness` directory, not the stale `readiness` name.

The runner also treats worker and observer exits as reportable outcomes instead
of checked subprocess hard-stops. A materializer failure now returns nonzero at
the end of the runner, but still writes `materializer_campaign_run.json`,
`queue_performance_summary.json`, `queue_feedback_replan_request.json`, and
`canonical_response_update_placeholder.json`.

## Authority Boundary

Malformed, failed, or truthy-authority queue performance output is converted to `experiment_queue_performance_summary_unavailable.v1` with blockers. The runner does not write poisoned telemetry as a consumable `experiment_queue_performance_summary.v1`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_executes_no_paid_inverse_scorer_chain_and_handoff src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_poison_summary_when_performance_stdout_invalid src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_poison_summary_when_performance_command_fails src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_poison_summary_with_nested_authority -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_inverse_steganalysis_acquisition.py -q`
- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_inverse_steganalysis_acquisition.py`
- `git diff --check -- tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- 2026-05-24 follow-up verification: `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
- 2026-05-24 follow-up verification: `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_experiment_queue.py -q`
- 2026-05-24 follow-up verification: `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_inverse_surface_cells_compile_to_action_functional_work_queue -q`
- 2026-05-24 follow-up verification: `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- 2026-05-24 follow-up verification: `.venv/bin/python tools/lane_maturity.py validate`

## Remaining Gap

The next tranche should wire runtime/cache identity production into the local runner invocation or run config so `queue_feedback_replan_request.json` can become ready automatically after a successful materializer run, still without granting score or dispatch authority.
