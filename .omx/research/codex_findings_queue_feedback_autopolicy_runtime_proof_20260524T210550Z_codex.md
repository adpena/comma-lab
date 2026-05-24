# Codex Findings: Queue Feedback Autopolicy Runtime Proof

UTC: 2026-05-24T21:05:50Z
Lane: codex_queue_feedback_replan_local_policy_20260524

## Finding

The materializer campaign feedback loop had two incomplete automation
boundaries. First, the local feedback follow-up queue could be resumed by an
autopolicy flag, but the resulting policy state was not surfaced in operator
briefing and the autopolicy guard did not check every false-authority alias used
by `experiment_queue.v1`. Second, family-agnostic materializer rows could count
as local queue success when the manifest had byte-closed output but did not prove
runtime consumption.

## Landing

- Added explicit local-autopolicy fields to materializer campaign summaries and
  operator briefing rows: policy, enabled state, blocker count, execution
  request, executed state, and success state.
- Hardened queue-feedback local autopolicy against the complete local
  false-authority alias set: required-false fields, false-or-missing fields, and
  `dispatch_packet_ready`.
- Tightened family-agnostic materializer postconditions so archive-section,
  packet-member, and tensor-factorize materializer rows require
  `receiver_contract_satisfied=true`,
  `receiver_verification.receiver_contract_satisfied=true`, and a nonempty
  `runtime_consumption_proof_path` before they can count as succeeded work.

## Authority Boundary

This remains local feedback-loop automation only. It is not score authority,
promotion authority, rank/kill authority, paid dispatch authority, or exact eval
authority. Exact CPU/CUDA auth eval still requires normal lane claim, runtime
custody, and auth-axis payload gates.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py tools/run_byte_shaving_materializer_campaign.py tools/operator_briefing.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py`
- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py -q`
  passed with `103 passed`.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_work_queue_wraps_archive_section_entropy_recode_adapter src/tac/tests/test_byte_shaving_campaign_queue.py::test_family_agnostic_candidate_postconditions_reject_weak_receiver_manifest -q`
  passed with `2 passed`.

## Remaining Gaps

- Local-autopolicy execution should become a typed dependent-queue actuator
  rather than direct runner-controlled queue control.
- Storage preflight should be required, or explicitly waived, for local
  execution that can materialize large outputs.
- `local_io_heavy` still needs a host-level claim/lock when multiple campaign
  queues or state files run on the same machine.
- The PR95 MLX audit found stale source-faithful blocker taxonomy and identified
  the next scorer-coupled loss attachment point at
  `src/tac/local_acceleration/pr95_hnerv_mlx.py::run_pr95_mlx_synthetic_timing_smoke`.
