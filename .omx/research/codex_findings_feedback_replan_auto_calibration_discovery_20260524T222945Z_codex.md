# Codex Findings: Feedback Replan Auto Calibration Discovery

UTC: 2026-05-24T22:29:45Z
Lane: `codex_feedback_replan_auto_calibration_discovery_20260524`

## Finding

The materializer campaign runner carried explicit exact-auth calibration packet
paths into queue-owned feedback replans, but did not discover existing paired
`tac_result_review_packet_v1` CPU/CUDA result-review packets from durable
research/result roots. That left valid exact-auth calibration signal dependent
on manual CLI wiring after every materializer run.

## Landing

- Added `--exact-auth-calibration-packet-root` and
  `--exact-auth-calibration-packet-glob` to
  `tools/run_byte_shaving_materializer_campaign.py`.
- Feedback replans now scan configured roots for paired `contest_cpu` and
  `contest_cuda` review packets sharing archive SHA, archive bytes, sample
  count, and runtime content tree hash.
- When no explicit calibration root is supplied, feedback replans now
  opportunistically scan the run directory, run-local exact-eval handoff
  directories, and run-local result-review packet directories. Empty run-derived
  roots are nonblocking; malformed run-derived packet pairs block with the
  canonical validation reason.
- Discovered pairs are validated through
  `paired_exact_auth_calibration_observations_from_review_packets(...)` before
  the feedback child queue is allowed to run.
- Validated packet paths and candidate id are carried into the suggested action
  functional command and child queue telemetry inputs.
- Explicit `--exact-auth-calibration-packet` inputs now use the same validator
  as discovered packets before a feedback replan can run.
- `queue_feedback_replan_policy.v1` now records whether exact-auth calibration
  is actually usable for the feedback trust region and blocks malformed/unpaired
  packet metadata instead of counting it as calibrated signal.
- Added an acquisition regression proving paired exact-auth calibration can
  demote only the measured configuration/trust region; it cannot retire a
  family or create score, promotion, rank/kill, or dispatch authority.
- Configured roots with no usable pair fail closed with
  `exact_auth_calibration_packet_pair_not_found`; invalid pairs surface the
  canonical validation reason.

## Authority Contract

This remains planner calibration only. The generated request, feedback queue,
and child action-functional command preserve false-authority fields:
`score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`,
and `ready_for_exact_eval_dispatch=false`. Paired exact-auth measurements are
used as trust-region calibration for the inverse-steganalysis planner, not as a
score claim, promotion gate, rank/kill decision, or dispatch authorization.

## Verification

- `.venv/bin/ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_preserves_calibration_inputs src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_discovers_calibration_packet_pair src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_blocks_on_missing_calibration_pair src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_blocks_on_invalid_calibration_pair -q`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_discovers_run_derived_packet_pair src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_treats_empty_run_root_as_optional src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_discovers_calibration_packet_pair src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_blocks_on_missing_calibration_pair src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_blocks_on_invalid_calibration_pair -q`
- `.venv/bin/ruff check src/comma_lab/scheduler/queue_feedback_replan_policy.py src/tac/tests/test_queue_feedback_replan_policy.py tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_preserves_calibration_inputs src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_discovers_run_derived_packet_pair src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_treats_empty_run_root_as_optional src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_discovers_calibration_packet_pair src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_blocks_on_missing_calibration_pair src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_blocks_on_invalid_calibration_pair -q`
- `.venv/bin/ruff check src/comma_lab/scheduler/queue_feedback_replan_policy.py src/tac/tests/test_queue_feedback_replan_policy.py tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_acquisition.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_inverse_steganalysis_acquisition.py::test_paired_exact_auth_calibration_demotes_regressed_measured_config src/tac/tests/test_inverse_steganalysis_acquisition.py::test_paired_exact_auth_calibration_requires_shared_archive_custody src/tac/tests/test_inverse_steganalysis_acquisition.py::test_paired_exact_auth_calibration_refuses_family_retirement_authority src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_preserves_calibration_inputs src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_discovers_run_derived_packet_pair src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_treats_empty_run_root_as_optional src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_discovers_calibration_packet_pair src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_blocks_on_missing_calibration_pair src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_blocks_on_invalid_calibration_pair -q`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_queue_feedback_replan_policy.py -q`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_inverse_steganalysis_action_functional_cli.py::test_cli_accepts_paired_exact_auth_calibration_packets src/tac/tests/test_inverse_steganalysis_acquisition.py::test_paired_exact_auth_calibration_demotes_regressed_measured_config src/tac/tests/test_inverse_steganalysis_acquisition.py::test_paired_exact_auth_calibration_requires_shared_archive_custody -q`

## Next

The next high-EV integration is to generalize beyond one paired packet into a
bounded calibration portfolio and let continuation policy reason about how many
calibration anchors are enough to resume local replanning.
