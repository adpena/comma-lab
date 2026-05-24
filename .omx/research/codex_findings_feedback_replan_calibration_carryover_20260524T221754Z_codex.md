# Codex Findings: Feedback Replan Calibration Carryover

Timestamp UTC: 2026-05-24T22:17:54Z

## Finding

`tools/run_byte_shaving_materializer_campaign.py` could build an initial inverse
steganalysis action functional with explicit observations and paired exact-auth
calibration packets, but its queue-owned feedback replan command only carried
the campaign plan plus queue-performance telemetry. That meant a bounded local
feedback loop could accidentally drop the strongest measured calibration signal
between iterations.

## Landing

- The materializer campaign feedback replan command now preserves explicit
  `--observation` inputs.
- It also preserves `--exact-auth-calibration-packet` inputs and
  `--exact-auth-calibration-candidate-id`.
- The feedback replan request records the observation paths, calibration packet
  paths, and calibration candidate id as planner metadata.
- The paused feedback child queue now records those observation/calibration
  inputs in its telemetry input artifact list.

## Authority Boundary

This carryover remains planner-only. Exact-auth calibration packets are consumed
by `tools/build_inverse_steganalysis_action_functional.py` as false-authority
trust-region updates. They do not grant score, promotion, rank/kill, paid
dispatch, or exact-eval authority. The feedback child queue remains paused,
local CPU only, and validated against forbidden dispatch flags.

## Verification

- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_replan_preserves_calibration_inputs -q`
- `.venv/bin/python -m pytest src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_action_functional_cli.py::test_cli_accepts_paired_exact_auth_calibration_packets src/tac/tests/test_inverse_steganalysis_acquisition.py::test_paired_exact_auth_calibration_demotes_regressed_measured_config -q`

## Remaining Gap

This patch preserves measured calibration through local replan iterations when
those packets already exist. It does not yet harvest completed remote
contest-CPU/contest-CUDA eval artifacts into paired calibration packets
automatically; that should be the next loop-closure patch.
