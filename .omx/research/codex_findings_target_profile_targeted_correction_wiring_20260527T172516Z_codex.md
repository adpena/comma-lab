# Codex Findings: Target Profile Targeted-Correction Wiring

Date: 2026-05-27T17:25:16Z
Agent: Codex

## Finding

The final-rate attack target optimization profile was already produced by the
feedback refresh, and the repair-budget waterfill queue already consumed it in
some surfaces. The targeted component-correction queue did not carry that
profile through its root metadata, per-experiment metadata, or per-request
metadata. That left downstream targeted-correction and post-auxiliary consumers
able to operate without a durable distinction between contest-video overfit
optimization and reusable corpus/video optimization.

## Fix Landed

- `build_frontier_targeted_component_correction_queue(...)` now accepts
  `target_optimization_profile_metadata` and validates it with the false-authority
  guard before embedding it into queue, experiment, and correction-request
  metadata.
- The blocked empty-selection targeted-correction queue now preserves the same
  target profile instead of dropping it at the no-actionable-row boundary.
- `write_frontier_refresh_artifacts(...)` passes the refresh target profile into
  both targeted component-correction and repair-budget waterfill queues.
- `write_targeted_component_correction_post_auxiliary_artifacts(...)` preserves
  the target profile explicitly, or recovers it from the queue root metadata.
- Both feedback-cycle CLIs now forward the profile rather than relying on
  downstream defaults.

## Why This Matters

This closes an automation and continual-learning gap: grouped SegNet/PoseNet
repair, receiver-budget spending, MLX acquisition, and queue-owned follow-up now
remain tied to the declared optimization target. That lets future consumers
learn from contest-video overfit runs without silently training the wrong lesson
for other videos or corpora.

## Verification

- `.venv/bin/python -m py_compile src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py tools/build_frontier_rate_attack_feedback_refresh.py tools/run_frontier_rate_attack_feedback_cycle.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py tools/build_frontier_rate_attack_feedback_refresh.py tools/run_frontier_rate_attack_feedback_cycle.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py::test_frontier_feedback_compiler_discovers_materializers_and_refreshes_dqs1_queue src/tac/tests/test_frontier_rate_attack_feedback.py::test_empty_targeted_component_correction_queue_emits_blocked_harvest src/tac/tests/test_frontier_rate_attack_feedback.py::test_post_auxiliary_targeted_component_refresh_reharvests_into_chain src/tac/tests/test_frontier_rate_attack_feedback.py::test_frontier_feedback_cli_writes_valid_followup_queue -q`

## Remaining Work

Push the same profile-contract discipline into downstream materialization
request queues, operation-chain compiler stage plans, materializer handoff rows,
and exact-readiness handoff reports so every automated rate/distortion attack
artifact remains target-declared end to end.
