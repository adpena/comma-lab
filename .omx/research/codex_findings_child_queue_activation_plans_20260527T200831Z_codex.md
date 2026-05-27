# Codex Findings: Child Queue Activation Plans

Generated: 2026-05-27T20:08:31Z

## Finding

The final-rate autoloop could classify a child queue as frozen, paused, or
disabled, but that still left the next action implicit. For repair waterfill
queues, the meaningful state is not simply "frozen"; it is the typed set of
missing response, receiver-closed byte-credit, exact-auth, and queue-status
evidence that would make the frozen encoder-side allocator executable.

## Fix

- `frontier_final_rate_attack_autoloop` now emits a
  `frontier_final_rate_attack_child_queue_activation_plan.v1` artifact beside
  each frozen/paused/disabled child-queue observer revalidation.
- The activation plan records blocked experiments, status, tags, lane id,
  activation blockers, blocker-to-evidence actions, step dependencies,
  telemetry input/output artifacts, and postcondition artifact refs.
- The plan carries the mathematical contract explicitly:
  minimize score delta under rate/runtime constraints using bytes delta,
  SegNet/PoseNet response, interaction terms, entropy position, and receiver
  runtime identity as state variables.
- The artifact is strictly false-authority: no score claim, no promotion, no
  rank/kill, and no paid dispatch authority.
- Child queue selection now prefers runnable queue definitions before frozen,
  paused, or disabled queue definitions. With a bounded child-queue limit, the
  loop executes available receiver/materializer repair work first and still
  writes activation plans for deferred frozen queues so no blocker signal is
  lost.
- Activation plans now also compile into
  `repair_campaign_blocked_learning_signal_report.v1` rows. The repair-campaign
  blocked-learning CLI accepts `--activation-plan`, so frozen child-queue
  blockers become posterior-consumable acquisition signals instead of
  operator-only prose.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py src/tac/tests/test_frontier_rate_attack_bootstrap.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`
  - Result: 30 passed in 4.39s.
- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py src/tac/optimization/repair_campaign_learning_signal.py tools/build_repair_campaign_blocked_learning_signals.py src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_repair_campaign_score_queue.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_repair_campaign_score_queue.py -q`
  - Result: 5 passed in 4.19s.
- Bounded queue-owned smoke:
  `.venv/bin/python tools/build_frontier_final_rate_attack_queue.py --output-dir .omx/research/frontier_final_rate_attack_activation_plan_smoke_20260527T2230Z --results-root /Volumes/VertigoDataTier/pact/frontier_final_rate_attack_activation_plan_smoke_20260527T2230Z --queue-id frontier_final_rate_attack_activation_plan_smoke_20260527T2230Z --max-steps 8 --max-parallel 2 --execute --post-execute-feedback-refresh --execute-post-feedback-queues --post-feedback-child-queue-limit 2 --post-feedback-child-queue-max-steps 1 --post-feedback-child-queue-max-parallel 1 --post-execute-feedback-candidate-limit 16 --post-execute-feedback-max-files-per-root 64`
  - Result: `failed_command_count=0`, feedback refresh return code `0`.
  - Executed runnable child queues: `operation_chain_compiler_queue` and
    `receiver_repair_queue`; both made progress and both observer
    revalidations were valid.
  - Deferred activation plans preserved four frozen queues:
    `autonomous_chain_optimization_queue`, `repair_campaign_score_queue`,
    `repair_budget_waterfill_queue`, and
    `targeted_component_correction_queue`.
  - Footprint: 2.6M under `.omx/research`, 1.3M under VertigoDataTier.

## Next Integration

The next hardening step is to let the acquisition policy consume these
activation plans directly. Frozen repair allocators should enqueue targeted
component-response harvests, receiver-closed byte-credit materializers, or
exact-auth handoff requests from the plan instead of relying on operator
interpretation of queue metadata.
