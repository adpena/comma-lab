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

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py src/tac/tests/test_frontier_rate_attack_bootstrap.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`
  - Result: 28 passed.

## Next Integration

The next hardening step is to let the acquisition policy consume these
activation plans directly. Frozen repair allocators should enqueue targeted
component-response harvests, receiver-closed byte-credit materializers, or
exact-auth handoff requests from the plan instead of relying on operator
interpretation of queue metadata.
