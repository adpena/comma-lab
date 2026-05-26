# Codex Findings: Repair Waterfill Action-Functional Queue

UTC: 2026-05-26T13:50:38Z

## Verdict

The action-functional work already existed in the frontier-rate stack, but the
repair side was still not a queue-owned executable edge. This landing wires the
existing rate-budget preservation/action-functional surface into an autonomous
child queue that emits encoder-side repair waterfill work orders.

## What Changed

- Added `frontier_rate_attack_repair_budget_waterfill_work_order.v1`.
- Added `frontier_rate_attack_repair_budget_waterfill_queue_metadata.v1`.
- Added `tools/build_frontier_repair_budget_waterfill_work_order.py`.
- Added `build_frontier_repair_budget_waterfill_queue(...)`.
- Changed `fit_segnet_posenet_repair_waterfill_policy` from an advisory source
  reference into a concrete `repair_budget_waterfill_queue` child action.
- Wrote `repair_budget_waterfill_queue.json` from both refresh writers before
  the parent autonomous-chain queue is built, so parent queue actuation sees the
  repair allocator as a real child queue artifact.
- Added an explicit action-functional lineage block tying the repair queue to
  `rate_budget_preservation_plan`, `cumulative_rate_attack`, and the existing
  waterfill objective rather than creating a parallel abstraction.
- Added the explicit `frontier_rate_attack_operator_action_functional.v1`
  contract to the preservation plan: operator `T`, receiver `R_T`, SegNet /
  PoseNet / rate terms, composition law, and exact-readiness constraints.

## Authority Boundary

- The repair queue is local and encoder-side only.
- The receiver remains deterministic decode-only.
- The queue can propose allocation rows from measured local component response
  and receiver-closed byte credit, but `budget_spend_allowed=false`.
- Exact CPU/CUDA component replay and exact readiness are still required before
  any score, promotion, rank/kill, budget-spend, or dispatch authority.

## Verification

- `ruff` on touched frontier feedback/cycle/tool/test files passed.
- `py_compile` on touched frontier feedback/cycle/tool/test files passed.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 33 passed.
- `pytest src/tac/substrates/_shared/tests/test_long_training_canonical.py -q`:
  64 passed.

## Next

The next gap is to make the emitted repair-waterfill work order materialize a
receiver-consumed spent-budget candidate archive as a child of the preserved
rate-only floor archive, then feed that archive through exact-readiness and
component replay.
