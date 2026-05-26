# Codex Findings: Autonomous Chain Queue Actuation

UTC: 2026-05-26T13:24:03Z

## Verdict

The frontier-rate optimizer was still one level too advisory: it selected broad
many-operator chains, but the normal operator surface could only inspect the
selected-chain JSON. This landing turns selected autonomous chains into
queue-owned local work orders that can validate and bounded-run concrete child
queues while preserving false authority.

## What Changed

- Added `frontier_rate_attack_autonomous_chain_work_order.v1` as the typed
  encoder/receiver work contract for a selected many-op chain.
- Added `frontier_rate_attack_autonomous_chain_optimization_queue_metadata.v1`
  and a parent `experiment_queue.v1` builder that emits chain work orders, then
  validates/runs concrete local child queues with bounded `run-worker` limits.
- Wrote the chain materializer handoff's embedded work queue as a standalone
  artifact so it is not orphaned inside a larger handoff JSON.
- Built materializer execution queues only when materializer work rows are
  actually executable; blocked materializer rows remain advisory with explicit
  blockers rather than pretending to be runnable.
- Added operator commands for the autonomous chain queue in both the refresh CLI
  and cycle writer.
- Hardened the parent queue after sidecar review: unresolved non-advisory child
  queue artifacts freeze the experiment instead of allowing a hollow queued row,
  and emitted work orders now carry child-queue binding/missing-key summaries.
- Added a post-receiver-repair refresh step for the targeted-chain materializer
  path when receiver context repair is required before a materializer execution
  queue can exist.

## Pipeline Boundary

Repair and the final rate attack are encoder-side responsibilities:

- encoder planner: choose many-op chains across byte, archive, tensor, pixel,
  region, boundary, frame, pair, batch, and scorer axes;
- encoder materializer/archive builder: apply byte-saving transforms and emit the
  byte-closed packet;
- encoder repair allocator: spend freed bytes only where measured SegNet/PoseNet
  marginal repair is worthwhile;
- receiver/inflate runtime: deterministically consume the transformed packet;
- exact eval: measure on contest CPU/CUDA axis and feed results back.

The receiver is not an optimizer and does not get score authority.

## False-Authority Guardrails

- All new work orders and queue metadata carry false authority.
- Parser/local proof still cannot become score, promotion, rank/kill, or dispatch
  authority.
- Advisory actions for waterfill fitting and exact-readiness replay remain
  source-artifact references until concrete worker artifacts exist.
- Child queue execution is local-only and bounded; exact auth eval remains a
  downstream measured gate.
- A parent autonomous queue row is runnable only when every non-advisory local
  action is bound to a concrete child queue artifact. Otherwise it is frozen
  with explicit actuation blockers.

## Verification

- `ruff` on touched feedback, cycle, tool, work-order CLI, and tests: passed.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 32 passed.
- `tools/lane_maturity.py validate`: 1396 lane(s) validated cleanly.
- Review tracker policy: clean on touched Python surfaces.

## Next

The next high-EV tranche is to emit the concrete response-harvest/waterfill
worker so the repair allocator stops being advisory, then feed exact-axis
component measurements into the same parent autonomous queue.
