# Codex Findings: Targeted Component Materialization Requests

UTC: 2026-05-26T05:22:08Z

## Verdict

The previous bridge harvested targeted SegNet/PoseNet correction responses, but
negative local Lagrangian rows still stopped at advisory response harvest. This
landing adds the missing queue-visible next stage: accepted response rows now
compile into grouped candidate materialization requests that preserve the
multi-operator basis across pixel, region, boundary, frame, pair, batch, and
full-video levels.

This is deliberately not score, promotion, rank/kill, budget-spend, or dispatch
authority. The new rows and queue remain local-only until a receiver-consumed
targeted correction materializer, full-frame inflate parity, exact-axis
component response, and exact auth eval evidence exist.

## What changed

- Added `frontier_rate_attack_targeted_component_correction_materialization_request_row.v1`.
- Added `frontier_rate_attack_targeted_component_correction_materialization_requests.v1`.
- Added `frontier_rate_attack_targeted_component_correction_materialization_queue_metadata.v1`.
- Preserved operation levels, targeted dimensions, command hints, and wire-in
  hooks through work orders and response-harvest rows.
- Added grouped request compilation from negative local Lagrangian response rows.
- Added a local materialization-request queue that emits request artifacts under
  false authority.
- Added `tools/build_frontier_targeted_component_correction_materialization_request.py`.
- Wired feedback refresh output to write
  `targeted_component_correction_materialization_requests.json` and, when
  accepted rows exist,
  `targeted_component_correction_materialization_queue.json`.

## Live artifact

Path:
`.omx/research/frontier_rate_attack_feedback_refresh_20260526T_materialization_requests_v2/`

Summary:

- materializer feedback payloads: 2
- receiver-closed saved bytes available for targeted correction budget: 258
- targeted correction acquisition rows: 5
- targeted response harvest rows: 5
- accepted local negative Lagrangian response rows: 0
- materialization request rows emitted: 0
- ready for budget spend: 0

The absence of request rows in the live artifact is the correct fail-closed
state: local CPU/MLX component response artifacts still need to run and produce
negative measured Lagrangian deltas before the request queue becomes active.

## Wire-in hooks

- Sensitivity-map contribution: ACTIVE through the carried
  `operation_levels`, `targeted_dimensions`, local component deltas, and
  correction-family basis rows.
- Pareto constraint: ACTIVE as the request objective keeps
  `delta_segnet + delta_posenet + lambda * delta_bytes` explicit and blocks
  budget spend until exact-axis component evidence exists.
- Bit allocator hook: ACTIVE by grouping accepted correction families by
  candidate and preserving receiver-closed saved-byte credit.
- Cathedral/autopilot dispatch hook: ACTIVE as a normalized queue artifact;
  cloud resources remain zero.
- Continual-learning posterior: ACTIVE through typed response-harvest and
  materialization-request schemas.
- Probe disambiguator: ACTIVE through separate local response harvest,
  materialization request, receiver materializer, full-frame parity, and exact
  auth gates.

## Verification

- `ruff` on touched scheduler/tool/test files: clean.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 24 passed.
- Live refresh smoke wrote the artifact directory above.
- `experiment_queue.py validate` passed for DQS1 follow-up, receiver repair,
  and targeted component correction queues from the live artifact.

Sidecar note: an xhigh reviewer was requested, but the agent pool was already at
the thread limit. I continued the critical path locally and preserved the review
risks as explicit tests and blockers rather than waiting.
