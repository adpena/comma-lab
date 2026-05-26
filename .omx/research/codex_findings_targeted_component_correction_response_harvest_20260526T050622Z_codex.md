# Codex Findings: Targeted Component Correction Response Harvest

UTC: 2026-05-26T05:06:22Z

## Verdict

The receiver-closed rate-credit queue is no longer one-way. Targeted
SegNet/PoseNet correction work orders now declare the component-delta response
contract, and each queued local CPU/MLX correction probe has a final harvest
step that converts measured response fields into typed local-acquisition
Lagrangian rows.

This is still false-authority. A negative local Lagrangian can recommend the
next receiver-consumed correction candidate, but it cannot spend budget, claim
score, promote, rank/kill, or dispatch exact eval.

## Landed Engineering

- Added `frontier_rate_attack_targeted_component_correction_response_harvest.v1`
  and row schema support in `src/comma_lab/scheduler/frontier_rate_attack_feedback.py`.
- Added a response-harvest contract to targeted component correction work
  orders: `segnet_delta`, `posenet_delta`, and correction byte/rate delta.
- Added `tools/harvest_frontier_targeted_component_correction_response.py`.
- Wired the targeted component correction queue so every experiment ends with
  `harvest_targeted_component_correction_response`.
- Changed targeted correction work dirs from candidate-only to
  candidate/acquisition-specific paths, preventing sibling correction-family
  races over shared `work_order.json` / advisory artifacts.
- Wired the refresh CLI to write
  `targeted_component_correction_response_harvest.json` and an operator
  inspection command.

## Live Smoke

Artifact root:

`.omx/research/frontier_rate_attack_feedback_refresh_20260526T_component_response_harvest/`

Observed:

- `receiver_closed_saved_bytes_total`: 258
- targeted correction families queued: 5
- targeted correction queue experiments: 5
- targeted correction queue steps: 25
- response harvest rows: 5
- local acquisition recommended rows: 0
- ready-for-budget-spend rows: 0

The five response rows are blocked because the local CPU/MLX component response
steps have not been executed yet. That is the intended fail-closed state: the
signal is visible and typed instead of silently missing.

## Verification

- `ruff check` on touched scheduler/tool/test files: passed
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 23 passed
- `tools/experiment_queue.py validate` on generated targeted correction queue:
  valid, 5 experiments, 25 steps
- `tools/experiment_queue.py validate` on generated receiver repair queue:
  valid, 4 experiments, 4 steps
- `tools/experiment_queue.py validate` on generated DQS1 follow-up queue:
  valid, 6 experiments, 42 steps

## Remaining Gap

The next autonomy step is to run or simulate at least one correction-family
candidate that emits the declared response deltas, then let the harvest promote
negative local Lagrangian rows into receiver-consumed correction materialization
requests while preserving exact-auth dispatch blockers.
