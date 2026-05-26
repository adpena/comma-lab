# Codex Findings: Autonomous Many-Op Chain Optimizer

UTC: 2026-05-26T13:04:44Z

## Finding

The frontier-rate feedback loop had strong leaf and mid-level artifacts:
operation portfolio rows, materializer bridge rows, targeted component
correction requests, chain work orders, and receiver/materializer handoff rows.
Those artifacts were individually useful, but the normal refresh/cycle outputs
still did not expose a single queue-owned control surface that composed:

- drop-many and pair/frame/batch distortion debt;
- receiver-closed rate credit from byte-saving materializers;
- SegNet/PoseNet component repair budget;
- packet/archive/tensor/byte-range/inverse-scorer target coverage;
- receiver runtime proof closure.

That made the system too leaf-like: every materializer row looked locally
important, while no first-class artifact decided which many-op campaign should
run next.

## Landing

Added `frontier_rate_attack_autonomous_chain_optimization.v1` in
`src/comma_lab/scheduler/frontier_rate_attack_feedback.py`.

The new builder consumes:

- `frontier_rate_attack_operation_portfolio.v1`
- `frontier_rate_attack_operation_materializer_bridge.v1`
- optional
  `frontier_rate_attack_targeted_component_correction_chain_materializer_handoff.v1`

It emits campaign rows such as:

- `global_many_op_rate_distortion_receiver_campaign`
- `receiver_closed_budget_reinvestment_campaign`
- `portfolio_materializer_context_closure_campaign`
- `segnet_posenet_geometry_drop_many_campaign`

Each row carries target classes, operation levels, correction families,
receiver-closure requirements, scheduler actions, and a
`frontier_rate_attack_repair_budget_waterfill_plan.v1` block. The waterfill
plan makes the intended allocator explicit: use receiver-closed rate credit to
repair SegNet/PoseNet distortion debt only where measured component marginal
return per byte is best.

## Wire-In

The artifact is now produced by the normal feedback refresh path and recomputed
after targeted component chain handoff is available in both:

- `src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py`
- `tools/build_frontier_rate_attack_feedback_refresh.py`

The cycle report now carries the optimizer in initial and post-harvest refresh
payloads, and the integration graph exposes:

- `autonomous_chain_optimization_to_queue_owned_many_op_plan`
- `many_op_plan_to_component_replay_and_exact_readiness_bridge`

The CLI result and refresh report now include an
`autonomous_chain_optimization_summary`, write
`autonomous_chain_optimization.json`, and expose an
`inspect_autonomous_chain_optimization` operator command.

## Authority And Contest-Compliance Boundary

The optimizer is planning signal only. It cannot claim score, promote, rank or
kill, dispatch exact eval, or spend correction budget. Parser-only and local
proofs remain insufficient. Budget spend and score claims remain blocked until
component replay, receiver consumption proof, and exact auth-axis evaluation
exist.

This is contest-compliant as queue planning infrastructure. Any concrete
candidate archive produced downstream must still pass the usual archive/runtime
custody, no-hidden-sidecar, no-scorer-at-inflate, full-frame replay, and
contest CPU/CUDA axis gates before it can be used as score authority.

## Verification

Passed:

- `ruff` on touched frontier feedback, cycle, CLI, and test files
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`

Focused regression coverage asserts that the many-op artifact:

- composes packet, archive, tensor, inverse-scorer, frame/pair/batch, and
  receiver targets;
- exposes SegNet/PoseNet repair waterfilling;
- remains false-authority;
- is written by the normal CLI and cycle artifact writers.
