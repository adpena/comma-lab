# Codex Findings: Grouped PacketIR Materializer Gap

Generated: 2026-05-25T15:16Z
Agent: Codex
Topic: automated final rate attack / grouped materializer execution

## Finding

The final-rate-attack stack has the right primitives, but the next missing
automation layer is grouped archive-state execution. PacketIR and inverse-action
operation sets can describe ordered multi-operation attacks, but the scheduler
currently lowers them into independent materializer rows. That loses interaction
state between operations and keeps the system closer to leaf byte shaving than
to queue-owned grouped attacks.

## Concrete Gap

Needed next:

1. A grouped PacketIR work row in the materializer queue: one
   `experiment_queue.v1` experiment with ordered steps from
   `chosen_operation_sequence`.
2. Chained context hydration: step `N` output archive becomes step `N+1` input
   archive, with deterministic output paths and per-step manifests.
3. A grouped result schema consumed by queue observer and frontier feedback,
   not only DQS1-specific eureka hints.
4. Receiver-hook resolution from cooperative receiver grammars into
   materializer readiness checks.
5. Strict false-authority until runtime consumption, receiver proof, same-runtime
   parity, and exact auth eval gates pass.

## Immediate Patch Target

Start in:

- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py`
- `src/comma_lab/scheduler/final_byte_operation_contexts.py`
- `src/comma_lab/scheduler/experiment_queue_observer.py`
- `src/comma_lab/scheduler/frontier_rate_attack_feedback.py`
- `src/tac/optimization/family_agnostic_materializers.py`

Tests should cover:

- PacketIR grouped queue lowering preserves ordered archive state
- missing archive/section/member/tensor manifests fail closed
- grouped result telemetry reseeds acquisition
- cooperative receiver hook metadata does not grant score authority

This is the architectural bridge from current leaf materializers to automated
inverse-steganalysis rate attacks.
