# Codex Findings: Inverse Steganalysis Layer Audit

Timestamp UTC: 2026-05-24T22:14:10Z

## Verdict

The repository is not merely doing manual leaf-level pixel or byte shaving. It
has real inverse-action machinery: MLX/scorer-response research rows, action
functional construction, water-fill portfolio search, PacketIR-style lowering,
materializer queues, parity probes, and exact-auth consumers. The main gap is
execution: the intelligence surface can rank richer action cells faster than
the materializer substrate can turn them into byte-closed candidate archives.

## Implemented Surfaces

- Inverse steganalysis/action functional:
  `src/tac/optimization/inverse_steganalysis_acquisition.py`
- Action-functional CLI:
  `tools/build_inverse_steganalysis_action_functional.py`
- Byte-shaving signal surfaces and campaign plans:
  `src/tac/optimization/byte_shaving_campaign.py`
- Inverse-scorer cell materialization/parity/exact-ready staging:
  `src/tac/optimization/inverse_scorer_cell_materializer.py`,
  `src/tac/optimization/inverse_scorer_cell_inflate_parity.py`,
  `src/tac/optimization/inverse_scorer_exact_eval_queue.py`
- Scheduler/materializer queue bridge and registry:
  `src/comma_lab/scheduler/byte_shaving_campaign_queue.py`,
  `src/comma_lab/scheduler/byte_shaving_materializer_registry.py`
- Exact-auth fail-closed consumer:
  `src/comma_lab/scheduler/materializer_exact_eval_consumer.py`

## Highest-EV Gaps

1. General materializers for non-DQS1 action cells remain insufficient,
   especially HNeRV/NeRV/boltons/packet-member/archive-section operations.
2. Context propagation from action functional to final materializer rows was the
   immediate active gap; this session landed PacketIR/backlog inline operation
   params to close the first slice.
3. Exact-auth harvest does not yet automatically retrain or reweight the next
   action functional.
4. MLX-selected operation sets are better at ranking candidates than producing
   final byte-closed archives.
5. PacketIR is not yet the canonical final-operation algebra across all
   substrate families.
6. Per-component scorer transparency is not yet calibrated deeply enough at
   boundary/region internals.
7. Infinite compression-time search exists as pieces, not yet as one always-on
   controller that keeps MLX/CPU/materializer/exact queues saturated.
8. Exact-ready authority is separated correctly, but not yet a dependency edge
   in every materializer DAG path.
9. Runtime and cost telemetry are stronger than score-delta telemetry.
10. Large scheduler/action modules increase invariant-break risk.

## Recommended Next Patches

1. Finish context propagation and make compiled operation selectors executable
   without duplicated artifact-map hints.
2. Add or harden one registry-backed non-DQS1 materializer for packet-member or
   archive-section recode against real champion artifacts.
3. Wire exact-auth observations into a feedback replan policy so exact results
   calibrate the next action surface.
4. Promote PacketIR operation sets as the canonical lowering boundary for final
   archive mutations.
5. Add a thin autonomy runner that delegates to existing modules: build action
   surface, compile materializer queue, run local materializers, parity-check,
   enqueue exact-ready packets, harvest, and replan.

## Authority Constraint

The recommended loop remains fail-closed: MLX/scorer-response/action rows may
choose local follow-up or exact-eval spend candidates, but every planner row
must stay `score_claim=false`, `promotion_eligible=false`, and
`rank_or_kill_eligible=false` until exact auth produces a byte-closed authority
artifact.
