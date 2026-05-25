# Codex Findings: Grouped PacketIR Materializer Queue Bridge

UTC: 2026-05-25T15:53:11Z

## Verdict

The materializer path is now less leaf/manual: PacketIR operation ordering is carried into final-byte contexts and materializer work rows, sibling archive-state materializers from the same PacketIR operation set are grouped into one queue-visible local proof chain, and the grouped runner rewrites each child command so step N feeds step N+1 through actual archive bytes.

This is still local proof-chain authority only. The grouped chain, PR95 MLX control profile briefing, and eureka/drop-many planning signals all preserve `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## What Changed

- `byte_shaving_campaign_queue.py` now emits `grouped_archive_state_materializer_request.v1` rows for executable same-PacketIR archive-state materializers.
- `tools/run_grouped_family_agnostic_materializer.py` executes a grouped request by chaining child materializer commands over the prior step's candidate archive and writing `grouped_archive_state_materializer_chain.v1`.
- Final-byte contexts and queue rows preserve `source_packet_ir_operation_indices` and `source_packet_ir_operation_indices_by_unit`, so operation ordering is not trapped in the planner.
- Context resolution now composes generic final-byte/PacketIR hints with row-specific executable materializer paths while preserving ambiguity blockers for competing source-unit contexts.
- `tools/operator_briefing.py` now surfaces PR95 MLX control profile health under Phase 6e and JSON `pr95_mlx_control_profiles`, with false-authority fields intact.

## Eureka / Distortion Signal Read

- Existing local eureka JSONs did not trigger exact-auth dispatch; they remain observe-only planning signals.
- Drop-two was better than the specific rank005 drop-one comparison (`r029_010_p0259_0376` projected 0.192029117 vs rank005 projected 0.192030283), but the current local set also has best drop-one rows near the same band. This means pair interaction cannot be inferred from inherited projected scores alone.
- The drop-many acquisition JSON generated 581 planning candidates, including 34 drop-many candidates, and attaches a rate-saved distortion repair budget to pairset candidates.
- The first pairwise-interaction build falsified its data-source premise: 100% artifact concentration from inherited `predicted_score_mean`, so the next valid build needs true paired CPU exact-eval anchors for drop-one/drop-two archive bytes before beam search can claim interaction geometry.
- SegNet/PoseNet-sensitive distortion probes are positive but advisory: per-class UNIWARD is partial, temporal Hinton KL is positive and plateaus around W=4..8, and per-segment UNIWARD improves over per-class but still misses the hard 0.5 threshold. The best next distortion surface is combined per-instance + multi-scale/boundary-aware masking, not a longer temporal window alone.

## Verification

- `ruff` clean on grouped materializer, scheduler, context, and operator briefing files.
- Focused grouped/request/briefing tests: 3 passed.
- Scheduler/context/operator briefing regression slice: 144 passed.
- `tools/lane_maturity.py validate`: 1341 lanes validated cleanly.
