# Targeted Drop-Many Chain Queue Wiring

## Finding

The targeted correction chain is no longer limited to a byte-range/materializer
handoff.  The operation-chain queue now emits a
`frontier_rate_attack_targeted_drop_many_stage_inputs.v1` packet and, when the
selector Pareto plus pair-frame geometry lattice are available, executes the
existing decoder-q pair-set acquisition planner from the same queue.

## Live Artifact

- Queue:
  `.omx/research/frontier_rate_attack_feedback_refresh_20260526T073640Z_reference_mlx_cache_reuse/targeted_component_correction_operation_chain_queue.json`
- Stage input:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/frontier_operation_chain_compiler/frontier_rate_attack_reference_mlx_cache_reuse_20260526t073640z_component_operation_chain/targeted_component_chain_targeted_component_materialization_packet_member_merge_cbe7d79124ba_001/targeted_drop_many_stage_inputs.json`
- Pair-set acquisition:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/frontier_operation_chain_compiler/frontier_rate_attack_reference_mlx_cache_reuse_20260526t073640z_component_operation_chain/targeted_component_chain_targeted_component_materialization_packet_member_merge_cbe7d79124ba_001/targeted_drop_many_pairset/targeted_drop_many_pairset_acquisition.json`

Follow-up durable artifact:
`.omx/research/frontier_rate_attack_feedback_refresh_20260526T083300Z_targeted_chain_member_inference/`

This follow-up also records byte-range member-name inference for the same
target-bound packet-merge chain. The byte-range stage now derives
`__packet_member_merge_v1.bin` from the strict single-member candidate archive
and leaves only the real missing PR110-specific schema/probe blockers:
`byte_range_stage_missing:schema_manifest` and
`byte_range_stage_missing:beam_probe_reports`.

## Result

Bounded local queue execution succeeded for the four current targeted-chain
steps:

1. emit operation-chain stage plan
2. emit byte-range stage inputs
3. emit targeted drop-many stage inputs
4. run targeted drop-many pair-set acquisition

The pair-set acquisition artifact contains 253 local planning candidates:
67 drop-many candidates, 6 pair-frame geometry candidates, 127 drop-two
candidates, 31 drop-one candidates, 12 prefix candidates, and 10 diversity
candidates.

## Authority

All artifacts remain local planning/proxy signal only:
`score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`,
and `ready_for_exact_eval_dispatch=false`.

The freed receiver-closed rate budget is available only as a bounded local
planning credit for targeted correction search.  It still requires paired
candidate/reference component response, total Lagrangian improvement, receiver
runtime proof, and exact-axis auth eval before any spend, score, dispatch, or
promotion claim.

## Remaining Gap

The queue now fans into many pair/frame/batch correction starts, but it still
does not materialize those starts into byte-closed candidate archives or run
component replay automatically.  Next work should consume the generated
pair-set acquisition artifact into the DQS1/local-first materialization queue
and then route only measured negative Lagrangian deltas back into targeted
component correction.
