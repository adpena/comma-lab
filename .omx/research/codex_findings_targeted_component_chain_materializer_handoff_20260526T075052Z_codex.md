# Targeted Component Chain Materializer Handoff

Generated: 2026-05-26T07:50:52Z

## Verdict

The targeted component-correction loop is now above the single-materializer
leaf level. Local paired CPU/MLX correction responses compile into grouped
materialization requests, then into staged operation-chain work orders, then
into typed materializer backlog/context/work-queue rows for every registered
chain target.

False authority remains intact: no score, promotion, rank/kill, GPU launch, or
exact-eval dispatch authority is granted by these artifacts.

## Landed Shape

- `build_frontier_targeted_component_correction_chain_work_orders(...)` converts
  grouped correction requests into a four-stage operation chain:
  scorer-sensitive operation selection, receiver-consumed correction synthesis,
  payload grammar/entropy materialization, and component-guarded budget replay.
- `build_frontier_targeted_component_correction_chain_materializer_handoff(...)`
  converts registered chain targets into the existing final-byte materializer
  backlog, context compiler, and materializer work queue surfaces.
- The harvest CLI can now emit response harvest, materialization requests,
  materialization queue, operation-chain work orders, operation-chain queue, and
  chain materializer handoff in one queue-owned flow.
- The refresh CLI now writes the handoff artifact and exposes an operator
  inspection command; its targeted chain queue command uses a wide bounded step
  count instead of stopping at the two planning steps.

## Live Artifact Evidence

Live directory:
`.omx/research/frontier_rate_attack_feedback_refresh_20260526T073640Z_reference_mlx_cache_reuse/`

Current generated targeted chain:

- accepted local acquisition rows: 5
- grouped materialization request rows: 1
- operation-chain work orders: 1
- operation-chain queue experiments: 1
- registered materializer targets: 6
- unregistered explicit scorer/component targets: 8
- materializer work rows: 6
- executable materializer work rows: 1

Registered handoff targets:

- `packet_member_merge_v1`
- `packet_member_zip_header_elide_v1`
- `byte_range_entropy_recode_v1`
- `tensor_quantize_v1`
- `tensor_prune_v1`
- `tensor_shared_codebook_v1`

Explicit non-registered chain targets:

- `drop_within_selected_set_masked_boundary`
- `full_video_batch_residual_budget_reallocation`
- `inverse_scorer_cell_basis_expansion`
- `pose_stable_pair_frame_motion_correction`
- `segnet_posenet_waterfill_region_repair`
- `segnet_component_response`
- `posenet_component_response`
- `full_video_lagrangian_response`

The current operation-chain queue validates and emits the stage plan plus
byte-range stage-input artifacts. It deliberately blocks byte-range execution
because target-bound context is missing `schema_manifest`, `beam_probe_reports`,
and `member_name`; the default PR103 context is explicitly disabled for this
target-bound chain.

## Executed Local Proof

The handoff exposed one executable materializer row:
`packet_member_zip_header_elide_v1`.

It was executed locally against the receiver-closed packet-member-merge
candidate. Result:

- source bytes: 345544
- candidate bytes: 345544
- saved bytes: 0
- receiver contract satisfied: true
- byte-closed candidate emitted: true
- blocker: `candidate_not_rate_positive`
- score authority: false
- exact dispatch authority: false

This is useful negative signal: header elision is saturated after the merge
candidate, so budget should not be spent there without a different context.

## Remaining Blocker

The next high-EV engineering gap is not another single materializer. It is the
receiver-consumed correction synthesis layer for the unregistered scorer and
component targets: masked drop-many, inverse-scorer basis expansion, pose-stable
pair/frame motion correction, SegNet/PoseNet waterfill, and component replay.
Those need concrete receiver/runtime adapters and component-response evaluators
before the saved rate budget can be spent on targeted corrections.

## Follow-up Binding Guard

Follow-up artifact:
`.omx/research/frontier_rate_attack_feedback_refresh_20260526T082300Z_targeted_chain_binding_guard_runtime/`

The targeted chain now carries explicit candidate/source receiver-runtime
binding from response harvest through materialization requests and operation
chain work orders. The byte-range stage input builder recognizes target-bound
targeted-component chains and disables the legacy PR103 default context instead
of silently executing against the wrong archive/runtime. The resulting
byte-range stage inputs are fail-closed with:

- `byte_range_stage_default_pr103_context_disabled_for_target_bound_chain`
- `byte_range_stage_missing:schema_manifest`
- `byte_range_stage_missing:beam_probe_reports`
- `byte_range_stage_missing:member_name`

The bound source is `submissions/robust_current/archive_correct.zip` with
`submissions/robust_current/inflate.sh`; the bound candidate is the
receiver-closed packet-member-merge archive under
`experiments/results/frontier_final_rate_attack/frontier_packet_merge_receiver_closed_20260526T015551Z/`.
This converts the prior silent wrong-context risk into an explicit readiness
blocker that cannot claim score, promote, rank/kill, or dispatch exact eval.
