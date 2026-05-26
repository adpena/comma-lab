# Codex Findings - MLX Repair Materializer Idempotent Closure

UTC: 2026-05-26T18:33:49Z

## Scope

Closed the queue-owned materializer execution path for the
`frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z` MLX repair
dynamics lane, with rerunnable materializer commands, DFL1 shell parity
overwrite safety, static-runtime submission closure, and chain-manifest harvest
filtering.

## Concrete Evidence

- Queue:
  `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_chain_materializer_execution_queue.json`
- Status artifact:
  `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_chain_materializer_execution_status_after_idempotent_closure.json`
- Result: 11/11 queue steps succeeded.
- DFL1 parity proof:
  `experiments/results/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/frontier_targeted_component_correction_chain_materializers/targeted_component_chain_materializers/targeted_component_chain_targeted_component_materialization_packet_member_merge_cbe7d79124ba_001/renderer_payload_dfl1_v1/exact_eval_handoff/renderer_payload_dfl1_shell_parity/shell_inflate_parity.json`
- Packet header-elide exact handoff:
  `experiments/results/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/frontier_targeted_component_correction_chain_materializers/targeted_component_chain_materializers/targeted_component_chain_targeted_component_materialization_packet_member_merge_cbe7d79124ba_001/packet_member_zip_header_elide_v1/exact_eval_handoff/dispatch_plan.json`

## What Changed

- Queue-owned materializer execution now adds `--allow-overwrite` and guarded
  `--expected-existing-*sha256` flags when prior output artifacts already exist.
  This makes reruns idempotent without permitting silent artifact replacement.
- DFL1 shell parity followups now pass `--overwrite`; the parity tool only
  removes an existing output directory when shell-parity marker files are
  present and refuses dangerous roots.
- Static archive transforms can close submission runtime packets using the
  source runtime when no generated adapter path exists, even if an earlier
  verifier carried a broad `runtime_adapter_ready=true` marker.
- Materializer harvest now filters work-queue sibling rows when an explicit
  `--chain-manifest` is supplied, preventing sibling candidate leakage into a
  single chain handoff.

## Contest Compliance

This remains false-authority local/MLX repair plumbing. The queue produced
materialized candidates, runtime-consumption artifacts, shell parity evidence,
submission closure packets, exact-readiness bridge reports, and dry-run dispatch
plans. It did not claim a score, promote a rank, execute exact CUDA/CPU auth
eval, or give the receiver any optimization authority.

## Repair-Dynamics Signal To Preserve

The operator-provided empirical canonical `K=16` palette from the live
`6bae0201` archive is a repair prior, not a score claim:

`[none, frame0_blue_chroma_amp_1, frame0_blue_chroma_amp_3,
frame0_luma_bias_{+1,-1,-2,-4}, frame0_rgb_bias_{m2_p1_p1,m4_p2_p2,p0_m1_p1,p0_m2_p2,p0_p1_m1,p0_p2_m2,p2_m1_m1,p4_m2_m2},
frame0_roll_dx+0_dy+1]`

The important structural observation is that 15/16 modes are frame-0 operators,
one mode is identity, and zero modes are frame-1 operators. The next repair
waterfill/acquisition layer should therefore treat frame asymmetry as a measured
prior to test, not as an assumed invariant: evaluate whether frame-0-only
operators are genuinely optimal under SegNet/PoseNet response, whether frame-1
operators are absent because they are harmful, redundant, or simply unsearched,
and whether stacked frame-0 repair composes with packet/rate operators without
distortion rebound.

## Next Integration Target

Feed the drained materializer evidence plus the K=16 frame-asymmetric palette
prior into the repair-budget waterfill queue as typed marginal actions:
`delta_bytes`, `delta_segnet`, `delta_posenet`, receiver proof, interaction key,
frame scope, and authority blockers. MLX can rank and batch the local
acquisition surface, but exact CPU/CUDA auth eval remains the only promotional
authority.
