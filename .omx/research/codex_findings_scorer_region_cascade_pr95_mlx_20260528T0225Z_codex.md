# Codex Findings - Scorer-Region Cascade And PR95 MLX Control Arm - 2026-05-28

## Scope

Queue-owned Cascade C materialization plus bounded PR95/HNeRV MLX control-arm
training smoke. All artifacts are local/proxy evidence only and carry
fail-closed authority fields.

## Landed And Executed

- Receiver-patch queue:
  `.omx/research/scorer_region_selector_chain_receiver_patch2_20260528Tlocal/queue.json`
  - 8/8 steps succeeded.
  - includes P19/P18 materialization, P11 selector recode, P15 repack,
    receiver-budget plan, and frame-1 region runtime patch materialization.
- Frame-1 region runtime patch:
  `/Volumes/VertigoDataTier/experiments/results/scorer_region_selector_chain_receiver_patch2_20260528Tlocal/frame1_region_waterfill_runtime_patch/frame1_region_waterfill_runtime_patch.json`
  - 12 P19/P18-selected pairs, 1 region per pair, RGB delta `[-1, -1, -1]`.
  - patched runtime compiles.
  - synthetic import/change probe changes only frame 1, not frame 0:
    `/Volumes/VertigoDataTier/experiments/results/scorer_region_selector_chain_receiver_patch2_20260528Tlocal/frame1_region_waterfill_runtime_patch/runtime_patch_synthetic_change_probe.json`
- P19 PoseNet-null pair artifact:
  `/Volumes/VertigoDataTier/experiments/results/scorer_region_selector_chain_p18p19_materialized_20260528Tlocal/p19_posenet_null_pairs.json`
  - 134 detected null-candidate pairs; 60 selected for budget planning.
- P18 SegNet-region waterfill artifact:
  `/Volumes/VertigoDataTier/experiments/results/scorer_region_selector_chain_p18p19_materialized_20260528Tlocal/p18_segnet_region_waterfill.json`
  - 60 selected pairs, top 4 regions per pair.
- Queue-owned P11/P15 chain:
  `.omx/research/scorer_region_selector_chain_p18p19_materialized_20260528Tlocal/queue.json`
  - 7/7 steps succeeded.
  - selected survivor: P11 selector context recode.
  - selected archive: `/Volumes/VertigoDataTier/experiments/results/scorer_region_selector_chain_p18p19_materialized_20260528Tlocal/p11_selector_context_recode/submission_dir/archive.zip`
  - local rate delta: 16 bytes saved versus source.
- Receiver-closed distortion-budget plan:
  `/Volumes/VertigoDataTier/experiments/results/scorer_region_selector_chain_p18p19_materialized_20260528Tlocal/receiver_closed_distortion_budget_attack_plan.json`
  - rate credit estimate: 0.000010653743249954742 score units.
  - still blocked by runtime patch, component spot check, and exact auth eval.

## PR95/HNeRV MLX Control Arm

Bounded stage-8 source-video smoke:
`/Volumes/VertigoDataTier/experiments/results/pr95_mlx_stage8_source_video_bounded16_20260528Tlocal/run_summary.json`

- 16 MLX steps, `pr95_stage8_muon_adamw_mlx`, source-video RGB+YUV6 proxy loss.
- 0.02966569275031361 seconds/step.
- PR95 public archive export emitted:
  `/Volumes/VertigoDataTier/experiments/results/pr95_mlx_stage8_source_video_bounded16_20260528Tlocal/pr95_public_archive.zip`
  - 230339 bytes, SHA-256 `06ed436b79ce62b78e4cd953bbb74639956113215df0e7a3b6350c05c61bfd99`.
- Runtime consumption proof passed.
- PyTorch export forward parity passed:
  - max abs `3.0517578125e-05`
  - mean abs `3.425766635700711e-06`.

## Current Blockers

- Cascade C now has materialized P19/P18/P11/P15 artifacts and a first
  frame-1 receiver patch, but still needs full runtime-consumption/output-change
  proof and component scoring before exact auth eval.
- PR95 MLX now proves source-video training plumbing, archive export,
  runtime consumption, and PyTorch forward parity, but still lacks full scorer
  loss wiring, source-faithful stage schedule closure, QAT/resume parity, and
  contest auth eval.

## Next Build Step

Materialize `frame1_region_waterfill_runtime_patch_v1` from the distortion
budget plan, run local MLX/CPU component spot checks, then exact-anchor only if
the eureka gate clears. In parallel, extend PR95 MLX from bounded smoke to a
queue-owned longer source-video run after scorer-loss wiring closes.
