# Codex Findings: Scorer-Region Selector Chain + PR95 MLX Queue Custody

Date: 2026-05-28T01:35Z
Author: Codex

## Landed Evidence

- Built `scorer_region_selector_chain_current_20260527Tlocal` as a strict
  `experiment_queue.v1` chain: P18/P19 context -> P11 selector-context recode
  -> P15 archive repack -> composed report.
- Current local survivor is P11-only: `fec10_adaptive_blend`, 16 bytes saved
  versus source, P15 repack added 0 bytes of further savings.
- P18/P19 are now explicit machine-readable blockers:
  `p19_posenet_null_pairs_missing` and `p18_segnet_region_masks_missing`.
- Built and executed `pr95_mlx_stage8_source_video_queue_20260527Tlocal` as a
  queue-owned Stage 8 PR95/HNeRV MLX source-video control arm using
  RGB+YUV6 timing loss, runtime-consumption proof, PyTorch export parity, and
  byte-closed archive output.

## Bugs Extincted

- PR95 MLX artifacts now emit the full local-training false-authority field
  set as explicit `false`, including reproduction and score-eligibility fields.
- PR95 runtime-consumption proof reruns are now guarded by
  `--allow-overwrite --expected-existing-sha256`, so queue rewind does not
  require deleting artifacts or silently trampling prior output.
- PR95 archive export, manifest, and proof now carry generic archive/proof
  custody fields the queue verifier can hash-check.
- Embedded runtime proof summaries are sanitized so the queue does not parse a
  candidate archive inside a proof as a second proof record.

## Next Operator-Routable Work

1. Materialize the missing P19 PoseNet-null subset and P18 SegNet-region masks,
   then rerun the same P18/P19 -> P11 -> P15 queue chain.
2. Promote those upstream artifacts into receiver-closed distortion-budget
   materializers, then use MLX acquisition with CPU spot checks.
3. Move PR95/HNeRV MLX from one-step source-video control smoke to bounded
   Stage 1/5/8 ladder execution, then longer local MLX training only after
   source-video/full-scorer blockers are reduced.
4. Reuse the PR95 custody fixes for HNeRV variants, BoostNeRV bolt-ons,
   NeRV-family, and non-NeRV local training queues.
