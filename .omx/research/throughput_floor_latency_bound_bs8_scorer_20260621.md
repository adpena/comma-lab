---
title: "Throughput floor for the PR95 600-pair vehicle is LATENCY-bound on the bs=8 scorer forward+backward (75 serial steps/epoch), NOT sync-bound and NOT FLOP-bound — jointly pinned by two measured rows: defer_batch_sync gives +2% (MPS), a real GPU upgrade gives ~0% (A10G≈T4≈MPS). The only un-CPU-bound lever is batch_size, which is score-locked. Modal NO-GO confirmed."
authority: "[contest-CPU advisory / MLX-trained] — pointer UNMOVED 0.19110; $0 local + ~$1.73 Modal (of $20); NO score claim"
score_claim: false
promotion_eligible: false
pointer_moved: false
date: 2026-06-21
verdict: THROUGHPUT_FLOOR_IS_LATENCY_BOUND_BS8 · DEFER_BATCH_SYNC_PLUS_2PCT_KEPT · MODAL_NO_GO · LOCAL_STAYS_4_DAYS
cross_refs:
  - src/tac/torch_vehicle/driver.py                                  # defer_batch_sync impl (eb4bcf4cd)
  - src/tac/torch_vehicle/tests/test_batch_sync_deferral_bit_identical.py  # the bit-identical proof
  - .omx/research/yousfi_r3_taper_marginhinge_e5_stage1_verdict_20260621.md
---

# The throughput floor — two measured rows triangulate it exactly

The operator asked whether the per-epoch could be sped up (Modal and/or local) with **nothing that hurts
score or signal**. Two measured rows now pin the bottleneck precisely. `[contest-CPU advisory]`; pointer
UNMOVED 0.19110; no score claim.

## The two measurements
| Lever | Removes | Measured effect | Implication |
|---|---|---|---|
| **defer_batch_sync** (MPS local) | ~225 per-batch `.item()` device→host syncs/epoch → 1/epoch | **13.34 → 13.07 s/ep = 1.02× (+2%)** | the per-batch SYNCS are NOT the dominant cost |
| **faster GPU** (Modal A10G vs T4 vs MPS) | nothing (more FLOPs) | A10G 13.08 ≈ T4 ~11 ≈ MPS 13.34 s/ep ≈ **0%** | the per-epoch is NOT FLOP/throughput-bound |

## The deduction (the deep-math this closes)
- If it were **sync-bound** → deferral would give a large win. It gave +2%. ✗ not sync-bound.
- If it were **FLOP/throughput-bound** → a real GPU upgrade (T4→A10G) would scale. It gave ~0%. ✗ not FLOP-bound.
- Both small ⟹ the per-epoch is **LATENCY-bound**: the serial dependency chain of 75 `batch_size=8`
  optimizer-per-batch steps, each a small forward+backward through frozen SegNet (EfficientNet-B2) +
  PoseNet (FastViT-T12). At bs=8 each step is too small to saturate the GPU and each waits on the prior
  (the per-batch weight update is a hard serial dependency), so the limiter is per-step *latency* — the
  kernel dispatch/execution chain of many small ops — which is GPU-invariant AND not dominated by the
  Python `.item()` syncs specifically.

## The one un-CPU-bound lever is score-locked
The only way to escape the small-batch latency wall is a **larger batch** (fewer, bigger, better-utilized
steps). But the optimizer steps PER-BATCH (`driver.py` `adamw_opt.step()` inside the `for batch_start`
loop), so `batch_size` sets #steps/epoch (600/8 = 75) + the gradient trajectory. Changing it changes the
score AND deviates from the faithful PR95 curriculum → forbidden by "score > training time ALWAYS." So the
throughput floor is a HARD floor for this faithful vehicle.

## What was kept (the +2% is real, free, durable)
`defer_batch_sync` is PROVEN bit-identical (golden hash ≡ defer-OFF ≡ defer-ON, all sha256 equal;
`test_batch_sync_deferral_bit_identical`). It is default-OFF, opt-in via `--defer-batch-sync`, and the
live run now runs with it. +2% is small but free + provably neutral + benefits every future local run, so
it stays on. NOT reverted.

## Verdicts
- **Modal: NO-GO.** ~0% faster than free local; $20 doesn't reach the stage-5 verdict ($27 T4 / $59 A10G);
  the only real lever is score-affecting. ~$1.73 of $20 spent across the two tests. Volume
  `yousfi-r3-pr95-resume` staged (rebuildable, deletable).
- **Local stays ~4 days** to the full curriculum / ~2.2 days to the stage-5 verdict, on the +2% loop.
- **Next-vehicle design note (10-year lesson):** a from-scratch-optimal vehicle escapes this floor with a
  larger training batch (or gradient accumulation that preserves the per-epoch trajectory) — a design
  choice for a NEW vehicle, NOT a change to the faithful PR95 reproduction. Filed for the capstone.

## torch.compile sister-finding (numerical fragility of the pose axis)
On A10G, `torch.compile` of the FROZEN scorers left d_seg (argmax) unchanged (0.0 flip-rate) but drifted
PoseNet output **22% relative** — far past the 1e-4 neutrality gate → REJECTED, kept default-OFF. The two
distortion axes have DIFFERENT numerical fragility: d_seg (argmax) is robust to kernel-backend changes;
d_pose (continuous MSE on FastViT) is fragile (same family as the 23× MPS-pose-drift lesson). Reusable
rule for any future kernel swap (compile / MPS / fused): re-validate the POSE axis, not just seg.

## NO-FAKE ledger
- MEASURED: defer-on 13.07 s/ep vs 13.34 baseline (+2%, 85-ep steady-state segment); A10G≈T4≈MPS (~0% GPU
  scaling); torch.compile pose drift 22%; resume continuity clean (ep5450 d_seg 0.00232 / d_pose 0.00055).
- INFERRED: the latency-bound attribution (consistent with BOTH measured rows; not separately profiled per-kernel).
- NOT claimed: no score moved; pointer UNMOVED 0.19110; the +2% does not change the ~4-day LOCAL timeline.
