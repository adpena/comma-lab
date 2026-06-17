# GENIUS BLIND-SPOT PROBE D — the sufficient-statistic byte floor ($0 CPU) — 2026-06-17

**Authority:** `[macOS-CPU advisory]` NON-PROMOTABLE. GT targets = the frozen contest scorer's OWN
outputs on GT (the literal d_seg/d_pose reference), read from the canonical capped-target cache
`experiments/results/capstone_gt_targets_cache/gt_targets_n100.pt` (GT decoded via
`upstream/frame_utils.yuv420_to_rgb` when built — never PyAV rgb24). $0 CPU, no GPU, no MPS, no PR.
n_pairs = 100. Exact pointer 0.19110 UNMOVED (this is a floor characterization, not a score row).

## The reframe (the probe question)

The contest scorer reads ONLY two objects per pair: the **SegNet-argmax 5-class partition of frame-1**
(the `d_seg` object) and the **PoseNet 6-dim output of the pair** (the `d_pose` object). That pair IS
the sufficient statistic; everything else in a reconstructed RGB frame is bits the scorer never reads.
The minimal contest compressor is the minimal joint code of (partition-stack, pose-trajectory). Q: are
we anchored on a NON-MINIMAL representation (paying to reconstruct unread pixels), or is the learned
decoder already the minimal sufficient-statistic compressor?

## What was built (top-AIML, NO FAKE — both coders decode(encode)==x)

1. **partition-H** — reused `tac.boundary_math.context_partition_codec` (the SOTA context-adaptive
   arithmetic codec for the 5-class smooth label stack; built 2026-06-16). d_seg = 0 (lossless).
2. **pose-H** — built the MISSING real reversible pose carrier in
   `tac.optimization.pose_trajectory_entropy`: `encode_pose_trajectory` / `decode_pose_trajectory` /
   `pose_carrier_real_bytes` — per-dim uniform-quantize → first-order temporal delta → range-code the
   delta stream (constriction) under a transmitted per-dim PMF (model bytes COUNTED, sister of the
   partition codec's transmitted-model discipline). The prior `pose_trajectory_entropy()` was a Shannon
   ESTIMATE only; this is the first REAL coded-bytes pose carrier (asserted bit-exact). Per-dim quant
   steps chosen so induced d_pose == the small-basis pose level (apples-to-apples pose term).
3. **probe** `experiments/sufficient_statistic_floor_probe.py` → `reports/sufficient_statistic_floor.json`.

## Measured sufficient-statistic floor (600-pair extrapolation; partition per-frame, pose per-pair-linear)

| half | coder | real bytes (600) | share | term |
|---|---|---:|---:|---|
| partition-H | context+temporal arithmetic | 271,788 | **98.1%** | d_seg = 0 |
| pose-H | range-coded temporal-delta carrier | 5,208 | 1.9% | √(10·d_pose) = 0.0585 (= small-basis pose) |
| **SS floor** | joint | **276,996** | 100% | **S_floor = 0.2429** |

(partition 452.98 B/frame temporal vs 533.66 spatial; pose 868 B measured @ n=100 → 5,208 B linear-600.)

## Comparison vs the two anchored vehicles

| vehicle | bytes | vs SS-store | rate+pose floor / total |
|---|---:|---:|---|
| **small basis** (base_ch20 HNeRV) | 89,136 | SS is **3.11× LARGER** | 0.1178 floor |
| **frontier** (0.19110) | 177,215 | SS is **1.56× LARGER** | 0.1911 |
| direct SS store (d_seg=0) | 276,996 | — | S_floor 0.2429 |

There is **NO headroom below either vehicle** for a direct sufficient-statistic store. The SS S-floor
(0.2429) is ABOVE the frontier (0.1911) by +0.052 and ABOVE the small-basis floor (0.1178) by +0.125.

## VERDICT: `SS_FLOOR_ABOVE_SMALL_BASIS` — the learned decoder is the cheaper SS carrier (we are NOT anchored on a non-minimal representation)

The blind-spot hypothesis (we pay to reconstruct pixels the scorer never reads, so a direct SS code
beats the decoder) is **REFUTED**. Storing the sufficient statistic DIRECTLY costs 3.11× the small
basis. The learned decoder is a **cheaper carrier of the SAME sufficient statistic** than an explicit
store — it amortizes the partition+pose into shared weights and exploits cross-pair/cross-region
structure the per-pixel context coder cannot. Two concrete consequences:

1. **The pose half is essentially free** — 5,208 B (1.9%) for the full coded pose trajectory at the
   small-basis pose level. Pose is NOT where the bytes are; the partition is 98% of the SS store. This
   re-confirms the CALCULUS finding (pose marginal ≈ 86% of d_seg's) is about the SCORE term, not the
   BYTE budget — pose costs almost nothing to carry.
2. **d_seg belongs IN training** — a renderer carries the partition cheaper than ANY direct partition
   store (271,788 B at d_seg=0 vs the small basis carrying d_seg≈0.0026 in 89,136 B total). The
   binding term is the partition, and the learned decoder already beats the explicit-store frontier by
   1.56–3.11×. This is consistent with yesterday's `yousfi_partition_store_topaiml_reopen` (the SOTA
   non-neural partition store is +0.0083 ABOVE the frontier even at d_seg=0).

The small basis is **near the minimal explicit-SS frontier** — we are NOT anchored on a non-minimal
representation. The sub-0.15 path remains the byte-neutral d_seg attack on the learned carrier (oomph
long-train + d_seg-aware taper), not a representation swap.

## Honest caveats

- 600-pair pose bytes are a LINEAR upper bound (the one-time per-dim PMF model is re-counted ~6× under
  linear scaling; the true 600-pair pose stream is slightly smaller — pose is already only 1.9%, so the
  conclusion is robust to this). Partition bytes scale per-frame (well-founded).
- n_pairs = 100 (not the full 600). The partition B/frame (452.98) is close to yesterday's n=24 measure
  (456.5), so it is stable; the floor would not move materially at full n.
- This is an EXPLICIT-store floor. It does NOT prove the learned decoder is GLOBALLY minimal — only that
  a direct context-coded SS store does not beat it. A learned/hybrid SS code (e.g. a tiny network that
  predicts the partition from a few latents) is exactly what the decoder already is.

## Wire-in / reusable surface (6-hook)

- `tac.optimization.pose_trajectory_entropy.{encode,decode}_pose_trajectory` + `pose_carrier_real_bytes`
  — a reusable REAL reversible pose carrier any pose-coding lane can import (sister of the partition
  codec; first real coded-bytes pose carrier, replacing the Shannon-estimate-only path for floor claims).
- #3 bit-allocator (the pose rate term is now a real measured quantity) ACTIVE; #6 probe-disambiguator
  (this probe answers "is the decoder non-minimal?") ACTIVE; #1/#2/#4/#5 N/A for a $0 advisory
  non-promotable floor characterization. 10 NO-FAKE tests
  (`src/tac/tests/test_pose_trajectory_carrier_codec.py`) + 23 existing module tests green.
