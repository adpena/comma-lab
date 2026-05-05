---
name: Lane H-V3 — Half-frame revival via JOINT warp-expansion training (curriculum 0.0→1.0 over 200 epochs)
description: 2026-04-28 design. Quantizr ships half-frame at 0.33 and beats us 0.33 vs 1.05. Our 4 prior half-frame attempts (V, V-V2, D, D-V3) all failed because we never properly implemented JOINT warp-expansion training. Lane H-V3 inherits from Lane G v3 1.05 anchor + adds curriculum mask_half_sim_prob=0.0→1.0 ramp over first 200 epochs + use_zoom_flow=True. Predicted band [0.55, 0.95] [contest-CUDA].
type: project
originSessionId: forensic-audit-20260428
---

## Why prior half-frame lanes failed (root cause)

| Lane | Failure | Root cause |
|------|---------|------------|
| Lane D (V1) | PoseNet=17.55 score | RETROFIT: `mask_half_sim_prob=0.5` mid-train on a renderer ALREADY locked into (e_t1-e_t).abs() diff features |
| Lane D-V3 | "half-frame broken" | Distribution mismatch: train endpoint=0.5, inflate=1.0 (same bug class as Lane M-V2 BUG-1) |
| Lane V | Channel crash at conv | `use_dsconv=True` + 88K-class arch wiring — channel-broadcast bug downstream of `warp_inverse_masks` |
| Lane V-V2 | Same as V | Inherits Lane V profile + adds annealing (annealing was correct, channel bug propagates) |

**The unifying lesson**: half-frame requires JOINT training where the renderer's motion module learns warp-expansion AS A FIRST-CLASS DISTRIBUTION, not as an after-the-fact retrofit OR as a half-attended training-time augmentation. Quantizr does this from epoch 0; we never have.

## Lane H-V3 design

### Anchor

**Lane G v3** (`DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL` minus the half-frame) = 1.05 [contest-CUDA]. 288K params, dilated h=64, NOT the 88K Quantizr-replica. Why: Lane V's channel bug suggests the DSConv 88K path has wiring issues; Lane G v3 has a known-working pipeline at 1.05.

### What we change

1. **Curriculum `mask_half_sim_prob` ramp 0.0 → 1.0 over first 200 epochs**
   - Epochs 0-100: full-frame (`mask_half_sim_prob=0.0`)
   - Epochs 100-300: linear ramp 0.0 → 1.0 (200 epochs, 10% of 1980 total)
   - Epochs 300-1980: half-frame (`mask_half_sim_prob=1.0`) — Quantizr endpoint
   - Schedule encoded as `mask_half_sim_prob_anneal: {start_value=0.0, end_value=1.0, ramp_start_frac=0.05, ramp_end_frac=0.15}` (relative to total epochs)
2. **`use_zoom_flow=True`** — wires `RadialZoomWarp` into both training and inflate. Required by preflight when `mask_half_sim_prob > 0`.
3. **`mask_half_sim_prob=1.0` (static endpoint)** — matches inflate-time distribution. Fixes Lane D-V3 mismatch.
4. **NEW augmentation**: at training time, the half-frame mask path mirrors inflate-time exactly. The same `RadialZoomWarp` instance + `warp_inverse_masks` call is used in both training step (line 2686 in train_renderer.py) AND in inflate-side `inflate_renderer.py:1042-1064`. **Symmetry is the contract.**
5. **KL distill weight = 0.002** (post-bugfix value, matches Lane V-V2).

### What stays the same

- 288K params (not 88K) — keeps known-working pipeline
- Phases (P1=400, P2=1080, P3=200, P4=200, P5=100, total=1980)
- Phase LRs (P1=1e-3, P2=5e-4, P3=2e-4, P4=1e-4, P5=2e-5)
- `eval_roundtrip=True` (NON-NEGOTIABLE per CLAUDE.md)
- All Fridrich aux losses
- 5-stage QAT pipeline + RESIDUAL FP4 codebook + robust_scale + stochastic rounding

### Predicted band

**[0.55, 0.95] [contest-CUDA]**

- **Floor 0.55**: Joint training works. Half-frame archive saves ~0.20 in rate vs full-frame; renderer trained for warp-expansion holds PoseNet ≤ 0.020 and SegNet ≤ 0.005. Lane G v3 anchor 1.05 - 0.20 (rate) - 0.30 (PoseNet headroom from joint training) ~ 0.55.
- **Ceiling 0.95**: Curriculum buys nothing; the renderer learns ONE distribution well (full-frame from epochs 0-100) and the late half-frame transition is too abrupt. Score lands at Lane G v3 - small rate gain ~ 0.95.
- **Above 1.05 (Lane G v3)**: Possible if the curriculum harms the full-frame distillation. We mitigate via warmup phase keeping full-frame at 0.0 for first 5%.

### Cost

~5h on RTX 4090 @ $0.25/hr = ~$1.25 (matches Lane D-V3 schedule). Plus ~30min pose TTO + ~15min auth eval = $1.50 total.

## Implementation artifacts

1. **Profile**: `H_V3_JOINT_HALFFRAME` in `src/tac/profiles.py`. Inherits from `DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL`. Overrides:
   - `mask_half_sim_prob: 1.0` (was 0.5)
   - `mask_half_sim_prob_anneal: {start_value=0.0, end_value=1.0, ramp_start_frac=0.05, ramp_end_frac=0.15}` (was 0.0→0.5 with 0.30→0.70 ramp)
   - `seed: 67` (different RNG basin from Lane G v3 seed=43)

2. **Script**: `scripts/remote_lane_h_v3_jointly_trained_halfframe.sh`. Mirrors `scripts/remote_lane_d_v3_full_engineering.sh` structure (Stage 0-5 with NVDEC probe, provenance, heartbeat, etc.).

3. **Tests**: `src/tac/tests/test_lane_h_v3_joint_halfframe.py`. 5 tests with magnitude anchors:
   - `test_profile_registered` — H_V3 in PROFILES
   - `test_curriculum_schedule_ramps_0_to_1` — explicit start/end + ramp-mid value at 0.5
   - `test_use_zoom_flow_true` — required by preflight
   - `test_static_endpoint_matches_inflate_distribution` — `mask_half_sim_prob == 1.0`
   - `test_arch_inherits_lane_g_v3` — base_ch=36, mid_ch=60, motion_hidden=32, depth=1

## Composability

Lane H-V3 stacks with:
- **Lane G v3 KL distill** (already inherited)
- **Lane W per-pair self-compress** (orthogonal: rate attack on already-quantized renderer)
- **Lane J-JBL Jaccard loss** (orthogonal: SegNet-side improvement)
- **Lane SAUG-V2** (orthogonal: input-numerics augmentation)

Stack moonshot: Lane H-V3 + Lane W + Lane J-JBL = predicted [0.30, 0.55] (sub-Quantizr territory).

## Risk register

1. **Curriculum too aggressive**: 200 epochs of ramp on 1980 total is 10% — short. Mitigation: profile carries the schedule; can tune to 0.05→0.20 (15% ramp) if first run plateaus.
2. **Channel bug from Lane V repeats**: Lane H-V3 uses 288K (NOT 88K) so the DSConv path is bypassed. Channel bug should not recur.
3. **PoseNet still underperforms**: if JOINT training doesn't fix PoseNet, the half-frame paradigm is genuinely incompatible with our renderer family. In that case: Lane H-V3 lands ~ 1.05-1.10 and we close the half-frame chapter. Cost: $1.50.

## Cross-references

- `project_killed_lanes_forensic_audit_20260428` (the audit that motivated this)
- `feedback_half_frame_breaks_posenet` (UPDATED 2026-04-28 with joint-training note)
- `project_lane_g_v3_landed_1_05_20260428` (anchor lane)
- `project_quantizr_full_intel_20260421` (what we're trying to match)
