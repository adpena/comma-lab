---
name: Lane Q-FAITHFUL — TRUE 1:1 Quantizr PR #55 architecture replica
description: 2026-04-28 corrected rebuild after Sherlock audit proved Lane V/V2/K kept the wrong architectural family. JointFrameGenerator (87,836 params) with NO motion module, NO warp, NO optical flow. Single-mask trunk + FiLM-on-pose dual-head. Predicted band [0.40, 0.80] [contest-CUDA].
type: project
originSessionId: lane-q-faithful-20260428
---

## Problem statement

Prior Quantizr replicas (Lane V, V2, K) all kept our **warp-based AsymmetricPairGenerator + dual-mask input + MotionPredictor**. The Sherlock audit (`.omx/research/quantizr_replica_audit_20260428.md`) proved this is the wrong architectural family: PR #55 (Quantizr, 0.33) **explicitly removed** the motion module. The PR body verbatim:

> "Changes made from #53 include **dropping optical flow** and **using Feature-wise Linear Modulation on pose vectors instead of using both masks**. As a result the **mask video only needs to encode half as many frames** and can be done at a higher CRF."

Lane V's 1-ch vs 3-ch crash was a symptom; the deeper bug was building the wrong model.

## What Lane Q-FAITHFUL actually is

**File**: `src/tac/quantizr_faithful_renderer.py` (verbatim port of `/tmp/quantizr_inflate.py:140-223`).

```
class JointFrameGenerator(nn.Module):
    def forward(self, mask2, pose6):
        coords = make_coord_grid(b, 384, 512, ...)
        shared_feat = self.shared_trunk(mask2, coords)   # SharedMaskDecoder
        pred_frame2 = self.frame2_head(shared_feat)      # UNCONDITIONAL
        cond_emb = self.pose_mlp(pose6)                  # Linear(6,48)+Linear(48,48)
        pred_frame1 = self.frame1_head(shared_feat, cond_emb)  # FiLM(pose)
        return pred_frame1, pred_frame2
```

**Key facts**:
- Total params: **87,836** (target 88K from PR; within ±2K band).
- NO motion module. NO warp. NO optical flow. NO dual-mask. Single odd-frame mask + 6-DOF pose ⇒ both reconstructions.
- `pose_mlp = Linear(6,48) → SiLU → Linear(48,48)` — `cond_dim=48`, not `pose_dim=6` fed direct.
- `SharedMaskDecoder(emb_dim=6, c1=56, c2=64, depth_mult=1)` per Quantizr's `compress.py:565-571`.
- DSConv (depthwise-separable) trunk via `SepConvGNAct` (DW + PW + GN + SiLU) and `SepResBlock`.
- `FiLMSepResBlock` modulation: `x * (1 + gamma) + beta` with `film_proj = Linear(48, 112)` (= 2 × 56 channels).

## Top architectural decisions

1. **Discard mask_t in the training shim.** The training loop expects `model(mask_t, mask_t1) → (B,2,H,W,3)` HWC pairs. The Q-FAITHFUL `_QuantizrFaithfulShim` wraps `JointFrameGenerator` and explicitly throws away `mask_t` — Quantizr's premise is that `mask_t1` (odd-frame) plus `pose6` fully determines both reconstructions. Tested via `test_inflate_shim_drops_mask_t_invariant` (varying mask_t with mask_t1 fixed produces bit-identical outputs).

2. **Single-mask reuse pattern.** Both `frame1` and `frame2` decode from the **same** `shared_feat`. The differentiation comes from `frame2_head` being unconditional and `frame1_head` having FiLM modulation. Tested via `test_frame2_unconditional_invariant_to_pose` (varying pose6 leaves frame2 BIT-IDENTICAL — proves no pose leak) and `test_frame1_changes_with_pose` (varying pose6 changes frame1 by > 1e-3 — proves FiLM is connected).

3. **QFAI binary format.** Simple: `[b"QFAI"][4-byte header_len][JSON header][torch.save(state_dict)]`. The inflate-side dispatch in `submissions/robust_current/inflate_renderer.py` reconstructs `JointFrameGenerator` from the header and wraps it in `_QuantizrFaithfulInflateShim` so the existing pair-iteration code path serves it without modification. `_is_asymmetric_model` returns True for `q_faithful=True` models.

## Where we deviated from Quantizr's exact code

- **Quantization**: Quantizr ships `FP4Codebook(pos_levels=[0,0.5,1,1.5,2,3,4,6], block_size=32)` with custom nibble packing. V1 of Lane Q-FAITHFUL stores **plain FP32** in QFAI; the QFAI header has `fp4_packed=False` and the 5-stage QAT pipeline runs but exports FP32. V2 will flip the flag and pack via our existing `tac.fp4_quantize` (RESIDUAL codebook is our advantage per the 5-stage memory). This means V1 archive will be larger than Quantizr's 64KB renderer.bin; V2 lands the FP4 win.
- **`QConv2d` / `QEmbedding` no-op subclasses**: Quantizr defines these for compress-time hooks. We use plain `nn.Conv2d` / `nn.Embedding` — at inflate time they're identical, and our quantization happens at export time via `tac.fp4_quantize.save_fp4` rather than in-place fake-quant during forward.
- **`make_coord_grid` shape**: Quantizr returns `(B, 2, H, W)` (channel-first for conv concat). Our existing `tac.renderer.make_coord_grid` returns `(1, H, W, 2)` (for grid_sample). We re-implement the channel-first variant inside `tac.quantizr_faithful_renderer` rather than reusing the existing helper.

## Test suite (15 passing)

- `test_param_count_within_88k_target_band` — assert 86K < n < 90K
- `test_no_motion_module_present` — state_dict has ZERO motion/warp/flow keys
- `test_forward_shape` — (B, 5, H, W) mask + (B, 6) pose → 2× (B, 3, H, W) frames in [0, 255]
- `test_frame2_unconditional_invariant_to_pose` — frame2 BIT-IDENTICAL under pose perturbation
- `test_frame1_changes_with_pose` — frame1 changes > 1e-3 under pose perturbation
- `test_film_gradient_flows_to_pose_mlp` — backward through frame1 produces finite, non-zero pose_mlp grads
- `test_5_step_training_loss_decreases` — 5-step Adam loss drops > 1%
- `test_film_layer_dimensions_match_quantizr` — Linear(6,48)+Linear(48,48), FiLM proj 48→112
- `test_make_coord_grid_layout_matches_quantizr` — (B, 2, H, W) in [-1, 1]
- `test_film_block_modulates_proportional_to_cond` — FiLM responds to cond
- `test_frame2_static_head_stable_under_input_perturbation` — finite, in-range
- `test_builder_returns_jointframegenerator` — builder smoke
- `test_quantizr_faithful_export_roundtrip` — save_qfai → load_qfai bit-identical
- `test_inflate_shim_returns_hwc_pair` — end-to-end QFAI → inflate → (B,2,H,W,3) pair
- `test_inflate_shim_drops_mask_t_invariant` — mask_t-discard verified

## Predicted score band [0.40, 0.80] [contest-CUDA]

- **Floor 0.40**: Quantizr ships at 0.33; we use Lane A's scorer-measured poses (load-bearing per `project_baseline_poses_load_bearing`) + DSConv (matches) + KL distill T=2.0 (matches) + 5-stage QAT with our RESIDUAL codebook (our advantage).
- **Anchor 0.55**: matches Quantizr 0.33 ± 0.5 typical lane variance.
- **Ceiling 0.80**: Lane A 1.15 minus rate gain. 88K renderer (~64KB FP4) saves ~225KB vs Lane A's 290KB renderer; at 25 bits/byte rate factor that's a 0.30 rate reduction. Even with PoseNet/SegNet matching Lane A exactly, archive shrinks; ceiling is rate-floored.

## Cost / wall-clock

- ~12h on RTX 4090 ($3.00 @ $0.25/hr); + ~30min QFAI export + ~15min auth-eval = $4-5 end-to-end.
- Cost cap $8 (hard kill at $8 budget cap). Wall-clock cap 32h.
- Hard-kill targets: pixel L1 < 14 at ep600, scorer < 4.0 at ep2100, scorer < 1.5 at ep2900.

## Files

- `src/tac/quantizr_faithful_renderer.py` — JointFrameGenerator + sub-modules
- `src/tac/quantizr_faithful_export.py` — save_qfai / load_qfai
- `src/tac/profiles.py::LANE_Q_FAITHFUL_88K` (key `q_faithful_dilated_88k`)
- `src/tac/experiments/train_renderer.py` — variant=quantizr_faithful arm + `_QuantizrFaithfulShim`
- `submissions/robust_current/inflate_renderer.py` — QFAI magic-byte arm + `_QuantizrFaithfulInflateShim`
- `src/tac/tests/test_quantizr_faithful_renderer.py` — 15 tests
- `scripts/remote_lane_q_faithful_jointgen.sh` — 6-stage deploy

## Cross-references

- Audit: `.omx/research/quantizr_replica_audit_20260428.md`
- PR: https://github.com/commaai/comma_video_compression_challenge/pull/55
- Memory: `project_quantizr_full_intel_20260421` (architectural ground truth correction appended)
- Memory: `feedback_half_frame_breaks_posenet` (clarified: bug class is "mixing half-frame WITH warp", not "half-frame")
- Memory: `project_lane_v_crashed_channel_mismatch_20260428` (the symptom that triggered the audit)
- Memory: `project_baseline_poses_load_bearing` (why we ship Lane A poses verbatim)
