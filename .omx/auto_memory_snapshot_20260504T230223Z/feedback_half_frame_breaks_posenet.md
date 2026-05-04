---
name: Half-frame mask trick BREAKS PoseNet on the dilated-h64 renderer WITHOUT joint warp-expansion training
description: Quantizr's "store only odd-frame masks, warp to recover even" trick produces correct-looking SegNet outputs but catastrophically wrong PoseNet outputs on our dilated-h64 renderer (PoseNet=28.7 vs expected ~0.01) WHEN the renderer is trained without joint warp-expansion. The motion module wasn't trained for warped masks; the warp-expansion zeroes the (e_t1 - e_t).abs() diff feature the predictor depends on. Quantizr SHIPS half-frame at 0.33 because they train jointly. Lane H-V3 (2026-04-28) is the proper revival via curriculum mask_half_sim_prob 0.0→1.0 over first 200 epochs.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

**2026-04-28 UPDATE — joint-training revival**: This memory previously implied "half-frame breaks PoseNet, period." That is WRONG. Half-frame breaks PoseNet WITHOUT joint warp-expansion training. Quantizr ships half-frame at 0.33; we score 1.05 on full-frame. The fix is JOINT training (renderer sees warped masks during training, not just at inflate). See `project_lane_h_v3_revival_design_20260428` for the proper revival design with curriculum ramp 0.0 → 1.0 over first 200 epochs of training.

**2026-04-28 CRUCIAL CLARIFICATION — bug class is "mixing half-frame WITH warp", not "half-frame"**: The Sherlock audit of Quantizr PR #55 (`.omx/research/quantizr_replica_audit_20260428.md`) revealed something deeper. **Half-frame breaks ONLY in our warp-based architecture.** Quantizr's actual architecture (`JointFrameGenerator`, PR #55, scores 0.33) has NO motion module, NO optical flow, NO warp at all. PR body verbatim: "dropping optical flow and using Feature-wise Linear Modulation on pose vectors instead of using both masks." His half-frame is TRIVIAL: store 600 odd-frame masks, at inflate run `(frame1, frame2) = generator(mask2, pose6)` from a single-mask + FiLM-on-pose dual-head. There is no warp to break. The bug class is therefore "mixing half-frame with a warp-based renderer", NOT "half-frame". Lane Q-FAITHFUL (`scripts/remote_lane_q_faithful_jointgen.sh`, profile `q_faithful_dilated_88k`) is the corrected rebuild — TRUE 1:1 Quantizr architecture, no motion module. In that architecture half-frame is the natural archive layout, not a retrofit. Predicted band [0.40, 0.80].

Prior failed half-frame attempts (2026-04-27 to 2026-04-28):
- Lane D (V1, V2, V3) — RETROFIT: `mask_half_sim_prob=0.5` mid-train on already-locked renderer; V3 had distribution mismatch (train endpoint=0.5, inflate=1.0)
- Lane V (V1, V2) — JOINT-from-epoch-0 but with channel-broadcast bug at conv layer (88K DSConv path)

The unifying lesson: half-frame requires (1) joint training from epoch 0 with a curriculum, AND (2) train/inflate distribution parity (i.e., if inflate sees 100% warped masks, training endpoint must be 100% warped too). See `feedback_check_42_train_inference_parity_20260428` for the bug-class catalog entry. See `project_killed_lanes_forensic_audit_20260428` for the full forensic audit of the 4 failed attempts.

---

**Verified 2026-04-27 on Oregon RTX 4090 instance 35668576, full contest-CUDA via experiments/contest_auth_eval.py:**

Built a half-frame archive via `experiments/build_baseline_archive.py --half-frame --crf 50`:
- 600 odd-indexed mask frames at 384x512, encoded AV1 CRF=50 = 236,524 bytes
- Renderer + poses + half-frame masks = 513,889 bytes (rate contribution 0.342)

Result on contest-CUDA upstream/evaluate.py:
- **Final score: 17.55**
- PoseNet dist: **28.72** (catastrophic; baseline 0.011)
- SegNet dist: 0.00262 (good; close to baseline 0.0024)
- Rate: 0.342 (matches our prediction)

**The breakdown is the diagnostic.** SegNet is fine — half-frame masks decode + warp produce semantically-correct masks at even frames. But PoseNet is 2,600x worse than the baseline. The renderer's MotionPredictor uses `(e_t1 - e_t).abs()` as a diff feature — when even-frame masks are RECONSTRUCTED via warp from odd-frame masks (instead of being independently SegNet-extracted), the diff feature becomes near-zero or wrong, and motion prediction fails. Frames look visually OK to SegNet but motion-incoherent to PoseNet.

`submissions/robust_current/inflate_renderer.py:1042-1064` documents this exact trade-off:
> "Half-frame masks (600 odd-frame only): we need to recover the even-frame (t) masks here. PROPER: when a RadialZoomWarp is available, warp t+1 → t via inverse zoom flow. This is Quantizr's paradigm and gives full-quality reconstruction."

**Quantizr's renderer was trained jointly with the warp-expansion**, so its motion module learned to handle the warped even-frame masks. Our dilated-h64 renderer was trained on independently-SegNet-extracted masks at every frame; substituting warped masks at half the frames breaks the motion path.

**How to apply:**
1. **NEVER ship a half-frame archive against the dilated-h64 renderer.** PoseNet is catastrophic.
2. The half-frame trick is a JOINT property of (renderer training regime, warp implementation, mask encoding). Mixing-and-matching across renderers doesn't work.
3. Future renderer training runs that want to use the half-frame trick must include warp-expansion in the training loop (i.e. train with `(SegNet_at_odd_frame, warped_even_frame)` pairs as input, not `(SegNet_at_odd, SegNet_at_even)`).
4. For our current dilated-h64, **only full-frame mask archives produce sensible PoseNet scores**. The historical 0.9001 archive (338KB total = ~26KB masks) is therefore mathematically impossible with our renderer at any sane resolution × CRF combination. The 0.9001 record is unreproducible from the current renderer + poses pair.

**Cost of this discovery:** ~$0.20 of CUDA spend on Oregon. Cheap, clean, definitive.
