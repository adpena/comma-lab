# Pose-sidecar reuse assessment — can we lift a public PR's pose blob into the task-space witness?

Date: 2026-07-08. READ-ONLY investigation (no GPU, no archive rebuild, no edits to any
`public_pr*_intake_*` clone — all PRISTINE per Catalog #109). Pointer UNMOVED at contest-CPU
**0.19110**. Nothing here moves it; this is a MEANS assessment.

## TL;DR bottom-line verdict

**No.** There is no public pose blob that is video-intrinsic + low-d_pose + legal + separable and
therefore "worth lifting as-is." Every low-d_pose public pose blob in the pose family
(qpose14 / quantizr / fp4_mask_gen — they are the SAME architecture) is **case (c): pose as a
byproduct of FULL-FRAME PHOTOMETRIC RGB RECONSTRUCTION**, WELDED to a `JointFrameGenerator` that
renders both RGB frames. The stored pose blob is TINY (~1.4–7 KB) and legal, but it is only a
FiLM CONDITIONING INPUT — it does nothing without the ~214 KB mask + ~56 KB frame-generator that
reconstructs the actual pair. Their low d_pose comes from reconstruction fidelity, not from the
blob addressing PoseNet.

Lifting the blob therefore = adopting "reconstruct the real pair" (Option B, rate-heavy: their
archives are ~175–277 KB, dominated by the mask video + decoder weights), which is exactly what our
task-space (non-photometric-frame1) witness deliberately avoids. **Storing a pose target is trivial
(we already do it, `src/tac/scorer_targets.py`, ~5 KB); HITTING it is the entire wall** — and our
known warp-model cap (~2.5 d_pose, MEMORY L68: planar-homography can't reproduce depth-parallax) is
precisely the reconstruction problem the public winners solve by rendering full RGB, not by any
cleverer blob. Do NOT oversell "just reuse it": the consuming mechanism is the blocker.

Also note (MEASURED, `inflate.py`): the public pose6 is the EGO-MOTION pose fed as a generator
input — it is NOT the PoseNet 6-scalar target that d_pose is computed against. So it is not even the
same object as our stored targets; it is welded conditioning.

## Decision table

### Row 1 — qpose14 (`qpose14_qzs3_filmq9g_slsb1_r55`, `qpose14_r55_segactions_minp`) — recurs PR81/82/94/100/105/106
1. **PR/score:** appears under many PRs (PR100 medal cluster among them). No axis-labeled score
   found in the pristine intake dirs (PR95 explicitly `score_claim: False`; PR100/101/103 carry only
   README, no report.txt). MEMORY records the medal cluster at ~0.193–0.199 [their-CPU] historically
   — NOT re-verified here, not authority for us.
2. **What the blob stores (classify):** **(c)** with (a)-flavored storage. `pose6` per-pair
   ego-motion vectors, quantized. Consumed as FiLM conditioning of a full-RGB frame generator.
   Evidence: `qpose14_qzs3_filmq9g_slsb1_r55/inflate.py:607-628` `JointFrameGenerator.forward(mask2,
   pose6)` → `pred_frame2 = frame2_head(shared_feat)`, `cond_emb = pose_mlp(pose6)`, `pred_frame1 =
   frame1_head(shared_feat, cond_emb)`; pose decode `inflate.py:786-812` (QP1 delta-varint or raw
   uint16, `pose_np[:,0]=q/512+20`, `pose_np[:,1:]=int16/2048`); optional `smooth_pose` poly_fourier
   correction `inflate.py:685-706,822-828`.
3. **Byte size (READ/computed):** packed single file `p`. From the packed math `inflate.py:747-765`:
   mask = 219472 B, model ≈ 56093–61147 B, `pose_q = payload[219472+model_len:]`. For the PR94
   archive (`p` = 276987 B, MEASURED `unzip -l`), pose_q ≈ **~1.4 KB** (QP1 stores mostly dim-0).
4. **d_pose for us?** Their d_pose is low ONLY because frame1/frame2 are RECONSTRUCTED RGB
   (`FrameHead.head = Conv(hidden,3,1)`, `sigmoid*255`) and PoseNet reads good frames. This is the
   ancestor full-RGB-photometric regime (MEMORY L68: 3.4e-5 is that regime, non-transferable to a
   non-reconstructing witness). Our witness is stuck ~1.79 because it does NOT reconstruct the pair.
5. **Legal?** YES. `grep` for safetensors/posenet/segnet/fastvit/efficientnet in inflate.py =
   EMPTY (MEASURED). No scorer loaded at inflate; blob is derived pose data.
6. **Separable?** NO. `pose6` only enters via `pose_mlp`→FiLM on `frame1_head`; with no frame
   reconstruction the blob is inert. Welded to the JointFrameGenerator + 219 KB mask + 56 KB weights.
7. **VERDICT:** **WELDED-NOT-REUSABLE** — the blob is tiny/legal but its value lives entirely in the
   full-RGB reconstruction it conditions; lifting it = lifting the rate-heavy renderer.

### Row 2 — quantizr (`.../submissions/quantizr/`, PR100/105 intake; inflate 12.4 KB)
1. **PR/score:** the canonical Quantizr submission. No axis-labeled score in pristine dir.
2. **Classify:** **(c)** — IDENTICAL architecture. Evidence: `quantizr/inflate.py:199-223`
   `JointFrameGenerator(mask2, pose6)` → `pred_frame2 = frame2_head(shared_feat)`,
   `cond_emb = pose_mlp(pose6)`, `pred_frame1 = frame1_head(shared_feat, cond_emb)`. Pose is a plain
   stored `pose.npy.br` input: `inflate.py:262,282-285` `pose_frames_all = np.load(...)`;
   fed at `inflate.py:301,309-311` `in_pose6 = file_poses[...]; fake1,fake2 = generator(in_mask2,
   in_pose6)`.
3. **Byte size:** `pose.npy.br` (brotli'd npy of ~600×6). Not separately built here; same order as
   fp4 codec (~2–7 KB). Total archive ~175–277 KB range.
4. **d_pose for us?** Same as Row 1 — reconstruction-fidelity-derived, non-transferable.
5. **Legal?** YES (grep EMPTY, MEASURED).
6. **Separable?** NO — same FiLM-into-frame-generator weld.
7. **VERDICT:** **WELDED-NOT-REUSABLE.**

### Row 3 — fp4_mask_gen (PR101/PR103 intake) — the CLEANEST pose codec
1. **PR/score:** part of the PR101/103 medal-cluster submission set. No axis-labeled score in dir.
2. **Classify:** **(c)** — same `JointFrameGenerator`. Evidence: `fp4_mask_gen/inflate.py:178-181`
   `forward(mask2, pose6): return frame1_head(feat, pose_mlp(pose6)), frame2_head(feat)`;
   `FrameHead.head = QConv2d(hidden, 3, 1)` (`inflate.py:166`) = **3-channel RGB**, `sigmoid*255`
   (`:167`), bilinear-upsampled to 874×1164 (`:246-247`) written to `.raw` (`:240`) = full
   photometric reconstruction.
3. **Byte size (READ):** cleanest declared layout — `pose.bin.br` = "12 fp32 (per-dim mn, mx) +
   N_PAIRS*6 uint16" (`inflate.py:8`, decoder `:69-75`) = 48 + 600·6·2 = **7248 B raw**, brotli'd
   ~2–4 KB.
4. **d_pose for us?** Same reconstruction-derived regime; non-transferable to a non-reconstructing
   witness.
5. **Legal?** YES (grep for scorer weights in inflate.py = EMPTY, MEASURED).
6. **Separable?** NO — identical FiLM-into-RGB-generator weld.
7. **VERDICT:** **WELDED-NOT-REUSABLE** (it IS the tidiest codec if we ever wanted to store a 6-dim
   per-pair vector, but the vector is a generator input, not a PoseNet-addressing sidecar).

### Row 4 — hnerv_lc_v2 (PR100), HNeRV medal family / PR95 root
1. **PR/score:** HNeRV substrate. PR95 intake `score_claim: False`
   (`pr95_hnerv_muon_packing_profile.md:8`); PR100 README only.
2. **Classify:** **(c)/other** — full-frame HNeRV reconstruction; pose is not a distinct addressable
   sidecar (docstring `hnerv_lc_v2/inflate.py:12` "minimize SegNet+PoseNet distortion"; no separate
   pose section). d_pose falls out of reconstructing both frames.
3. **Byte size:** n/a as a standalone pose blob (no separable pose section).
4. **d_pose for us?** Reconstruction-derived; non-transferable.
5. **Legal?** YES.
6. **Separable?** NO — there is no separable pose blob to lift.
7. **VERDICT:** **NOT-A-POSE-SIDECAR** (pose is implicit in full-frame reconstruction).

### Row 5 — PR94 (`public_pr94_qpose_intake`, archive-only)
1. Archive-only intake (`archive.zip` → single `p`, 276987 B, MEASURED `unzip -l`); no source. Its
   packed layout matches the qpose14 packer exactly (Row 1). No report/comment file present.
2–7. Same as Row 1 (qpose family). **VERDICT: WELDED-NOT-REUSABLE** (same packer, same weld).

## Why this settles the operator's question
"Can we just use their actual sidecar blob without recreating it, and did they score low enough on
pose for it to work?" — The honest answer: their low pose is **welded to full-frame reconstruction**.
The blob is tiny/legal/video-intrinsic-ish, but it is a FiLM conditioning input to a JointFrame
Generator that renders both RGB frames; PoseNet's low distortion is bought by reconstructing the
frames (rate-heavy: ~175–277 KB archives, mask+weights dominated). Lifting the blob = adopting
Option B "store/reconstruct the real pair," not a free pose fix. Storing a 6-dim pose target is
trivial and already ours (`scorer_targets.py`); the wall is a render that HITS it, which is our
open warp-model cap. Pointer stays 0.19110.

Evidence tags: architecture/consumption/legality/RGB-output = MEASURED (file:line above, grep
EMPTY for scorer loads). Byte sizes = READ (declared section constants) / computed from packed math.
Reconstruction-vs-target d_pose attribution = INFERRED from the measured forward pass + MEMORY L68.
No axis-labeled contest score was present in the pristine intake dirs; historical cluster numbers
are MEMORY-recorded, not authority.
