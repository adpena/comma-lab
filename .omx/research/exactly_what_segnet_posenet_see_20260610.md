# EXACTLY what SegNet and PoseNet see (line-cited from modules.py + frame_utils.py, 2026-06-10)

The archive must produce frames `(B, seq_len=2, H=874, W=1164, 3)` uint8 RGB (camera_size=(1164,874),
seq_len=2). Both nets take the SAME camera pair; they preprocess DIFFERENTLY. d_seg/d_pose are computed
on these exact transforms (modules.py compute_distortion).

## SegNet sees (modules.py SegNet.preprocess_input + compute_distortion)
1. `x[:, -1, ...]` — ONLY frame1 (the LAST frame). frame0 is 100% unused by SegNet.
2. `interpolate(frame1, size=(384,512), mode='bilinear')` — resize 874×1164 → **384×512** (a 2.27×
   bilinear DOWNSAMPLE per axis). NO mean/std normalization (smp.Unet on raw resized RGB [0,255]).
3. d_seg = mean over the **196,608** (=384·512) pixels of `argmax_5class(out1) != argmax_5class(out2)`,
   averaged over 600 pairs. **SegNet scores the 384×512 ARGMAX PARTITION of a BLURRED frame1.** Nothing
   else about frame1 matters — not appearance, not camera-resolution detail, only which of 5 classes
   wins per pixel at 384×512.

## PoseNet sees (modules.py PoseNet.preprocess_input + forward + compute_distortion)
1. BOTH frames → `interpolate(size=(384,512), bilinear)` → `rgb_to_yuv6` → `b (t c) h w` = **12 channels**
   (2 frames × 6).
2. CRITICAL (frame_utils rgb_to_yuv6): YUV6 outputs at **HALF** the resized res. y00,y10,y01,y11 are the
   four 2×2 luma sub-phases at **192×256**; U_sub,V_sub are 2×2 BOX-AVERAGED chroma at 192×256. So
   PoseNet's fastvit_t12 input is **192×256×12** — even lower-res than SegNet, luma reorganized (full
   luma preserved across the 4 phases) but **chroma 2×2-averaged**.
3. `(x-127.5)/63.75` → fastvit → 2048 → 512 → Hydra → pose head **12 dims**.
4. d_pose = MSE on the **FIRST 6 of 12** pose dims, both frames, averaged over 600 pairs. **Dims 7-12 are
   never scored.** PoseNet scores **6 scalars per pair** off the 192×256 luma-dominant two-frame stack.

## The precise eurekas (what this licenses)
- **E-A (per-flip score quantum, exact):** one argmax flip on one pair = 1/(600·196,608) = 8.48e-9 in
  d_seg → 8.48e-7 in score. Current d_seg=5.6e-4 ⇒ exactly **66,060 flips** across the video (matches the
  66,039 flip map). Halving d_seg = fix ~33k flips = −0.028 score.
- **E-B (SegNet's two nulls, exact):** (i) frame0 entirely free; (ii) frame1 only through 384×512
  bilinear pooling (the resize null, 80.67%) AND only the argmax (the margin null — any logit move inside
  the margin is free). SegNet is invariant to everything except the 384×512 class partition.
- **E-C (PoseNet's three nulls, exact):** (i) output dims 7-12 unscored; (ii) input chroma 2×2-averaged
  (chroma perturbations vanishing under box-average are pose-free); (iii) 192×256 spatial = even more
  appearance freedom than SegNet. Pose is **6 scalars** — a tiny, smooth (Taylor-able) target.
- **E-D (the decomposition the score ACTUALLY measures):** {frame1's 384×512 5-class argmax partition}
  + {the pair's first-6 pose scalars at 192×256 luma}. THAT is the entire scored content. A
  camera-resolution RGB renderer (ours, Quantizr, the whole leaderboard) pays for ~3M RGB pixels to
  specify ~196k labels + 6 scalars per pair — orders of magnitude of waste. This is the precise,
  line-cited proof of the score-native class shift.
- **E-E (frame0 fully characterized):** frame0's ONLY scored job is its contribution to the 6 pose dims
  via the 192×256 YUV6 two-frame stack. It can be ANY frame holding those 6 scalars given frame1 — and
  frame0 = warp(frame1, ego-motion) lives in rate-free inflate code (E1). PR110's frame0-trick used
  only null (i); this uses the full equivalence class.

## The direct targets (for the variational solve + lever B smoke)
Store/generate: (1) the 384×512 argmax partition of frame1 (piecewise-constant, ~196k labels of 5
classes — highly compressible, and only needs to survive the blur+argmax, not match RGB); (2) the
600×6 pose trajectory (smooth, ~KB); (3) minimal margin-holding + pose-hitting appearance. NOT RGB,
NOT camera resolution. d_seg is attacked by getting the PARTITION right (generate it, don't render it);
d_pose by the 6-scalar trajectory; rate collapses because you stop paying for invisible appearance.

Consumers: the direct differential-geometric solve memo; lever B (score-native carrier) + G
(argmax-invariance budget) from the offensive plan; F (the floor IS the min bits to specify this
content). The #1 smoke: train a tiny generator to hit the frozen SegNet 384×512 argmax (CE+argmax-hinge)
+ store the 6 pose dims — exact-CPU d_seg + blob bytes.
