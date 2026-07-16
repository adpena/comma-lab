# Frozen scorer — EXACT flattened factorization (modules.py + evaluate.py + frame_utils.py)

**Operator directive 2026-07-15:** *"modules.py and evaluate.py show exact order and resizing and yuv and
chroma and luma, if we're not deconstructing and flattening and understanding and factorizing all we are
not doing deep math and geometry correctly and risk naive and toy."* This is the deconstruction — every
map in the scored forward chain, in exact order, factorized into (linear · linear · frozen-nonlinear ·
decision). Read directly from the pinned upstream source (NEVER a reimplementation). Pointer 0.19108 UNMOVED
— this is means (the exact white-box the c2 levers optimize against).

## 0. The witness produces (what we control)
Per pair, at `camera_size = (W=1164, H=874)`, 3-ch uint8 RGB (via inflate → the R operator). The contest
batches pairs `seq_len=2`: `x` shape `(B, 2, 874, 1164, 3)` uint8. Everything below is the FROZEN map from
that tensor to the score. Our only DOF is the pixel values (and, upstream, the render grid the R operator
downsamples FROM).

## 1. DistortionNet.preprocess_input (modules.py:143-148)
- `rearrange 'b t h w c -> b t c h w' .float()` → `(B,2,3,874,1164)` float ∈ [0,255].  **[LINEAR: a permute]**
- fork into the two scorer paths (below). Frame_0 and frame_1 both go to PoseNet; **only frame_1 (last) goes to SegNet.**

## 2. SegNet path — the d_seg path (OUR primary controllable)
| # | op | source | algebraic type |
|---|----|--------|----------------|
| S1 | `x[:, -1, ...]` take LAST frame only | modules.py:108 | **LINEAR (selection); frame_0 has ZERO d_seg obligation** |
| S2 | bilinear resize (874,1164)→(384,512) | modules.py:109, `size=(segnet_model_input_size[1],[0])=(384,512)` | **LINEAR: fixed matrix A_seg (per channel), rank-deficient (downsample)** |
| S3 | EfficientNet-B2 U-net → 5-class logits `(B,5,384,512)`; RGB **NOT normalized** (smp default off, encoder_weights=None); stride-2 stem | modules.py:105 | **FROZEN NONLINEAR N_seg** (the only irreducible nonlinearity) |
| S4 | `argmax(dim=1)`; d_seg = mean over pixels of `[argmax(comp)≠argmax(gt)]` | modules.py:112-113 | **DECISION (discontinuous); smooth surrogate = top1−top2 logit margin, zero-set = boundary** |

`d_seg = mean_pixels 1[argmax N_seg(A_seg · frame1_comp) ≠ argmax N_seg(A_seg · frame1_gt)]`.

## 3. PoseNet path — d_pose (BANKED: R1 dxi, d_pose 0.001610)
| # | op | source | algebraic type |
|---|----|--------|----------------|
| P1 | `rearrange '(b t) c h w'` both frames | modules.py:72 | LINEAR (permute) |
| P2 | bilinear resize (874,1164)→(384,512) | modules.py:73 | **LINEAR: A_pose ≡ A_seg — THE SAME kernel** |
| P3 | `rgb_to_yuv6` | frame_utils:51-79 | **AFFINE (pre-clamp) + 2×2 box-avg + space-to-depth**; see §5 |
| P4 | `rearrange '(b t) c h w -> b (t c) h w'` → `(B,12,192,256)` (2 frames × 6ch) | modules.py:74 | LINEAR (permute) |
| P5 | `(x − 127.5)/63.75` (mean 255/2, std 255/4) | modules.py:64-65,77 | **AFFINE** |
| P6 | fastvit_t12 → 2048 → summarizer 512 → Hydra → 12-dim pose | modules.py:66-79 | **FROZEN NONLINEAR N_pose** |
| P7 | d_pose = MSE on FIRST 6 of 12 pose dims (`h.out//2`) | modules.py:84 | **QUADRATIC readout; only 6 scalars scored** |

## 4. Score (evaluate.py)
`S = 100·d_seg + √(10·d_pose) + 25·(archive.zip bytes / uncompressed)`  (evaluate.py:92; rate bytes :63).

## 5. rgb_to_yuv6 FULLY factorized (frame_utils.py:51-79) — the chroma/luma the operator flagged
Pre-clamp, YUV is an **exact linear map M** of RGB:
- **Y (luma) = 0.299 R + 0.587 G + 0.114 B**  → luma direction **ℓ = (0.299, 0.587, 0.114)** (BT.601). NOT (1,1,1)/√3.
- U = (B−Y)/1.772 + 128 = **(−0.299, −0.587, +0.886)/1.772 · RGB** + 128
- V = (R−Y)/1.402 + 128 = **(+0.701, −0.587, −0.114)/1.402 · RGB** + 128
- The **CHROMA PLANE = span{U-row, V-row}** = the 2D orthogonal complement of ℓ in RGB. (Clamps to [0,255] are inactive in the interior → the map is affine except at the gamut boundary.)
- **6 channels** = `[y00, y10, y01, y11, U_sub, V_sub]`: luma is a **2×2 space-to-depth** (4 phases, LOSSLESS full-res info) at (192,256); chroma U,V are **2×2 BOX-AVERAGED** (mean of the block) at (192,256) — **LOSSY, half-resolution.**

## 6. What the flattening REVEALS (the deep-math, not the slogan)
1. **ONE shared resize.** A_seg ≡ A_pose (modules.py:109 == :73): both scorers only ever see the witness through the *same* bilinear downsample to (512,384). ⟹ the witness's effective target space is (512,384); everything at camera-res lives in the **preimage of a rank-deficient linear map** — this IS the blind-coordinate (#401) and sub-pixel-placement (#149) exploit, now *derived* not asserted. Pull ∂/∂pixel back through the single A adjoint (#391), not two.
2. **frame_0 is seg-free.** S1 discards it ⟹ d_seg obligation ≈ 0 on frame_0 (DAG L86 / Unit-C) — the cheapest place to carry pose output.
3. **Only TWO irreducible nonlinearities per path.** SegNet = {N_seg, argmax}; PoseNet = {N_pose, MSE} (+ near-inactive YUV clamps). Every other stage (both resizes, the YUV map, space-to-depth, normalization) is linear/affine and composes into single matrices. The witness→score map factorizes as **(decision/readout) ∘ (frozen net) ∘ (fixed linear R+resize+YUV)**. The whole white-box optimization is: shape the frozen net's *input* (a linear image of our pixels) to move the decision — an inverse problem against a known oracle.
4. **The BT.601 chroma plane is the exact RGB-at-boundaries basis.** "Does color decide this boundary?" = project the SegNet margin-Jacobian ∂(top1−top2)/∂(R,G,B) onto span{U-row,V-row}. Luma-decidable boundaries (sky/road) have ~0 chroma-plane component; lane-paint / car-edge boundaries do not.
5. **PoseNet is chroma-blind below 2px.** U_sub/V_sub 2×2 box-average ⟹ chroma structure finer than 2px @ (512,384) (~4px @ camera) is INVISIBLE to PoseNet. ⟹ a fine-scale boundary-RGB carry is **pose-safe by construction** — it cannot raise banked d_pose. Chroma is the only lossy PoseNet channel (luma is lossless space-to-depth).
6. **Only 6 pose scalars, only the last seg frame** — the scored sufficient statistics are tiny; the witness is over-serving both readouts everywhere except the boundary annulus (#333, ~97% of d_seg) and the 6-dim pose subspace.

## 7. Consumers (this is the white-box the c2 levers optimize against)
- RGB-at-boundaries lever (`p0_rgb_at_boundaries_derivation`): §5 chroma basis + §6.4/6.5.
- #514 whitebox full campaign: §6 is the frozen-structure inventory the composite lattice enumerates over.
- Metal VJP (DONE, `metal_grouped_conv_backward` ~18× + `metal_fused_r_operator` transpose-VJP): differentiates N_seg/N_pose + the A adjoint — the engine that makes §6.1/6.4 measurable at n600.
- Every ∂/∂pixel or ∂margin/∂chroma claim goes through the real N via mx.vjp / the Metal VJP, verdict through the real byte-closed decode.
