# PTNC — PoseNet-Tube-Native Carrier + frame1 dual-fidelity (task #61) — PRE-REGISTRATION

**Subagent:** `task61_ptnc_frame1_dual_fidelity`. **Written BEFORE any measurement** (the
pre-registration the verdict memo will be scored against). **Authority of every number this task will
produce:** `[local CPU-torch advisory]` — exact upstream PoseNet/SegNet (`DistortionNet`) on CPU, GT
decoded via `upstream/frame_utils.yuv420_to_rgb` ONLY, S recomputed from components (the rounded field
lies). `[macOS-MLX research-signal]` for any MLX forward. **NO MPS.** `$0` unless the paired exact eval
fires (then ≤$1). `promotable=false`, `score_claim=false`, `ready_for_exact_eval_dispatch=false` until a
byte-closed archive clears the advisory gate.

**Frontier to beat (pointer, NOT hardcoded):** `0.19109982` `[contest-CPU]`, 177,169 bytes
(`lane_pr110_payload_entropy_recode_20260610`). Sub-0.15 is the secondary gate.

---

## 0. The wall #57 proved (precise)
- frame0 pose carrier WORKS: 13 KB INR, d_pose 0.0036 (beats naive low-res 20×) — BUT the coordinate-INR
  RD curve is NON-monotone with a ceiling at d_pose ≈ 0.0036 (124× above the tube 2.9e-5). More capacity
  trained WORSE (0.0036 tiny → 0.0158 small → 0.0144 medium).
- palette frame1 ALONE destroys pose: d_pose 12.14 (it is a label map, pose-blind, no luma motion).
- The binding constraint: frame1 has a DUAL constraint — SegNet argmax (d_seg) AND PoseNet luma (d_pose).

## 1. PTNC mechanism — the genuine, non-rename distinction (NO-FAKE class 6/8 guard)
The existing trainer (`score_native_train_luma_carrier.py`) ALREADY optimizes the exact 6-dim PoseNet
pose-MSE. So "optimize pose output" is NOT the PTNC novelty — claiming it would be a rename (fake). The
genuine, falsifiable PTNC distinction is the **input-domain anchor**:

- **Existing loss:** `loss = (1 - 0.5·w_pose)·pix_MSE + 50·w_pose·pose_MSE·1e4`. The `pix_MSE` term has
  weight ≥ 0.5 ALWAYS — it forces the carrier to reproduce the FULL appearance (every luma pixel),
  including pose-IRRELEVANT pixels. Capacity is spent where PoseNet is blind. This is the source of the
  0.0036 ceiling (the verdict's "spends capacity on pose-irrelevant luma").
- **PTNC loss (IDSE with the MEASURED frozen-PoseNet Jacobian):** replace the DENSE pix_MSE anchor with a
  **PoseNet-Jacobian-saliency-WEIGHTED recon anchor**: `pix_anchor = mean( w_pix · (carrier − GT)² )`,
  where `w_pix(x,y)` ∝ the per-pixel PoseNet pixel-Jacobian-norm (high where pose is sensitive, ~0 in the
  pose-null). The carrier is then FREE to be wrong wherever the Jacobian is ~0 (most of the luma field),
  so it amortizes to a smaller weight set and every trained bit lands in the pose-tube directions. The
  exact pose-MSE term stays as the objective; the Jacobian-weighting changes WHERE recon capacity is
  spent. This is literally USC's IDSE `‖J_P·(x̂−x)‖²` but with the EXACT measured atlas Jacobian, not a
  per-image Taylor approximation.

**The 3 falsifiable tests (class-2 behavior guards):**
1. The Jacobian-weighted recon loss reduces d_pose MORE PER BYTE than plain dense-MSE at matched capacity
   (the headline; if it does not, the projection is a no-op rename → FAIL).
2. A CONSTANT carrier frame fails (does not reduce d_pose vs the constant baseline).
3. Replacing the Jacobian weight field with IDENTITY (uniform weight = dense MSE) recovers the old
   behavior — proving the weight field is load-bearing, not cosmetic.

The Jacobian weight field is computed via finite differences on the EXACT frozen CPU PoseNet (the
`tac.differentiable_eval_roundtrip` differentiable-YUV6 path) per pair, or the analytic gradient through
the patched preprocess — a MEASURED field, not a fabricated constant.

## 2. frame1 dual-fidelity carrier — the wall's actual gate
frame1 must (a) land the SegNet argmax (d_seg) AND (b) carry pose luma (d_pose). Three candidate frame1
representations, RD-swept and the one minimizing FULL S is chosen:
- **(F1a) PTNC-on-frame1 confined to the seg-null:** a per-pair RGB carrier for frame1, but the luma it
  paints is confined to the **interior of each SegNet argmax cell** (the margin-polytope interior the
  argmax can't see). The argmax is set by the seg generator / palette; PTNC adds interior luma texture
  only where it does not flip a pixel (margin > 0 band). Test: d_seg preserved (≈ palette's) while d_pose
  drops vs palette's 12.14.
- **(F1b) low-res GT luma carrier:** downsampled GT frame1 luma (+ bilinear or learned upsample),
  byte-counted, composited onto the seg-argmax-fixed chroma. #56 hinted this reached d_pose 0.0007 (better
  than the INR 0.0036) at higher byte. Compare RD.
- **(F1c) palette (baseline):** the #57 dead end (d_pose 12.14) — the reference.

## 3. PREDICTION (pre-registered)
PTNC breaks the 0.0036 ceiling toward the tube (≤ ~1e-4) at ≤ ~15 KB for frame0, because the
Jacobian-weighted anchor stops wasting capacity on pose-null luma. For the FULL candidate, the frame1
dual-fidelity carrier (best of F1a/F1b) gets the candidate's total d_pose low enough that total
`S = 100·d_seg + sqrt(10·d_pose) + 25·B/D` beats frontier 0.19110 — OR it does not, and frame1 needs
near-full-RGB fidelity to hold the tube.

## 4. KILL / DEFER CRITERION (pre-registered, honest)
If PTNC cannot get the FULL candidate's total S below frontier (0.19110) — or sub-0.15 — at a byte cost
that preserves the rate advantage (i.e. frame1 needs near-full-RGB fidelity, converging to HNeRV-class),
then **DEFER to lever C** and record the honest finding: pose-relevant luma is NOT cheap enough; the
score-native pose axis converges to HNeRV-class. This is a DEFER, not a KILL (Forbidden premature KILL).

**Lever C reactivation campaign (pre-registered):** fresh-init, JOINTLY-trained smaller seg+pose frame1
carrier (a single per-pair-latent CONVOLUTIONAL decoder, NOT a coordinate-MLP — the non-monotone RD curve
says coordinate-MLP capacity does not help) trained against BOTH SegNet (d_seg) AND PoseNet (d_pose) on
the contest video, with the archive grammar + scorer-free inflate declared export-first. Predicted byte
band: 40–120 KB (between the score-native 85 KB and the frontier 177 KB). The open question: can a
per-pair-latent conv decoder reach the frontier's d_seg=5.6e-4 + d_pose=2.9e-5 below 177 KB.

## 5. Build plan
- Extend `src/tac/boundary_math/amortized_luma_carrier.py` (numpy-portable) + the torch trainer with the
  PTNC Jacobian-weighted loss + the frame1 seg-null confinement. NEW module
  `src/tac/boundary_math/posenet_jacobian_saliency.py` (the measured per-pixel pose-Jacobian field +
  numpy-portable weight map). RD-sweep tool. ≥15 behavior tests. Byte-closed candidate assembly +
  scorer-free inflate + lossless parity + exact advisory S. Conditional paired eval if advisory beats
  frontier.

## 6. Wire-in (Catalog #125) — to be filled in the verdict
sensitivity-map / Pareto / bit-allocator / cathedral-autopilot / continual-learning / probe-disambiguator.
