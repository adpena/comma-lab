# UPSTREAM SCORER — FULL-DIMENSION RE-READ (chroma · luma · YUV · hydra · order · dtype) — 2026-07-10

**Operator (2026-07-10):** *"review upstream evaluate.py and modules.py again but not just from
resolution perspective but also chroma and luma and yuv and hydra and order and all dimensions"* — in
light of the palette finding (`palette_artifact_probe_20260710.md`): SegNet argmax is
TEXTURE/CONTEXT-dominated, Road/Lane NEVER win on colour alone (constant-colour tiles → Undrivable
195/216). Re-read every scored dimension from the ACTUAL sources, verify EXPLOITED / PRICED-NOT-EXPLOITED
/ OVERLOOKED.

**Sources read line-by-line (verified, not memory):** `upstream/evaluate.py` (113 L),
`upstream/modules.py` (187 L), `upstream/frame_utils.py` (271 L). **$0, read + one cheap CPU measurement.**
**NO upstream edits.** **Pointer contest-CPU 0.19110 UNMOVED — this is MEANS.**

**STORES CONSULTED:** `palette_artifact_probe_20260710.md` (context-domination) ·
`cascade_c_prime_frame_1_segnet_waterfill_...20260526.md` (frame-0 seg_delta=0.0 structural) ·
`cascade_c_posenet_null_segnet_region_waterfill_..._landed_20260526.md` (blue/DCT-chroma = pose-null
bottom decile) · MEMORY L68/L73/L74/L75 · #149 camera-res placement · #276 chroma DOF ·
`docs/operating_manual_craft_handoff.md` (§4 re-derive-from-primary, §5 label MEASURED/DERIVED).

---

## THE EXACT SCORED PATH (from code, every op)

```
comp uint8 RGB (B,2,874,1164,3)          # TensorVideoDataset reads .raw uint8; camera_size=(W1164,H874)
  └ DistortionNet.preprocess_input        # modules:143-148  'b t h w c -> b t c h w' .float()  [0-255]
     ├ PoseNet.preprocess_input            # modules:70-74
     │   x = interpolate(BOTH frames, size=(384,512), bilinear, NO align_corners, NO antialias)
     │   yuv = rgb_to_yuv6(x)              # frame_utils:51-78  → 6ch @ 192×256
     │   → (B, 12, 192, 256)              # 2 frames × 6ch = 12; order [y00,y10,y01,y11,U,V]×[f0,f1]
     │   (x-127.5)/63.75 → FastViT-T12 → summarizer(2048→512) → Hydra → pose[:6]   (MSE on 6 dims)
     └ SegNet.preprocess_input             # modules:107-109
         x = x[:, -1, ...]                # LAST frame ONLY (frame_1); frame_0 UNSEEN by SegNet
         x = interpolate(size=(384,512), bilinear, NO AA)   # RGB 0-255, NO normalization
         → EfficientNet-B2 U-net → 5-class logits @ 384×512 → argmax (only argmax matters)
score = 100·mean_seg(argmax≠) + √(10·mean_pose(MSE6)) + 25·|archive.zip|/Σfiles(videos/)   # evaluate:92
```

---

## DIMENSION TABLE (each: exact code semantics → status → note)

| # | Dimension | Exact code semantics | Status | Note |
|---|---|---|---|---|
| A | **Rate: numerator** | `(submission_dir/'archive.zip').stat().st_size` ONLY (`evaluate:63`); inflate.py/.sh NOT sized; NO time term | EXPLOITED | rule-118 compile-generator known |
| A2 | **Rate: denominator** | `Σ stat().st_size for ALL files in videos/` (`evaluate:64`) = fixed 37,545,489; 25× coeff | PRICED/N-A | constant; not a lever |
| B | **Score reduction** | `dists += dist.sum()` then `/batch_sizes` (`evaluate:81-92`); pose = `√(10·mean)` | EXPLOITED | order-independent; √ → pose marginal→∞ near 0 (op-point rule) |
| C | **SegNet frame select** | `x[:, -1, ...]` (`modules:108`) — ONLY frame_1 scored by seg | **OVERLOOKED (v8)** | frame_0 has ZERO d_seg obligation; validated seg_delta=0.0. Old selector-codec used it; witness/v8 does NOT yet. **RANK 1** |
| D | **SegNet input space** | raw RGB 0-255, NO norm; `interpolate(384,512) bilinear NO antialias` (`modules:107-109`) | PRICED/partial | 874→384 (2.28×) downscale w/o AA → aliasing; argmax @384×512. #149 placement partial |
| E | **SegNet decision geometry** | argmax over 5 classes; texture/context-dominated (palette probe: const colour → Undrivable 195/216, Road=1 Lane=0) | KNOWN (07-10) | colour alone can't make Road/Lane win → seg lever = TEXTURE not colour |
| F | **PoseNet luma** | `y00,y10,y01,y11` = 2×2 POLYPHASE of 384×512 luma → 4ch @192×256 (`frame_utils:74-77`) | EXPLOITED | luma carried at FULL res (4 phases lossless); phase = ego-dist = screw ξ (L73) |
| G | **PoseNet chroma** | `U_sub,V_sub` = 2×2 BOX-AVERAGE → 192×256 (`frame_utils:65-72`); MEASURED 11.1× less pose-sensitive/plane; high-freq above 2×2 grid EXACTLY pose-null | **OVERLOOKED (joint)** | chroma-high-freq FREE for pose × NEEDED for seg (E). **RANK 2** |
| H | **YUV coefficients** | BT.601 FULL-range Y=.299R+.587G+.114B; U=(B−Y)/1.772+128; V=(R−Y)/1.402+128; clamp 0-255 (`frame_utils:60-63`) | PRICED | blue-chroma∝U, red∝V modes known; clamp dead-zone at saturated colours = pose-free (minor) |
| I | **PoseNet norm** | uniform `_mean=127.5 _std=63.75` ALL 12ch (`modules:64-65,77`) | N/A | no per-channel dead-zone asymmetry |
| J | **uint8 boundary** | comp stored uint8 raw (`frame_utils:218-231`); witness rounds to uint8 @camera-res | EXPLOITED | R uint8-STE known; symmetric 1/255 step |
| K | **Hydra head** | `Linear(512,32)→ResBlock→Linear(32,12)`, score uses `[...,:6]` (`modules:82-84`); dims 6-11 UNUSED | N/A | output-null; nothing we ship touches head. 6 used dims = the stored-target sidecar |
| L | **Order dims** | `seq_len=2` non-overlap (`frame_utils:10`); 600 pairs/1200 frames; frame-within-pair order fixed (f0→f1 = pose direction) | EXPLOITED | pair/batch order irrelevant (sum-then-÷); frame order = pose sign |
| M | **PoseNet frames** | BOTH frames fed (`modules:71-74`); pose = differential | EXPLOITED | f0/f1 pose energy balanced (MEASURED f1/f0=0.86×) |

---

## THE CHEAP MEASUREMENT — per-channel PoseNet Jacobian (luma vs chroma)

`[macOS-CPU advisory . real-GT-pair n≤4 . NON-PROMOTABLE]` — PoseNet weights + `gt_n6.npz` real pairs.
Per-channel energy = Σ over 6 scored pose dims of (∂pose_k/∂yuv_channel)² summed spatially:

```
  f0_y00 5.53e-3  f0_y10 5.50e-3  f0_y01 5.38e-3  f0_y11 5.37e-3   f0_U 4.57e-4  f0_V 7.15e-4
  f1_y00 4.67e-3  f1_y10 4.93e-3  f1_y01 4.60e-3  f1_y11 4.86e-3   f1_U 2.44e-4  f1_V 4.20e-4
  LUMA total (8 planes) = 4.08e-2   CHROMA total (4 planes) = 1.84e-3
  → luma/chroma per-plane sensitivity = 11.1×   (aggregate 22.2×)   frame1/frame0 = 0.86×
```

**Two independent chroma-cheapness facts stack:** (1) at the yuv6 input, chroma is **11.1× less
pose-sensitive per plane** (measured); (2) chroma is a 2×2 BOX-AVERAGE of RGB (`frame_utils:65-72`) →
its frequency response is a sinc that **NULLS at the 2×2 grid**, so RGB chroma detail ABOVE that grid is
**exactly pose-invariant** (derived from code, not measured). Cross-check with stores: blue-chroma /
DCT-chroma modes are the measured PoseNet-null bottom decile (`cascade_c_..._landed_20260526.md`).

---

## TOP-3 OVERLOOKED (ranked by expected value)

### 1. frame_0 is SegNet-FREE — the witness/v8 has never priced it (RANK 1)
`x[:, -1, ...]` (`modules:108`) means SegNet sees ONLY frame_1. **frame_0's d_seg constraint is
identically ZERO** (structurally validated: all 87 frame-0 perturbation modes show `seg_delta=0.0`,
`cascade_c_prime_..._20260526.md`). The old PR106/PR110 **selector codec** exploited this (Atick-Redlich
asymmetric channel); the **witness / v8** paradigm does NOT — it renders both frames symmetrically. This
is a large unpriced DOF: **half the rendered pixels (all of frame_0) owe nothing to d_seg** and can be a
PURE pose-carrier — spend frame_0's entire byte/fidelity budget on the 6 pose scalars, render it as
cheaply/coarsely as pose allows with NO seg-texture obligation. v8's per-class carriers currently pay
seg-fidelity on both frames; frame_0 should drop to a pose-only representation. **EV: high** — directly
cuts v8 seg-carrier bytes ~in half on the frame axis, orthogonal to every other lever.

### 2. Chroma-high-freq: FREE-for-pose × NEEDED-for-seg — the palette-dilemma escape (RANK 2)
The palette finding said flat colour can't make Road/Lane win (seg is texture-dominated) → v8 "must carry
per-pixel texture." The scorer geometry says WHERE that texture is cheapest: **chroma above the 2×2
yuv-grid is exactly pose-null** (G, measured 11×; derived exactly-null at high freq) **yet feeds SegNet at
FULL 384×512 RGB resolution** (D — SegNet takes RGB, not yuv). So a **chroma high-frequency texture
carrier** gives SegNet the context/texture it needs to hold Road/Lane/Movable argmax at **≈zero pose
cost**. This is the one lever that resolves the palette dilemma without paying pose. Under-exploited:
prior chroma work (blue-chroma modes) used chroma as a pose-NULL *perturbation*; here it is a **seg
TEXTURE carrier**, a different use. **EV: high** — a seg lever that is pose-free by construction. Owed: a
through-R measurement that a chroma-high-freq dither reduces d_seg on the flat-paint-misread classes
(Movable/MyCar/Road, per palette per-class table) without moving d_pose.

### 3. No-antialias bilinear downsample → predictable aliasing (RANK 3)
BOTH nets do `interpolate(size=(384,512), mode='bilinear')` with `antialias=False` (default) on a 2.28×
downscale from 874×1164 (`modules:73,109`). Sub-Nyquist detail folds through a KNOWN, fixed 2-tap kernel
— boundary placement can be tuned so a witness edge lands on the aliasing grid that SegNet's downsample
actually samples (vs being averaged away). Partially covered by #149 camera-res placement, but the
specific no-AA bilinear phase is not yet a placement constraint. **EV: medium** — refinement of existing
placement lever, not a new axis.

---

## VERDICT (verdict_scope: FORMULATION — scorer-dimension characterisation; NO kill)

The two RANK-1/2 overlooked levers are **structural asymmetries in the scorer, both pose-free seg DOFs**,
and both are UNPRICED in the current witness/v8 build (they were priced only in the retired selector
codec). They compose (frame-0 seg-freedom is a FRAME axis; chroma-high-freq is a CHANNEL×FREQUENCY axis).
Neither is a score yet — the pointer moves only through a byte-closed n600 exact eval. Every claim here is
MEASURED (Jacobian, seg_delta=0.0, palette context-domination) or DERIVED-from-code (box-average null,
frame select). Pointer 0.19110 UNMOVED (means).

## Triality legs
- **DAG:** FEED-alldim (this landing).
- **Equations:** `posenet_luma_chroma_sensitivity_asymmetry_v1` (registered; VERIFIED_VIA_EMPIRICAL_ANCHOR
  — luma/chroma 11.1× per-plane + exact high-freq chroma null + frame-0 seg-freedom).
- **DSL/verdicts:** no trainer knob (this is a scorer-geometry characterisation); routes #385 build-wave
  (frame-0 pose-only carrier + chroma-high-freq seg-texture carrier for v8).

**Pointer 0.19110 UNMOVED (means).**
