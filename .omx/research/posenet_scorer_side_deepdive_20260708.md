# POSENET SCORER-SIDE DEEP DIVE — how to fool the PoseNet (texture-statistics recommendation, scene-invariance bound, ranked failure modes)

**Date:** 2026-07-08 · **Axis:** research / design memo, `[no-triality]` · **Read-only** (no run dirs
touched, no jobs). **Pointer contest-CPU 0.19110 UNMOVED — MEANS.** Answers §6(a) of
`pose_legible_witness_aperture_design_20260708.md` (494f1a35c) and supplies the texture spec the T-probe
(A0T/A2T) consumes.

STORES CONSULTED: `pose_legible_witness_aperture_design_20260708` (the aperture diagnosis + 3-layer T/D/S
design + $0 probe) · `pose_mladder_depthwarp_measured_20260708` (A0 1.685 / A2 1.486 / A2+ 1.223,
corr(d_pose,|t|) = −0.446/−0.676, off-plane mass 0.5%) · `upstream/modules.py` + `upstream/evaluate.py` +
`upstream/frame_utils.py` (READ verbatim, line cites below) · `src/tac/clip_profile.py` (comma2k19 RAV4
provenance) · CLAUDE.md L80 (class order) + §"Exact scorer architectures" · web literature (FastViT ICCV23,
self-supervised VO domain-gap, structure-tensor / aperture, Lucas-Kanade subpixel). Every number tagged
**MEASURED(read)** (from code/introspection), **DERIVED** (arithmetic from measured facts), **INFERRED**
(literature/framing), or **ASSUMED**.

---

## 0. HEADLINE (the actionable answer)

**The texture the probe should paint is ISOTROPIC (2-D, broadband-in-orientation) band-limited noise at a
spatial period of ~24–32 px on the 874×1164 render grid (sweep 16/24/32/48), zero-mean, peak amplitude
~8–16 uint8 steps (sweep 4/8/16/32), applied to BOTH luma and chroma, advected ξ-consistently.** The band
is bounded HARD below by the scorer's own resolution path (DERIVED): the PoseNet luma readout grid is
384×512 reached by a **non-anti-aliased bilinear 2.276× downsample from 874** — so any render texture with
period < ~9 px @874 (2× the 384-Nyquist) is **aliased or killed** before it ever reaches the network. It is
bounded above by the requirement that texture stay *within* a Morse cell (< cell size). The scene-invariance
bound is honest and non-zero: comma's PoseNet keys partly on scene-appearance priors, so even perfect texture
leaves a residual = the net's semantic-prior weight — **the probe (A0T) measures exactly this residual and
is decisive either way.**

---

## 1. WHAT POSENET ACTUALLY COMPUTES (MEASURED, from the real code)

### 1a. The exact input pipeline (modules.py:70–80, frame_utils.py:50–78)
Submission inflate emits frames at `camera_size = (1164, 874)` uint8 (frame_utils.py:11). Then per pair
(seq_len=2, frame_utils.py:10):
1. `DistortionNet.preprocess_input` (modules.py:143–148): `(B,2,874,1164,3) → (B,2,3,874,1164)` float.
2. `PoseNet.preprocess_input` (modules.py:70–74):
   - flatten frames: `(B·2, 3, 874, 1164)`.
   - `F.interpolate(size=(384, 512), mode='bilinear')` — **`segnet_model_input_size=(512,384)` → size=(H=384,
     W=512)**. **MEASURED: no `antialias=` arg and no `align_corners=` arg → torch defaults
     `antialias=False, align_corners=False`.** This is a 874→384 (2.276×) / 1164→512 (2.273×) decimation with
     a 2-tap bilinear kernel that reads only a 2×2 source neighborhood per output pixel — it does **not**
     anti-alias a >2× downsample. **DERIVED: render spatial frequencies with period < ~2×2.276 ≈ 4.5 px @874
     are dropped/aliased; period < ~9 px @874 is near-Nyquist and unreliable.** This is the binding band gate.
   - `rgb_to_yuv6(x)` (frame_utils.py:51–78): from the 384×512 RGB, compute Y=0.299R+0.587G+0.114B (BT.601,
     clamped 0–255), U=(B−Y)/1.772+128, V=(R−Y)/1.402+128. Then **luma space-to-depth**: the 4 phases
     `y00,y10,y01,y11` of every 2×2 luma block become 4 channels at **192×256** (full 384×512 luma is
     preserved *losslessly*, repacked into 4 channels — NOT downsampled). **Chroma is genuinely subsampled**:
     `U_sub,V_sub` = 2×2 average → 192×256. Output 6 ch/frame.
   - reassemble both frames: **`(B, 12, 192, 256)`** — this is the actual FastViT input.
3. Normalization (modules.py:64–65, 77): `(x − 127.5)/63.75`, per channel, **all 12 channels share the same
   mean/std** (buffers `_mean=255/2`, `_std=255/4`). Input lands ≈ [−2, +2].

**Consequence for texture design (DERIVED):** luma detail is carried at the **384×512** effective grid
(via the 4 space-to-depth channels); chroma at **192×256**. Texture that varies at the *2-px luma-block*
scale of the 384 grid degenerates into pure *inter-channel DC* across y00..y11 with **zero within-channel
spatial gradient** — dead for flow (see §3). Texture must vary at a scale **coarser than the 2×2 luma
block** (period ≳ 4 px @384) so each yuv channel itself carries a trackable spatial gradient.

### 1b. FastViT-T12 architecture + effective band-pass (MEASURED via timm introspection)
`timm.create_model('fastvit_t12', in_chans=12, num_classes=2048)`, traced on a (1,12,192,256) tensor:

| stage | op (MEASURED) | out spatial | out ch | cum. stride |
|---|---|---|---|---|
| stem[0] | MobileOne 3×3 **stride-2** conv 12→64 | 96×128 | 64 | /2 |
| stem[1] | MobileOne 3×3 **stride-2** depthwise 64→64 | 48×64 | 64 | /4 |
| stem[2] | MobileOne 1×1 stride-1 64→64 | 48×64 | 64 | /4 |
| stage0 | RepMixer blocks (depthwise reparam token-mix) | 48×64 | 64 | /4 |
| stage1 | downsample + RepMixer | 24×32 | 128 | /8 |
| stage2 | downsample + RepMixer | 12×16 | 256 | /16 |
| stage3 | downsample + **attention** (global over 6×8) | 6×8 | 512 | /32 |

Then global pool → 2048 (num_classes head) → `summarizer` Linear(2048→512)+ReLU+ResBlock (modules.py:67) →
`Hydra`: ResBlock(512) → per-head `in_layer` Linear(512→32)+ReLU → `res_layer` → `final_layer`
Linear(32→12) (modules.py:45–59). Total FastViT params **8.63 M** (MEASURED).

**Band-pass reading (DERIVED + INFERRED):**
- **The stem's first op is a stride-2 3×3 conv on the 192×256 yuv grid** — again a decimating filter with
  no anti-alias, so yuv-grid frequencies near period 2–4 px alias in the stem. The first grid at which flow
  is *resolved* as spatial structure is stage-0's **48×64** (= 4× the 192 grid ⇒ ~4 px @192 = ~8 px @384 =
  ~18 px @874). Below that the stem records texture *energy statistics*, not resolvable displacement.
- **RepMixer = localized (depthwise conv) receptive field**; the paper adds large-kernel depthwise convs in
  FFN/patch-embed to grow RF cheaply (INFERRED, FastViT ICCV23). Only **stage-3 attention is global** (over
  the 6×8 grid) and the **global pool** aggregates everything. **DERIVED: flow information anywhere in the
  frame can reach the 6 scored outputs** (global pool + late attention) — so texture does *not* need to be
  everywhere to be read, but denser well-conditioned texture = more flow-carrying tokens = higher SNR into
  the pool. There is no known closed-form "frequency response"; the operative fact is the two cascaded
  non-anti-aliased stride-2 stages (bilinear 874→384, then stem 192→96→48) set a **low-pass corner around
  a resolved scale of ~8 px @384 / ~18 px @874**, with everything finer folded into unreliable aliased
  energy.

### 1c. The scored quantity (MEASURED, modules.py:82–84, evaluate.py:90–92)
`PoseNet.forward` → Hydra 'pose' head → **12-dim** output; `compute_distortion` takes
`out[..., :out//2]` = **first 6 dims** and returns **per-pair MSE** over those 6. evaluate.py aggregates
`√(10·mean_pose_MSE)`. **INFERRED: the 6 scored dims are the 6-DOF relative 2-frame ego-pose (3 rot + 3
trans); the other 6 are aux/uncertainty** (unlabeled in code). **CRITICAL (MEASURED framing):** the target
is `PoseNet(real_pair)`, the frozen authority — **not** physical ground-truth motion. We must make the net
emit *the same 6 numbers it emits on the real scene*. Any appearance-prior the net couples into those 6
numbers becomes residual if our synthetic render mismatches it (see §5).

---

## 2. POSENET'S TRAINING DOMAIN + SCENE-INVARIANCE BOUND

### 2a. Provenance (MEASURED + INFERRED)
evaluate.py:9 argparse describes "a comma2k19 compression submission"; `clip_profile.py:20,240` pins the
clip as the **comma2k19 RAV4 segment** (37,545,489 B = the rate denominator). **INFERRED: the PoseNet
weights (`models/posenet.safetensors`) are comma's own ego-motion estimator trained on comma's real
driving fleet / comma2k19-family data** — a FastViT-T12 12-ch two-frame ego-motion regressor. It is a
*supervised-target* frozen teacher here; its training objective (self-supervised photometric vs supervised
against a SLAM/INS pose) is not in-repo (ASSUMED: comma-internal). It is a **real-driving-domain** net.

### 2b. What the VO literature says about scene-invariance (INFERRED)
Self-supervised / learned monocular ego-motion nets (SfMLearner lineage) are **measurably domain-sensitive**:
- ORB-SfMLearner needs **"selective online adaptation"** to hold accuracy across domains (KITTI→vKITTI)
  — i.e. the pose net does NOT transfer cleanly to a shifted appearance domain without adaptation.
- AF-SfMLearner introduces **"appearance flow"** precisely because brightness/appearance shifts break the
  photometric assumption and hurt cross-dataset generalization.
- Two-stream / sparsity-augmentation pose nets improve generalization by **regularizing the pose net away
  from appearance** — evidence that vanilla pose nets *lean on appearance/scene layout*, not flow alone.

**THE BOUND (the honest answer to §6a "is it scene-invariant"):** a trained driving ego-motion net is
**partially** flow-driven and **partially** scene-prior-driven (horizon row, road-plane vanishing geometry,
vehicle-scale cues, global luma/chroma statistics). A synthetic render that depicts the *correct flow*
densely will drive the flow pathway well, but will leave a **non-zero residual equal to the net's
scene-prior weight**. That weight is **unknown a priori** and is exactly what the probe measures:
- **If A0T ≪ A0 (1.685):** the deficit was flow-observability (aperture) — texture rescues it, residual
  small, build the pose-legibility term.
- **If A0T ≈ A0:** semantic priors dominate the 6 outputs — a texture-only synthetic render cannot elicit
  the target, and pose reverts to a budget item (design §4 fallback + #238).

This bound is consistent with the measured M-ladder: A2/A2+ floored at ~1.2–1.5 on a **flat** render because
a low-DOF warp of a flat cartoon cannot reach the real-pair PoseNet target — but that experiment **could not
separate** "aperture deficit" from "semantic-prior deficit" because the flat render supplied *neither* dense
flow *nor* real appearance. A0T is the clean separator (adds dense flow, holds cartoon appearance).

---

## 3. OPTIMAL TEXTURE STATISTICS FOR FLOW READABILITY (the recommendation the probe consumes)

### 3a. Orientation content — ISOTROPIC, not oriented (DERIVED from the aperture problem + structure tensor)
The **aperture problem**: through a local window, only the flow component **normal to the intensity gradient**
is observable; a 1-D (single-orientation) texture reveals nothing about flow **parallel** to its edges.
The **structure tensor** J = Σ∇I∇Iᵀ must be **well-conditioned (both eigenvalues large — "corner-like")**
for the full 2-D local flow to be observable; a single-orientation grating gives one large + one ~zero
eigenvalue (rank-deficient → aperture-blind along one axis).

Ego-motion flow on this clip is a **2-D radial field** (forward translation → expansion about the FOE +
yaw/pitch rotation), so the local flow direction **varies across the frame**. Therefore the texture must be
**isotropic (broadband in orientation)** so that J is well-conditioned everywhere and *every* local flow
direction is observable. **Recommendation: isotropic band-limited noise (2-D), NOT oriented gratings /
stripes.** (Oriented texture would only help if the local flow were 1-D and its direction known — it is
neither.)

### 3b. Spatial band — the R-chain sets a hard window (DERIVED)
Scale-map from render to the two readout grids:
- render 874 → bilinear → **384** (factor **2.276**, non-anti-aliased) → luma readout grid.
- 384 → space-to-depth /2 → **192** yuv channel grid; render→yuv factor = 2.276×2 = **4.55**.

Constraints:
1. **Survive the 874→384 bilinear (binding):** period must exceed 2× the 384-Nyquist = ~4.5 px @874 to
   exist at all, and ≳ **9 px @874** to avoid near-Nyquist attenuation/aliasing.
2. **Carry a within-channel gradient (avoid space-to-depth degeneracy):** period ≳ 4 px @384 ⇒ ≳ **~9 px
   @874** (same order).
3. **Be resolved as displaceable structure by the stem** (first resolved grid 48×64 = ~18 px @874): period
   ≳ **~18 px @874** for the *displacement* (not just the energy) to be read.
4. **Stay within a Morse cell** (within-cell texture, does not touch separatrices): period ≪ cell size
   (cells here are 10²–10³ px @874) ⇒ upper bound ~48 px is safe.

**Recommendation: band period ∈ [16, 48] px @874, sweet spot ~24–32 px @874** (≈ 5–7 px at the 192 yuv grid
— safely above space-to-depth degeneracy, safely below the aliasing corner, resolved by the stem). **Probe
sweep: {16, 24, 32, 48} px @874.** (16 is the risk edge — expect it to under-perform if aliasing bites;
48 is safe but coarser flow.)

### 3c. Amplitude — above the quant floor, below the SegNet margin (DERIVED)
- **Floor:** post-bilinear the zero-mean texture is attenuated; it must still land ≥ **1–2 uint8 LSB** at the
  384 grid to beat the uint8 quantization noise (~0.5 LSB) + resize noise. **Recommendation: render peak
  amplitude sweep {4, 8, 16, 32} uint8, predicted sweet spot ~8–16.**
- **Ceiling (d_seg guard):** on f1 the texture must stay **sub-SegNet-margin** so argmax does not flip
  (#141 margin field p50≈0.9 ⇒ large headroom; **MEASURE flips, do not assume**). f0 is SegNet-free
  (modules.py:108 uses `x[:,-1]` = last frame only) → f0 texture amplitude is *unconstrained by d_seg*.
- **Both frames, advected:** f1 texture = f0 texture advected by the per-cell model flow (design §2 T);
  amplitude equal in both frames (pair-consistency law — mixing real+cartoon measured 10.42, DEAD).

### 3d. Chroma (DERIVED + guard)
PoseNet reads `U_sub, V_sub` (2 of 6 channels, at 192×256). **Recommendation: texture chroma too** (or at
minimum match real-driving chroma DC), and **sweep chroma-on vs chroma-off** so the probe *measures* chroma
dependence rather than assuming it. Chroma is subsampled (lower weight likely) but non-zero.

### 3e. Global-statistics match (DERIVED — BN operating point, see §5.2)
FastViT BN + the summarizer/Hydra `AllNorm` (BatchNorm over flattened features, modules.py:28–33) run in
**eval with frozen running stats calibrated on real driving frames**. **Recommendation: add a "match global
luma/chroma DC + variance to real-driving frames" toggle to the sweep** so a synthetic-statistics shift does
not push activations off the frozen-BN operating point.

**Probe grid (compact):** {band 16/24/32/48} × {amp 4/8/16/32} × {chroma on/off} × {DC-match on/off},
isotropic noise, ξ-advected, on the existing A0T harness; report d_pose vs A0=1.685 (n24) + d_seg-flip guard.

---

## 4. THE ANTI-CORRELATION (d_pose worst on LOW-|t|) — the small-flow readability threshold (DERIVED)

Measured: corr(d_pose, |ξ_translation|) = **−0.446 (n24) / −0.676 (n8)** — small-motion pairs are the
ill-conditioned tail. Mechanism (DERIVED):

Apparent luma change from a flow u is ΔI ≈ ∇I·u (brightness-constancy linearization). To be *readable*,
ΔI must exceed the readout noise floor ε ≈ 0.5–1 uint8 LSB (quant + bilinear-resize).
- **Flat cell interior:** ∇I ≈ 0 ⇒ ΔI ≈ 0 for **any** |u| ⇒ interior flow **unobservable at all magnitudes**;
  only boundary normal-flow survives, and even there the readable signal |∇I|·(u·n) shrinks with |u|, so the
  **smallest flows drop below ε first** — exactly the measured low-|t| tail.
- **Textured interior:** for isotropic texture amplitude A (uint8) and period λ (px @readout grid), peak
  gradient |∇I| ≈ 2πA/λ per px. Minimum readable flow: **u_min ≈ ε·λ / (2πA)**.
  - Worked (DERIVED): A=8, λ=8 px @384, ε=1 LSB ⇒ **u_min ≈ 8/(2π·8) ≈ 0.16 px**. A=16, λ=8 ⇒ **~0.08 px**.
  - So texture makes ~0.1-px ego flows readable, precisely rescuing the low-|t| tail. On a flat render that
    same displacement is unreadable in interiors (u_min → ∞) — the anti-correlation IS the aperture deficit.

**Prediction (labeled PREDICTION):** if the aperture diagnosis is right, A0T should *most* improve the
low-|t| pairs (flatten the −0.45 correlation toward 0). The probe should **report d_pose stratified by |t|**,
not just the mean — a selective low-|t| improvement is the diagnostic signature; a uniform-but-small
improvement points at a global scene-prior residual (§5.1) instead.

---

## 5. ADVERSARIAL HONESTY — ranked failure modes (even with perfect texture)

**#1 — PoseNet keys on absolute scene content / semantic priors (HIGHEST risk).**
Evidence: self-supervised VO domain-gap literature (§2b: online adaptation / appearance-flow needed
cross-domain); the **12-ch input mixes both frames at conv-1 (flow-capable) but also exposes each frame's
absolute content** (scene-layout-capable); M-ladder A2/A2+ floored ~1.2 on a flat cartoon (reachable set of
a low-DOF warp does not contain the real-pair target). If the 6 outputs are strongly coupled to horizon
row / road-plane geometry / vehicle-scale / learned road appearance, a cartoon-textured render mismatches
those and leaves a large residual. **This is the make-or-break; A0T measures it directly.** Mitigations if
partial: render class/depth-stratified geometry faithfully (design D — correct horizon, road plane), and let
the 6-DOF solve (design S) calibrate the residual.

**#2 — Normalization / frozen-BN operating-point shift (MEDIUM).**
FastViT BN + `AllNorm` run eval with **frozen running stats from real driving frames** (modules.py:28–33,
evaluate.py:52 `.eval()`). A synthetic render with different global luma/chroma mean/variance pushes
activations off that operating point → distorted 6 outputs even with correct flow. Mitigation: §3e DC/variance
match toggle. This is *silent* (no error) — must be swept, not assumed.

**#3 — Chroma dependence (MEDIUM-LOW).**
PoseNet reads U_sub/V_sub. A grayscale or wrong-chroma render leaves 2/6 channels constant; if the net
learned chroma flow/appearance cues, that is residual. Mitigation: §3d chroma texture + chroma-DC match;
the chroma-on/off sweep measures the magnitude.

**#4 — Texture aliasing / band mismatch through the non-anti-aliased R chain (MEDIUM, implementation not
paradigm).** Too-fine texture (< ~9 px @874) aliases in the bilinear 874→384 *and* the stem stride-2 →
**false flow** (worse than no texture: aliased patterns move incoherently under advection). Too-coarse →
under-resolved, weak signal. Controlled by the §3b band sweep; the failure signature is d_pose *rising* at
the fine end.

**#5 — Space-to-depth phase degeneracy (LOW).**
Texture varying at the 2-px luma-block scale @384 becomes pure inter-channel DC (zero within-channel
gradient) → dead for flow (§1a). Avoided by keeping period ≳ 4 px @384 (already in the §3b band).

**#6 — f1 texture flips SegNet argmax → d_seg regression (LOW-MEDIUM).**
Sub-margin amplitude should prevent (f0 is SegNet-free; only f1 is read by SegNet, `x[:,-1]`). Guarded by the
probe's mandatory d_seg-flip count vs untextured — **measure, don't assume** (#141 margin p50≈0.9 gives
headroom but the tail matters).

**#7 — The 6 scored dims include non-pose aux the render can't touch (LOW).**
If any of dims 0..5 encode something other than geometric ego-pose (INFERRED they are the 6-DOF pose, but
unconfirmed in-repo), texture cannot address it. The probe implicitly tests this: if A0T plateaus above a
floor uncorrelated with |t|, suspect a non-flow component in the scored 6.

---

## 6. BOTTOM LINE FOR THE BUILD PHASE

1. **Texture spec (probe consumes):** isotropic band-limited noise, period **24–32 px @874** (sweep
   16/24/32/48), zero-mean peak amp **8–16 uint8** (sweep 4/8/16/32), **luma + chroma**, ξ-advected
   f0→f1, with a **global-DC-match** toggle and a **chroma-on/off** toggle. Report d_pose **stratified by
   |t|** + d_seg-flip guard.
2. **Scene-invariance bound (honest):** comma's driving PoseNet is partially flow-driven, partially
   scene-prior-driven; perfect texture leaves a residual = the net's semantic-prior weight, which is
   **unmeasured** and which **A0T decides** (A0T ≪ 1.685 ⇒ build the term; A0T ≈ 1.685 ⇒ pose = budget item).
3. **Ranked failure modes:** semantic-prior dominance (#1, make-or-break, directly probed) > frozen-BN
   statistics shift (#2) > chroma dependence (#3) > aliasing/band mismatch (#4) > space-to-depth degeneracy
   (#5) > d_seg flip (#6) > non-pose aux dims (#7).

**DSL/lever obligation (for the BUILD phase, not this memo):** if the probe is GREEN and a pose-legibility
texture term/stage is added to the witness, it MUST land as a `Lever` factory in
`src/tac/witness_dsl/curriculum_dsl.py` (never a hand-added trainer flag) with the band/amp/chroma/DC-match
as swept parameters, per the triality "DSL HOLDS every designed lever" discipline. This memo fires no lever
(`[no-triality]` — research/design only).

**FINAL STATE:** read-only; no run dirs touched; no jobs; no score claimed. **Pointer 0.19110 UNMOVED —
MEANS.**
