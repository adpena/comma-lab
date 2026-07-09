# Cells2Pixels (SIGGRAPH'26 NCA+LPPN) — deep-dive + bridge to the AMBER / 7-dim d_seg crux

**Date:** 2026-07-09 · **Author:** deep-research subagent (READ-ONLY; no trainer edits, no GPU).
**Operator directive:** "Deepest dive bridge all information."

## STORES CONSULTED
- Paper: arXiv 2506.22899 "Neural Cellular Automata: From Cells to Pixels" (Pajouheshgar, Xu, Abbasi,
  **Mordvintsev**, Jakob, Süsstrunk — EPFL + the Growing-NCA author). SIGGRAPH 2026. Project page
  cells2pixels.github.io. [abstract page + arxiv HTML v1 fetched; PDF too large for the fetcher]
- Repo: github.com/TheDevilWillBeBee/Cells2Pixels — `models/{nca2d,nca3d,meshnca,siren,optic_flow}.py`,
  `losses/{appearance,image,motion,rf,voxel,loss}.py`, `training/{common.py,tasks/growing_2d.py,...}`,
  `configs/nca2d/{growing,dynamic_texture,pbr_texture}.yaml`. Source files fetched raw.
- OUR priors: DAG `FEED-cells2pixels`, `FEED-amber-unblock`, `FEED-crux-7dim`, `FEED-ff` (Kuramoto/NCA
  design-verdict + reactivation trigger); MEMORY.md L17 (AMBER decisive), L11 (spine/crux).
- Contest law: S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489; pointer **0.19110 UNMOVED**;
  only `upstream/evaluate.py` on exact bytes = a real score. This memo is a MEANS — moves NOTHING.

Labels: **MEASURED** = read from repo source / config. **DERIVED** = computed by me from those.
**INFERRED** = reasoned from method, not stated. Numbers quoted verbatim where possible.

---

## 1. EXTRACTED ARCHITECTURE (MEASURED from source + `configs/nca2d/growing.yaml`)

### NCA core (`models/nca2d.py`)
- **cell-state channels `channels = 32`** (MEASURED, growing.yaml).
- **perception = 5 FIXED depthwise kernels**: `filters = torch.stack([ident, sobel_x, sobel_x.T, lap_x, lap_x.T])`
  (identity + 2 directional Sobel + 2 Laplacian). Depthwise `conv2d`, groups = b·ch. → perceived = 32·5 = 160.
- **update MLP = single hidden layer, two 1×1 convs**:
  `w1 = Conv2d(channels·5 + cond_chn → fc_dim=256, 1)` → `relu` → `w2 = Conv2d(256 → 32, 1, bias=False)`.
- **stability init (MEASURED)**: `xavier_normal_(w1.weight, gain=0.2)`; **`zeros_(w2.weight)`** = residual/identity
  init (NCA starts as a no-op, grows gradually — the canonical Growing-NCA trick).
- **stochastic fire-rate mask**: `M = (rand(...) + update_prob).floor()` multiplicative on the update.
- **alive masking** (growing variant): `new_s *= logical_and(pre_life_mask, post_life_mask)` (alpha maxpool gate).
- **DERIVED param count (cond_chn=0):** w1 = 160·256+256 = **41,216**; w2 = 256·32 = **8,192** → **≈ 49.4K params**.

### LPPN = the SIREN implicit decoder (`models/siren.py`, growing.yaml)
- The paper's "LPPN" is realized as a **SIREN** (sinusoidal MLP), applied **independently per cell** to
  `(s̄(p), u(p))` = locally-averaged cell state + **local continuous coordinate** (MEASURED: paper §; code
  `x = torch.cat([coords, cell_states], dim=-1); output = self.net(x)`).
- **config (growing.yaml): `hidden_features=64, hidden_layers=2, num_frequencies=1`**, output 4 (RGBA),
  `omega_0=30` sine activation. Positional enc: `cat([sin(aug), cos(aug)])`.
- **DERIVED param count:** input ≈ 32 + coord-enc(~4) = 36; 36·64+64 + 2·(64·64+64) + 64·4+4 ≈ **≈ 11K params**.
  → 11K/49K = **22%**, matching the paper's admitted "**roughly 20∼30% more parameters**" for the decoder.
- **TOTAL COUNTED ≈ 60K params** (49.4K NCA + 11K SIREN).

### Resolution (MEASURED)
- growing.yaml: `image_size [512,512]`, `scale_factor 8` → **coarse NCA lattice = 64×64**, SIREN decodes each
  coarse cell to an 8×8 pixel block → 512×512. Paper claims decode to **1024² and, WITHOUT retraining, 8192²**;
  "full-HD in real time." Because the SIREN input coordinate `u(p)` is a **continuous real** and the activation
  is sine (omega_0=30 → high-freq detail), the decode is **continuous / sub-pixel by construction** (MEASURED
  mechanism; the arbitrary-res claim is the paper's headline).

## 2. EXTRACTED TRAINING + STABILITY (MEASURED)
- optimizer input: `self._optimizer(list(model.parameters()) + list(siren.parameters()))` — **NCA + SIREN
  trained JOINTLY under ONE optimizer**. lr **0.001**, batch **8**, `epochs 4000 × num_repetitions 5 ≈ 20K iters`.
- **LR decay** `lr_decay_steps [1000,2000,3000] gamma 0.3`.
- **THE stabilizer — per-parameter gradient normalization** (`training/common.py`):
  ```python
  def normalize_model_grads(model):
      for p in model.parameters():
          if p.grad is not None:
              p.grad /= p.grad.norm() + 1e-8
  ```
  called after backward. This makes each step **magnitude-invariant per parameter** — the canonical
  Mordvintsev Growing-NCA trick; **stronger than clip** (it fully renormalizes, not just caps).
- **state POOL (MEASURED, `growing_2d.py`)**: `pool = model.seed(pool_size=1024, ...)`; per-iter
  `batch_idx = choice(len(pool), batch_size=8)`; **worst/first sample re-seeded** `x[:1] = model.seed(1,...)`
  every `inject_seed_interval=32`; **write-back** `pool[batch_idx] = x`. → trains a persistent attractor
  (recover-from-arbitrary-state), which suppresses grow-then-blow-up divergence.
- **random unroll** `step_n = randint(*step_range=[32,96])` per iteration.
- **overflow loss** `overflow_loss_weight = 100.0` (state-clamp penalty on |state| out of range) +
  `image_loss (l2=1, l1=1, lpips=1)`; textures use **OT / relaxed-EMD appearance loss** over VGG16 layers
  `(1,6,11,18,25)=relu1_1..relu5_1` in **YUV** space (color match + moment mu/cov + style + optional FFT
  autocorrelation), clamped `[1e-5,1e5]` (`losses/appearance_loss.py`).
- mixed precision with grad scaler.

**Claims/scope (MEASURED from abstract/HTML):** 2D grids, 3D voxel grids, 3D meshes; morphogenesis +
texture synthesis; one rule generalizes across resolution (retrain-free 8192²). **Limitations the authors
name** = the 3 NCA problems they FIX: (1) quadratic train/memory in grid size; (2) strictly-local propagation
→ weak long-range comm; (3) heavy high-res inference. Quantitative PSNR/SSIM/LPIPS numbers were **deferred**
("more details released soon") on the HTML — a real GAP; treat quality claims as unverified.

---

## 3. THE SIX BRIDGE VERDICTS (each with evidence + relative-significance; gap-to-target S−0.15 = 0.191−0.15 = 0.041)

**1 — res/scale (#149 vehicle): REAL but NECESSARY-NOT-SUFFICIENT.** The SIREN decodes from continuous
coords with sine activation → it CAN emit sub-pixel structure and decode the seg-frame at camera-res
874×1164 (MEASURED mechanism; retrain-free 8192² claim). **Adversarial correction (I refuted my own strong
claim):** R = bicubic↑(384→874)→uint8→bilinear↓(512×384) AVERAGES regardless of decode source-res — a
sharper flip at 874 does NOT bypass R's averaging. So the LPPN alone does NOT "recover the wall"; it supplies
the **sub-pixel placement DOF** that **training-THROUGH-R** exploits (pre-distort the band so the post-average
argmax flips right). The AMBER's `boundary_band_flip 0.079` (**half the ~0.16 rep-independent wall**) came from
training-through-R — OUR objective, not the paper's. Verdict: LPPN = the right *machinery* for the #149 lever,
win still owned by the through-R objective. **Relative-significance: HIGH as a mechanism** (targets ~half the
dominant d_seg residual term), **not a standalone frontier beater** (AMBER standalone S 0.415 > 0.19110).

**2 — direction/place/chroma/luma/time: PARTIAL map.**
- **place: STRONG** — SIREN local coord `u(p)` = literal sub-cell placement DOF (maps our margin/annulus/
  sub-pixel-t-localizer #275/#333).
- **direction: WEAK/present** — perception has 2 fixed Sobel (directional) + Laplacian, so the update *can*
  express gradient anisotropy, but it is **isotropic-by-default** (no learned orientation field). NOT the
  −48% all-class directional-curvelet basis (#277); would need OUR anisotropic basis bolted onto perception.
- **chroma+luma: present-as-appearance** — SIREN outputs 4 channels; appearance loss is in **YUV** (Y luma +
  UV chroma). So chroma/luma ARE expressible — but trained for TEXTURE match, **not SegNet-argmax**; chroma-as-
  d_seg-lever (#276) is OURS to add.
- **time: present-as-dynamic-texture** — `optic_flow.py` + `motion_loss.py` exist; maps loosely to our
  ξ/keyframe (#148/#193), NOT ego-rigid se(3). **scale: present** — coarse→fine is the NCA's native regime.

**3 — training stability (the BRIDGE to our collapse fix): their tricks MATCH+EXCEED our plan.**
Our planned fix = grad-clip + pose-eps-floor + stage-boundary weights. Theirs =
**(a) per-param grad NORMALIZATION** (stronger than clip; renormalizes w_seg=100 and the sqrt-pose-eps 5e4
blowup away per-parameter) + **(b) overflow loss (wt 100)** state-clamp (we LACK this) + **(c) state pool
1024 + seed re-injection** persistence (our per-pair batch=1 has NO pool → exactly what exposes the blowup) +
**(d) random unroll [32,96]** + **(e) zero-init last conv** residual identity. **Highest-value adoptions:
(a) grad-normalize + (b) overflow loss.** These directly de-risk the AMBER un-collapse. **Adversarial caveat
(refuting "it fixes collapse"):** grad-normalize CHANGES the optimization geometry — it kills the relative
seg-vs-pose grad scale our stage-boundary weights rely on, and could HURT the delicate boundary descent. It is
an **owed A/B, not a proven fix** — but it's a training-only lever (touches NO archive bytes → #205
byte-identity preserved), so it is SAFE to A/B behind a default-off flag against the existing immune system
(gnorm_hijack alarm, spike-guard).

**4 — rate / rule-118: FITS the band, but ONLY at ~4-bit.** DERIVED: ~60K counted params. At **FP4** (PR95
discipline) → **≈ 30 KB** → rate = 25·30000/37,545,489 = **≈ 0.0200** — right at the top of the proven AMBER
band (0.0072–0.019). At FP8 → 60 KB → rate 0.040 (OVER band). So the rule MUST quantize to ~4-bit to stay
cheap (MEASURE byte-closed with brotli, do not assume). rule-118: NCA rule + SIREN weights = **COUNTED**
(learned/video-derived); the iteration + continuous-coord decode = **FREE** in inflate.py. One rule is shared
across all frames of a video → 60K amortizes cheaply per-video.

**5 — originality / NO-FAKE #7: GENUINE COMPOSITION iff we do the accounting.** THEIRS (borrowed substrate) =
the coarse-NCA + SIREN-implicit-decoder architecture + the per-param-grad-norm/pool/overflow stabilizers.
OURS = (i) training the whole thing **THROUGH R against the SegNet argmax d_seg objective** (they do OT/VGG
texture + morphogenesis, NEVER a scorer); (ii) camera-res boundary-band-flip placement for d_seg (#149);
(iii) chroma-as-argmax-lever; (iv) the costate joint optimizer over the 7 dims; (v) byte-close into L13 +
rule-118 compile + FP4 quant. The **task-space SegNet-through-R objective is the originality line** — retrain
their pipeline on our frames with THEIR objective = borrowed FAKE; retarget the DECODER to the scorer manifold
= genuine. Requires an itemized `borrowed_substrate_accounting` on any submission.

**6 — HONEST VERDICT: top-AIML RE-OPEN vehicle for the AMBER — CONFIRMED, STAGED (per FEED-ff trigger).**
- **ADOPT specifically:** (i) SIREN continuous-coord decoder = the #149 camera-res sub-pixel placement
  machinery; (ii) per-param grad-normalize + overflow-loss + state-pool + random-unroll = the BRIDGE-#3
  collapse fix (exceeds our plan; adopt as default-OFF opt-in levers, A/B against #205 immune system).
- **DO NOT adopt:** their OT/VGG appearance objective (we need SegNet-argmax-through-R, not texture match);
  their isotropic Sobel perception as-is (bolt OUR anisotropic directional-curvelet basis for the −48% lever).
- **GAPS still owed (all heavy, operator-GO gated — CONTAINMENT, #205 owns GPU):** (a) the un-collapse A/B
  proving grad-normalize+overflow actually un-sticks the AMBER without harming the seg descent; (b) across-600
  **amortization** (one rule for all pairs vs per-video adaptive #211 — still UNMEASURED, n_converged=0 was
  collapse-confounded); (c) byte-closed FP4 quant keeping rate ≤0.019 MEASURED; (d) joint 7-dim descent
  convergence. **Reactivate WHEN (FEED-ff trigger): (a) #205 SDF witness walls on realized boundary_band_flip
  AND (b) the collapse fix is in.** No drop-everything. **Pointer 0.19110 UNMOVED — this is a MEANS.**

## verdict_scope + relative-significance
**verdict_scope: FORMULATION (positive re-open, no kill).** #143 flat-partition-NCA RED + #146 AMBER shelving
= implementation/formulation-level negatives (specific NCA formulations + a DIAGNOSED training-collapse BUG),
NOT a family/paradigm kill. cells2pixels is a POSITIVE re-open of the NCA d_seg-core family.
**Relative-significance:** the AMBER d_seg-core targets `boundary_band_flip 0.079` = HALF the ~0.16
rep-independent polynomial wall = the dominant term in (S_current − S_target); so re-open value is HIGH **as a
mechanism**, not as a standalone beater (AMBER standalone S 0.415 > pointer). Staging is MEASUREMENT-grounded
(cited exit criteria: collapse bug + unmeasured amortization), not a magnitude eyeball-dismissal.
