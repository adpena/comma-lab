# Deep HiNeRV/SNeRV/NeRV-family fidelity review vs the frozen `evaluate.py`

Date: 2026-06-09
Author: Claude MAX-REASONING research subagent (READ-ONLY; no code edits, no commit, no training launched)
Axis discipline: every numeric in this memo is tagged. The only score authority is exact
`upstream/evaluate.py` on `[contest-CPU]` (Linux x86_64) or `[contest-CUDA]` (T4). The d_seg=0.5075 /
d_pose=205.8 trace cited here is `[macOS-MLX research-signal]` (live MLX render scored through the
exact DistortionNet on macOS) — it is a mechanism diagnostic, NOT a score claim. PSNR is
`[advisory only]` and is never a score.

Status: research_only=true. No promotion, no rank/kill. This memo is a fidelity manifest +
root-cause analysis + ranked actionable findings. Per CLAUDE.md "Forbidden premature KILL": nothing
here kills the HiNeRV/SNeRV paradigm; it falsifies the *current default carrier configuration* at the
implementation level (Catalog #307 IMPLEMENTATION-LEVEL), and names the cheap experiments that test
the reactivation paths.

---

## 0. The decisive finding this memo explains (given, not re-derived)

Our HiNeRV carrier, even under the clean PR95-faithful curriculum (zero novelty), at ep~1000, scored
through the exact DistortionNet on a live MLX render:

- d_seg = 0.5075 `[macOS-MLX research-signal]` — **SegNet collapses to ~one dominant class** (a near-uniform
  render makes SegNet argmax ~constant, so ~50% of pixels disagree with the source's varied 5-class map).
- d_pose = 205.8 `[macOS-MLX research-signal]` — catastrophic; PR95 reached d_pose ~3.5e-5.
- Pure-recon PSNR plateaus at ~21.7 dB `[advisory only]` (blurry mean-field; good video INR is 35-40 dB).
- The export/inflate BRIDGE is EXONERATED (live ≈ archive; fp16 ≈ int8; roundtrip atol 5e-2).

PR95's similar-size 229K HNeRV decoder reached ~0.193 with d_seg ~6e-4, d_pose ~3.5e-5
`[contest-CUDA historical, via canonical frontier pointer]`. So the gap is **architecture +
training-schedule + objective-shape**, not the bridge, not bytes, not quantization.

This memo's thesis, stated up front: **our "HiNeRV" carrier is, in its default/PR95-faithful
configuration, a vanilla NeRV (sin + PixelShuffle, no skip, no positional grid) wearing a HiNeRV
label. It is missing BOTH of HiNeRV's two defining components (hierarchical grid positional encoding,
hierarchical skip connections) AND the PR95 reference decoder's bilinear-skip + refine-residual.
Those missing residual/high-frequency paths are the direct mechanical cause of the blurry mean-field
that collapses SegNet to one class.** A secondary objective-shape mismatch (recon-MSE + learnable
distillation-head surrogate vs PR95's direct frozen-SegNet margin loss with NO recon term) compounds it.

---

## 1. PR95-fidelity manifest (MATCH / APPROXIMATES / MISSING, per term)

The PR95 reference decoder source is
`experiments/results/public_pr_archive_kaggle_mirror/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/model.py`
(VERIFIED, read in full). Our carrier is `src/tac/substrates/hi_nerv/architecture.py` (PyTorch oracle)
and `src/tac/substrates/hi_nerv/mlx_renderer.py` (the MLX path that is actually trained + scored).

PR95 reference `HNeRVDecoder.forward` (VERIFIED, model.py:42-54):

```python
x = self.stem(z).view(B, C0, 6, 8)
x = torch.sin(x)
for block, skip in zip(self.blocks, self.skips):
    identity = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # bilinear-skip
    identity = skip(identity)                                                            # 1x1 channel match
    x = self.ps(block(x))                                                                # PixelShuffle(conv)
    x = torch.sin(x + identity)                                                          # sin(PS + skip)
x = x + 0.1 * torch.sin(self.refine(x))                                                  # dilated-conv HF refine residual
f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
```

| # | PR95-family term (source) | PR95 reference | Our HiNeRV carrier | Verdict | Evidence |
|---|---|---|---|---|---|
| L19 | Per-frame-PAIR latent | single 28-d latent → 2 frames | coarse16 + mid20 + fine24 = 60-d, injected at 3 depths | **APPROXIMATES** (richer, multi-scale; not wrong but more latent bytes/pair) | architecture.py:105-129, model.py:14 |
| L18a | Upsample operator | PixelShuffle(2) per stage | PixelShuffle(2) per stage (`_pixel_shuffle_2x_nhwc`) | **MATCH** | mlx_renderer.py:577, model.py:32,49 |
| L18b | **bilinear-SKIP per block** | `identity = bilinear(x); x = sin(PS(conv) + identity)` | **NONE** — `_UpBlockMLX` is `pixel_shuffle(sin(w*conv(x)))`; no skip, no residual | **MISSING (critical)** | mlx_renderer.py:561-577, architecture.py:321-329 vs model.py:46-50 |
| L18c | sin applied to **(PS + skip)** | `torch.sin(x + identity)` | sin applied to conv output **only**: `mx.sin(self.w*conv)` | **MISSING (consequence of L18b)** | mlx_renderer.py:577 vs model.py:50 |
| L18d | **refine HF residual** near heads | `x = x + 0.1*torch.sin(self.refine(x))` (dilated conv) | **NONE** | **MISSING** | architecture.py:508-518 vs model.py:35-38,51 |
| L18e | sin frequency | implicit `sin(x)` (w≈1) on stem; per-block sin on (PS+skip) | global SIREN `sin_frequency = 30.0` everywhere + SIREN-uniform init | **DIVERGES** (see §2.4 — w=30 with no skip is a spectral-bias trap) | architecture.py:121, mlx_renderer.py:1055 vs model.py:45,50 |
| — | channel taper | `[C,C,C,.75C,.58C,.5C,.5C]`, base C=36 → [36,36,36,27,20,18,18] | `(48,40,32,24,20,16,12)` monotonic taper from embed_dim 64 | **APPROXIMATES** | architecture.py:114-119 vs model.py:21 |
| — | initial grid | 6×8 → ×2^6 → 384×512 (6 stages) | 3×4 → ×2^7 → 384×512 (7 stages) then bilinear-resize to (384,512) | **APPROXIMATES** (7 vs 6 stages; final global resize) | architecture.py:116-123, mlx_renderer.py:7314 vs model.py:17,27 |
| — | separate rgb_0 / rgb_1 sigmoid heads | yes | yes (`head_rgb_0`, `head_rgb_1`, sigmoid×255) | **MATCH** | mlx_renderer.py:7319-7344, model.py:39-40,52-53 |
| L14 | 8-stage 29,650-epoch curriculum | CE→tau_softplus→smooth→QAT→C1a-L7→λ→σ→Muon | factory exists (`pr95_faithful_curriculum.py`) but the run that produced the trace was ep~1000 | **APPROXIMATES (under-trained)** | profile_pr95...md:32-39; trace candidate `hinerv_full600_long_native_muon_qat`, stage 5, ep~1000 |
| L15 | Muon final stage only | 177,156/228,958 params under Muon, stage 8 only | factory honors `faithful_stage8_only`; trace observed `pact_muon_adamw` partition | **MATCH (factory)** | pr95_faithful_curriculum.py:190-195; telemetry `optimizer_muon_observed=True` |
| L16 | C1a coder-aware reg | `cat_entropy_v2(sigma)` weight-distribution entropy | present in MLX coder_qat / loss | **MATCH** (not score-relevant to the d_seg=0.5 problem) | losses.py:79-113 |
| L17 | sigma noise schedule 0.2→0.1 | yes | present | **MATCH** | losses.py:11, profile...md:32-39 |
| — | **primary loss = SegNet margin, NO recon term** | `loss = 100*seg_margin + 1*pose + λ*c1a` (NO RGB MSE) | `loss = recon_MSE + distill(learnable head→cached SegNet) + pose + guards` | **DIVERGES (significant)** | losses.py:13-14, ce/tau_softplus/smooth/l7 vs `mlx_score_aware/loss.py:4` "reconstruction MSE + optional distilled surrogate" |
| — | seg target path | candidate RGB → **frozen SegNet** → margin on real logits | candidate RGB → decoded frame → **learnable student head** distilled toward cached teacher logits | **DIVERGES (significant — see §3)** | loss.py:11-16 |

**Two concrete MISSING components (not approximations): the bilinear-skip (L18b/c) and the
refine HF residual (L18d).** Plus the HiNeRV-defining hierarchical grid PE + ConvNeXt are present in
the code but **OFF by default** (`HinervConfig.use_hierarchical_feature_grid=False`,
`use_convnext_blocks=False`; architecture.py:136-140). So the default/PR95-faithful carrier is
strictly a vanilla NeRV.

### 1.1 HiNeRV-paper fidelity (the carrier is named "HiNeRV" but isn't)

VERIFIED from the HiNeRV paper (arXiv 2306.09818, NeurIPS 2023; ar5iv + NeurIPS proceedings):
HiNeRV's two defining ideas are (a) **"a new upsampling layer which embodies bilinear interpolation
with hierarchical encoding sampled from multi-resolution local feature grids"** — i.e. it *replaces*
sub-pixel/PixelShuffle with bilinear-up + grid PE; and (b) **"grid-based positional encodings,
hierarchical skip connections, patchwise processing, and bilinear upsampling with ConvNeXt blocks."**
The paper's own ablation: removing hierarchical encoding drops PSNR 34.69→32.16 dB.

Our default carrier uses PixelShuffle (the thing HiNeRV replaces), no grid PE, no skip, no ConvNeXt.
**It is a NeRV/HNeRV hybrid, not HiNeRV.** This is fine as a label-quibble — but the missing grid PE is
exactly the mechanism HiNeRV introduced to beat spectral bias, and it is OFF.

---

## 2. Root-cause analysis of the blurry mean-field (ranked hypotheses, with math)

The render is so low-frequency that SegNet argmax collapses to ~one class. Ranked by likely
contribution to the d_seg=0.5 / 21.7 dB plateau.

### H1 (TOP) — Missing residual high-frequency paths (bilinear-skip + refine), confidence HIGH

**Mechanism.** A NeRV upsampling stack `sin(PixelShuffle(conv(·)))` with no skip must synthesize ALL
spatial structure through the conv weights at each scale. PR95 instead computes
`x_{l+1} = sin(PixelShuffle(conv(x_l)) + bilinear_up(x_l))`. The `bilinear_up(x_l)` term carries the
previous scale's signal forward as a residual baseline, and the conv branch only has to learn the
*high-frequency correction* on top of it. This is a standard residual-learning argument: the optimizer
descends a much better-conditioned objective because the identity (coarse) path is free and gradients
reach early layers without passing through every sin nonlinearity. PR95 then adds an explicit terminal
HF residual `x + 0.1*sin(refine(x))` (dilated conv = larger receptive field for thin boundaries).

Without these, the only HF source is the conv kernels, and `sin` with high `w` (see H4) tends to drive
the network to a smooth low-curvature solution that minimizes recon-MSE cheaply (the DC/mean image is
the global MSE minimizer when HF is hard to fit). The result is the observed plateau: mean is right,
variance is a few percent of target — exactly the "one-class flat image" that
`initialize_output_head_contrast_from_targets` (mlx_renderer.py:1159) was bolted on to patch, and which
its own docstring (lines 1170-1177) explicitly describes: *"the scorer-resolution output starts with the
right mean but only a few percent of the target RGB variance, so SegNet sees a one-class flat image."*
That bolt-on is a band-aid on the symptom; the architectural cause is the missing residual path.

**Why it's the top hypothesis.** It is the single largest *verified structural* difference vs the
reference that won. PR95's residual-skip is the canonical fix for exactly this failure mode, and the
codebase already documents the one-class symptom. PSNR ~21.7 dB is consistent with a residual-free
NeRV that has converged its low-frequency content and stalled on HF.

### H2 — Hierarchical grid positional encoding is OFF, confidence HIGH (independent of H1)

**Mechanism.** NeRV/HNeRV/HiNeRV decoders take a per-frame latent and must paint a 384×512 image. With
no per-pixel coordinate signal, the decoder relies entirely on the spatial inductive bias of conv +
upsample to differentiate pixel locations. HiNeRV's hierarchical encoding injects a *coordinate-indexed*
multi-resolution feature grid at each scale (VERIFIED: "bilinear interpolation with hierarchical
encoding sampled from multi-resolution local feature grids"). This is the mechanism that gives the
decoder location-specific high-frequency content (the paper's 2.5 dB ablation). Our `HierarchicalFeatureGrid`
exists (architecture.py:208-266; `trilinear_upsample` modulo-indexed local grids) but is gated OFF by
default (`use_hierarchical_feature_grid=False`). So the carrier has no coordinate input at all — a pure
`latent → stem → conv-stack` map. That is precisely the configuration most prone to a smooth mean-field.

**Falsifiable distinction from H1.** H1 is about residual *connectivity*; H2 is about coordinate
*conditioning*. Either alone can cause low-frequency collapse; turning on grid PE supplies HF that the
conv stack cannot synthesize from a single latent. (NB: the local-grid path is flagged
`official_core_forward_parity_proven=False`, architecture.py:92 — its receiver semantics are still
research-only, so it needs a parse-back proof before it can be a promotion path. But for testing the
*spectral-bias hypothesis* locally, it's a legitimate ablation.)

### H3 — Objective shape: recon-MSE base + learnable-distillation-head surrogate vs PR95's
direct frozen-SegNet margin loss, confidence MEDIUM-HIGH (compounds, see §3)

PR95's loss has NO RGB-recon term; the *primary* signal is the SegNet argmax-margin surrogate computed
on the **frozen** SegNet logits of the candidate RGB, weighted 100× (matching `100*d_seg`). Our loss is
`recon_MSE + distill + pose + guards` where `distill` flows through a *learnable student head*. Two
problems: (a) an RGB-MSE base loss *rewards the mean-field* (MSE is minimized by the conditional mean →
blur), pulling against sharp boundaries; (b) a learnable student head can satisfy the distillation
surrogate while the *frozen* SegNet (the contest's actual scorer) still sees one class — the head
absorbs the discrepancy. PR95 never has this gap because it differentiates straight through the frozen
SegNet. See §3 for the geometry.

### H4 — SIREN frequency w=30 *without a skip* is a spectral-bias trap, confidence MEDIUM

**Mechanism.** SIREN uses `sin(w·x)` with large `w` (≈30) specifically to *inject* high frequencies —
but SIREN is an MLP on *coordinates*, where the high-frequency input lets the network represent sharp
signals. In our carrier, `sin(30·conv(x))` is applied to *feature maps with no coordinate input and no
skip*. High `w` on a slowly-varying feature map mostly aliases/saturates: `sin` wraps through many
periods for small input changes, producing a near-random high-frequency carrier that gradient descent
cannot organize into coherent image structure, so it collapses the useful signal to the low-curvature
(near-DC) regime where `sin(w·x) ≈ w·x` is locally linear. PR95 sidesteps this by (i) applying `sin` to
`(PS + bilinear_skip)` so there's always a coherent low-frequency carrier inside the sin, and (ii) using
an implicit `w≈1` on the stem. **w=30 is HARD-EARNED for coordinate-MLP SIREN but CARGO-CULTED for a
skip-free feature-map NeRV decoder.** This is a per-assumption fork candidate (CLAUDE.md 18-assumption
discipline).

### H5 — Under-training (ep~1000 vs 29,650), confidence LOW-MEDIUM as the *primary* cause

The trace candidate was at PR95 stage 5, ep~1000. PR95's curriculum is 29,650 epochs and HF converges
last (spectral-bias: low frequencies first). So some of the blur is just incomplete training. BUT: d_seg=0.5
with SegNet at *one class* is not "HF not yet sharp" — it's "the image has almost no variance," which a
residual-free NeRV reaches early and then *stalls* on (the H1/H4 plateau). More epochs on the *same
architecture* will sharpen slowly but the residual-free + no-PE + w=30 configuration has a low ceiling
(21.7 dB is already a plateau, per the prompt). Under-training is real but secondary; it is not why a
229K decoder that PR95 drove to d_seg~6e-4 is stuck at 0.5. **Falsify cheaply by checking the recon-PSNR
slope at ep1000: if it is ~flat (plateau), H5 is not primary; if still rising fast, give it more epochs
before changing architecture.**

**Ranked verdict:** H1 (missing residual paths) ≈ H2 (PE off) > H3 (objective shape) ≈ H4 (w=30) > H5
(under-training). H1 and H2 are the two structural causes; H3/H4 compound; H5 is a confound to control.

---

## 3. Evaluator-geometry alignment analysis

The contest metric (VERIFIED, `contest_eval_contract.py`:71-72, `modules.py` snippets):
`d_seg = mean(argmax(out1) != argmax(out2))` on the **last frame only**; PoseNet uses **both** frames via
YUV6. The d_seg surface is a **0/1 argmax-disagreement rate** — flat (zero gradient) everywhere except at
SegNet decision boundaries (where the top-2 logit margin crosses 0). The only way to move d_seg is to
push boundary pixels across the argmax line.

**PR95's loss is exactly this geometry** (VERIFIED losses.py):
- `smooth_disagreement_seg_loss` = `sigmoid(-margin/tau).mean()`, `margin = target_logit - max_other_logit`.
  Its gradient `d/dmargin sigmoid(-margin/tau)` is a **bell curve peaking at margin=0** — i.e. it puts ALL
  the optimization pressure exactly on pixels sitting on the decision boundary, which is where d_seg lives.
- `l7_softplus_seg_loss` then *reweights* hard pixels (margin < threshold) by 4×.
- Weight is `100*seg`, matching the contest's `100*d_seg` coefficient.
- Crucially: the candidate RGB goes straight into the **frozen** SegNet; the loss is on the *real* logits
  the contest will see. There is no recon-MSE term to pull toward the mean.

**Our loss optimizes a surrogate that admits the mean-field minimum** (VERIFIED loss.py:4-16):
1. The **base term is reconstruction MSE.** MSE's minimizer is the conditional mean → it *actively
   rewards* the blurry mean-field. PR95 has no recon term at all.
2. The seg term is a **learnable student head** distilled (KL) toward a cached SegNet teacher
   (loss.py:11-16). The student head is a free function approximator inserted between the decoded frame
   and the loss. It can drive the distillation loss down while the *frozen* SegNet — which has NO student
   head at eval — still argmaxes to one class on the actual blurry render. **This is a surrogate-vs-
   authority gap of exactly the kind CLAUDE.md "NO FAKE / Tier-A" warns about, but at the gradient level:
   the gradient optimizes a quantity (student KL) that is not the authority (frozen-SegNet argmax).**

**Geometric conclusion.** Even with a sophisticated worst-connected-region p50-margin objective present
in the MLX adapter (adapter.py:633-786 — this part IS argmax-aware and good), the *base recon-MSE term +
learnable-head indirection* means the composite objective has a low-frequency mean-field local minimum
that PR95's objective does not have. The architecture (H1/H2) makes the mean-field *easy to reach*; the
objective (H3) makes it *not penalized enough*. Both must be fixed; fixing only one is likely insufficient.

A subtle corollary on the contest's `x[:,-1,...]` SegNet asymmetry (contract:226-233): d_seg depends
**only on frame_1**. Our recon-MSE weights frame_0 and frame_1 equally, spending half the capacity on a
frame SegNet never sees (it does matter for pose — see §5 — but not for seg). PR95's separate rgb_0/rgb_1
heads + seg-on-frame_1 already handle this; our equal-weight recon does not exploit the asymmetry.

---

## 4. SNeRV assessment (arXiv 2501.01681; `src/tac/substrates/snerv_inverse_steg_carrier`)

**What SNeRV is (VERIFIED abstract + our `__init__.py`):** SNeRV uses a 2D DWT to split each frame into
LF (coarse approximation) and HF (detail). It **encodes only the LF into network parameters and
GENERATES the HF with a decoder** (MFU = multi-resolution fusion, HFR = high-frequency restorer, TUB =
temporal up-sampling block). Its explicit thesis is that "neural networks learn HF components at a slower
rate than LF components" — i.e. SNeRV is a *direct architectural answer to the exact spectral-bias
failure we are seeing in H1/H4.* Our substrate also has `dwt.py` (orthonormal DWT + exact synthesis
adjoint, rel-residual 0.0) and a "store-LF / generate-HF" carrier (the cure for the "Z8 disease" of
storing the raw HF detail blob).

**Is SNeRV a viable carrier per the evaluator geometry?** Assessment (INFERRED from mechanism +
evaluator geometry; not yet empirically scored here):

- **For d_seg (argmax boundaries):** Boundaries ARE high-frequency. SNeRV's HFR explicitly restores fine
  texture/edges — structurally the right tool for boundary-class fidelity. If the HFR genuinely recovers
  edges that the LF-only path blurs, SNeRV should reach lower d_seg than a residual-free NeRV at equal
  bytes. **But** SNeRV's HFR is *generated* (predicted from LF), not *stored* — so its boundary fidelity
  is only as good as the LF→HF predictor generalizes on THIS video. For single-video memorization (our
  regime), a stored/residual approach (PR95's bilinear-skip) may actually be more reliable than a learned
  HF generator, because we can overfit the one video. So SNeRV's advantage is *parameter efficiency* (HF
  is free), not necessarily *lower achievable d_seg* than a well-trained PR95-skip decoder.
- **For d_pose (both-frame YUV6):** SNeRV's TUB captures temporal correlation between frames — directly
  relevant to the two-frame pose signal (see §5). This is a genuine SNeRV advantage over a per-frame NeRV.
- **For bytes:** "store LF, generate HF" is the strongest byte story of the three (LF is a small fraction
  of the DWT coefficients). Aligns with the contest's `25*bytes/N` term.

**SNeRV blocker (from our substrate + AGENTS.md):** the hard blocker is "official MFU/HFR/TUB
source-forward train/export/runtime binding plus LF/HF representation collapse under real byte pressure."
I.e. SNeRV is currently a research carrier without a proven byte-closed inflate path, and there's a known
risk that under aggressive byte pressure the LF/HF split collapses (the HF generator outputs ~0 and you're
back to LF-only blur — the SAME mean-field, just reached differently). **Verdict: SNeRV is the most
principled architectural answer to the spectral-bias root cause, but it is NOT a drop-in fix — it shares
the same "HF can collapse" failure risk and adds an unproven export/runtime contract. It is a strong
parallel campaign, not the cheapest next step.** The cheapest next step is to fix the carrier we already
score (HiNeRV-with-skip), which is one residual line of code.

---

## 5. The d_pose ~ 200 explosion analysis

PoseNet (VERIFIED contract:157-170, modules.py): two RGB frames → bilinear resize → `rgb_to_yuv6`
(mean=127.5, std=63.75) → 12-channel tensor → FastViT-T12 → MSE on first 6 of 12 pose dims. PoseNet reads
**both** frames and is fundamentally a *temporal/motion* estimator: it infers ego-motion from the
*difference/structure between frame_0 and frame_1*.

**Why a mean-field render gives d_pose ~ 200:**

1. **No inter-frame motion signal.** A blurry mean-field render produces frame_0 ≈ frame_1 ≈ (smooth
   blob). PoseNet sees ~zero apparent motion / ~degenerate optical structure, so its 6-dim pose output is
   far from the source's true pose (which encodes real camera motion). The MSE between a near-constant
   predicted pose and the true varied pose is large and roughly constant across pairs — consistent with
   the observed `pose_tail_burst_baseline_median ~ 3.6` per-window and aggregate ~200 over the run
   `[macOS-MLX research-signal]`, telemetry from `hinerv_full600...`.
2. **YUV6 amplifies the variance deficit.** YUV6 normalizes with std=63.75. A render with a few percent of
   target RGB variance has *even less* YUV6 chroma/luma structure after this normalization, so the
   PoseNet input is nearly featureless — the worst case for a motion estimator.
3. **The recon-MSE objective doesn't supply motion structure.** Per-frame MSE toward two blurry targets
   does not reward getting the *frame_0 → frame_1 delta* right; PR95's `pose_loss = sqrt(10*MSE)` is
   computed through the *frozen PoseNet on both frames*, so it directly rewards reproducing the pose-
   relevant temporal structure.

**What restores the two-frame pose signal:**

- (a) **Distinct, sharp frame_0 and frame_1** with correct *relative* structure — which the bilinear-skip
  + refine residual (H1) and grid PE (H2) provide by raising HF fidelity on BOTH heads. d_pose and d_seg
  share the same root cause: a sharp render fixes both.
- (b) **A pose loss through the frozen PoseNet on both frames** (PR95 does this; our pose_distill exists
  but, like seg, may route through indirection — worth auditing that the pose term differentiates through
  the frozen PoseNet, not a learnable pose head).
- (c) **SNeRV's TUB** (temporal up-sampling) is the architectural version of (a) — it explicitly models
  the frame_0↔frame_1 temporal correlation.
- (d) The codebase already has `posenet_temporal_signal_floor_weight` and
  `posenet_yuv6_geometry_tether_weight` guards (trainer config lines 199-201) — these are the right idea
  (force minimum temporal/YUV6 structure) but they are *floor penalties*, not the primary objective;
  they patch the symptom. The primary fix is the sharp render.

**Note:** at PR95's operating point (pose_avg ~3.4e-5) the pose marginal is 2.71× SegNet's (CLAUDE.md
"SegNet vs PoseNet importance"). But at OUR operating point (d_pose=205.8, far above the 2.5e-4 crossover)
we are nowhere near that regime — both terms are catastrophic and the *seg* term (100*0.5075 = 50.75)
dominates the *pose* term (sqrt(10*205.8) = 45.4). So **fix d_seg first** (it's the larger contributor AND
the easier-to-attribute argmax collapse), and d_pose will co-improve because both are downstream of the
same mean-field.

---

## 6. RANKED ACTIONABLE FINDINGS

Each finding: the change, the predicted mechanism, and how to falsify it cheaply (all local MLX,
`[macOS-MLX research-signal]`, $0 — promote only via byte-closed archive + paired CPU/CUDA per CLAUDE.md).
These are research proposals; per "Forbidden premature KILL" nothing here retires the paradigm.

### F1 (HIGHEST VALUE, CHEAPEST) — Add the PR95 bilinear-skip + refine residual to `_UpBlockMLX`

- **Change:** make each upsample block `sin(PixelShuffle(conv(x)) + skip(bilinear_up(x)))` (1×1 channel-
  match skip when in≠out), and add the terminal `x + 0.1*sin(refine(x))` dilated-conv residual before the
  RGB heads. This is the exact PR95 reference (model.py:46-51) and the PyTorch oracle
  (architecture.py:321-329) must match too. ~10-20 lines.
- **Predicted mechanism (H1):** the identity/coarse path becomes free; the conv branch learns only HF
  corrections; the optimizer escapes the mean-field; recon PSNR breaks the 21.7 dB plateau (expect ≥30 dB
  on a single video), SegNet stops collapsing to one class, d_seg drops by an order of magnitude or more,
  d_pose co-improves.
- **Falsify cheaply:** retrain the SAME short curriculum (ep~1000, same losses, same latents) with ONLY
  the skip+refine added. Score the live MLX render through DistortionNet. **Prediction: d_seg drops from
  ~0.5 toward <0.1 and PSNR > 28 dB within the same epoch budget.** If d_seg stays ~0.5 with the skip
  added, H1 is falsified and H2/H3 dominate — escalate to F2. This is a clean single-variable ablation
  and the single most important experiment in this memo.

### F2 — Turn ON the hierarchical grid positional encoding (`use_hierarchical_feature_grid=True`)

- **Change:** flip the config flag (architecture.py:136); the `HierarchicalFeatureGrid` machinery already
  exists. Optionally also `use_convnext_blocks=True`. (For *local testing of the spectral-bias hypothesis*
  this is valid; for *promotion* the local-grid path needs the parse-back/receiver proof per
  architecture.py:92 `official_core_forward_parity_proven=False` — flag this as a gating follow-up.)
- **Predicted mechanism (H2):** coordinate-indexed multi-resolution grids inject location-specific HF the
  conv stack cannot synthesize from a single latent (HiNeRV's own ablation = +2.5 dB PSNR). This is the
  literal HiNeRV mechanism; turning it on makes the carrier actually HiNeRV.
- **Falsify cheaply:** ablation grid {skip off/on} × {PE off/on}, 4 short runs, score each through
  DistortionNet. If PE-on with skip-off still collapses, the residual connectivity (F1) was the binding
  constraint; if PE-on alone fixes it, H2 was binding. Run F1 and F2 as a 2×2 — it isolates both top
  hypotheses in 4 cheap runs.

### F3 — Drop (or heavily downweight) the recon-MSE base term; make the frozen-SegNet margin loss primary

- **Change:** move toward PR95's `loss = 100*seg_margin(frozen SegNet) + pose(frozen PoseNet) + λ*c1a`
  with NO recon-MSE base (or recon weight → a small anneal-to-zero anchor only). Critically, ensure the
  seg margin is computed on the **frozen** SegNet logits of the candidate RGB (the `smooth_disagreement` /
  `l7_softplus` bell-curve-at-margin-0 surrogate, losses.py:39-62), NOT through a learnable student head.
- **Predicted mechanism (H3):** removes the mean-field reward (MSE) and the surrogate-vs-authority gap
  (learnable head); puts all gradient pressure on boundary pixels where d_seg actually lives.
- **Falsify cheaply:** A/B the seg loss path {learnable-head distillation vs direct frozen-SegNet margin}
  at fixed architecture+epochs; compare d_seg through DistortionNet. **Prediction: direct frozen-SegNet
  margin reaches lower d_seg.** Caveat: this requires the differentiable-SegNet path
  (`tac.differentiable_eval_roundtrip`) to be gradient-reachable through the frozen SegNet — verify that
  first (it's a CLAUDE.md non-negotiable and the seg scorer must not be `@torch.no_grad()`-severed).

### F4 — Lower the SIREN frequency for the feature-map sin (w=30 → ~1-6), or restrict w=30 to a coordinate-PE input only

- **Change:** set `HinervConfig.sin_frequency` from 30 to ~1 (PR95-implicit) or sweep {1, 6, 30}; if grid
  PE is on (F2), keep high w only on the *coordinate/PE* branch where it belongs (SIREN's actual design).
- **Predicted mechanism (H4):** removes the high-w aliasing trap on coordinate-free feature maps that
  pushes the network into the near-linear (DC) regime.
- **Falsify cheaply:** w-sweep {1,6,30} at fixed (skip-on) architecture; recon PSNR + d_seg. **Prediction:
  w≈1 with skip beats w=30 with skip on this single-video memorization task.** Cheap (3 runs).

### F5 — Control for under-training before any architecture verdict (H5)

- **Change:** none — diagnostic only. Plot recon-PSNR vs epoch for the EXISTING run around ep1000.
- **Predicted mechanism (H5):** if the slope is ~flat, the carrier has plateaued (architecture-limited →
  F1/F2 needed); if still climbing steeply, give it more epochs first.
- **Falsify cheaply:** read the existing telemetry JSONL slope. ~zero cost; do this BEFORE F1 to set the
  baseline and confirm the plateau is architectural, not budgetary.

### F6 (parallel campaign, not the cheapest) — Pursue SNeRV (store-LF/generate-HF + TUB) as the
principled spectral-bias-native carrier

- **Change:** advance `snerv_inverse_steg_carrier` toward a byte-closed archive + parse-back proof; the
  DWT (exact adjoint), MFU/HFR/TUB, and store-LF/generate-HF carrier already exist.
- **Predicted mechanism (§4):** HFR is the architectural answer to spectral bias; TUB restores the two-
  frame pose signal; store-LF/generate-HF is the best byte story. But it shares the "HF generator can
  collapse" risk and needs the export/runtime binding (the AGENTS.md SNeRV hard blocker).
- **Falsify cheaply:** first run a *non-byte-closed* SNeRV recon smoke on the contest video and score the
  live MLX render through DistortionNet — does the HFR actually recover edges (PSNR > 30, d_seg < 0.1) or
  does it collapse to LF (PSNR ~21, d_seg ~0.5)? If it collapses, SNeRV inherits the same disease and F1/F3
  are still the priority. Run AFTER F1's result is known.

---

## 7. Summary verdict

- The bridge is exonerated; the gap is **architecture + objective-shape**, not bytes/quantization/export.
- The PR95-fidelity manifest found **concrete MISSING components**, not mere approximations: the
  **bilinear-skip per block (L18b/c)** and the **terminal refine HF residual (L18d)** are absent from
  `_UpBlockMLX`; and the two HiNeRV-defining features (**grid positional encoding**, **ConvNeXt**) are
  present but **OFF by default**. The default/PR95-faithful carrier is a vanilla skip-free NeRV mislabeled
  as HiNeRV.
- These missing residual/HF paths are the mechanical cause of the blurry mean-field that collapses SegNet
  to one class (d_seg=0.5) and zeroes the two-frame pose signal (d_pose=205.8). A secondary objective-shape
  mismatch (recon-MSE base + learnable-head SegNet surrogate vs PR95's recon-free direct frozen-SegNet
  margin) rewards and hides the mean-field.
- Highest-value cheapest experiment: **F1 — add the bilinear-skip + refine residual and retrain the same
  short curriculum.** It is one block-definition change and a clean single-variable test of the dominant
  hypothesis.

### Evidence tags

VERIFIED (file:line): PR95 decoder (model.py:13-54), PR95 losses (losses.py:25-162), PR95 curriculum
(profile_pr95_hnerv_muon_intake.md:32-39), our MLX block (mlx_renderer.py:545-577), our PyTorch block
(architecture.py:321-329, 472-518), config flags OFF (architecture.py:136-140), one-class symptom
documented in our own bolt-on (mlx_renderer.py:1170-1177), MLX loss = recon-MSE + distill (loss.py:1-16),
evaluator geometry (contest_eval_contract.py:71-72, 143-170, 226-233), crossover 2.5e-4
(score_geometry.py:13-16), SNeRV mechanism (snerv .../__init__.py + dwt.py).

VERIFIED (web): HiNeRV bilinear-up + grid PE + hierarchical skip + ConvNeXt + 34.69→32.16 dB ablation
(arXiv 2306.09818, ar5iv/NeurIPS 2023); SNeRV DWT store-LF/generate-HF + MFU/HFR/TUB + spectral-bias
thesis (arXiv 2501.01681 abstract). Full PDFs were NOT fetchable (>10 MB / abstract-only landing pages);
architecture specifics above are from the ar5iv summary + abstracts + my own knowledge, tagged
accordingly.

INFERRED: the surrogate-vs-authority gradient gap of the learnable distillation head (mechanism argument,
not measured); SNeRV's relative d_seg advantage vs a well-trained PR95-skip decoder (mechanism, not
measured); d_pose mechanism attribution (consistent with telemetry but not isolated by ablation).

No fabricated numbers. The only quantitative score-axis claims (d_seg=0.5075, d_pose=205.8, PSNR 21.7,
PR95 0.193 / d_seg 6e-4 / d_pose 3.5e-5) are carried from the prompt/canonical pointers and tagged by axis.
