# Ledger 02 — NASA Goddard / Ames / Glenn spacecraft-autonomy lineage (2026-05-13)

**Lane:** `lane_expert_team_hardware_physics_future_alien_tech_20260513`.
**Persona:** NASA Goddard Space Flight Center (Greenbelt MD) — EOS Terra/Aqua MODIS, Landsat 8/9 OLI, MUSE-Mars Ultraviolet Spectrograph teams; NASA Ames Research Center (Moffett Field CA) — autonomy-for-Mars PASS-AI; NASA Glenn Research Center (Cleveland OH) — radiation-hardened ASIC design, LEON3 SPARC V8, RAD750 PowerPC. We compress to **survive the deep-space link** and to fit inside **single-chip radiation-hardened budgets**.
**Mode:** READ-ONLY engineering-analog derivation. `research_only=true`. NO archive bytes mutated.
**Evidence:** `[engineering-analog]`, `[mathematical-derivation]`, `[literature-prediction]`.

---

## 0. The NASA frame

Three operating regimes shape our compression practice:

1. **Deep-space link** (DSN downlink at ~10 kbps from Mars, ~3 kbps from Jupiter, ~16 bps from Voyager). Every bit costs ~$1-10 in DSN time. Compression-quality tradeoffs are bit-counted, not eyeballed.
2. **Radiation-hardened computers** (RAD750 @ 200 MHz, LEON3 @ 80 MHz, BAE RAD5545 quad-core PowerPC). No FPU on the rad-hard die. All onboard compression is **fixed-point integer arithmetic**.
3. **Mission-critical autonomy** (Mars Curiosity rover PASS-AI, Mars 2020 Perseverance terrain-relative navigation, OSIRIS-REx asteroid touch-and-go autonomy). The vehicle plans inside a **world prior** encoded in its onboard neural model; only **prior-model deltas** are sent home.

References operationalized below:
- **CCSDS 122.0-B-2** — Image Data Compression Recommended Standard, 2017.
- **CCSDS 123.0-B-2** — Low-Complexity Lossless and Near-Lossless Multispectral and Hyperspectral Image Compression, 2019.
- **MUSE imager**: NASA Goddard Mars-orbiter UV spectrograph, lossy-compression team (Brad Sandel / Bill McClintock).
- **PASS-AI / Mars Curiosity AEGIS**: autonomous target selection on Mars, Estlin et al. 2012, *AI Magazine* 33(2).

---

## 1. CCSDS 122.0-B-2 wavelet compression on the renderer's output

### 1.1 Background

CCSDS 122 (2017, NASA/ESA Bluebook) uses **9/7 biorthogonal wavelet** transform + bit-plane coding + post-processing entropy coder. Deployed on:

- Mars Reconnaissance Orbiter HiRISE (25 cm/pixel imaging at 0.5 bits/pixel).
- James Webb NIRCam (4 µm imaging at 0.5-2 bits/pixel).
- Sentinel-2 MSI (Sentinel Earth-observation, 10-60 m/pixel).

Performance: at 2 bits/pixel ≥ 40 dB PSNR on natural imagery; at 0.5 bits/pixel ≥ 30 dB.

### 1.2 Contest analog

The contest video is 1200 frames × 384×512×3 = 706 Mbits raw. CCSDS 122 at 0.5 bits/pixel → 88 Mbits = 11 MB. Way too big.

But: **per-pixel rate allocation is the knob**. Allocate:
- 5 bits/pixel to foveation region (~5% of frame) → 5 × 0.05 × 196608 / 8 = 6.1 KB/frame foveation
- 0.05 bits/pixel to periphery (~95% of frame) → 0.05 × 0.95 × 196608 / 8 = 1.2 KB/frame periphery
- **Total per frame ≈ 7.3 KB**

Across 1200 frames: 8.8 MB. Still way too big.

### 1.3 Per-pair-frame reuse (the real path)

The contest scorer consumes **pairs** of frames at a time (PoseNet) and **only the LAST frame of each pair** (SegNet). If we encode just the **600 reference frames** (every other frame) at the budget above, and the inflate runtime **synthesizes the intervening 600 frames via temporal interpolation** (warp from neighbors using stored optical-flow proxies):

- 600 frames × 7.3 KB = 4.4 MB stored
- 600 frames × ~0.5 KB optical-flow proxy = 0.3 MB
- **Total: ~4.7 MB**. Still too big for contest budget (~300 KB max archive).

### 1.4 The truly buildable variant

Run CCSDS 122 **only on residuals** from a tiny neural predictor. The neural predictor (~20 KB) handles the bulk; CCSDS 122 cleans up the residuals at very low rate. Predicted savings: marginal vs current PR101 architecture, but the **wavelet basis as a fixed transform** (Daubechies 9/7 is ~30 multiplies/pixel, ~5 LOC PyTorch) is a free knob to add to any pipeline.

### 1.5 Score-impact prediction

**Standalone replacement:** worse than PR101. Don't pursue.

**As a residual cleaner on top of an existing renderer:** -0.0002 to -0.0005 depending on residual statistics. [literature-prediction]

### 1.6 Reactivation

If a fresh renderer substrate emerges with high-residual-entropy regions, wavelet residual coding becomes a natural cleanup pass. Cross-link with `wavelet_telescopic_foveation_reactivation_20260509_codex.md`.

---

## 2. MUSE bit-allocator: rate-per-pixel by scorer-utility

### 2.1 Background

MUSE (Mars Ultraviolet Spectrograph, MAVEN orbiter) downlinks UV spectra from Mars at limited DSN bandwidth. The MUSE bit allocator computes **radiometric utility per spectral bin** and allocates bits proportional to scientific value. Quiet sky background gets 0.1 bits/bin; auroral emission lines get 4 bits/bin.

### 2.2 Contest analog

Compute **scorer utility per pixel** via gradient:
```python
def compute_scorer_utility_map(video, segnet, posenet):
    """Per-pixel |∂score/∂pixel_value|. Returns shape (H, W) average across frames."""
    utility = torch.zeros(H, W)
    for t in range(T):
        frame = video[t].requires_grad_(True)
        ...
        seg_loss = segnet_distortion(frame)
        pose_loss = posenet_distortion(frame, video[t+1])
        score_loss = seg_loss + sqrt(10) * pose_loss
        score_loss.backward()
        utility += frame.grad.abs().mean(dim=0)  # average across RGB channels
    return utility / T
```

**Empirical prediction** (from FastViT-T12 stride-2 + EfficientNet-B2 stride-2 stems):
- Top 5% pixels (class boundaries + ego-axis vertical strip): utility ~ 1.0
- Middle 15% pixels (textured road, near vehicles): utility ~ 0.1
- Bottom 80% pixels (sky, distant background, hood): utility ~ 0.01

A 100× rate ratio between top and bottom is justified.

### 2.3 Bit budget

If current archive spends 50% of bytes on pixel-level content (latents + residuals), redistributing those bytes per utility ratio could compact the top-utility region's representation by ~3× and trim periphery by ~10×. Savings ~30-50% of pixel-level bytes = 20-40 KB.

### 2.4 Score-impact prediction

20-40 KB rate savings = -0.00050 to -0.00100. Distortion impact: scorer is **less sensitive** to periphery (per definition); distortion change should be neutral or favorable. **Net: -0.0005 to -0.0010.** [mathematical-derivation]

### 2.5 Coordination

Cross-link with `wavelet_telescopic_foveation_reactivation_20260509_codex.md`. The MUSE allocator is a **static, scorer-derived prior**; the wavelet-foveation reactivation is a learned-attention variant. Both produce per-pixel utility maps. **Recommend MUSE first** because it's analytically derivable and reviewable in minutes.

---

## 3. PASS-AI / AEGIS: world-model-as-prior

### 3.1 Background

Mars Curiosity's AEGIS (Autonomous Exploration for Gathering Increased Science) lets the rover **autonomously select drill targets** by encoding a "good drill target" prior in its onboard neural net. Ground operators send only **deltas** ("here's what changed since you last asked"). The rover spends compute generating-and-testing hypotheses inside the prior.

### 3.2 Contest analog

This is the **most important** translation in this ledger. **The cooperative receiver (SegNet + PoseNet) IS our world model.** Their conv weights encode:
- ~5M parameters of SegNet driving-image priors (road/lane/vehicle/person classifications).
- ~12M parameters of PoseNet trajectory priors (vehicle kinematics + ego-motion).

**The 17M-parameter world model is sitting at inflate-time, free.** We are obliged to use it.

### 3.3 Three concrete implementations

**Implementation A — Penultimate-feature reuse:**
The scorer's penultimate features (just before the final classifier) encode **scene structure** that our renderer is laboriously re-learning. Train a tiny renderer that **CONDITIONS on the scorer's penultimate features** of the previous frame. The next frame is predictable from the previous frame's scorer features + a tiny per-pair residual.

```python
def render_frame(prev_frame, residual_bytes, segnet, posenet):
    prev_seg_features = segnet.encoder(prev_frame)        # cached
    prev_pose_features = posenet.vision(prev_frame_yuv6)  # cached
    residual = decode_residual(residual_bytes)            # ~30 bytes
    new_frame = mini_renderer(prev_seg_features, prev_pose_features, residual)
    return new_frame
```

Tiny renderer: ~10-20 KB. Per-pair residual: ~30 bytes × 1199 = 36 KB. **Total: ~50 KB archive** vs PR101's 229 KB.

**Implementation B — Hinton distillation of scorer into renderer:**
Train the renderer with a **distillation loss** against the scorer's intermediate features. The renderer absorbs the scorer's world-model into its own weights; per-pair latents shrink because the renderer "already knows" what to draw.

**Implementation C — Score-gradient-driven sample selection:**
At training time, sample input video patches **proportional to scorer gradient norm**. The renderer focuses parameter capacity where the scorer cares. (Sister to §2 MUSE allocator, applied to training-time data selection.)

### 3.4 Score-impact prediction

This is **the single largest potential gain in the ledger**. Predicted savings 100-150 KB → -0.0025 to -0.0040 score.

**Caveat:** building this requires the score-gradient saliency from CLAUDE.md Catalog #123, plus the differentiable-eval-roundtrip discipline, plus the score-aware-loss-from-byte-zero discipline (HNeRV parity lesson 1). All three already exist in the codebase. The remaining work is **substrate engineering**: 3-4 weeks of architecture iteration, multiple Modal A100 dispatches.

### 3.5 Reactivation

Register as `lane_substrate_world_model_prior_renderer` at L0 SKETCH. Reactivation requires:
- Operator approval (substrate engineering = multi-week + multi-Modal-dispatch spend)
- Prior `[contest-CUDA]` anchor on the score-gradient saliency primitive (Catalog #123 + #124 evidence)
- Council deliberation per CLAUDE.md "Design decisions" non-negotiable

---

## 4. LEON3 fixed-point integer-only inflate

### 4.1 Background

NASA Glenn's radiation-hardened RAD750 (PowerPC 7457, ~200 MHz, no FPU on rad-hard die) and ESA's LEON3 (SPARC V8, 80 MHz) run **all onboard compression in fixed-point Q16/Q32 integer arithmetic**. Reasons:
- No FPU on rad-hard CPU.
- Floating-point denormals trigger traps; integer arithmetic is exception-free.
- Byte-deterministic across processors.

### 4.2 Contest analog

Our current `inflate.py` uses PyTorch FP32. The **CUDA-vs-CPU drift** documented in `feedback_cuda_cpu_auth_eval_drift_pr102_pr104_20260508.md` (CUDA−CPU gap +0.033 on PR102) is partly due to **floating-point determinism differences** between CUDA cuBLAS / cuDNN kernels and CPU MKL / OpenBLAS kernels.

A **fully integer inflate** (Q16 fixed-point matmul, integer quantized activations, integer rendering) would:
1. **Close the CUDA-CPU drift** — same bytes in, same bytes out, regardless of which kernel ran.
2. **Be byte-deterministic** — bit-identical output across runs.
3. **Reduce dependency closure** — no need for FP-aware libraries.
4. **Open the CPU-leaderboard frontier** — currently PR102 is gold at `[contest-CPU]` 0.19538 while our PR107 is CUDA-anchored at 0.22936; if CPU/CUDA converge, internal CUDA-training reaches the CPU leaderboard.

### 4.3 Implementation challenge

PyTorch doesn't have a native integer-only inference path for the renderer architecture. Options:
- **TFLite Micro** (TensorFlow Lite for Microcontrollers): native integer-only, but doesn't support our exact architecture.
- **ONNX-Runtime with INT8 quantization** + manual fixed-point post-processing: feasible but ~2-week build.
- **Custom integer kernels via PyTorch tensor ops** with int32 accumulation: most flexible but requires hand-coded quantization-aware matmul.

### 4.4 Score-impact prediction

**No direct score gain.** Indirect: if it closes the CPU/CUDA gap, internal CUDA-training results predict CPU leaderboard ~1:1. This is **strategically larger** than any single bolt-on — it changes which optimization signal we trust.

### 4.5 Reactivation

Register as `lane_substrate_integer_only_inflate` at L0 SKETCH. Reactivation requires:
- Empirical CUDA-CPU gap measurement on a current candidate (post-fix replay)
- Comparison against the existing scorer-preprocess fix and differentiable-YUV6 patches
- Operator approval (multi-week build)

---

## 5. CCSDS 123.0-B-2 lossless multispectral residual coding

### 5.1 Background

CCSDS 123 is the **lossless** counterpart to CCSDS 122. It's used for multispectral / hyperspectral imagery (Landsat 9 OLI-2, Sentinel-3 OLCI, EnMAP, HISUI). Achieves typical compression ratio 2-4× on natural multispectral imagery, **lossless**. Algorithm: predictive (3D causal predictor) + arithmetic-coded residual.

### 5.2 Contest analog

If our renderer outputs are within ε of ground truth (small residual), code the residual losslessly with CCSDS 123. Residual stream is then **fully recoverable**, no distortion penalty.

**Practical challenge:** the residual at PR101's operating point is ~50-100 ε per pixel (8-bit-equivalent). Lossless coding of 1200 × 200K × 1 byte residuals = 240 MB. Way too big.

Lossless on residuals only works if the residuals are **already very sparse / very low-entropy**. They're not, at PR101's operating point.

### 5.3 Score-impact prediction

Not viable as a standalone path. **Discard for now, reactivation criteria**: if a substrate emerges with very-low-entropy residuals (≤ 0.1 bits/pixel), CCSDS 123 becomes natural cleanup.

---

## 6. Status / cross-references / next steps

- **Companion:** master memo `expert_team_hardware_physics_future_alien_tech_20260513.md`.
- **Sister memos in flight:** Bell Labs / Lincoln Lab / NSA / MIT-LIDS signal-processing; Skunkworks / CIA stealth-analytic.
- **Active codex work:**
  - `wavelet_telescopic_foveation_reactivation_20260509_codex.md` — overlaps §1 / §2.
  - `feedback_cuda_cpu_auth_eval_drift_pr102_pr104_20260508` — motivates §4.
- **Wire-in hooks** declared in master memo §9.
- **Reactivation:** all techniques `research_only`; none proceed to archive-bytes-change without explicit operator approval + grand-council review + Catalog #124 representation-archive-grammar-at-design-time discipline.

**Per CLAUDE.md "KILL is LAST RESORT":** all techniques DEFER-pending-research with reactivation criteria. CCSDS 123 lossless residual coding (§5) is DEFER-pending-low-entropy-residual-substrate.
