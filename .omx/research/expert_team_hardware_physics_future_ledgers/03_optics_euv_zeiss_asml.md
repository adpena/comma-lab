# Ledger 03 — Optics + EUV (Zeiss SMT + ASML) lineage (2026-05-13)

**Lane:** `lane_expert_team_hardware_physics_future_alien_tech_20260513`.
**Persona:** Zeiss SMT optics designers (Oberkochen, Germany — EUV mirror polishing to sub-50-pm rms), ASML EUV-lithography systems engineers (Veldhoven, Netherlands — NXE:3400C, NXE:3600D, EXE:5000 high-NA), Berkeley LBNL CXRO computational-imaging team. We work at 13.5 nm wavelength, NA 0.55, photon-count-limited dose, with multi-billion-dollar tools where every photon matters.
**Mode:** READ-ONLY first-principles physics derivation. `research_only=true`. NO archive bytes mutated.
**Evidence:** `[physics-bound]`, `[mathematical-derivation]`, `[literature-prediction]`.

---

## 0. The optics frame

Three immutable physics constants shape our practice:
1. **Abbe diffraction limit**: smallest resolvable feature = λ / (2 · NA). No amount of clever algorithm beats this.
2. **Photon shot noise**: Poisson statistics; below ~100 photons/pixel, noise dominates signal.
3. **Information capacity per area**: bounded by (1) and (2).

References operationalized:
- **Abbe 1873** (original diffraction limit derivation).
- **Goodman, *Introduction to Fourier Optics*, 4th ed.** (canonical text on coherent + incoherent imaging).
- **Ozcan group (UCLA), *Science* 2018** — diffractive deep neural networks (D²NN).
- **Bell Labs FlatCam / Rice FlatCam 2017** — lensless coded-aperture imaging.
- **CXRO Berkeley** — EUV computational imaging.

---

## 1. Abbe-limit-aware spatial-frequency pruning

### 1.1 The physics

Abbe diffraction limit:
```
d_min = λ / (2 · NA)
```
For the IMX390 camera in comma.ai panda hardware:
- λ ≈ 550 nm (visible center)
- NA ≈ 0.3 (F/1.7 lens at F-number ≈ 1.7 implies NA ≈ 1/(2·F) ≈ 0.29)
- d_min ≈ 0.9 µm

The IMX390 has 1.6 µm pixel pitch. **The optics resolve features ~1.8× finer than the pixel pitch — wait, actually coarser.** Let me redo: pixel pitch 1.6 µm; Nyquist at the sensor = 1/(2·1.6) = 0.3125 cycles/µm. Optical cutoff at 0.9 µm → 1/0.9 = 1.11 cycles/µm. Nyquist of the sensor is LOWER than optical cutoff → **the sensor is under-sampling the optical image** (aliasing!).

But the lens has an anti-aliasing low-pass filter (OLPF) or the optics intentionally diffract slightly to suppress aliasing. **Effective spatial cutoff after OLPF ≈ 0.6 × Nyquist ≈ 0.2 cycles/µm = 0.32 cycles/pixel.**

### 1.2 Translation to encoding

Any spatial frequency above 0.32 cycles/pixel in the camera output is **dominated by aliasing + sensor noise, not by scene structure**. The scorer's stride-2 conv stems further low-pass to ~0.16 cycles/pixel effective (Nyquist of the half-resolution feature map).

**Implication:** the renderer only needs to reconstruct content up to ~0.16-0.32 cycles/pixel. Anything higher is wasted bits.

### 1.3 Implementation sketch

```python
# In renderer's output layer, apply a Butterworth low-pass at cutoff 0.32 cycles/pixel
def abbe_lowpass(frame, cutoff=0.32):
    F = torch.fft.rfft2(frame, dim=(-2, -1))
    H, W = frame.shape[-2:]
    fy = torch.fft.fftfreq(H).abs().unsqueeze(1)
    fx = torch.fft.rfftfreq(W).abs().unsqueeze(0)
    mask = 1 / (1 + (torch.sqrt(fx**2 + fy**2) / cutoff)**8)
    F_filtered = F * mask
    return torch.fft.irfft2(F_filtered, s=(H, W))
```

### 1.4 Bit budget

Currently the renderer's per-pair latent encodes ~30 cycles of spatial content; pruning to 0.32 cycles/pixel × 196608 pixels / 4 (decimation) = ~50K relevant cycles. Latent compactness improves ~30-50%. Estimated savings 20-40 KB per archive.

### 1.5 Score-impact prediction

20-40 KB rate savings × 1/37.5 MB = -0.00050 to -0.00100. Distortion impact: **none** if cutoff is set correctly (we're discarding noise, not signal). **Net: -0.0005 to -0.0010** [physics-bound, mathematical-derivation].

### 1.6 Caveats

- Some sensor noise has structure (banding, fixed-pattern noise) that the scorer might use as artifact features. **Test against the actual scorer** before assuming high-frequency content is noise.
- The scorer's conv stem isn't a true low-pass filter; it has stride-2 + 3×3 kernel. Effective frequency response is **not** an ideal box filter. Empirical measurement of `|H(f)|` for the scorer stem is the right calibration step.

### 1.7 Reactivation

Register as `lane_abbe_limit_spatial_lowpass` at L0 SKETCH. Reactivation requires:
- Empirical scorer frequency-response measurement (~1 day work).
- Comparison against current renderer's spatial-frequency content.
- Loss-function ablation to confirm low-pass doesn't hurt training.

---

## 2. Photon-shot-noise entropy floor

### 2.1 The physics

A pixel measuring N photons has Poisson variance N. Entropy per pixel (Gaussian approximation, valid for N > 10):
```
H_pixel ≈ (1/2) · log2(2πe · N)
```

For typical 8-bit ADC output covering 0-255 grey levels with full-well capacity ~50K photons:

| Pixel value | Photons (approx) | Entropy (bits) |
|---|---|---|
| 1-3 (shadow) | 10 | 3.4 |
| 50 (mid-tone road) | 10³ | 5.4 |
| 200 (bright sky) | 4·10⁴ | 7.0 |
| 250 (sun-hit) | 5·10⁴ | 7.2 |

### 2.2 Translation

A region with N=10 photons/pixel has **at most 3.4 bits/pixel of meaningful information**. Coding it at 8 bits/pixel wastes 4.6 bits/pixel. For our 200K-pixel frame × 1200 frames, if 10% of pixels are in shadow at 10 photons/pixel, the waste is 200K × 0.1 × 4.6 / 8 × 1200 = 13.8 MB.

**That's enormous.** But it's already mostly captured by JPEG-style quantization and Brotli entropy coding — both adapt rate to local entropy. The novelty here is **explicitly bounding the per-region entropy from physical photon statistics**.

### 2.3 Implementation sketch

Precompute a per-pixel photon-count proxy from local pixel brightness:
```python
def shot_noise_entropy_map(video):
    """Returns per-pixel entropy floor in bits, from photon shot noise."""
    photons_per_pixel = video * 200  # rough calibration: pixel_value × 200 ≈ photons
    entropy = 0.5 * torch.log2(2 * torch.pi * torch.e * photons_per_pixel.clamp_min(1))
    return entropy
```

Use this as a **per-pixel rate budget cap**. The bit-allocator (sister to NASA §2 MUSE) never spends more than `entropy[h,w]` bits on pixel (h,w).

### 2.4 Score-impact prediction

5-15 KB savings on shadow / sky regions. -0.00015 to -0.00040 [physics-bound, mathematical-derivation].

### 2.5 Reactivation

Register as `lane_photon_shot_noise_entropy_floor` at L0 SKETCH. Cheap to test; ~3 LOC addition to existing bit-allocator. **Recommend testing this week** alongside MUSE bit-allocator.

---

## 3. Diffractive deep neural network → matched-filter conv stem

### 3.1 The physics

Ozcan group (UCLA, 2018, *Science* 361:1004): **D²NN** = stacked phase-mask layers between camera and detector. Each phase mask is a 3D thin-screen modulating optical phase. Light diffracts through the stack; the final detector intensity is a **learned function** of input scene structure.

**Each phase mask is a convolutional layer implemented in optics.** Trained via standard backprop.

### 3.2 Contest analog

The scorer's first conv layer is a learned matched filter implemented in silicon. If our renderer's output is **physically realizable as the input to that matched filter** — i.e. the renderer respects the same diffraction-limit + photon-shot-noise priors the real camera obeys — then the scorer's first layer extracts the renderer's signal at maximum SNR for minimum bit cost.

This is the **physics-grounded version** of sister Bell-Labs Technique B1 (matched-filter source coding). My contribution: **the receiver IS a camera + optics + sensor stack**; the encoder must respect that stack's physical priors to compress optimally.

### 3.3 Implementation sketch

Augment renderer training loss with **physically-realizable-image priors**:
- Penalize content above Abbe cutoff (§1).
- Penalize per-pixel rate above photon-shot-noise entropy (§2).
- Penalize physically-impossible color combinations (negative values, super-bright values).
- Penalize physically-impossible spatial gradients (sharper than Abbe limit allows).

```python
def physical_prior_loss(rendered_frame):
    abbe_violation = high_freq_content(rendered_frame, cutoff=0.32).norm()
    shot_noise_violation = excess_entropy_over_floor(rendered_frame).sum()
    color_violation = ((rendered_frame < 0) | (rendered_frame > 1)).float().sum()
    return abbe_violation + shot_noise_violation + color_violation
```

### 3.4 Score-impact prediction

Speculative. The prior should make renderer training **more sample-efficient** (smaller-parameter models reach equivalent score). Predicted savings 10-30 KB at no distortion cost. -0.00025 to -0.00075 [physics-bound, literature-prediction].

### 3.5 Coordination

Cross-link with sister Bell-Labs memo B1 (matched-filter source coding). Both predict same primitive (renderer output co-designed with scorer first-layer). My contribution: the **physics derivation** of the cutoffs (Abbe + photon shot noise) gives non-arbitrary cutoff values; Bell-Labs B1 derives the cutoffs from receiver-SNR-maximization. **Equivalent endpoints from different reasoning paths — strong evidence the primitive is real.**

---

## 4. Lensless / coded-aperture imaging analog

### 4.1 The physics

Bell Labs FlatCam (Asif et al., *IEEE TPAMI* 2017) and Rice University FlatCam: replace the lens with a **coded mask** placed close to the sensor. Each scene point illuminates a known pattern on the sensor; the sensor image is a **convolution** of the scene with the mask. Computational inversion recovers the scene.

Information-theoretically, lensless imaging **multiplexes** scene information across all pixels before measurement. A single dim scene point contributes to ~10⁴ sensor pixels — high redundancy, robust to per-pixel noise.

### 4.2 Contest analog

Instead of storing per-pixel latents (currently ~30 bytes/pixel × foveation area), store a **single coded multiplexer matrix M + a low-dimensional latent vector z** per frame.

```python
# Inflate-time
def lensless_decode(M, z_per_frame):
    """M: shape (K, P) where K is sensor pixels, P is latent dim.
       z_per_frame: shape (1200, P), per-frame latent.
       Returns: 1200 frames of shape (K,).
    """
    return torch.einsum("kp,tp->tk", M, z_per_frame)
```

If P = 50 (latent dim), per-frame bytes = 50 × 4 = 200 bytes. For 1200 frames: 240 KB. **Still too big.** Need P closer to 10-20 to fit.

Alternatively, share M across all frames (stored once, ~10 KB) and use small per-frame z (P=20, 80 bytes/frame, 96 KB total).

### 4.3 Bit budget

- Shared multiplexer M: 10 KB
- Per-frame z: 1200 × 80 bytes = 96 KB
- Per-frame fine residual: 1200 × 30 bytes = 36 KB
- **Total: ~142 KB**

This is **smaller than PR101's 229 KB** but larger than the time-traveler's 50 KB target (§7).

### 4.4 Score-impact prediction

Speculative. Lensless coded imaging exploits **scene sparsity** (most scene points contribute very little energy). Driving scenes have ~10²-10³ effective independent scene-points per frame; 20-50 latent dimensions should capture them.

Predicted: -0.005 to -0.010 if the inverse multiplexer is small enough (~5 KB) and the per-frame latent compresses well. [physics-bound, literature-prediction]

### 4.5 Reactivation

Register as `lane_lensless_coded_aperture_renderer` at L0 SKETCH. Substrate-engineering tier (multi-week). Reactivation requires:
- Operator approval (substrate work).
- Proof-of-concept reconstruction quality on a single contest frame (~1 day).
- Council deliberation per CLAUDE.md "Design decisions" non-negotiable.

---

## 5. Status / cross-references / next steps

- **Companion:** master memo `expert_team_hardware_physics_future_alien_tech_20260513.md`.
- **Cross-link to sister memos:**
  - Bell Labs B1 (matched-filter source coding) — §3 here is the physics-grounded version.
  - NASA Goddard §2 (MUSE bit-allocator) — sibling to §2 here (photon-shot-noise floor).
  - Lincoln Lab radar pulse compression — likely sibling concepts.
- **Active codex work:** `wavelet_telescopic_foveation_reactivation_20260509_codex.md` — overlaps with bit-allocation framing.
- **Wire-in hooks** declared in master memo §9.
- **Reactivation:** all techniques `research_only`. Photon-shot-noise floor (§2) is cheapest to test (~3 LOC); recommend running this week. Lensless coded imaging (§4) is substrate-tier; defer until operator approval.

**Per CLAUDE.md "KILL is LAST RESORT":** all techniques DEFER-pending-research with reactivation criteria.
