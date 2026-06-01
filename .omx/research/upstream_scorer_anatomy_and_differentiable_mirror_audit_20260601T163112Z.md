# Upstream scorer anatomy + tac differentiable-mirror audit

- **Lane**: `lane_boundary_aware_rd_allocation_grammar_20260601`
- **UTC**: 2026-06-01T16:31:12Z
- **Author**: subagent `boundary_aware_rd_anatomy_20260601`
- **Cost**: $0 — macOS-CPU local. Every numeric result below is `[macOS-CPU advisory]`, **NON-PROMOTABLE** (no score claim, no MPS authority, no contest-CUDA/contest-CPU claim).
- **Mission contribution** (Catalog #300): `frontier_breaking_enabler` — this is the foundation anatomy enabling exact boundary-aware (`s_seg`) + pose-Fisher (`s_pose`) RD bit allocation per the operator's joint P18/P19 water-fill direction.
- **Real assets used**: `upstream/models/segnet.safetensors` (36.7M), `upstream/models/posenet.safetensors` (53.2M), `upstream/videos/0.mkv` (35.8M = the entire public test set; Catalog #213). NO synthetic fixtures for the real-weight sections.
- **Harness**: `tools/verify_upstream_scorer_mirror_fidelity.py` (NO-FAKE — loads real weights, decodes real frames, measures divergence; never asserts constants). **Tests**: `src/tac/tests/test_upstream_scorer_mirror_fidelity.py` (5 tests, all pass; the 4 real-weight tests SKIP if assets absent — they are not faked).
- **Raw JSON**: `.omx/tmp/scorer_mirror_fidelity_20260601.json`.

---

## 1. Step-by-step pipeline (end-to-end, every numerical step that affects d_seg/d_pose/rate)

Source files (READ-ONLY pinned snapshot, never edited): `upstream/frame_utils.py`, `upstream/evaluate.py`, `upstream/modules.py`.

### 1.1 Decode (CPU path = `AVVideoDataset` / `TensorVideoDataset`)
- **Camera-native resolution**: `camera_size = (1164, 874)` = (W, H). Frames are `(H=874, W=1164, 3)` uint8.
- **`seq_len = 2`**: every "pair" is 2 **consecutive non-overlapping** frames. The dataset batches into `(B, 2, H, W, 3)`.
- **CPU decode** (`frame_utils.yuv420_to_rgb`, lines 159-183): yuv420 → RGB via **bilinear chroma upsampling + BT.601 LIMITED range** (Y offset 16, scale 255/219; chroma offset 128, scale 255/224; matrix R=Y+1.402·V, G=Y−0.344136·U−0.714136·V, B=Y+1.772·U), `.clamp(0,255).round().to(uint8)`. This is the *decode-from-mp4* transform, distinct from the *scorer-input* `rgb_to_yuv6`.
- **The submission contract**: `evaluate.py` reads ground-truth frames from `--uncompressed-dir` (`./videos/`) and candidate frames from `--submission-dir/inflated/` (raw uint8 `.raw` files, `(N, 874, 1164, 3)`, via `TensorVideoDataset`). `inflate.sh` (the submission runtime) must materialize `inflated/<name>.raw` at exactly `(N, 874, 1164, 3)` uint8 (asserted at `evaluate.py:77`). The `archive.zip` is the **scored byte payload**.

### 1.2 DistortionNet preprocess (`modules.py:143-148`)
- Input `(B, 2, H, W, 3)` uint8 → `einops.rearrange('b t h w c -> b t c h w').float()` → `(B, 2, 3, 874, 1164)` float in `[0,255]`. **This CHW float tensor is what both scorers' `preprocess_input` receive.**

### 1.3 SegNet path (`modules.py:103-113`)
- **Architecture**: `smp.Unet('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)` (EfficientNet-B2 encoder, 5-class decoder, raw logits, no activation).
- **`preprocess_input`**: `x = x[:, -1, ...]` → **ONLY the LAST frame** of each pair → `F.interpolate(size=(384, 512), mode='bilinear')` → `(B, 3, 384, 512)`. (Note: `segnet_model_input_size = (512, 384) = (W, H)`, so the interpolate target is `(H=384, W=512)`.)
- **Forward** → logits `(B, 5, 384, 512)`.
- **`compute_distortion`** (`modules.py:111-113`): `diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float()`; `d_seg = diff.mean(per-pixel)`. ✅ **Confirmed: d_seg = mean per-pixel argmax-FLIP RATE at 384×512** (fraction of pixels whose 5-class winner changed between GT and candidate's last frame). It is a 0/1 disagreement averaged over the 384·512 = 196,608 pixels.

### 1.4 PoseNet path (`modules.py:61-84`)
- **Architecture**: FastViT-T12 backbone (`timm.create_model('fastvit_t12', in_chans=12, num_classes=2048, act_layer=gelu_tanh)`) → summarizer (Linear 2048→512 + ReLU + ResBlock) → Hydra head → `pose` output of **12 dims**.
- **`preprocess_input`** (`modules.py:70-74`): rearrange `b t c h w -> (b t) c h w` → `F.interpolate(size=(384,512), mode='bilinear')` → `rgb_to_yuv6(x)` → rearrange `(b t) c h w -> b (t c) h w` → `(B, 12, 192, 256)`. **Uses BOTH frames** (12 = 6 YUV6 channels × 2 frames). Spatial is halved by the 4:2:0 subsample (384/2=192, 512/2=256).
- **Forward** (`modules.py:76-80`): normalize `(x − 255/2) / (255/4)` (mean 127.5, std 63.75) → vision → summarizer → hydra → `pose` `(B, 12)`.
- **`compute_distortion`** (`modules.py:82-84`): `(out1['pose'][..., :6] − out2['pose'][..., :6]).pow(2).mean(per-sample)`. ✅ **Confirmed: d_pose = MSE on the FIRST 6 of 12 pose dims** (`out // 2 = 12 // 2 = 6`). Dims 7–12 are scored-IRRELEVANT.

### 1.5 Aggregation + score (`evaluate.py:79-92`)
- Per batch: `posenet_dist, segnet_dist = distortion_net.compute_distortion(batch_gt, batch_comp)` then sum over all pairs; final `posenet_dist = Σ/N`, `segnet_dist = Σ/N` (N = total pairs).
- **`rate = compressed_size / uncompressed_size`** where `compressed_size = (submission_dir/'archive.zip').stat().st_size` and `uncompressed_size = sum(f.stat().st_size for f in uncompressed_dir.rglob('*') if f.is_file())`.
- ✅ **Confirmed score formula** (`evaluate.py:92`):
  ```
  score = 100 * segnet_dist + sqrt(posenet_dist * 10) + 25 * rate
  ```

---

## 2. Math derivations (derived, then numerically verified)

### 2.1 `rgb_to_yuv6` — BT.601 FULL-range + 4:2:0 (verified bit-exact)
Upstream (`frame_utils.py:50-78`):
```
Y = clamp(0.299·R + 0.587·G + 0.114·B, 0, 255)
U = clamp((B − Y)/1.772 + 128, 0, 255)
V = clamp((R − Y)/1.402 + 128, 0, 255)
U_sub, V_sub = mean of each 2×2 block (×0.25)            # 4:2:0 chroma subsample
y00,y10,y01,y11 = the 4 luma phases of each 2×2 block    # luma quad (full res preserved)
out = stack([y00, y10, y01, y11, U_sub, V_sub])           # 6 channels @ (H/2, W/2)
```
- **Citation**: ITU-R BT.601-7 luma `Y' = 0.299R' + 0.587G' + 0.114B'`. The chroma denominators are the BT.601 FULL-range (0–255) form: `Cb = (B−Y)/1.772 + 128`, `Cr = (R−Y)/1.402 + 128`, where `1.772 = 2·(1−0.114)` and `1.402 = 2·(1−0.299)`. This is full-range (NOT the limited-range 16–235/16–240 used by `yuv420_to_rgb` in §1.1 for mp4 decode) — the two YUV transforms in the codebase are DIFFERENT and serve different stages.
- The luma quad `[y00,y10,y01,y11]` keeps luma at FULL spatial resolution (split into 4 phase planes), only chroma is subsampled — a deliberate "keep luma sharp, decimate chroma" design.
- **In-place `.clamp_()` + `@torch.no_grad()` (autograd)**: VERIFIED empirically. Under `@torch.no_grad()` the output `requires_grad=False` regardless of the input → **gradients are fully severed** (the dominant gradient-killer). Even without the decorator, `.clamp_()` in-place on an autograd-tracked tensor is autograd-unsafe. The mirror removes BOTH: no decorator + out-of-place `.clamp()`. `clamp` has the correct saturation subgradient (verified: input 300→grad 0, 100→grad 1, −5→grad 0).
- **VERDICT (measured)**: `differentiable_rgb_to_yuv6` vs upstream `rgb_to_yuv6` → **max abs error = 0.0** across 8 random RGB samples. Bit-exact. BT.601 coefficients are exact float32 rationals, so 0 error is expected and observed.

### 2.2 `d_seg` — argmax-flip RATE + DeepFool nearest-boundary saliency
- ✅ The exact metric is the per-pixel argmax-flip RATE (§1.3) — NOT a softmax/logit distance. A pixel contributes 1 iff its 5-class winner changes.
- **DeepFool multiclass nearest-boundary** (Moosavi-Dezfooli, Fawzi, Frossard, CVPR 2016, "DeepFool: a simple and accurate method to fool deep neural networks"): for a pixel currently classified `k*` (winner) with runner-up `k̂`, the minimal input perturbation `Δx` to flip the argmax to the nearest competing class is approximately
  ```
  ||Δx||  ≈  (z_{k*} − z_{k̂}) / ||∇_x (z_{k*} − z_{k̂})||      # nearest decision boundary
  ```
  i.e. the (positive) top-2 logit MARGIN divided by the gradient norm of the margin. Pixels with **small margin AND large gradient** are the cheapest to flip. The task's saliency
  ```
  s_seg_i = ||∇_{x_i}(z_{k̂} − z_{k*})||² / (z_{k̂} − z_{k*})²
  ```
  is exactly the **inverse-squared** DeepFool distance (flip-RISK = 1/Δx²): high where flipping is cheap. This matches the actual SegNet output structure (5 logits/pixel, argmax decision) — the metric is the per-pixel boundary the d_seg flip-rate counts.
- **Cheap gradient-free proxies** (no backward; for a fast prior):
  - **PointRend top-2 softmax gap** (Kirillov et al., CVPR 2020): `1 − (p_{top1} − p_{top2})` — uncertainty = small top-2 gap = near boundary. No gradient needed.
  - **ABL adjacent-pixel KL** (Active Boundary Loss, Wang et al., AAAI 2022): KL divergence between a pixel's class distribution and its neighbours' — boundary pixels have high adjacent-KL. No gradient needed.
- **VERDICT (measured, real SegNet on real 0.mkv last frame)**: gradient finite ✓, nonzero everywhere (frac 1.0), boundary/interior flip-risk ratio = **20,816×** (`flip_risk_boundary_mean=3971.6` vs `flip_risk_interior_mean=0.19`), `margin_min ≈ 0.0004` (a near-flip pixel exists). `s_seg` is sharply boundary-peaked exactly as DeepFool predicts. **`s_seg` is exactly computable against the real frozen SegNet.**

### 2.3 `d_pose` — MSE on first-6 dims + per-pixel Fisher information
- ✅ d_pose = MSE on `pose[..., :6]` (§1.4).
- **`s_pose` derivation**: Treat pose as a function `g: x → pose_{1..6}(x)`. Under iid Gaussian pixel noise `x + ε`, `ε ~ N(0, σ²I)`, the per-pixel Fisher information of the (weighted) pose output is the diagonal of the Gram matrix `JᵀJ` where `J = ∂pose_{1..6}/∂x`:
  ```
  s_pose_i = Σ_{k=1..6} w_k (∂pose_k / ∂x_i)²          # diag(JᵀJ), w_k = per-dim weight (=1 in MSE)
  ```
  This is the squared input-Jacobian = local sensitivity of the scored pose to a perturbation at pixel `i` = the natural RD "how-many-bits-does-this-pixel-deserve" weight for the pose axis. It REQUIRES (a) the **differentiable `rgb_to_yuv6`** (§2.1) and (b) the **differentiable bilinear resize** (`F.interpolate(mode='bilinear')` is differentiable in PyTorch). Both confirmed present in the mirror.
- **VERDICT (measured, real PoseNet on real 0.mkv pair via differentiable mirror)**: gradient finite ✓, nontrivial ✓ (`s_pose_max ≈ 2.6e-6`), 77% of pixels nonzero, top-10%-energy-fraction = 0.943. Magnitudes are tiny because pose is a near-invariant low-dim global function of the frame (ego-motion changes little under per-pixel perturbation — physically correct). **`s_pose` is exactly computable against the real frozen PoseNet through the differentiable mirror.**

### 2.4 Rate / water level (verified against real assets)
- ✅ `rate = archive.zip_bytes / uncompressed_total` (§1.5).
- **`uncompressed_total` for the public test set = 37,545,489 bytes** — VERIFIED: `sum(f.stat().st_size for f in upstream/videos/.rglob('*'))` = 37,545,489 = size of `0.mkv` (the only file). **Exactly matches CLAUDE.md.** (The denominator is the sum of *file sizes on disk* under `--uncompressed-dir`, which for the public set is just `0.mkv`.)
- **Contest-fixed reverse-waterfilling water level**: the rate term contributes `25 · B / N` to the score, so the marginal score-per-byte (the Lagrange "water level" λ for the bit-allocation problem) is
  ```
  λ = 25 / N = 25 / 37,545,489 = 6.659e-7  score-units per byte
  ```
- **Sanity check** — bytes to move the score by 0.001: `25·ΔB/N = 0.001  ⇒  ΔB = 0.001·N/25 = 1,501.82 bytes`. ✅ **~1,502 bytes ↔ 0.001 score.** This is the canonical RD exchange rate: a per-pixel saliency-driven allocation that saves ~1.5 KB of archive is worth as much as a 0.001 distortion improvement.

---

## 3. Differentiable-mirror mapping table (per-step fidelity verdict; measured)

Real-weight measurements: 2 pairs (4 frames) from `0.mkv`, CPU, frozen `segnet.safetensors` + `posenet.safetensors`. "max abs diff" = mirror forward vs upstream forward on identical CHW input.

| # | Upstream step (file:line) | tac mirror function | Fidelity verdict (MEASURED) |
|---|---|---|---|
| 1 | `frame_utils.rgb_to_yuv6` (BT.601 full-range + 4:2:0) | `tac.differentiable_eval_roundtrip.differentiable_rgb_to_yuv6` (and the inline `_rgb_to_yuv6_diff` in `tac.scorer.make_scorers_differentiable`) | **max abs error 0.0** (8 samples). Bit-exact. |
| 2 | `frame_utils.rgb_to_yuv6` `@torch.no_grad` + `.clamp_()` (severs grads) | mirror: no decorator + out-of-place `.clamp()` | **gradient reachability RESTORED** — verified `requires_grad=True`, finite nonzero leaf grad. |
| 3 | `PoseNet.preprocess_input` resize `F.interpolate(size=(384,512), mode='bilinear')` (no align_corners arg) | mirror `_diff_preprocess` `mode='bilinear', align_corners=False` | **bit-exact**: `bilinear` no-arg ≡ `align_corners=False` (verified max_abs_diff 0.0; `align_corners=True` would diverge 0.56). |
| 4 | `PoseNet.forward` normalize `(x−255/2)/(255/4)` + FastViT-T12 + summarizer + Hydra | unchanged frozen weights; `AllNorm.forward` patched `view`→`reshape` (numerically identical) | **PoseNet pose max abs diff 0.0** (first-6 too). Bit-identical on real frames. |
| 5 | `SegNet.preprocess_input` `x[:,-1]` + `F.interpolate(size=(384,512), mode='bilinear')` | NOT patched (upstream method reused directly) | **SegNet logits max abs diff 0.0**, **argmax disagreement 0.0**. Bit-identical. |
| 6 | `PoseNet.compute_distortion` MSE on `pose[...,:6]` | `tac.losses.core.scorer_loss_terms_btchw` `pose_dist = (fp[...,:6]−gp[...,:6]).pow(2).mean()` | **EXACT match** to upstream formula (same slice, same pow-2-mean). |
| 7 | `SegNet.compute_distortion` argmax-flip rate | `tac.losses.core.segnet_surrogate_per_pixel` (SOFT cosine/sinkhorn/fisher-rao) — **NOT the hard argmax** | **DELIBERATE DIVERGENCE**: a differentiable surrogate for the non-differentiable argmax-flip. Training proxy only. The `scorer_loss_terms_btchw` loss is `100·seg_surrogate + sqrt(10·pose_dist + 1e-8)`. ⚠ See GAP-1. |
| 8 | eval inflate→raw uint8 roundtrip (resolution + uint8 quantization) | `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` (bicubic↑874×1164 + bilinear↓ + `Uint8STE`) + `Uint8STE` | TRAINING-TIME simulation of the inflate contract, NOT a step inside the scorer forward. ⚠ See GAP-2. |
| 9 | `compute_distortion` `@torch.inference_mode` | mirror runs WITHOUT inference_mode (gradients enabled) | correct — the mirror must allow backward; numerics identical (verified). |

**Canonical loaders**: `tac.scorer.load_default_scorers` (upstream-faithful, no grads) and `tac.scorer.load_differentiable_scorers` (= `load_default_scorers` + `make_scorers_differentiable`). `score_pair_components` (`tac.substrates.score_aware_common`) routes through `scorer_loss_terms_btchw`. `compute_proxy_score` (`tac.scorer`) applies the full eval-roundtrip ladder + `Uint8STE`.

---

## 4. Saliency computability verdict

| Saliency | Definition | Computable against real frozen weights? | Evidence |
|---|---|---|---|
| **`s_seg`** | `‖∇_{x_i}(z_{k̂}−z_{k*})‖² / (z_{k̂}−z_{k*})²` (DeepFool flip-risk on 5-class SegNet logits, last frame) | **YES — EXACT** | finite ✓, nonzero frac 1.0, boundary/interior ratio 20,816×, margin_min 4e-4 |
| **`s_pose`** | `Σ_{k=1..6} w_k (∂pose_k/∂x_i)²` (pixel-Fisher, both frames, through differentiable rgb_to_yuv6) | **YES — EXACT** | finite ✓, nontrivial ✓, nonzero frac 0.77, top-10% energy 0.943 |

Both saliencies are gradient-reachable through the tac differentiable mirror against the REAL frozen SegNet + PoseNet on REAL `0.mkv` frames, at CPU bit-fidelity to the contest forward. **The foundation for boundary-aware (`s_seg`) + pose-Fisher (`s_pose`) RD bit allocation is verified-faithful.**

---

## 5. GAP list (what's missing to compute either saliency EXACTLY / faithfully)

- **GAP-1 (seg surrogate vs hard argmax-flip)** — `scorer_loss_terms_btchw` uses a SOFT seg surrogate (`segnet_surrogate_per_pixel`, cosine/sinkhorn/fisher-rao), NOT the hard argmax-flip the contest scores. For computing `s_seg` EXACTLY this is fine (the harness backpropagates the raw top-2 logit margin directly, bypassing the surrogate). But any *training loss* that uses `scorer_loss_terms_btchw` optimizes a proxy whose gradient ≠ the true flip-rate gradient at non-boundary pixels. **Resolution for s_seg**: compute the DeepFool margin saliency directly from `segnet(seg_in)` logits (as the harness does) — do NOT route through the soft surrogate.

- **GAP-2 (eval-roundtrip is NOT in the scorer forward)** — `apply_eval_roundtrip_during_training` (bicubic↑874×1164 + bilinear↓384×512 + `Uint8STE`) is a TRAINING-TIME simulation of the inflate→raw-uint8→eval-decode contract. The upstream scorer forward operates on *already-inflated raw uint8 frames* at 874×1164, resized once to 384×512. For s_seg/s_pose to be allocation-faithful to what the contest actually scores, the saliency should be computed on frames that have gone through (or simulate) the same inflate→uint8 roundtrip the submission will produce. The harness here computes saliency on the directly-decoded GT frames (the cleanest signal); a *deployment-faithful* saliency would compose `apply_eval_roundtrip_during_training` (or the real inflate output) BEFORE the scorer. This is an allocation-fidelity refinement, not a correctness blocker — the gradient is correct for the frames it's given.

- **GAP-3 (`s_pose` resolution attribution)** — `s_pose` is computed wrt the 874×1164 camera-native input, but PoseNet internally resizes to 384×512 then 4:2:0-subsamples to 192×256. The gradient correctly flows back through both resizes to camera-native pixels (verified finite). For a bit-allocator that operates in the wavelet/DWT detail-coefficient domain (the Z8 RD lever per the operator's joint P18/P19 direction), `s_pose` must be pushed from camera-native pixel space → the substrate's representation domain via the exact adjoint of that substrate's synthesis (e.g. orthonormal Daubechies analysis DWT). That adjoint is substrate-specific and is the integration point, not a scorer-anatomy gap.

- **GAP-4 (CPU-only fidelity; CUDA axis untested)** — all fidelity numbers are CPU. CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + "MPS auth eval is NOISE": the bit-exactness verdicts are `[macOS-CPU advisory]`. The contest CUDA forward (T4) may diverge in low-order bits (TF32 matmul, kernel order) — NOT measured here. For a *score claim* the saliency-driven allocation must be re-verified on contest-CUDA. For *bit-allocation prior* purposes the CPU saliency is a faithful research signal.

- **GAP-5 (multi-pair / full-video aggregation)** — saliency was computed on 1–2 pairs. The contest d_seg/d_pose are means over all 600 pairs (1200 frames / 2). A faithful RD allocator needs per-pair saliency aggregated over the full video; the harness scales linearly (`--num-pairs N`) but a full-video pass (~600 pairs) is the production input, not run here ($0 budget; per-pair structure confirmed).

- **NO blocker on the two core saliencies**: `s_seg` and `s_pose` are EXACTLY computable against the real frozen weights TODAY. The gaps above are allocation-fidelity refinements (domain adjoint, eval-roundtrip composition, CUDA re-verification, full-video aggregation), not correctness blockers.

---

## Fidelity verdict (one line)
The tac differentiable mirror reproduces the upstream scorer forward **bit-exactly on CPU** (YUV6 0.0, PoseNet pose 0.0, SegNet logits 0.0, SegNet argmax disagreement 0.0) against the real frozen weights on real `0.mkv` frames; the only deliberate divergence is the SOFT seg surrogate used for *training loss* (NOT for saliency).

## Math-verification verdict (one line)
All four math claims verified: YUV6 = BT.601 full-range + 4:2:0 (bit-exact); d_seg = per-pixel argmax-flip rate with DeepFool nearest-boundary saliency (boundary-peaked 20,816×); d_pose = MSE on first-6 pose dims with squared input-Jacobian Fisher saliency (finite, spread); rate denominator N = 37,545,489 and ΔB = 1,501.82 bytes ↔ 0.001 score.

## GAP verdict (one line)
Both `s_seg` (DeepFool flip-risk) and `s_pose` (pixel-Fisher) are EXACTLY computable against the real frozen scorers today; remaining gaps are allocation-fidelity refinements only (soft-surrogate avoidance GAP-1, eval-roundtrip composition GAP-2, representation-domain adjoint GAP-3, CUDA re-verification GAP-4, full-video aggregation GAP-5) — none blocks faithful saliency computation.
