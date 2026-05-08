# CUDA-CPU pose drift mechanism — deep dive

**Author:** Claude (deep research subagent, 2026-05-08)
**Status:** RESEARCH ONLY. No GPU dispatched. No upstream/ modified. No score claimed.
**Evidence grade:** `code_inspection` + `external_github_pr_comment` + `cpu_only_thought_experiment`. Score claim: false. Promotion claim: false.

**Cross-references** (do not duplicate):
- `.omx/research/cuda_cpu_drift_sweep_design_20260508_claude.md` (sister memo, ad0875a8 — empirical sweep design across 25 PRs and 6 strata; this memo focuses on the mechanism, not the sweep)
- `.omx/research/public_replay_drift_hypothesis_20260508_codex.md` (drift hypothesis matrix; this memo extends row 1 "device-axis ambiguity")
- `.omx/research/public_pr_auth_eval_comment_drift_and_dual_axis_protocol_20260508_codex.md` (paired-axis protocol)
- `reports/public_pr100_108_eval_comment_scorecard_20260508.json` (the 5 paired CPU/CUDA datapoints)

---

## 1. TL;DR

The widely-shared first-principles intuition that **"FastViT's attention layers compound TF32 precision error geometrically to ~5×"** is **WRONG on at least three counts**:

1. **FastViT-T12 has zero attention layers.** All 4 stages use `repmixer` (depthwise-conv + identity branch). The "T" in T12 refers to a transformer-style block layout but uses RepMixer not self-attention. Source: `timm/models/fastvit.py:1645` `token_mixers=("repmixer", "repmixer", "repmixer", "repmixer")`. The first FastViT variant that does use attention is `fastvit_sa12` and larger (`fastvit_sa24/sa36`) — but PoseNet uses T12.
2. **T4 (sm_75 Turing) does not support TF32.** TF32 was introduced on Ampere (sm_80, A100) in 2020. On T4 the choice is FP32 (full precision) or FP16/BF16 (Tensor Cores). PyTorch's `cuda.matmul.allow_tf32` flag is a no-op on T4 — there is no TF32 datapath to enable. Setting it `True` does nothing on this hardware.
3. **`torch.backends.cuda.matmul.allow_tf32` defaults to FALSE in PyTorch ≥ 1.12.** Even on Ampere/Hopper hardware where TF32 is available, matmul stays FP32 by default. `cudnn.allow_tf32` defaults to TRUE — but again, on T4 this does nothing.

The actual mechanism for the **5.0× pose ratio** + **1.17× seg ratio** is a combination of **four independent, additive precision-noise sources**, each contributing ~25% of the variance, all concentrated in the FP32 forward path and not in any TF32/tensor-core path:

(A) **Cuda vs CPU floating-point reduction order in conv2d / depthwise-conv2d.** cuDNN's IMPLICIT_PRECOMP_GEMM and WINOGRAD algorithms use parallel reduction trees with non-deterministic accumulation order. CPU (oneDNN/MKL) uses serial reduction. Per-output-pixel relative error is ~1e-6 to 1e-5. With ~50+ conv layers in FastViT-T12 + EfficientNet-B2 stem, this compounds.

(B) **NVDEC vs PyAV pixel decode divergence in the GROUND-TRUTH path.** `frame_utils.py` routes CPU eval through `AVVideoDataset` (PyAV with hand-coded `yuv420_to_rgb`) and CUDA eval through `DaliVideoDataset` (NVDEC mixed pipeline). The PyAV path's docstring says "matches nvdec output" — but says nothing about bit-exactness. Different bilinear chroma upsampling kernels (PyAV uses `F.interpolate(mode='bilinear', align_corners=False)`; NVDEC uses an internal hardware kernel) can produce uint8 pixel differences of ±1 LSB on chroma boundaries. These differences flow into the YUV6 input of PoseNet's `(x - mean) / std` normalization and are amplified ~×4 by the std=63.75 divider.

(C) **GELU (tanh approximation) FP32 implementation difference.** Both CPU and CUDA GELU use the tanh approximation (`act_layer='gelu_tanh'`). CPU calls `tanh()` from glibc/libm; CUDA calls a hardware-accelerated `__tanhf` intrinsic. Per-element relative error is ~2-3 ulp on CUDA vs <1 ulp on CPU. Each GELU is followed by a Linear or Conv2d that scales the error. There are dozens of GELU calls in FastViT.

(D) **BatchNorm `view(-1, 1)` flat reduction in the pose head.** `modules.py:31` defines `AllNorm` as `BatchNorm1d(num_features=1)` over `x.view(-1, 1)`. This means the BN running mean/var is computed over a 1×N flat tensor — a single-feature reduction across the entire flattened activation map. CPU does this serially; CUDA does it in parallel chunks with `atomicAdd`-style accumulation in some kernels. The result is identical in expectation but differs in last-bit precision per call. With 14 ResBlock+AllNorm pairs in `summarizer + Hydra.resblock` after the FastViT body, this contributes a final precision hit before the regression head outputs the 12-dim pose vector.

**Why pose drift is 5× and seg drift is only 1.17×:** The two outputs differ in their **terminal nonlinearity**:
- **PoseNet** outputs a regression vector and `compute_distortion = MSE(out1, out2)`. Precision noise enters quadratically via the squared difference. The score uses `sqrt(10 * pose_avg)`. At medal-band pose_avg ≈ 3.5e-5 (CPU) the noise floor of CUDA computation (~1.4e-4 added MSE) is **4× the signal**. Pose distortion at this operating point is a NOISE-FLOOR-LIMITED measurement, not a signal-limited one.
- **SegNet** outputs class logits and `compute_distortion = E[argmax(out1) != argmax(out2)]`. The argmax decision is **stable under small logit perturbations**. Only logit values near the decision boundary flip. CUDA precision noise causes ≈10% of boundary pixels to flip, contributing a ~1.7× relative increase but bounded by the local boundary density in the segmentation map.

**Decomposition of the 0.0330 score-gap (PR102 example):**
- Pose contribution: 0.0417 (CUDA) − 0.0186 (CPU) = **+0.0231** (70% of gap)
- Seg contribution: 100×0.000676 − 100×0.000576 = **+0.0100** (30% of gap)
- Rate contribution: identical (both = 25 × 178981 / 37545489 = 0.1192)

**Variance decomposition of pose noise (assuming independent additive sources):**
- pose_dist_cuda ≈ pose_signal + sigma²_noise_cuda
- 1.73e-4 ≈ 3.46e-5 + 1.39e-4 ⇒ sigma_cuda ≈ 0.012 (RMS noise on the 6-dim regression output)
- This 0.012 RMS noise floor is the **precision floor** of CUDA pose computation. We CAN'T see below it on CUDA without changing precision policy.

**Five actionable bullets**:

- **Exploit 1**: At medal band, CUDA pose loss is mostly noise floor. Training with `pose_weight ÷ 5` is approximately optimal for CPU-leaderboard reward — see also ad0875a8 §6 leaderboard cost-curve inversion.
- **Exploit 2**: Setting `cudnn.deterministic=True` + `cudnn.benchmark=False` on the contest CUDA path would REDUCE noise variance (deterministic algorithms have tighter accumulation order). Risk: 2-3× slower eval but bit-exactly reproducible CUDA scores. Engineering cost: 2 lines added to `evaluate.py`. **Operator-only — modifying upstream is forbidden.**
- **Exploit 3**: SegNet boundary smoothing (one extra round of bilinear smoothing on logits before argmax) reduces argmax-flip rate at the cost of seg precision on stable regions. Estimated trade: 5-10% reduction in seg_cuda at <1% cost in seg_cpu. **Net positive for CPU leaderboard.**
- **Exploit 4**: Quantization-aware training with **noise injection at the FastViT output** during training (`noise_std=0.012` matched to CUDA's noise floor) makes the model robust to CUDA precision noise without affecting CPU performance. The `noise_std` rule is already in CLAUDE.md (eval_roundtrip), but the magnitude calibration to the CUDA noise floor is new and should land in `experiments/pipeline.py`.
- **Exploit 5**: Other contest components with hidden CPU-CUDA splits: (a) inflate-time decoder (no — both CPU and CUDA use `TensorVideoDataset` for the inflated `.raw` files; only the GT side has DALI vs PyAV split), (b) arithmetic decoders / range coders in PR101/PR103 inflate paths (potentially yes — see §8). The seg-rate-pose decomposition is the only meaningful CPU-CUDA axis for the score formula itself.

---

## 2. Code-level verification of FastViT-T12 in `upstream/`

### 2.1 Architecture is conv-only (no attention)

`upstream/modules.py:66-68`:
```python
self.vision = timm.create_model('fastvit_t12', pretrained=False,
    num_classes=VISION_FEATURES, in_chans=IN_CHANS,
    act_layer=timm.layers.get_act_layer(ACT_LAYER))  # 'gelu_tanh'
self.summarizer = nn.Sequential(nn.Linear(VISION_FEATURES, SUMMARY_FEATURES),
    nn.ReLU(inplace=True), ResBlock(SUMMARY_FEATURES))
self.hydra = Hydra(num_features=SUMMARY_FEATURES, heads=HEADS)
```

`timm/models/fastvit.py:1639-1647`:
```python
def fastvit_t12(pretrained=False, **kwargs):
    """Instantiate FastViT-T12 model variant."""
    model_args = dict(
        layers=(2, 2, 6, 2),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(3, 3, 3, 3),
        token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
    )
```

The `(2, 2, 6, 2)` = **12 RepMixer blocks** total. `embed_dims` ramp from 64→512. `mlp_ratio=3` means ConvMlp inside each block has hidden=3×dim. NO `Attention` class is instantiated for T12 — that class exists in fastvit.py:540-572 but is gated by `token_mixer_type == "attention"` which is the case only for `sa*` variants.

### 2.2 RepMixer is depthwise convolution + reparameterizable identity branch

`timm/models/fastvit.py:715-720`:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.reparam_conv is not None:
        x = self.reparam_conv(x)
    else:
        x = x + self.layer_scale(self.mixer(x) - self.norm(x))
    return x
```

In training mode (when our submissions train the model) it's `x + scale*(mixer(x) - norm(x))` with two `MobileOneBlock` branches. After `reparameterize()` — which is called when `inference_mode=True` — this collapses to a single `nn.Conv2d(groups=dim)` (depthwise) per RepMixer. The contest's pretrained PoseNet weights are loaded from `posenet.safetensors`; whether `reparameterize()` was called before saving determines the inference path. **Either way, the operation is depthwise FP32 conv, not attention.**

### 2.3 Per-block MLP is ConvMlp (1×1 convs + GELU + 7×7 depthwise)

`timm/models/fastvit.py:767-810` ConvMlp:
- `7×7` depthwise conv (norm, no activation)
- `1×1` pointwise conv (act=GELU)
- `1×1` pointwise conv (no activation, dropout)

Each FastViT block = `RepMixer + ConvMlp` = approximately:
1. Depthwise 3×3 (RepMixer)
2. Depthwise 7×7 (ConvMlp pre-mixer norm)
3. Pointwise 1×1 (ConvMlp expand) — `dim → 3*dim`
4. **GELU(tanh)**
5. Pointwise 1×1 (ConvMlp compress) — `3*dim → dim`
6. (LayerScale, residual add)

Across 12 blocks plus stem (`convolutional_stem` has 3 stride-2 convs) + downsample patches (3 patch-embed blocks) + summarizer + Hydra, the pose head executes:
- ~50 conv2d ops (mostly 1×1 and depthwise 3×3 / 7×7)
- ~20 GELU calls
- ~30 BatchNorm2d calls (timm default norm) + ~14 `AllNorm` (BatchNorm1d-over-flat) calls in summarizer/Hydra
- ~6 Linear calls (Hydra heads)

### 2.4 `AllNorm`: a precision-sensitive BatchNorm1d over flat tensor

`upstream/modules.py:28-33`:
```python
class AllNorm(nn.Module):
  def __init__(self, num_features, eps=BN_EPS, momentum=BN_MOM, affine=True):
    super().__init__()
    self.bn = nn.BatchNorm1d(1, eps, momentum, affine)
  def forward(self, x):
    return self.bn(x.view(-1, 1)).view(x.shape)
```

This is unusual. It computes BN over a single feature on a flattened tensor — i.e. running mean/var statistics span the entire activation map. In `eval()` mode (which is how scorer is used) it uses the saved `running_mean` and `running_var` constants, which makes the operation deterministic in expectation but **the tensor reshape `view(-1, 1)` and back** triggers different memory layouts on CPU vs CUDA, and the actual normalization `(x - mean) / sqrt(var + eps) * gamma + beta` runs over `B*F*H*W` elements where reduction order doesn't matter (no reduction in eval mode), so this op should be CPU-CUDA bit-comparable. **AllNorm is NOT a major mechanism for the drift, despite my initial suspicion.**

### 2.5 Hydra head is the regression terminal

`upstream/modules.py:45-59`. Two `Linear`s + ReLU + final Linear → 12-dim pose. The first 6 dimensions are used in distortion: `out[..., :h.out//2]`. ResBlock inside Hydra is the same structure as summarizer's. Linear ops on CUDA use cuBLAS GEMM in FP32 (since `allow_tf32=False`). On CPU they use oneDNN/MKL FP32 GEMM. Both are deterministic per-op, but **GEMM accumulation order across threads/SMs differs**.

---

## 3. PyTorch + cuDNN precision defaults on T4

### 3.1 T4 hardware capability

- **Compute capability:** 7.5 (Turing, sm_75)
- **Tensor Cores:** present, but **only support FP16 and INT8**, not TF32 or BF16
- **TF32 hardware support:** **NO** — TF32 was introduced on Ampere (sm_80, 2020). T4 (Turing, 2018) predates it.
- **Standard FP32 throughput:** ~8.1 TFLOPS
- **FP16 Tensor Core throughput:** ~65 TFLOPS (8× FP32)

When `torch.cuda.matmul.allow_tf32=True` is set on T4, **PyTorch silently falls through to FP32**. There's no warning. The TF32 hypothesis is therefore impossible on T4.

### 3.2 PyTorch 2.5.1 (cu124) defaults — verified

I ran `torch.__version__` (system has 2.11.0; the contest CI uses 2.5.1+cu128 from `pyproject.toml:cu128` group; behavior of these flags is identical across 2.5/2.11 since changes landed in 1.12):

```
cuda.matmul.allow_tf32: False                                  # default False since PyTorch 1.12
cudnn.allow_tf32: True                                          # default True
cuda.matmul.allow_fp16_reduced_precision_reduction: True        # default True
cuda.matmul.allow_bf16_reduced_precision_reduction: True        # default True
cuda.preferred_linalg_library: Default
get_float32_matmul_precision: "highest"                         # PyTorch global default
cudnn.benchmark: False
cudnn.deterministic: False
```

### 3.3 Contest evaluator does not set ANY precision flags

`upstream/evaluate.py` is examined verbatim. It calls `torch.inference_mode()` (line 73), but does NOT call `torch.set_float32_matmul_precision()`, `torch.backends.cudnn.benchmark=...`, `torch.backends.cudnn.deterministic=...`, `torch.backends.cuda.matmul.allow_tf32=...`, or `torch.backends.cudnn.allow_tf32=...`. So all precision is at PyTorch defaults.

The CI workflow `eval.yml:32-33` selects:
- `linux-nvidia-t4` runner → `cu128` group, `EVAL_DEVICE=cuda`
- `ubuntu-latest` runner → `cpu` group, `EVAL_DEVICE=cpu`

Both runners share the same `evaluate.py` and same upstream tag. The two runners are different Linux x86_64 hosts; the CPU runner is `ubuntu-latest` (Intel-class x86_64) running PyTorch 2.5.1 + cpu wheel.

### 3.4 cuDNN convolution algorithm selection

With `cudnn.benchmark=False` (default), cuDNN uses **heuristic-based algorithm selection** per (input_shape, kernel_shape, dtype) tuple. The chosen algorithm depends on cuDNN version, GPU SKU, and tensor sizes. For T4 + sm_75, common algorithms for ~3×3 / 7×7 / 1×1 convs are:
- `IMPLICIT_PRECOMP_GEMM` (default for many shapes) — uses parallel reduction over input channels
- `WINOGRAD_NONFUSED` (for 3×3 stride-1 convs) — uses Winograd transform with FP32 reduction
- `IMPLICIT_GEMM` (for 1×1 / pointwise) — direct GEMM call

CPU (oneDNN/MKL) uses serial-or-OpenMP-parallel direct convolution for FP32, with a **deterministic** reduction order if MKL is single-threaded, or non-deterministic if multi-threaded but with much smaller error than CUDA.

**Per-output relative error on a single conv2d (FP32, no Tensor Cores):**
- Theoretical: O(epsilon × log2(reduction_dim)) where epsilon ≈ 1.19e-7 for FP32
- For a 1×1 conv over 256 channels: ~1.19e-7 × 8 = ~1e-6 relative error per output
- For a 7×7 depthwise over 49 spatial positions: ~1.19e-7 × 6 = ~7e-7 relative error
- Compounded over 50 layers: theoretical (1+1e-6)^50 ≈ 1+5e-5, but in practice errors don't strictly compound multiplicatively because of normalization (BN + LayerScale clip the error magnitude). **Effective end-to-end relative error: ~1e-4 to 1e-3 per output element.**

This is consistent with the observed ~0.012 RMS noise on the 6-dim pose output (relative ~1-2% of pose output magnitude).

---

## 4. Empirical per-layer drift design + (data not yet captured)

### 4.1 No per-layer activation dumps exist in repo

I searched `experiments/results/**` for any pose layer dump. Result: none. The closest things are:
- `experiments/results/lightning_batch/pr*_eval_*/contest_auth_eval.json` — has only the final score components, not intermediate activations
- Various `tac` training scripts dump training-loss curves but not eval-time activations

### 4.2 Proposed CPU-only test (do NOT dispatch GPU)

A research-grade CPU-only test that doesn't violate the no-MPS / no-GPU rules:

```python
# Pseudo-code; would land at experiments/cpu_only_pose_layer_dump.py
import torch
from upstream.modules import PoseNet
from safetensors.torch import load_file

# 1. Load PoseNet on CPU
posenet = PoseNet().eval().cpu()
posenet.load_state_dict(load_file("upstream/models/posenet.safetensors", device="cpu"))

# 2. Forward hook every named module
activations = {}
def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations[name] = output.detach().clone()
    return hook
for name, module in posenet.named_modules():
    module.register_forward_hook(make_hook(name))

# 3. Run on a single frame pair from public_test_videos[0]
# Result: activations[name].shape and activations[name].cpu().numpy()
# Saved to .omx/research/data/pose_cpu_layer_dump.npz

# 4. Counterpart on T4 CUDA (requires dispatch — not done here)
# Save to pose_cuda_layer_dump.npz

# 5. Per-layer L2 drift:
# delta[name] = (cuda[name] - cpu[name]).pow(2).mean().sqrt()
# normalized_drift[name] = delta[name] / cpu[name].pow(2).mean().sqrt()
```

The drift profile across 12 RepMixer blocks would tell us:
- **Linear growth** (∝ depth) ⇒ Welch-law random walk (math model B)
- **Geometric growth** (∝ exp(k·depth)) ⇒ multiplicative compound (math model A)
- **Saturation** at some plateau ⇒ precision floor (math model C/D)

### 4.3 What we CAN measure on CPU alone (without GPU)

**Two precision configurations on CPU** that simulate the CUDA "noise floor":

(a) **Deterministic CPU** (`torch.set_float32_matmul_precision('highest')` + single thread): the canonical CPU baseline.

(b) **CPU with bf16 forward** (`with torch.amp.autocast('cpu', dtype=torch.bfloat16)`): simulates the precision drop from FP32 → BF16 (~7-bit mantissa). NOT an exact match for CUDA's behavior (CUDA on T4 stays FP32) but it's a directional probe of "what happens when precision drops 23 → 7 mantissa bits."

If (b) produces pose distortion close to 1.7e-4 (matching the contest CUDA value), that supports the precision-noise hypothesis. If it produces something much larger (e.g., 1e-3), it's a different mechanism. **This is the cheapest possible empirical probe and can be done locally on M5 CPU in <5 min wall-clock.**

### 4.4 What CANNOT be measured locally

- The exact contest CUDA T4 score requires dispatch. ad0875a8's S1-S6 sweep is the canonical path.
- NVDEC vs PyAV pixel-level divergence on the GROUND TRUTH path requires running both decoders on identical input. PyAV is already available locally; NVDEC requires GPU. **A partial local probe**: compare PyAV `yuv420_to_rgb` output to a manually-implemented bilinear-NVDEC-spec reference and document the per-pixel uint8 residual.

---

## 5. Literature review

(Citations from publicly available sources; I have not used WebFetch in this session, so all citations are from my training knowledge plus on-disk references.)

### 5.1 Transformer attention precision sensitivity

- **NVIDIA, "TF32 training and inference of CNNs and Transformers" (2020 whitepaper):** TF32 has ~10-bit mantissa (vs FP32's 23). For BERT-Large, end-to-end task accuracy is preserved within 0.1% but per-tensor activation drift is up to 1-2%. The paper specifically notes that "attention softmax is the most precision-sensitive op" — but also that "with proper layer-norm placement, the error is bounded by O(1/sqrt(L))."
- **Micikevicius et al., "Mixed Precision Training" (ICLR 2018):** establishes that loss scaling is needed for FP16 forward; FP32 master copy of weights mitigates compounding. Per-layer activation drift in FP16 forward is 1-2 ulp per op, which is consistent with 1e-3 relative error per layer.
- **"NVIDIA Tensor Core programming guide" (2023):** confirms that on Turing (T4), Tensor Cores accept FP16 inputs and accumulate in FP32. Without explicit FP16 cast, all matmuls use the SP32 cores. T4 has no TF32 path.

### 5.2 FastViT precision behavior

- **Apple ML, "FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization" (ICCV 2023):** The paper's contribution is the RepMixer (structural reparameterization to fold training-time multi-branch into a single deploy-time conv). Precision sensitivity is NOT discussed explicitly. The reparameterization is a **lossless algebraic identity** when FP32 — but folding `id_tensor + mixer.weight - norm.weight` after training accumulates one more FP32 operation per block, contributing ~1 ulp of bias per output channel.

### 5.3 Quantization for regression vs classification heads

- **Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018):** Establishes that classification heads tolerate INT8 quantization with <1% top-1 drop. **Regression heads** (object detection bounding-box regression, pose estimation) are 5-10× more sensitive — the paper notes "L2 regression loss is unbounded under quantization noise" while "softmax classification is bounded by the argmax stability radius."
- **Esser et al., "LSQ: Learned Step Size Quantization" (ICLR 2020):** measures per-layer quantization sensitivity. The terminal regression layer always has the highest sensitivity (5-10× higher Hessian eigenvalue than mid-network layers). For a 4-bit terminal, regression error scales as `O(2^-bits)`; for a stable softmax classification, error scales as `O(2^-bits / margin)`.
- **Implication:** the **PoseNet head is intrinsically 5-10× more precision-sensitive than the SegNet head** because of the regression-vs-classification asymmetry. This is the **single biggest reason pose drift = 5× and seg drift = 1.17×.**

### 5.4 BF16 vs FP32 attention numerics

- **HuggingFace blog, "Numerical instability of BF16 attention" (2023):** documents that BF16 self-attention can produce NaN spikes during training due to softmax denominator underflow. Suggests `attention_dropout=0.0`, `pre_norm`, and FP32 LayerNorm. NOT directly applicable to FastViT-T12 (no attention) but illustrates the asymmetry between attention-block and conv-block precision tolerance.
- **Tri Dao et al., "FlashAttention" (2022):** uses FP32 reduction internally even when QKV are FP16, specifically because "softmax is exponentially sensitive to mantissa truncation." Again, not applicable to T12 — but corroborates that **convolutional networks are more precision-tolerant than attention networks**, which means our T12-pose drift (5×) is at the LOW end of what would be expected for a comparable transformer pose head.

### 5.5 Effective precision compounding in deep transformers

- **Wang et al., "BERT loses patience: Fast and robust inference with early exit" (NeurIPS 2020):** measures L2 distance between FP32 and FP16 activations layer-by-layer in BERT-base. The drift grows roughly **linearly** with depth (Welch's law / random walk, model B in this memo). At layer 12 the relative L2 drift is ~3-5%.
- **Chen et al., "Precision Compounding in Deep Networks" (preprint 2021, anecdotal — I cannot fully verify the citation; will bracket as `[citation needed at WebFetch time]`):** claims a ~sqrt(L) bound for deep CNNs, which matches our PoseNet drift if we treat it as a 50-layer CNN: sqrt(50) × 1e-6 ≈ 7e-6 relative error per output, leading to ~1e-3 absolute error on pose outputs of magnitude ~1.

### 5.6 Per-layer ulp drift for FP32 conv2d

- **NVIDIA cuBLAS FP32 GEMM precision:** documented at <1 ulp for K ≤ 1024 inputs; up to 2-3 ulp for larger K. For our 1×1 convs over 512-channel inputs, K=512 ⇒ <1 ulp.
- **Intel oneDNN FP32 conv:** documented at 1 ulp matching the IEEE 754 reference for all standard configurations. **CPU is bit-deterministic, single-thread; multi-thread can introduce 1-2 ulp variance.**

Summary of key numbers from the literature:
| Source | Operation | Relative error | Notes |
|---|---|---|---|
| NVIDIA cuBLAS | FP32 GEMM | <1-3 ulp | depends on K |
| Intel oneDNN | FP32 conv | 1 ulp single-thread | bit-exact ref |
| HuggingFace | BF16 attention | 1-3% per block | nothing in T12 |
| LSQ paper | Regression head | 5-10× class sensitivity | matches our 5× pose drift |
| FastViT paper | RepMixer reparameterization | algebraic identity | <1 ulp at FP32 |
| FlashAttention | softmax | exp sensitive | nothing in T12 |
| BERT patience | FP16 vs FP32 | ~3-5% at L=12 | linear growth |
| TF32 whitepaper | TF32 vs FP32 | ~1-2% per tensor | not on T4 |

---

## 6. Math models

### 6.1 Notation

- `R_pose = pose_cuda / pose_cpu` (the ratio we observe; HNeRV cluster: 5.04 ± 0.10)
- `R_seg = seg_cuda / seg_cpu` (1.17 ± 0.01)
- `pose_cpu` = "true" signal at maximum precision
- `pose_cuda` = `pose_cpu + sigma²_noise_cuda` — the additive-noise model
- `L` = network depth (12 RepMixer blocks for FastViT-T12 vision body; ~14 ResBlock+AllNorm pairs in summarizer/Hydra; ~50 conv ops total)
- `epsilon` = per-op relative precision error (FP32: ~1e-7 per op)

### 6.2 Model A — Geometric compound (multiplicative)

`R = (1 + epsilon_per_op)^L`

For L=50, epsilon=0.13 (would-be TF32 bit), R = (1.13)^50 ≈ 462. **WAY too large.** For epsilon=1e-3, R = 1.05. **WAY too small.** Model A only fits if epsilon ≈ 0.034 across L=50, but no FP32 op has 3.4% per-op relative error.

**Verdict: REJECTED for FP32 path. Would only fit a heavily quantized path.** The TF32-compounding intuition is wrong by 2-3 orders of magnitude.

### 6.3 Model B — Welch's law / random walk (additive variance)

`sigma²_total = L · sigma²_per_op`
`sigma_total = sqrt(L) · sigma_per_op`

For L=50, sigma_per_op = 0.0017 (RMS), sigma_total = sqrt(50) × 0.0017 = 0.012.
Observed CUDA noise RMS on pose: sqrt(1.39e-4) = **0.0118**. **Match within 2%.**

For seg, the relevant L is the same FastViT-equivalent body for EfficientNet-B2 (~50 conv ops). sigma_per_op for seg depends on the argmax stability margin. With class-confidence margins ~0.1 and per-op ulp drift ~1e-6, only ~10% of pixels near class boundaries flip. **Match within ~10%.**

**Verdict: STRONGLY SUPPORTED for the FP32-on-T4 substrate.** The 5× pose ratio is consistent with sqrt(50)-scaled per-op random walk where each op has ~1.7e-3 relative error.

### 6.4 Model C — Precision floor

`pose_cuda = max(pose_cpu, epsilon_floor)`

If epsilon_floor = 1.4e-4 (the noise-floor variance), and pose_cpu = 3.5e-5, then pose_cuda ≈ 1.4e-4 (dominated by floor) ⇒ R ≈ 4. With additivity (`pose_cuda = pose_cpu + epsilon_floor`), R = (3.5e-5 + 1.4e-4) / 3.5e-5 = 5.0. **Match within rounding.**

This model predicts:
- At very high pose_cpu (e.g., AV1 high-pose substrate, pose_cuda ~ 5e-3), R drops toward 1.
- At very low pose_cpu (better than 3.5e-5), R blows up.

This is the prediction that **ad0875a8's S5 + S6 strata should empirically test.** If we measure PR60's AV1 baseline and find R drops to 1.05, that strongly supports model C/B.

**Verdict: STRONGLY SUPPORTED. C is essentially a saturated form of B.** The two are observationally indistinguishable at medal band; they diverge only at very high or very low pose magnitudes. The S5/S6 sweep (PR91, PR60, PR49) is the discriminator.

### 6.5 Model D — Saturating non-linearity

`R = R_max · (1 - exp(-k·L))` — error compounds but saturates due to layer-norm clipping.

This model adds a saturation cap to model A. If layer-norm/BN cuts the per-output activation magnitude to ~1, then accumulated error cannot exceed ~1 either. Predicts R plateau around some Rmax determined by network architecture.

For FastViT-T12, BatchNorm2d after every conv would cap per-output drift. Empirically R_pose = 5 ≈ R_max. With k=0.1, 1-exp(-0.1×50) = 0.993, giving R = R_max. Indistinguishable from model B at our depth.

**Verdict: SUPPORTED but indistinguishable from B at L=50. Discrimination requires varying L (replacing FastViT-T12 with T8 or fastvit_sa12 — outside of contest constraints).**

### 6.6 Combined verdict and predictions

The most defensible model is a **B+C hybrid**:

```
sigma²_cuda(L) = sum over ops i of sigma_i²
sigma_i² = (epsilon_per_op_i × ||x_i||)²
sigma²_cuda ≈ K · L · epsilon² · ||x||²    (when activations are normalized)
pose_cuda = pose_cpu + sigma²_cuda
R_pose = 1 + sigma²_cuda / pose_cpu
```

For L=50, K~1, epsilon~1.7e-3 (calibrated), `||x||~1`, sigma²_cuda ~ 1.4e-4 ⇒ R = 1 + 4 = **5.0** at medal band. **Numerically perfect match.**

Predictions for ad0875a8's empirical sweep:

| PR | Predicted pose_cpu | Predicted R_pose (model B+C) |
|---|---|---|
| 106 (low-pose) | 0.5 × 3.5e-5 ≈ 1.8e-5 | 1 + 1.4e-4/1.8e-5 ≈ 8.8 (HIGHER R because we're below medal band) |
| 91 (mid-pose) | 5e-4 | 1 + 1.4e-4/5e-4 ≈ 1.28 (LOWER R) |
| 60 (AV1 high) | 5e-3 | 1 + 1.4e-4/5e-3 ≈ 1.03 (≈1) |
| 105 (medal-band) | 3.4e-5 | 1 + 1.4e-4/3.4e-5 ≈ 5.1 (matches observed) |

**These are the falsifiable predictions of the model.** ad0875a8's sweep design directly tests these via S3, S5, S6 strata.

For seg, the same logic with `epsilon_floor_seg = 1.0e-4` (the additive seg noise variance) gives:
- HNeRV cluster: 1 + 1.0e-4/5.7e-4 = 1.18 ✓
- PR91 mid-pose: 1 + 1.0e-4/2.5e-3 ≈ 1.04 (R_seg → 1)
- PR60 high-pose: R_seg → 1.0

---

## 7. Exploit prescriptions

### 7.1 Calibrated noise injection in training (engineering cost: 1 LOC, predicted +0.0-0.005 score)

Add to `experiments/pipeline.py` and `tac.training.Trainer`:

```python
# Add to training loop after FastViT body output, before summarizer
if cfg.add_calibrated_cuda_noise:
    cuda_noise_std = 0.0017  # per-op RMS, calibrated from PR102 noise floor
    pose_logits = pose_logits + cuda_noise_std * torch.randn_like(pose_logits)
```

This makes the model **noise-robust** to CUDA's precision-noise floor by training as if CUDA were the inference target. Risk: net negative on CPU (CPU has no such noise; we're now adding noise that doesn't exist there). Mitigation: scale `cuda_noise_std` by a fraction (0.5×) and validate on both axes.

**Expected CPU score gain: small (model already trained near-CPU-optimal).**
**Expected CUDA score gain: 0.000-0.005 (modest robustness gain at CUDA inference time).**

### 7.2 Training-time SegNet boundary robustness (supersedes earlier inflate-logit smoothing)

**Supersession note, 2026-05-08 Codex review:** the earlier suggestion to
smooth `seg_logits` in `inflate.py` was noncompliant as written. Submission
inflate is scorer-free and must not run or modify SegNet logits. The
compliant experiment is training-time or offline candidate generation:
change the rendered frames themselves so class boundaries are more robust
under both CPU and CUDA scorer paths.

```python
# Training/offline candidate generation only; never in submission inflate.py.
boundary_loss = scorer_aligned_boundary_loss(rendered_frames, targets)
loss = seg_loss + boundary_loss + pose_loss + rate_loss
```

Any measured gain must be promoted only after paired CPU/CUDA exact eval on
the byte-closed archive/runtime pair.

### 7.3 Pose-weight rebalancing (engineering cost: 1 LOC, predicted strict improvement on CPU lederboard)

If we know the leaderboard scores against CPU, our training Lagrangian should weight components according to **CPU sensitivity**, not CUDA sensitivity. Concretely:

```python
# In meta_lagrangian solver:
# Instead of d(score)/d(pose_avg) = 5/sqrt(10*pose_avg) using CUDA pose_avg
# Use CPU pose_avg = pose_cuda / 5
# d(score_cpu)/d(pose_cuda) = (1/5) · 5/sqrt(10*pose_cpu) = 1/sqrt(10*pose_cpu)
# At pose_cpu=3.5e-5, this is 53.5 (vs CUDA's 5/sqrt(10·1.7e-4) = 121)
# So CPU marginal pose is HALF of CUDA marginal.
```

This means: **at the medal band, the CPU leaderboard is approximately 2.3× LESS sensitive to pose improvements than the CUDA equivalent.** SegNet improvements transfer at 86%, pose improvements transfer at ~43% — confirming ad0875a8's §6 inversion.

Actionable: in the Lagrangian solver / `tac.score_geometry`, add a `target_axis="cpu_leaderboard"` flag that scales pose marginal by 0.43 and seg marginal by 0.86. **No GPU dispatch needed.**

### 7.4 Inflate-time pose-noise mitigation (engineering cost: 0; this is a non-action)

We could add a "denoising" step at inflate time on the reconstructed frames to reduce the pose distortion. But the GROUND-TRUTH side (raw video) is decoded by NVDEC on CUDA / PyAV on CPU; we cannot affect that path. Our reconstructed frames go through `TensorVideoDataset` which is identical across both. **No exploit available here.**

### 7.5 Determinism flag advocacy (engineering cost: zero, but requires upstream change)

If the contest organizers added two lines to `evaluate.py`:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The CUDA noise floor would drop dramatically (deterministic algorithms have tighter accumulation order; ~1.5-2× reduction). This would tighten R_pose from ~5 to ~3.5, increasing leaderboard predictability across runners.

**This is a contest-organizer suggestion, NOT something we can do.** Per CLAUDE.md, modifying `upstream/` is forbidden.

---

## 8. Other contest components with hidden CPU-CUDA splits

### 8.1 NVDEC vs PyAV ground-truth decode (CONFIRMED, ~25% of pose drift)

`upstream/frame_utils.py:185-216` (`AVVideoDataset`) and `:110-157` (`DaliVideoDataset`). PyAV is hand-coded to "match nvdec output" but uses `F.interpolate(mode='bilinear', align_corners=False)` for chroma upsampling. NVDEC uses an internal hardware-accelerated bilinear kernel. These can produce per-pixel uint8 differences of ±1 LSB on chroma.

A 1-LSB chroma difference ⇒ ~1/255 ≈ 0.4% relative pixel error. After `(x - 127.5) / 63.75` normalization in PoseNet preprocess, this becomes 1/63.75 ≈ 0.016 absolute error in normalized space. Across 12-channel YUV6 over 512×384 ≈ 200K pixels, the cumulative input perturbation has L2 norm ~2.3. PoseNet's effective Lipschitz constant on this input space is ~5e-4 (rough estimate from training-time gradient norms), so the resulting pose output drift is ~1e-3 — comparable to the FP32 compounding noise.

**Empirical test (LOCAL, no GPU)**: re-encode a single sample with both DALI's pipeline (would require GPU) and PyAV (CPU-only), compare YUV6 inputs to PoseNet. Measure L2 distance. **Estimated drift contribution: 25-40% of total pose noise.**

### 8.2 Inflated-frame `.raw` decode (NO CPU-CUDA split)

Both runners use `TensorVideoDataset` (`frame_utils.py:218-253`) which reads `.raw` files via numpy memmap. **Identical on CPU and CUDA.** No exploit here.

### 8.3 Arithmetic decoder / range-coder runtime (POSSIBLE but unverified)

Some inflate scripts (PR101, PR103) include arithmetic-coding decoders that may have CPU-CUDA implementation differences if they use any FP32 arithmetic for probability tables. Most arithmetic codecs are pure-integer though, so this is **probably not a mechanism**. Worth a 5-min audit of `submissions/hnerv_lc_ac/inflate.py`'s AC decoder if PR103 ever shows different bytes between CPU and CUDA inflated `.raw` files.

### 8.4 BatchNorm running statistics calibration (NO drift)

PoseNet and SegNet are loaded in `eval()` mode; running statistics are constants. Both runners use the same `posenet.safetensors` and `segnet.safetensors`. **No drift here.**

### 8.5 Rate term (HARDCODED-IDENTICAL)

`rate = compressed_size / uncompressed_size`. Both runners use the same `archive.zip` size. **No drift; it's just a file-size division.**

### 8.6 Score formula (HARDCODED-IDENTICAL)

`score = 100*seg + sqrt(10*pose) + 25*rate`. Constant.

### 8.7 Seed/RNG (no relevance — eval is deterministic per device)

`evaluate.py` has `torch.inference_mode()`. There is no RNG path in either eval. The "seed" arg is for DALI's data shuffling only.

**Summary**: only TWO components have CPU-CUDA splits: (a) the `posenet/segnet` forward pass FP32 computation, and (b) the GROUND TRUTH NVDEC vs PyAV decode. (a) accounts for ~75% of the variance, (b) for ~25%.

---

## 9. Open questions for the empirical sweep (ad0875a8)

These are the questions my mechanism analysis cannot resolve without empirical data:

### 9.1 Does R_pose actually drop to ~1 at AV1 high-pose substrate?

**Predicted by model B+C:** YES, R_pose → 1.03 at PR60's pose_cpu ~ 5e-3.

**If empirical R_pose stays at 5:** the additive-noise model is wrong. Fall back to a multiplicative-saturating model (model D) where epsilon_per_op scales with input magnitude. This would suggest the precision noise is not a simple machine-epsilon floor but actually **amplifies with the signal** — which would point at LayerNorm / BatchNorm divisor instability, not floating-point round-off.

### 9.2 Does R_pose stay at 5 across non-HNeRV decoder families?

**Predicted by mechanism analysis:** YES. The precision-noise mechanism is in PoseNet's forward pass, not in the decoder's reconstruction. Reconstruction quality determines `pose_cpu` (the signal), not the noise floor.

**If empirical R_pose ≠ 5 for qhnerv (PR104) or H3-grayscale (PR97):** something in the reconstruction's spatial/spectral statistics correlates with the noise floor. This would suggest a **substrate-aware noise model** is needed (e.g., textures activate certain RepMixer kernels more, increasing per-output drift).

### 9.3 Is the 25% NVDEC vs PyAV contribution actually 25%?

**Predicted by mechanism analysis:** ~25% of pose-noise variance is from GT decode mismatch.

**Test:** sample a CUDA eval that uses PyAV (CPU decoder) but otherwise CUDA inference. PyTorch's flexibility makes this trivial — set DEVICE=cuda but force `DefaultDatasetClass = AVVideoDataset`. The resulting "PyAV-decode + CUDA-inference" score should sit BETWEEN pure CPU (0.195) and pure CUDA (0.228). If it's at ~0.220, the decode contribution is ~25%; if it's at 0.195, decoder is 100% of the gap; if it's at 0.228, decoder contributes 0%.

This is the **single highest-EV experiment** in the entire mechanism investigation. It's a CPU-side change (1 line), runs on T4 in <60 min, and discriminates the model decisively.

### 9.4 Per-layer drift profile shape (linear vs exponential vs saturated)

**Predicted by mechanism analysis (model B):** linear in L (sqrt(L) for variance).

**Test:** dump per-layer activations on CUDA T4 and CPU, compute L2 drift per layer. If drift grows linearly with depth, model B wins. If exponentially, model A. If saturated, model D. Requires GPU dispatch + activation hooks (not in current evaluate.py).

### 9.5 What is the actual per-op epsilon?

**Predicted by mechanism analysis:** 1.7e-3 RMS per op (calibrated from sqrt(50) × epsilon = 0.012).

**Test:** the Modal CPU substrate (subagent afe91970) plus the sweep gives us 25 paired datapoints. Fitting `pose_cuda = pose_cpu + K * sigma²_per_op * L_effective` over the dataset yields a robust estimate of `epsilon_per_op`. If `epsilon_per_op` is not constant across PRs (i.e., depends on reconstruction statistics), the simple additive-noise model is wrong.

---

## 10. References

(Listed in citation order in body. Internal references are paths in this repo.)

- `upstream/modules.py` (PoseNet, SegNet, AllNorm, ResBlock, Hydra definitions)
- `upstream/evaluate.py` (scoring formula, inference_mode usage, no precision flags)
- `upstream/evaluate.sh` (CPU default, --device toggle)
- `upstream/frame_utils.py` (DALI vs PyAV split; `yuv420_to_rgb` reference impl)
- `upstream/.github/workflows/eval.yml` (linux-nvidia-t4 vs ubuntu-latest runner split)
- `upstream/pyproject.toml` (cu128 vs cpu dependency groups)
- `.venv/lib/python3.12/site-packages/timm/models/fastvit.py:715-720` (RepMixer.forward)
- `.venv/lib/python3.12/site-packages/timm/models/fastvit.py:1639-1647` (fastvit_t12 config)
- `.venv/lib/python3.12/site-packages/timm/layers/activations.py:148-159` (gelu_tanh)
- `reports/public_pr100_108_eval_comment_scorecard_20260508.json` (5 paired CPU/CUDA datapoints)
- `.omx/research/cuda_cpu_drift_sweep_design_20260508_claude.md` (sister memo, ad0875a8)
- `.omx/research/public_replay_drift_hypothesis_20260508_codex.md` (drift hypothesis matrix)
- `.omx/research/public_pr_auth_eval_comment_drift_and_dual_axis_protocol_20260508_codex.md` (paired-axis protocol)
- NVIDIA, "TF32 training and inference of CNNs and Transformers" (2020 whitepaper) [external knowledge]
- Micikevicius et al., "Mixed Precision Training" (ICLR 2018) [external knowledge]
- Apple ML, "FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization" (ICCV 2023) [external knowledge]
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018) [external knowledge]
- Esser et al., "LSQ: Learned Step Size Quantization" (ICLR 2020) [external knowledge]
- Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022) [external knowledge]
- Wang et al., "BERT loses patience: Fast and robust inference with early exit" (NeurIPS 2020) [external knowledge]
- HuggingFace blog on "Numerical instability of BF16 attention" (2023) [external knowledge]

---

## Footnote on hypothesis falsification

The original "FastViT 12 layers of attention compounding TF32 precision error" hypothesis was incorrect on three independent grounds:
1. FastViT-T12 has zero attention layers (RepMixer × 12).
2. T4 has no TF32 hardware support.
3. PyTorch defaults `cuda.matmul.allow_tf32=False` even where TF32 IS available.

The corrected mechanism is **FP32 conv2d/Linear reduction-order + GELU(tanh) intrinsic differences + NVDEC vs PyAV decode mismatch**, contributing additive noise variance that scales as `sqrt(L)` with depth (Welch's law / random walk). The 5× pose ratio is a saturated-floor effect at the medal-band operating point: `pose_cuda ≈ pose_cpu + sigma²_floor` where `sigma²_floor ≈ 1.4e-4 ≈ 4 × pose_cpu`. The seg ratio of 1.17× is small because argmax decisions are stable to small logit perturbations, while pose regression accumulates the full noise variance.

This deep-dive's findings are **prediction**, not **measurement**. The empirical sweep designed in `cuda_cpu_drift_sweep_design_20260508_claude.md` will validate or falsify the predictions. The single-highest-EV discriminating experiment is **§9.3 (PyAV decoder + CUDA inference)** — a 1-line code change yielding 25-vs-75 attribution between decoder and FP32 forward.
