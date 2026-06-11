# Running the comma video-compression-challenge evaluator fast AND at fidelity on Apple Silicon

> **STATUS: DRAFT — external sharing pending operator approval.**
> This document is an internal draft prepared for possible later sharing with comma.ai and the
> Apple-Silicon ML community. It has not been reviewed for public release. Do not publish, email,
> or mirror to an external repository without explicit sign-off.

---

## TL;DR

The comma video-compression-challenge scorer is two PyTorch networks — a **SegNet** (segmentation)
and a **PoseNet** (ego-motion) — that turn a candidate decoded video into the two distortion terms
that drive the leaderboard. Running that evaluator on **Apple Silicon GPUs** is awkward: PyTorch has
no Metal training backend, its MPS path is numerically unfaithful for this scorer, and CoreML defaults
to FP16. So we ported both networks to **MLX** (Apple's array framework), loading the upstream PyTorch
checkpoints, and built a **parity-audit harness** that measures exactly the quantities the contest
charges.

The genuinely-interesting, reusable result:

- **MLX-CPU is bit-faithful to PyTorch-CPU** for the charged metrics (2 argmax disagreements out of
  19.66M pixels, both at genuine float32 ties; pose component drift ~1e-10).
- **MLX-GPU (Metal) is fast and usable** with a small, boundary-confined, *characterized* drift whose
  root cause is **Metal vs CPU float32 reduction-order non-associativity** (not a porting bug) — first
  amplified at the Squeeze-and-Excite global-average-pool.
- The practical pattern: use **MLX-GPU for the fast signal**, recompute the **absolute metrics on
  PyTorch-CPU** as the authority gate (especially the pose term, which has a steep marginal near the
  operating point).

All MLX numbers below are tagged `[MLX-CPU]` or `[MLX-GPU]`; **PyTorch-CPU is the only authority**, and
any number from it is tagged `[torch-CPU authority]`. No leaderboard scores or strategy appear here.

---

## 1. The problem: GPU-fast scoring on a Mac, without losing fidelity

The challenge scorer is argmax-sensitive. The segmentation distortion term, `d_seg`, is the **rate at
which the per-pixel class argmax disagrees** between the reference and the candidate (computed on the
last frame). The pose distortion term, `d_pose`, is a **mean-squared error on the first 6 PoseNet
output dimensions** of a two-frame input. Because `d_seg` is a discrete argmax-flip count and `d_pose`
is squared error at a small operating point, **tiny numerical drift can be score-material** — a few
millivolts of logit noise at a class boundary flips a pixel; a 1e-4 pose drift can rival the pose
signal itself when the candidate is good.

That makes the usual "just run it on the GPU" options unsafe:

| Path | GPU? | Fidelity vs PyTorch-CPU authority | Verdict |
|---|---|---|---|
| PyTorch-CPU | No (no Metal backend) | **AUTHORITY** | The oracle — but slow, no GPU |
| PyTorch **MPS** | Yes | Numerically unfaithful for this scorer (large pose drift; reorders most rankings) | Avoid for fidelity |
| `torch.compile` Metal / Inductor | Yes | Same MPS numerics underneath (a speed layer over MPS) | Not a fidelity escape |
| **CoreML** (coremltools) | Yes (GPU/ANE) | Defaults to FP16; FP32 forceable but GPU/ANE numerics uncertain | Risky for argmax-flip `d_seg` |
| **MLX** (CPU + GPU) | Yes | High; numpy-like IEEE float32 semantics | **The path** |

MLX is the only Apple-GPU path that keeps the IEEE-float32 semantics this argmax-sensitive scorer needs.
The rest of this guide describes the port and, more usefully, the **measured fidelity envelope** so you
know exactly when an MLX number can be trusted and when it must be gated by a PyTorch-CPU recompute.

> Why "numerically unfaithful" for MPS rather than a specific number: the MPS drift class for this scorer
> is large enough (especially on the pose head) that we treat MPS as a non-authority signal entirely. The
> reusable point is the *envelope*, which we measured carefully on MLX; the MPS figure is not needed to
> make the case and we deliberately omit it.

---

## 2. The port

The port lives in two repo-relative modules:

- `tac/local_acceleration/mlx_scorer_adapters.py` — the MLX port itself (the layer adapters and the
  weight-conversion entry points).
- `tac/local_acceleration/mlx_scorer_torch_parity.py` — the parity-audit harness that measures the
  charged quantities against PyTorch.

Both are MIT-licensed (`SPDX-License-Identifier: MIT`).

### 2.1 Architecture coverage

Both networks are fully ported as small, composable **adapter** classes. Each adapter takes the
corresponding upstream PyTorch module and exposes an MLX forward; a top-level converter
(`torch_distortion_net_to_mlx`) walks the upstream `DistortionNet` and builds the whole MLX scorer with
weights copied from the PyTorch checkpoint.

**SegNet** = an EfficientNet-B2 encoder + a Unet decoder + a 5-class segmentation head
(the upstream model is `segmentation_models_pytorch`'s `Unet` over a `tu-efficientnet_b2` encoder):

- `MLXEfficientNetStemAdapter`, `MLXEfficientNetStageAdapter`, `MLXInvertedResidualAdapter`,
  `MLXDepthwiseSeparableConvAdapter` — the EfficientNet-B2 backbone blocks.
- `MLXEfficientNetSqueezeExciteAdapter` / `MLXSEModuleAdapter` — the Squeeze-and-Excite blocks
  (global-average-pool → reduce → SiLU/ReLU → expand → sigmoid gate). *This is the first drift
  amplifier on GPU — see §3.*
- `MLXUnetDecoderAdapter` (+ decoder blocks with bilinear upsample, skip concat, conv) and
  `MLXSegmentationHeadAdapter` (final 5-class logits).
- Assembled by `MLXSegNetAdapter` (`encoder → decoder → segmentation_head`).

**PoseNet** = a FastViT-T12 backbone + a "Hydra" pose head (the upstream model is a `timm` FastViT in
eval/inference mode):

- `MLXMobileOneBlockAdapter` — the MobileOne stem blocks (multi-branch conv + optional SE).
- `MLXRepMixerBlockAdapter` (+ `MLXRepMixerAdapter`) — the FastViT token-mixer blocks.
- `MLXPatchEmbedAdapter` — the patch-embed downsampling between stages.
- `MLXFastVitVisionAdapter` — the FastViT stage stack.
- `MLXHydraAdapter` — the pose-regression head (vision features → summary → ResBlock → 12-dim pose,
  first 6 used by the scorer).
- Assembled by `MLXPoseNetAdapter`.

**The scorer** ties them together: `MLXDistortionScorerAdapter` runs `posenet(yuv6_pair)` and
`segnet(last_rgb)`; `scorer_distortion_components_numpy(...)` computes the two charged terms from the
outputs — `d_seg` as the per-pixel argmax-disagreement rate and `d_pose` as the MSE on the first 6 pose
dims. These are the exact quantities the contest charges.

The port is **NCHW-aware at the boundary**: upstream tensors are NCHW; MLX convs work in NHWC. The
adapters wrap each forward with `nchw_to_nhwc` / `nhwc_to_nchw` so callers can pass the same tensors they
would pass to PyTorch and get NCHW outputs back.

### 2.2 Eval-roundtrip and YUV6 — the details that matter for matching the evaluator

Two preprocessing details are load-bearing for matching the evaluator and were reproduced carefully:

- **Eval-roundtrip.** The evaluator decodes a candidate through the same resize/quantize roundtrip the
  contest applies before scoring. Any scoring that skips that roundtrip measures a different surface than
  the evaluator. The MLX scorer is fed inputs from a cache builder whose ground-truth RGB decode is a
  **byte-identical structural copy of the upstream `frame_utils.yuv420_to_rgb`** — same plane extraction,
  bilinear chroma upsample (`align_corners=False`), BT.601 limited-range, clamp/round/uint8. This is what
  makes the comparison apples-to-apples rather than "a model that resembles the evaluator."
- **YUV6 input for PoseNet.** PoseNet consumes a 12-channel two-frame YUV6 tensor (per frame: 4 luma
  taps + 2 subsampled chroma), resized and normalized exactly as upstream. The cache supplies
  `posenet_yuv6_pair` in this layout; the adapter consumes it directly.

The SegNet side takes the **last frame's RGB** (`segnet_last_rgb`), matching the upstream slice.

### 2.3 The parity-audit harness

`mlx_scorer_torch_parity.py` builds a manifest that compares the MLX scorer against the upstream PyTorch
scorer on a fixed window of real scorer-input pairs, reporting:

- `segnet_argmax_diff_pixels` — the `d_seg` charge (count of flipped argmax pixels), with a **strict
  threshold of 0**;
- `posenet_component_abs_max` — the `d_pose` charge (per-dim, the way the scorer charges it), with a
  strict threshold of `2.0e-5`;
- raw logit / output deltas for diagnostics.

It also ships:

- `build_mlx_segnet_layer_trace_manifest` — a layer-by-layer drift trace that localizes *where* GPU drift
  first appears (used to find the root cause in §3);
- a conv2d accumulation probe (`build_mlx_conv2d_accumulation_probe_manifest`) and a
  `MLXReferenceConv2dAdapter` with `fixed_fp32` / `kahan_fp32` / `fixed_fp64` accumulation modes, so the
  drift can be tightened deterministically when needed;
- a GPU fail-closed guard (`GPU_RESEARCH_SIGNAL_BLOCKER`) that prevents an MLX-GPU run from being misread
  as an authority score — GPU manifests must be explicitly opted into as research-signal.

---

## 3. The fidelity result (the genuinely-interesting part)

Measured on the real reference-video scorer cache (600 two-frame pairs; representative sample = first 100
pairs = **19,660,800** SegNet argmax pixels). PyTorch-CPU is the authority; MLX numbers are the
comparison.

| Quantity | `[MLX-CPU]` vs `[torch-CPU authority]` | `[MLX-GPU]` vs `[torch-CPU authority]` |
|---|---|---|
| **`d_seg` argmax pixels flipped** (of 19.66M) | **2** | **243** |
| `d_seg` flip rate (overall) | 1.0e-7 | **1.24e-5** |
| min top-2 logit margin at a flipped pixel (mean) | 2.4e-7 (≈ float32 ULP) | 5.2e-5 |
| SegNet logit abs-max delta | 5.7e-5 | 9.6e-2 |
| **`d_pose` component abs-max** (the charged term) | **8.7e-11** | **2.76e-4** (mean 2.22e-4) |
| PoseNet raw 12-dim output abs-max delta | 2.3e-5 | 4.1e-2 |
| pure scorer-forward throughput (SegNet+PoseNet) | — | **~8.7 pairs/s** (≈920 ms / batch-8) `[MLX-GPU]` |

Source: the drift audit memo. Hardware: M5 Max, MLX 0.31.1, PyTorch 2.11.0 (CUDA unavailable → PyTorch-CPU
is the authority). No MPS used anywhere.

**Reading the table:**

- **MLX-CPU is essentially bit-faithful.** 2 flipped pixels out of 19.66M, and both sit at near-perfect
  ties (mean top-2 margin ≈ float32 ULP) — i.e. genuine argmax ties, not a port defect. Pose component
  drift ~1e-10 is float32 round-off. For the metrics the contest charges, **MLX-CPU = PyTorch-CPU at the
  charged precision.** It is a fully usable, near-authority cross-check that you can run on the Apple GPU
  machine's CPU.
- **MLX-GPU drifts but stays tiny on `d_seg`.** 243 flipped pixels / 19.66M = **0.00124% of pixels**, and
  **every** flip is at a segmentation decision boundary (mean margin 5.2e-5) where the logit drift just
  crosses a near-tie. A raw-logit drift this size that does *not* cross a class boundary is harmless. For
  training and sensitivity work this `d_seg` signal is trustworthy.
- **MLX-GPU pose is the one caution.** The pose-component abs drift (2.76e-4) is small in absolute terms,
  but a good candidate's `d_pose` is itself small — so **near a strong operating point the GPU pose drift
  can rival the pose signal.** MLX-GPU pose is fine as a *relative* gradient/ranking signal; it is **not**
  trustworthy for an *absolute* `d_pose` readout near the operating point without a PyTorch-CPU recompute.

### 3.1 Root cause: Metal vs CPU float32 reduction-order non-associativity

A SegNet GPU layer trace (two real pairs) shows the first divergence cliff and how it propagates:

| Layer | max abs delta | note |
|---|---|---|
| `encoder.stage_0.block_0.se` | 3.97e-3 | **first cliff** — Squeeze-and-Excite global-average-pool reduction |
| `encoder.stage_0.block_1.conv_dw` | 3.46e-2 | depthwise conv accumulation |
| `encoder.stage_1/2/3 .conv_pw / .bn2` | 3.5e-2 – 6.5e-2 | pointwise GEMM reduction-order drift accumulates |
| `decoder.block_1` | 6.87e-2 | upsample + concat + conv amplifies |
| `segmentation_head.logits` | 9.64e-2 | final logits — but only flips argmax at near-ties |

The CPU path has ~0 flips, which proves the port is **numerically correct** — there is no layout,
PixelShuffle, transpose, interpolation-mode, BatchNorm-eps, YUV6-basis, or argmax-tie-break bug. The
GPU drift is the classic **GPU-vs-CPU float32 non-associativity** in conv / GEMM / pooling accumulation:
summing the same numbers in a different order gives a slightly different float32 result. The
**Squeeze-and-Excite global-average-pool is the first amplifier** because reductions over H×W (here
192×256) are the most order-sensitive operation in the network. In the port the SE pool is literally a
nested mean reduction (`mean over W` then `mean over H`), which is exactly the kind of reduction whose
order Metal and the CPU disagree on.

**This drift characterization is the reusable insight.** Anyone running an argmax-sensitive vision model
on Apple GPUs will hit the same class of drift; the playbook is: (1) keep a CPU oracle, (2) trace the
network to find the first reduction-order amplifier (usually a global pool or a large GEMM), (3) decide
per-metric whether the drift is below your decision threshold, and (4) where it isn't, route the
offending reduction through a fixed-order or Kahan/fp64 accumulator. The port already ships the
deterministic accumulator modes (`fixed_fp32` / `kahan_fp32` / `fixed_fp64`) for exactly this; for the
scorer they are largely unnecessary because the GPU `d_seg` flip rate (~1.2e-5, boundary-confined) is
already negligible for training.

### 3.2 Practical guidance (the operating pattern)

- **Use MLX-GPU for the fast signal.** SegNet gradients/sensitivities and relative pose rankings during
  training are trustworthy on MLX-GPU; it is the lever that turns a CPU-infeasible loop into a GPU-fast one.
- **Gate absolute metrics on PyTorch-CPU.** Recompute the absolute `d_seg` / `d_pose` on PyTorch-CPU
  before any decision that depends on the absolute value (promotion, comparison, reporting). MLX-CPU can
  serve as a cheap near-authority cross-check because it is bit-faithful.
- **Watch pose near the operating point.** When the candidate is good and `d_pose` is small, the
  PyTorch-CPU pose recompute is mandatory — that is the one regime where the GPU drift can be the same
  order as the signal.

---

## 4. The reusable landscape — build on these, don't hand-port from scratch

If you want your own MLX scorer (or any EfficientNet/Unet/FastViT-class model on Apple GPUs), much of the
work is already done in the open-source ecosystem. Credit and pointers:

- **mlx-image** (`github.com/riccardomusmeci/mlx-image`) — MLX **EfficientNet-B0..B7 (including B2)**, the
  SegNet *backbone*, with automatic PyTorch `.pth` → MLX safetensors conversion and ImageNet-validated
  parity. Reuse the encoder + its weight-conversion machinery (encoders only — no Unet decoder).
- **apple/ml-fastvit** (`github.com/apple/ml-fastvit`) — the official FastViT (the PoseNet backbone),
  authored by Apple, with PyTorch + CoreML export. No ready MLX port, but it is the authoritative
  reference to port FastViT-T12 from (Apple authored both FastViT and MLX).
- **ExecuTorch MLX delegate** (PyTorch-official, 2025) — runs PyTorch models on the Apple GPU via MLX with
  FP32 support; great as a **zero-port spike** if your model fits its supported ATen-op set (it cannot run
  arbitrary `torch.nn` modules — custom blocks like the smp-Unet decoder, FastViT RepMixer, or the Hydra
  head may not export).
- **torch2mlx** (`github.com/SynapticSage/torch2mlx`) and **Xforge** (`github.com/SattamAltwaim/Xforge`)
  — general PyTorch → MLX converters; Xforge has built-in parity testing and auto-detects ViT/ResNet.
  Reuse for weight conversion + an automated parity harness.
- **YOLO-MLX** (`github.com/thewebAI/yolo-mlx`) — demonstrates a CNN + a segmentation/mask **decoder head**
  running in MLX at parity and faster than MPS; a working reference that Unet-style decoders are feasible
  and fast in MLX.
- **coremltools** (`apple.github.io/coremltools`) — viable if you can force FP32 and validate the GPU/ANE
  numerics against a CPU oracle; the FP16 default is the trap for argmax-sensitive metrics.

**Recommended strategy** (avoids re-porting the backbones): try the ExecuTorch-MLX FP32 zero-port spike
first; if custom ops block it, reuse **mlx-image EfficientNet-B2** + **apple/ml-fastvit** for the heavy
backbones and hand-port only the small custom heads (Unet decoder + 5-class seg head; Hydra pose head),
using Xforge/torch2mlx for weight conversion and a parity harness. The invariant either way: a CPU oracle
stays the authority, and every MLX piece ships with a **real-input** parity gate (not a zero-init gate —
zero-init parity hides weight-key mismatches).

---

## 5. Benchmark: ours vs alternatives

We measured every viable Apple-GPU backend for this scorer against the PyTorch-CPU authority, on the same
real reference-video scorer-input cache (the byte-identical `frame_utils.yuv420_to_rgb` decode, not a
PyAV rgb24 path). The fidelity columns are the exact contest charges: `d_seg` is the per-pixel
argmax-disagreement count vs `[torch-CPU authority]`, and `d_pose` is the abs drift on the first-6 PoseNet
dims (the way the scorer charges it). Every non-authority number is `[research-signal]` and
**non-promotable** — this is a portability + fidelity + speed characterization, not a score claim.

### 5.1 The unified table

| Backend | Device | `d_seg` flip vs authority | `d_pose` abs drift vs authority | GPU? | port effort | fidelity verdict |
|---|---|---|---|---|---|---|
| **PyTorch-CPU** | CPU | **0 (AUTHORITY)** | **0 (AUTHORITY)** | No | none (the oracle) | `[torch-CPU authority]` — the oracle, but no GPU |
| **owned MLX port** | `[MLX-CPU]` | **2 / 19.66M** (1.0e-7; both fp32 ties) | 8.7e-11 (≈round-off) | No | reuse (built) | **bit-faithful** = PyTorch-CPU |
| **owned MLX port** | `[MLX-GPU]` (default arch) | 243 / 19.66M (1.24e-5; all boundary near-ties) | 2.76e-4 (mean 2.22e-4) | Yes | reuse (built) | seg OK for training; pose needs authority gate near the operating point |
| **owned MLX port** | `[MLX-GPU]` **+ arch override** | **0 / 19.66M** (0.0) | **8.7e-11** (≈round-off) | Yes | reuse (built) + 1 env var | **FP32-exact** — see §5.2 |
| **ExecuTorch-MLX delegate** PoseNet | `[ExecuTorch-MLX]` | n/a (PoseNet only) | **7.6e-6** (rel-MSE ~5e-14) | Yes | none (zero-port) | **FP32-exact** ✅ |
| **ExecuTorch-MLX delegate** SegNet | `[ExecuTorch-MLX]` **+ arch override** | **0 / 1.57M** (0.0; logit Δ 4.3e-5) | n/a | Yes | none (zero-port) + 1 env var | **FP32-exact** ✅ (was a kernel-bug NO-GO without the override — see §5.2) |
| **CoreML** FP32 SegNet | `[CoreML-FP32]` | **0 / 1.97M** (0.0; logit Δ 4.5e-5) | n/a | Yes (GPU/ANE) | reuse (trivial export) | **FP32-exact** ✅ |
| **CoreML** FP32 PoseNet | `[CoreML-FP32]` | n/a | **1.9e-5** (rel-MSE ~6e-14) | Yes (GPU/ANE) | reuse (trivial export) | **FP32-exact** ✅ |
| **mlx-image** EfficientNet-B2 | `[MLX-GPU]` | **arch mismatch** (not a SegNet drop-in) | n/a (encoder-only) | Yes | hand-port (timm→torchvision remap) | backbone port itself is exact (cosine 0.99999999 vs torchvision-CPU) but it is the **torchvision** B2, **not** the **timm** B2 the contest SegNet uses |

Notes on the table:

- All fidelity numbers are traced to the measured memos (the owned-port envelope is the §3 drift audit; the
  arch-override, ExecuTorch, CoreML, and mlx-image rows are the backend-comparison + arch-override audits).
  CoreML / ExecuTorch per-call latency is round-trip-dominated, not a throughput ceiling, so we report
  **fidelity** (the load-bearing result) rather than head-to-head throughput; the owned MLX-GPU forward
  runs at ~8.7 pairs/s (SegNet+PoseNet, batch-8) `[MLX-GPU]` from §3.
- **mlx-image is faithful but the wrong variant.** Its EfficientNet-B2 is essentially exact (cosine
  0.99999999, top-1 20/20) and ~30× faster on GPU than torchvision-CPU — but it is the *torchvision* B2,
  whose feature-pyramid taper (`[32, 16, 24, 48, 88, 120, 208, 352, 1408]`) differs from the *timm*
  `tu-efficientnet_b2` encoder our SegNet uses (`[3, 16, 24, 48, 120, 352]`). Hosting the contest SegNet
  encoder there would need a verified per-tensor timm→torchvision weight remap **and** a re-derivation of
  the Unet decoder's skip-connection indices — a real hand-port, not a drop-in. It is the right *reference +
  tooling*, not a reusable SegNet.

### 5.2 The share-worthy reusable insight: `MLX_METAL_GPU_ARCH` makes MLX-GPU FP32-exact on NAX-capable Apple GPUs

This is the genuinely-useful finding for the Apple-Silicon ML community, and it generalizes far beyond this
scorer. §3 traced the MLX-GPU drift to **Metal-vs-CPU float32 reduction-order non-associativity**, first
amplified at the Squeeze-and-Excite global-average-pool and accumulating through the pointwise GEMMs. We
have since localized the *specific* cause and a one-line fix:

- On recent Apple GPUs that are **NAX-capable** (the M5's GPU reports architecture `applegpu_g17s`), MLX
  routes fused convolution/GEMM epilogues (the many 1×1 pointwise convs in EfficientNet-B2, the FastViT
  GEMMs) through a **NAX-tile reduction-order kernel** (`steel_gemm_fused_nax`). That tile's accumulation
  order differs from PyTorch-CPU's at float32 — **this NAX reduction order is the drift source.**
- Forcing a **non-NAX GPU arch** routes the same ops through the plain `steel_gemm_fused` kernel, whose
  reduction order matches PyTorch-CPU at float32. MLX exposes a documented env var for exactly this kernel
  selection:

  ```
  MLX_METAL_GPU_ARCH=applegpu_g15      # (or g14 / g16 — any non-NAX arch)
  ```

  The variable must be set **before the MLX runtime initializes** (before `import mlx` in the process).

Measured effect on the owned MLX-GPU scorer, over the same 100 real pairs (19.66M argmax pixels):

| Quantity | `[MLX-GPU]` default (g17s / NAX) | `[MLX-GPU]` + override (g15 / non-NAX) |
|---|---|---|
| `d_seg` argmax flips (of 19.66M) | 243 | **0** |
| `d_seg` flip rate | 1.24e-5 | **0.0** |
| SegNet logit abs-max delta | 9.64e-2 | **1.02e-4** (float32 ULP) |
| `d_pose` component abs-max (the charged term) | 2.76e-4 | **8.7e-11** (≈round-off) |
| parity gate verdict | FAIL | **PASS** |
| forward throughput | baseline | **unchanged** (≤4% noise at batch 4/8/16) |

Three properties make this trustworthy rather than a coincidence:

- **It is kernel-selection, not a numerics change.** The fixed result is **arch-invariant** across g14 /
  g15 / g16 (all non-NAX → the same plain kernel), giving identical 0-flip / ~2e-11 results. The override
  changes which Metal GEMM kernel JIT-compiles; it does not alter what the scorer computes.
- **It is free.** Forward throughput is unchanged within noise at every batch size — the "more faithful AND
  not slower" outcome. The deterministic Kahan / fp64 accumulator modes the port ships (§3.1) become
  unnecessary in production: the override alone reaches 0 flips.
- **It generalizes.** The same one-line override independently makes the PyTorch-official **ExecuTorch-MLX
  delegate** FP32-exact on the Apple GPU for **both** nets — its SegNet path was otherwise a hard NO-GO,
  crashing in the NAX GEMM JIT (a missing-type-name compile error in the NAX-tile kernel). The fix is the
  same: route ExecuTorch's MLX runtime to the non-NAX kernel. So this is not a quirk of one hand-port; it
  is a property of the NAX kernel path that any MLX-on-Apple-GPU user of an argmax-sensitive vision model
  can apply.

> **Anti-knob (load-bearing):** **never set `MLX_ENABLE_TF32`** for a fidelity path. TF32 is lower than
> float32 and would silently re-introduce drift. The arch override reaches float32-exactness *with* MLX's
> default fast-math on; the only knob that matters is `MLX_METAL_GPU_ARCH`, and the only knob to avoid is
> `MLX_ENABLE_TF32`. (`MLX_DISABLE_COMPILE` is numerics-neutral and only slows things down; leave it unset.)

One integration caveat: `MLX_METAL_GPU_ARCH` is **process-wide**, so it forces *all* MLX work in that
process onto the non-NAX kernels. For a scorer that is fine (it's the win); for a co-resident MLX workload
that genuinely wants the native NAX kernels, either run the scorer in its own process or accept the
override process-wide after confirming the rest of the pipeline tolerates non-NAX (a small decoder, for
instance, does not hit the NAX-fused-GEMM path and is unaffected).

### 5.3 Recommended backend pairing

Putting the table and the override together, the practical recommendation for a GPU-fast, fidelity-safe
scorer on Apple Silicon:

- **PoseNet → ExecuTorch-MLX FP32 (or CoreML-FP32).** Both are FP32-exact and zero-port. PoseNet is the
  net whose drift can rival its own signal near a good operating point, so a faithful zero-port GPU PoseNet
  is the higher-value win.
- **SegNet → MLX-GPU with the `MLX_METAL_GPU_ARCH=applegpu_g15` override** (already built, MLX-native so
  gradients are available for a training loss, and FP32-exact under the override). CoreML-FP32 SegNet (also
  0 flips) is an excellent forward-only alternative when you want the most faithful absolute readout and do
  not need an MLX-native gradient path.
- **PyTorch-CPU stays the authority gate.** Recompute the absolute `d_seg` / `d_pose` on PyTorch-CPU before
  any decision that depends on the absolute value. With the arch override the GPU↔CPU gap is at the float32
  floor, so the gate becomes a periodic confirmation rather than a per-step necessity — but it remains the
  only authority, and `[MLX-CPU]` is the cheap bit-faithful cross-check between gates.

mlx-image EfficientNet-B2 is **not** recommended as a SegNet path (wrong B2 variant; would require a
hand-port). It remains the best open reference if you ever do port the SegNet encoder from scratch.

---

## 6. What is cleanly extractable as standalone OSS

A clean, genuinely-useful standalone repo (separate from any contest-strategy work) could contain:

**Extractable (proposed standalone scope):**

- The **MLX scorer adapters** — the EfficientNet-B2 + Unet + 5-class-head SegNet and the FastViT-T12 +
  Hydra-head PoseNet, plus the NCHW↔NHWC boundary wrappers and the `MLXReferenceConv2dAdapter` with its
  deterministic accumulation modes. (Repo-relative source: `tac/local_acceleration/mlx_scorer_adapters.py`.)
- The **PyTorch-parity framework** — the manifest builder, the `d_seg` argmax-flip and `d_pose`
  component-drift metrics, the layer-trace drift localizer, and the conv2d accumulation probe.
  (Repo-relative source: `tac/local_acceleration/mlx_scorer_torch_parity.py`.)
- A **minimal example** — load the upstream checkpoints, build the MLX scorer, run a parity check on a
  handful of frames, print the `[MLX-CPU]` / `[MLX-GPU]` vs `[torch-CPU]` drift table from §3.

These are MIT-licensed in our tree and have no dependency on any private infrastructure or contest
strategy — they are a general "fast + faithful Apple-GPU scorer" tool.

**Stays internal:** anything tied to our research state, training loops, dispatch tooling, score history,
or strategy. The standalone repo would be the *scorer port + parity harness + example* only.

**License / asset note:** the port code is MIT and ours to release. The **upstream model checkpoints and
the contest video/evaluator assets are the contest's**, not ours to redistribute — a standalone repo must
point users at the official challenge for those, ship only the port + harness + example code, and load
weights from the user's own copy of the upstream checkpoints.

---

## Provenance and tagging

Every quantitative claim in this guide comes from the measured drift audit and the port/harness source.
Numbers are tagged by axis throughout: `[torch-CPU authority]` is the only authority; `[MLX-CPU]` is the
bit-faithful near-authority cross-check; `[MLX-GPU]` is the fast research signal that must be gated by a
PyTorch-CPU recompute for absolute metrics. No leaderboard scores, score-lowering strategy, credentials,
private infrastructure, or local filesystem paths appear in this document by design.

> **STATUS: DRAFT — external sharing pending operator approval.**
