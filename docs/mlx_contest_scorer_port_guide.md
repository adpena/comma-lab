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

> **PENDING** — this section will be filled in with the ours-vs-others fidelity + performance comparison
> currently running. The intended table compares the owned MLX port against the OSS reuse paths in §4
> (e.g. mlx-image EfficientNet-B2 + ported heads, and the ExecuTorch-MLX delegate where it exports) on:
> per-net forward throughput, `d_seg` argmax-flip rate vs PyTorch-CPU, and `d_pose` component drift vs
> PyTorch-CPU, on both MLX-CPU and MLX-GPU. Until then, the measured envelope in §3 is the authoritative
> fidelity reference for the owned port.

| Path | Device | Throughput | `d_seg` flip rate vs torch-CPU | `d_pose` abs drift vs torch-CPU |
|---|---|---|---|---|
| _owned MLX port_ | MLX-CPU | _pending_ | 1.0e-7 (measured, §3) | 8.7e-11 (measured, §3) |
| _owned MLX port_ | MLX-GPU | ~8.7 pairs/s (measured, §3) | 1.24e-5 (measured, §3) | 2.76e-4 (measured, §3) |
| _mlx-image EfficientNet-B2 + ported heads_ | MLX-CPU/GPU | _pending_ | _pending_ | _pending_ |
| _ExecuTorch-MLX delegate (if exports)_ | MLX-GPU | _pending_ | _pending_ | _pending_ |

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
