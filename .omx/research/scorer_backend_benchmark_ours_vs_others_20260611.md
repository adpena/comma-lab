# Scorer-backend benchmark: OURS vs OTHERS — unified fidelity + performance table (2026-06-11)

**Operator ask (2026-06-11):** *"test and compare the outcomes and performance and results of ours
versus those others — more signal is better."* Produce the UNIFIED fidelity+performance comparison of
every viable scorer-backend option for the comma contest scorer (SegNet + PoseNet), against
**torch-CPU authority**, to fill the guide's benchmark placeholder and tell the training wire-in which
backend to use per net.

**Authority discipline (CLAUDE.md, binding):** torch-CPU is the **ONLY** fidelity authority. Every
non-torch-CPU row is `[macOS-MLX/ExecuTorch/CoreML research-signal]` and **NON-PROMOTABLE**. NO score /
promotion / rank / kill claim is made from any number here — this is portability + fidelity + speed
characterization for the training wire-in decision. MPS never touched. Real 0.mkv frames via the
byte-identical `frame_utils.yuv420_to_rgb` decode (the reference-video scorer-input cache, NOT PyAV
rgb24). The exact frontier pointer did NOT move; this is an enabler.

**Method (efficiency: consumed the already-measured numbers, newly measured only the gaps):**
- **CONSUMED** (not re-measured): ExecuTorch spike (`executorch_mlx_delegate_scorer_spike_20260611.md`),
  our-MLX drift audit (`mlx_scorer_port_drift_audit_20260611.md`), OSS survey
  (`mlx_scorer_existing_oss_derisk_survey_20260611.md`).
- **NEWLY MEASURED** (this session): (1) **mlx-image EfficientNet-B2** fidelity + speed; (2)
  **CoreML-FP32** SegNet d_seg-flip + PoseNet d_pose drift; (3) **ExecuTorch SegNet kernel-pin**
  feasibility (diagnosed, not chased — see §4).
- Sample: real GT-decoded frames from
  `experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600` (`segnet_last_rgb`
  = last-frame resized 384×512 SegNet input; `posenet_yuv6_pair` = the 12-ch YUV6 pair PoseNet sees).
  20–30 pairs for the new measurements (drift is stationary across windows per the consumed audit, so a
  small sample is representative for the comparison).
- All speeds measured with the capstone daemon (pid 72123) contending for torch-CPU → torch-CPU
  throughput numbers are a **lower bound** (uncontended is faster); the GPU/CPU ratio is conservative.

---

## THE UNIFIED TABLE

`d_seg-flip` = per-pixel argmax-disagreement count vs torch-CPU (the exact contest d_seg charge; lower =
more faithful). `d_pose drift` = abs delta on the first-6 PoseNet dims (the d_pose charge), with the
relative-MSE in parens. Speed columns: `fwd pairs/s` is the measured forward throughput on the listed
device; CoreML rows are **per-call `.predict()` latency-dominated**, NOT a throughput ceiling.

| Backend | SegNet d_seg-flip vs authority | PoseNet d_pose drift vs authority | fwd pairs/s | est min/epoch @600 | port effort | GPU? | fidelity verdict |
|---|---|---|---|---|---|---|---|
| **torch-CPU** (authority) | **0 (AUTHORITY)** | **0 (AUTHORITY)** | SegNet 1.50¹ · Pose 8.9 · enc 2.62 | ~6.7 (full) | none (the oracle) | **NO** | AUTHORITY — but no GPU |
| **our MLX-CPU** | **2 / 19.66M** (1.0e-7; both fp32 ties) | 8.7e-11 (≈0) | ~CPU-bound | — | **reuse (built)** | NO | **bit-faithful** = torch-CPU |
| **our MLX-GPU** | 243 / 19.66M (1.24e-5; all at boundary near-ties) | 2.76e-4 (mean 2.22e-4) | **8.7** (SegNet+Pose) | **~1.1** | **reuse (built)** | **YES** | seg OK for training; **pose needs authority gate near frontier** |
| **ExecuTorch PoseNet GPU** | n/a (PoseNet only) | **7.6e-6** (rel_mse 4.97e-14) | ~7² (0.14s/call) | — | **none (zero-port)** | **YES** | **FP32-EXACT** ✅ |
| **ExecuTorch SegNet GPU** | **BLOCKED** (didn't run) | n/a | — | — | none-if-fixed | YES(part.) | **kernel-bug NO-GO** (steel_gemm JIT; §4) |
| **mlx-image EffNet-B2** | **ARCH MISMATCH** (not a SegNet drop-in; §2) | n/a (encoder-only) | **123**³ (cls@288²) | — | **hand-port** (timm→tv remap + decoder reindex) | **YES** | backbone port **exact** (cos 0.99999999) but **wrong B2 variant** |
| **CoreML-FP32 SegNet** | **0 / 1.97M** (flip rate 0.0; logit Δ 4.5e-5) | n/a | 2.67² | — | **reuse (trivial export)** | **YES** | **d_seg-EXACT** ✅ |
| **CoreML-FP32 PoseNet** | n/a | **1.9e-5** (rel_mse 5.99e-14) | 8.7² | — | **reuse (trivial export)** | **YES** | **d_pose-EXACT** ✅ |

¹ torch-CPU full-SegNet 1.50 pairs/s and PoseNet 8.9 pairs/s were measured *contended*; uncontended is
faster. The full DistortionNet (SegNet+PoseNet) at ~1.5 pairs/s → ~6.7 min/epoch @600 is the bottleneck
the GPU paths attack.
² CoreML / ExecuTorch per-call rows are dominated by the Python `.predict()` / single-call round-trip
overhead, not GPU compute — a batched/persistent-session path would be much faster. The fidelity is the
load-bearing result; the throughput numbers here are conservative call-latency, not a ceiling.
³ mlx-image B2 = 123 pairs/s is the full backbone+classifier at ImageNet 288² (its native eval res), GPU.
At the SegNet 384×512 it would be slower but still GPU-fast; this number certifies the backbone is fast,
not that it's a SegNet drop-in.

### Headline rows of newly-measured backends (the JSON receipts)

- **mlx-image B2 implementation fidelity** (apples-to-apples: mlx-image-GPU vs torchvision-CPU, SAME
  ImageNet weights, identical pixels): logit `max_abs_delta=1.67e-5`, `cosine_mean=0.99999999`,
  `top1_argmax_agree=20/20`, **123 pairs/s GPU vs 4.08 pairs/s torchvision-CPU (~30×)**. The port is
  numerically excellent.
- **CoreML-FP32 SegNet**: `argmax_flips=0 / 1,966,080` (10 real frames), `logit_abs_max_delta=4.5e-5`.
  **d_seg-exact** — better than our MLX-GPU SegNet (243 flips).
- **CoreML-FP32 PoseNet**: `pose_component_abs_max_delta=1.9e-5`, `pose_rel_mse_vs_torch=5.99e-14`.
  **d_pose-exact.**

---

## §1 — torch-CPU is the only fidelity authority (and the only one with no GPU)

PyTorch has no Metal backend; torch-CPU is the oracle but cannot use the Apple GPU. Every GPU path below
is judged ONLY by how faithfully it reproduces torch-CPU's exact charged quantities (argmax-flip d_seg,
first-6-dim d_pose). Full DistortionNet on torch-CPU is ~1.5 pairs/s (≈6.7 min/epoch @600 contended) —
the bottleneck the GPU backends exist to remove, with torch-CPU retained as the periodic authority gate.

## §2 — mlx-image EfficientNet-B2: a FAITHFUL, FAST backbone — but the WRONG B2 variant for our SegNet

This is the key nuance the survey's "REUSE the SegNet encoder" optimism missed. **Two separable results:**

1. **The mlx-image B2 *implementation* is essentially exact** (cosine 0.99999999, max_abs 1.67e-5 vs
   torchvision-CPU, top-1 20/20) and **~30× faster on GPU** (123 vs 4.08 pairs/s). The port quality is
   excellent; this is a genuinely reusable, fidelity-faithful, GPU-fast EfficientNet-B2.

2. **But it is the *torchvision* B2, not the *timm* B2 our SegNet uses.** The contest SegNet is
   `smp.Unet('tu-efficientnet_b2', ...)` whose encoder is **timm `EfficientNetFeatures`**. Measured
   feature-pyramid taper mismatch:
   - **Contest timm encoder** returns 6 maps, channels **`[3, 16, 24, 48, 120, 352]`** (the `out_indices`
     smp's `TimmUniversalEncoder` selects for the Unet skip connections).
   - **mlx-image torchvision B2** has 9 `features` entries, channels
     **`[32, 16, 24, 48, 88, 120, 208, 352, 1408]`** — an extra 88-ch stage timm's pyramid skips, and a
     different MBConv block-grouping / SE-reduction-channel layout.
   - Consequence: mlx-image cannot host the contest SegNet encoder weights without **(a)** a verified
     per-tensor timm→torchvision remap of ~500 weights AND **(b)** re-deriving the smp Unet decoder's
     skip-connection indices for the torchvision taper. That is a real **hand-port**, not a reuse.

   **Verdict:** mlx-image is the right *reference + tooling* if we ever port the SegNet encoder, but it is
   **not a drop-in** for the timm-arch contest SegNet. Our already-built MLX SegNet port (timm-arch,
   bit-faithful on CPU) and CoreML-FP32 SegNet (d_seg-exact) both dominate it for our actual net.

## §3 — CoreML-FP32: the survey's pessimism is REFUTED — both nets are exact at FP32

The survey flagged CoreML as "defaults to FP16 (fidelity loss); FP32 forceable but ANE/GPU numerics
uncertain → risky for argmax-flip d_seg." **Measured, with `compute_precision=ct.precision.FLOAT32` +
`compute_units=CPU_AND_GPU` + `convert_to='mlprogram'`:**
- **SegNet: 0 argmax flips / 1.97M pixels** (d_seg-EXACT), logit Δ 4.5e-5.
- **PoseNet: pose rel_mse 5.99e-14** (d_pose-EXACT), abs max Δ 1.9e-5 (fp32 ULP on the dominant ~34 dim).
- **Both nets export trivially** (no kernel bug, no arch mismatch — `torch.jit.trace` + `ct.convert`,
  765 ops for PoseNet, exports in ~2–4 s).

So the FP32-forced CoreML path is a **fully-faithful, zero-hand-port GPU path for BOTH nets** — the
single best-fidelity GPU option measured for SegNet (ties ExecuTorch on PoseNet). Caveat: the per-call
`.predict()` latency is high (2.67–8.7 pairs/s here is round-trip-dominated, not throughput); a batched
or persistent-session integration is needed before CoreML beats our MLX-GPU on raw speed, and the FP32
fidelity must be re-confirmed under batching + on a larger sample before any wire-in.

## §4 — ExecuTorch SegNet kernel-pin: diagnosed as NOT pip-pinnable (left as the spike found it)

The spike's SegNet blocker was `steel_gemm_fused_nax.h:23:20: error: unknown type name 'GEMMParams'` — a
**version/build skew between ExecuTorch's *vendored* MLX kernel sources** (`backends/mlx/third-party/mlx`)
**and the installed mlx-metal JIT**. Because the ExecuTorch MLX backend ships its OWN vendored kernel
sources, pinning a different `mlx-metal` version *in the venv* does NOT change the kernels ExecuTorch
compiles — the fix lives in ExecuTorch's vendored tree, i.e. it is an upstream build-skew bug requiring an
ExecuTorch rebuild/patch, not a `pip install mlx-metal==X` pin. **I did NOT chase it** because it is (a)
not quickly pinnable and (b) **fully mooted**: CoreML-FP32 already provides a d_seg-exact GPU SegNet path,
so the ExecuTorch SegNet blocker no longer gates a GPU-fast faithful scorer. ExecuTorch PoseNet (FP32-exact,
zero-port) remains a clean GO and is consumed unchanged from the spike.

---

## RECOMMENDATION — which backend per net for the GPU-fast 600-pair training loop

**Authority invariant (all options):** torch-CPU stays the ONLY authority — periodic torch-CPU re-score
(argmax-flip d_seg + first-6-dim d_pose) gates any promotion; no score/kill from a GPU backend.

### PoseNet → **ExecuTorch-MLX-delegate FP32 (primary)**, our-MLX-GPU (relative-signal fallback)
- ExecuTorch PoseNet is **FP32-EXACT** (rel_mse 4.97e-14), **zero-port**, GPU, and PoseNet was historically
  the worst MPS-drift offender — so a faithful GPU PoseNet is the higher-value win. CoreML-FP32 PoseNet is
  an equally-exact alternative if ExecuTorch integration friction is higher.
- Our MLX-GPU PoseNet (drift 2.76e-4) is fine as the *relative* per-step training gradient signal, but its
  drift can exceed the d_pose signal itself near the PR106 frontier (d_pose≈3.4e-5) → never trust its
  *absolute* d_pose without the torch-CPU gate. Use it only if a single MLX-native VJP path is wanted for
  both nets and ExecuTorch/CoreML gradient wiring is deferred.

### SegNet → **our already-built MLX-GPU SegNet (primary)**, CoreML-FP32 (best-fidelity gate alt)
- Our MLX SegNet is **already built, MLX-native (so `mx.vjp` gradients are available for the training
  loss), bit-faithful on CPU, and 1.24e-5 flip-rate on GPU** (243/19.66M, all boundary near-ties =
  negligible for training/atlas). It is the lowest-integration-effort GPU SegNet for the loss because the
  gradient path is already MLX.
- **CoreML-FP32 SegNet is the higher-fidelity option (0 flips)** but is forward-only / per-call-latency
  heavy and has no easy MLX gradient path → best used as a **cheap near-authority cross-check** between
  torch-CPU gates, or if a forward-only sensitivity/atlas pass wants the most faithful GPU SegNet.
- mlx-image B2 is **not recommended** as a SegNet path (wrong B2 variant; would require a hand-port).

### The recommended pairing for the GPU-fast 600-pair loop
> **PoseNet = ExecuTorch-MLX FP32 (or CoreML-FP32) — FP32-exact, GPU, zero-port.
> SegNet = our MLX-GPU port (MLX-native gradients, 1.24e-5 flip-rate, already built).
> torch-CPU = the periodic authority gate (every N epochs / pre-promotion) for BOTH nets.**

Reasoning: this pairing maximizes (fidelity × speed × integration-effort) jointly. PoseNet's higher
near-frontier drift sensitivity is solved exactly by the zero-port ExecuTorch/CoreML FP32 path; SegNet's
training-loss gradient is solved by the already-built MLX-native port whose GPU drift is negligible
(boundary near-ties only); and the one non-negotiable — torch-CPU as the absolute authority — is preserved
as the periodic gate. If a single-backend MLX-native VJP for both nets is preferred for simplicity, fall
back to our MLX-GPU for both with a tightened torch-CPU pose gate; if maximum SegNet fidelity is needed for
an absolute (not relative) readout, swap in CoreML-FP32 SegNet for that pass.

---

## Reproduce / artifacts (all rebuildable; throwaway venvs recorded for cleanup)

New measurement scripts (kept in `.omx/tmp/`, small):
- `.omx/tmp/scorer_backend_benchmark_torch_ref.py N` — torch-CPU authority (SegNet encoder + full forward).
- `.omx/tmp/mlximage_b2_fidelity.py N` — mlx-image B2 logits + GPU speed (throwaway venv).
- `.omx/tmp/torchvision_b2_compare.py` — torchvision-CPU vs mlx-image-GPU fidelity (main venv).
- `.omx/tmp/coreml_fp32_posenet.py N` / `.omx/tmp/coreml_fp32_segnet.py N` — CoreML-FP32 drift (throwaway venv).

Throwaway venvs (certified rebuildable per the disk-hygiene rule — **DELETE after review**):
- `.venv_mlximage_spike` (~) — rebuild: `VIRTUAL_ENV=.venv_mlximage_spike uv venv .venv_mlximage_spike --python 3.13 && VIRTUAL_ENV=.venv_mlximage_spike uv pip install mlx-image`.
- `.venv_coreml_spike` (~) — rebuild: `VIRTUAL_ENV=.venv_coreml_spike uv venv .venv_coreml_spike --python 3.13 && VIRTUAL_ENV=.venv_coreml_spike uv pip install coremltools "torch==2.11.0" timm einops segmentation-models-pytorch safetensors numpy`.
- (ExecuTorch venv was already deleted by the prior spike; not recreated this session.)

Versions recorded: mlx-image 0.1.10 (`mlxim`); mlx-vision/efficientnet_b2-mlxim weights; coremltools 9.0 +
torch 2.11.0 (throwaway); main venv mlx 0.31.1 + torchvision 0.26.0 + torch (CUDA unavailable → torch-CPU
authority axis). Machine: M5 Max, macOS, arm64.

## NO-FAKE / caveats
- Every non-torch-CPU number is `[research-signal]`, non-promotable; no score/promotion/kill claim.
- mlx-image is the *torchvision* B2 — its 0.99999999 cosine is vs torchvision-CPU (its matching arch), NOT
  vs the contest timm-arch SegNet encoder; it is explicitly NOT a measured SegNet drop-in (the channel-taper
  mismatch is the verdict, not a fidelity number).
- CoreML/ExecuTorch per-call throughput rows are call-latency-dominated, not throughput ceilings; the
  fidelity numbers are the load-bearing results. CoreML FP32 fidelity was measured on 10–20 frames at
  per-sample predict; re-confirm under batching + larger sample before wire-in.
- torch-CPU speeds measured contended with the capstone daemon → conservative lower bounds.
- Sample = 10–30 of 600 real pairs; drift is stationary across windows per the consumed drift audit.
