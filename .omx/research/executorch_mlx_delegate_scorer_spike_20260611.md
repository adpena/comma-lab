# SPIKE: ExecuTorch MLX delegate vs the exact contest scorer — GO/NO-GO (2026-06-11)

**Operator ask (2026-06-11, "so we don't do a lot of work for nothing"):** test whether the
PyTorch-official **ExecuTorch MLX delegate** can run our EXACT contest scorer (SegNet +
PoseNet) on the Apple GPU at FP32 fidelity — the ZERO-PORT path that, if it works, moots
hand-porting the scorer to MLX. Bounded feasibility spike, not a production integration.

**Authority discipline (CLAUDE.md):** torch-CPU is the ONLY authority. Everything below the
torch-CPU column is `[macOS-MLX/ExecuTorch research-signal]` and **non-promotable**. Real
0.mkv frames decoded via `frame_utils.yuv420_to_rgb` (never PyAV rgb24). MPS never touched.

Sibling subagents `a458be6fa8effdfa5` (existing-MLX audit) and `a642408922c41f2f3`
(existing-MLX-vs-torch drift) are DISJOINT; this spike tested only the OFFICIAL ExecuTorch path.

---

## Environment (recorded for reproducibility + cleanup)

- Machine: Apple Silicon `arm64`, macOS 26.4 (build 25E246), M5 Max.
- Throwaway venv: **`.venv_executorch_spike`** (Python 3.13.12). **~1.0 GB — DELETE after review**
  (rebuildable via the install command below; certified rebuildable per the disk-hygiene rule).
- Install: `VIRTUAL_ENV=.venv_executorch_spike uv pip install executorch` (single command).
- Resolved versions: **executorch 1.3.1**, **mlx 0.31.2 + mlx-metal 0.31.2**, torch 2.12.0,
  timm 1.0.27, segmentation-models-pytorch 0.5.0, coremltools 9.0. (pyav NOT installed in the
  throwaway env — real frames were decoded in the trusted main `.venv` and the preprocessed
  input tensors + torch-CPU reference outputs were handed to the spike venv.)
- ExecuTorch MLX backend present at `executorch/backends/mlx/` with `MLXPartitioner` + `MLXBackend`.
  Example models shipped are ALL transformers (llm / voxtral / whisper) — consistent with the
  documented transformer focus, but the op registry is broader (98 registered ATen overloads
  including conv1/2/3d, conv_transpose, batch_norm, gelu, silu, sigmoid, relu, linear, addmm).

Spike script (committed): `experiments/executorch_mlx_delegate_scorer_spike.py`.

---

## Step 1 — Install: **GO.** ExecuTorch + the MLX delegate install cleanly on this machine
(arm64, macOS 26.4, Py3.13) in one `uv pip install`. Not a NO-GO.

## Step 2 — Export through ExecuTorch: **GO for BOTH nets.**
The hard part the survey warned about ("cannot run arbitrary nn.Modules") is **NOT a blocker**:
- `torch.export.export(...)` succeeds on **both** SegNet (smp.Unet/efficientnet-b2) and PoseNet
  (timm fastvit_t12 + Hydra head). No tracing failures.
- `to_edge_transform_and_lower(..., partitioner=[MLXPartitioner()])` partitions BOTH into GPU
  delegate subgraphs with a small CPU fallback set:
  - **SegNet** → **6 GPU delegate partitions**; CPU fallback = only `aten.upsample_nearest2d`
    ×5 (the U-Net decoder's `interpolate(mode='nearest')`). All convolutions delegate to MLX.
  - **PoseNet** → **9 GPU delegate partitions**; CPU fallback = only `_native_batch_norm` ×8
    (the 2D BatchNorm1d in the Hydra `AllNorm` head; the 4D batch-norms delegate fine). All
    convolutions delegate to MLX.
- **Diagnostic caveat (op-form, not a real gap):** the *naive* `to_edge` path mis-reports
  `aten.convolution.default` as "unsupported" because the default decomposition rewrites
  `conv2d` → `convolution.default(transposed=False)`, and the MLX `convolution.default` handler
  is coded to accept only `transposed=True` (`ops.py:2589`, "use aten.conv{1,2,3}d instead").
  The **partitioner path** (`ops_to_not_decompose`) preserves `conv2d.default`, which the MLX
  backend DOES support — so convolution is delegatable. Anyone re-measuring must use the
  partitioner path, not bare `to_edge`, or they'll get a false NO-GO on conv.

## Step 3 — Run on Apple GPU + fidelity vs torch-CPU

### PoseNet — **runs on GPU, FP32 fidelity is essentially exact.** ✅
On REAL 0.mkv pair-0 (the contest path), Apple GPU, FP32:
| metric | value |
|---|---|
| `pose_rel_mse` (vs torch-CPU) | **4.97e-14** (≈ 0) |
| `pose_max_abs_err` | **7.6e-06** (float32 ULP; dominant dim 34.244167 vs 34.244168) |
| GPU run | 0.138 s |
torch-CPU ref `[34.2442, -6.7e-5, 1.7e-3, -8.6e-4, -1.27e-2, -2.1e-4]` vs MLX-GPU
`[34.2442, -6.7e-5, 1.7e-3, -8.6e-4, -1.27e-2, -2.1e-4]` — identical to 5+ sig figs. d_pose
(the scored MSE on the first 6 dims) is preserved at FP32. (Also confirmed on random input:
rel_mse 0.0.)

### SegNet — **partitions correctly but FAILS at GPU execute** due to an MLX-Metal kernel bug. ❌
`method->execute() failed with error 0x1`, with the underlying Metal compiler error:
```
[metal::Device] Unable to build metal library from source
mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.h:23:20:
    error: unknown type name 'GEMMParams'
```
This is **NOT** a fidelity failure, an op-coverage gap, or an "arbitrary-module" limitation:
- Raw `mlx.core.matmul` and `mlx.core.conv2d` both run fine on the GPU in this exact env.
- A minimal single `Conv2d` AND a minimal `Linear` both run fine through the MLX delegate.
- The crash is the **fused-GEMM `steel` kernel JIT failing to compile** for the specific
  conv shapes EfficientNet-B2 routes through (likely grouped/depthwise or particular channel
  counts). It's a **version/build skew** between the ExecuTorch MLX backend's vendored MLX
  kernel sources (`backends/mlx/third-party/mlx`) and the installed `mlx-metal 0.31.2` JIT —
  the `GEMMParams` type the header references is missing from the compiled kernel set. A
  fixable toolchain bug, not an architecture incompatibility.

---

## GO/NO-GO verdict

| Net | export | GPU partition | GPU run | FP32 fidelity vs torch-CPU | verdict |
|---|---|---|---|---|---|
| **PoseNet** (fastvit_t12 + Hydra) | ✅ | 9 partitions (BN1d on CPU) | ✅ 0.14s | rel_mse 5e-14, max_err 7.6e-6 | **GO** (zero-port viable) |
| **SegNet** (smp.Unet/efficientnet_b2) | ✅ | 6 partitions (upsample on CPU) | ❌ steel_gemm JIT bug | unmeasured (didn't run) | **CONDITIONAL NO-GO** (fixable kernel bug, not arch) |

**Bottom line:** the zero-port ExecuTorch+MLX path is **HALF proven** in a few hours. PoseNet is
a clean GO — exact-FP32 on the Apple GPU with near-zero porting. SegNet is blocked ONLY by a
fused-GEMM Metal-kernel JIT bug in this mlx-metal build (the model exports + partitions fine and
all its convs are delegatable), so it is a **conditional NO-GO pending a kernel/version fix**, NOT
a fundamental incompatibility. This is materially better than the survey's pessimistic prior
("conv-heavy nets may not export") — both nets export and partition; only one hits a build bug.

## Recommended next steps (do NOT do as part of this spike)
1. **Chase the SegNet steel_gemm bug (cheapest, highest payoff):** try (a) a different mlx /
   mlx-metal version pin (the bug is a header/runtime skew — a nearby version may resolve it),
   (b) an ExecuTorch MLX backend env flag to disable the fused-GEMM/steel path and use the plain
   GEMM that minimal conv used successfully, (c) report upstream. If any works → SegNet flips to
   GO and the **whole scorer is zero-port + GPU + FP32-faithful**.
2. **If the kernel bug is not quickly fixable:** the Tier-2 fallback is now SMALLER than the
   survey assumed — PoseNet needs NO port at all; only SegNet would need the
   mlx-image-EfficientNet-B2 + ported-Unet-decoder route. Most likely just keep SegNet on
   torch-CPU and run only PoseNet on the GPU via ExecuTorch (PoseNet was historically the worse
   MPS-drift offender, so a faithful GPU PoseNet is the higher-value win anyway).
3. **Before any "scorer-on-GPU" promotion:** ship a real-input torch-CPU parity GATE (argmax-flip
   count for d_seg, rel-MSE for d_pose) — this spike's two metrics ARE that gate's first rows.

## Artifacts (rebuildable; cleaned up post-spike per the disk-hygiene rule)
- Result JSONs: `.omx/tmp/executorch_mlx_delegate_scorer_spike_result.json`,
  `.omx/tmp/spike_real_frame_fidelity.json` (kept — small).
- `.pte` blobs (~217 MB) + `.venv_executorch_spike` (~1.0 GB): **deleted after the report
  landed** — both fully rebuildable from the committed spike script + the one-line uv install.
