# FIX: ExecuTorch MLX-delegate SegNet runs on the Apple GPU — FP32-EXACT (2026-06-11)

**Operator ask (2026-06-11, "try out the executorch segnet with the fix"):** the prior spike
(`executorch_mlx_delegate_scorer_spike_20260611.md`) found the ExecuTorch MLX-delegate SegNet
(smp.Unet/efficientnet-b2) EXPORTS + partitions to the GPU fine but its GPU RUN failed with
`steel_gemm_fused_nax.h:23:20: error: unknown type name 'GEMMParams'`. The PLAIN-GEMM path worked
(minimal Conv2d/Linear/raw matmul ran on GPU), so the directive was: force ExecuTorch's MLX backend
onto the plain (non-steel-fused) GEMM path, and if SegNet then runs, measure its d_seg fidelity.

**RESULT: FIXED with a one-line env var.** `MLX_METAL_GPU_ARCH=applegpu_g15` (any non-NAX arch) makes
SegNet run on the Apple GPU and it is **d_seg-EXACT vs torch-CPU** (0 argmax flips / 1,572,864 pixels
across 8 real 0.mkv frames; logit max-abs-err 4.3e-5 = float32 ULP). The SAME override keeps PoseNet
FP32-exact (rel_mse 0.0, max-abs-err 7.6e-6). **ExecuTorch now gives BOTH nets zero-port FP32-exact on
the Apple GPU.**

**Authority discipline (CLAUDE.md):** torch-CPU is the ONLY authority. Everything below the torch-CPU
column is `[macOS-MLX/ExecuTorch research-signal]` and **non-promotable**. Real 0.mkv frames from the
byte-identical reference scorer-input cache (`mlx_scorer_input_cache_reference_video_20260521T2304Z_full600`,
`segnet_last_rgb` = last-frame resized 384×512, the exact SegNet conv-body input). MPS never touched.
The exact frontier pointer did NOT move; this is an enabler.

---

## Root cause — it is a NAX-tile GEMM JIT bug, gated on GPU arch (NOT a vendored-kernel skew)

The prior benchmark memo (`scorer_backend_benchmark_ours_vs_others_20260611.md` §4) concluded the bug
lived in ExecuTorch's *vendored* MLX kernel sources and was "NOT pip-pinnable." **This fix refines that
diagnosis: the kernels are NOT vendored — they come from the installed `mlx` / `mlx-metal`** (the
`backends/mlx/third-party/mlx` tree ships only python/docs/examples, no kernels and no compiled lib).
The ExecuTorch MLX C++ runtime JIT-compiles Metal kernels from `mlx/include/.../steel/gemm/kernels/`.

- M5 Max reports GPU architecture **`applegpu_g17s`** (`mx.device_info()['architecture']`).
- On g17 (NAX-capable: Apple's `MetalPerformancePrimitives` / NAX-tile hardware), MLX 0.31.2 routes
  fused GEMM (the conv+bias/addmm epilogue that EfficientNet-B2's many 1×1 pointwise convs hit) to
  **`steel_gemm_fused_nax.h`**, whose JIT assembly is **missing the `GEMMParams` include** in this
  mlx-metal build → `unknown type name 'GEMMParams'` → `method->execute() failed with error 0x1`.
- On any **non-NAX arch** (g14/g15/g16), MLX routes the same op to the plain **`steel_gemm_fused.h`**,
  which JIT-compiles cleanly. `MLX_METAL_GPU_ARCH` (a documented MLX env var, present in libmlx's string
  table alongside `MLX_DISABLE_COMPILE`) overrides the detected arch used for kernel selection.

So the "fix" the directive asked for — **force the plain (non-steel-fused-NAX) GEMM path** — is exactly
`MLX_METAL_GPU_ARCH=<non-NAX arch>`. Fix #1 (cheapest in the directive's list: an env flag that disables
the fused/steel path) is the winner; no version-combo pinning or kernel patch was needed.

Note this is consistent with raw `mlx.core` matmul/addmm/conv2d all working at default arch (they take a
different, non-fused-NAX dispatch); only the ExecuTorch delegate's fused conv-GEMM hits the broken
`_nax` epilogue kernel.

---

## Measurements (real 0.mkv frames; torch-CPU = authority)

### SegNet — runs on GPU, d_seg-EXACT across 8 real frames

| arch override | mlx detected | GPU run | argmax flips | flip rate | logit max-abs-err | verdict |
|---|---|---|---|---|---|---|
| **(none) = g17s** | applegpu_g17s | ❌ `steel_gemm_fused_nax` JIT | — | — | — | crash reproduced |
| **applegpu_g15** | applegpu_g15 | ✅ | **0 / 1,572,864** | **0.0** | **4.3e-5** | **d_seg-EXACT** |
| applegpu_g16 | applegpu_g16 | ✅ | 0 (frame 0) | 0.0 | 3.4e-5 | d_seg-EXACT |
| applegpu_g14 | applegpu_g14 | ✅ | 0 (frame 0) | 0.0 | 3.4e-5 | d_seg-EXACT |

8-frame run @ g15: **0 flips total / 1,572,864 px**, per-frame logit-err 2.6e-5–4.3e-5, ~11.9 frames/s
(single-frame per-call; not a batched ceiling). The 3.4e-5 logit-err is **arch-invariant** across
g14/g15/g16 → the non-NAX steel GEMM is numerically equivalent to torch-CPU at FP32 (the tiny err is
float32 ULP, identical to the spike's PoseNet 7.6e-6).

### PoseNet — same override, still FP32-exact

`MLX_METAL_GPU_ARCH=applegpu_g15`, input (12,384,512) (matches the prior spike): **rel_mse 0.0,
max-abs-err 7.6e-6, GPU run 0.20s.** Identical to the spike's default-arch PoseNet result → the override
does not regress PoseNet. (A BatchNorm1d node still falls back to CPU in the mixed partition, exactly as
the spike documented — log noise, run succeeds.)

---

## VERDICT

**ExecuTorch now gives BOTH nets zero-port FP32-exact on the Apple GPU**, under a single env var. The
prior spike's "SegNet conditional NO-GO" flips to **GO**:

| Net | export | GPU partition | GPU run (with fix) | FP32 fidelity vs torch-CPU | verdict |
|---|---|---|---|---|---|
| PoseNet | ✅ | 9 partitions | ✅ 0.20s | rel_mse 0.0, max-err 7.6e-6 | **GO** (was already GO) |
| **SegNet** | ✅ | 6 partitions | ✅ (`MLX_METAL_GPU_ARCH=applegpu_g15`) | **0 flips, logit-err 4.3e-5** | **GO** (newly fixed) |

**The fix:** export+partition unchanged; set `MLX_METAL_GPU_ARCH=applegpu_g15` (or any g14/g15/g16) in the
process env before `import mlx` so the ExecuTorch MLX runtime selects the plain `steel_gemm_fused` kernel
instead of the broken `steel_gemm_fused_nax`. The env var must be set BEFORE the MLX runtime initializes.

**CoreML-FP32 SegNet (0 flips, already measured) is no longer the only d_seg-exact GPU SegNet path** — it
remains a valid alternative, but ExecuTorch is now a fully-faithful zero-port GPU path for the whole
scorer. Trade-off to note for any future wire-in: the g15 override forces the WHOLE process onto the
non-NAX kernel set, so it would also affect any other MLX work in that process (e.g. our MLX SegNet/atlas
training) — the scorer eval should run in its own process, or the override scoped to it, if the rest of the
pipeline wants the native g17/NAX kernels.

## NO-FAKE / caveats
- Every number here is `[macOS-MLX/ExecuTorch research-signal]`, non-promotable; no score/promotion/kill
  claim. torch-CPU is the sole authority; SegNet GPU output was compared argmax-for-argmax against the
  torch-CPU reference logits on the same real frames.
- Sample = 8 of 600 real pairs for SegNet; drift is stationary across windows per the consumed drift audit,
  and 0 flips at FP32 across 8 frames + arch-invariant logit-err is a strong signal. Re-confirm on the full
  600 + under batching before any wire-in.
- The fix is an arch-detection override, not a numerics change — confirmed by the arch-invariant 3.4e-5
  logit-err (g14=g15=g16). It does not alter what the scorer computes; it only changes which Metal GEMM
  kernel JIT-compiles.

## Environment (recorded for reproducibility + cleanup)
- Machine: Apple M5 Max, macOS 26.4, arm64. GPU `applegpu_g17s` (NAX-capable).
- Throwaway venv: **`.venv_executorch_segnet_fix`** (Python 3.13.12, ~1.0 GB) — **DELETED after this report**
  per the disk-hygiene rule (certified rebuildable).
- Rebuild: `VIRTUAL_ENV=.venv_executorch_segnet_fix uv venv .venv_executorch_segnet_fix --python 3.13 && VIRTUAL_ENV=.venv_executorch_segnet_fix uv pip install executorch segmentation-models-pytorch timm einops safetensors numpy "mlx==0.31.2"`
- Resolved: executorch 1.3.1, mlx 0.31.2 + mlx-metal 0.31.2, torch 2.12.0, timm 1.0.27,
  segmentation-models-pytorch 0.5.0.
- Scripts (kept in `.omx/tmp/`, small): `segnet_torch_cpu_ref.py` (torch-CPU SegNet reference, run in main
  `.venv`), `segnet_executorch_gpu_fix.py` (single-frame GPU run + crash repro), `segnet_executorch_gpu_allframes.py`
  (8-frame d_seg fidelity), `posenet_executorch_gpu_archcheck.py` (PoseNet under override).
- Run: e.g. `MLX_METAL_GPU_ARCH=applegpu_g15 .venv_executorch_segnet_fix/bin/python .omx/tmp/segnet_executorch_gpu_allframes.py`
- Bulk `.pte` blobs (~240 MB) + reference `.npy` (~50 MB) + venv (~1.0 GB): **deleted after the report
  landed** — all rebuildable from the committed scripts + the one-line uv install + the committed reference
  input cache. Result JSONs (small) kept under `.omx/tmp/segnet_fix/`.
