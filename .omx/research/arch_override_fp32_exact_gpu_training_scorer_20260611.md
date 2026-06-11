# Arch override makes the MLX-GPU training scorer FP32-EXACT — at zero throughput cost (2026-06-11)

**Operator ask (2026-06-11):** *"maybe there's a way on gpu to force the cpu stuff to use tools that are
faithful AND optimal"* + *"optimize compress-time too, all must be fully optimized."* The ExecuTorch
SegNet fix proved `MLX_METAL_GPU_ARCH=applegpu_g15` forces the M5's buggy NAX GEMM kernel to the clean
non-NAX path → FP32-exact (0 flips). The NAX reduction-order kernel is ALSO the root cause of OUR
MLX-GPU scorer bridge's drift (243 flips / 19.66M, pose 2.76e-4). **Hypothesis: the same arch override
eliminates that drift → the GPU training scorer becomes authority-grade.** Tested + measured below.

**RESULT: HYPOTHESIS CONFIRMED, decisively.** `MLX_METAL_GPU_ARCH=applegpu_g15` (or any g14/g15/g16,
non-NAX) makes our existing `MLXGpuScorerBridge` **FP32-exact** vs torch-CPU on the real 0.mkv cache:
**243 → 0 d_seg flips**, **pose 2.76e-4 → 8.7e-11** (a 7-orders-of-magnitude collapse to fp32 round-off),
and the gate verdict flips **FAIL → PASS**. The backward/VJP gradient (what training uses) tightens to the
**MLX-CPU bit-faithful reference** (cosine 0.99992, identical to MLX-CPU). Throughput is **unchanged** (within
1–4% noise at bs=4/8/16). The decoder is **not hurt** by the process-wide override (marginally faster,
fp32-ULP-identical output) → **process isolation is NOT required**. This is a ~zero-cost upgrade to the
wire-in that makes the GPU training scorer authority-grade for d_seg + (forward) d_pose.

**Authority discipline (CLAUDE.md NO-FAKE class 8 + MLX authority ladder):** torch-CPU exact is the ONLY
authority. Every number here is `[macOS-MLX research-signal]`, **non-promotable** — no score/promotion/kill
claim. Real 0.mkv frames from the byte-identical reference scorer-input cache
(`mlx_scorer_input_cache_reference_video_20260521T2304Z_full600`) + real upstream SegNet/PoseNet + real
trained-init render. No MPS touched. The exact frontier pointer did NOT move; this is an enabler.

**Daemons:** the running capstone daemon (pid 72123, 48-pair run) + atlas workers + the capstone-spec
subagent's files were NOT touched. The bridge code (`mlx_gpu_score_bridge.py`, commit 57d3a83ff) was NOT
edited. This is a measurement + an integration *proposal* (an env-var addition), per the directive.

---

## Why the override touches OUR bridge (not just ExecuTorch)

The drift audit (`mlx_scorer_port_drift_audit_20260611.md` §2b) localized our bridge's GPU drift to
**Metal-vs-CPU fp32 reduction-order non-associativity** in conv/GEMM/pool accumulation — the SE
global-avg-pool was the first cliff (3.97e-3), amplifying through the EfficientNet-B2 pointwise GEMMs to
9.6e-2 logit drift at the head. **That accumulation order is exactly what the NAX-tile GEMM kernel
changes.** On the M5's `applegpu_g17s` (NAX-capable), MLX routes the scorer's many 1×1 pointwise convs +
the FastViT GEMMs through the NAX-tile `steel_gemm_fused_nax` reduction order, which differs from
torch-CPU's. On any non-NAX arch (g14/g15/g16), MLX routes the same ops through the plain
`steel_gemm_fused` kernel, whose reduction order matches torch-CPU at fp32. The override is a
**kernel-selection** change, not a numerics change — confirmed by arch-invariance below.

`MLX_METAL_GPU_ARCH=applegpu_g15` overrides the arch MLX uses for kernel selection; it must be set in the
process env BEFORE `import mlx`. (A trivial `mx.matmul` at small size shows identical sums on both arches —
the NAX path only diverges through the *deep accumulated* scorer, which is why this needed the real-net test.)

---

## Test 1 — FP32-exactness (forward AND backward), torch-CPU authority

### Forward (the contest-charged d_seg + d_pose), 100 real 0.mkv pairs = 19.66M argmax px

Canonical drift sweep (`build_mlx_scorer_torch_parity_sweep_manifest`, the framework that measures the
exact charged quantities). Reused the audit's `mlx_drift_run.py` verbatim, twice — env arch off vs on:

| Quantity | GPU default (g17s/NAX) | **GPU override (g15/non-NAX)** | MLX-CPU ref (bit-faithful) |
|---|---|---|---|
| d_seg argmax flips (total / 19.66M) | **243** | **0** | 2 |
| d_seg flip rate | 1.24e-5 | **0.0** | 1.0e-7 |
| SegNet logit abs max delta | 9.64e-2 | **1.02e-4** (fp32 ULP) | 5.7e-5 |
| PoseNet raw output abs max delta | 4.07e-2 | **2.29e-5** | 2.3e-5 |
| **pose_component_abs_max** (the charged term) | **2.76e-4** | **8.73e-11** | 8.7e-11 |
| parity gate verdict | **FAIL** | **PASS** | PASS |

The override makes the GPU forward **bit-faithful to torch-CPU**: 0 flips, and pose drift collapses
7 orders of magnitude (2.76e-4 → 8.7e-11) to *exactly* the MLX-CPU round-off floor. **The pose drift that
"can exceed the frontier d_pose ~3.4e-5" is GONE** — under the override, GPU pose is trustworthy in
absolute terms at the frontier (drift 8.7e-11 ≪ 3.4e-5).

### Backward / VJP (the training gradient — what the optimizer consumes), 8 real pairs

The training step uses the pixel cotangent `dL/d(render)` from `loss_and_pixel_grad`. Reused the bridge's
own NO-FAKE real-net setup (`_build_real_setup`: real scorer, real GT, real trained-init render):

| Quantity | GPU default (g17s) | **GPU override (g15)** | MLX-CPU ref |
|---|---|---|---|
| grad cosine vs torch-CPU | 0.99986 | **0.99992** | 0.99992 |
| grad rel-L2 error vs torch-CPU | 1.69e-2 | **1.25e-2** | 1.2487e-2 |
| grad abs-max delta | 4.05e-4 | **2.29e-4** | — |
| total-loss rel error | 5.23e-4 | **5.6e-6** | 5.7e-5 |
| seg-loss abs delta | 1.52e-3 | **9.3e-6** | 1.5e-4 |
| d_seg (gpu vs torch) | 0.0 delta | 0.0 delta | 0.0 delta |
| grad non-zero (NO-FAKE) | ✅ | ✅ | ✅ |

**The backward gradient under the override is indistinguishable from the bit-faithful MLX-CPU reference**
(cosine 0.99992 and rel-L2 1.2523e-2 vs MLX-CPU's 0.99992 / 1.2487e-2). The residual ~1e-4 cosine gap to
torch-CPU is the inherent MLX-VJP-vs-torch-autograd fp32 difference that EVEN bit-faithful MLX-CPU has — it
is NOT GPU reduction-order drift. The GPU-specific drift is fully eliminated in the backward too. (The
forward loss/d_seg the gradient is built from is fp32-exact, per Test 1 forward.)

**Verdict (Test 1):** under the override the MLX-GPU training scorer is **authority-grade for d_seg
(0 flips) and (forward) d_pose (8.7e-11 ≪ frontier signal)**; the training gradient matches the
bit-faithful CPU reference. A torch-CPU authority re-score is still the canonical gate per CLAUDE.md, but
the GPU↔torch-CPU gap that *necessitated* frequent re-scoring is now at the fp32 floor.

### Arch-invariance (proves it's kernel-selection, not a numerics accident)

g16 forward, 24 pairs: **0 flips, pose 2.18e-11, PASS** — identical to g15. The FP32-exactness is robust
across g14/g15/g16 (all non-NAX → plain `steel_gemm_fused`), exactly mirroring the ExecuTorch finding.

---

## Test 2 — Compress-time throughput under the override (bs=4/8/16)

Reused the real-net bundle; timed `loss_and_pixel_grad` (full fwd+bwd training step) + a fwd-only path.
**Uncontended** (daemon was idle between its own steps; numbers are a clean read):

| bs | fwd+bwd g17s (p/s) | **fwd+bwd g15 (p/s)** | fwd-only g17s | fwd-only g15 |
|---|---|---|---|---|
| 4 | 0.636 | **0.659** | 5.07 | 4.98 |
| 8 | 0.724 | **0.720** | 8.36 | 8.61 |
| 16 | 0.768 | **0.767** | 12.74 | 12.47 |

**Throughput is identical between arches (within 1–4% noise) at every batch size.** Two notes:
- The override is **FP32-exact AND not-slower** — the "more faithful AND not slower" win the operator
  hypothesized. The non-NAX `steel_gemm_fused` kernel is numerically faithful at no measured perf cost here.
- The wire-in memo's **bs=16 backward collapse (0.61x)** did NOT reproduce in this uncontended run (bs=16
  fwd+bwd was the *fastest* at 0.768 p/s). The cliff was a **daemon-contention / memory-pressure artifact**,
  not arch-related — and crucially **the override neither introduces nor worsens it.** (bs≤8 remains the
  safe operating point under contention; the override doesn't change that calculus.)

---

## Test 3 — Integration design: does the process-wide override hurt the DECODER?

The override is process-wide → it would force the decoder's MLX kernels onto non-NAX too. Measured the
capstone VQ-NeRV decoder forward (`CapstoneVqNervBundle(idx, pose)`, base_ch=20, 48 pairs):

| Quantity | decoder g17s | **decoder g15** |
|---|---|---|
| decoder fwd pairs/s | 412.7 | **428.4** (marginally faster) |
| render sum checksum (7.2e9 scale) | 7220606464 | 7220606976 (Δ 5e-7 rel = fp32 ULP) |
| render mean | 127.52050 | 127.52051 (identical to 6 digits) |

**The decoder is NOT hurt by the override** — marginally faster, fp32-ULP-identical output. The decoder is
small (base_ch=20) and does not hit the NAX-fused-GEMM path the heavy EfficientNet-B2 scorer does, so
forcing non-NAX has no decoder cost.

### Recommended integration: process-wide override, NO process isolation needed

Since the decoder is unaffected (Test 3) and the scorer becomes FP32-exact (Test 1) at zero throughput cost
(Test 2), the **cleanest integration is the simplest one:**

> **Set `MLX_METAL_GPU_ARCH=applegpu_g15` in the training-process env (before `import mlx`) whenever
> `--scorer-backend mlx_gpu` is selected.** No separate scorer process, no override scoping — the whole
> capstone training process runs non-NAX, the scorer is FP32-exact, the decoder is unaffected.

Concrete wire-in options (proposed, NOT applied — the bridge was not edited):
1. **CLI/launcher env (lowest-touch, recommended):** the 600-pair launch command and any `--scorer-backend
   mlx_gpu` invocation export `MLX_METAL_GPU_ARCH=applegpu_g15` before the python process starts. One line
   in the launch wrapper / the daemon launcher; zero code change. This is the cleanest and is what I
   recommend.
2. **Self-set in the trainer:** `CapstoneTrainConfig.scorer_backend == "mlx_gpu"` could `os.environ
   .setdefault("MLX_METAL_GPU_ARCH", "applegpu_g15")` at the *very top* of `run_capstone_campaign.py` BEFORE
   any `import mlx` (the import-order constraint is real — MLX reads the env at runtime init). Slightly more
   robust (can't forget the env), but must be guarded to run before the first MLX import.
3. **Process isolation (NOT needed):** the directive flagged this as a possible requirement; Test 3 shows
   it is unnecessary — the decoder tolerates non-NAX. Skip it (avoids IPC complexity).

The bridge already documents the trade-off (`mlx_gpu_score_bridge` docstring); the ExecuTorch memo §VERDICT
flagged "scorer eval should run in its own process … if the rest of the pipeline wants the native NAX
kernels." Test 3 resolves that open question for the **capstone** pipeline: the rest of the pipeline (the
decoder) does NOT benefit from NAX, so the whole-process override is safe and simplest.

---

## Test 4 — Other "force faithful + optimal" kernel knobs

Surveyed MLX 0.31.1's env knobs (`strings libmlx.dylib | grep ^MLX_`) + `mx.fast` / `mx.compile`:

| Knob | Effect on fidelity | Effect on speed | Recommendation |
|---|---|---|---|
| **`MLX_METAL_GPU_ARCH=applegpu_g15`** | **FP32-exact (the win)** | **neutral** (≤4% noise) | **ADOPT** for the scorer process |
| `MLX_ENABLE_TF32` | LOWER precision (TF32 < fp32) | faster matmul | **DO NOT SET** — confirmed unset by default (= fp32); enabling it would re-introduce drift |
| `MLX_DISABLE_COMPILE` | none (numerics same) | slower (disables graph fusion) | leave UNSET (keep compile) |
| `MLX_METAL_FAST_SYNCH` | none (sync only) | minor latency | not fidelity-relevant; ignore |
| `mx.compile` (graph fusion) | numerics-neutral here | the bridge already benefits from MLX's default lazy graph | already in effect; no action |
| Metal `setFastMathEnabled` | MLX JIT-compiles kernels with `__METAL_FAST_MATH__` hardcoded; **no runtime env to disable it** | — | not tunable from our side; the g15 path is already fp32-exact *with* fast-math on, so this is moot |
| `MLXReferenceConv2dAdapter` Kahan/fp64 accum (in-port) | tighter accum on SE pool + head | slower | **NOT NEEDED** — the g15 override already achieves 0 flips, so the Kahan/fp64 reference accumulator the port ships for forensics is unnecessary in production |

**Top recommendation beyond the arch override:** none needed for fidelity — the arch override alone reaches
fp32-exact. The one *anti*-recommendation is load-bearing: **never set `MLX_ENABLE_TF32`** (it would silently
re-introduce drift). For speed, the dominant cost remains the VJP backward through the frozen scorer on both
backends (the wire-in memo's finding); no kernel knob changes that — the real backward-speed levers are
batch amortization (bs≤8 sweet spot) and the int8/carrier choices, unchanged by this work.

---

## NO-FAKE accounting

- Every number is `[macOS-MLX research-signal]`, non-promotable. torch-CPU is the sole authority; the GPU
  forward was compared argmax-for-argmax + per-pose-component against torch-CPU on the same real frames.
- Forward sample = 100 of 600 real pairs (19.66M argmax px); backward = 8 real pairs; throughput = bs 4/8/16
  on 24 real pairs; decoder = 48 pairs. The forward 0-flip + arch-invariance (g15=g16) + the pose collapse to
  the MLX-CPU round-off floor are a strong joint signal. A full-600 + comma10k generalization slice is the
  obvious extension (not needed to reach the verdict).
- The fix is an arch-detection override, not a numerics change — proven by arch-invariance (g15 and g16 give
  identical 0-flip / ~2e-11 results). It does not alter what the scorer computes; it changes which Metal GEMM
  kernel JIT-compiles.
- The exact frontier pointer is UNMOVED. This is an enabler: it upgrades the GPU training scorer from
  "fast relative signal needing frequent torch-CPU re-scoring" to "fp32-exact, authority-grade d_seg + pose."
- Bridge code NOT edited; daemons NOT touched. Integration is a *proposed* one-line env addition.

## Reproduce + cleanup

Durable scripts + result JSONs committed at
`.omx/research/arch_override_fp32_exact_artifacts_20260611/` (40 KB; small, kept):
```
A=.omx/research/mlx_scorer_drift_audit_20260611_artifacts          # the canonical forward drift sweep
B=.omx/research/arch_override_fp32_exact_artifacts_20260611         # this work's scripts

# Forward FP32-exactness (off vs on):
OMP_NUM_THREADS=4 .venv/bin/python $A/mlx_drift_run.py gpu 100                    # g17s: 243 flips, FAIL
MLX_METAL_GPU_ARCH=applegpu_g15 OMP_NUM_THREADS=4 .venv/bin/python $A/mlx_drift_run.py gpu 100   # g15: 0 flips, PASS

# Backward/VJP gradient drift (off vs on):
OMP_NUM_THREADS=4 .venv/bin/python $B/arch_override_grad_drift.py 8
MLX_METAL_GPU_ARCH=applegpu_g15 OMP_NUM_THREADS=4 .venv/bin/python $B/arch_override_grad_drift.py 8

# Throughput (off vs on) + decoder-under-override:
[MLX_METAL_GPU_ARCH=applegpu_g15] OMP_NUM_THREADS=4 .venv/bin/python $B/arch_override_throughput.py
[MLX_METAL_GPU_ARCH=applegpu_g15] OMP_NUM_THREADS=4 .venv/bin/python $B/arch_override_decoder.py
```
No bulk artifacts produced (the scorer cache + GT-targets cache are pre-existing committed inputs). Scratch
under `.omx/tmp/arch_override_out/` (result JSONs) is duplicated into `$B` (durable) and is safe to delete.

## Environment
- Apple M5 Max, macOS 26.4, arm64. GPU `applegpu_g17s` (NAX-capable, the buggy reduction-order kernel).
- mlx 0.31.1, torch 2.11.0 (CUDA unavailable → torch-CPU is the authority axis). No MPS.
- Reference scorer cache: `experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600`.
- GT targets: `experiments/results/capstone_gt_targets_cache/gt_targets_n{8,24,48,100}.pt`.
