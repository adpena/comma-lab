# MLX vs PyTorch-MPS — critical-path engineering benchmark + MLX-superiority roadmap

`[macOS-MLX/MPS engineering benchmark]` — ADVISORY engineering speed/correctness
study. Benchmarking MPS *as a compute substrate* is legitimate here; it does NOT
make MPS a score authority. Contest score authority is unchanged: numpy-fp32
CPU / CUDA; MPS is gradient-only-never-score. A faster kernel ≠ a better score.

- **Date (UTC):** 2026-06-30T16:11Z
- **Host:** M5 Max, 128 GB unified, Metal GPU; macOS 26.4 (25E246), arm64
- **Versions:** Python 3.13.12, MLX 0.31.1, PyTorch 2.11.0 (`torch.backends.mps.is_available()=True`)
- **Harness:** `tools/bench_mlx_vs_torch_mps.py` (resumable, argparse, JSON out, parity-gated)
- **Operator authorization:** days/weeks campaign to push MLX SUPERIOR to torch-MPS on macOS (2026-06-30)

---

## 0. GPU-exclusivity status — GPU is OWNED → GPU sweep is PENDING

At session start a live witness training arm owns the single Metal GPU under the
one-GPU rule:

```
PID 98764  train_levelset_witness_realized_through_R_mlx.py ... --mlx-device gpu
           (safe_run label=levelset_thetastar_muon_arm; muon-start-epoch 726; ~1h43m elapsed)
PID 77003  witness_per_stage_annulus_attribution.py ... (CPU, 99% one core)
```

A GPU sweep alongside it would contaminate BOTH the benchmark and the training,
so **all GPU numbers below are PENDING-GPU.** The harness enforces this: `--gpu-sweep`
**refuses** (rc=3) while a `thetastar`/`train_levelset*` arm is alive (verified —
it correctly refused this session). Everything not needing the exclusive GPU is
done: harness built, correctness-parity established (CPU), literature scanned,
gap analysis + push plan drafted.

> NOTE: the first `ps | grep` of this session returned empty and nearly led me to
> run the GPU sweep — a second, thorough `ps -axww -o pid,command` revealed the
> live arm. The lesson is baked into the harness gate (it shells `ps` itself).

### Fire the GPU sweep when the GPU frees (one command)

```bash
# Only run when NO thetastar/train_levelset arm is alive (the harness re-checks and
# refuses otherwise). Full sweep, parity-gated, resumable, ~5-10 min:
.venv/bin/python tools/bench_mlx_vs_torch_mps.py --gpu-sweep \
  --out .omx/research/mlx_vs_torch_mps_bench_gpu_$(date -u +%Y%m%dT%H%M%SZ).json
```

It writes incrementally (resume on interrupt) and prints the GAP TABLE at the end.
Run with both `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` (default) and `=0` to quantify the
custom-Metal-backward lever on M5 GPU directly.

---

## 1. Methodology (NO-FAKE)

- **Parity FIRST, then speed.** For every op, MLX and torch run on the SAME seeded
  inputs; outputs are compared on numpy (`max |Δ|`, `max relrel`) BEFORE timing.
  A faster *wrong* kernel is gated out. Argmax (the d_seg path) requires exact
  integer match (tol=0).
- **Warmed timing.** ≥5 warmup, then N timed trials (default 25; `--quick`=5),
  **median + IQR** (not single shots). Compute is forced inside the timed region:
  MLX `mx.eval(out)` + `mx.synchronize()`; torch `torch.mps.synchronize()` — so we
  time compute, not lazy-graph construction.
- **Peak unified memory** per candidate (`mx.get_peak_memory` / `torch.mps.driver_allocated_memory`).
- **Seeded + deterministic** (`np.random.default_rng(seed)`); device/OS/lib versions recorded in JSON `meta`.
- **Identical weights** for the scorer ops: the SAME `upstream/models/{segnet,posenet}.safetensors`
  load both ways (`tac.scorer.load_default_scorers` for torch; `load_mlx_distortion_scorer_adapter_from_upstream`
  converts the same CPU torch load to MLX). Authority-grade parity (MLX vs CPU-torch,
  argmax pixel-identical, posenet component Δ ≤ 2e-5) is separately established in-repo
  via `build_mlx_scorer_torch_parity_manifest` (thresholds: segnet logit 1e-2, segnet
  argmax 0 px, posenet output 2e-3, posenet component 2e-5). The bench's same-device
  gate re-verifies MLX≈torch on the bench device.

### Ops covered (13)

| category | ops |
|---|---|
| primitive (landscape) | matmul, conv2d (std), depthwise_fwd, groupnorm, layernorm, **argmax (d_seg path)**, gather, elementwise (relu+silu) |
| critical_path | **grouped_conv_backward** (mlx_custom Metal vs mlx_ref python-loop vs torch), **inr_trunk** (Linear/FiLM/softmax at P=196608), **render_R** (bicubic↑874 → uint8 → bilinear↓384), **segnet_fwd** (NHWC 384×512→5cls), **posenet_fwd** (NHWC 192×256×12→pose) |

Critical-path shapes are the REAL witness shapes: render 384×512 (P_px=196608),
hidden 96, 4 hidden layers, 5 classes; SegNet input (N,3,384,512); PoseNet YUV6
(N,12,192,256); the broken-native strided depthwise backward (groups=C, stride 2).

---

## 2. MEASURED — CPU dry-run (parity validation; advisory, NOT the GPU comparison)

These run MLX-CPU vs torch-CPU (`--device cpu`, non-contending with the GPU arm).
They **validate the harness and establish correctness parity**; they are NOT the
MLX-GPU-vs-torch-MPS comparison the campaign targets (that is §0 PENDING). The CPU
*ratios* are nonetheless an informative first read of MLX's lazy-eval/overhead
profile. Artifact: `.omx/research/mlx_vs_torch_mps_bench_cpu_dryrun_20260630T161107Z.json`.

| op | size | parity max\|Δ\| | gate | mlx_ms (CPU) | torch_ms (CPU) | ratio = torch/mlx |
|---|---|---|---|---:|---:|---:|
| matmul | 128³ | 0.0 | ok | 0.024 | 0.005 | 0.23 (torch) |
| matmul | 512³ | 0.0 | ok | 0.203 | 0.195 | 0.96 |
| matmul | 2048³ | 0.0 | ok | 11.38 | 12.03 | 1.06 |
| matmul | 4096×4096×1024 | 0.0 | ok | 34.6 | 28.6 | 0.83 |
| conv2d std | N1 C3→32 384×512 s2 | parity ok | ok | — | — | (see JSON) |
| depthwise_fwd | N4 C96 48×64 s2 | 2.4e-7 | ok | 3.40 | 8.44 | **2.48 (MLX)** |
| depthwise_fwd | N4 C64 48×64 s1 | 4.8e-7 | ok | 4.94 | 0.22 | 0.04 (torch) |
| argmax (d_seg) | N4 384×512 C5 | 0 (exact) | ok | 3.45 | 1.91 | 0.55 (torch) |
| **inr_trunk** | P196608 in88 h96 | 3.0e-5 | ok | 74.9 | 65.2 | 0.87 |
| inr_trunk | P49152 | 3.0e-5 | ok | 11.1 | 20.6 | **1.85 (MLX)** |
| **render_R** | N1 384→874→384 | 0.93 LSB | ok* | 32.5 | 3.7 | 0.11 (torch) |
| render_R | N4 | 0.94 LSB | ok* | 216 | 15.1 | 0.07 (torch) |
| **segnet_fwd** | N1 384×512 | 1.2e-4 | ok | 764 | 1372 | **1.80 (MLX)** |
| segnet_fwd | N4 | 1.1e-4 | ok | 2181 | 2560 | **1.17 (MLX)** |
| **posenet_fwd** | N1 192×256×12 | 7.0e-6 | ok | 432 | 111 | 0.26 (torch) |
| posenet_fwd | N4 | 1.5e-5 | ok | 718 | 1054 | **1.47 (MLX)** |
| grouped_conv_backward | mlx_ref vs torch (CPU) | 1e-4..3e-4 | ok | 312–1879 (ref) | 10–33 (torch) | (custom Metal is GPU-only → §3) |

\* `render_R` parity is informational: MLX's contest-faithful bicubic uses a=-0.5
vs torch a=-0.75, so a sub-1-LSB average delta on the up-step is EXPECTED and
correct (the MLX path is the witness's production R; this is a known algorithmic
difference, not a kernel bug).

**CPU parity verdict: PASS across all ops** (max non-algorithmic Δ = 1.2e-4 on
SegNet logits, far inside the 1e-2 threshold; argmax exact; INR/PoseNet ≤ 3e-5).
The harness is trustworthy; the speed comparison just needs the GPU.

**CPU ratio reading (the lazy-eval signature, will re-shape on GPU):** MLX shows
a clear **per-call overhead that amortizes with batch** — PoseNet 0.26× at N=1 →
1.47× at N=4; inr_trunk 0.87× at P=196k but 1.85× at P=49k (smaller = fewer big
matmuls, overhead-relative); depthwise wins or loses by stride/shape. This is the
same overhead story the literature reports (small-op overhead, big-op wins).

---

## 3. The headline lever (MEASURED on GPU previously, in-repo): custom Metal grouped backward

The single biggest critical-path fact, already measured on M-series GPU and now
DEFAULT-ON (`TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`), per
`src/tac/local_acceleration/metal_grouped_conv_backward.py` +
`.omx/research/mlx_custom_grouped_backward_kernel_makes_mlx_gpu_fast_20260612.md`:

- MLX **native** strided grouped/depthwise Conv2d **backward is numerically WRONG**
  (grad cosine ~0.025, magnitude 5–25× too large). Forward `mx.conv2d` is bit-exact + fast.
- The repo's custom `@mx.custom_function` (two `mx.fast.metal_kernel` programs for
  grad_input + grad_weight) is **correct AND fast**: full-scorer (SegNet+PoseNet)
  backward **~18× faster** (11,149 ms → 621 ms at B=4); SegNet backward **12.9×
  (B=4) → 35.5× (B=8)**. The **backward is >97% of the training step**, so this is
  THE throughput lever that makes MLX-GPU the fast local backward backend.

The CPU dry-run (§2) shows *why this exists*: the correct python-loop reference
backward costs 312–1879 ms — unusable per step. The custom Metal kernel replaces it.
**This lever is the reason MLX-GPU is already viable for our training; the §5 push
plan extends it.** The PENDING GPU sweep will re-confirm the 18× on M5 directly
(run with `=1` vs `=0`).

---

## 4. Literature / OSS landscape (cited)

Concrete per-op MLX-GPU-vs-torch-MPS data (highly chip-dependent — M5 Max must be
measured, §0):

- **Per-op ratios** (`TristanBilot/mlx-benchmark`, averaged across M1–M5; +% = MLX
  faster): Conv2d **+7669% (M1 Pro) … −76% (M1)** (enormous variance by chip);
  MatMul +19%…+255%; Softmax +341%…−36%; Sort +1039%…+8534% (MLX dominant);
  **Argmax +503%…−59%**; Scatter +5002%…−55%; Gather +504%…−4%; activations
  (ReLU/SeLU/PReLU) often **slower** on MLX. Takeaway: MLX wins compute-heavy
  (conv/sort/matmul-large) on most chips, loses on small elementwise/activation and
  is wildly chip-dependent on conv.
- **Small matmul overhead:** PyTorch-MPS **5.5× faster** than MLX at 128×128 (10k
  iters: 0.21 s vs 1.15 s, M3 Pro) — overhead-bound, no compute explanation; matches
  our CPU 128³ read (0.23×) (Kevin Martin Jose).
- **MLX conv pain points (open issues):** `conv_general` (3D) **10–150× slower**
  than MPS (#1409, 22× on a (8,16,128,128,32) case); `conv1d`/`conv_transpose1d`
  **2.9–4.5× slower** than MPS (#2180); 1D group conv **2× faster as multiply+sum**
  than native (#2369); **`mx.fast.metal_kernel` called >10k× is 2.5× slower** due to
  lazy-eval + numpy-recovery dispatch overhead (#1828) — directly relevant to our
  per-step custom kernel; `metal_kernel` can give **wrong results on M1 Max but
  correct on M3 Max** (#2205) — a cross-chip correctness caveat for custom kernels.
- **Where MLX structurally leads:** unified memory (zero host↔device copies — the
  big real-workload win), lazy eval + graph fusion (`mx.compile`), strong large-matmul
  + LLM decode (2–3× over MPS), native quantization. (TDS "How Fast Is MLX"; MetalCloud;
  Apple ML Research M5 neural-accelerator post.)

Sources:
- [TristanBilot/mlx-benchmark](https://github.com/TristanBilot/mlx-benchmark)
- [Kevin Martin Jose — matmul MPS faster than MLX](https://kevinmartinjose.com/2025/04/21/matmul-using-pytorchs-mps-backend-is-faster-than-apples-mlx/)
- [ml-explore/mlx #1409 conv_general slow](https://github.com/ml-explore/mlx/issues/1409)
- [ml-explore/mlx #2180 conv1d slower than MPS](https://github.com/ml-explore/mlx/issues/2180)
- [ml-explore/mlx #2369 optimize 1D group conv](https://github.com/ml-explore/mlx/issues/2369)
- [ml-explore/mlx #1828 metal_kernel repeated-call overhead](https://github.com/ml-explore/mlx/issues/1828)
- [ml-explore/mlx #2205 metal_kernel wrong on M1 Max](https://github.com/ml-explore/mlx/issues/2205)
- [How Fast Is MLX? (Towards Data Science)](https://towardsdatascience.com/how-fast-is-mlx-a-comprehensive-benchmark-on-8-apple-silicon-chips-and-4-cuda-gpus-378a0ae356a0/)
- [MLX vs PyTorch (MetalCloud)](https://metalcloud.space/blog/mlx-vs-pytorch-comparison/)
- [richiksc/mlx-benchmarks](https://github.com/richiksc/mlx-benchmarks)

---

## 5. GAP TABLE — MLX vs torch-MPS per op (status-tagged)

`weight` = our critical-path importance. `status`: GPU-MEASURED (in-repo) /
LIT (literature-expected, confirm on GPU) / CPU-DRYRUN (this session) / PENDING.

| op | weight | MLX vs torch-MPS (expected) | status | basis |
|---|---|---|---|---|
| **strided grouped/depthwise conv BACKWARD** | ★★★★★ (>97% of step) | MLX **WINS big** with custom Metal kernel (~18× full-scorer); native MLX is WRONG → must use custom | GPU-MEASURED | repo memo 2026-06-12 |
| SegNet forward (conv-heavy) | ★★★★ | likely MLX ≥ torch (native conv fwd bit-exact; CPU 1.2–1.8×) but conv2d is chip-variant | CPU-DRYRUN + LIT | §2, TristanBilot |
| PoseNet forward (FastViT) | ★★★ | overhead-bound at N=1 (torch wins), MLX wins at batch | CPU-DRYRUN | §2 (0.26×→1.47×) |
| **R operator (bicubic↑/uint8/bilinear↓)** | ★★★★ (2×/pair) | **torch likely wins** — MLX separable resize is slow (CPU 0.07–0.11×); prime custom-kernel target | CPU-DRYRUN + LIT | §2, #1409 |
| INR trunk (Linear/FiLM/softmax) | ★★★ | near-parity, MLX competitive at scale; fuseable | CPU-DRYRUN | §2 (0.87–1.85×) |
| matmul (large) | ★★ | parity-to-MLX-win | CPU-DRYRUN + LIT | §2, TristanBilot |
| matmul (small) | ★ | torch wins (overhead) | CPU-DRYRUN + LIT | §2, Kevin Jose |
| argmax (d_seg, in verdict/reorient) | ★★ | chip-dependent (+503%…−59%) | LIT + CPU | TristanBilot |
| gather/scatter (code table, palette) | ★ | mixed, usually MLX-favorable | LIT | TristanBilot |
| elementwise / activations | ★ | torch often wins; MLX needs fusion | CPU + LIT | §2, TristanBilot |
| optimizer (Muon/AdamW) + EMA | ★★ | MLX-only (no torch twin); profile-only | PENDING | — |

---

## 6. PRIORITIZED MLX-SUPERIORITY PUSH PLAN (the multi-week roadmap)

Ranked by **(critical-path weight × current gap × inverse-effort)**. Each item:
the concrete optimization, the expected win, and effort. Magnitudes confirm on the
PENDING GPU sweep; the ranking is robust to that.

### P1 — Harden + extend the custom Metal grouped-conv backward (HIGHEST weight; partly won)
The 18× win is banked and default-ON, but per-step overhead is the remaining gap.
- **P1a. Fuse grad_input + grad_weight into ONE kernel dispatch** (currently two
  `mx.fast.metal_kernel` launches per conv per step). Halves launches on the >97%
  path. *Effort: M. Win: 10–30% backward.*
- **P1b. Kill per-step lazy-eval/host-sync overhead** (issue #1828): ensure the
  backward stays fully on-device across the accum loop (no mid-loop numpy recovery;
  single `mx.eval` per accumulated step), and probe `mx.compile` around the
  grad closure. *Effort: M. Win: up to 2–2.5× at high call counts.*
- **P1c. Autotune threadgroup** (fixed 256) per shape; the grad_weight kernel is
  one-thread-per-weight (tiny grid for depthwise) — try reduction-style tiling.
  *Effort: M. Win: 10–40% on small-grid configs.*
- **P1d. Cross-chip correctness guard** (issue #2205): add an M5-vs-CPU bit-parity
  CI assertion for the kernels (we already validate vs reference; pin it per chip).
  *Effort: S. Win: correctness insurance, not speed.*

### P2 — Fused R-operator (bicubic↑ → uint8 STE → bilinear↓) Metal kernel (high weight, large gap)
MLX's separable per-axis resize is the clearest forward gap (CPU 0.07–0.11× vs torch;
literature conv_general/resample slowness). The R roundtrip runs **twice per pair**
(f0,f1) inside the loss. A single fused `mx.fast.metal_kernel` doing bicubic-up +
uint8-STE + bilinear-down (contest-faithful coefficients, with VJP for the STE
passthrough) would collapse ~4 resize passes + a pad/round into one kernel.
- *Effort: M–L (write fwd + vjp, validate vs `apply_contest_faithful_roundtrip_nhwc`).
  Win: potentially 3–10× on R; this is the most likely NEW MLX-superiority kernel.*

### P3 — Forward conv coverage on M5 (high weight, chip-dependent gap)
Measure SegNet/PoseNet conv forward configs on M5 GPU (PENDING). For any config where
native `mx.conv2d` loses to MPS, route through **im2col + matmul** (MLX large-matmul
is strong) or a custom forward kernel. Depthwise/SE blocks (FastViT, EfficientNet-B2)
are the suspects.
- *Effort: M. Win: close the worst forward-conv cells; convert "torch faster" rows to "MLX faster".*

### P4 — `mx.compile` fusion of the INR trunk + scorer pre/post chains (low effort, broad)
Wrap the INR trunk (in_proj → 4×[Linear·FiLM·act] → out_sdf → softmax → palette →
sigmoid) and the elementwise/activation glue in `mx.compile` to fuse the many small
Linear+bias+FiLM+activation launches into fewer kernels (kills the small-op overhead
the CPU run + literature show).
- *Effort: S. Win: 10–30% on the forward; removes launch overhead.*

### P5 — Exploit unified memory + async authority eval (structural MLX edge)
The async-CPU-authority pattern already in `driver.py` (train MLX-GPU, authority
CPU-torch in a background thread, zero device copies) is MLX's structural advantage
over an MPS pipeline that would copy. Formalize + measure the end-to-end step
throughput (train + verdict) as the real-workload metric where MLX's no-copy wins
even where a single op is at parity.
- *Effort: S (measure) / M (optimize). Win: real-workload throughput, not microbench.*

### P6 — Standing benchmark + CI gate (campaign infrastructure)
Promote `tools/bench_mlx_vs_torch_mps.py` to a scheduled GPU sweep (when GPU free) that
appends a JSON row per chip/version and regresses on any op that drops below torch-MPS,
so MLX-superiority is *maintained*, not a one-shot. Add `--dtypes fp32,bf16,fp16` (torch-MPS
supports low precision; the witness is fp32 but landscape coverage matters).
- *Effort: S. Win: durable; catches MLX/torch version regressions.*

### Push-plan ranking summary
1. **P1 grouped-conv backward fusion/overhead** — the >97% path; biggest absolute step win; partly banked.
2. **P2 fused R-operator kernel** — largest *forward* gap, 2×/pair; best NEW custom-kernel opportunity.
3. **P3 forward conv coverage** — chip-dependent; convert losing forward cells.
4. **P4 mx.compile trunk fusion** — cheap, broad overhead kill.
5. **P5 unified-memory async pipeline** — structural edge, real-workload metric.
6. **P6 standing CI sweep** — keep MLX superior over time.

---

## 7. What's MEASURED vs PENDING (honest ledger)

- **MEASURED (this session, CPU dry-run, advisory):** harness correctness, parity
  across all 13 ops (PASS), MLX-CPU-vs-torch-CPU ratios (lazy-eval/overhead signature).
- **MEASURED (in-repo, GPU, prior):** custom Metal grouped backward ~18× (the lever).
- **PENDING-GPU (fires when the witness arm frees, one command in §0):** the actual
  MLX-GPU-vs-torch-MPS per-op ratios on M5 Max; the `=1` vs `=0` backward-lever delta;
  R-operator + forward-conv GPU gaps that drive P2/P3 magnitudes.

Files:
- harness: `tools/bench_mlx_vs_torch_mps.py`
- CPU dry-run JSON: `.omx/research/mlx_vs_torch_mps_bench_cpu_dryrun_20260630T161107Z.json`
- this memo: `.omx/research/mlx_vs_torch_mps_bench_20260630T161107Z.md`
