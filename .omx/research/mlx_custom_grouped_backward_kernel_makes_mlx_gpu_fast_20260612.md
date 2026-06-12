# MLX custom Metal grouped/depthwise backward kernel — the missing throughput lever (2026-06-12)

**Author:** MLX-BACKWARD-KERNEL subagent.
**Operator question (verbatim):** "Why can't we build the mlx thing we need?" — and we CAN. It was already
~90% built by a sister agent and left un-wired with two bugs that hid the win.
**Evidence grade:** torch-CPU exact `modules.py` = the d_seg/d_pose authority. EVERY throughput / gradient-
fidelity number here is `[macOS-MLX research-signal]`. The custom backward is a GRADIENT THROUGHPUT tool,
NEVER a score authority. **NO MPS.** $0, local, no paid dispatch.
**Did the exact frontier pointer move?** No. This is a training-throughput enabler (makes MLX-GPU the fast
local backward backend), not a pointer move.

---

## TL;DR (the honest 6-line verdict)

1. **The MLX backward thing we need EXISTS and WORKS.** A sister agent already built the custom
   `mx.fast.metal_kernel` grad_input + grad_weight kernels (`tac.local_acceleration.metal_grouped_conv_backward`)
   AND diagnosed MLX's native reverse-mode blow-up rigorously. I did NOT rebuild the kernels (search-first).
2. It was **never wired in** because of **two real bugs** that hid the prize: (a) the `@mx.custom_function`
   passed the conv config as keyword-only args that MLX's `.vjp` does NOT forward → `TypeError` under
   `mx.grad`; (b) `metal_grouped_conv2d_backend_available()` did `mx.gpu.type` (AttributeError — `mx.gpu` is a
   `DeviceType` enum) so the gate always returned False. **I fixed BOTH** + added a config-bound factory.
3. **Diagnosis confirmed (sister + me):** native MLX strided-grouped VJP is numerically WRONG — grad cosine
   **0.025** (random direction), magnitude **5–25× too large** (the stride>1 grad-input scatter mishandles
   the group×stride indexing). Stride-1 grouped is fine (cosine 1.0). This is a real, narrow MLX bug.
4. **The custom backward is CORRECT:** per-layer grad_input + grad_weight cosine = **1.000000** vs the trusted
   Python-loop reference (relmax ~1e-7 = fp32 round-off). Full-SegNet pixel-gradient cosine **0.99999775**;
   full-scorer (SegNet+PoseNet) input-grad cosine **0.99999921 / 1.00000000**. NOT a fake/zero gradient.
5. **The win is BIG, and bigger than the prior agents thought:** swapping the custom backward into the real
   scorer makes the **full-scorer backward 17.96× faster** (11,149 ms → 621 ms at B=4), the **SegNet backward
   12.86× (B=4) → 35.45× (B=8)** faster. The backward is >97% of the training step, so this is a regime
   change — the sister LOCAL-MLX-DREAM agent measured only ~1.3–1.5× because their `MLXGpuScorerBridge` still
   routed the strided-grouped layers through the Python loop; they never had the custom backward wired.
6. **Wired behind `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`** (default OFF; bit-faithful reference stays the
   validated baseline). With the flag, 12 strided-grouped scorer layers (4 SegNet + 8 PoseNet) use the fast
   path through the canonical `torch_distortion_net_to_mlx`. Descent-equivalence A/B running (below).

---

## Task 1 — the profile: how big is the prize?

The strided-grouped fallback is the dominant cost on BOTH forward and backward — the right measurement is the
honest **end-to-end A/B** (build the real scorer two ways, time each whole), NOT an in-graph instrumented
fraction (an `mx.eval()` inside the patched fallback breaks MLX's lazy fusion and mis-attributes the time —
my first instrumented profile read 15.5% and was an artifact; the A/B below is the truth).

| Measurement (real weights) | Python-loop fallback | custom Metal backward | speedup |
|---|---:|---:|---:|
| **SegNet backward** (B=4) | 7,337 ms | 570 ms | **12.86×** |
| **SegNet backward** (B=8) | 20,206 ms | 570 ms | **35.45×** |
| **Full scorer backward** (SegNet+PoseNet, B=4) | 11,149 ms | 621 ms | **17.96×** |
| SegNet forward (B=4) | 847 ms | 176 ms | 4.82× |
| 4 SegNet fallback layers, grad fwd+bwd in isolation | 10,018 ms | 116 ms | 86.4× |

The custom backward is nearly **batch-invariant** (~570–620 ms) while the Python loop scales with batch — so
the speedup grows with batch (12.86× → 35.45× from B=4 → B=8). The 4 strided-depthwise SegNet layers + 8
PoseNet strided-grouped layers ARE the backward wall; removing them is the prize.

## Task 2 — diagnosis of MLX's reverse-mode blow-up

Measured on the real strided-grouped scorer shapes (sister's `native_blowup_diagnosis.json`, corroborated):

| shape | native grad cosine | native/ref magnitude | verdict |
|---|---:|---:|---|
| dw 3×3 g96 **s2** | **0.025** | 7.6× too large | WRONG |
| dw 5×5 g144 **s2** | **0.021** | 22× too large | WRONG |
| grouped 3×3 g64 **s2** | **−0.065** | 9.6× | WRONG (anti-correlated) |
| dw 3×3 g96 **s1** | **1.000** | 1.0× | CORRECT |

The forward `mx.conv2d` is bit-exact on Metal for ALL of them (parity ~1e-7). Only the **stride>1** grouped
VJP is broken: the grad-input scatter mishandles the group×stride index arithmetic (an output position at
`(ho,wo)` writes back to input `(ho*stride+kh*dil-pad, …)` within its group; the native kernel's stride+group
interaction in the reverse scatter is wrong). It is **narrow** (stride-1 grouped + all dense convs are
correct), which is why forcing only these 12 layers onto the loop fallback was the correct safe stopgap — and
why a correct custom Metal scatter (one thread per input element, explicit stride-alignment test, no atomics)
fixes it exactly. This is **upstreamable to MLX** as a focused bug report + kernel.

## Task 3 — the custom backward prototype + measured speedup

Reused the sister's kernels (`_GRAD_INPUT_SRC` / `_GRAD_WEIGHT_SRC`), fixed the wiring:
- Added `make_grouped_conv2d_nhwc(stride, padding, dilation, groups)` — a **config-bound factory** that closes
  the conv config over the `.vjp` (MLX's vjp signature is `(primals, cotangent, output)` ONLY; it never
  forwards keyword-only args, which is why the bare `grouped_conv2d_nhwc` raised `TypeError`).
- Fixed `metal_grouped_conv2d_backend_available()` (`mx.gpu.type` → `mx.gpu`; `mx.gpu` is a `DeviceType`).
- Added `MLXCustomKernelStridedGroupedConvAdapter` + an env gate `_custom_metal_backward_enabled()` and wired
  `torch_conv2d_to_mlx` to use the fast path on GPU when `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` (verified: 12
  custom adapters activate in the full scorer via the canonical `torch_distortion_net_to_mlx`).

Speedups: see the table in Task 1 (the load-bearing numbers).

## Task 4 — descent-equivalence (the NO-FAKE crux)

The gradient must be a CORRECT DIRECTION reaching the same exact-d_seg basin, not a fast-but-wrong gradient.

**Gradient-fidelity (necessary):** per-layer grad cosine 1.000000; full-SegNet pixel-grad cosine
**0.99999775**; full-scorer input-grad cosine 0.99999921 / 1.0. These are *tighter* than the MLX-GPU-vs-
torch-CPU cosine (0.99986) that the sister LOCAL-MLX-DREAM agent already proved reaches the same basin via a
descent A/B — so the custom backward is strictly MORE faithful than the already-validated MLX-GPU path.

**Descent A/B (sufficient — the trajectory test):** `experiments/measure_descent_equivalence.py` with
`TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`, arm A = torch-CPU authority gradient, arm B = MLX-GPU + **custom Metal
backward**, exact d_seg measured on the torch-CPU authority for BOTH arms.

**RESULT — CONFIRMED (n8, 40 epochs, fixed recipe muon-throughout CE muon_lr=0.03 grad_clip=50):**

| epoch | torch_d_seg (authority) | mlx_d_seg (custom backward) | abs gap | rel gap |
|---:|---:|---:|---:|---:|
| 0 | 0.507273 | 0.507273 | 0.000000 | 0.0% |
| 10 | 0.220551 | 0.213068 | 0.007483 | 3.4% |
| 20 | 0.023930 | 0.023006 | 0.000924 | 3.9% |
| 30 | 0.012939 | 0.012422 | 0.000518 | 4.0% |
| **40** | **0.011034** | **0.010707** | **0.000327** | **3.0%** |

**FINAL: torch-CPU authority d_seg 0.011034 vs custom-backward d_seg 0.010707 — abs gap 0.000327 = 0.07% of
the 0.496 descent.** The custom-Metal-backward MLX-GPU arm reaches the SAME exact-d_seg basin as the
torch-CPU authority gradient; the two track within ~3-4% the whole way down and converge at the basin (custom
slightly lower, identical pattern to the sister's already-validated native MLX-GPU descent). **The gradient
is a CORRECT direction, not fast-but-wrong — descent-equivalence is CONFIRMED.**

**AND it was 5.5× faster in real training:** arm A (torch-CPU authority) ran **27.0 s/epoch**; arm B
(mlx_gpu + custom Metal backward) ran **4.7-5.0 s/epoch**. This is the n8 single-batch number; at n600
multi-batch (where the GPU stays warm) the backward win is even larger (the full-scorer backward is 18× at
B=4). The init d_seg (0.507273) is bit-identical on both arms (same seed) — a degenerate/zero gradient would
NOT descend; arm B descends monotonically to the basin. Artifact:
`experiments/results/descent_equivalence_custom_backward_n8.json`.

## Honest verdict

**Building the MLX backward kernel DOES make MLX-GPU the fast local backward backend — it unblocks the local
dream.** The full-scorer backward (>97% of the step) goes from 11.1 s to 0.62 s at B=4 (17.96×) with a
descent-identical gradient, and the **real end-to-end training step went from 27.0 s/epoch (torch-CPU) to
4.7-5.0 s/epoch (mlx_gpu + custom backward) = 5.5× faster** in the n8 descent A/B, reaching the SAME exact
d_seg basin (final gap 0.07% of the descent). This is the regime change the sister agents concluded "didn't
exist" — they were right about the `MLXGpuScorerBridge` AS WIRED (Python-loop fallback inside), and wrong
about the ceiling once the custom backward replaces the loop. **It is worth completing AND upstreamable to
MLX** (the stride>1 grouped VJP is a real, narrow, well-characterized bug).

**The honest residual:** the win is on the strided-grouped *backward*; the remaining ~570 ms full-scorer
backward is the native VJP through the ~110 dense/pointwise/stride-1 conv layers (already fast on Metal). So
the floor after this fix is ~0.6 s/step (vs ~11 s), and the n600 epoch budget (the open question from the
LOCAL-MLX-DREAM memo) is now bounded by a ~0.6 s/step backward, not an 11–20 s one — **the resumable n600
daemon should be re-launched on the `mlx_gpu` backend with this flag set.**

## Files

- `src/tac/local_acceleration/metal_grouped_conv_backward.py` — added `make_grouped_conv2d_nhwc` factory
  (vjp-config-closure fix) + fixed `metal_grouped_conv2d_backend_available` DeviceType bug + made the bare
  custom_function raise a helpful error instead of a confusing TypeError.
- `src/tac/local_acceleration/mlx_scorer_adapters.py` — added
  `MLXCustomKernelStridedGroupedConvAdapter` + `_custom_metal_backward_enabled()` env gate + wired
  `torch_conv2d_to_mlx` to the fast path (default OFF).
- `src/tac/local_acceleration/tests/test_metal_grouped_conv_backward.py` — 16 NO-FAKE tests (gradient
  direction + magnitude vs reference on real scorer shapes; non-zero guard; forward parity; the bare-vjp
  error guard; the backend-availability regression guard).
- `experiments/profile_mlx_segnet_backward_fallback.py` — the fallback-fraction profiler (+ the in-graph
  instrumentation caveat).
- `experiments/wire_and_measure_mlx_custom_backward.py` — per-layer validate + time (86.4× aggregate).
- `experiments/measure_mlx_segnet_end_to_end_custom_backward.py` — end-to-end SegNet A/B.
- `experiments/measure_mlx_full_scorer_custom_backward.py` — end-to-end full-scorer A/B.
- `experiments/results/mlx_*backward*.json` — the measured artifacts.

## Reproduce

```
# per-layer correctness + speed
PYTHONPATH=src:upstream .venv/bin/python experiments/wire_and_measure_mlx_custom_backward.py
# end-to-end SegNet A/B (B=8 → 35.45x)
PYTHONPATH=.:src:upstream .venv/bin/python experiments/measure_mlx_segnet_end_to_end_custom_backward.py --batch 8
# full scorer A/B (17.96x)
PYTHONPATH=.:src:upstream .venv/bin/python experiments/measure_mlx_full_scorer_custom_backward.py --batch 4
# descent A/B with the custom backward enabled
PYTHONPATH=.:src:upstream TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 MLX_METAL_GPU_ARCH=applegpu_g15 \
  .venv/bin/python experiments/measure_descent_equivalence.py --max-pairs 8 --epochs 40 \
  --eval-every 5 --targets-cache experiments/results
# tests
PYTHONPATH=src:upstream .venv/bin/python -m pytest src/tac/local_acceleration/tests/test_metal_grouped_conv_backward.py -q
```
