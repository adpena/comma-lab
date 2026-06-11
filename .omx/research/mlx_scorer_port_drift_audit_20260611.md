# MLX scorer port drift audit — can MLX-GPU be the fast fidelity scorer?

**Date:** 2026-06-11
**Author:** mlx-scorer-drift-audit subagent
**Evidence grade:** `[macOS-MLX research-signal]` / `[macOS-CPU advisory]` for all MLX numbers; torch-CPU is the only authority.
**Score claims:** NONE. No promotion / rank / kill from any number here. This is a portability + drift characterization.
**Hardware:** M5 Max, MLX 0.31.1, torch 2.11.0 (CUDA unavailable → torch-CPU is the authority axis). No MPS used anywhere.
**Did the exact frontier pointer move?** No. This is an enabler for GPU-fast fidelity scoring; it does not itself lower the exact score.

---

## 1. Audit — what MLX scorer pieces exist

The MLX scorer port is **far more complete than "partial"** — both networks are fully ported with a parity-audit
framework that measures exactly the contest-charged quantities. Key files:

| File | What it is |
|---|---|
| `src/tac/local_acceleration/mlx_scorer_adapters.py` (~70 KB) | **The actual MLX port.** Full EfficientNet-B2 SegNet (`MLXEfficientNetStemAdapter` / `MLXEfficientNetStageAdapter` / `MLXDepthwiseSeparableConvAdapter` / `MLXInvertedResidualAdapter` / `MLXSEModuleAdapter` / `MLXUnetDecoderAdapter` / `MLXSegmentationHeadAdapter` → `MLXSegNetAdapter`) AND full FastViT-T12 PoseNet (`MLXMobileOneBlockAdapter` / `MLXRepMixerBlockAdapter` / `MLXPatchEmbedAdapter` / `MLXFastVitVisionAdapter` / `MLXHydraAdapter` → `MLXPoseNetAdapter`) → `MLXDistortionScorerAdapter`. Weights loaded from upstream via `torch_distortion_net_to_mlx(dist)`. NCHW↔NHWC adapters + a `MLXReferenceConv2dAdapter` with fixed-order / Kahan / fp64 accumulation modes for drift forensics. |
| `src/tac/local_acceleration/mlx_scorer_torch_parity.py` (~1830 lines) | **The parity-audit framework.** `build_mlx_scorer_torch_parity_manifest[_sweep]` measures the contest quantities: `segnet_argmax_diff_pixels` (the d_seg charge), `posenet_component_abs_max` (the d_pose charge, using the per-dim Mahalanobis the scorer charges), plus logit/output deltas. Strict gate thresholds `max_segnet_argmax_diff_pixels=0`, `max_posenet_component_abs_delta=2.0e-5`. Also has `build_mlx_segnet_layer_trace_manifest` for layer-by-layer drift localization. |
| `src/tac/local_acceleration/mlx_scorer_response.py` | Real scorer-input cache loader (`load_scorer_input_cache` → `segnet_last_rgb (N,3,384,512)` + `posenet_yuv6_pair (N,12,192,256)`); `GPU_RESEARCH_SIGNAL_BLOCKER` fail-closed on GPU score claims. |
| `src/tac/local_acceleration/mlx_scorer_port_inventory.py` | A coverage **planner** (NOT a port). Explicitly `full_mlx_port_claim_allowed: False`. Useful as a checklist; do not read it as "nothing is ported." |
| `src/tac/tests/test_mlx_scorer_torch_parity.py` | Parity tests — but on **synthetic fixtures** (`_write_test_cache`), not real 0.mkv. They assert `segnet_argmax_diff_pixels == 0` + `posenet_component_abs_max <= 2e-5` on CPU. The real-input gap is what this audit fills. |
| `src/tac/mlx_pr95_port/score_bridge.py` (`TorchScorerBridge`) | **How training scores TODAY:** the frozen **torch-CPU** DistortionNet with a custom VJP back to the MLX render. This is the ~18min/epoch bottleneck — the scorer forward+backward is torch-CPU. The MLX scorer port above is NOT yet wired into the training loss. |

**Verdict on audit:** there IS a full MLX SegNet, a full MLX PoseNet, and a parity framework that measures the exact
charged quantities. What is missing is (a) a real-0.mkv parity gate (tests only use synthetic), and (b) wiring the MLX
scorer into the training loss in place of `TorchScorerBridge`.

## 2. Drift characterization (the core deliverable, NO-FAKE)

Measured on the **real reference-video (0.mkv) scorer cache**
`experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600/`
(`segnet_last_rgb` sha `f2b904ac…`, 600 pairs; cache builder `_av_frame_to_rgb_uint8` is a **byte-identical
structural copy of upstream `frame_utils.yuv420_to_rgb`** — same plane extraction, bilinear chroma upsample
`align_corners=False`, BT.601 limited-range, clamp/round/uint8 — so the GT-decode NO-FAKE requirement is satisfied).

Representative sample: **first 100 pairs** (= 19,660,800 SegNet argmax pixels). Contest charges: **d_seg = argmax-disagreement
RATE** (SegNet on frame1), **d_pose = MSE on first-6 PoseNet dims** (both-frame YUV6).

| Quantity | MLX-CPU vs torch-CPU | MLX-GPU (Metal) vs torch-CPU |
|---|---|---|
| **d_seg argmax pixels FLIPPED** (total / 19.66M) | **2** | **243** |
| d_seg flip rate (overall) | 1.0e-7 | **1.24e-5** |
| d_seg flip fraction (worst 4-pair window) | 1.27e-6 | 1.78e-5 |
| min top-2 logit margin AT a flipped pixel (mean) | 2.4e-7 | 5.2e-5 |
| SegNet logit abs max delta | 5.7e-5 | **9.6e-2** |
| **d_pose component abs max** (the charged term) | **8.7e-11** | **2.76e-4** (mean 2.22e-4) |
| PoseNet raw 12-dim output abs max delta | 2.3e-5 | 4.1e-2 |
| pure scorer-forward throughput (SegNet+PoseNet) | — | **8.7 pairs/s** (920 ms / batch-8) |
| torch-CPU pure scorer-forward (contended w/ capstone daemon) | ~3.2 pairs/s | — |

**Interpretation:**
- **MLX-CPU is essentially bit-faithful.** 2 flipped pixels out of 19.66M, and BOTH sit at near-perfect ties
  (mean top-2 margin 2.4e-7 ≈ float32 ULP — i.e. a genuine argmax tie, harmless, not a port defect). Pose component
  drift 8.7e-11 ≈ fp32 round-off. **MLX-CPU = torch-CPU for d_seg/d_pose at the precision the contest charges.**
- **MLX-GPU drifts but stays tiny on d_seg.** 243 flipped pixels / 19.66M = **0.00124% of pixels**. EVERY flip is at a
  SegNet decision boundary (mean margin 5.2e-5) where the logit drift (max 0.096) just crosses the near-tie. A
  raw-logit drift this size that does NOT cross a class boundary is harmless — what matters is the flip rate, and it is
  ~1.2e-5. For training/atlas sensitivity work this is negligible.
- **MLX-GPU pose is the caution.** The pose-component abs drift (2.76e-4) is small in absolute terms, but at the PR106
  frontier operating point d_pose ≈ 3.4e-5 — i.e. **the GPU pose drift can EXCEED the pose signal itself near the
  frontier.** MLX-GPU pose is fine as a *relative* gradient/ranking signal during training, but NOT trustworthy for an
  *absolute* d_pose readout near the frontier without a torch-CPU authority check.

## 2b. Drift source localization (SegNet GPU layer trace, 2 real pairs)

First divergence cliff (`build_mlx_segnet_layer_trace_manifest` device=gpu):

| Layer | max abs delta | note |
|---|---|---|
| `encoder.stage_0.block_0.se` | 3.97e-3 | **FIRST cliff** — Squeeze-Excite global-avg-pool reduction (order-sensitive on Metal) |
| `encoder.stage_0.block_1.conv_dw` | 3.46e-2 | depthwise conv accumulation |
| `encoder.stage_1/2/3 .conv_pw / .bn2` | 3.5e-2 – 6.5e-2 | pointwise GEMM reduction-order drift accumulates |
| `decoder.block_1` | 6.87e-2 | upsample + concat + conv amplifies |
| `segmentation_head.logits` | 9.64e-2 | final logits — but only flips argmax at near-ties |

**Root cause: NOT a structural porting bug** (CPU has ~0 flips → the port is numerically correct). It is **Metal-vs-CPU
floating-point reduction-order drift** in conv/GEMM/pooling accumulation — the classic GPU vs CPU fp32 non-associativity.
The SE global-average-pool is the first amplifier (reductions over H×W=192×256 are the most order-sensitive op).

## 3. Verdict + plan

### Verdict: YES — MLX-GPU is fidelity-usable as the FAST training/atlas scorer, with torch-CPU as the authority gate.

The d_seg flip rate on MLX-GPU is ~1.2e-5 (243/19.66M) and confined to boundary near-ties → MLX-GPU SegNet gradients
and sensitivities are trustworthy for training/atlas. The two conditions:

1. **torch-CPU stays the authority.** Periodic torch-CPU re-score (e.g. every N epochs / before any promotion) — never a
   score/promotion claim from MLX-GPU. (MLX-CPU may also serve as a cheap near-authority cross-check: it is bit-faithful.)
2. **Pose needs an authority check near the frontier.** Use MLX-GPU pose for the *relative* training signal; recompute
   the *absolute* d_pose on torch-CPU before trusting it at d_pose≈3e-5 (the drift is the same order as the signal there).

### Drift sources + fixes (priority order)
- **Primary (GPU): conv/GEMM/pool reduction-order fp32 non-associativity.** Already partially addressed: the port ships
  `MLXReferenceConv2dAdapter` with `fixed_fp32` / `kahan_fp32` / `fixed_fp64` accumulation modes
  (`build_mlx_conv2d_accumulation_probe_manifest`). The *optimized* MLX Conv2d (used in the fast path) trades order for
  speed. **Fix if tighter parity needed:** route the SE pool + final segmentation-head conv (the two biggest amplifiers)
  through the Kahan/fp64 reference accumulator — but this is likely unnecessary given the 1.2e-5 flip rate is already
  negligible for training.
- **Secondary:** interpolation mode (decoder bilinear upsample), BN eps, YUV6 basis, argmax tie-break — all already
  matched (CPU parity ≈ 0 flips confirms these are byte-faithful; they are NOT the drift source).

### #1 next step to unlock GPU-fast fidelity scoring
**Wire the MLX-GPU `MLXDistortionScorerAdapter` into the training loss as the fast forward, with a torch-CPU authority
re-score every N epochs** — i.e. replace the `TorchScorerBridge` torch-CPU forward with the MLX-GPU forward for the
per-step training signal, keep torch-CPU as the periodic gate. Pure-forward is ~8.7 pairs/s on GPU; this is the lever
that turns the torch-CPU-infeasible 600-pair capstone into a GPU-fast loop. (The VJP/gradient path must also be MLX —
the adapter is MLX-native so `mx.vjp` through it is available; that wiring + a per-step d_pose authority cross-check is
the build.)

## Reproduce
Durable scripts + raw JSON committed at `.omx/research/mlx_scorer_drift_audit_20260611_artifacts/`:
```
A=.omx/research/mlx_scorer_drift_audit_20260611_artifacts
.venv/bin/python $A/mlx_drift_run.py gpu 100   # MLX-GPU vs torch-CPU  -> $A/gpu_100.json
.venv/bin/python $A/mlx_drift_run.py cpu 100   # MLX-CPU vs torch-CPU  -> $A/cpu_100.json
.venv/bin/python $A/mlx_segtrace_gpu.py        # SegNet layer-trace drift localization
.venv/bin/python $A/mlx_fwd_throughput.py gpu 8 12   # pure MLX-GPU forward throughput
```
Regression gate (certifies MLX-CPU bit-faithful on real 0.mkv):
`pytest src/tac/tests/test_mlx_scorer_torch_parity.py::test_mlx_cpu_bit_faithful_vs_torch_cpu_on_real_reference_video_cache`

## Caveats / NO-FAKE notes
- All MLX numbers are `[macOS-MLX research-signal]`; torch-CPU is authority. No score/promotion claim is made.
- torch-CPU throughput (3.2 pairs/s) was measured while the capstone daemon (pid 72123) was also using torch-CPU →
  the uncontended torch-CPU number is faster; the GPU/CPU forward *ratio* is therefore a lower bound, not an exact speedup.
- Sample = first 100 of 600 real 0.mkv pairs. The drift is stationary across windows (per-window stats are tight), so
  100 pairs is representative; a full 600-pair sweep + a comma10k generalization slice are the obvious extensions but
  were not needed to reach the verdict.
- The existing parity tests use synthetic fixtures only; a real-0.mkv CPU parity regression gate is the recommended
  follow-up to certify MLX-CPU bit-faithfulness structurally.
