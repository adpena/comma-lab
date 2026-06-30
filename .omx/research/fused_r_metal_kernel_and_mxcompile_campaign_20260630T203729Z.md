# Fused R-operator Metal kernel + mx.compile d_seg step — MLX-superiority P2 / P1b·P4

`[macOS-MLX/Metal engineering]` — ADVISORY wall-clock engineering. **means≠ends:**
this is throughput tooling for FUTURE witness runs behind default-OFF flags; the
canonical frontier pointer **0.19110 is UNMOVED** and moves only on a byte-closed
exact-eval row. MLX/Metal is a gradient/compute substrate, **never** a score
authority (numpy-fp32 CPU / contest-CUDA only).

- **Date (UTC):** 2026-06-30T20:37Z · **HEAD:** 565de4372 · **Host:** M5 Max, 128 GB, Metal, macOS 26.4, MLX 0.31.2
- **Campaign:** `mlx_vs_torch_mps_bench_20260630T161107Z.md` push-plan **P2** (fused R-operator, top NEW-kernel EV) + **P4/P1b** (`mx.compile` fusion of the INR trunk + d_seg grad closure).
- **Hard rail honored:** the live n600 arm (`levelset_n600_v2_attrclean_20260630T194549Z`) owns the single Metal GPU. **No GPU kernel was executed.** All BUILD + numpy/CPU-oracle parity + tests are CPU/MLX-CPU (non-contending). The Metal-on-GPU validation is a ready-to-run gated harness (fires when the GPU frees).

---

## Deliverable 1 (TOP EV) — fused R-operator Metal kernel

`src/tac/local_acceleration/metal_fused_r_operator.py` (+ tests
`src/tac/tests/test_metal_fused_r_operator.py`, 12 CPU tests PASS).

### What it is
The contest-faithful R (eval roundtrip), realized twice per pair inside the d_seg
loss:

    render (Hin×Win) --bicubic↑ CAMERA(874×1164)--> uint8 STE @ CAMERA --bilinear↓ SCORER(384×512)

The MLX production R (`pr95_hnerv_mlx_training.apply_contest_faithful_roundtrip_nhwc`)
does this as **5 separable per-axis passes + a round**, each materializing a large
`(B,out,taps,W,C)` intermediate — the clearest *forward* gap vs torch-MPS (CPU
render_R **0.08–0.14× torch**, re-measured this session below). The kernel collapses
it into **2 on-device `mx.fast.metal_kernel` launches**: (1) fused bicubic-up + clip
+ round @ camera, (2) fused bilinear-down to scorer — W-outer/H-inner tap order
== MLX's H-pass-then-W-pass order, so **bit-faithful**.

### Coefficient correction (NO-FAKE: match the REAL oracle)
The campaign memo/prompt loosely say bicubic `a=-0.5`. The **actual** MLX production
R uses **`a=-0.75`** (PyTorch default; `_cubic_convolution_weight`). The kernel +
numpy oracle match the real code (`a=-0.75`). `mx.round` verified = round-half-to-even
on this host (matches `np.rint`); the Metal kernel uses `rint()`.

### Parity (MEASURED, CPU — the math is validated now)
numpy oracle (`fused_r_forward_numpy`) vs the MLX production R, MLX-CPU:

| shape | fwd max\|Δ\| | frac bit-identical |
|---|---:|---:|
| **full witness N1 (384×512→874×1164→384×512)** | **0.000000** | **1.000000** |
| **full witness N4** | **0.000000** | **1.000000** |
| tiny odd shapes (24×32→55×73) | ≤ 1 LSB | ≥ 0.999 |

On the **real witness shapes the oracle is BIT-IDENTICAL** to the MLX production R.
The ≤1-LSB worst case only appears on tiny odd-size test shapes (edge undershoot +
fp partition-of-unity near a round half-boundary) — the documented sub-LSB note.
The Metal kernel uses the SAME exact tap-sum order as the oracle, so **GPU bit-identity
is expected** (the gated harness confirms on-device).

VJP: `fused_r_vjp_numpy` (analytic transpose of the two linear resamples + STE-round
passthrough + clip subgradient) matches MLX autodiff of the oracle **within rtol 2e-3 /
atol 2e-4** (verified MLX-CPU). STE/clip semantics tested: fully-saturated input
(>255) → grad exactly 0 (clip kills); interior input → grad == no-round transpose
(STE passthrough).

### Design (honest scope)
- **FORWARD:** fast custom Metal (2 kernels). The measured gap, the real win.
- **VJP:** `mx.vjp` of the bit-faithful pure-MLX oracle (correct STE/clip/transpose
  grad, near-zero cross-chip risk). Rationale: unlike the strided grouped-conv backward
  (numerically WRONG natively → needs a custom grad kernel), the resize backward is
  *correct* in native MLX (only the forward is slow), and R-backward is a small fraction
  of the step vs the grouped-conv backward (the >97% lever, handled separately). A
  fully-fused Metal transpose VJP is a documented future extension (**P2b**).
- **Per-chip guard (#2205):** `assert_metal_matches_cpu_oracle` asserts metal-forward ==
  numpy oracle (atol 0) + metal-VJP == numpy analytic VJP (within tol) on THIS chip.

### Flag wiring (default OFF — for FUTURE runs)
The drop-in is `apply_contest_faithful_roundtrip_nhwc` → `fused_r_roundtrip` at the
two call sites in `experiments/train_witness_realized_through_R_mlx.py`
(`render_through_R_mlx` :353, `render_batch_through_R_mlx` :371). To avoid touching the
LIVE trainer + colliding with sibling subagents, the trainer edit is **documented, not
applied** (the flag + dispatch live in the module API: `metal_fused_r_available()` gates,
`fused_r_roundtrip_reference` is the OFF/non-Metal fallback). Ready-to-apply patch:

```python
# argparse: ap.add_argument("--fused-r-kernel", action="store_true", default=False)
# in render_*_through_R_mlx (both sites):
if getattr(args, "fused_r_kernel", False) and metal_fused_r_available():
    r = fused_r_roundtrip(rgb, camera_hw=(874,1164), output_hw=(SEG_H,SEG_W), ste_round=True)
else:
    r = apply_contest_faithful_roundtrip_nhwc(rgb, output_hw=(SEG_H,SEG_W), ste_round=True)
```

### Expected win
3–10× on R (P2 estimate); R runs twice/pair in the loss forward, so this is the
largest NEW MLX-superiority forward kernel. Magnitude confirms on the GPU sweep.

---

## Deliverable 2 — `mx.compile` the d_seg step (EXCLUDING pose)

`src/tac/local_acceleration/mlx_compile_step.py` (+ tests
`src/tac/tests/test_mlx_compile_step.py`, 4 CPU tests PASS).

- `compile_loss_and_grad(loss_and_grad_fn, *, enabled=…, shapeless=…)` — `mx.compile`
  wrapper; `enabled` = the `--compile-step` flag (default OFF). MUST wrap the **seg-only**
  loss-and-grad closure — **pose is EXCLUDED** (pose explodes under compile; in the
  witness it is `w-pose=0` monitoring-only on the async CPU-authority thread, already off
  the training step).
- `assert_compile_bit_identical(...)` — equivalence gate + determinism gate.
- `build_representative_dseg_trunk` / `representative_dseg_loss_and_grad` — a trunk
  mirroring the real witness (`in_proj→4×[Linear·FiLM·act]→out_sdf→softmax→palette→sigmoid`)
  + a seg-only CE surrogate (NO pose), to prove the technique on CPU.

### MEASURED (CPU) — `mx.compile` is "same math" within fp-fusion ULP
`mx.compile` is graph fusion and CAN reorder fused multiply-adds → fp32 ULP-scale diff,
**NOT a math change** (this matches the spec's `<1e-6` bound, not `=0`):

| metric | value | gate |
|---|---:|---|
| compiled-vs-uncompiled **loss** max\|Δ\| | **2.38e-7** | < 1e-6 ✓ |
| compiled-vs-uncompiled **grad** max\|Δ\| (17 leaves) | **2.68e-7** | < 1e-6 ✓ |
| **determinism** (compiled re-run, same inputs) max\|Δ\| | **0.0** | EXACT ✓ |

So: equivalence within `atol=1e-6` (graph-fusion ULP), determinism **byte-exact**.

### Composition + wiring (documented)
Wrap the trainer's `value_and_grad = nn.value_and_grad(model, loss_fn)` (line 1736 of
`train_levelset_witness_realized_through_R_mlx.py`) with `compile_loss_and_grad(...,
enabled=args.compile_step)` — using the canonical MLX state-capture pattern
(`inputs=[model.state, opt.state]`) for the full step. `mx.compile` composes with the
custom grouped-conv backward and (when on) the fused-R kernel: it captures the
surrounding graph and the `@mx.custom_function` ops keep their registered VJPs. The
trainer edit is **documented not applied** (live-run + sibling-subagent safety); CPU
proves the compile bit-identity of the pure-MLX trunk, the GPU harness proves full
composition.

### Expected win
10–30% on the forward (kills small-launch overhead). Magnitude confirms on GPU.

---

## Deliverable 3 — bench extension

`tools/bench_mlx_vs_torch_mps.py` `_build_render_R` now adds an `mlx_fused` candidate
(GPU-gated: only when a Metal default device is active — the custom kernel can't run on
CPU). `parity_ref` switched `torch`→`mlx` so the fused candidate is gated against its
true oracle (the MLX production R, ~0 Δ); torch's row stays informational (the known
`a` coeff note, |Δ|≈0.93) under the generous tol. Re-measured CPU dry-run (this session,
non-contending) reconfirms the forward gap P2 targets:

| op | mlx_ms (CPU) | torch_ms (CPU) | ratio torch/mlx |
|---|---:|---:|---:|
| render_R N1 | 27.3 | 3.7 | 0.14 (torch faster) |
| render_R N4 | 176.0 | 14.0 | 0.08 (torch faster) |

### GPU validation command (DEFERRED — fire when the GPU frees; refuses while training)
```bash
# 1) per-chip correctness (forward bit-identical + VJP within tol), tiny/bounded:
.venv/bin/python -c "import mlx.core as mx; mx.set_default_device(mx.gpu); \
from tac.local_acceleration.metal_fused_r_operator import assert_metal_matches_cpu_oracle as A; \
print(A(in_hw=(384,512), camera_hw=(874,1164), output_hw=(384,512), batch=1))"

# 2) full timing sweep incl. the mlx_fused candidate (only when no thetastar/train_levelset arm alive):
.venv/bin/python tools/bench_mlx_vs_torch_mps.py --gpu-sweep --op render_R \
  --out .omx/research/mlx_vs_torch_mps_bench_gpu_fusedR_$(date -u +%Y%m%dT%H%M%SZ).json
```

---

## Files
- `src/tac/local_acceleration/metal_fused_r_operator.py` (NEW) — fused-R kernel + numpy oracle (fwd+VJP) + per-chip guard.
- `src/tac/local_acceleration/mlx_compile_step.py` (NEW) — `mx.compile` d_seg-step wrapper + bit-identity/determinism harness + representative trunk.
- `src/tac/tests/test_metal_fused_r_operator.py` (NEW, 12 tests) · `src/tac/tests/test_mlx_compile_step.py` (NEW, 4 tests).
- `tools/bench_mlx_vs_torch_mps.py` (edit) — `mlx_fused` candidate (GPU-gated) + parity_ref→mlx.

## Ledger — MEASURED vs PENDING
- **MEASURED (CPU, now):** numpy oracle == MLX production R **bit-identical** on witness shapes; analytic VJP == MLX autodiff (within tol); `mx.compile` equivalence 2.4e-7 (<1e-6) + determinism exact; 16/16 CPU tests green; bench builds+runs.
- **PENDING-GPU (gated harness, fires when GPU free):** metal-forward on-device bit-identity (expected 0), metal-VJP on-device, the 3–10× R / 10–30% fwd timing on M5.
- **Pointer:** 0.19110 UNMOVED (this is a MEANS — a faster R/step buys wall-clock for future witness runs; only a byte-closed exact row moves the score).
