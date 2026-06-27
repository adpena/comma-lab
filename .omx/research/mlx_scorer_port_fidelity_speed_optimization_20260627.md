# MLX scorer-port fidelity×speed optimization + 4th-measurement-artifact diagnosis (2026-06-27)

Tasks #88/#89 — operator directive 2026-06-26 "optimize the MLX scorer ports for signal
fidelity + speed." $0 CPU benchmark + brief GPU fidelity probes. MLX 0.31.1, torch 2.11.0,
M5 Max (Apple GPU). Evidence: `[macOS-MLX research-signal]` / `[Apple-GPU-vs-torch-CPU parity,
exact-measured]`. NO score claim; pointer UNMOVED. Authority = torch-CPU / numpy-fp32.

## BENCHMARK TABLE (device × speed × fidelity-px)

Two distinct precision surfaces. The argmax verdict is hypersensitive: sub-0.15 d_seg
budget ≈ **143 argmax px/pair** (7.3e-4 of 196608 px).

SCORER PORT (SegNet `tu-efficientnet_b2` UNet + PoseNet FastViT, on cached real 0.mkv frames):

| device | fwd pairs/s (batch4) | speedup vs torch-CPU | seg argmax px/pair (max·mean) | seg logit L∞ | pose comp L∞ | grade |
|---|---|---|---|---|---|---|
| torch-CPU | 2.01 | 1.0× (AUTHORITY) | 0 (by def) | 0 | 0 | verdict authority |
| MLX-CPU | 1.46 | **0.73× (SLOWER)** | 0 · 0 | 4.7e-5 | 2.2e-11 | fp32-exact, but no speed win |
| MLX-GPU | 7.45 | **3.71×** | 13 · 10 | 0.096 | 2.5e-4 | reduced-ORDER → gradient-only |

WITNESS RENDER (coord-INR MLP, high-omega SIREN/hosc → RGB; the verdict feeds this to torch-CPU SegNet):

| render device | RGB max-Δ /255 | uint8-differing px | realized SegNet argmax px/pair | grade |
|---|---|---|---|---|
| numpy-fp32 | ~1e-3 (vs fp64) | 0 (byte-close) | **0** | AUTHORITY (== inflate.py) |
| MLX-CPU | ~1e-3 | ~8 | 3–18 | monitoring-ok (<<143) |
| MLX-GPU | ~0.19 (real) | ~all | **495–1672** | NOT verdict-trustworthy |

## DRIFT DIAGNOSIS (which op, localized — PROVEN)

Controlled probes isolate the cause to **fp32 reduction-ORDER non-associativity**, NOT
reduced-precision accumulation and NOT fast-math transcendentals:

- `matmul` K=2048 fp32: MLX-GPU-vs-fp64 = **1.8e-4** (BETTER than numpy-fp32's 3.6e-4);
  MLX-GPU-vs-numpy = 4.0e-4. ⇒ GPU matmul accumulates in **true fp32** (tf32 would be ~1e-2).
  The GPU↔numpy delta is purely the tiled/SIMD reduction order vs numpy's order.
- `sin` (ω=30): MLX-GPU-vs-numpy = **1.19e-7**; `exp` = 8.9e-8. ⇒ NOT Metal fast-math.
- SegNet layer trace (GPU): cliffs are the **pointwise 1×1 conv (`conv_pw`, matmul) + `bn2`**
  layers, compounding through ~500 layers → `segmentation_head.logits` L∞ 0.096 → ~10 near-tie
  argmax flips/pair (all flips have top2-margin < the 0.096 drift bound).
- Witness MLP: the same ~4e-4 reduction-order Δ is **amplified by high-ω activations**
  (d/du sin(ωu)=ω·cos, ω≈30 per layer) → ~0.19/255 RGB → hundreds of argmax flips. MLX-CPU's
  reduction order is near-numpy ⇒ stays at 3–18 px; MLX-GPU's tiled order diverges → 495–1672.

Root: GPU parallel reduction order ≠ CPU/numpy order; fp32 add is non-associative. Inherent.

## OPTIMIZATION (GPU made fp32-exact? → NO; optimal policy instead)

GPU **cannot** be made bit-exact in MLX 0.31.1:
- `mlx_runtime_determinism_contract` → `framework_different_no_public_deterministic_reduction_flag`.
- `dir(mx)` / `mx.fast` / `mx.matmul` doc: NO precision / determinism / accum / tf32 control.
- `mx.float64` exists but **Metal has no fp64** → CPU-only; cannot fix GPU.

So "fast AND bit-exact GPU verdict" is unreachable. The **optimal device-split policy** (already
implemented in the witness trainer, and the canonical recommendation for the level-set sibling):
- **Training gradient → MLX-GPU** (3.71× scorer-forward speedup; reduction-order is fine for the
  gradient — through-R grad cosine ~0.9999 / 0.99996; CHAOS verdict: the gradient is per-step
  correct, only the optimizer can diverge).
- **Verdict (d_seg/d_pose) → numpy-fp32 render → torch-CPU SegNet/PoseNet** (0 px, byte-identical
  to inflate.py = the byte-close authority).
- **In-loop monitoring → MLX-CPU render** (3–18 px << 143 budget; fast for the witness MLP) →
  torch-CPU scorer.
- **MLX-CPU is NOT useful for the SCORER** (1.46 < 2.01 pps: slower than torch-CPU AND not the
  authority) — torch-CPU is both authority and faster, so the verdict scorer stays torch-CPU.

## VALIDATION (fidelity vs oracle + speedup)

- MLX-CPU scorer == torch-CPU oracle: **0 argmax px** / logit L∞ 4.7e-5 / pose L∞ 2.2e-11 (8 pairs).
- numpy-fp32 render == fp64 ref to ~1e-3/255 → 0 byte-close px (matches the deployed
  `_witness_forward_numpy` == inflate.py template; regression-guarded by byte-close parity_on_inflated).
- MLX-GPU scorer drift reproduced exactly vs the 2026-06-11 audit (max 13–14 px, logit 0.096).
- MLX-GPU forward speedup **3.71×** vs torch-CPU (pure forward, batch 4).

## WALL-CLOCK IMPACT (trustworthy-fast verdict?)

- There is **no** trustworthy-fast GPU verdict; the trustworthy verdict is torch-CPU-bound
  (~2 pps), mitigated by pair-subset + batched single-forward (already in eval_verdict).
- The real wall-clock win is on the **gradient** (MLX-GPU 3.71×), which is what makes n600
  tractable; the verdict correctly does NOT ride the GPU.
- 4th measurement artifact = the MLX-GPU **witness render** in the verdict (495–1672 px). FIXED
  by rendering the verdict in numpy-fp32 (DAG FEED-br, already in the witness trainer). The
  level-set sibling (a8c34178) must adopt the same numpy-fp32 verdict render (do NOT use
  mlx-gpu for any scored verdict).

## SYSTEM-INTELLIGENCE WIRE-IN

Codified as `tac.local_acceleration.mlx_scorer_torch_parity.mlx_scorer_device_fidelity_speed_profile()`
(measured numbers + proven diagnosis + policy, false-authority-flagged) so the verdict path,
the level-set sibling, and the autopilot consume the policy instead of re-deriving it.
Bench scripts: `.omx/tmp/mlx_scorer_bench/` (sweep/throughput/probe/witness_probe — ephemeral scratch).
