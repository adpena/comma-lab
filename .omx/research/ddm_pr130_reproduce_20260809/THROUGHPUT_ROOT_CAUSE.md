# PR130 local training throughput — measured root cause

**Date** 2026-08-09 · **Authority** `[macOS-Metal advisory]` · `score_claim=false`
**Host** Darwin arm64, M5 Max, 128 GB · torch 2.10.0 (pinned runtime
`/Volumes/VertigoDataTier/pact/ddm_pq1_runtime_20260809/venv`)
**Question** (operator): *"It seems like it's way too slow for some reason. like, we have
everything we should that should make it faster."* + *"We have a ton of compute and memory
we can totally saturate."*

Every number below is measured on this host, on PR130's real semantic trainer at its real
config (width 96, blocks 4, `quant_bits=4`, 66,339 params, 384×512 token grid). Receipts in
`/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/`.

---

## 0. MAIN's own arithmetic correction (stated first — it inflated "too slow")

I reported **~0.48 s/step** earlier from a single data point (100 steps + load + one eval =
77 s) without separating the fixed cost. Two-point solve:

| batch | 30 steps total |
|---|---|
| 2 | 65.1 s |
| 8 | 79.5 s |
| 16 | 96.3 s |

⇒ **per-step @bs2 = 0.170 s**, **fixed per-invocation = 60.0 s** (torch import + model/GT
load + one final n600 eval). The 6,000-step semantic tail is **~25 min**, not ~1 h.
`LOCAL_TRAINING_AUDIT.md`'s "~1 h" is superseded by this file.

## 1. It is COMPUTE-BOUND-SHAPED but running at 7.10% of the machine

> **CORRECTED 2026-08-09 by ddm_rr2's independent re-derivation.** This section first said **5.5%**.
> My FLOP denominator covered only the four `TokenBlock`s, **not the full renderer** — it omitted the
> stem/head/embedding path. rr2's independent full-renderer lower bound is **20.6486 GFLOP/fwd-image**
> (mine: 15.8545). Under the same `backward ≈ 2× forward` convention that is **123.8914 GFLOP at B=2**,
> so 120.5976 ms ⇒ **1,027.3 GFLOP/s = 7.10% of the measured 14,471.765 GFLOP/s ceiling.**
> **The 75.40% renderer share and the memory-bandwidth-bound diagnosis are UNAFFECTED** — both rest on
> measured ms, not on the FLOP denominator. Only the 5.5% figure and its scope were wrong.

`step_cost_isolation_bs2.json` — isolation reproduces training exactly
(80.0 ms/pair measured here vs 80.0 ms/pair in the real loop, **ratio 1.000**), so the split
is trustworthy:

| component | ms (B=2) | share |
|---|---|---|
| renderer fwd+bwd | 120.60 | **75.40%** |
| frozen SegNet (by subtraction) | 37.27 | 23.30% |
| R-chain (interp↑ / ste_uint8 / interp↓) | 2.07 | 1.29% |
| dense fp32 GEMM 4096³ (device ceiling) | 9.50 | → **14,472 GFLOP/s** |

**Our #449 finding ("frozen SegNet ≈95% of wall-clock") does NOT transfer to this vehicle.**
It was measured on the WITNESS vehicle with an MLX scorer; here the 66,339-param renderer
costs 3.2× the 8M-param frozen scorer. Borrowed number, correctly refused (m21/L18).

Renderer arithmetic from `TokenBlock` read at source (`semantic_renderer_oracle.py:59-77`):
depthwise 3×3 `groups=96` + pointwise 1×1 + GroupNorm + FiLM + GELU + residual, ×4 blocks
over 196,608 px ⇒ ≈15.9 GFLOP fwd/img, ≈47.6 GFLOP fwd+bwd, **95 GFLOP at B=2**.
95 GFLOP / 120.6 ms = **789 GFLOP/s = 5.5% of the measured 14,472 GFLOP/s ceiling.**

## 2. WHY: 60% of the renderer is ops that do ~no arithmetic

`renderer_op_breakdown_bs2.json`, per block at (2,96,384,512):

| operator | ms | achieved | FLOPs |
|---|---|---|---|
| depthwise 3×3 | 7.01 | 291 GFLOP/s | real, bandwidth-bound (9 MACs/weight) |
| pointwise 1×1 | 7.40 | 2,940 GFLOP/s | real GEMM |
| **GroupNorm** | **16.22** | — | **~zero** |
| GELU | 5.72 | — | ~zero |
| QAT re-param + fwd | 8.04 | — | per-step `fake_quantize`+`functional_call` |

**GroupNorm alone is ~45% of a block and computes nothing.** GroupNorm + GELU ≈ 60%.
A (2,96,384,512) fp32 tensor is 150 MB; GroupNorm's reduce+normalize+backward traverses it
several times, so the renderer is **memory-bandwidth bound, not compute bound**.

That also explains the flat batch curve — per-PAIR time is 0.0850 / 0.0813 / 0.0756 s at
batch 2/8/16 (throughput 11.8→13.2 pairs/s, +12% for 8× batch). There is no idle occupancy
for batching to fill, so **batching cannot convert memory headroom into throughput.**

## 3. BOTH standard levers are MEASURABLY UNAVAILABLE on PyTorch-MPS

`precision_fusion_bs2.json`, one real `TokenBlock` fwd+bwd:

| arm | ms | vs fp32 |
|---|---|---|
| fp32 | 30.15 | 1.00× |
| fp16 autocast | 31.15 | **0.97× (worse)** |
| bf16 autocast | 31.52 | **0.96× (worse)** |
| torch.compile | — | **CRASHES**: `AssertionError: expected size 96==96, stride 196608==1 …` in `aten.convolution_backward` |

- **"Halve the bytes" does not exist through autocast here.** On a bandwidth-bound loop fp16
  should have paid; it did not. Cast insertion adds traffic and MPS kernels do not gain.
  (This ALSO means pp2's "ZERO AMP in PR130's trainer" is not a missing optimization —
  there is nothing to miss on this backend.)
- **The standard fusion path is broken on MPS** for this block's layout.

⇒ On this backend the throughput lever is **ours to build or ours to move**, not a flag.

## 4. What this leaves (each mapped to the operator's steer)

| lever | measured basis | status |
|---|---|---|
| Fused Metal kernel: dw→pw→norm→FiLM→GELU→residual in one pass | kills most of the 60% norm+GELU traffic; torch.compile can't do it (§3) | **our #478 conv suite / #356 megakernel exist for exactly this** — the port is the work |
| Move the trainer to MLX | our fused stack + custom Metal kernels are MLX-side; PR130's trainer is torch. The two have never met | REAL but a port, with an argmax-parity gate owed |
| Drop/replace GroupNorm | 45% of a block for ~0 FLOPs | **architectural** — changes the trained object, needs its own A/B |
| **CPU/ANE offload of the periodic eval** | 24 evals × 19 s = **7.6 min of the 25 min tail = 30%**, pure frozen inference, no gradient needed | **cleanest win, no numerics risk** — separate process, Metal keeps training |
| ANE for the frozen SegNet | 23.3% of the step, fixed inference graph | blocked: we need ∂loss/∂frame and ANE gives no backward. Only via #455/#456 |
| Concurrency | peak RSS **3,397 MiB of 128 GB = 2.7%** | many independent legs can run at once; this is what "saturate" actually looks like given flat per-pair scaling |
| Fixed per-invocation cost | 60.0 s (import + load + one eval) | amortize by longer runs / warm workers, not by faster kernels |

## 5. Honesty bars

- Everything above is wall-clock on Metal. **No score claim.** MPS is never authority.
- fp16/bf16 and any GroupNorm change alter numerics inside a **QAT** loop; adopting either
  requires an argmax-parity gate on the deployed path first. §3/§4 **price** levers; they
  **adopt** nothing.
- The FLOP figures in §1 are DERIVED from the architecture read at source, with the
  formula shown; the ms figures and the 14,472 GFLOP/s ceiling are MEASURED.
- The 30% eval share uses the measured 19 s/n600-eval from the quantized repro receipt and
  the trainer's default eval cadence; if the real cadence differs the share scales with it.

## 6. Receipts

- `step_cost_isolation_bs2.json` — component split + device ceiling + ratio-1.000 control
- `renderer_op_breakdown_bs2.json` — per-operator table
- `precision_fusion_bs2.json` — fp16/bf16/compile arms
- `quantized_repro_receipt.json` — the 19 s/n600 eval figure + semantic-leg reproduction
