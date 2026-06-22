# Cross-pair latent dedup — information bound + live measurement: a SMALL d_seg-neutral lever, NOT "the biggest" (2026-06-21)

**Operator ask (2026-06-21):** derive the cross-pair latent dedup information bound (the score-lowering rate
lever, task #106 D1 "UNBUILT, biggest"). **THEORY + $0 MEASUREMENT** (latents are 600×28, tiny → no rendering,
no contention with the live run). Authority: `[contest-CPU advisory]`, NON-PROMOTABLE, pointer UNMOVED 0.19110.

---

## 1. The bound (Gauss-Markov)
Store 600 per-pair 28-d latents {z_i}. Independent storage costs `R_indep = 600·H(z)`. The minimum is the joint
entropy `H(z_1..600)`; the dedup saving is the total redundancy `Σ_i I(z_i; z_{<i})`. First-order (AR(1))
model `z_i = A z_{i-1} + ε`:
  **dedup saving ≈ (600−1)·I(z_i;z_{i-1}) = (599)·½·log₂(det Σ_z / det Σ_ε) bits**
(Σ_z = marginal 28×28 covariance, Σ_ε = AR(1) innovation covariance). Three redundancy sources: temporal
(cross-pair, the AR term), cross-dimension (off-diagonal Σ_z → KLT), and global low-rank (SVD of the 600×28
matrix). Byte→S: ΔS = −25·bytes/37,545,489 (d_seg-NEUTRAL — pure rate axis, no distortion risk).

## 2. Live measurement (EMA latents, stage-2 ep~5675, float64, $0)
- **Per-dim lag-1 autocorrelation: median ρ = 0.66** (p10 0.29, p90 0.84) → real temporal correlation (adjacent
  dashcam pairs are similar), but far from ρ→1.
- **AR(1) temporal MI ≈ 20.4 bits/step → max first-order dedup ≈ 1.53 KB** vs INDEPENDENT storage → ΔS ≈ **−0.0010**.
- **Cross-dim (KLT) redundancy: ~0.14 KB** → ΔS ≈ −0.0001 (the 28 dims are nearly decorrelated already).
- **Global low-rank: SVD eff-rank r90=23/28, r99=28/28, top-5 energy 0.31** → **near-FULL-rank → essentially NO
  low-rank/global redundancy.** The 28 dims are well-used; there is no hidden low-dim manifold to factor.

## 3. The honest verdict — SMALL, and mostly ALREADY CAPTURED (retires the "biggest lever" framing)
1. **Ceiling is ~1.5 KB** (first-order temporal, vs independent) → ΔS ≈ −0.001. Cross-dim + low-rank add ≈ 0.
2. **The frontier ALREADY temporal-delta-codes the latents** (CLAUDE.md L25: `lat[i]=lat[i-1]+delta[i]`) +
   raw-LZMA (L24, blob = 15.4 KB). Naive delta assumes the AR coefficient A=1; the optimal at ρ=0.66 is A=0.66.
   The gap between naive-delta and optimal-AR(1): delta-innovation var = 2σ²(1−ρ)=0.68σ² vs AR(1)=σ²(1−ρ²)=0.564σ²
   → optimal saves only ½log₂(0.68/0.564)=0.135 bits/step → **599·0.135/8 ≈ 10 bytes.** Negligible.
3. **→ The ADDITIONAL dedup beyond the existing delta-coding is tens of bytes to maybe ~0.5 KB (ΔS ~ −0.0001
   to −0.0003) — effectively negligible.** The near-full-rank result kills the global-dedup hope.

**Conclusion: latent dedup is NOT "the biggest unbuilt rate lever" — it is a near-exhausted minor one.** The
latents are only ~15 KB of a ~130–177 KB archive, they're near-full-rank (no global structure), and the
frontier already delta-codes the temporal correlation. **The rate actually lives in the DECODER WEIGHTS (~94%
of the archive)** — the score-aware weight re-quant (WRQ) / weight-entropy / per-tensor bit-allocation levers
(tasks #69/#154) are where rate-axis bytes are, NOT the latents. Re-point #106's "biggest" framing accordingly.

## 4. The one caveat that keeps a door open (training, not codec)
The ρ=0.66 is the NATURAL temporal structure. A **latent-structure-inducing regularizer** (task #110) could
TRAIN the latents to be more temporally predictable (push ρ↑) or low-rank, raising the dedup ceiling beyond the
natural 1.5 KB. But that is a TRAINING lever that changes the latents (and possibly the score), not a pure
codec dedup — and its EV is bounded by the same ~15 KB latent-blob ceiling. Modest even if it works.

## NO-FAKE ledger
- MEASURED (live EMA latents, $0): ρ=0.66 median lag-1; AR(1) MI 20.4 bits/step → 1.53 KB max first-order
  dedup; SVD near-full-rank (r99=28/28); cross-dim ~0.14 KB.
- DERIVED: the Gauss-Markov bound; the naive-delta-vs-optimal-AR(1) gap ≈ 10 bytes.
- HONEST CORRECTION: latent dedup is a small/near-exhausted lever (ΔS ~ −0.0001 to −0.001 ceiling), NOT the
  "biggest" — that framing (task #106) is retired; decoder-weight rate is the real axis.
- NOT claimed: no score moved; pointer UNMOVED 0.19110; latents measured mid-training (ρ unlikely to shift much —
  it's a scene property — but the snapshot caveat stands).

## Cross-references
- `dseg_boundary_hessian_conditioning_20260621.md` (the d_seg side; §6.3 boundary-sidecar economics).
- tasks #69 (score-aware weight re-quant) / #154 (weight-entropy) — the ACTUAL rate axis (decoder weights).
- CLAUDE.md L24/L25 (the frontier's existing raw-LZMA + temporal-delta latent coding this measures the headroom over).
