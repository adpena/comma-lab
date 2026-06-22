# Decoder-weight rate axis: entropy-coding EXHAUSTED, quant BLOCKED by the shallow boundary — and the unifying sweep synthesis (2026-06-21)

**Operator ask (2026-06-21):** the decoder-weight rate axis (the real 94% of the archive) — WRQ / weight-entropy.
**THEORY + $0 MEASUREMENT** (decoder state_dict, ~83K params, no rendering, no contention). Authority:
`[contest-CPU advisory]`, NON-PROMOTABLE, pointer UNMOVED 0.19110.

---

## 1. Measurement (live bc20 EMA-shadow decoder, int8, $0)
- 30 tensors, **83,422 params**. Uniform int8 = 83.4 KB.
- **Marginal int8 symbol entropy H(W) = 6.884 bits/param** → order-0 floor = 71.8 KB.
- **brotli-q11 ACTUAL = 71.9 KB = 6.891 bits/param** → brotli is **AT the marginal entropy floor** (gap 0.007
  bits/param / 0.07 KB). Decoder rate term = 25·71,854/37.5M = **0.0478**.
- **Arithmetic/range-coder headroom over brotli ≤ 0.07 KB → ΔS ≤ −0.00005. EXHAUSTED.**

## 2. The three decoder-rate levers, ranked by what the data says
1. **Entropy coding beyond brotli — EXHAUSTED.** brotli-q11 ≈ H(W) marginal (measured). A custom
   arithmetic/range coder gains ≤0.07 KB. The PR95 L21–L32 stack (per-tensor byte-maps, split brotli streams,
   canonical Huffman) already extracts the codeable redundancy. ✗
2. **Uniform sub-8-bit quant — BLOCKED by d_seg.** Lowering the 6.88 bits/param (int8→int5) was TESTED: it
   caps **S~0.49** (d_seg plateaus ~0.0035, 6× above floor; `feedback_frontier_int5...path_b_caps`). The reason
   is §4: the shallow d_seg boundary can't tolerate the quant noise. ✗
3. **WRQ (score-aware per-weight realloc) — modest + constrained.** The only principled remaining quant lever.
   Reverse-waterfill `b_i* = ½log₂(s_iσ_i/√θ)` saves rate ∝ the log-variance of per-weight score-sensitivity
   s_i. But the measured sensitivity is **~5.5× flat** (task #121) — only 2.5 bits of dynamic range → modest
   reallocation win — AND it must spend MORE bits on the boundary-controlling weights (§4), not fewer. Bounded.
4. **Structural (weight-tie / low-rank / prune) — the REAL rate lever** (capstone L1 weight-tied upsample
   blocks). Trades params for inflate-compute; cuts rate by reducing PARAM COUNT, not bits/param. This is where
   decoder rate actually moves — and it's an ARCHITECTURE choice, not a post-hoc codec.

## 3. The unifying synthesis — the shallow d_seg boundary is the load-bearing structural fact
Today's sweep bounded every post-hoc codec lever, and they all trace to ONE fact: **the d_seg residual is
shallow** (66.5% of flips lost by <0.5 logit, conditioning memo §6). That shallowness:

**BLOCKS the post-hoc rate axis three ways:**
- **Per-flip d_seg sidecar → break-even** (1.273 B/flip; even LOCATING a flip ≈ its worth).
- **Sub-8-bit uniform quant → hurts d_seg** (int5 caps S~0.49 — shallow flips are quant-fragile; small noise
  pushes the <0.5-logit pixels over the edge).
- **Entropy coding → already at the floor** (brotli ≈ H(W)).
- (Latent dedup → small/near-full-rank, the one not boundary-driven — separate near-exhaustion.)

**ENABLES the d_seg axis one way:**
- **Training (Muon κ-buster, stage 8, 0 bytes) → fixes the shallow flips for free** (a tiny nudge each; the
  conditioning analysis showed Muon converges O(ln 1/ε) regardless of κ≈19).

→ **The score moves on TRAINING (Muon, free, d_seg) and STRUCTURAL architecture (weight-tie/low-rank, rate) —
NOT on post-hoc codec levers (entropy / quant / sidecar / latent-dedup), which are exhausted or boundary-blocked.**
The rate term (0.0478 for the bc20 decoder) is at the entropy floor; cutting it further needs fewer PARAMS
(structural), and the conditioning "NOT capacity-limited" finding says there IS spare capacity to shrink — BUT
the shrink must PRESERVE the boundary representation (tie/share only the d_seg-irrelevant weights), because the
shallow boundary is quant-/perturbation-fragile.

## 4. Why shallow ⇒ quant-fragile (the int5-cap mechanism, derived)
A flipped pixel at margin m_p (|m_p|<0.5 for 66.5% of flips) is corrected by a small weight change. Quantizing
weights to b bits injects per-weight noise ε ∝ σ·2^{−b}, which propagates to a logit perturbation δz ≈ Σ_i
(∂z/∂w_i)·ε_i. When the boundary margin is small (<0.5) and δz is comparable, the argmax flips — so aggressive
quantization (int5: 2^{−5} step vs int8: 2^{−8}, ~8× more noise) flips the shallow-margin pixels → d_seg rises.
This is the SAME shallowness that makes the residual training-fixable: a small change either way moves it. So
the boundary weights need PROTECTION (more bits, the WRQ high-bit set), and uniform sub-8-bit is structurally
wrong. **WRQ is the correct frame for any decoder bit reduction — but score-aware, boundary-protecting, and
bounded by the flat sensitivity.**

## NO-FAKE ledger
- MEASURED ($0): decoder int8 H(W)=6.884, brotli-q11=6.891 bits/param (at floor); rate term 0.0478; per-tensor
  scale spread 9.7×.
- REASONED: WRQ saving ∝ log-var(sensitivity), bounded by the measured ~5.5× flat sensitivity; the int5-cap
  mechanism (shallow boundary → quant-fragile).
- HONEST VERDICT: decoder entropy coding EXHAUSTED, uniform sub-8-bit BLOCKED, WRQ modest+constrained,
  structural (weight-tie) is the real rate lever. The post-hoc codec rate axis is largely closed.
- NOT claimed: no score moved; pointer UNMOVED 0.19110; sensitivity 5.5× is a prior anchor (task #121), not
  re-measured here (fresh ∂S/∂w needs scorer backprop = render contention, deferred).

## Cross-references
- `dseg_boundary_hessian_conditioning_20260621.md` (the shallow boundary + Muon κ-buster; §6.3 sidecar economics).
- `latent_dedup_information_bound_20260621.md` (the latent rate axis — small/near-full-rank).
- `feedback_frontier_int5_score_aware_qat_finetune_path_b_caps_20260618.md` (the int5 S~0.49 cap this explains).
- tasks #69 (WRQ) / #154 (weight-entropy) / capstone L1 weight-tie (`optimal_capstone_vehicle_spec_20260611.md`).
