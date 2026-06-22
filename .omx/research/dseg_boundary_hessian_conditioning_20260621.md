# d_seg-boundary Hessian conditioning → the convergence bound, the Muon connection, and the plateau diagnosis (2026-06-21)

**Operator ask (2026-06-21):** derive the condition number κ of the SegNet argmax-boundary Hessian, the
noise-floor convergence bound, turn the B=64 "8× fewer updates" bet into a number, and answer *is the d_seg
plateau conditioning-limited?* **DESIGN/THEORY MEMO** — $0, no training touched. Authority: `[contest-CPU
advisory]`, NON-PROMOTABLE, pointer UNMOVED 0.19110. The live margin measurement (subagent `a688…`, N=24 EMA
shadow, OMP-capped) plugs into §6; thresholds are pre-registered in §5 so the diagnosis is falsifiable.

---

## 1. The d_seg surrogate Hessian is BOUNDARY-DOMINATED (structure)

Decoder θ → frame x(θ) → SegNet logits z ∈ ℝ^{P×5} (P ≈ 197K pixels/frame × 600). Per-pixel GT-class margin
`m_p(θ) = z_p[c_gt] − max_{c≠c_gt} z_p[c]`; pixel correct iff m_p>0, flipped iff m_p<0. The exact
`d_seg = (1/P)Σ_p 1[m_p<0]` is a step (non-diff); training uses a smooth surrogate ℓ(m_p) with ℓ'<0 (pushes m
up) and **ℓ''(m_p) ≥ 0 concentrated near m=0** (the steep part of the sigmoid/hinge).

Gauss-Newton Hessian (j_p ≡ ∂m_p/∂θ):
  **H ≈ (1/P) Σ_p ℓ''(m_p) · j_p j_pᵀ**

H is a sum of rank-1 outer products weighted by ℓ''(m_p). Interior pixels (large |m_p|, flat surrogate)
contribute ℓ''≈0; **only BOUNDARY pixels (small |m_p|) carry curvature.** So the optimization geometry is
entirely set by the thin boundary band — which is exactly why d_seg "concentrates at the SegNet decision
boundaries" (CLAUDE.md). The whole d_seg problem is the conditioning of this boundary Hessian.

## 2. κ from two factors (one of them directly measurable)

κ(H) = λ_max/λ_min over the active (boundary) subspace = **(weight spread) × (geometric factor):**
1. **Margin/weight spread** — ℓ''(m_p) varies across boundary pixels by how close each sits to m=0. A wide
   spread of flipped-pixel |m_p| → wide spread of ℓ'' weights → wide eigenvalue spread → high κ. **This is the
   measurable proxy: the flipped-pixel margin distribution's spread (§6).**
2. **Geometric alignment of {j_p}** — if boundary pixels' margin-gradients are nearly parallel (a contiguous
   contour moving together) the curvature concentrates in few directions; if spread across independent
   directions, higher effective rank. (Harder to measure directly; the margin spread is the tractable handle.)

## 3. The convergence bound (how many steps to resolve the flips)

For a quadratic with Hessian H, GD at LR η is stable only for η < 2/λ_max, and the slowest direction (λ_min)
then converges at rate (1 − 2/κ) per step. To shrink the worst-direction residual by ε:
  **n_steps ≈ (κ/2) · ln(1/ε)**  (the classic condition-number bound).
For the d_seg polish, "resolve a flip" = push m_p from its current negative value past 0. So the steps to clear
the flip set scale **linearly in κ**. Tightly-clustered near-zero margins (low κ) → few steps. Widely-spread
deep margins (high κ) → many steps.

## 4. The MUON connection — orthogonalization IS the conditioning-buster (and this re-confirms B-invariance)

Muon's `polar(M) = UVᵀ` sets every singular value to 1 — it makes the update **isotropic in the spectral
basis**, which is precisely a (whitening) PRECONDITIONER: it flattens the eigenvalue spectrum κ → ~1 for the
directions it covers. So Muon converges in **O(ln(1/ε))** steps, NOT O(κ·ln(1/ε)). **This is why PR95 uses Muon
for the FINAL d_seg-finishing polish (stage 8):** it is the conditioning-buster for the high-κ boundary
geometry that plain AdamW grinds through linearly in κ.

This CONNECTS to the prior batch-scaling result: Muon's value is the spectral flattening (conditioning), and
conditioning is **magnitude-independent** → the same reason its LR is batch-invariant (§ the B=64 spec). Two
derivations, one mechanism: Muon decouples from magnitude, which simultaneously (a) busts κ and (b) makes η
batch-invariant.

## 5. The fewer-updates bet (B=64), refined — and pre-registered diagnosis thresholds

The B=64 risk is "8× fewer updates." But the bound says the risk is **NOT uniform across stages:**
- **Stage 8 (Muon): O(ln 1/ε), κ-busted** → the needed step count is small + κ-independent → 8× fewer Muon
  steps is LIKELY fine. Low risk.
- **Stages 1–7 (AdamW): O(κ·ln 1/ε), κ-LIMITED** → step count scales with κ → fewer updates is RISKIER here,
  concentrated in the big d_seg stage (stage 5 C1a, 9000 ep). **Refinement to the B=64 spec: if extending
  epochs to recover the fewer-updates deficit, extend the AdamW κ-limited stages (esp. stage 5), NOT stage 8.**

**Pre-registered plateau diagnosis (falsifiable; §6 measurement decides):**
- **EASY / not-conditioning-limited** if flipped-pixel **median |margin| < 0.5** AND **κ-proxy (p90/p10 |margin|)
  < ~5**: the plateau is LR/temperature/under-training, not geometry → the live run's later stages (esp. the
  Muon polish + the σ/LR anneal) WILL break it. GOOD for stage-5/8.
- **HARD / conditioning-or-capacity-limited** if **median |margin| > 2.0** OR **κ-proxy > ~20** with a deep
  tail: the residual flips are deep-margin (the decoder can't put those pixels on the right side) → more epochs
  won't help; need capacity (base_ch↑) or a different lever (sub-pixel boundary / sidecar). BAD — re-route.
- Intermediate → mixed; the fraction of flips with |margin|<0.5 (near-boundary, fixable) vs >2.0 (deep) is the
  actionable split.

## 6. MEASUREMENT (live EMA-shadow, N=24, CPU OMP=2, 13.8s, read-only — run untouched)
Probe `a688…`; render path verified (subset d_seg 0.002049 ≈ run's 0.002278). JSON:
`.omx/research/dseg_margin_distribution_capstone_ema_shadow_n24_20260621.json` (commit `97605526c`).

Flipped-pixel |GT-class margin| (n=9,667): **median 0.320, p10 0.050, p50 0.320, p90 0.952, max 5.095**.
Histogram (width 0.51): `[6488, 2407, 598, 135, 21, 10, 2, 2, 2, 2]` — front-loaded at the boundary.
- **|margin| < 0.5 (near-boundary, fixable): 66.5%** · 0.5–2.0: 33.0% · **> 2.0 (deep): 0.44%** · <1.0: 91.5%
- **κ-proxy (p90/p10): 18.98**
- Full-pixel boundary band |top1−top2|<0.5: **0.43%** of all pixels (the SegNet field is confident almost
  everywhere; flips concentrate in that sparse band, flip-rate 0.20% ≈ half the band).

### 6.1 Verdict (read HONESTLY against §5's pre-registered thresholds — it's the intermediate case)
The thresholds SPLIT, exactly as §5 anticipated for the mixed case:
- **median |margin| 0.320 < 0.5 → meets the EASY median criterion.** ✓
- **κ-proxy 18.98 → fails EASY (<5), just under HARD (>20).** ✗ for clean-easy.
- **deep-tail (>2.0) = 0.44% → the HARD deep-tail condition is NOT met.** ✗ for hard.

Decompose the high κ to resolve the split: κ=p90/p10=0.952/0.050. The wide spread is driven by the **LOW end**
(p10=0.05 — a huge mass of flips lost by ≈nothing), NOT a deep high end (p90 still < 1.0; only 0.44% > 2.0).
So the residual is **conditioning-WIDE but SHALLOW.** The honest verdict:
- **NOT capacity-limited.** 99.6% of flips are < 2.0 logit, 91.5% < 1.0, the deep tail is negligible. The
  base_ch decoder represents the frame well enough to put almost every pixel within ~1 logit of correct →
  **capacity is NOT the binding lever; more base_ch won't unlock the plateau.**
- **IS optimizer-conditioning-limited.** κ≈19 (wide, from the near-zero-margin mass) → plain AdamW grinds
  through it at O(κ·ln 1/ε) → THIS is the "plateau" feel: many thin-margin flips, each a small nudge, but
  spread enough that κ-limited AdamW resolves them slowly.
- **Muon-ADDRESSABLE.** The κ-buster (§4) converges O(ln 1/ε) regardless of κ=19 → **stage-8 Muon is exactly
  the lever to break this plateau.** The live run reaching stage 8 with Muon working is the expected breaker.
  GOOD news for the decisive stage-5→8 verdict.

### 6.2 Consequences (sharpened by the measurement)
1. **The B=64 AdamW-stage fewer-updates risk is REAL and quantified:** κ≈19 is substantial, and the AdamW
   stages are O(κ·ln 1/ε) → 8× fewer AdamW updates genuinely slows the κ-limited grind. **Extend stage 5
   (C1a, the big d_seg AdamW stage) at B=64; stage 8 (Muon, κ-busted) is fine as-is.** Confirms §5's refinement.
2. **A boundary-targeted SIDECAR is a CONDITIONAL win at the break-even knife-edge (CORRECTED — see §6.3).**
   The residual is shallow + sparse + near-boundary (66.5% of flips <0.5 logit, 0.43%-of-pixels band). But the
   $0 byte-cost arithmetic (§6.3) shows the sidecar is net-positive ONLY if the per-flip cost beats the
   break-even **1.273 B/flip** — and iid LOCATION alone is already **1.125 B/flip**, with Lever-D's MEASURED
   all-in cost ~1.27 = break-even (survival overhead eats the headroom). The shallow margins make the
   *correction* nearly free but do NOT cheapen the *location*; the win hinges on survival-aware **contour +
   temporal-delta** location coding (exploiting the band clustering + smooth boundary motion). So the PRIMARY
   lever for this residual is **TRAINING (Muon stage 8, 0 bytes, §4)**; the sidecar is a conditional fallback
   whose viability needs a survival-aware location-coding measurement (task #137). [This corrects the earlier
   over-enthusiastic "green-light" — the arithmetic says knife-edge, not free win.]
3. **The plateau is NOT a wall — it's a κ-limited grind with a FREE training buster (Muon) and a knife-edge
   sidecar fallback.** The training lever (Muon stage 8, 0 bytes) is primary; the codec lever (boundary
   sidecar) is conditional (§6.3).

### 6.3 Boundary-sidecar byte economics — the break-even knife-edge ($0 arithmetic)
Total scored pixels N_px = 600·512·384 = **117,964,800**. The marginals:
- ∂S per flip removed = 100/N_px = **8.477e-7**;  ∂S per byte = 25/37,545,489 = **6.659e-7**.
- **Break-even = 8.477e-7 / 6.659e-7 = 1.273 B/flip** — the sidecar helps iff each fixed flip costs < 1.273 B.
  (Matches Lever-D's independently-MEASURED "below 1.27 B/flip" anchor — the all-in cost there was ~1.27 =
  break-even, because survival overhead — corrections erased by the uint8 round-trip + SegNet forward — eats the
  headroom.)

To reach the sub-0.15 d_seg target (0.000322) from 0.002278: fix **230,739 flips (86%)**. iid LOCATION-only
entropy = 230,739·log₂(N_px/230,739) = **259.5 KB = 1.125 B/flip** — already within 0.148 B/flip of break-even,
leaving almost nothing for the correction. Net ΔS at various per-flip costs:

| per-flip cost | bytes | seg_save | rate_add | net ΔS | source of headroom |
|---|---|---|---|---|---|
| 1.273 (Lever-D measured) | 294 KB | −0.196 | +0.196 | **0.000** | none (break-even) |
| 1.125 (iid location, ~free corr) | 260 KB | −0.196 | +0.173 | **−0.023** | sparse-mask entropy floor |
| 1.0 | 231 KB | −0.196 | +0.154 | −0.042 | contour beats iid |
| 0.8 | 185 KB | −0.196 | +0.123 | −0.073 | + 0.43%-band clustering |
| 0.5 | 115 KB | −0.196 | +0.077 | **−0.119** | + temporal-delta (boundaries move smoothly) |

**The deep structural fact:** even the *location* of a flip (1.125 B) nearly costs more than the flip is worth
(break-even 1.273 B) — the rate-axis-must-be-d_seg-neutral law at its sharpest. The ONLY headroom is the flips'
spatial+temporal STRUCTURE (contour + cross-frame coherence), NOT their shallowness (shallowness only frees the
~0.15 B/flip correction budget + helps survival). **Verdict: conditional win — net-positive ONLY if
survival-aware contour+temporal location coding beats ~1.1 B/flip; the measured anchor is break-even. Primary
lever = Muon training (free); sidecar = fallback pending the task-#137 survival-aware location measurement.**
Arithmetic reproducible inline; no artifact written (pure closed-form).

## 7. Consequences (results→intelligence, regardless of the §6 number)
1. **The d_seg problem IS a conditioning problem** — the Hessian is boundary-dominated, κ set by the margin
   spread. This reframes "the d_seg wall" as "the boundary κ," a measurable, attackable quantity (not a mystery).
2. **Muon is the principled d_seg-finisher** (κ-buster), and its batch-invariance + its conditioning role are
   the SAME magnitude-decoupling property. The capstone should keep Muon as the final stage and NOT swap it.
3. **B=64 fewer-updates risk is AdamW-stage-concentrated** → extend stage 5, not stage 8, if needed.
4. **Capstone lever implication:** if §6 says HARD (deep margins), the capstone needs capacity or a boundary
   sidecar, NOT just more training — directly informs the base_ch=24-vs-higher + sidecar decision.

## NO-FAKE ledger
- DERIVED (this memo): the GN boundary-Hessian structure; n ≈ (κ/2)ln(1/ε); Muon = spectral preconditioner →
  O(ln 1/ε) + the B-invariance connection; the AdamW-concentrated fewer-updates risk.
- PENDING MEASUREMENT (§6): the live flipped-pixel margin distribution → the actual κ-proxy + the plateau verdict.
- NOT claimed: no score moved; pointer UNMOVED 0.19110; κ here is the boundary-Hessian conditioning proxy from
  margins, not a full eigendecomposition (the geometric factor §2.2 is bounded, not measured).

## Cross-references
- `capstone_batch_size_fixed_point_B64_launch_spec_20260621.md` §2.1 (Muon B-invariance — same magnitude-decoupling mechanism).
- `throughput_floor_latency_bound_bs8_scorer_20260621.md` (the bs=8 floor the fewer-updates bet trades against).
- `lensA_dseg_optimal_loss_geometry_ce_vs_margin_hinge_20260619.md` (the margin-hinge surrogate ℓ(m) this analyzes).
- `tac.margin_saliency_map` / `tac.substrates.d1_segnet_margin_polytope.margin_map.compute_logit_margin_map` (the margin machinery).
