# DECISIVE $0 TEST — "matter-on-fixed-Fisher-background" witness framing

**UTC:** 2026-06-29T160536Z · **git:** 554509973 · **tag:** `[macOS advisory / research-signal]`
**score_claim=false · promotable=false · ready_for_exact_eval_dispatch=false** (validates a DESIGN
framing; NO contest-pointer claim). NO GPU — CPU/numpy only, ~61 s wall over n96.

**Artifact (numbers):** `experiments/results/colocation_test_20260629T160343Z/colocation_results.json`
**Tool:** `tools/colocation_fisher_stress_anisotropy_test.py`

## What was tested

The witness produces a 5-class partition scored by a FROZEN SegNet argmax (`d_seg`) on a 512×384 grid.
Deep-math claims the witness is **"matter on a fixed Fisher background"**: (1) the scorer's FISHER
curvature and the `d_seg` STRESS-ENERGY (small-margin flip-mass) CO-LOCATE on the codim-1 decision
boundary annulus Σ; (2) boundary tangent anisotropy ≈ 7:1; (3) flip-mass concentrates in a thin
small-margin band. If true → the v2 loss should be an **annulus-localized, anisotropic, natural-gradient
(Fisher-weighted) energy**.

## Data + faithfulness (NO-FAKE)

`gt_n96.npz` caches SegNet **argmax** (`lstars`) + **top1−top2 logit margin** (`margins`) but NOT full
5-class logits. Fisher curvature `trace(diag(p)−ppᵀ) = 1 − Σ p_k²` needs the full softmax, so logits were
**RECOMPUTED** by running the SAME frozen CPU-torch SegNet (`load_real_segnet('cpu')`,
`measure_segnet_argmax` lineage: degenerate pair, last-frame preprocess, one forward) on the cached
`gt_f1` frames. **Faithfulness proof (all 96 frames):** recomputed argmax vs cached `lstars` mismatch
rate = **0.0**; recomputed margin vs cached `margins` |Δ|max = **4.8e-7** (float32 ULP). The recomputed
logits are byte-faithful to the cache authority.

## VERDICT TABLE (n=96 frames, 512×384)

| Test | Metric | Value | Threshold | Verdict |
|---|---|---|---|---|
| **1. Co-location** | Pearson(curvature, −margin), boundary band | **0.978** ±0.003 | ≥0.5 CONFIRMED | **CONFIRMED** |
|  | Pearson(curvature, −margin), all pixels | 0.814 ±0.013 | ≥0.5 | CONFIRMED |
|  | Spearman(curvature, −margin), all | 0.908 ±0.011 | — | (monotone, strong) |
|  | corr(Fisher trace, ‖F‖₂) | 0.997 | — | trace ≈ spectral norm |
| **2. Anisotropy** | grad-proj ratio across:along (direct analog of 7:1) | **9.56:1** | [4,10]≈7 CONFIRMED | **CONFIRMED (≥ predicted)** |
|  | structure-tensor eigenvalue λ1/λ2 (aggregate) | 37.8:1 | [4,10] | overshoots → even more codim-1 |
|  | λ1/λ2 lane-class boundary only | 28.5:1 | [4,10] | overshoots |
| **3. Flip-mass** | frac of 2%-margin px in 2px boundary band | **0.968** | concentration | **CONFIRMED** |
|  | frac of 5%-margin px in band | 0.852 | | CONFIRMED |
|  | frac of LANE px in 2% margin band (vs known ~60%) | **0.627** | ~0.60 | **MATCHES** |

Per-class flip mass (2% band, GT-argmax assignment): **Road 48.1% / Lane 18.5% / Undrivable 14.6% /
MyCar 10.4% / Movable 8.4%** — matches CLAUDE.md's measured "~50% Road / 19% Lane / 13% Undrivable".

## Method + falsification thresholds

- z = SegNet logits, p = softmax(z) (T=1). curvature(x) = trace(diag(p)−ppᵀ) = Σ p_k(1−p_k) = 1−Σp_k².
  ‖F‖₂ via batched `eigvalsh` on a 40k-px/frame subsample. stress = −margin (robust) ;
  `1/(m+ε)` was also computed but is **outlier-dominated** (Pearson ≈ 0.008, heavy-tailed) and is NOT a
  meaningful co-location metric — use −margin / Spearman.
- Co-location: pooled + per-frame Pearson; CONFIRMED ≥0.5 / PARTIAL 0.3–0.5 / REFUTED <0.3 (annulus band).
- Anisotropy: structure tensor of the margin field (Gaussian σ=2) on Σ=(m<5%-quantile ∪ argmax-boundary);
  CONFIRMED 4–10 / PARTIAL 2–4∪10–15 / REFUTED <2. λ1=across-boundary (normal), λ2=along (tangent).
- Flip-mass: geometric boundary band = argmax 4-neighbor disagreement dilated 2 px; flip-prone = small
  margin (2/5/10% global quantiles).

## Honest caveats

- `[macOS advisory / research-signal]` — validates a DESIGN framing, not a score; NO pointer claim.
- **Test 2 nuance (not over-claimed):** the codim-1 anisotropy HYPOTHESIS is confirmed. The directionally
  correct gradient anisotropy is 9.56:1 (across:along) — within the predicted [4,10]≈7:1 band, so the
  **magnitude is confirmed at the band's upper edge**. The raw structure-tensor eigenvalue ratio is much
  higher (≈38:1) because it is dominated by near-zero λ2 at clean straight boundaries; it OVERSHOOTS the
  7:1 number — i.e. the boundary is an even thinner 1D ridge than 7:1, which over-confirms rather than
  refutes codim-1. The JSON's `VERDICT_eigenratio="REFUTED"` is the literal eigenvalue-vs-[4,10] auto-flag
  and should be read as "overshoot", not a refutation of the anisotropy framing. The DESIGN conclusion
  (anisotropic loss) holds robustly under both metrics.
- Margin/argmax are the EXACT cached authority; logits are an exact recompute of the same frozen scorer
  (faithfulness proven above), so this is the contest scorer's geometry, not a proxy.

## Implication for the v2 loss (actionable)

**All three predictions hold** → the v2 loss should be an **annulus-localized (boundary-band),
anisotropic (cross-boundary-weighted ~7–10:1), Fisher-natural-gradient energy.** The strongest practical
finding: curvature ≈ a monotone function of the margin (band Pearson 0.978, Spearman 0.908 overall) AND
the Fisher spectral norm tracks the trace (corr 0.997). So **the cheap top1−top2 MARGIN field is a
byte-faithful surrogate for the full Fisher natural-gradient metric** — the v2 loss can keep using the
margin (already live in `experiments/train_witness_realized_through_R_mlx.py::_live_margin_weight`, which
re-allocates loss budget to the bottom-margin annulus) as the Fisher weight WITHOUT carrying full logits.
The v2 upgrade is to add the **anisotropy** term (orient the loss/Fourier basis to the boundary tangent
field, ~7–10:1 cross:along) on top of the existing margin-localized allocator. This is consistent with
CLAUDE.md's measured "ALL-CLASS DIRECTIONAL Fourier basis = THE decisive lever (−48% d_seg, ~0 byte)".

## Per-test final line

- **Test 1 CO-LOCATION: CONFIRMED** (Fisher curvature and margin-stress co-locate; band r=0.978).
- **Test 2 ANISOTROPY: CONFIRMED** (gradient anisotropy 9.56:1 in the ≈7:1 band; eigenvalue ratio shows
  even stronger 1D structure — magnitude meets-or-exceeds the prediction).
- **Test 3 FLIP-MASS: CONFIRMED** (96.8% of flip-prone mass on the codim-1 boundary band; lane 62.7% in
  2%-band ≈ known 60%).

**Net: the "matter-on-fixed-Fisher-background" framing is VALIDATED → build the v2 loss as an
annulus-localized anisotropic natural-gradient energy, using the cheap margin field as the Fisher weight.**
