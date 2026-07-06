# Chernoff information vs raw margin — $0 advisory geometry probe (2026-07-06)

**Status:** ADVISORY GEOMETRY ONLY. `score_claim=false`, `promotion_eligible=false`,
pointer UNMOVED. No training, no paid launch, no MLX/MPS score. CPU-torch frozen upstream
SegNet is the authority.

- git: `002257c6655869721658c05341497e01b900c494`
- seed: 0 · scorer: `upstream/models/segnet.safetensors` (frozen SegNet, `tu-efficientnet_b2`, 5-class)
- forward faithful to `upstream/modules.py` SegNet.preprocess_input (last-frame → bilinear
  interp to 512×384 → raw 0-255 float → 5-class logits); argmax reproduced GT cache on
  **597/600** frames; boundary-pixel margin mean |Δ| vs cache = **2.1e-6** (authority-faithful).
- tool: `tools/chernoff_vs_margin_probe.py` · artifacts:
  `experiments/results/chernoff_probe_20260706T125835Z/{result.json,chernoff_vs_margin.png}`
- primary cache: `gt_n600.npz` (n600, 2,551,382 boundary/separatrix pixels)
- survival cache: `gt_strided_n200.npz` + `wave0_residual_id_20260628/baseline_flips.npz`
  (lstars align frac = 1.0; witness-render flip label; **n200 advisory**)

## The deep-math being tested
Frank Nielsen info-geometry: the principled bits-to-flip quantity is the multi-class
**Chernoff information** from the FULL 5-class softmax, which differs from raw margin
(top1−top2) ONLY where 3+ classes compete (lane / triple-junction). Model (documented,
honest first-order surrogate): equal-variance Gaussian-logit-noise ⇒ pairwise Chernoff
exponent `C_j = (z_top1 − z_j)²/8`; aggregate flip-resistance
`chernoff = −log( Σ_{j≠top1} exp(−C_j) )` (sees ALL runner-ups, not just the nearest).
Junction pixel := ≥3 classes with softmax prob ≥ 0.15.

## Results (n600 boundary pixels)

| quantity | value |
|---|---|
| **Spearman(raw_margin, chernoff)** | **0.8206** |
| baseline class-1 (lane) frac of boundary | 0.203 |
| baseline junction frac of boundary | 0.016 (1.6%) |
| top-decile-disagreement class-1 frac | 0.075 → **concentration 0.37×** (DEPLETED) |
| top-decile-disagreement junction frac | 0.132 → **concentration 8.27×** (ENRICHED) |

### Survival-predictive test [n200 advisory, witness-render flip label]
852,013 boundary px · 121,719 flips (boundary flip rate 14.3%):

| predictor | AUC(predicts flip) |
|---|---|
| raw margin | **0.7774** |
| chernoff | 0.7303 |
| Δ (chernoff − margin) | **−0.0471** (chernoff WORSE) |

## Interpretation
1. **Re-ranking is real but not degenerate-identical** (Spearman 0.82, not ~1.0) — Chernoff
   does move ranks vs top-2 margin.
2. **The re-ranking lands in the WRONG place for our residual.** Disagreement pixels are
   8.3× enriched on triple-**junction** pixels (mechanistically expected: that is exactly
   where 3+ classes compete) but 2.7× **DEPLETED** on class-1 **lane** pixels. Our measured
   d_seg residual lives on lane/along-tangent pixels (~19% lane flip mass; the lane-dash
   long-tail), NOT on junctions — and junctions are only 1.6% of the separatrix.
3. **Where Chernoff differs, it predicts the actual flip WORSE than plain margin**
   (AUC 0.730 vs 0.777, −0.047). So the multi-class aggregation does not sharpen the
   flip-resistance ranking that Lever-D / margin-saliency care about; it dilutes it.

## VERDICT — DEGENERATE-EQUIVALENT-OR-WORSE; close the lead
Multi-class Chernoff does **NOT** add actionable signal beyond raw margin for the
lane/along-tangent pixels where our d_seg residual actually lives. Raw margin (already the
Fisher-metric surrogate, Pearson 0.978) remains the correct, simpler, and empirically
**superior** flip-resistance quantity. The Chernoff refinement re-ranks only the rare (1.6%)
junction pixels and does so in the wrong direction for flip prediction.

**ONE concrete next action:** KEEP the raw margin field as the margin-saliency pixel-weight
and the Lever-D b/flip prioritization signal — do **not** swap in a multi-class-Chernoff
weight. Close this info-geometry lead. (If junction pixels ever become a distinct residual
component, revisit — but margin already out-predicts Chernoff even there.)

## Honesty caveats
- Chernoff uses an equal-variance Gaussian-logit-noise surrogate for the true (unknown)
  SegNet logit perturbation law — the defensible first-order form, not a measured noise model.
- Survival test is the witness-**render** flip label (a real per-pixel d_seg-error outcome),
  a proxy for Lever-D parseback flip-survival, on n200 strided frames → tagged advisory.
- No pointer moved. Advisory geometry only.
