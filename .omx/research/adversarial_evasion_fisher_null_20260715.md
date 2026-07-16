# Adversarial / detection-evasion (UNIWARD dual): the Fisher-flat interior is ALREADY null — evasion is pre-exhausted, the residual is boundary-structural

**Date:** 2026-07-15/16 UTC · **Operator P0:** "pursue all as p0" — the ADVERSARIAL /
DETECTION-EVASION facet of the projection/preimage unification (Yousfi / UNIWARD, the DUAL of
projection). **Authority:** `[macOS-CPU advisory]` — cached frozen-SegNet per-pixel margin field
(`gt_n96.npz` margins/lstars), which reproduces the frozen SegNet argmax BIT-EXACTLY (0 mismatches
/ 96 frames, `segnet_recursive_fractal_factorization_20260715`). `research_only=true`;
`score_claim=false`; **pointer 0.19108 UNMOVED (MEANS).**

**Frame.** The witness = PROJECTION onto the correct argmax cell. Some necessary flips are
uint8-UNREACHABLE (the realization-limited sub-LSB tail, `segnet_head_rank4_linear_flipdist_v1`).
For that irreducible residual projection fails; the DUAL strategy is EVASION — do not try to flip
where unreachable, instead PLACE the unavoidable d_seg cost where the detector (SegNet) is
Fisher-FLAT (large margin, low sensitivity), so the same visual error costs less DETECTION. The
scorer IS a detector; d_seg IS its detection rate; the margin field IS the Fisher surrogate
(`tr g = ½·sech²(m/2)`, curvature↔(−margin) Pearson 0.978, `tac.information_geometry.optimal_metric`).

**Probe (cached-only, standalone):** `tools/adversarial_evasion_fisher_null_probe.py` → artifact
`experiments/results/adversarial_evasion_fisher_null_20260715/evasion_probe.json`
(sha256 `ed0b9a98fc9c66b071cee512206bd7c865f71c1f1817d8d93c4c281826f0458c`; dir gitignored, deterministic rerun).
No live trainer, no SegNet forward: reads the cached margin field + the closed-form head gain anchor.

---

## The one-line verdict

**For THIS scorer the detection-evasion dual has ~ZERO independent headroom: the Fisher-flat
interior is ALREADY null (no realistic render error flips a high-margin pixel), so there is nowhere
to relocate error that isn't already hidden. 100% of the modeled d_seg lives on the boundary
annulus. The residual there splits into a genuinely-paid, realization-irreducible sub-LSB tail
(d_seg ≈ 0.0008, Lane-dominated — projection CANNOT fix it) and a precision-reachable band that is
closed by SHARPER BOUNDARY PROJECTION (lower render error), NOT by UNIWARD relocation. Evasion is
pre-exhausted by the scorer's own margin geometry.**

**Evadable-by-relocation fraction (route error into the Fisher-null interior): ≈ 0.**
The only surviving UNIWARD-shaped lever is the CONDITIONAL one (§4): IF the witness's render error
is boundary-CONCENTRATED (Gibbs/ringing, the L1 spatial dual of erasure), then suppressing that
boundary-excess IS msal_uni — but its ceiling is the boundary-excess over uniform, not the interior.

---

## §A. The detectability landscape (MEASURED, exact from cached margins + the sech² law)

Per-pixel Fisher trace `F_p = ½ sech²(m_p/2)` = detectability per unit render perturbation.

| quantity | value |
|---|---|
| margin median | **5.82 logits** (most pixels DEEP in the flat interior) |
| margin p1 / p4.7 / p10 | 0.377 / **2.013** / 3.99 |
| annulus (m<2.0) area | **4.67%** (≈ the #333 ~4.7% annulus) holding **62.4%** of Fisher mass |
| m<0.5 | 1.32% area / 22.1% Fisher mass |
| m<0.2 | 0.53% area / 9.1% Fisher mass |
| m<0.05 | 0.13% area / 2.3% Fisher mass (**17× concentration**) |
| area holding 50% / 90% / 97% of Fisher mass | 3.3% / 32% / 66% |

The interior is not merely large-margin, it is **quadratically protected**: `sech²(m/2)` decays
exponentially, so a pixel at m=2 has Fisher trace ~7% of a boundary pixel's, and at m=5.8 (the
median) it is ~10⁻³. This is why (§C) the interior carries no d_seg.

## §B. The irreducible realization-floor residual (DERIVED scale, quantization-only)

Even a PERFECT continuous render must emit uint8 → a half-LSB quantization jitter (RMS 0.289 in
0-255 units). Pulled through the closed-form head gain (anchor: median Road-Lane boundary margin
0.516 flips at first-order input-L2 8.8 over the frame ⟹ `‖∇m‖ = 0.0586` at the anchor,
`‖∇m_p‖ ≈ 0.0606·sech(m_p/2)`), the quantization margin-jitter RMS ≈ `0.289·‖∇m_p‖`. A pixel is
realization-lost when its margin falls below that jitter:

| threshold | d_seg floor rate | Lane | Movable | Road | Undrivable | MyCar |
|---|---|---|---|---|---|---|
| 1σ | **0.000465** | **0.0166** | 0.00272 | 0.00091 | 0.00013 | 0.00020 |
| 2σ | **0.000929** | **0.0331** | 0.00545 | 0.00185 | 0.00026 | 0.00038 |

**The irreducible floor is ~0.0005–0.0009 d_seg and is LANE-DOMINATED (Lane 18–36× the bulk
classes).** Contextualized against the current witness d_seg ≈ 0.005 (mod32cap), ~10–20% of the
current gap is genuinely realization-irreducible — projection cannot reach it; it is the boundary
itself at sub-LSB margin. Lane is the binding class exactly as in every other lens (rank-4 head Lane
normals largest; stride-2 skip 77% Lane; #333 annulus lane orbit).

## §C. The UNIWARD evasion counterfactual (DERIVED, first-order, margin-blind-error model)

Detectability COST density `cost(m) = (m/G)·cosh(m/2)` = minimal aligned render-L2 to flip a pixel
of margin m. It grows ~`m·e^{m/2}` — **flipping the annulus (m→0) is free (cost→0 = maximally
detectable); flipping the interior costs exponentially (cost→∞ = evasion-safe).** Sweep a naive
per-element render-error RMS ρ (margin-blind, uniform = MSE outcome):

| ρ (0-255 RMS) | naive d_seg | UNIWARD floor (intrinsic annulus) | interior-leak (m≥2.0) |
|---|---|---|---|
| 0.25 | 0.000403 | 0.000403 | **0** |
| 0.5 | 0.000803 | 0.000803 | **0** |
| 1.0 | 0.001614 | 0.000803 | **0** |
| 2.0 | 0.003228 | 0.000803 | **0** |
| 4.0 | 0.006409 | 0.000803 | **0** |

**LOAD-BEARING MEASURED RESULT: `interior_leak = 0` at EVERY ρ up to 4 LSB RMS.** No render error
short of catastrophic (ρ ≈ 51 LSB to flip an m=2 pixel) puts a single flip in the Fisher-flat
interior. **100% of d_seg is on the boundary annulus** — even stronger than #333's 97%. This result
does NOT depend on the gain model: it is a direct consequence of `sech²(m/2)→0`, i.e. exact from the
cached margin distribution.

The apparent "evadable fraction" (0 at ρ≤0.5, 50% at ρ=1, 87% at ρ=4) is the part of the naive flips
sitting ABOVE the deepest intrinsic-annulus floor (m ∈ ~[0.03, ρ·gain]). **These are NOT relocatable
to the interior** — they are still low-margin NEAR-BOUNDARY pixels. They are closed by RENDERING THE
BOUNDARY MORE PRECISELY (reducing ρ), i.e. by PROJECTION, not by evasion. As ρ→0 (a better witness)
they vanish and only the §B intrinsic floor (~0.0008, Lane) remains.

## §4. What survives of the evasion lever (the CONDITIONAL, scoped)

The §C model assumes margin-BLIND render error. The ONE way UNIWARD keeps headroom: if the witness's
error is boundary-CONCENTRATED — Gibbs ringing (the L1 spatial dual of lane-erasure) dumps error ON
the annulus — then the effective ρ AT the boundary exceeds the frame-average ρ, and suppressing that
boundary-excess IS the msal_uni / UNIWARD lever. Its ceiling is the **boundary-excess over uniform**,
NOT the (already-null) interior. This is unmeasurable from cached data (no witness render loaded); it
is the owed n600 measurement: correlate the witness's per-pixel render error with the margin field.
If error is ~uniform, the lever's headroom ≈ 0; if anti-correlated (Gibbs), headroom = the excess.

## §5. Routing to levers (V9·CGauge)

1. **DO NOT build a UNIWARD/msal_uni amplitude-RELOCATION lever expecting independent d_seg.** Its
   interior target is null (interior_leak=0). The scorer's sech²(m/2) geometry pre-exhausts the
   evasion dual. (Directly bounds `msal_uni` LEVER-4's EV — sharpens the "Lever-4 msal_uni texture
   proxy INERT vs through-R" finding L76: not just proxy-inert, the *target region* is empty.)
2. **The residual is 100% boundary-structural**, so the productive levers are the PROJECTION carriers:
   (a) sharper boundary render (reduce ρ — the §C precision-reachable band); (b) the §B
   realization-irreducible sub-LSB tail (Lane ~0.0008) which is uint8-unreachable and needs GEOMETRY /
   PHASE placement (analytic lane band L71 d_seg 0.00087; ξ-transport / appearance-phase endgame L85/L86)
   to put the boundary sub-pixel-correctly rather than by amplitude. Evasion ≠ the tool; sub-pixel
   geometry IS.
3. **The ONLY admissible UNIWARD variant is CONDITIONAL (§4):** a margin-weighted boundary-error
   penalty whose JOB is to STOP the render from concentrating Gibbs energy on the annulus (keep error
   ≤ uniform there), not to relocate error into the interior. Gate it on the owed n600 error↔margin
   correlation measurement; if error is already uniform, skip it.

## §6. verdict_scope

- `interior_leak = 0` / annulus-100%: **MEASURED**, exact from cached n96 margin field + the exact
  `½sech²(m/2)` law; robust (does not use the gain model). Scope: n96 real GT frame_1.
- realization floor §B (~0.0005–0.0009, Lane): **DERIVED** — margin field MEASURED, quantization scale
  from the closed-form head gain anchor (first-order pixel-space pullback). verdict_scope: FORMULATION
  (quantization-only, margin-blind, first-order). n600 generalization owed if a lever consumes it.
- evadable-by-relocation ≈ 0: **FORMULATION-level** negative on the UNIWARD-relocation formulation of
  the evasion dual; the CONDITIONAL boundary-de-concentration variant (§4) is UNMEASURED (owed:
  witness error↔margin correlation, n600) — one failed formulation is not the family dead.

## STORES CONSULTED

CLAUDE.md; AGENTS.md; docs/operating_manual_craft_handoff.md;
segnet_recursive_fractal_factorization_20260715.md (rank-4 head + realization-limited);
tac.canonical_equations.segnet_head_rank4_flipdist_20260715; tac.information_geometry.optimal_metric
(sech² Fisher law, 0.978); #333 annulus; L71/L85/L86 (analytic lane band + appearance-phase endgame);
L76 (msal_uni Lever-4 inert); gt_n96 cache (margins/lstars); MEMORY graph-recall (Fisher/margin/UNIWARD).
