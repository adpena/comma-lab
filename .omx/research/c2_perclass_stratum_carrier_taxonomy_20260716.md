# c2 per-class × per-stratum residual carrier taxonomy + separatrix asymmetry (2026-07-16)

**Source:** #515 FINAL-OPTIMAL c2 witness-design campaign (operator P0 2026-07-16) + operator deep-math
add-on (separatrix asymmetry, one-sided carrier). Extends `necessity_dseg_calibration_20260715.md`
(min-S knee: ε=0-lossless + static hood-tex, d_seg_real **0.01328** n600) by decomposing that residual
per class × stratum, measuring its static/dynamic character, its input-space cure drivers, the n600
separatrix asymmetry (upgrading eq `separatrix_asymmetry_t_subpixel_boundary_localizer_v1` from n6),
and $0 carrier-form smokes. **Pointer 0.19108 UNMOVED — everything here is MEANS** (c2 design inputs);
the exact `upstream/evaluate.py` row is the only score authority.

**Axis / honesty:** `[macOS-CPU advisory]` — frozen CPU-torch fp32 SegNet, bit-exact cached GT argmax
(`gt_n600.npz`). n600 unless labelled; smokes are stride-5 subsets (120 frames, ranking-only, winner
owed an n600 re-measure). `research_only; score_claim=false; promotable=false`.

**Tool:** `tools/c2_perclass_stratum_carrier_analysis.py` (stages decomp/temporal/slope/sens/smoke;
resumable). **Artifacts:** `experiments/results/c2_perclass_stratum_20260716/` (decomp_rows.jsonl ·
dis_masks_packbits.u8 · temporal.json 0a4b9a41 · slope.json b88ab29b · sens.json ded60916 ·
smoke_*.json).

---

## 1. Per-class × per-stratum decomposition of the 0.01328 residual (n600, MEASURED)

Stratum priority saddle > edge(1px) > near(≤3px) > far; class = the pixel's own GT class.
persist = P(same px disagrees at frame i+1 | disagrees at i); occ = per-pixel disagreement
frequency over n600 (static error sites → high occ).

| bucket | d_seg contrib | % of residual | persist_next | occupancy | character |
|---|---:|---:|---:|---:|---|
| Road\|edge | 0.004684 | 35.3% | 0.365 | 0.183 | boundary jitter, semi-static |
| Movable\|far | 0.003222 | 24.3% | **0.865** | 0.113 | object-interior, object-tracking |
| Road\|near | 0.001586 | 11.9% | 0.434 | 0.242 | boundary jitter halo |
| Movable\|near | 0.001101 | 8.3% | 0.609 | 0.088 | object border halo |
| Lane\|edge | 0.001098 | 8.3% | 0.304 | 0.126 | thin-stripe jitter |
| Movable\|edge | 0.000614 | 4.6% | 0.461 | 0.086 | object border |
| Undrivable\|edge+near | 0.000585 | 4.4% | ~0.31 | ~0.10 | horizon band |
| MyCar\|edge | 0.000140 | 1.1% | 0.270 | 0.261 | hood rim (post-hood-seed) |
| all saddle buckets | 0.000100 | 0.75% | ~0.45 | ~0.16 | measure-zero |
| everything else | 0.000150 | 1.1% | — | — | — |

Sum of buckets = 0.01328 exactly. **Top-6 buckets = 92.6%.** By class: Road 48.2% · Movable 37.4% ·
Lane 8.4% · Undrivable 4.6% · MyCar 1.5%.

**Static-vs-dynamic (MEASURED):** NO bucket is truly static (occupancy ≤ 0.26 everywhere — the hood
seed already removed the static core). Movable interiors are **object-persistent** (persist 0.865 at
the 2-frame horizon: the lead car is quasi-stationary in image position → ξ-transportable, NOT
per-frame-independent). Boundary buckets persist 0.27–0.46 (slow scene advection). ⟹ no further
"static seed" bucket exists; remaining carriers are motion-aware or render-side.

**Confusion (erasure direction, THIS vehicle):** edge|Road→Lane 240k vs edge|Lane→Road 123k — flips
concentrate on the **Road side** = lane **DILATION** (the crisp flat palette over-contrasts the lane),
the mirror image of the witness's lane EROSION (L65). Road→MyCar 182k + near|Road→MyCar 117k = hood
dilation. far|Movable→MyCar 234k / →Road 103k / →Undrivable 43k = Movable interiors read as
ego-car/background. Pair-side split (edge+near): Road-MyCar|Road 0.2534 · Road-Lane|Road 0.2196 ·
Road-Undrivable|Road 0.1277 · Road-Lane|Lane 0.1093 · Road-Movable|Movable 0.0964 ·
Undrivable-Movable|Movable 0.0741 (contrib ×100 units).

## 2. Separatrix asymmetry at n600 (extends the n6 t-localizer; MEASURED, cached bit-exact margins)

One-sided margin profile m_med(k), k=1..6px from each pair boundary (stride-5, 120 frames, ~0.9M–6k px
per pair); slope=(m(3)−m(1))/2; shallow side = where crossing is cheap.

| pair | slope shallow-side | slope deep-side | asym ratio | shallow side | m1 lo/hi |
|---|---:|---:|---:|---|---|
| Road-Undrivable | 0.45 (Road) | 1.25 (Undriv) | **2.78** | **Road** | 1.325/1.475 |
| Lane-MyCar | 0.45 (Lane) | 1.05 (MyCar) | **2.33** | **Lane** | 1.925/2.025 |
| Road-Lane | 0.70 (Lane) | 0.95 (Road) | 1.36 | **Lane** | 1.475/1.775 |
| Road-MyCar | 1.15 (MyCar) | 1.375 (Road) | 1.20 | MyCar | 1.475/1.475 |
| Undrivable-Movable | 0.625/0.675 | — | 1.08 | ~symmetric | 0.825/0.825 |
| Road-Movable | 0.675/0.70 | — | 1.04 | ~symmetric | 0.975/0.975 |

**Law (MEASURED): the asymmetry is PER-PAIR, not a global fine-class rule.** Lane is the shallow side
of both its major pairs (the erasure signature, as predicted). But Road-Undrivable is strongly
**Road**-shallow (far-field road at the horizon is fragile), and the two Movable pairs are symmetric
AND have the LOWEST absolute margins (m1 0.83–0.98 — the whole Movable annulus is fragile on both
sides). The flip SIDE observed in §1 is carrier-dependent (this vehicle dilates the lane; the witness
erodes it); the slope field is vehicle-agnostic and prices margin restoration per px on each side.

## 3. Cure-driver sensitivity at the disagreeing pixels (VJP through real SegNet, 119 samples)

∇(logit_gt − logit_pred) at the palette-render input, per bucket medians:

- **LUMA carries 92–94% of cure-gradient energy in EVERY bucket** (chroma 6–8%). The carrier driver is
  BT.601-luma texture. (Chroma-decides applies at GT-textured inputs; at the flat render, cure = luma.)
- **NON-LOCAL:** energy within r=4px (camera) is 0.00–0.15; within r=36px only 0.13–0.67. Movable|far
  cure is 83% OUTSIDE r36 and 58–64% on the PRED-side region — SegNet reads REGION signatures.
  Lane is the most local bucket (r4 0.15, r36 0.67).
- **FLAT-SHIFT-ORTHOGONAL:** flat coherence |Σg_luma|/Σ|g_luma| ≤ 0.24 everywhere (mostly 0.00–0.03) —
  a uniform value shift is nearly orthogonal to the cure; the required signal is sign-alternating
  spatial structure = **texture/gradient**. This is the photometric wall (L68) measured at the input
  surface. **Exception: Lane** (coh 0.12–0.24 on the GT side, sign +10/1 consistent = "brighten the
  lane side") — the one bucket with a genuine flat one-sided component.
- MyCar|edge sign 0+/6− (consistent darken); Movable signs mixed (2+/8− edge) — no single push works
  per-class there in the flat basis; the band-contrast form (below) is what works.

## 4. Carrier-form smokes (stride-5 subset = 120 frames, LABELLED ranking-only; baseline 0.013044)

| variant | counted B/frame | d_seg (subset) | Δ vs base | verdict |
|---|---:|---:|---:|---|
| **oneside_movable band β=2.0** | **0** | **0.009196** | **−29.5%** | WINNER: cures 83% of Movable\|far, 73% near, 66% edge from a 2px band |
| oneside_movable β=1.5 | 0 | 0.009518 | −27.0% | monotone in β (0.5→2.0) |
| oneside_shallow {Lane,Movable} β=1.5 | 0 | 0.009607 | −26.4% | Lane push at 1.5 overshoots (dilation) |
| oneside_shallow β=0.5 | 0 | 0.011013 | −15.6% | |
| oneside_lane β=0.5 | 0 | 0.012812 | −1.8% | small; β must stay small |
| movable_meancolor (per-frame mean RGB) | 3 | 0.012282 | −5.8% | dominated by 0-B band |
| blur σ=2 | 0 | 0.013991 | +7.3% | blur adds no structure |
| symmetric band β=0.5 | 0 | 0.022572 | +73% | deep-side half poisons |
| oneside_deep {Road,Undriv,MyCar} β=0.5 | 0 | 0.025854 | +98% | pushing the deep side moves pixels OFF their own class manifold |
| tex_movable_ds8 (GT oracle) | 1749 | 0.008690 | −33.4% | oracle ceiling for Movable; bytes dominated (1.05MB n600) |
| tex_global_ds16 (GT oracle) | 6952 | 0.023194 | +78% | **blurry real texture LOSES to the crisp cartoon** — the cure is high-frequency structure, not low-freq realism |
| tex_band_ds4 (GT band paste) | 7795 | 0.024455 | +87% | INSTANCE-scoped negative: texture-band-in-flat-interior creates phantom region boundaries |

**Asymmetric dominance CONFIRMED (the operator's prediction):** the one-sided carrier on the correct
side beats the symmetric one categorically (−29.5% vs +73%) at ZERO bytes. The correct side for the
band-contrast form is **per the confusion+margin structure** (Movable both-sides-fragile annulus →
push the Movable side away from its partner; Lane shallow side at small β), NOT a blanket
fine-class rule. The deep-side/symmetric failure is the measured cost of placing perturbation where
the OWN-class manifold is left (a flat push has no cure component there, only collateral crossings).

**Region-from-boundary law (NEW, MEASURED):** a 2px boundary-band perturbation re-classifies entire
Movable INTERIORS (83% of Movable|far cured; those pixels are ≥3px, mostly ≥10px away). SegNet reads
region identity substantially from border contrast — carriers for interior buckets should be PLACED
on the boundary annulus. This is the mechanism behind #333 (97% of d_seg in the annulus) seen from
the cure side, and it makes v8/v9 per-object Laguerre-cell carriers cheap: the byte content that
matters is the cell BORDER profile, not the interior fill.

## 5. The carrier taxonomy (c2 design inputs, ranked by residual value)

| class × stratum | % | natural minimal carrier (generator, seed) | measured support |
|---|---:|---|---|
| Road\|edge+near (jitter at Road-Lane/MyCar/Undriv) | 47% | JOINT-trained render boundary profile + sub-pixel appearance-phase geometry (L85/L86); one-sided placement per §2 slopes; NOT flat pushes (deep-side +98%), NOT band texture paste (+87%) | sens: non-local, flat-orthogonal; slope: per-pair shallow sides |
| Movable\|far+near+edge | 37% | **one-sided border-contrast band on the Movable side** (0-B generator term, β≈2, MEASURED −29.5% total) + per-object cell generator with ξ-transport (persist 0.865) for the residual 0.006-of-0.013 share | smoke winner; oracle ds8 ceiling −33% |
| Lane\|edge | 8.3% | small-β one-sided luma brightening on the Lane side (the ONE flat-coherent bucket, +10/1 sign) + dash-phase/polynomial generator (L71/L73) for geometry | sens coh 0.12–0.24; smoke −1.8% @ β=0.5 (needs per-pair β tuning, β≤0.5) |
| Undrivable\|edge+near (horizon) | 4.4% | Road-side is the shallow side (2.78×): one-sided horizon-band treatment on the ROAD side; ties into Road\|edge carrier | slope table |
| MyCar\|edge (hood rim) | 1.1% | extend the static hood-tex seed to its boundary band (static, occ 0.26; darken direction 0/6−) | sens signs; hood seed precedent |
| saddles (all) | 0.75% | NO bytes: precision annotation on edge carriers (unchanged from necessity solver) | §1 |

**What this settles for c2:** (a) the trained render's d_seg budget should be concentrated on the
**boundary annulus luma texture** (region identity is border-driven, luma-driven, non-local); (b) the
archive's cheap carriers are ONE-SIDED per-pair band terms with sides read from the measured slope
field (a few constants per pair — a generator term, ~0 counted bytes); (c) Movable needs ξ-tracked
per-object border carriers, not interior texture; (d) no further static-seed bucket exists after the
hood; (e) blurry low-frequency realism is anti-productive — spend on high-frequency boundary
structure only.

## 6. Round-1 adversarial review (own attack) + boundaries

- **Subset caveat:** all §4 smokes are stride-5 (120-frame) rankings; the winner (β=2 movable band) is
  OWED an n600 re-measure + a through-R/byte-closed A/B before any composition claim. §1/§2 are n600
  (§2 slope stride-5 but ~0.9M px/pair samples).
- **Vehicle caveat:** the residual decomposed here is the PALETTE render's (its flip direction is
  dilation; the witness's is erosion). The slope field (§2), the luma/non-local/flat-orthogonal cure
  structure (§3), and the region-from-boundary law (§4) are vehicle-facing measurements of the FROZEN
  scorer and transfer; the specific bucket weights (§1) are vehicle-specific.
- **β=2.0 not converged** (still improving at sweep edge) and kpx=2 unswept; per-pair β untuned —
  optimal form NOT reached; numbers are lower bounds of the form.
- **NOT measured:** a real FiLM-conditioned trained residual (out of $0 scope — the oracle smokes
  bracket it); pose interaction of the band carrier (PoseNet chroma-blind <2px per
  frozen_scorer_exact_factorization §6, and the band is luma at 2px scorer-res ⇒ pose risk small but
  UNMEASURED); n600 verdicts for smoke variants; the witness's own per-bucket decomposition (owed:
  run stage decomp on witness output frames).
- Naive-negative scoping: tex_band_ds4 and tex_global_ds16 negatives are INSTANCE-scoped (naive paste
  forms); the family "band texture, blended/trained" remains open.

## 7. Triality + stores consulted

- **DAG:** FEED-c2-taxonomy appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **equations:** `perclass_stratum_residual_carrier_taxonomy_v1` registered (anchors: decomposition ·
  n600 slope asymmetry · one-sided-dominance smoke). Extends
  `separatrix_asymmetry_t_subpixel_boundary_localizer_v1` (n6→n600 upgrade of the asymmetry surface)
  and `necessity_generator_seed_dseg_calibration_v1` (whose residual it decomposes).
- **DSL:** OWED — the one-sided per-pair band-contrast term is a config surface (per-pair side + β +
  kpx) that belongs as an archive-build/generator `Lever` when the carrier ships; recorded owed, not
  hand-added.
- **STORES CONSULTED:** necessity_dseg_calibration (parent knee) · necessity_solver_inverse_factorization
  (strata, K-ladder) · frozen_scorer_exact_factorization (luma/chroma bases, resize) ·
  segnet_recursive_fractal (rank-4 head, ERF) · separatrix_asymmetry_t eq (prior n6 study) · #333
  annulus · L65/L68/L71/L73/L85/L86 · SPEC_v8 · rate_law_ladder.

**Pointer 0.19108 UNMOVED — MEANS.**
