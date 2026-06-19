---
title: Polynomial-fill survival gate VERDICT — AMBER (interior SOLVED, boundary-band wall HOLDS)
authority: "[contest-CPU advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: frontier_breaking_enabler
date: 2026-06-18
verdict: AMBER_POLYNOMIAL_FILL_IMPROVES_BUT_PLATEAUS_ABOVE_SUB015
wall_diagnosis: BOUNDARY_BAND_SURVIVAL_WALL (interior solved by the gradient; the 1px boundary band is representation-independent)
cross_refs:
  - experiments/probe_polynomial_fill_survival_gate.py
  - experiments/results/polynomial_fill_survival_probe/gate_state.json
  - .omx/research/polynomial_fill_survival_gate_20260619T014622Z.json
  - .omx/research/curve_core_gate_RED_survival_wall_and_the_pincer_20260618.md
  - .omx/research/eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md
  - .omx/research/factored_lf_core_capacity_gate_20260618T233913Z.md
tasks:
  - "#137/#138 (road<->lane / openpilot geometric prior): a per-region gradient fill IS that prior; this gate measures its scorer survival"
---

# Polynomial-fill survival gate — VERDICT: AMBER (interior SOLVED, boundary-band wall HOLDS)

**Does a CONTINUOUS per-region polynomial gradient fill (photoshop gradient-fill-per-selection) beat
the FLAT-COLOUR survival wall that capped the curve-core, and at what byte cost?** Measured answer:
**partly.** The polynomial gradient SOLVES the region INTERIORS (interior realized d_seg collapses
0.00557 → 0.00005, a ~100× win — continuous gradients survive the resize) but the **1px boundary band
caps realized d_seg at ~0.0061 regardless of polynomial order** (boundary_band_flip stuck ~0.15
across k=0..10). Best operating point k=5: realized d_seg **0.00609** (2.4× frontier), **S = 0.697** —
nowhere near sub-0.15. All `[contest-CPU advisory]` NON-PROMOTABLE; exact pointer **UNMOVED at 0.19110**
(stated plainly per the GOAL firewall: this unit did NOT move the pointer). $0, CPU only, no paid GPU, no PR.

## The order ↔ d_seg ↔ bytes table (k=0..10, oracle GT partition, 3 GT frames; least-squares fit to GT RGB)
| k | coeffs | fit_resid | geo_dseg | **realized** | bnd_flip | **int_flip** | rate | S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (flat)  | 119  | 14.8 | 0.01424 | 0.01516 | 0.240 | 0.00557 | 0.00399 | 1.579 |
| 1 (linear)| 287  | 13.4 | 0.00956 | 0.00923 | 0.187 | 0.00160 | 0.00962 | 0.991 |
| 2 (quad)  | 419  | 10.8 | 0.00705 | 0.00727 | 0.170 | 0.00033 | 0.01404 | 0.800 |
| 3 (cubic) | 524  | 10.2 | 0.00825 | 0.00954 | 0.194 | 0.00166 | 0.01756 | 1.030 |
| 4         | 693  |  9.5 | 0.00729 | 0.00746 | 0.175 | 0.00035 | 0.02322 | 0.827 |
| **5**     | 879  |  9.2 | 0.00605 | **0.00609** | 0.148 | **0.00005** | 0.02945 | **0.697** |
| 6         | 1037 |  8.9 | 0.00739 | 0.00753 | 0.180 | 0.00016 | 0.03475 | 0.846 |
| 7         | 1257 |  8.6 | 0.00673 | 0.00690 | 0.165 | 0.00012 | 0.04152 | 0.790 |
| 8         | 1395 |  8.3 | 0.00678 | 0.00658 | 0.160 | 0.00010 | 0.04675 | 0.763 |
| 10        | 1939 |  7.9 | 0.00628 | 0.00633 | 0.155 | 0.00010 | 0.06497 | 0.757 |

`fit_resid` = per-pixel RGB L2 of the LS fit (drops monotonically — the polynomial fits GT better and better);
`geo_dseg` = filled-frame SegNet argmax, NO roundtrip; **`realized` = hard fill THROUGH the EXACT uint8
roundtrip (bicubic-874 → bilinear-384 → round) + real frozen SegNet vs L* = the AUTHORITY**;
`int_flip`/`bnd_flip` = realized flip in the interior vs the 1px-dilated boundary band.
S recomputed from components (k=5): 100·0.00609 + √(10·0.00034) + 0.02945 = 0.609 + 0.0583 + 0.0295 =
**0.697 ✓** (matches stored — no label inversion, NO-FAKE clean). GREEN rows (rate<0.05 ∧ realized<0.0012
∧ S<0.15): **NONE**.

## The decisive mechanism (the 3-way decomposition is the finding)
- **The gradient SOLVES the interior.** `int_flip` collapses **0.00557 (k=0 flat) → 0.00005 (k=5)** — a
  ~100× drop, far BELOW the frontier d_seg (0.00257) and even the sub-0.15 target (0.0006). A continuous
  per-region polynomial gradient lands INSIDE the per-pixel argmax polytope in region interiors where a
  flat colour did not. **This is a real, large win — the continuous-gradient hypothesis is CONFIRMED for
  interiors.** It also confirms the eval-roundtrip-deep-math finding: the resize is benign on smooth
  texture; the wall was never the resize per se.
- **The boundary band does NOT move.** `bnd_flip` is stuck at **~0.15–0.24 across ALL orders k=0..10** — it
  does not descend with polynomial order. The 1px band where two regions meet is mixed by the
  camera→384 bilinear downsample ACROSS the SegNet decision boundary, and **no amount of per-region
  interior gradient can fix a band whose pixels straddle two regions** (the polynomial is defined
  per-region; the boundary pixel belongs to the wrong-side polynomial after mixing). This is
  **representation-independent**: it held for the static partition store (flat), the differentiable
  curve-core (flat + backpropped colour/band offset), and now the oracle-fit polynomial gradient.
- **realized d_seg ≈ (boundary fraction) × bnd_flip.** At k=5, realized 0.00609 is almost ENTIRELY
  boundary-band flips (interior contributes 0.00005). The whole remaining gap is the 1px boundary band.

## The localized signal (high-value, forward-looking)
**If the boundary-band wall vanished**, S at the k=5 operating point would be
`100·0.00005 + √(10·0.00034) + 0.02945` = **0.093 — comfortably sub-0.15.** The interior is already
sub-0.15-grade; the ENTIRE sub-0.15 deficit for a continuous-fill vehicle is now localized to the 1px
boundary band. This is a much sharper target than "the survival wall" — it is specifically the
cross-region boundary-pixel mixing under the bilinear downsample.

## Honest caveats (NO-FAKE)
1. **k=0 here (0.01516) ≠ the curve-gate flat wall (0.00673).** My k=0 uses the per-region GT-MEAN colour
   over the ORACLE GT partition; the curve gate's flat baseline used SegNet-OPTIMAL flat colours
   BACKPROPPED through the roundtrip. GT-mean is a photographic mean, not the SegNet-optimal flat, so it
   floors higher. **The polynomial sweep is apples-to-apples WITHIN itself** (every order is a GT
   least-squares fit). The harness sanity check soft-flags this delta transparently in the JSON — it is a
   colour-choice difference, not a bug. Crucially: even the polynomial sweep's BEST (k=5, 0.00609) only
   ~matches the curve-gate's SegNet-OPTIMAL flat (0.00673) on the boundary — the gradient's win is the
   interior, not the boundary.
2. **Oracle partition + oracle fit (upper bound).** Both the partition (GT L*) and the per-region
   coefficients (LS to GT RGB) are oracle/upper-bound. A real byte-cheap vehicle would store a quantized
   partition + quantized coeffs and do WORSE, not better. So the boundary-band wall is, if anything,
   even more binding in practice. The reported rate counts coeffs only (fill-only; geometry is the oracle
   and NOT counted — stated explicitly).
3. **n=3 frames** (matching the curve/NCA gates for apples-to-apples). The boundary-band plateau is
   stable across all 3 frames and all 10 orders (the per-frame rows in the JSON agree within noise).
4. **Non-monotone in k.** realized oscillates ~0.006–0.0095 from k=2 onward (small-region ringing/overfit
   at high order). The PLATEAU (not a descent) is the robust signal; k=5's dip is the floor, confirmed by
   k=7/8/10 staying ~0.0063–0.0069.

## VERDICT: AMBER — the order↔d_seg↔byte curve, partial win, and where it plateaus
**AMBER_POLYNOMIAL_FILL_IMPROVES_BUT_PLATEAUS_ABOVE_SUB015.** A continuous polynomial gradient beats the
flat survival wall **in the interior** (a genuine ~100× d_seg win there) but **plateaus above sub-0.15**
because the 1px boundary band is representation-independent and caps realized d_seg ~0.0061 (S ~0.70).
There is NO (k, rate) with realized < 0.0012 ∧ rate < 0.05 ∧ S < 0.15.

This **refines** the curve-core RED (`curve_core_gate_RED_survival_wall_and_the_pincer_20260618.md`) into
a precise, localized mechanism: the survival wall is NOT the whole frame — it is **specifically the
cross-region boundary-band mixing under the bilinear downsample**. The interior is solved by any
continuous gradient.

## What this means for the pincer + the next move
The curve-gate pincer ("flat-region representations are survival-walled; continuous-texture
representations are capacity-walled") is **partly broken**: a continuous gradient is byte-cheap (k=5 fill
rate 0.029, no capacity wall — it is a closed-form fit, not a learned decoder) AND solves the interior
(beating the survival wall there). The ONLY remaining wall is the 1px boundary band. So the sub-0.15
question for the continuous-fill family is now narrowed to ONE concrete sub-problem:

> **Can the cross-region boundary band be represented so its post-downsample SegNet argmax matches L*?**

Candidate mechanisms this gate makes concrete (forward-looking, NOT done here):
1. **Boundary-pixel anti-aliasing / pre-compensation.** Render the camera-res frame with a boundary band
   whose colours pre-distort so the bilinear-mixed 384-res pixel argmaxes to the correct side (a
   classic anti-alias inverse). The curve-core's backpropped band-offset attempted a 1px version and did
   NOT beat the wall — but it optimized a flat-class offset, not a true sub-pixel boundary placement.
2. **Sub-pixel boundary placement at camera-res.** The argmax flip is set at camera-res (1164×874) where
   the boundary has ~3× the pixels; a sub-pixel-accurate camera-res boundary may downsample to a band
   that lands on the right side. The polynomial here is on the 384 grid; a camera-res render is untested.
3. **The frontier's actual answer = continuous HF texture across the boundary.** The frontier decoder
   reaches 0.00056 because it paints a continuous frame with NO hard region boundary at all — the
   "boundary" is a smooth texture gradient the resize preserves. The interior-win here (0.00005) is
   exactly what the frontier does everywhere; the frontier just also blends the boundary continuously.
   This points back to a learned-texture decoder OR a hand-built continuous boundary blend, NOT a
   hard-partition fill.

**The terminal reframe is NOT yet reached.** This is an AMBER that opens a sharper, smaller corner (the
boundary band) rather than closing the family. The next $0 gate should be **boundary-band-specific**:
measure whether a camera-res sub-pixel boundary OR a learned 1px boundary-blend can move bnd_flip below
~0.05 (which, with the already-solved interior, would give S ~0.10).

## Connection to tasks #137/#138 (road↔lane / openpilot geometric prior)
A per-region gradient fill IS the openpilot geometric-prior representation those tasks want (regions =
road/lane/etc., each a smooth gradient). This gate's result tells those tasks: **the geometric-prior
gradient representation is scorer-viable in region interiors (sub-0.15-grade there) but needs an explicit
boundary-band model** to be scorer-viable end-to-end. The road↔lane boundary is exactly the dominant
boundary band; a geometric prior that models the lane-marking boundary sub-pixel-accurately is the
on-ramp.

## Process notes (self-protect)
- The probe uses `python -u` for all runs; the full sweep + the k=7/8/10 confirmatory tail completed in
  <1 min (closed-form LS, no gradient, no MPS) so no daemon was needed (folds in the curve-gate's
  daemon-death lesson: this gate is fast enough to run in the foreground with a resumable per-k JSON
  checkpoint at `experiments/results/polynomial_fill_survival_probe/gate_state.json`).
- 14 NO-FAKE tests (`src/tac/tests/test_polynomial_fill_survival_gate.py`): the fill is an ACTUAL
  polynomial gradient for k≥1 (non-flat) and flat for k=0; a higher order fits a curved frame strictly
  better (real LS); k=0 == per-region mean; the roundtrip resizes+clamps+rounds; realized d_seg goes
  through the REAL SegNet (self-match-is-zero); a degenerate thin region stays finite/valid (numerical
  guard); byte cost monotone; S arithmetic is the exact contest functional.

## Cross-refs
`curve_core_gate_RED_survival_wall_and_the_pincer_*` (the RED this refines — the wall is the boundary band,
not the whole frame), `eval_roundtrip_deep_math_pr95_handling_and_exploits_*` (the resize-is-benign /
texture-dependence mechanism this confirms at the interior), `factored_lf_core_capacity_gate_*` (the
continuous-texture capacity wall — sidestepped by the closed-form fill, which has no capacity wall).
Tasks #137/#138 (geometric prior). Pointer UNMOVED 0.19110.
