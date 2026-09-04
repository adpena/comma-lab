---
title: "ddm_lb1 pre-registration — the analytic lane-band carrier's ceiling on the born field's persistent set"
arm: ddm_lb1
charter: .omx/research/charters/ddm_lb1_lane_band_carrier_ceiling_on_born_field_20260904.md
charter_commit: 29a303192
utc: 2026-09-04T15:05:00Z
verdict_scope: "[macOS-CPU advisory . LABEL-SPACE CEILING (composition into the retained argmax, NOT realized through R) . frozen CPU-torch SegNet . QBF1-born vehicle, cold control seed 20260902, terminal EMA shadow step 5000 . n32 sealed selection . DALI authority, PyAV beside it . NON-PROMOTABLE . no score claim . 0 Metal / 0 Modal / 0 contest eval]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# Pre-registration — written and committed BEFORE the numbers

## What this arm prices, and what it does NOT

It prices a **CEILING**, in **LABEL SPACE**. The composition below overwrites the born field's
terminal argmax at sites the carrier claims. The real carrier
(`src/tac/boundary_math/analytic_lane_render_band.py`) composites lane appearance into the **RGB
render before the contest R operator**, and the frozen SegNet then decides; the module's own
docstring records that the naive form of that composite **HURT** realized d_seg by +25%
(0.00333 → 0.00415, n600, l7-best levelset). Label-space substitution assumes perfect label
authority at every claimed pixel and is therefore a strict **UPPER BOUND** on what any realized
composite of this carrier could deliver. No row from this arm is a realized-through-R measurement,
and none may be cited as one.

## The object scored against

md1's retained partition (456c74551, store `/Volumes/APDataStore/pact/ddm_md1_micro_macro/`),
**loaded, never recomputed**: the terminal EMA-shadow argmax at step 5,000 of the cold control
(`payloads/cold_control_seed_20260902/shadow_step_005000.npz::argmax_u8`) and the six-class site
partition (`site_classes_cold_control_seed_20260902_shadow_dali.npz::site_class_u8`). Scoring is
md1's exact integer HT path: `W = Σ_p w_p·n_wrong(p)`, `w ∈ {15, 30}`
(`ddm_qbt1_qbflow_trainer.py:112`), denominator `600 · 384 · 512 = 117,964,800`
(`ddm_md1_micro_to_macro.py:655`), against DALI `gt_cache_dali.pt` as authority with PyAV beside it.

## Pre-registered prediction (charter §4, verbatim)

> Rule (a) removes **≥ 50%** of the persistent set at **≤ 2 KB** with harm **< 20%** of the removal.

## Pre-registered falsifier (charter §4, verbatim)

> **< 25% removed**, **or** harm ≥ removal, **or** bytes > 6 KB.

Any one of the three clauses firing falsifies the arm's premise that the landed lane-band carrier
is the different representation the persistent set demands.

## Definitions fixed before the measurement

* **Removal** — the reduction in the integer HT numerator carried by sites md1 classes
  `PERSISTENT`, expressed as a fraction of that class's terminal numerator (205,305 units, DALI
  shadow). It is measured on the numerator, not on site counts, because the estimator is
  HT-weighted and two site classes carry different weights.
* **Harm (B / H / W)** — a partition of every site by the composition's effect:
  **H**ealed = wrong before, correct after; **B**roken = correct before, wrong after;
  **W**rong-still = wrong before and after (including wrong→differently-wrong). Reported in HT
  numerator units. "Harm < 20% of the removal" reads `B_numerator < 0.20 · H_numerator`.
* **Bytes** — the coded size of the counted video-derived statistic (the per-pair lane-line
  coefficients) through the module's OWN bit-exact serializer, with the render performed on the
  **dequantized** lines (measure-what-you-ship). n32 total and per-pair; the n600 read is DERIVED
  and labelled TRANSFERRED.
* **Composition rules** (all three measured, none chosen after the fact):
  * **(a) REPLACE** — argmax := Lane where carrier coverage ≥ 0.5; where born said Lane and the
    carrier does not, argmax := the born field's own **runner-up** class (second-place logit).
  * **(b) UNION** — argmax := Lane where carrier coverage ≥ 0.5; born prediction kept elsewhere.
  * **(c) BAND** — rule (a) restricted to a dilated neighbourhood of the carrier curve; outside the
    dilated band the born prediction is kept untouched.
* **Lane class** is **self-detected** from the ground truth by spatial/geometric signature per the
  CLAUDE.md class-order law, then asserted against the module's default. It is never assumed.

## Calibration gates declared in advance

1. The re-run forward's argmax must reproduce md1's retained `argmax_u8` at step 5,000 **bit-for-bit**
   (same code, same weights, same CPU device). Differing sites are reported whatever they are.
2. The integer HT numerator recomputed from the retained argmax must reproduce md1's sealed
   `terminal_d_seg_hat` 0.0028065999348958334 exactly.
3. B + H + W + (always-correct, unchanged) must partition all 6,291,456 sites exactly, in integers.

Failing gate 1 or 2 does not stop the arm; it is reported and travels with every number.

## What a PASS would and would not authorize

A pass authorizes exactly one thing: a charter for the born trainer with Lane **held by the carrier
in-loop** (the field trained on the other four classes), whose verdict would be realized through R.
It authorizes no score claim, no pointer move, and no transfer of the n32 fraction to n600.
