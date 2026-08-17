# The wall-clock prefix-bias law — every window on this trainer was over-priced, by a factor that GROWS with the window (4.9× at n=600, 6.8× at n=3,000)

**Status:** MEASURED (2026-08-17, MAIN, $0 — a by-product of firing jr1's `A2_repeat`).
**Axis:** wall-clock only. No score claim. Frontier untouched.
**Prediction pre-registered below BEFORE the confirming run lands.**

> ⚠ **CORRECTED 2026-08-17 by `ddm_aa3` (fresh-eyes audit).** Three defects, all landed at source
> in this file, in `.omx/research/ddm_l3000_no_descent_verdict_20260817.md`, in the registered
> equation `wallclock_fixed_cost_prefix_bias_v1`, and in the DAG leg:
> 1. **The headline universalised an `n`-dependent factor.** "Every window … 4.9×" is the n=600
>    value; this file's own re-pricing table already carried 6.8× for n=3,000. Corrected above and
>    in §"What it re-prices".
> 2. **The "run-to-run floor" is quoted at the single step where it is smallest** (§"The reason
>    this got fired at all"). Corrected there with the full six-step measurement.
> 3. **The checkpoint counts (6 / 12) were the `n / --checkpoint-every` arithmetic, not the
>    receipts** (7 / 13). The *difference* is 6 either way, so the +89 s attribution is unaffected.
>
> The `(F, r)` separation, the pre-registered prediction, and the +6.1% score are **unaffected and
> reproduce exactly** from the receipts (`elapsed_s` 408.258 and 1552.209).

## The finding

`ddm_b2e_sealed_launch_ticket_20260816.md:3` records, verbatim:

> 166.30 s / 50 steps end-to-end … **Derived window: ≤3.33 s/step → 3,000 steps ≈ 2.8 h**

That 3.33 s/step became the campaign's cost model. It propagated into `ddm_jr1`'s
"~33 min per 600-step arm", and from there into my own "Leg C = 2.20 h vs a 2.75 h window".

`A2_repeat` (this turn, 600 real steps, **7** checkpoint saves per its own
`checkpoints_written` receipt) took **408.258 s**. End-to-end that is **0.680 s/step** — the
sealed budget is **4.90× too expensive at n = 600**.

⚠ The factor is **not** a constant. It is `3.326·n / (F + r·n)`: **1.00× at n=50** (by
construction — that is where the smoke was measured), **4.89× at n=600**, **6.82× at n=3,000**,
asymptote **7.57×**. Quote it with its `n`, or quote `(F, r)` instead.

*(`verdict = PASS` was cited here as corroboration in the original draft. It corroborates nothing:
`ddm_pl1` measured that this trainer reports `verdict = PASS` on every arm regardless of whether it
descended — including this one, which did not. Removed rather than repeated.)*

## The mechanism, separated exactly

An **end-to-end** rate is not the marginal rate. It is `r + F/n`, where `F` is fixed cost
(process start, `gt_cache` load, model init, first eval) paid once regardless of `n`.
Two points separate the terms:

| n | total | end-to-end s/step |
|---:|---:|---:|
| 50 (b2e smoke) | 166.30 s | 3.326 |
| 600 (A2_repeat) | 408.0 s | 0.680 |

→ **`r` = 0.4395 s/step · `F` = 144.3 s.**

Internal check: the fit reproduces b2e's own quoted 3.33 at n=50 to three digits
(3.326). At n=50, **87% of the measured wall-clock was fixed cost.**

## Why this is a genus, not an arithmetic slip

This is the prefix-bias law ([[m88]], [[m96]] — a prefix of a skewed population is a
different population) on the **time** axis. Same shape, different quantity: extrapolating a
short-prefix *rate* to a long run overstates by exactly the fixed-cost fraction. Nobody
mis-multiplied; the number was measured honestly and then carried across a scope boundary
where its denominator changed. Sister of the standing law that an instrument's
**units × level × aggregation** are part of its claim
([[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]]).

**The cure is structural, not vigilance:** a wall-clock budget derived from a smoke must
quote `(F, r)` from ≥2 points, or declare itself an upper bound. One point cannot separate
them, and one point is what every budget on this trainer had.

## Pre-registered prediction (write-before-measure)

`L3000_off` (3,000 steps, lr 2e-5, band OFF, pid 67567, launched 2026-08-17) should take
**24.4 min** = `144.3 + 3000 × 0.4395` s.

- b2e's model predicts **2.77 h** (6.8× the fit).
- A naive re-extrapolation from the 600-step end-to-end rate predicts **34.0 min** (1.4× the fit).

**Named caveat, declared before the result:** `L3000_off` writes 12 checkpoints
(`--checkpoint-every 250`) against `A2_repeat`'s 6 (`--checkpoint-every 100`). If per-checkpoint
cost is material, the actual will exceed 24.4 min and the two-point fit will need a third
term. That over-run would be a *finding about checkpoint cost*, not a refutation of the
fixed-vs-marginal split.

## What it re-prices

| window | b2e model | measured fit | over-pricing factor |
|---|---:|---:|---:|
| 50-step smoke | 2.8 min | 2.8 min | **1.00×** (the fit point) |
| 600-step arm | 33 min | **6.8 min** | **4.89×** |
| Leg C (4 × 600, separate processes) | 2.20 h | **27.2 min** | **4.89×** |
| 3,000-step window | 2.77 h | **24.4 min** | **6.82×** |
| (asymptote, n → ∞) | — | — | **7.57×** |

The factor column is the correction `ddm_aa3` landed: it was always in this table implicitly, and
the headline flattened it to a single number.

**The sequencing consequence.** I corrected jr1's "Leg C gates at ~1/10 the cost" to 80% and
routed on scarcity. The *ratio* survives — both numbers scale by the same factor — but the
**decision it drove evaporates**: at 27 min and 24 min there is no scarcity to sequence
around. Both run. A ratio can be right while the decision it licenses is wrong, because a
ratio discards the scale that made the trade-off real.

## The reason this got fired at all

`A2_repeat` was fired to establish the **run-to-run floor**. It did — but the floor is
**step-dependent**, and the original draft of this section quoted only the step where it is
smallest.

⚠ **CORRECTED 2026-08-17 by `ddm_aa3`.** The original text read: *"peak 27,098.0 vs A2's 27,170.0 =
72 flips, 0.2650%, against a 6.1% band effect — 23× headroom. The instrument can separate the
arms."* That is true **at step 100 and nowhere else**. A2 and `A2_repeat` are the same config and
the same seed (20260715 — argv diff: `--band-objective-weight` explicit-0.0 vs defaulted-0.0,
`2e-5` vs `2.0e-5`, and the `--save` basename; nothing else), so every difference below is pure MPS
run-to-run nondeterminism:

| step | A2 | A2_repeat | Δ | Δ as % of A2 |
|---:|---:|---:|---:|---:|
| **100** | 27,170 | 27,098 | **−72** | **0.26%** |
| 200 | 14,237 | 12,460 | −1,777 | **12.48%** |
| 300 | 12,009 | 11,146 | −863 | 7.19% |
| 400 | 9,607 | 10,415 | +808 | 8.41% |
| 500 | 8,415 | 9,419 | +1,004 | 11.93% |
| **600** | 8,049 | 8,654 | **+605** | **7.52%** |

**Steps 200–600: 7.19% – 12.48%, mean 9.51% — 28× the step-100 figure.** Step 100 is the peak of a
spike that both runs hit hard and identically; it is the *least* representative step in the window,
not a summary of it.

**What this changes.** "23× headroom" holds **only for a PEAK-vs-PEAK comparison at step 100**
(6.1% ÷ 0.265%). Measured against the step-600 floor the same arithmetic gives **0.81× — the noise
exceeds the effect.** So:

- **jr1 §5.3's Leg C design is unaffected**: it interpolates band arms to matched `‖Δw‖₁₀₀` and
  compares *peaks* against A2's 27,170. Peak comparison, peak floor — internally consistent, and
  the pre-registered `27,170 − 3 × spread` bar still stands at step 100.
- **The sentence "the instrument can separate the arms" is withdrawn as a general claim.** It is
  true of the peak and false of the end. jr1 §4's "6.4% fewer flips at step 600" sits *inside* a
  7.52% run-to-run floor — which is exactly what jr1 §4 already said ("every single number here is
  inside the instrument's noise"); this measurement supplies the number that sentence lacked.
- **Any future comparison on this trainer must state WHICH STEP its noise floor was measured at.**
  One re-run gives six floors, spanning 48×.

⚠ And the floor itself rests on **n = 1** — a single A/A pair. It is an observed difference, not an
estimated spread; three-sigma bars built on it are decorative. A second repeat is the cheap fix
(6.8 min by the law above).

The wall-clock law is a free by-product of running a real 600-step arm where every prior number came
from a 50-step smoke. That part is unaffected.

## NEXT_IF_RESUMED

1. ~~When `L3000_off` lands, check the elapsed against **24.4 min**… `A2_repeat` sat **+8,654 flips
   above init** and was still falling **765 flips/100 steps** at the 600 cap (~1,131 steps from
   parity at that rate), so 600 steps was plausibly a **truncation**, not a wall.~~
   **DONE — and the truncation half is REFUTED** by `ddm_l3000_no_descent_verdict_20260817`: 2,400
   further steps closed 3,767 of the 8,654 with the rate decaying 32×. Linear extrapolation of a
   decaying tail was the wrong instrument.
   ⚠ `ddm_aa3` adds a **second, independent** defect in the same sentence: **−765 flips/100 is
   `A2_repeat`'s tail rate, and A2's own tail rate for the identical config is −366/100.** One A/A
   pair spans **2.1×** on this quantity, so "~1,131 steps" was `1,131–2,199` before it was anything.
   **Never quote a tail rate from a single run on this trainer.**
2. Re-price every open window on this trainer with `(F=144.3 s, r=0.4395 s/step)` before
   routing on cost. No budget on this vehicle derived from a single smoke survives.
   ⚠ Do **not** stack the `~14.8 s/save` attribution on top of this pair: `F = 144.3 s` was fit from
   two points that each already *include* their own saves (7 in `A2_repeat`), so `F` already carries
   ~7 saves' cost. Adding a separate save term double-counts ~104 s. That is precisely why the
   three-term model fails to close against the n=50 point (§L3000 verdict). Use `(F, r)` as a pair,
   or re-fit all three terms from ≥3 points with recorded save counts.
3. The curriculum **stretches** with the run: `--ce-fraction 0.50 --softplus-fraction 0.85` are
   fractions, so a 3,000-step run is a *different schedule*, not the 600-step run continued.
   Do not read its trajectory as a continuation of `A2_repeat`'s.
