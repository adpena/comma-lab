# The wall-clock prefix-bias law — every window on this trainer was priced 4.9× too expensive

**Status:** MEASURED (2026-08-17, MAIN, $0 — a by-product of firing jr1's `A2_repeat`).
**Axis:** wall-clock only. No score claim. Frontier untouched.
**Prediction pre-registered below BEFORE the confirming run lands.**

## The finding

`ddm_b2e_sealed_launch_ticket_20260816.md:3` records, verbatim:

> 166.30 s / 50 steps end-to-end … **Derived window: ≤3.33 s/step → 3,000 steps ≈ 2.8 h**

That 3.33 s/step became the campaign's cost model. It propagated into `ddm_jr1`'s
"~33 min per 600-step arm", and from there into my own "Leg C = 2.20 h vs a 2.75 h window".

`A2_repeat` (this turn, 600 real steps, 6 checkpoints, `verdict = PASS`) took **408 s**.
End-to-end that is **0.680 s/step** — the sealed budget is **4.90× too expensive**.

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

| window | b2e model | measured fit |
|---|---:|---:|
| 600-step arm | 33 min | **6.8 min** |
| Leg C (4 × 600, separate processes) | 2.20 h | **27.2 min** |
| 3,000-step window | 2.77 h | **24.4 min** |

**The sequencing consequence.** I corrected jr1's "Leg C gates at ~1/10 the cost" to 80% and
routed on scarcity. The *ratio* survives — both numbers scale by the same factor — but the
**decision it drove evaporates**: at 27 min and 24 min there is no scarcity to sequence
around. Both run. A ratio can be right while the decision it licenses is wrong, because a
ratio discards the scale that made the trade-off real.

## The reason this got fired at all

`A2_repeat` was fired to establish the **run-to-run floor**, and it did: peak 27,098.0 vs A2's
27,170.0 = **72 flips, 0.2650%**, against a 6.1% band effect — **23× headroom**. The instrument
can separate the arms. The wall-clock law is a free by-product of running a real 600-step arm
where every prior number came from a 50-step smoke.

## NEXT_IF_RESUMED

1. When `L3000_off` lands, check the elapsed against **24.4 min** and record which of the three
   models held. Then read `improved_over_init` — `A2_repeat` sat **+8,654 flips above init**
   and was still falling **765 flips/100 steps** at the 600 cap (~1,131 steps from parity at
   that rate), so 600 steps was plausibly a **truncation**, not a wall.
2. Re-price every open window on this trainer with `(F=144.3 s, r=0.4395 s/step)` before
   routing on cost. No budget on this vehicle derived from a single smoke survives.
3. The curriculum **stretches** with the run: `--ce-fraction 0.50 --softplus-fraction 0.85` are
   fractions, so a 3,000-step run is a *different schedule*, not the 600-step run continued.
   Do not read its trajectory as a continuation of `A2_repeat`'s.
