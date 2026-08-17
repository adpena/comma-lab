# The per-event wall-clock cost is the EVAL, not the save — and 70.4% of a probe run is evaluation

**Status:** MEASURED (2026-08-17, MAIN, $0 — a by-product of the `ce1` curriculum probes).
**Axis:** wall-clock only. No score claim. Frontier untouched.
**Supersedes:** the save-cost attribution in `ddm_wallclock_prefix_bias_law_20260817.md` and
`ddm_l3000_no_descent_verdict_20260817.md`. The `(F, r)` two-point law is **re-fit**, not discarded.

## What happened

`ddm_aa3` struck the `~14.8 s/save` term as circular — back-solved from the residual it explains,
then reused as if measured — and left a NEXT_IF_RESUMED: *"re-fit all three terms from ≥3 points
with recorded save counts."* `CE0` supplies the third point. The re-fit **refutes the save model.**

All three elapsed figures come from **one instrument** — the launcher's `detached_local_process_done.v2`
receipt `elapsed_s` (checked at source before fitting, because mixing a launcher-measured elapsed with
a trainer-measured one is exactly this campaign's recurring defect):

| run | steps | evals | saves | elapsed_s (done receipt) |
|---|---:|---:|---:|---:|
| `A2_repeat` | 600 | 6 | 7 | 408.29374 |
| `L3000_off` | 3,000 | 30 | 13 | 1552.30873 |
| `CE0` | 600 | 24 | 25 | 865.50101 |

*(The prior memos cite 408.258 / 1552.209 — off by 0.01–0.02%, immaterial to the fit, but the
receipt values are used here.)*

## The refutation, and why it is informative

Three points fit three parameters. Two rival three-parameter models:

| model | F (fixed) | r (per step) | per-event | verdict |
|---|---:|---:|---:|---|
| **A:** `F + r·n + s·SAVES` (assumes eval cost ≈ 0) | **−17.41 s** | 0.41317 | 25.400 s/save | **REFUTED — F < 0 is unphysical** |
| **B:** `F + r·n + e·EVALS` (assumes save cost ≈ 0) | **+122.29 s** | 0.22267 | 25.400 s/eval | **PHYSICAL** |

A fixed cost is process start + `gt_cache` load + model init. It cannot be negative. **The data
rejects the save-dominated model and accepts the eval-dominated one.** This is not "e = 25.400 s is
measured" — with 3 points and 4 unknowns `(F, r, e, s)` I cannot separate eval from save. It is the
weaker, sound claim: *the per-event cost is attributable to evaluation, because attributing it to
saves alone forces an impossible fixed cost.* Mechanistically this is what one would expect — an
eval is a full n600 SegNet forward; a save is a few MB to SSD.

**Why the error was invisible for three runs:** in `A2_repeat` and `CE0`, `--eval-every` **equals**
`--checkpoint-every`, so evals and saves are collinear (6/7 and 24/25) and interchangeable in any
fit. Only `L3000_off` (30 evals, 13 saves) breaks the collinearity. I changed *two* knobs when I
set CE0's cadence to 25 and attributed the effect to one of them. Genus: the confounded-2×2 —
sister of [[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]] and of
the measured-object-vs-named-object law.

## The prior law was conflated, and by how much

`wallclock_fixed_cost_prefix_bias_v1` registered **r = 0.4395 s/step** from two points that
**both ran `--eval-every 100`**. At a fixed cadence, `e·evals = (e/100)·n` — perfectly collinear
with `r·n`. So the fitted `r` was never the marginal training rate; it was

`r_apparent = r + e/100 = 0.22267 + 0.25400 = 0.47667`

against the fitted 0.43950 (the residual is `b2e`'s different cadence). **Roughly 53% of what the
law called "training rate" was evaluation.** The law's *predictions* stay good wherever
`--eval-every 100` holds — that is why it scored +6.1% on `L3000_off` — and silently break the
moment the cadence changes, which is precisely what CE0 did.

## Where the wall-clock actually goes

| run | total | fixed | training | **evaluation** |
|---|---:|---:|---:|---:|
| `A2_repeat` | 408.3 s | 122.3 | 133.6 (32.7%) | **152.4 (37.3%)** |
| `L3000_off` | 1552.3 s | 122.3 | 668.0 (43.0%) | **762.0 (49.1%)** |
| `CE0` | 865.5 s | 122.3 | 133.6 (15.4%) | **609.6 (70.4%)** |

**CE0 spent 4.6× more wall-clock observing itself than training.** That is not waste — those evals
are the trajectory I read all session, and the fine cadence is what let CE0's excursion be located
at all. It is an **unpriced choice**: the observation cadence has never been selected deliberately,
and it now has a price — **25.4 s per sample.**

This is the *"where spent"* axis in its wall-clock form, and it has the same shape as the other two
budgets measured today: allocated by a **fixed cadence**, not by marginal value. `ddm_ws4` is
auditing whether that shape generalizes; this row is one confirmed instance, not the general law.

## Pre-registered prediction (write-before-measure)

`EF0` (600 steps, `--eval-every 25 --checkpoint-every 25` — identical cadence and length to `CE0`,
differing only in `--softplus-fraction`) is running now. Model B predicts

`122.29 + 600×0.22267 + 24×25.400` = **865.5 s**

— identical to CE0 by construction, so this is a **reproducibility test, not a new design point.**
A material deviation means run-to-run wall-clock spread exceeds the model's precision, and the
three-parameter exact fit (zero residual, no goodness-of-fit by construction) is over-trusted.

## NEXT_IF_RESUMED

1. **Score the EF0 prediction against 865.5 s.** Then the model has one validation point.
2. **A 4th point that breaks collinearity the other way** — same steps, same evals, *different*
   save cadence — separates `e` from `s` and turns the assumption into a measurement. Cheapest:
   600 steps, `--eval-every 25 --checkpoint-every 200`. ~6.8 min by Model B.
3. **Re-price every open window with `(F=122.29, r=0.22267, e=25.400)` AND its eval cadence.**
   A budget quoted without its cadence is under-specified — the same defect the prefix-bias law
   named for `n`, one level down.
4. **Update the registered equation** `wallclock_fixed_cost_prefix_bias_v1`: its `(F, r)` are
   cadence-conflated. Do not delete it — its consumer is `--walltime-cap-s`, and it *over*-predicts
   at eval-every-100 (the safe direction). Add the cadence term and the collinearity warning.
5. **Do not read the training-rate improvement as a speedup.** `r` fell 0.4395 → 0.2227 because a
   term moved out of it, not because anything got faster.
