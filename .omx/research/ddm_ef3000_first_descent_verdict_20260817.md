# THE SEG REGIME DESCENDS — ten runs said it could not, and the difference is two flags

**Status:** MEASURED (2026-08-17, MAIN, $0 local Metal). `EF3000`, 3,000 steps, lr 2e-5,
`--ce-fraction 0.0 --softplus-fraction 0.0`, seed 20260715, rc=0, 1,445 s.
**Axis:** `[macOS-MPS training-signal]`, `quantized_exact_seg` on the trainer's own instrument.
**No score claim. Frontier untouched.** Payload `/Volumes/APDataStore/pact/ddm_ce1/EF3000/`.

## The verdict — the pre-registered fork took branch 1

```
best_step          = 3000        (was 0 on every prior run)
improved_over_init = True        (was False on every prior run)
endpoint           = −2,286 flips BELOW init   (min over all 31 evals, AT the cap)
```

**Pre-registered before the result landed:** `best_step > 0` ⇒ *the aligned objective CROSSES its
handed-down floor and the ten-run verdict is configuration, not physics.* That is what happened.

−2,286 is **3.78× the measured A/A noise floor** (605 flips at step 600), so the crossing is well
outside n=1 nondeterminism. The tail is still falling: **−430 / −356 / −111 / −234** flips per 100
over the last four evals. **It did not converge — it ran out of steps.**

## What this does — and does NOT — overturn

**verdict_scope: INSTANCE** — one seed (20260715), one config, 3,000 steps, on the burn-2/PR130-lineage
QAT base, measured on the trainer's advisory `quantized_exact_seg`. An existence result is
INSTANCE-scoped until repeated; the second seed is owed.

⚠ **MAIN's first headline said `ddm_l3000_no_descent_verdict_20260817` is REFUTED. WITHDRAWN, same
turn, before re-citation — and the truth is better than the claim.** L3000's scope reads:

> **verdict_scope: FORMULATION** — this training formulation (QAT fine-tune from the PR130-lineage
> init **under the CE → softplus_margin → expected_flip curriculum**) does not descend …

**The curriculum is INSIDE its formulation.** EF3000 ran `ce=0, softplus=0` — a *different*
curriculum, therefore a *different formulation*. Nothing in EF3000 tested L3000's formulation at a
longer window. **L3000's verdict is INTACT AS SCOPED.** It is SUPERSEDED FOR ROUTING — its
formulation is no longer the live one — which is a different and weaker statement than refuted.

**And L3000 named this experiment itself.** Its own §"Not closed by this":

> a *different* formulation (different init, **different curriculum shape**, absolute-step stage
> boundaries, a schedule that does not spike +49,580 flips in its first 100 steps) is untested.
> **The CE phase's opening excursion is where the entire debt is created** — every later phase is
> repayment that never finishes.

That is EF3000's design, its mechanism, and its result, written a day early. **The scope ladder did
its job; the apparatus worked.** My "the verdict named its own blind spot and nobody read it as one"
was false — L3000 read it correctly and said so under a heading that means exactly that. The reader
who missed it was me, twice: once treating the no-descent result as closing the question, and once
today calling the named successor a refutation. Both corrected here.

The measured comparison stands unchanged:

| arm | curriculum | steps | endpoint vs init |
|---|---|---:|---:|
| C0/A1/A2/A3 | 81.19% `ce` | 600 | +8,049 … +27,170, `best_step=0` |
| `L3000_off` | 81.19% `ce` | 3,000 | +4,887, `best_step=0` |
| `CE0` | 0% `ce` | 600 | +4,852, `best_step=0` |
| `EF0` | 100% `expected_flip` | 600 | +636, `best_step=0` |
| **`EF3000`** | **100% `expected_flip`** | **3,000** | **−2,286, `best_step=3000`** |

Both ingredients were necessary. EF0 (aligned, short) reached parity but did not cross;
`L3000_off` (long, misaligned) closed 43.5% of its debt and stalled. **Aligned × long crosses.**

| arm | curriculum | steps | endpoint vs init |
|---|---|---:|---:|
| C0/A1/A2/A3 | 81.19% `ce` | 600 | +8,049 … +27,170, `best_step=0` |
| `L3000_off` | 81.19% `ce` | 3,000 | +4,887, `best_step=0` |
| `CE0` | 0% `ce` | 600 | +4,852, `best_step=0` |
| `EF0` | 100% `expected_flip` | 600 | +636, `best_step=0` |
| **`EF3000`** | **100% `expected_flip`** | **3,000** | **−2,286, `best_step=3000`** |

Both ingredients were necessary. EF0 (aligned, short) reached parity but did not cross;
`L3000_off` (long, misaligned) closed 43.5% of its debt and stalled. **Aligned × long crosses.**

## Scope, stated plainly

**INSTANCE.** This is the trainer's own advisory `quantized_exact_seg` on the burn-2 base at n600
— NOT the `hv1` frontier vehicle, NOT byte-closed, NOT exact-eval'd. In S units the endpoint is
**−1.938e-3 seg** (2,286 / 117,964,800 × 100), which would be 20.2% of the −0.0095973 gap **if it
transferred and byte-closed** — and neither is measured. The measured claim is narrow and real:
*this regime descends.* Whether the descent survives realization and reaches the shipping vehicle
is the next question, not this one.

## Wall-clock: a fourth term the law does not model

Predicted 1,552.3 s (`F + 3000·r + 30·e`, the exact `L3000_off` configuration). Measured **1,445 s
— 6.9% faster at identical steps and evals.** EF0 was −2.7% on its point. The residual is not noise:
**`r` depends on WHICH LOSS IS ACTIVE.** `expected_flip` is cheaper per step than `ce`, and the
cadence law treats `r` as curriculum-independent. Same genus as this morning's `e`-vs-`r` conflation,
one level in: a rate fitted across a mixture is a mixture rate.

## NEXT

1. **Second seed.** `best_step > 0` rests on n=1. ~24 min by the (corrected) law.
2. **Keep going.** It was still descending at 3,000 with the anneal exhausted. The window is not
   the limit that has been found yet — the SCHEDULE is. 6,000 steps ≈ 48 min.
3. **The transfer question, which is now the real one.** Does an aligned-objective descent on this
   base produce a byte-closeable candidate on the frontier vehicle? That is a different measurement
   and it is where the S actually lives.
4. **`cos(sign g)` for unmeasured objectives.** 0.6185 is the best of three, not a ceiling.
5. **Re-price the wall-clock law per-curriculum**, or quote `r` with its active loss.
