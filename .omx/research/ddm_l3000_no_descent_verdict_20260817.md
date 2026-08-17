# 5× the window does NOT reach parity — the seg training regime asymptotes ABOVE its own initialization

**Status:** MEASURED (2026-08-17, MAIN, $0 local Metal). `L3000_off`, 3,000 steps, lr 2e-5,
band OFF, rc=0, 1,552 s. Payload `/Volumes/APDataStore/pact/ddm_jr1/L3000_off/` (13 saves).
**Axis:** `[macOS-MPS training-signal]`, `quantized_exact_seg` on the trainer's own instrument.
No score claim. Frontier untouched.

## The verdict

`best_step = 0` · `improved_over_init = False` · final seg **exactly equals** init seg
(0.00028616163465711804) because the best checkpoint *is* the initialization.

| step | phase | flips above init | Δ per 100 |
|---:|---|---:|---:|
| 100 | ce | +49,580 | +49,580 |
| 900 | ce | +16,680 | −3,378 |
| 1,500 | ce | +6,510 | −743 |
| 1,800 | softplus_margin | **+7,469** | **+320 (rises)** |
| 2,400 | softplus_margin | +5,787 | −19 |
| 3,000 | expected_flip | **+4,887** | **−104** |

**I predicted this was a truncation. It is not.** From `A2_repeat`'s tail (−765 flips/100 at the
600 cap, +8,654 above init) I estimated ~1,131 more steps to parity. The real run went **2,400
more steps**, closed only **3,767** of those flips, and its rate **decayed 32×** from the CE
phase (−3,378/100) to the tail (−104/100). At the final rate the remaining 4,887 flips would
need ~4,700 more steps *and the rate is still falling*. This is an **asymptote above init**, not
a truncation. Linear extrapolation of a decaying tail was the wrong instrument, and it was mine.

**The two runs are not comparable at matched steps** — exactly the caveat pre-registered in
`ddm_wallclock_prefix_bias_law_20260817`. `--ce-fraction 0.50` is a fraction *of the run*, so CE
ends at step 300 in the 600-step arm and step 1,500 in the 3,000-step arm. At step 600 the short
arm was already in softplus at +8,654 while the long arm was still in CE at +27,047. Only the
**end states** compare: 5× the steps ends 43.5% closer to parity, with the rate 7.4× slower.

## What it closes

Every stock arm now measured: C0/A1/A2/A3 (600 steps, three decades of lr) all `best_step = 0`;
`b2e` (3,000 steps, lr 2e-7) did not descend; `L3000_off` (3,000 steps, lr 2e-5) does not descend.

**verdict_scope: FORMULATION** — this training formulation (QAT fine-tune from the PR130-lineage
init under the CE → softplus_margin → expected_flip curriculum) does not descend below its own
initialization on `quantized_exact_seg`, at any tested lr or window length.

**This moots R6 rather than refuting it.** R6 (band objective in training) is a *reallocation of
gradient mass* inside this process. `rg1b` measured the misallocation as real (2.161% of gradient
mass on a band holding 99.22% of the debt — 45.9×, 83.3°), and that finding stands. But a better
allocation of a descent that ends **+4,887 flips worse than where it started** cannot supply the
gap. The blocker was never the objective or the judge; it is the regime. This is jr1's
pre-registered DIRECTION NULL branch, arrived at with a measured curve instead of a null result:
*"the pixel-reweighting family closes at FORMULATION scope with a measured law — a real result."*

**Not closed by this:** a *different* formulation (different init, different curriculum shape,
absolute-step stage boundaries, a schedule that does not spike +49,580 flips in its first 100
steps) is untested. The CE phase's opening excursion is where the entire debt is created —
every later phase is repayment that never finishes.

## The wall-clock prediction, scored

Pre-registered **24.4 min** (1,463 s), before the run. Actual **1,552 s** — **+6.1%**.
b2e's model predicted 9,972 s (**6.4× off**); naive re-extrapolation from the 600-step end-to-end
rate predicted 2,040 s (**+31%**). The two-point `(F = 144.3 s, r = 0.4395 s/step)` separation
held.

**The residual, honestly bounded.** `L3000_off` wrote 13 saves vs `A2_repeat`'s 7 — 6 extra ×
14.8 s each accounts for the full +89 s, exactly the caveat declared in advance. But this is an
**attribution, not an isolated measurement**: two equations, three unknowns. It also does not
close cleanly against b2e's 50-step point (predicting ~107 s vs the measured 166.30 s), so either
the smoke saved more often or a further term is unmodelled. Recorded as consistent-with, not
measured.

## NEXT_IF_RESUMED

1. **Do not fire Leg C.** It ranks damage *rates* between objectives inside a regime that does
   not descend. The comparison is well-powered (72-flip floor, 23× headroom) and would be
   answering a question that no longer routes.
2. The live question is the **opening excursion**: +49,580 flips in the first 100 CE steps, which
   every subsequent phase spends the run repaying. Ask why CE creates that debt before asking
   which objective repays it faster.
3. Pin stage boundaries in **absolute steps** before any future length comparison on this
   trainer, or length and schedule stay confounded.
4. Re-price with `(F = 144.3 s, r = 0.4395 s/step, ~14.8 s/save)`; a 3,000-step arm is ~26 min,
   not 2.8 h.
