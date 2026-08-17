# The seg "wall" is 92.7% CONFIGURATION — aiming the budget at the aligned objective closes 13.6× of it

**Status:** MEASURED (2026-08-17, MAIN, $0 local Metal). Three 600-step arms, **same init, same seed
(20260715), same lr (2e-5), same everything except the curriculum FRACTIONS.**
**Axis:** `[macOS-MPS training-signal]`, `quantized_exact_seg` on the trainer's own instrument.
No score claim. Frontier untouched.

## The ladder

| arm | curriculum | `cos(sign g)` | flips above init @600 | vs A/A noise floor (605) |
|---|---|---:|---:|---:|
| control (`A2_repeat`) | 81.19% `ce` → `softplus` → `expected_flip` | 0.2087 (weighted) | **+8,654** | 14.3× |
| `CE0` | `--ce-fraction 0.0` | mixed | **+4,852** | 8.0× |
| **`EF0`** | `--ce-fraction 0.0 --softplus-fraction 0.0` | **0.6185** | **+636** | **1.05×** |

**control → EF0: 13.6×. 8,018 of the control's 8,654 excess flips removed = 92.7%.**
Monotone at every one of the six checkpoints (100/200/300/400/500/600). The control→EF0 gap is
**4–13× the measured absolute A/A spread** (605–1,777 flips), so the ORDERING is well outside n=1 noise.

**One flag. No code change.** `--ce-fraction`/`--softplus-fraction` already existed with no validator;
`_phase_for_step` reads them as fractions of the run.

## What this does and does not overturn

**`best_step = 0` on all three.** EF0 never dipped below init at any eval point, so the ten-run
statement *"this formulation does not descend below its own initialization"* survives **literally**.

**But the margin collapsed from 14.3× the noise floor to 1.05×.** +636 sits INSIDE the measured
absolute A/A spread at step 600 (605 flips). At n=1 this endpoint is **indistinguishable from
parity**, where the control was unambiguously above it. That is a different regime, not a different
number.

**MAIN's CE0 read was UNDERSTATED.** I graded CE0's falsifier PARTIAL and wrote that the no-descent
floor "is NOT the CE phase," then pre-registered the middle fork — *allocation is real but inert*.
**EF0 refutes that bet.** CE0 was PARTIAL because it removed only the worst phase while leaving
`softplus_margin` (cos 0.5235) holding 18% of the budget; the residual was the residual of a
still-misaligned mixture, not of the mechanism. Corrected here at source.

## Why the sweep never found this

Nine retained runs swept **three decades of learning rate** and **two window lengths** across a
curriculum shape that was **never varied once**. `--ce-fraction 0.50` / `--softplus-fraction 0.85`
are inherited, underived, and — being fractions OF THE RUN — invariant to exactly the two axes that
were swept. The search explored the axis that didn't matter and held the one that did.
Sister of the CHARTER-TIME OPTIMAL-FORM LAW: the naive element was born in the config, and every
downstream verdict inherited it.

## Where the residue lives — MAIN's first reading was WRONG, and it inverts

⚠ **CORRECTED at source, same turn, before this memo was cited again.** I wrote and told the
operator that the residue is *"`rt1`'s manufactured-error residue in the realization path
(render→R→uint8→SegNet), not the weights,"* implying it is downstream of training and unreachable.
I cited "95% manufactured" four times without re-reading `rt1`. Re-read at source
(`ddm_rt1_seg_roundtrip_decomposition_20260816.md`), **three things are wrong:**

1. **`R` supplies EXACTLY ZERO** flips on piecewise-constant content (`rt1` §ANSWER-FIRST item 5,
   MEASURED). Naming "render→**R**→uint8→SegNet" as the mechanism is false; the R operator is
   innocent. The manufacture happens between the labels we ship and the argmax read back — but
   not in R.
2. **The residual is a TIE, not a wall** (item 6). At **98.3%** of flips the wanted class is
   already the RUNNER-UP; median logit deficit **0.105**; half the axis needs **< 0.1 logits**;
   84.5% < 0.3; none > 3 — while *correct* pixels sit at margins of 3–10. `rt1`'s own words: *the
   render put the scorer within a hair of the answer and lost on the last step.* A 0.1-logit tie
   is precisely what a training-side lever reaches. **My "no training-side lever can reach it" is
   the opposite of what the measurement implies.**
3. **`rt1`'s own verdict routes the seg axis TO THE RENDERER** — after its byte-carrying
   correction channel passed the coder gate (32,270 B) and then FAILED its realization bar
   (η 0.6235 vs 0.753, non-supplier at +0.0025 S). The measured routing is *toward* the trained
   object, not away from it.

Numbers corrected too: the round trip is **33,743 flips = 0.028604 S**, **96.6%** of the seg axis
and **2.98×** the gap (I quoted 0.028155 and 2.9×). 99.22% sits exactly ON the transmitted label
boundary; the label interior carries **7 flips in 104 million pixels** (203,000× enrichment);
flips are isolated single pixels (mean run 1.110). The lever `rt1` names is **sub-pixel edge
placement on the Road↔Lane boundary** (that edge alone = 43.4%).

### The arithmetic that reframes this ladder

**The `ce1` init is 33,757 flips. `rt1`'s measured round trip is 33,743. They agree to 0.041%.**

If those are the same object — INFERRED, not proven, since `rt1` measured on `hv1 ep0634` and
these arms initialize on the burn-2/PR130-lineage QAT base — then the whole ladder reads
differently: **all three arms START at the round-trip floor and the training only ever ADDS error
on top of it.** `best_step = 0` then means *no arm went below the floor it was handed*, not *the
regime cannot descend*. The control added 8,654 to that floor; EF0 added 636. The measured
quantity was never the floor; it was the surcharge, and the surcharge is 92.7% configuration.

**Cheap check owed before this is asserted:** confirm whether the `ce1` init object and `hv1
ep0634` share a lineage, or whether 0.041% is a 1-in-2,400 coincidence.

**The $0 falsifier stands but its branches flip.** Decompose all three endpoints into
boundary-tie vs interior flips. `rt1` predicts ~99.2% boundary and a median deficit near 0.105 on
the *floor*; if the ~636 EF0 surcharge shows the SAME tie signature, it is reachable by exactly
the lever that produced it. If the surcharge is interior/systematic while the floor is boundary,
they are two different problems and only one is a training question.

## The wall-clock prediction, scored (pre-registered this morning)

`predicted_total_seconds_with_cadence(600, 25)` = **865.5 s**. EF0 measured **842.0 s** — **−2.7%**,
and CE0 (byte-identical cadence and length) measured 865.5 s, so the two identically-configured runs
differ by 23.5 s = 2.7%: **the model's error equals the run-to-run spread.** The three-point
cadence fit (`F=122.29 · r=0.22267 · e=25.400`) reproduces out-of-sample on its first test.

## NEXT

1. **`EF3000` FIRED** (pid 74123, 3,000 steps, 100% `expected_flip`, ~26 min): EF0 was **still
   descending at the cap** (2,019 → 927 → 636 over its last 75 steps) with lr annealed to 2e-7 — the
   descent slowed because the ANNEAL ran out, not the objective. Fractions stretch, so 3,000 steps
   is 5× the aligned window, not a repeat. **Pre-registered fork — branch 2 RE-SPECIFIED by the
   correction above, before the result landed:** `best_step > 0` ⇒ the aligned objective CROSSES
   its handed-down floor and the ten-run verdict is configuration, not physics · `best_step = 0`
   with endpoint inside the 605-flip floor ⇒ alignment drives the **surcharge** to ~0 and the
   remaining object is `rt1`'s 33,743-flip boundary-tie floor. ~~"the residue is the realization
   path"~~ — **withdrawn**: `R` supplies exactly zero, and a 0.105-logit tie is a training-reachable
   regime, so branch 2 is NOT a hand-off away from training. It says the surcharge is solved and
   the FLOOR becomes the object, reachable by sub-pixel edge placement on Road↔Lane.
2. **Second EF0 seed** (~14 min at this cadence): is +636 real or noise? The floor rests on n=1.
3. **The $0 manufactured-vs-label decomposition** of all three retained endpoints.
4. **`cos(sign g)` for objectives we have NOT measured.** 0.6185 is the best of three, not a ceiling;
   nothing says the aligned-objective axis is exhausted at `expected_flip`.
