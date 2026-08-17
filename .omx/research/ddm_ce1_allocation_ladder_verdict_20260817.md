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

## Where the residue lives (INFERRED — the named next measurement)

After alignment, ~636 flips remain. `rt1`/`td1` measured that **95% of seg error is MANUFACTURED by
the render→SegNet round trip, not label error** (0.028155 S, 2.9× the whole remaining gap). If most
error is created downstream of the weights, no training-side lever can reach it — which is
consistent with alignment collapsing the gap 13.6× and then stopping at parity.

**This is a hypothesis, not a measurement**, and it has a $0 falsifier: decompose all three
endpoints' remaining flips into manufactured-vs-label. If the manufactured fraction is INVARIANT
across arms while the total varies 13.6×, the floor is in the realization path, not the trainer.

## The wall-clock prediction, scored (pre-registered this morning)

`predicted_total_seconds_with_cadence(600, 25)` = **865.5 s**. EF0 measured **842.0 s** — **−2.7%**,
and CE0 (byte-identical cadence and length) measured 865.5 s, so the two identically-configured runs
differ by 23.5 s = 2.7%: **the model's error equals the run-to-run spread.** The three-point
cadence fit (`F=122.29 · r=0.22267 · e=25.400`) reproduces out-of-sample on its first test.

## NEXT

1. **`EF3000` FIRED** (pid 74123, 3,000 steps, 100% `expected_flip`, ~26 min): EF0 was **still
   descending at the cap** (2,019 → 927 → 636 over its last 75 steps) with lr annealed to 2e-7 — the
   descent slowed because the ANNEAL ran out, not the objective. Fractions stretch, so 3,000 steps
   is 5× the aligned window, not a repeat. **Pre-registered:** `best_step > 0` ⇒ the aligned
   objective CROSSES and the ten-run verdict is configuration, not physics · `best_step = 0` with
   endpoint inside the 605-flip floor ⇒ alignment saturates AT parity and the residue is the
   realization path.
2. **Second EF0 seed** (~14 min at this cadence): is +636 real or noise? The floor rests on n=1.
3. **The $0 manufactured-vs-label decomposition** of all three retained endpoints.
4. **`cos(sign g)` for objectives we have NOT measured.** 0.6185 is the best of three, not a ceiling;
   nothing says the aligned-objective axis is exhausted at `expected_flip`.
