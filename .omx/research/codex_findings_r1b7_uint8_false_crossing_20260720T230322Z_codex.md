# Codex finding — R1b7 integer false-crossing admission

Date: 2026-07-20  
Lane: `r1b7_uint8_survival_carrier`  
Authority: `[macOS-CPU advisory]`, `score_claim=false`  
Pointer: `0.1910828242 [contest-CPU] UNMOVED`

## Finding

`BUG_CLASS_EXTINCTED`: the bounded integer search admitted a proposal whenever
its post-perturbation target/rival margin was positive. It did not require the
baseline decision to be wrong. Therefore an already-correct site could be
mislabeled as a newly crossed hard decision.

All four originally reported candidates were already correct at baseline:

| Site offset | Baseline margin | Proposal margin |
|---:|---:|---:|
| 2 | 0.4798073769 | 0.4798383713 |
| 3 | 0.4778380394 | 0.4778385162 |
| 6 | 0.3192958832 | 0.3192944527 |
| 7 | 0.6401209831 | 0.6401195526 |

The complete set of original positive-margin candidates is those four rows, so
the corrected wrong-to-target crossing count for the measured top-8 prefix is
exactly zero. This is a re-derivation from the existing receipt; no measurement
was rerun.

## Permanent fix

`tools/measure_r1b7_uint8_survival_carrier.py` now admits a hard crossing only
when all four conditions hold: baseline prediction is not the target, proposal
prediction is the target, baseline margin is at or below the configured gate,
and proposal margin is above it. Regression tests cover the already-correct
false-positive and the real wrong-to-target transition.

The four-site sealed archive remains valid diagnostic custody: it changes Seg
flips by zero, slightly harms Pose, costs 184 bytes, and does not pay its rate.
It is no longer described as a successful integer counterarm.

## Scope and remaining blocker

The full 498-site fixed arm remains measured nonpositive and byte-identical to
R1b6. The top-8 bounded integer prefix produced no new crossing. A marginal
receiver-composed prefix waterfill was not executed; individual prefixes remain
open until each is measured with collateral, Pose debt, and exact byte delta.

The corrected receipt, result memo, DAG feed, probe supersession, and equation
domain refinement all require MAIN landing review before merge.
