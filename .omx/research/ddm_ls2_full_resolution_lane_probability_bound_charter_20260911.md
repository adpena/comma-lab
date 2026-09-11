# LS2 — preserve the shipped probabilities while pricing the remaining Lane context

Disposition: **QUEUED-WITH-A-FIRE-ORDER**. Owner **MAIN**, execution owner **ddm_ls2** after
harvest/assignment. Consumer store:
`/Volumes/VertigoDataTier/pact/ddm_ls2_full_resolution_lane_probability_bound/`.
Fire trigger: MAIN harvests the verified LS1 unit and binds the still-current archive/field.
This document orders no training, paid dispatch, scorer or candidate build during LS1.

Read `.omx/tmp/codex_runs/_common_contract.md` and the LS1 memo first. Reuse LS1's retained
integer-frequency rows, exact context assignments and full receiver checkpoints; do not repeat
the full trace if their hashes and move-44 binding remain valid. Rebind after a pointer change.

## Why this is the next measurement

LS1 measured 17,534.216 B of Miller–Madow estimated gross gain in its receiver-visible
constant-cell table. That table discards the shipped model's within-bin probability information
and grants its many fitted parameters for free. It therefore proves neither that a small
full-resolution correction can recover that gain nor that a better model cannot exceed it.
The strongest granted-row map costs 594,003 physical bytes and is not a carrier candidate.
The remaining question is the actual gain of a compact correction that retains the full prior.

The next object is a **finite, explicitly priced additive probability model**, not another
constant-cell entropy table and not a transfer of PC2's pose-carrier price.

## Concrete measurement

1. Keep every shipped probability, decoded token, model, adaptive state and counted configuration
   fixed. Use all 600 pairs. Form causal count/expected-mass features using the two LS1 receiver
   distance bins and the HPAC group phase. Counts update only from completed earlier planes;
   same-frame reads obey the actual HPAC group order. Explicitly verify this causality.
2. Predeclare two distinct compact families before fitting: separate context features and their
   joint distance feature. Freeze the inherited weights for the first comparison. State the
   exact coefficient count, integer representation and header cost for each; no empirical table
   or spatial selector is free. Preserve the original probability at zero added coefficients.
3. Reuse TC1's full-resolution categorical log-loss/supporting-plane instrument. Report numerical
   lower-loss bounds for the stated coefficient box, with any omitted symbols receiving explicit
   optimistic credit. Price rounded coefficients on **every** symbol through the real RC64
   frequency conversion. Retain all proposed parameter vectors, not only a winner.
4. Add seeded pair-level held-out validation across the full n600 population. Report total,
   per-class and patch-row-boundary gains; fitting and selection bias must remain visible.
   Old HC1/HC2/MI1/TC4 results are recall constraints, not current numerical predictions.
5. Write a receiver-work estimate from the actual required operations. TC4's later failed contest
   result forbids treating a tiny weight payload as evidence of a cheap decoder. Any production
   proposal needs a measured native implementation cost and the live timing gate.

## Fire and stop rules

- If neither specified family can achieve positive net loss reduction after its exact parameter
  charge, record a formulation-scoped refusal and stop. Do not extrapolate that to all context
  models or generators.
- If a rounded full-n600 model has a net saving and the source-defined receiver timing gate can
  admit it, queue **one** byte-identical lossless recode for MAIN under the applicable build charter.
  A supporting bound alone is insufficient to fire a candidate.
- Compare the actual net result with the live demand, rederived from components. A smaller
  improvement may justify a lower exact row; it must not be called a sub-0.12 solution.
- A successful build must retain twins, preserve public full-field identity, charge every
  video-fitted byte and meet the live complete-evaluation timing contract before any exact dispatch.
  MAIN owns that future authorization and lane claim; this charter does not fire it.

Deliver `RESULT.json`, exact parameter payloads, numerical-bound receipts, held-out/full-population
tables and a typed fire-or-refusal receipt in the consumer store. Finish with an honest finite-family
verdict. This remains a $0 scorer-free measurement until a separately bound candidate exists.

<!-- # FORMALIZATION_PENDING: proposed measurement of two finite probability-correction families; no new measured law or runtime actuator is claimed. -->
