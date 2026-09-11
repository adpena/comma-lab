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

## OPTIMAL FORM (appended by MAIN for the spawn-site lint)
- Reference form: ls1's exact per-symbol instrument on the SHIPPED decoder (surprise reconciled to the real stream within 0.0006 %) and
  its Miller–Madow oracle ladder; this charter prices the full-resolution probability correction with the SAME instrument and the
  SAME shipped field. Any reduced subset is SCOPE (declared, no verdict); a proxy model or a different coder is MECHANISM (toy-bracketed).
- Provenance pins (sha256 prefixes): ls1 memo b729faa146b62ea3…; this charter as written by ls1 8392a530f73ca2eb…; the shipped field subset6.u8 (a92e7d90…,
  pin it from the encode receipt); pointer move 44 commit 99625f32f / archive 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e;
  retained ls1 store `/Volumes/VertigoDataTier/pact/ddm_ls1/` (record the atlas RESULT sha you consume).

## Prior negatives accounted (operator 2026-08-15)
- ls1: fixed-cell replacements fall short even with free parameters — the full-resolution bound must be priced as MARGINAL savings
  under the shipped model, with the correction's own serialized cost, never as a sum of per-context gains (m164 UNION ≠ SUM; m166).
- gdc4 / available-vs-authoritative field: pin the shipped field by the encoder's receipt; never pass6.u8.
- tc1 / tc2: oracle gain ≠ map cost; report both columns.
- eb2: the receiver holds no partition/pose — a receiver-visible bound may use only the decoded token plane and the carrier.
Serializer commit LAST; rc 17 is NOT a stop; checkpoint as `ddm_ls2` and mark it COMPLETE at the end. Final message ends with the
frontier line `composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`.

<!-- # FORMALIZATION_PENDING: measurement charter; no measured row -->
