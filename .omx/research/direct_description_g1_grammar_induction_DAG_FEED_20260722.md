---
schema: ddm_g1_grammar_induction_dag_feed.v1
date_utc: 2026-07-22
lane_id: ddm_g1_grammar_induction
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true
---

# FEED — DDM G1 per-stratum grammar induction

## Executable readiness DAG delta

```text
[frozen n600 lstars, SHA-bound]
        |
        +--> [Movable external islands]
        |       -> 48px Hungarian slots
        |       -> BIRTH / active PERSIST / DIE
        |       -> centroid + absolute-or-morph shape
        |
        +--> [Lane fitted curves]
        |       -> coherent slots
        |       -> EVENT + CENTER + WIDTH + DASH + RANGE
        |       -> birth absolute / active persistence delta / absent free
        |
        +--> [all-class transition support]
                -> ARC_EVENT + ARC_VERTEX
                -> scoped negative at measured rate/fidelity
        |
        v
[per-production actual coder contest]
  Brotli q11 vs raw LZMA1 preset1/dict1MiB vs zlib9
  codec tag + raw/coded lengths + framing counted
        |
        v
[G1S1 parse + semantic reconstruction]
  exact rows equality-gated; lossy rows mask-compared
        |
        v
[two-part MDL table + tolerance ladders]
        |
        v
[DERIVED clean-rest composition]
  Lane 27,692 B / 0.004945687188
  Movable 29,810 B / 0.000282948812
  union upper 57,502 B / 0.005228635999
        |
        +--> bytes <=60,000: PASS
        +--> d_seg <=0.005: FAIL by 26,971 mask errors
        +--> receiver RGB / through-R / Pose / final ZIP: NOT MEASURED
        |
        v
[NO ARCHIVE, NO DISPATCH, POINTER UNCHANGED]
```

## Triality

- DSL/code: `experiments/direct_description/induce_per_stratum_grammar.py` is a fail-closed,
  research-only typed production/envelope surface. It accepts only `--execution-allowed false`,
  SHA-checks the frozen cache, preserves atomic per-stage checkpoints, and emits no archive.
- DAG: this file. The next consumer is DDM v13 round 2, but only after MAIN landing review.
- Equations: `direct_description_g1_grammar_induction_equations_20260722.md` records the measured
  two-part MDL and composition law as a candidate; no canonical registry append or promotion occurs.

## Six-hook wire-in

1. Sensitivity map: consumes the v12 per-stratum ranking. No scorer forward or Fisher field was run,
   so mask error is not silently promoted to a sensitivity update.
2. Pareto constraint: each row binds `(counted bytes, decoded mask errors, IoU)`. The joint 60-KB / 
   0.005 gate remains false.
3. Bit allocator: production bytes expose the measured rate owners—Lane center 12,367 B, dash
   6,314 B, width 5,632 B, range 3,176 B; Movable shape 27,032 B dominates its knee.
4. Cathedral/autopilot: `research_only=true`, `execution_allowed=false`, no dispatch edge and no
   archive candidate. MAIN review is the next authority boundary.
5. Continual learning: the compact receipt retains the failing joint gate, exact lossless floors,
   and the morph-vs-absolute crossover so later builders do not rediscover them.
6. Probe disambiguators: absolute versus morph shape, delta-dash versus persistent dash, and exact
   rows versus parametric grammar are all emitted and decided by real bytes plus semantic fidelity.

## Decision and negative scope

Admit the table as measured candidate-set evidence; do not admit the grammar as a receiver or score
vehicle. The negative closes only the emitted Lane/Movable/Boundary formulations at their measured
tolerances and coders. It does not close xi transport, receiver-derived cross-stratum constraints,
split/merge syntax, curvelet/shearlet residuals, or future grammars with semantic parse-back.

Blocker delta versus #603: actual predictor-native syntax is now priced, replacing the unmeasured
inventory question on this surface. At 60 KB it owes 26,971 mask-error removals, receiver-visible
RGB/through-R construction, cross-stratum ordering, Pose custody, and final-container byte closure.

## STORES CONSULTED

See the paired landing memo. This FEED additionally binds the compact receipt SHA chain, the three
SSD stage checkpoints, the local lane registration, and both delegation inboxes through the final
measurement checkpoint. It follows `docs/operating_manual_craft_handoff.md` and preserves the
operator's Fisher/margin, corrected-inner-Jacobian, curvelet/shearlet, xi-factorization, and
reverse-waterfill rules for any downstream residual work.

Pointer `0.1910828242 [contest-CPU]` unchanged. MAIN landing review required.
