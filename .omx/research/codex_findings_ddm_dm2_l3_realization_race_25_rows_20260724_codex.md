---
research_only: true
execution_allowed: false
score_claim: false
promotion_eligible: false
pointer_moved: false
main_review_required: true
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
---

# Codex findings — DM2 exact semantic rows through L3 RGB realization

## Verdict

`MEASURED / INSTANCE`: all 25 sealed DM1 semantic records are independently
realizable through camera-resolution RGB, real `R`, uint8, and frozen SegNet in
the bounded candidate menu. Fresh joint composition is also exact, but its
3,960,549-byte price for 1,569 semantic bytes is a `2524.250478x` constructive
upper bound because pairs 55, 60, and 90 required full solved-target fallbacks
after measured non-telescoping union conflicts.

This is not a score, a minimum-preimage certificate, a family verdict, or a
promotion. Pointer: `0.1910828242 [contest-CPU] UNMOVED`.

## Authority and custody

- Authority SHA-256: `21b8f410c6fd6f65db48218b12a30dfb6d9c893352bb3ce43274db7aa50bbf9b`.
- Sealed DM1 receipt SHA-256:
  `4c2fe77927e300e341d5ce9ce00ae8a37c58dbebbde8e5860fe514958990de28`.
- Final DM2 receipt:
  `.omx/research/ddm_dm2_l3_realization_race_25_rows_20260724T133300Z/ddm_dm2_l3_realization_race_receipt.json`.
- Final DM2 receipt SHA-256:
  `8897241b7fc0ded7d4d6d1100c4d23ea162111050754e36c8ad8b3e57e294229`.
- Preserved adversarial round-0 receipt SHA-256:
  `a429194edcc129d7d84a96a87918cead5e09d35769485216d4e52fb89d707c37`.
  It records the initially failed aggregate semantics for pairs 55, 60, and 90.
- Torch CPU threads were pinned to 4; deterministic algorithms and seed 1234
  were enabled. Every named input and scorer weight was SHA-revalidated.
- The 25 row checkpoints retain their producer config/implementation SHA pair;
  the joint-only correction accepts exactly that pair and records it in final
  custody rather than pretending the rows were regenerated.

## Headline exchange

| Quantity | MEASURED value | Interpretation |
|---|---:|---|
| Exact joint semantic price | 1,569 B | DM1 exact shared-context record |
| Exact joint realized L3 RGB price | 3,960,549 B | lzma9 winner, parse-back exact |
| Realized / semantic | 2524.250478x | constructive upper bound |
| Positive collateral byte equivalent | 0 B | collateral was beneficial; no byte credit taken |
| Joint collateral score delta | -0.2857199805 | off-support Seg plus Pose |
| Joint rate score delta | +2.6371670109 | exact bytes at contest rate term |
| Joint score delta | +2.3508358312 | advisory; not a contest score |
| #613 tangent context | 25.630972x | context only, no box arithmetic |

`DERIVED`: one byte costs `25/37,545,489 = 6.658589531e-7` score units.
The current aggregate is rate-dominated even though collateral improves the
advisory Seg/Pose terms.

## Per-row ratio table and first rung

Every row status is `SUCCESS_EXACT_L4_RECORD_THROUGH_L3_RGB`. “Pose Δ” is exact
affected-pair PoseNet first-six MSE delta. A negative value is nonharm. “Next”
is the implied measurement, not authorization to dispatch or train.

| Row | Pair | Home | n | Sem B | Selected L3 candidate | RGB B | RGB/Sem | Pose Δ | Joint ΔS | Next exact measurement |
|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 523 | FIBER | 34 | 157 | local_r8_q128 | 3,265 | 20.796 | -0.032469 | +0.001755 | compose pair; exact Seg/Pose remeasure |
| 1 | 523 | SKELETON | 291 | 258 | global_all_q255 | 498,817 | 1933.399 | -174.650082 | +0.289993 | corrected-J/shearlet hard admission |
| 2 | 523 | FIBER | 36 | 161 | local_r64_q32 | 58,797 | 365.199 | -0.315034 | +0.037499 | compose pair; exact Seg/Pose remeasure |
| 3 | 54 | FIBER | 23 | 155 | local_r32_q32 | 6,961 | 44.910 | -0.046641 | +0.033255 | compose pair; exact Seg/Pose remeasure |
| 4 | 90 | SKELETON | 6 | 162 | local_r128_q128 | 141,473 | 873.290 | -0.802449 | +0.092635 | compose pair; exact Seg/Pose remeasure |
| 5 | 90 | SKELETON | 2 | 156 | local_r32_q128 | 12,601 | 80.776 | +0.031546 | +0.008195 | pose-null/SE(3) candidate, then compose |
| 6 | 446 | SKELETON | 190 | 223 | global_all_q255 | 516,581 | 2316.507 | -177.135070 | +0.295624 | corrected-J/shearlet hard admission |
| 7 | 446 | FIBER | 51 | 164 | local_r64_target | 106,621 | 650.128 | -0.015875 | +0.062667 | compose pair; exact Seg/Pose remeasure |
| 8 | 0 | SKELETON | 4 | 158 | local_r1_q128 | 292 | 1.848 | -0.003222 | +0.000178 | compose pair; exact Seg/Pose remeasure |
| 9 | 14 | FIBER | 1 | 150 | local_r32_target | 12,037 | 80.247 | -0.016019 | +0.006236 | compose pair; exact Seg/Pose remeasure |
| 10 | 327 | FIBER | 3 | 148 | local_r128_q64 | 66,429 | 448.845 | +0.565948 | +0.049526 | pose-null/SE(3) candidate, then compose |
| 11 | 60 | SKELETON | 2 | 158 | local_r64_q64 | 42,701 | 270.259 | +0.096446 | +0.026449 | pose-null/SE(3) candidate, then compose |
| 12 | 60 | FIBER | 3 | 148 | local_r0_target | 105 | 0.709 | +0.000809 | +0.000068 | pose-null/SE(3) candidate, then compose |
| 13 | 323 | SKELETON | 5 | 161 | local_r8_target | 1,617 | 10.043 | -0.006640 | +0.002010 | compose pair; exact Seg/Pose remeasure |
| 14 | 323 | SKELETON | 17 | 167 | local_r4_q32 | 1,089 | 6.521 | -0.002806 | -0.000632 | compose pair; exact Seg/Pose remeasure |
| 15 | 38 | SKELETON | 8 | 165 | global_all_q255 | 515,377 | 3123.497 | -189.543368 | +0.300112 | corrected-J/shearlet hard admission |
| 16 | 42 | SKELETON | 2 | 158 | local_r128_q64 | 215,597 | 1364.538 | -0.111315 | +0.140765 | compose pair; exact Seg/Pose remeasure |
| 17 | 4 | SKELETON | 12 | 178 | local_r128_target | 155,761 | 875.062 | -0.138304 | +0.100210 | compose pair; exact Seg/Pose remeasure |
| 18 | 55 | SKELETON | 2 | 159 | local_r32_target | 11,057 | 69.541 | -0.003768 | +0.007195 | compose pair; exact Seg/Pose remeasure |
| 19 | 55 | SKELETON | 2 | 153 | global_all_q2 | 122,229 | 798.882 | -0.000049 | +0.081221 | corrected-J/shearlet hard admission |
| 20 | 56 | SKELETON | 3 | 161 | local_r32_target | 12,049 | 74.839 | -0.015549 | +0.007556 | compose pair; exact Seg/Pose remeasure |
| 21 | 56 | FIBER | 18 | 158 | local_r64_q64 | 46,345 | 293.323 | -0.012034 | +0.028962 | compose pair; exact Seg/Pose remeasure |
| 22 | 56 | SKELETON | 4 | 159 | local_r1_q128 | 217 | 1.365 | -0.058771 | +0.000128 | compose pair; exact Seg/Pose remeasure |
| 23 | 16 | FIBER | 6 | 151 | local_r16_q32 | 2,089 | 13.834 | +0.021551 | +0.004183 | pose-null/SE(3) candidate, then compose |
| 24 | 16 | SKELETON | 2 | 156 | global_all_q255 | 498,677 | 3196.647 | -189.126063 | +0.288850 | corrected-J/shearlet hard admission |

Twenty selected rows are local and five are global. Twenty are Pose-nonharm.
Only row 14 has negative independent joint ΔS; no row is promoted from this
advisory measurement.

## Required decompositions

| Decomposition | Rows | Independent semantic B | Independent realized B | Ratio of sums | Pose-nonharm |
|---|---:|---:|---:|---:|---:|
| FIBER | 9 | 1,392 | 302,649 | 217.420259x | 6 |
| SKELETON | 16 | 2,732 | 2,746,135 | 1005.173865x | 14 |

`DERIVED / INSTANCE`: SKELETON is `4.623184x` dearer by the independent
ratio-of-sums. This supports “interface placement is dearer here,” but support
size, pair identity, and the five global selections confound a causal or
family-level statement.

| Support n | Rows | Independent semantic B | Independent realized B | Ratio of sums |
|---|---:|---:|---:|---:|
| [1,4) | 10 | 1,547 | 993,482 | 642.199095x |
| [4,16) | 7 | 1,134 | 816,826 | 720.305115x |
| [16,64) | 6 | 962 | 223,078 | 231.889813x |
| [64,256) | 1 | 223 | 516,581 | 2316.506726x |
| [256,inf) | 1 | 258 | 498,817 | 1933.399225x |

`MEASURED`: price is not monotone in support count. Support geometry,
quantization threshold, pair context, and whether the bounded menu falls back
to a global write matter more than `n` alone.

## Family (d) versus family (b), context only

`FORMULATION implication`: a family-(d) semantic emitter cannot claim the DM1
1,569-byte information price as a receiver price. Its emitted sufficient
statistic must produce camera RGB that survives this same nonlinear
R/uint8/Seg/Pose gate. The present literal fixed-quantum realization is
rate-inadmissible, so the next useful family-(d) measurement is a compact
corrected-inner-Jacobian/shearlet emitter, not more sparse record coding.

This does not prove family (b) wins. A family-(b) pixel/field realizer must pay
the same gate and can exploit cross-row/pair structure absent from this bounded
menu. The #613 154,522-byte value remains tangent context only; the measured
joint realization is `25.630972x` it, with no slack-pool subtraction.

## Canonicalized system intelligence

- Callable equation:
  `src/tac/canonical_equations/ddm_dm2_semantic_realization_exchange.py`.
- Equation ID: `ddm_dm2_semantic_to_realized_rgb_exchange_v1`.
- Registry event: append-only line 840, event-line SHA-256
  `1b0ad0d84610110338a5ab28e112d18a39eb7b28ee24ac33c367a50f1435ae17`.
- DAG leg:
  `.omx/research/ddm_dm2_l3_realization_race_25_rows_DAG_FEED_20260724.md`.
- Machine-readable per-row first rungs live in the final receipt.

## Directive-consumption table

| Directive UTC | Status | Application |
|---|---|---|
| 2026-07-19T19:42:07Z | `CONSUMED` | Exact coder bytes and the measured rate dual govern every ratio. No blanket fix or inferred positive-EV admission. |
| 2026-07-19T19:48:01Z | `CONSUMED` | Pairwise margins remain the semantic metric. Global next rungs use corrected inner-Jacobian and curvelet/shearlet support; Fourier is excluded. Pose-harm rows name pose-null/SE(3) remeasurement. |

No newer per-arm or broadcast directive was present through cursor
`2026-07-21T13:15:53Z`.

## Round-1 adversarial review

1. `FOUND AND FIXED`: independently exact row writes did not telescope. The
   original aggregate failed rows on pairs 55, 60, and 90. That receipt is
   preserved; the final aggregate remeasures all pairs and labels full-target
   fallbacks as an upper bound.
2. `CLEAN / SCOPED`: every final joint record round-trips to the exact camera
   bytes and every target row is freshly checked after composition.
3. `CLEAN / SCOPED`: “minimal” means the best exact member of the finite
   preregistered menu after bounded Pose screening. No global optimizer or
   minimum certificate is claimed.
4. `CLEAN / PRICED`: five rows are Pose-harmful in the screened menu. They are
   not hidden or declared pose-safe; exact Pose deltas enter collateral and
   each names a pose-null next measurement.
5. `CLEAN / NON-AUTHORITY`: no archive, contest eval, training, descent, paid
   dispatch, frontier mutation, or score claim occurred.
6. `MAIN REVIEW REQUIRED`: review the full-target fallback scope, prior
   checkpoint custody exception, exchange-law registry row, and the
   SKELETON/FIBER interpretation before merge.

Machine-readable review receipt:
`.omx/research/reviews/ddm_dm2_l3_realization_race_round1_20260724.json`.

| Clean pass | Evidence | Result |
|---:|---|---|
| 1 | ruff, py_compile, 10 focused tests, staged diff hygiene, source/receipt SHA closure | `CLEAN` |
| 2 | 52 DM2/DM1/receiver regressions plus blind checkpoint, aggregate, ratio, pointer, and registry re-derivation | `CLEAN` |
| 3 | 77 equation/registry/receiver tests, 15-record codec fuzz, corruption refusal, ten input SHA closures, targeted lane and dispatch-import audits | `CLEAN` |

The repository-wide lane validator separately reports 110 inherited
missing-evidence paths in older lanes. That pre-existing debt was not counted
as a DM2 clean pass; this lane's targeted structure was checked directly.

## STORES CONSULTED

Authority prompt; `CLAUDE.md`; `AGENTS.md`; craft handoff manual; binding DM1
and IS1 findings/receipts/configs; PF2 index and receipt; G2 source config and
start receipt; `upstream/modules.py`; scorer weights; full-kernel, lattice,
v17/v19, menu1/fr1 realization code; lane registry; lane maturity audit;
subagent-progress ledger; canonical-equation registry; per-arm inbox; broadcast
inbox. No stale score or branch snapshot was used as authority.
