# ddm_vf1 evaluator-visible floor — STOP: the strict byte demand is 42,382 B

**Date:** 2026-08-22  
**Disposition:** `STOPPED-CHARTER-ARITHMETIC-DISAGREEMENT`  
**Verdict scope:** `INSTANCE` — the fixed-distortion, strict-sub-0.12 byte threshold in this charter only  
**Authority:** exact arithmetic over the retained DX2 `[contest-CUDA T4, n600]` pointer receipt; this arm ran no scorer and created no score row

## Conclusion

The charter's mandatory independent check disagrees with its stated demand by one byte. At the retained DX2 distortion, the continuous boundary is

\[
B_{0.12}=\frac{(0.12-D_{\mathrm{DX2}})\,37{,}545{,}489}{25}
=137{,}986.8387944436\ \mathrm{B}.
\]

Because the target is the strict inequality `S < 0.12`, the largest admissible integer archive is **137,986 B**. Starting from 180,368 B therefore requires **42,382 B** of reduction. The charter's 42,381 B reduction lands at 137,987 B and scores **0.12000010734016302**, which fails the target. This is not a display-rounding issue.

The charter says to stop and report any independent arithmetic disagreement. Accordingly, VF1 did not adjudicate the downstream load-bearing census, MDL floor, realization pricing, prior-law prediction, or an owed scorer run. Populating those fields after the failed precondition would falsely claim that the charter had been executed on its registered decision boundary.

## Mandatory arithmetic gate

The inputs below are retained measurements from the existing DX2 pointer; the calculations are `DERIVED` by this arm.

| Quantity | Status | Value |
|---|---|---:|
| DX2 archive size | `MEASURED [contest-CUDA T4, n600]`, retained | 180,368 B |
| DX2 archive SHA-256 | `MEASURED`, retained | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` |
| `d_seg` | `MEASURED [contest-CUDA T4, n600]`, retained | 0.00020139 |
| `d_pose` | `MEASURED [contest-CUDA T4, n600]`, retained | 0.00000637 |
| Seg contribution `100 d_seg` | `DERIVED` | 0.020139 |
| Pose contribution `sqrt(10 d_pose)` | `DERIVED` | 0.007981227975693966 |
| Rate contribution `25 B / 37,545,489` | `DERIVED` | 0.12009964765673980 |
| Recomputed DX2 score | `DERIVED`, agrees with pointer | 0.14821987563243377 |
| Fixed distortion contribution | `DERIVED` | 0.028120227975693966 |
| Continuous byte boundary at `S = 0.12` | `DERIVED` | 137,986.8387944436 B |
| `S(137,986 B)` | `DERIVED` | 0.11999944148120990 — PASS |
| `S(137,987 B)` | `DERIVED` | 0.12000010734016302 — FAIL |
| Largest integer archive with `S < 0.12` | `DERIVED` | **137,986 B** |
| Required reduction from DX2 | `DERIVED` | **42,382 B** |
| Bytes bought by 0.001 S of distortion | `DERIVED` | 1,501.81956 B |

The charter's prose `archive < 137,987 B` is compatible with a maximum of 137,986 B. Its subsequent **42,381 B** demand is not: subtracting 42,381 B from 180,368 B produces the excluded 137,987 B endpoint. The charter itself later names 42,382 B as unassigned prior negative signal, reinforcing that the discrepancy is internal to the registered demand rather than a changed pointer.

## Provenance and custody

All five required seed pins matched before adjudication:

| Required source | Verified SHA-256 |
|---|---|
| `.omx/research/ddm_rb1_rate_bound_decomposition_20260822.md` | `fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09` |
| `.omx/research/ddm_xt1_exact_solve_teacher_student_20260822.md` | `6437bc53d96e527049c3fd6cd60b91af220305881a7bcc68195fece15a728867` |
| `.omx/research/ddm_tk1_20260806/RECEIPT.md` | `5519cce5a986ffd1536233c2f0865a1ce2f95996293f230cb8a0da0f30e09861` |
| `.omx/research/ddm_fp1_class_field_projection_20260731.md` | `b594de4b53d58a1535466f8dc94f14b6fbb87c4d16d8be53b01089996aeef42d` |
| `.omx/research/ddm_nl1_never_fired_levers_20260822.md` | `a11e56b228513c066b803cb6c03e7ce31d2af40d7271b812abaff5e16b5ced3a` |

This arm made a `SCOPE` reduction to arithmetic-preflight disposition because the charter required it. It made no mechanism reduction and assigned no `UNMEASURED` token or cell to `INERT`.

No scorer, Metal, local advisory, or Modal job ran. The sacred JO r9 directory was not read. `upstream/`, NR1, RC1, DB1, their trees, and the staged index were not modified. No payload was materialized, so there is no payload-retention event or bulky receipt.

## Unexecuted deliverable fields

| Field | Honest status | Boundary |
|---|---|---|
| Per-token/per-cell load-bearing census | `NOT ADJUDICATED` | No token/cell classification is emitted; the registered threshold failed its precondition. |
| Measured-subset denominator | `NOT ADJUDICATED` | There is no VF1 census denominator. |
| Evaluator-equivalence MDL floor | `NOT DERIVED` | No inert mass was promoted, so no bytes are credited. |
| Inert-class realization price | `NOT DERIVED` | No zero-rate derivability or cheaper stand-in is asserted. |
| Prior-law prediction | `UNTESTED` | Neither confirmed nor refuted. |
| One owed scorer measurement | `NOT SELECTED` | The charter stopped before a census could be found inconclusive; inventing a scorer request here would bypass the arithmetic repair. |

## RECALL EVIDENCE

The recall scope covered governing state plus content searches across `.omx/research/` memos and receipts, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, `harness_tasklist_bridge_20260803.jsonl`, and `canonical_task_status.jsonl`. Queries included `evaluator-equivalence`, `score quotient`, `token-by-token`, `768 cells`, `4 rungs`, `three-way edit drop keep`, `token drop`, `rung-4`, `jg3`, `fs2`, and `fs3`. The canonical registry was enumerated with `.venv/bin/python tools/list_canonical_equations.py --json`.

Beyond the charter seeds, recall found three boundaries that would matter after the arithmetic repair:

1. JG3's named three-way `{edit, drop, keep}` solve shipped `edit + keep`; its token-drop branch was explicitly not implemented because it required a receiver change. JG5 later measured pair-level KEEP/DROP admission over edited pairs, which is not a token-field omission census.
2. Harness task 869 remains `pending`: the retained HV2 surface is exact-key preparation for four orders, not the claimed 768-cell × 4-rung joint scorer remeasurement. Those cells therefore cannot be counted as measured or inert from that task row.
3. The canonical equations registry marks `argmax_cell_identity_ideal_bytes_v1` as a known-site ideal that excludes site locations, headers, receiver, and realization, and marks `ddm_score_quotient_functional_v1` incomplete. `token_rate_model_direction_dependence_v1` also forbids substituting its directional model for a real re-encode when making a decision.

These findings would have changed a resumed census from a presumed full-field aggregation into a typed measured-subset audit with substantial `UNMEASURED` mass. They do not cure the earlier arithmetic failure, so they were not converted into classifications or byte credit. In the bounded index/DAG/task-ledger scope, no completed current-DX2 768×4 joint receipt or receiver-closed evaluator quotient was found.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — **owner:** MAIN / charter issuer; **consumer store:** `.omx/research/charters/ddm_vf1_evaluator_visible_floor_charter_20260822.md` and this VF1 memo; **fire trigger:** repin or explicitly ratify a strict maximum of 137,986 B and a 42,382 B demand, then refire the retained-receipt-only census from an empty classification table.

## LIVE-HYPOTHESES

- A from-birth evaluator quotient may still remove substantial exact-plane cost because the scorers observe fewer degrees of freedom than the shipped token field. It remains plausible from the objective geometry, but VF1 measured no quotient mass.
- The measured subset may contain meaningful inert mass, but receiver realization could consume much of its ideal saving. The known-site ideal explicitly omits the costs that decide whether omission is legal and byte-positive.

## DEAD-ENDS

- Treating 42,381 B as the strict fixed-distortion sub-0.12 demand is closed: it lands at 137,987 B, whose exact score is above 0.12.
- Treating task 869's exact-key preparation as a completed 768×4 joint measurement is closed by the task ledger's `pending` status.
- Treating JG3 as a completed token-level `{edit, drop, keep}` census is closed because its drop mechanism was not implemented; JG5's later DROP/KEEP states have a different pair-level denominator.

Own-vehicle frontier: **DX2 S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600], UNMOVED.**
