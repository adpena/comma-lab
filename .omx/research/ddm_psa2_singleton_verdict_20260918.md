# ddm_psa2 — singleton verdict: the per-pair int4 RGB head bias family is CLOSED at formulation scope (MAIN, 2026-09-18)

**Result (MEASURED, 60 of 60 pairs, exhaustive 4,096-vector grid per pair, pose re-solved, both codecs priced by real encode):**
k_net_positive = **0 / 60**. The pre-registered falsifier (k < 6) FIRED.

| quantity | value |
|---|---|
| pairs with any seg credit after the pose re-solve | 7 / 60 (five 1-cell, two 2-cell) |
| pairs that collapse to the zero-bias control | 53 / 60 |
| section overhead per pair (cheapest codec q11) | 42 B (45–46 B on two pairs) |
| best singleton net | pair 53: +2.636e-5 S (2 cells = −1.7e-6; 42 B = +2.80e-5) |
| worst singleton net | pair 406: +3.116e-5 S |
| total seg credit if EVERY nonzero pair were free | −7.63e-6 S (0.11 bar) |

**Mechanism.** One cell is worth 8.5e-7 S and one byte 6.7e-7 S, so a 1-cell pair pays for itself only
under 1.27 B of carrier. The three-dof bias codes plus section framing cost 42 B on every pair, 33× the
largest credit observed. The pose re-solve absorbs the rest (53 pairs end at zero bias). psa1's projection
(12/36 could pay) was drawn before the re-solve and before the section price; both were the missing terms.
verdict_scope: formulation — per-pair three-dof int4 RGB head bias on move 52's renderer, priced as singletons.
Not claimed: a set admission (the charter's owed item) — with every singleton +2.6e-5 or worse and credits
sub-additive on seg (composition law), a set cannot net negative; the owed real-price ladder is FOLDED.

Artifacts: `/Volumes/APDataStore/pact/ddm_psa2/SINGLETON_SUMMARY.json` (sha deeb4135d13bc825…), `/Volumes/APDataStore/pact/ddm_psa2/SINGLETONS.json` (sha f1415ca389efba88…), per-pair
`pairs/pair_NNN/{SEARCH,RESOLVED}.json`, `prices/pair_NNN/PRICE.json`; follow-through v3 receipt
`.omx/tmp/codex_runs/ddm_psa2_followthrough_v3.done` (rc 0, 1,812 s); supersession record
`FOLLOWTHROUGH_V3_SUPERSESSION.json`. Axis: macOS-CPU advisory, frozen CPU scorers; no score claim.
# FORMALIZATION_PENDING: one negative on a per-pair actuator family; the exchange arithmetic is the existing rate law (25·B/37,545,489) and the seg cell price (100/(600·384·512)); no new equation.
