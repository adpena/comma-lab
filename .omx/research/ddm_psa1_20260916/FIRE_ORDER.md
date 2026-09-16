# ddm_psa1 harvest fire orders

## PSA2 — QUEUED-WITH-A-FIRE-ORDER

Owner: MAIN (executing successor: ddm_psa2). Consumer store:
`/Volumes/APDataStore/pact/ddm_psa2/`; canonical queue:
`.omx/state/canonical_task_status.jsonl`, task
`ddm_psa2_pair_selective_rgb_bias_realization_20260916`.

Fire trigger: MAIN lands/verifies the psa1 handoff, issues the scorer-step charter with
`--owns-scorer` after pd6 and any other full-n600 owner release the slot, and revalidates
move52 archive/runtime custody (or explicitly re-prices on a moved pointer).

Concrete consumer inputs: psa1 `RESULT.json`, `SAMPLE.json`, B3's counted sections and
byte-identical research ZIP twins under `/Volumes/APDataStore/pact/ddm_psa1/prices/B3/`.
B3's complete singleton q11 cost is 42 B; 12/36 pairs have enough whole seg debt to pay,
including the 24 B empty-section framing. The appended-SM1X final-archive leg passes 5/36.
No command here launches, authorizes, or assigns a scorer slot.

On that new charter: implement the actual three-channel pair-selective head-bias receiver
in an isolated candidate, realize/optimize the form on the same 36 pairs, price its actual
values, measure seg and resolved pose, and verify all600 collateral in chunks <=120.
Require real byte-closed public receiver support before any exact-eval claim. Sparse
price instances are not effective-repair values. Entire-pair debt is only an optimistic
budget ceiling. No Modal/fire/packet authority is inherited from psa1.

Shared-section A/C/D leads are FOLDED into MAIN's psa2 design review (not independently
queued scorer jobs): retained joint36 prices are A 343/339 B, C 255/312 B, D 383/385 B
(q11/appended final archives). They must not be closed from singleton failure, nor assigned
individual marginal prices from total/36. Their activation trigger is a separate MAIN
scope decision if an effective multi-pair composition requires them.

## Landing — QUEUED-WITH-A-FIRE-ORDER if serializer returns rc17

Owner: MAIN. Consumer: committed repository plus the two task rows in the canonical queue.
Task: `ddm_psa1_serializer_bundle_landing_20260916`.
Fire trigger: the serializer produces a verified bundle in this handoff's `serializer/`
folder. Apply only that intended commit using MAIN's permitted Git context; preserve the
shared index and sibling changes. The arm must not report fallback bundle creation as a
landed main-branch commit. Exact fallback receipts will be recorded after the final attempt.
