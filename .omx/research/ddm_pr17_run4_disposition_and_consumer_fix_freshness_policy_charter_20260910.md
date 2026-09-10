# ddm_pr17 — ratify ffi6's typed consumer-fix row; rule on run4's disposition and on consumer fixes landing under an in-flight intent (charter, MAIN 2026-09-10)

You are the pr-family reviewer (pr8→pr16 lineage; second family; read-only). Facts:
- **run4** (call fc-01M26JVVPJX57J7YTYENQZXW6A, intent v4, authorization v7, amendment[3]) HARVESTED an exact row:
  S 0.13724490417134017 @ 180,406 B [contest-CUDA T4 n600], d_seg 0.00010345, d_pose 4.59e-06 (= move 43's exactly; raw identity
  held on the T4), Modal wall 1,184.7 s; receipt `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run4/MODAL_REMOTE_RESULT.json`
  (read-only). pr16 forbade v3 completion until its completion-custody amendments landed.
- **ffi6** (memo `.omx/research/ddm_ffi6_completion_custody_and_consumer_fix_rows_20260910.md`, sha 5c52aa5f0f6ba990…; landed ea09174b; 231 tests)
  implemented pr16's completion custody and typed `prefire_contract_consumer_fix.v1` rows; its run4 DRY custody validation passes
  across 48 files; it measured that appending the new freeze row makes intent v4 STALE (latest-row freshness), so run4's receipt is
  not completable under the current contract state. MAIN appended the typed row (validated by `_pf_freeze_history`) and is firing
  **run5** (intent v5, authorization v8, same bytes) so a completable row exists under the current state.
- Landing ffi6 changed `src/tac/candidate_seal.py`, which every amendment manifest pins with live-file equality — so after ANY consumer
  landing, every in-flight intent is stale by construction. This is the policy question.

## What you adjudicate (RATIFY / AMEND with literal text / REFUSE, with pr12 sentence → code → test)
1. ffi6's landing + the typed row MAIN appended (read the freeze; verify chain digests, manifest identity, adjudication memo pin).
2. **run4's disposition:** its receipt is a real, authority-grade exact row on the exact bytes. Under pr12, may a receipt produced
   under contract state X be completed under state Y when the ONLY change X→Y is consumer code (definition unchanged) and the
   completion validators of Y pass on X's retained evidence? Rule "as-dispatched completion allowed with a typed provenance link"
   or "re-fire required". If re-fire is required, run5's row (expected byte-identical) is the pointer's source and run4's receipt is
   retained as a corroborating measurement, never scored twice.
3. **Freshness policy:** a rule for consumer fixes landing while an intent is in flight (e.g., stale-intent GRACE bound to the
   authorization nonce: an intent whose authorization was reserved before the fix may complete iff the fix batch's tests pass on
   its retained evidence), or an explicit "no grace" rule with the cost stated (one re-fire per consumer landing).
4. Custody completeness of runs 1–5 (authorizations v1–v8, nonces, claims, ledger rows, reconciliation receipts): name gaps.

## Method
Read pr12 (sha 50d00e3956dc7ae5…), pr13–pr16, ffi6's memo + tests, the freeze (amendments[0..4]), `src/tac/candidate_seal.py`
(diff ea09174b^..ea09174b), run4's retained dir (read-only), `.omx/research/ddm_rlc5_20260910/` (intents v1–v5, authorizations,
RUN3_RECONCILIATION.json). Host suites. Read-only on code and `/Volumes/...`; no Modal, no fires. Memo
`.omx/research/ddm_pr17_run4_disposition_and_consumer_fix_freshness_policy_20260910.md` with a trailing
`<!-- # FORMALIZATION_PENDING:<rationale> -->`; serializer commit LAST; rc 17/19 is NOT a stop. Checkpoint as `ddm_pr17`.

## OPTIMAL FORM
- Reference form: pr12's contract + pr13–pr16 rulings as normative text; the landed code at ea09174b as the object; pr13's clause-table form. No delta.
- Provenance pins (sha256 prefixes): pr16 memo cb678491cc6859f7…; ffi6 memo 5c52aa5f0f6ba990…; ffi6 landing ea09174b; run4 receipt path above (record sha);
  freeze receipt (record sha); pointer move 43 commit 48109233e / archive 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e.

## Prior negatives accounted (operator 2026-08-15)
- pr10 (rule tuned after data): item 3 IS a rule; write it prospectively, name what it forbids, never loosen the latest-row check.
- dwc1: eight pass-path defects were found by real runs; a policy that forces a re-fire per consumer landing is honest but must be
  costed against "bias toward the lower exact score soonest".
- r9m: none of this touches digests; if you find a digest change, REFUSE.

Final message: four verdicts (one line each), any AMEND text, custody gaps, serializer rc, and the frontier line quoted from
`.omx/state/canonical_frontier_pointer.json`.

<!-- # FORMALIZATION_PENDING: review charter; no measured row -->
