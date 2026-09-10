# ddm_cust1 — close pr17's custody gaps for the rlc5 first-measurement runs 1–5 (append-only; charter, MAIN 2026-09-10)

pr17 (`.omx/research/ddm_pr17_run4_disposition_and_consumer_fix_freshness_policy_20260910.md`, sha ab2e0c89016b618c…, landed) ruled NO-GRACE
freshness and named four custody gaps. Close them APPEND-ONLY (never edit an existing record):
1. **run3 reconciliation in pr16's schema**: replace MAIN's `.omx/research/ddm_rlc5_20260910/RUN3_RECONCILIATION.json` (wrong schema;
   keep it, do not delete) with a new pr16-conformant receipt: exact intent (v3) and authorization (v6) references
   `{path,bytes,sha256}`, explicit `nonce_reusable=false`, claim-closure custody (the terminal claim row's hash/identity),
   canonical ledger-row hashes/identities, and the linked `reconciled_terminal_failure` ledger event written through
   `tac.deploy.modal.call_id_ledger.update_call_id_outcome` (read pr16's memo §5 and `candidate_seal.py` for the exact field set;
   if the schema is only described in prose, implement it as the memo states and say so).
2. **run1 / run4 corrective claim rows** (append-only via `tools/claim_lane_dispatch.py claim --force --status …`): run1's
   unsupported `dispatched` wording (it refused before dispatch) and run4's stale `pending_v3_completion` wording (run4 is
   retained corroboration, non-promotable per pr17) — new rows with exact statuses; existing rows untouched.
3. **run5 terminal reconciliation**: once `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run5/MODAL_REMOTE_RESULT.json`
   exists (poll the file, artifact-bound), write its pr16-schema reconciliation receipt (success variant) and the linked ledger
   event; if MAIN has already completed v3 by then, reference the seal by `{path,bytes,sha256}`.
4. **Authorizations v1–v8 register**: one append-only JSON listing each authorization `{path,bytes,sha256, state (unconsumed-superseded /
   reserved-consumed / completed), nonce reusable=false, job id, output dir, outcome}` — read from disk, never inferred.
Deliverable: the receipts under `.omx/research/ddm_rlc5_20260910/custody/`, memo
`.omx/research/ddm_cust1_first_measurement_custody_reconciliation_20260910.md` with a gap → receipt → verification table;
serializer commit LAST (`REVIEW_GATE_OVERRIDE=1` ok for non-.py); rc 17 is NOT a stop. Checkpoint as `ddm_cust1` and mark it
COMPLETE at the end (an unclosed checkpoint trips the #340 guard on MAIN's landing).

## Boundaries
Read-only on every `/Volumes/...` path and on all existing records; no Modal, no fires, no completion (MAIN's); no edits to
`src/` or `tools/`; do not touch gdc2's directory.

## OPTIMAL FORM
- Reference form: pr16 §5 reconciliation requirements + pr17 §4 custody list, implemented verbatim; the canonical ledger helper
  for events; the claim tool for rows. No delta.
- Provenance pins (sha256 prefixes): pr17 memo ab2e0c89016b618c…; pr16 memo (record sha); `RUN3_RECONCILIATION.json` (record sha);
  run4 receipt `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run4/MODAL_REMOTE_RESULT.json` (record sha); pointer
  move 43 commit 48109233e / archive 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e.

## Prior negatives accounted (operator 2026-08-15)
- Catalog #110/#113: historical provenance is append-only — corrective rows, never edits.
- vr7 (MOVE labels ≠ custody): hash every referenced artifact from disk.
- pr17: run4 is corroboration only; nothing you write may present it as promotable.

Final message: gap → receipt table, the ledger events written, serializer rc, and the frontier line quoted from
`.omx/state/canonical_frontier_pointer.json`.

<!-- # FORMALIZATION_PENDING: custody charter; no measured row -->
