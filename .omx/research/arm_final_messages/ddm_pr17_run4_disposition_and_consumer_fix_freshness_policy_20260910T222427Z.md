Completed the review. The ruling is strict latest-row freshness: run4 is valid retained evidence, but only run5 can produce the current-state seal and pointer move.

The full ruling is in [ddm_pr17 review memo](/Users/adpena/Projects/pact/.omx/research/ddm_pr17_run4_disposition_and_consumer_fix_freshness_policy_20260910.md).

1. **RATIFY:** ffi6 commit `ea09174b` and typed consumer row `e77ed88e…` validate; all 17 manifest entries match committed and live files.
2. **REFUSE:** run4 cannot complete under the later freeze row. Keep its exact `S=0.13724490417134017` receipt as corroboration only; never score it twice.
3. **AMEND-PROSPECTIVE:** adopt an explicit NO-GRACE rule. Each consumer-fix landing batch crossing an in-flight authorization costs one fresh lifecycle/fire.
4. **AMEND:** custody remains incomplete because run3 lacks pr16’s required reconciliation schema/event, historical run1/run4 claim wording needs append-only corrections, and run5 was still in flight at the `2026-09-10T22:22:09Z` cutoff.

Literal AMEND:

> **Consumer-fix freshness — NO GRACE.** A pre-fire intent and every authorization binding it are usable only while its `latest_row_sha256` equals the current last freeze-row digest. Appending a consumer-fix row invalidates earlier intents at dispatch and completion, including reserved or spawned authorizations. Earlier results may be harvested and retained, but cannot create v3, receive promotion, or move the pointer. After terminal reconciliation and single-flight clearance, MAIN must issue a new intent, authorization, nonce, job/output, and fire. Identical candidate bytes and provenance links create no exception.

Custody gaps:

- `RUN3_RECONCILIATION.json` uses the wrong schema, lacks exact intent/authorization references, explicit `nonce_reusable=false`, claim-closure custody, ledger-row hashes/identities, and the linked `reconciled_terminal_failure` event.
- Run1’s unsupported `dispatched` wording and run4’s stale `pending_v3_completion`/P0 route require new corrective rows, not edits.
- Run5 was `SPAWNED` as call `fc-01M26NNV3WR2XXXV914S8BTDR4`; its fresh v5/v8 lineage, nonce, claim, and dispatch ledger row exist, but no terminal receipt existed at cutoff.

Validation: **231 tests passed** and Ruff passed. Run4’s 48-file custody and `_pf_completion_facts` passed; intent v4 refused solely on latest-row freshness while v5 passed.

Serializer returned **rc 17** because Git object writes were denied. The shared index stayed empty. The verified fallback is [intended-commit.bundle](/Volumes/VertigoDataTier/pact/ddm_pr17/serializer_fallbacks/20260910T222304.696741Z-54954/intended-commit.bundle), fallback commit `dc284e499229c6ca6c4cf6384dff676b7be2f1f2`. Memo SHA-256: `ab2e0c89016b618ca6d27641edc30850e9d4ab56eeb7f8c4da539f1e9583b787`.

Frontier unchanged: **“S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600]”**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store: the verified pr17 bundle plus `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run5/`; trigger: land the bundle and receive run5’s terminal receipt.** Revalidate v5 from disk, then complete v3/pointer move 44 only if every current check passes; otherwise retain a typed refusal.

## LIVE-HYPOTHESES

- Run5 should reproduce run4 closely because it carries the identical 180,406-byte archive under pinned deterministic evaluator inputs.
- Batching fixes before authorization should make the one-refire penalty uncommon without weakening freshness.
- Run3’s missing typed reconciliation can be appended from its preserved artifacts without modifying history.

## DEAD-ENDS

- Grandfathering run4 through a provenance link.
- Reusing run4’s nonce, authorization, job, or receipt.
- Discarding or calling run4 fake; it is real corroboration, just non-promotable.
- Treating identical generated raw as identical archive custody.
- Treating the existing run3 reconciliation as pr16-complete.
- Calling commit `3dfbd36e` a valid amendment append.
- Relaxing latest-row freshness to save a paid fire.

