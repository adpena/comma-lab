# ddm_cust1 — first-measurement custody reconciliation

**PARTIAL: receipts and corrective claims are complete; the two required linked ledger events are BLOCKED by the canonical helper's event whitelist.** No exact score was measured or pointer moved by ddm_cust1. This memo closes the bounded custody arm with an explicit implementation handoff, not a claim that every charter gap is closed.

| Gap | Append-only receipt | Verification / disposition |
|---|---|---|
| Run3 pr16 custody | `ddm_rlc5_20260910/custody/RUN3_TERMINAL_RECONCILIATION.json` | Exact intent v3/auth v6 references, nonce, job/lane/output, failure/result/provenance, terminal claims and canonical rows 951–952 pinned; no replay or score authority. Linked event BLOCKED. |
| Run1 unsupported dispatched claim | `ddm_rlc5_20260910/custody/CLAIM_CORRECTIONS.json` | Claim tool appended `refused_before_provider_dispatch_no_call_run1_run2_terminal`; original 20:08:11 row preserved. Run1/run2 share the job identity; both remain terminal without a provider dispatch. |
| Run4 stale completion promise | `ddm_rlc5_20260910/custody/CLAIM_CORRECTIONS.json` | Claim tool appended `completed_retained_corroboration_non_promotable_no_v3_completion`; original 21:38:52 row preserved. Run4 is corroboration only. |
| Run5 terminal custody | `ddm_rlc5_20260910/custody/RUN5_TERMINAL_RECONCILIATION.json` and `RUN5_COMPLETED_SEAL_ADDENDUM.json` | Artifact-bound terminal success; intent v5/auth v8/result/nonce/claims/canonical rows 955–956 pinned. MAIN's subsequently observed v3 seal joins the exact three references. Linked event BLOCKED. |
| Authorizations v1–v8 | `ddm_rlc5_20260910/custody/AUTHORIZATIONS_V1_V8_REGISTER.json` | 8/8 authorization files hashed; six distinct nonces, four nonce records. v2–v4 share one permanently consumed nonce. No historical authorization changed. |

All receipt paths in the table resolve under `.omx/research/`. `VERIFICATION.json` supplies the cross-piece checks; `SOURCE_PINS.json` pins 66 source/control files at the initial capture.

## Exact ledger blocker

The real `tac.deploy.modal.call_id_ledger.update_call_id_outcome` was called separately for each already-existing call, with the receipt's exact path/bytes/SHA and no numeric score. Both calls raised `ValueError` before writing:

- Run3 `fc-01M26G7JYMY39TVN1ET3JJV2YD`: `reconciled_terminal_failure` is not in `VALID_EVENT_TYPES`.
- Run5 `fc-01M26NNV3WR2XXXV914S8BTDR4`: `reconciled_terminal_success` is not in `VALID_EVENT_TYPES`.

The accepted set is `dispatched`, `failed`, `harvested`, `manually_terminated`, `pre_spawn_fatal`, `stale`. **Ledger events written by ddm_cust1: 0.** Full requests, exact exceptions, helper taxonomy, and equal before/after ledger hashes are retained in `LEDGER_EVENT_ATTEMPTS.json`. No second dispatched row or substitute event was appended. Extending the helper would edit `src/`, expressly forbidden by this charter. The blocked work is QUEUED-WITH-A-FIRE-ORDER to MAIN in `NEXT_FIRE_ORDERS.json`.

## Schema and scope

pr16 describes `candidate_first_measurement_terminal_reconciliation.v1` in prose; `candidate_seal.py` contains no constructor or validator for that schema. The receipts implement the enumerated fields directly. The success variant retains the same field set and false authority flags. `reconciled_terminal_success` is the symmetric success event name selected for this charter's success variant; pr16 explicitly names only the failure event. MAIN must ratify its canonical taxonomy when implementing support.

Run3's result is a terminal worker failure, return code 1, before scoring. Its result/provenance and stderr identify a runtime-tree root mismatch. The latest historical claim's different failure wording is preserved as claim-closure identity, never promoted into cause authority. `_pf_completion_facts` still refuses run3 with `FIRST_MEASUREMENT_RESULT_REFUSED: authorization not harvested`.

Run5 arrived during this arm. Its retained receipt reports successful n600 T4 evaluation; the arithmetic replay from its retained components is `100*0.00010345 + sqrt(10*0.00000459) + 25*180406/37545489 = 0.13724490417134017` [contest-CUDA T4 n600 retained-result arithmetic, not a new evaluation]. Its false score/promotion flags remain intact. `_pf_completion_facts` passed read-only. This is not full seal validation or promotion. MAIN created the v3 seal while this reconciliation was in progress; the new addendum pins it and verifies its intent/auth/result join. The initial receipt's no-seal-at-capture statement remains historical. `SEAL_SEARCH.json` records the bounded search.

The authorization register uses the requested state vocabulary with explicit meanings: `completed` means a terminal HARVESTED nonce, not a v3 seal. v1/v5 are unconsumed-superseded; v2/v3 are reserved-consumed through the shared v4 nonce, and their own digest/output mismatch to the reservation is explicit; v4/v6 are reserved-consumed; v7/v8 are completed terminal lifecycles. Every row has `nonce_reusable=false`. No absence of a nonce file grants reuse. Outcomes are tied to on-disk controls and the pr17 ruling.

## Verification and provenance

Axis: **host custody / scorer-free**. Both receipt reference sets rehash; all four historical call rows rehash at their recorded line identities; terminal claim rows remain present; every pre-existing claim line is preserved; the original canonical call-ledger prefix is preserved. JSON row hashes use sorted keys, compact separators and ASCII escaping; raw row hashes exclude the newline. Claim line numbers are capture-time only because the canonical claim writer inserts newest rows at the top; exact row text/hash and retained claim snapshots survive later insertions.

- pr17 memo SHA `ab2e0c89016b618ca6d27641edc30850e9d4ab56eeb7f8c4da539f1e9583b787` (required prefix matches).
- pr16 memo SHA `cb678491cc6859f709828017d0ed76f35d7a4143855c5e56e42690a5a8b0cf15`.
- Original run3 reconciliation SHA `553d8c8baa8163df73280697fd4b12b25993fbe7a3541fe34ae1d6dcb16f69ed`; unchanged and retained.
- Run4 terminal receipt SHA `6a7267f4efe7ab54edc6e556030dbfa174d6d027e3cb6b008ad159cce51c0136`.
- Charter move-43 commit `48109233e`; archive `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`.

The pointer file changed concurrently after the initial source pin; its effective score/archive remained move 43 at final capture. `VERIFICATION.json` records this as a concurrent source change, not an edit by this arm. No scorer, decoder, Modal API, dispatch, nonce transition, completion writer, packet, or pointer writer was called. Every `/Volumes/...` access was read-only. No edits to `src/`, `tools/`, upstream, existing receipts, authorizations, or gdc2's directory. The two claims were added only through the required canonical tool. The staged index was not manually touched.

## RECALL EVIDENCE

`RECALL_COMMANDS.json` preserves exact commands, exit codes and output hashes. Content searches covered `.omx/research/` (excluding gdc2's directory), canonical research indexes and `sub015_DAG_*`, design/SPEC-named docs and task-ledger surfaces. Queries: `terminal.reconciliation|reconciled_terminal|nonce.*reus|first.measurement.*custody`; `claim.closure|terminal.custody|nonce|first.measurement`; and `terminal.reconciliation|first.measurement|nonce|claim.closure`. The full canonical equation tool produced 483 rows; searching their serialized contents for `terminal_reconciliation`, `first_measurement`, and `nonce` found no matching equation in that scope.

Beyond the charter seeds, ffi6's memo explains that successful provenance can be JSON embedded inside `MODAL_REMOTE_RESULT.json`, with no standalone provenance file. This changed the run5 receipt to pin the containing file plus exact field path and decoded-content SHA, rather than invent a standalone path. The task store contains an older rlc5 harvest output path; the live authorization and actual retained run5 result determine this handoff instead. The source helper's whitelist reveals the literal-event blocker, absent from the charter's expected closure. The DAG hits concern other first-measurement surfaces and do not alter this custody contract. Memory's serializer-denial precedent was used only to prepare an honest fallback, not as evidence that this landing failed.

## Landing

The final serializer attempt and verification are recorded separately under this custody directory so existing evidence stays immutable. Expected hashes are post-edit hashes for each intended file; no attribution trailer; `[no-triality] [p0-ledger-ok]`. Only this arm's two claim rows belong in its patch, preserving unrelated work. The checkpoint is marked complete as charter-required arm termination; that does not reclassify the blocked ledger writes as complete. No claim of committed custody is made until the serializer receipt establishes it.

Frontier quoted from the canonical pointer at capture: **“S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600]”**. This arm did not lower it.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store `ddm_rlc5_20260910/custody/LEDGER_EVENT_ATTEMPTS.json` and `.omx/state/modal_call_id_ledger.jsonl`; trigger: canonical support for the reconciliation event types is authorized and landed, and the pinned receipts/rows rehash.** Append the two exact linked events once, including the run5 seal addendum; preserve false authority flags and permanent nonce consumption. No refire.

## LIVE-HYPOTHESES

- A narrow canonical event-taxonomy extension can close both residual ledger links without touching historical rows; both real helper calls failed at the whitelist before mutation, and their full valid custody payloads are retained. This implementation remains untested and requires MAIN's authorized source landing.

## DEAD-ENDS

- Literal reconciliation events through the present helper are closed at INSTANCE scope: both fail its explicit whitelist before writing.
- Promoting run4 or reviving its pending completion wording is closed by pr17's stale-intent NO-GRACE ruling.
- Treating run3 as successful or resetting its nonce is closed by the retained worker failure and permanent RESERVED record.
- Treating v2/v3 as independently reusable is closed by v4's on-disk reservation of their shared nonce.
- Treating the pointer file's changed hash as a score move is closed: its effective score/archive remained move 43 at the observed capture.

<!-- # FORMALIZATION_PENDING: custody-only receipt reconciliation; two canonical event types remain blocked by the existing helper whitelist -->
