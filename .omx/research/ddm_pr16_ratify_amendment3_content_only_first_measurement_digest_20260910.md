# ddm_pr16 — ratification of amendment[3] and the run3→run4 custody chain

Date: 2026-09-10

Role: second-family, read-only pr reviewer

Tokens: `[no-triality] [p0-ledger-ok]`

## Outcome

Amendment `[3]` is a valid, fully pinned implementation re-pin. Its content-only first-measurement comparison is not a relaxation of runtime byte identity: it compares the same executed file rows and dependency inputs while intentionally omitting only deployment-location fields. The dispatcher, worker, and evaluator all check that content digest, and a one-byte runtime change refuses.

The chain is not ready for a completed `candidate_seal.v3`. The completion consumer does not require the content flag or join the request/result/provenance content digest, and the unchanged `t4_direct` builder records the normal projected path-tree digest rather than the actual retained-worker tree. Run4 is still `SPAWNED` with no result receipt at the audit cutoff. These are custody gaps, not evidence that amendment `[3]` failed and not authority to refire.

The canonical frontier remained `effective_frontier.score = 0.1372848557085275`, archive `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`, axis `contest_cuda`, lane `ddm_sj1_t4_token_predistortion_pass6_20260910` at the final pre-serialization read. This review moved no pointer and achieved no lower exact score.

## Five verdicts

1. **AMEND (content comparison RATIFIED; completion custody not ratified).** Content-only identity is at least as strong on every executed byte, but a completed v3 seal and its `t4_direct` leg must explicitly validate and record both the content digest and the actual retained-root path-tree digest.
2. **RATIFY.** The prescribed fire path registers once in the repository-root ledger with that ledger's matching lock and does not write a snapshot ledger; run4 is the real positive control.
3. **RATIFY.** The pr15 authorize-import AMEND is implemented exactly by adding `REPO` after `REPO/src`, and the foreign-working-directory regression passes.
4. **AMEND-PROSPECTIVE.** Add a distinct append-only consumer-fix row class, with a chained definition parent and batched typed fixes, while preserving an exact latest-implementation-row check.
5. **AMEND / CUSTODY INCOMPLETE.** Run3 is non-replayable and its bytes are retained, but its canonical lifecycle is not fully reconciled; run4 has a fresh immutable intent/auth/nonce/job pairing and a clean dispatch row, but harvest and terminal custody are pending.

## Clause → code → test

| Question | Controlling sentence | Executed code path | Test or retained evidence | Verdict |
|---|---|---|---|---|
| First-measurement runtime identity | pr12 requires recomputation of the runtime identity with the named function and exact equality; completion must bind the remote runtime, intent, authorization, and call. A one-measurement exception cannot weaken non-timing identity. | `modal_uploaded_submission_dir_runtime_manifest` hashes every runtime row as `(relative_path, bytes, sha256)`, external dependency-root contents, the repo-local `tac` dependency manifest, and upstream `evaluate.py`. Only absolute/root-location fields are absent from `runtime_content_tree_sha256`. `fire_modal_auth_eval.py` derives the independent local pin; `modal_auth_eval.py` validates it locally and forwards it; `contest_auth_eval.py::_validate_expected_runtime_tree` checks it against the retained worker manifest. | `test_first_measurement_argv_and_manifest_name_both_real_digests`, `test_content_pin_accepts_relocation_but_refuses_one_changed_byte`, `test_local_content_pin_validation_and_worker_argv`, and `test_content_pin_survives_fail_closed_wrapper` pass. Independent run3 reconstruction gives equal content SHA and equal 48 file rows across different roots. | Content selection **RATIFY**; completion consumer **AMEND**. |
| Canonical call registration | pr12 requires a durable call-id row tied to the reserved authorization before completion; snapshot-local state cannot satisfy repository custody. | `modal_auth_eval.py` passes `Path.cwd()/.omx/state/modal_call_id_ledger.jsonl` and the exactly matching `.jsonl.lock` to the same registration helper for both normal and first-measurement calls. `fire_modal_auth_eval.py` launches the snapshot entrypoint with `cwd=REPO`. Registration remains after successful detached spawn, matching the normal path's ordering. | `test_local_registration_uses_repo_ledger_when_imported_from_snapshot` passes for both modes. Run4 has exactly one canonical dispatched row and no ledger or ledger lock under its source snapshot. | **RATIFY**. |
| Authorize import closure | pr15 requires the committed authorize consumer to import all transitive runtime-digest dependencies outside the repository working directory. | `tools/authorize_candidate_first_measurement.py` uses `sys.path.insert(0, str(REPO / "src"))` followed by `sys.path.insert(1, str(REPO))`; this makes the deferred `experiments.contest_auth_eval` dependency reachable without relying on pytest cwd. | `test_authorize_tool_imports_experiments_from_a_foreign_cwd` passes in a bare subprocess with `PYTHONPATH` removed. | **RATIFY**. |
| Freeze amendments | pr12/pr14 require immutable append-only contract custody, live/committed source identity, and refusal of an intent pinned to anything other than the latest frozen implementation. | Freeze rows `[0..3]` preserve the pr14 definition and identifier byte-for-byte while repinning consumer snapshots. Row `[3]` points to commit `0919b62e4e4afba47124e77dd9c3e958b30e477a`; all 16 manifest rows match both the live files and `git show 0919b62e:<path>`, and that commit is an ancestor of freeze append `974dda54c`. The schema nevertheless represents fixes `[1..3]` as repeated definition rows whose semantic distinction exists only in free-text `note`. | Existing latest-row/live-committed-source negatives pass. Freeze SHA `9557af75c031b06eaf6a190fed03ec32afe11a994478149f6ce215e8cc605415`; landed amendment[3] manifest SHA `97b739cd45da328fab204e2aa249b9126718e4f2bc9bfff92ee8a87c400c07df`. | Row `[3]` **RATIFY**; future row schema **AMEND**. |
| Run3→run4 nonce/job custody | pr12 requires atomic nonce reservation before spawn, forward-only lifecycle transitions, one call per authorization, and no reset after failure or ambiguity. | The consumption files are immutable per nonce. Run3's old nonce remains `RESERVED`; run4's distinct nonce is `SPAWNED` and binds call, job, lane, output, intent, and authorization. `_pf_registered_call` can accept run4's complete canonical row, but cannot reconstruct run3 from its incomplete manual row. | Direct inspection of the canonical ledger, consumption files, run3 retained result/provenance, and run4 fire manifest. No Modal call or mutation was performed. | **AMEND / incomplete terminal custody**. |

## Why content-only is not a loosening

The digest pair has deliberately different jobs:

| Digest | Includes | Omits | Result on run3 |
|---|---|---|---|
| `runtime_content_tree_sha256` | Every shippable runtime file's relative path, size, and SHA-256; external dependency roots and their file rows; repo-local runtime dependency rows; upstream evaluator bytes | Physical extraction/upload root names and other location-only fields | Local = worker = `e1e6d1252b56b66b7f7559fe0df751ffbf38559d99abf21f06f00134f685bcaf` |
| `runtime_tree_sha256` | The same content inputs plus deployment/root-location identity | Nothing in its path-custody scope | Normal local projection `e3d2371917920ce3dad002761203af48cfa72d98b07f82f077ef6dc0115b1891`; actual retained worker `baeb53afc8d1bb0c43bb2ce91fe1e09e845809b76fc3047bc8ed3397254155af` |
| `runtime_files_sha256` | The 48 runtime file rows | Dependency and root metadata | Local = worker = `fb1f6295a3527fdc07b3682d7e8ed42354e14e54dbbb4317deaa9f4f27ba21d4` |

Projecting the independently computed local manifest onto run3's recorded retained root reproduces `baeb53af...` exactly. The runtime row maps are identical, including all 48 `(relative_path, bytes, sha256)` tuples. Changing one real source byte changes the content digest and refuses. The normal path still checks the path-tree digest when its roots coincide. First-measurement mode therefore changes the equality relation from “same bytes at the same physical root spelling” to “same executed bytes under an independently retained physical root,” while preserving the path digest as custody evidence.

This is host/provenance proof, not scorer authority. It is runnable without another dispatch. Cold n600 success, actual decode time, exact Seg/Pose/rate, and run4's terminal retained digests require the already-running remote job and were not runnable in this read-only review.

## Literal AMEND 1 — completion runtime custody

Apply prospectively to `candidate_seal.v3` first-measurement completion and to the `t4_direct` leg produced for that completion. Do not alter legacy non-v3 `t4_direct` validation.

> A first-measurement completion MUST recompute the current candidate's `runtime_content_tree_sha256` with `tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest`. It MUST require one exact `--expected-runtime-content-tree-sha256` argv value and exact equality across the independently recomputed value, `FIRST_MEASUREMENT_CONTEXT.json`, the retained local request, `MODAL_REMOTE_RESULT.json`, and `artifacts.provenance.json.inflate_runtime_manifest.runtime_content_tree_sha256`. Missing or unequal values MUST REFUSE with `FIRST_MEASUREMENT_RESULT_REFUSED`.
>
> Completion MUST read the actual retained runtime root and `runtime_tree_sha256` from the SHA-pinned provenance artifact. It MUST project the independently recomputed local manifest onto that exact retained root and require the resulting path-tree digest to equal the worker's recorded digest. It MUST NOT require the actual retained-root tree digest to equal the normal upload-root prediction.
>
> The completed `candidate_seal.v3` and its embedded first-measurement `t4_direct` leg MUST contain byte-identical `first_measurement_runtime_custody` objects with: schema; content and path-tree digest-definition identifiers; expected content SHA; worker content SHA; normal projected tree SHA; actual retained root; actual retained tree SHA; and SHA-pinned source field paths for the context, request, result, and provenance. Validation MUST freshly rederive every local value and re-read every pinned source. The candidate seal MUST refuse if the two objects differ.
>
> Required negatives: missing or duplicate content argv flag; wrong context/request/result/provenance content SHA; missing provenance; wrong retained root; wrong actual tree; altered runtime byte; unequal seal/leg custody objects. Required positives: relocation with identical content and correctly projected actual tree; unchanged legacy `t4_direct` fixture remains valid.

The existing `candidate_seal.py::_pf_completion_facts` checks other exact argv values but not the content flag, and it builds the unchanged direct leg without joining the worker provenance. `build_t4_direct_leg` records `measure_t4_runtime_digest(runtime_dir)` and reads the receipt's outer `expected_runtime_tree_sha256`; neither is the actual retained-worker tree. The existing completed fixture contains no content digest and still passes. That is why the completion half is AMEND rather than RATIFY.

Until this is implemented and appended under the consumer-fix schema below, run4 may be harvested and retained as a real positive/negative control, but it MUST NOT produce a completed v3 seal or move the pointer. Because the latest-row rule is preserved, any post-amendment scored fire needs a newly emitted intent and fresh authorization; run4's nonce is never reused.

## Literal AMEND 2 — typed consumer-fix freeze rows

Do not edit freeze rows `[0..3]`. Add this class prospectively and allow one row to batch multiple causally related fixes:

```json
{
  "schema": "prefire_contract_consumer_fix.v1",
  "fix_batch_id": "<unique stable id>",
  "previous_row_sha256": "<canonical sha256 of immediately preceding row>",
  "definition_parent_sha256": "<canonical sha256 of latest preceding prefire_contract_amendment.v1 definition row>",
  "definition_change": false,
  "consumer_fixes": [
    {
      "fix_id": "<unique stable id>",
      "scope": "<exact consumer/path/behavior>",
      "reason": "<retained causal finding>",
      "causal_commit": "<full commit sha>",
      "tests": ["<exact test name>"]
    }
  ],
  "implementation_commit": "<full reachable commit sha>",
  "implementation_manifest": {"path": "<absolute path>", "bytes": 1, "sha256": "<64 hex>"},
  "adjudication_memo": {"path": "<absolute path>", "bytes": 1, "sha256": "<64 hex>"},
  "score_claim": false
}
```

Normative validation:

1. A consumer-fix row MUST contain no contract-definition, candidate/reference receiver, threshold, or evidence-value override. `definition_change` MUST be exactly `false`.
2. `definition_parent_sha256` MUST equal the canonical digest of the latest preceding definition row. `previous_row_sha256` MUST chain the complete append-only history.
3. `consumer_fixes` MUST be nonempty, sorted by unique `fix_id`, and each item MUST name a narrow consumer scope, retained cause, full causal commit, and at least one exact test. A batch is one implementation snapshot, not permission to hide unrelated rule changes.
4. The full implementation manifest MUST match the live files and committed blobs, and both the causal and implementation commits MUST have the required ancestry.
5. An intent MUST pin both the current definition-parent digest and the exact canonical digest of the latest freeze row. If the latest row is a consumer-fix row, an intent pinned to the preceding definition or consumer row MUST refuse. Thus batching changes representation, not freshness strength.
6. Required negatives: stale latest row, broken previous-row chain, missing/duplicate fix id, wrong definition parent, any definition-bearing key in a consumer-fix row, manifest drift, unreachable commit, or empty tests. Required positive: multiple typed fixes in one row with an unchanged definition parent.

Rows `[1..3]` remain historical, valid snapshots under their then-current schema. Amendment `[3]` changes consumers, not the pr14 scoped-risk definition. The new class prevents future implementation fixes from masquerading as repeated definition amendments and lets one second-family ratification close a typed batch without weakening exact latest-row refusal.

## Run3 and run4 custody

Audit cutoff: `2026-09-10T21:31:13Z`.

### Run3

- Call `fc-01M26G7JYMY39TVN1ET3JJV2YD`; job `ddm_rlc5_first_measurement_t4_run3_20260910`; authorization v6 nonce file `9ced38b39888d5d3a767f36d67e241f44aef8c967d2f7294c7516727cd5337c1.json`.
- Retained result: 80,671 B, SHA `525a9bb57f5462410f321fabb57410c8e5e59bc2c57b3079176e15feb00f72f5`; retained provenance: 42,595 B, SHA `4cd0463214a90b8919489eabe77062df8720c5d8405faf2e3d5ef10ddbd74405`.
- The failure is terminal and not a score: `passed=false`, return code 1, path-root mismatch before evaluation. The claim is closed.
- Non-replay custody is safe because the nonce remains durably `RESERVED`; it cannot return to unused. Terminal lifecycle custody is incomplete: the consumption file was never advanced to `SPAWNED`/`HARVESTED`, canonical row 951 is a manual `dispatched` row with axis `cuda` and without intent/authorization/file pins, and row 952 is a separate `failed` row.

Required reconciliation, append-only:

> Emit one `candidate_first_measurement_terminal_reconciliation.v1` record. Do not edit the authorization, nonce record, result, provenance, or existing ledger rows, and do not append a second `dispatched` event. Bind the exact call, v6 authorization path/bytes/SHA, intent path/bytes/SHA, nonce, job, lane, output, result path/bytes/SHA, provenance path/bytes/SHA, terminal provider/worker failure, claim closure, and the two existing ledger row hashes/line identities. Append one typed `reconciled_terminal_failure` ledger event referring to that immutable record. The record MUST state `score_claim=false`, `promotion_eligible=false`, and `nonce_reusable=false`; it supplies custody only and MUST NOT make run3 pass `_pf_completion_facts`.

### Run4

- Call `fc-01M26JVVPJX57J7YTYENQZXW6A`; intent v4 canonical digest `ad149549d58c06eca96ee999cc8b4eed7ca0079a888a494f292bc923c1ce02ff`; authorization v7 SHA `f7e229a17d05cd216faa943e6244b54a4f2cbcc39b6e43b29e9207ac81ca6b7e`; fresh nonce `2ca56bcc195e2749b44a4c2930953bce3cc8cd144877a73ad0780b6029f44c68`; job `ddm_rlc5_first_measurement_t4_run4_20260910`.
- The intent and authorization files are byte-identical to their containing commits (`6481aa...` and `979d51...` respectively). The consumption file is `SPAWNED` and binds the call/job/lane/output.
- Canonical ledger row 953 is the only row for this call and contains `contest_cuda`, T4, archive, intent, authorization, and file-identity fields. The source snapshot has no `.omx/state/modal_call_id_ledger.jsonl` and no matching lock. This is the real ledger positive control.
- `FIRE_MANIFEST.json` is 6,104 B, SHA `7239ff39dcefa6b682a4a5ecc5c32393e64eae770203e7b4b3019a489f1a3f5d`, and carries both expected tree `e3d23719...` and expected content `e1e6d125...` plus the exact call and worker argv.
- At cutoff there is no `MODAL_REMOTE_RESULT.json`; the consumption state remains `SPAWNED`. Score, time, n600, actual worker content/tree digests, result hash, harvest transition, claim closure, completion, and pointer disposition are therefore pending. No inference is made from elapsed wall time.

## Provenance pins

| Object | SHA-256 / commit |
|---|---|
| pr12 memo | `50d00e3956dc7ae5d3b15379d2ae6f8704119817b0b58f50aa30413b97eacadc` |
| pr13 memo | `3156992449ea4f71966dfe96adea1471aa3b735dae3bac4a9f49c4a16ce4eba3` |
| pr14 memo | `6e1732baea1912731fef6a53abca0cb94faf2e9440accc304dd9b34e593bade9` |
| pr15 memo | `39efb49c2e0f34b9c36f7b8a858afe9e68ff7878efa4f637ef650167eedd7560` |
| ffi5 memo | `2482e7eeee94b832eb7d1e3d00f1cf0d328eeea16cc3396f5d1aabaa03b7d28c` |
| freeze after rows `[0..3]` | `9557af75c031b06eaf6a190fed03ec32afe11a994478149f6ce215e8cc605415` |
| amendment[3] implementation | `0919b62e4e4afba47124e77dd9c3e958b30e477a` |
| amendment[3] implementation manifest | `97b739cd45da328fab204e2aa249b9126718e4f2bc9bfff92ee8a87c400c07df` |
| freeze append | `974dda54c` |

## Validation

- `.venv/bin/python -m pytest src/tac/tests/test_candidate_prefire_intent.py src/tac/tests/test_candidate_seal.py src/tac/tests/test_decode_wall_clock_t4_direct.py -q` — **146 passed**.
- `.venv/bin/python -m pytest src/tac/tests/test_candidate_prefire_intent.py src/tac/tests/test_candidate_seal.py src/tac/tests/test_decode_wall_clock_t4_direct.py src/tac/tests/test_modal_source_snapshot.py -q -k 'not test_mount_ignore_still_excludes_git_tree_and_special_files'` — **181 passed, 1 deselected**.
- The same adjacent suite unfiltered — **181 passed, 1 failed**. The sole failure is `test_mount_ignore_still_excludes_git_tree_and_special_files`: its local Unix-socket setup receives sandbox `PermissionError: [Errno 1] Operation not permitted`; no product assertion ran. This is an environment limitation, not a positive pass claim.
- `.venv/bin/python -m ruff check` over the amendment[3] code/test surface — **all checks passed**.
- Read-only Git verification: all 16 manifest rows match live and committed blobs at `0919b62e`; `0919b62e` is reachable from the committed freeze; rows `[0..3]` are unchanged.

These host checks cover the byte-comparison algorithm, changed-byte and relocation branches, argument flow, ledger selection/lock pairing, import closure, latest-freeze refusal, and existing completion behavior. They do not cover run4 remote termination, exact scoring, timing, or the proposed completion AMEND; those remain explicit gates.

## RECALL EVIDENCE

I searched the research corpus, canonical equation registry, `sub015_DAG_*`, design/spec docs, task/ledger state, and Git history for `runtime tree`, `runtime_content_tree_sha256`, `measure_t4_runtime_digest`, `first measurement`, `ledger`, `nonce`, and `consumer amendment`, in addition to the named pr12–pr15/ffi5 seeds.

Two non-seed precedents changed the adjudication:

1. `.omx/research/ddm_r9m_first_contest_cpu_row_20260804/runtime_tree_hash_verification_20260804.md` records the same genus: byte-identical file rows can receive different path-projected tree hashes. That required verifying both content identity and actual-root path custody rather than treating either digest as a replacement for the other.
2. `.omx/research/codex_findings_modal_runtime_content_fail_closed_20260523T205211Z_codex.md` separates runtime content identity from package/location identity. That prompted the downstream completion audit and exposed the absent content/provenance join in `_pf_completion_facts` and `build_t4_direct_leg`.

No competing canonical equation, design row, DAG row, or task row was found in the searched scope. The live task and ledger state confirmed run4 ownership and pending harvest; it did not authorize a second dispatch or a score claim. External memory lookup had no directly relevant pr16 entry, and no memory-derived fact is used here.

## Boundaries

No code, test, runtime, archive, payload, volume artifact, authorization, nonce, result, claim, call ledger, lane, pointer, upstream tree, or shared staged index was changed. No Modal API/CLI or fire was invoked. The only durable review artifact is this memo; the charter-authorized `ddm_pr16` progress checkpoint was written before serialization. Run3 and run4 payloads remain retained. Unrelated dirty work is preserved.

## NEXT_IF_RESUMED

- **AMEND / owner MAIN implementation-and-freeze landing / consumer store** `src/tac/candidate_seal.py`, `src/tac/decode_wall_clock.py`, completion regressions, one typed `prefire_contract_consumer_fix.v1` batch, and a fresh implementation manifest; **fire trigger** run4 reaches a retained terminal state and the exact completion-custody and consumer-fix text above is implemented, reviewed, committed, and appended without editing freeze rows `[0..3]`.
- **RECONCILE / owner MAIN custody ledger / consumer store** one immutable run3 `candidate_first_measurement_terminal_reconciliation.v1` receipt plus one linked canonical `reconciled_terminal_failure` event; **fire trigger** all named run3 artifacts and existing ledger rows rehash exactly and provider/claim terminal state is confirmed; never reopen or reuse the nonce.
- **HARVEST-ONLY / owner MAIN / consumer store** `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run4/` and the canonical nonce/call/claim ledgers; **fire trigger** the existing call emits its retained terminal receipt. Preserve it as the amendment positive/negative control, do not refire, and do not complete v3 or move the pointer under the unamended consumer.
- **MEASURE / owner MAIN fresh first-measurement lifecycle / consumer store** a newly emitted intent, newly committed authorization, fresh nonce/job/output, completed v3 seal, exact evaluation ledger, and pointer packet; **fire trigger** both AMENDs are landed and frozen, the prior call is terminal, all usual lane/cost/single-flight/pointer gates pass, and the new intent pins the latest consumer-fix row.

## LIVE-HYPOTHESES

- Run4 will accept relocated runtime bytes because its manifest carries the independently derived content digest and the landed evaluator checks it; only the retained terminal artifacts can confirm this.
- Adding an explicit completion custody object will turn the presently transitive retained-tree evidence into a fresh, locally rederived v3 invariant without changing the legacy timing contract.
- A two-level freeze chain—definition parent plus exact latest implementation row—can batch consumer repairs while remaining strictly fail-closed against stale intents.

## DEAD-ENDS

- Requiring equality between normal-upload and retained-volume path-tree hashes is closed by exact run3 root projection; it tests location spelling, not byte identity.
- Dropping the path-tree digest entirely is closed; it remains necessary custody evidence for where the worker actually executed.
- Treating amendment `[3]` as a rule change is closed: the pr14 definition is unchanged and all 16 consumer blobs are pinned.
- Treating run3 as replayable, successful, or scored is closed; its reservation is permanent and its retained result is a failure.
- Treating run4 dispatch as completion is closed; no result existed at cutoff, and the current completion consumer lacks the required digest join.

<!-- # FORMALIZATION_PENDING: second-family review memo; literal consumer-custody and freeze-schema amendments require implementation and tests -->
