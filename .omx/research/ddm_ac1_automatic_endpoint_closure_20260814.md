# DDM AC1 — automatic Modal endpoint closure (2026-08-14)

## Result

The dispatch-to-endpoint chain is now executable apparatus. The new
`tools/modal_endpoint_close.py` composes the canonical Modal poller, claim writer,
call-ID ledger writer, NP1 `NEXT_IF_RESUMED` reader, detached launcher, and commit
serializer. The EC2 reference dispatcher emits a complete typed closure manifest
and arms this closer immediately after the provider call and accepted claim are
recorded.

The closer does real work: it waits for provider terminality, derives success or
failure from the returned object, closes the claim row before the call-ID row,
downloads every declared payload, verifies exact bytes and SHA-256, optionally
commits a declared git-blocked memo, extracts the NP1 continuation surface, and
writes typed closure and done receipts. A provider poll deadline is explicitly
nonterminal and does not falsify either ledger. A payload mismatch or download
failure refuses closure custody and preserves the remote object plus any local
staging bytes.

This is dispatch apparatus, not a score-bearing candidate. No scorer, renderer,
training job, or exact evaluation ran; the frontier did not move.

## Closure-manifest convention

Every data-bearing dispatcher result should carry this additive block. Paths are
relative to the named Modal volume, not local mount paths. The payload list must
contain every exact record in every returned `retained_payloads` and `payloads`
block, plus terminal documents such as the final result, selected result, and
worker log when those are retained remotely.

```json
{
  "schema": "modal_endpoint_closure_manifest.v1",
  "volume_name": "comma-example-retained",
  "lane_id": "canonical_lane_id",
  "instance_job_id": "modal:unique-run-id",
  "payloads": [
    {
      "name": "endpoint.retained_payloads.argmax_n600",
      "remote_path": "unique-run-id/endpoint/retained/argmax_n600.npy",
      "bytes": 117964928,
      "sha256": "64-lowercase-hex-characters"
    }
  ],
  "memo_handoff": {
    "schema": "modal_endpoint_memo_handoff.v1",
    "path": ".omx/research/example.md",
    "sha256": "64-lowercase-hex-characters",
    "base_sha256": "optional-pre-edit-sha-or-omitted-for-new-files",
    "message": "example [no-triality] [p0-ledger-ok]"
  }
}
```

`memo_handoff` is optional. The same typed memo object may instead appear as
`git_blocked_memo` in the returned result or in a fenced JSON block in a retained
arm final message. Conflicting declarations refuse. The serializer is invoked
only after the declared path exists, its SHA matches, and the required message
tags are present.

The EC2 emitter walks its real final result, converts Modal-internal paths to
volume-relative paths through the sealed `run_id`, de-duplicates only identical
path identities, and adds `FINAL_RESULT.json`, `SELECTED_RESULT.json`, and
`worker.log`. Both success and failure returns carry a manifest; a failure still
retains the worker log. This addition does not alter the sealed request or fire
inputs.

## Endpoint state contract

The terminal classification is data-driven:

| Returned evidence | Ledger outcome | Claim outcome |
|---|---|---|
| `training_complete=true`, `rc=0`, or a declared success status | `harvested` | `completed_endpoint_harvested` |
| nonzero rc, error field, `training_complete=false`, or failure status | `failed` | typed `failed_endpoint_*` |
| provider exception returned by the canonical poller | `failed` | `failed_endpoint_remote_exception` |
| local polling deadline | no write | no write |
| ambiguous terminal object | `failed` | `failed_endpoint_ambiguous_result` |

The claim row is terminalized first so the canonical call-ledger writer cannot
emit the after-the-fact active-claim blocker on the normal path. Both canonical
stores are then read back and must be terminal. A completed closure receipt is
the idempotency key: a second invocation returns it without appending either
ledger, re-downloading payloads, re-committing a memo, or duplicating a queue row.

`tools/preflight_hook.py` now has an additive warn-only check over newly added
`experiments/*_modal_*.py` files containing `.spawn(`. It reports a missing
manifest convention. A genuinely data-free spawn can use a same-line
`MODAL_CLOSURE_MANIFEST_FREE:<substantive rationale>` waiver; short or placeholder
rationales are rejected. The check is deliberately labelled lexical coverage,
not proof that a runtime returned complete custody.

## Dogfood receipt

The closer was run in dry-run mode against the retained EC2 R4 result and BG1
final result:

- call ID: `fc-01M0073TSNJEKW2BA4XTGF950X`;
- retained EC2 final-result SHA-256:
  `b9df6af2175b2003e8a236e67e3315b60a065c7913792fcd93464c203c321a1b`;
- typed receipt: `/Volumes/VertigoDataTier/pact/ddm_ac1_20260814/retained/dogfood_r4/ENDPOINT_CLOSURE.receipt.json`,
  8,408 B, SHA-256
  `b9624202314bde5c39db178eb7960060e311ebd312d4315b23f4eff1f5ce1f4e`;
- done receipt: `/Volumes/VertigoDataTier/pact/ddm_ac1_20260814/retained/dogfood_r4/ENDPOINT_CLOSURE.done.json`,
  415 B, SHA-256
  `aad1f611b8a7af83c4d3c2ec148e897bba878ab293e22039c4f81ca38a05b74c`.

The receipt status is `DRY_RUN_VALIDATED`. Both existing canonical ledgers were
terminal and therefore no-ops. The manifest enumerated 5/5 payloads: the
117,964,928-B argmax field, 67,055-B batch receipts, two 187,723-B archives, and
the 1,369-B adapter. Three retained local fixtures verified against the pinned
hashes `803a1d8755ca...`, `ffa88ae44787...`, and `9559c2ab5128...`; the two archives
were absent from the bounded fixture store and were honestly reported as
`would_pull`. Dry-run wrote no claim, call-ledger, or NP1 queue row.

## Verification

- New AC1 fault-injection suite: **36 passed**. This includes the terminal-status
  table, remote exception and local deadline split, manifest completeness,
  path traversal refusal, destination-directory creation, download failure,
  SHA mismatch preservation, memo SHA and serializer gates, NP1 heading and
  omission behavior, receipt round-trip, real temporary dual-ledger closure,
  and an idempotent second run with exactly 2 call rows and 2 claim rows.
- Existing EC2 and preflight-hook suites: **95 passed**.
- Total focused tests: **131 passed**.
- Ruff on all five Python surfaces: passed. CPython compilation and `git diff
  --check`: passed.
- Two review-tracker passes were recorded for every changed Python file. No
  `REVIEW_GATE_OVERRIDE` was used on Python.

## RECALL EVIDENCE

The recall pass searched the complete `.omx/research/` memo and arm-receipt
corpus by content with `endpoint closure|modal harvest|dual-ledger|volume get|
manual harvest|hand-poll|poller`; the current claim/call ledgers with the exact
R1-R4 call and job IDs; `CANONICAL_RESEARCH_INDEX*`; `sub015_DAG_*` FEED blocks;
the hot state and task ledgers with `AC1|EC2|BG1|BG2|MC35|#978|#982|#984`; the
v7.5/v8 and operating surfaces; and all 435 canonical equations using
`endpoint closure|dual-ledger|modal harvest|payload harvest`.

Beyond the charter seeds, recall found:

1. `modal_harvest_ledger_execute_fix_20260516_codex.md` had already fixed the
   older harvester's false `--execute` path and terminal call-ledger duplicate.
   AC1 therefore reuses the call ledger and adds the missing dispatch-time
   closer, claim closure, payload custody, memo handoff, and receipt contract;
   it does not build another general harvester.
2. The DAG's FEED-513 entry had already landed Modal single-flight and
   dual-ledger reconciliation. That changed AC1 from a new reconciliation
   store into a composition of the canonical writers with read-back proof.
3. HG1 had already measured recovery idempotence on a separate PO1 dispatcher
   using check-before-append plus a lock. AC1 adopts the stronger completed-
   receipt no-op at the whole endpoint boundary and tests exact row counts.
4. JS1C records the concrete Modal CLI trap: a nonexistent destination
   directory is interpreted as the target filename. That made parent-directory
   creation a pre-download invariant and a tested condition.
5. DT1 explicitly assigns Modal endpoint closure to AC1. No duplicate cure was
   found in its work surface. The 435-equation registry had 0 exact hits for the
   AC1 surface, so no mathematical identity superseded this empirical apparatus.
6. BG2 and MC35 are already terminal local work, not live endpoints to mutate.
   The live use of AC1 is therefore the next data-bearing Modal dispatch in
   those successor families, not a fabricated retroactive fire.

## Boundaries

- `[local CPU unit/integration tests]`: 131 focused tests only.
- `[scorer-free retained-fixture dry-run]`: the R4/BG1 receipt and five-entry
  payload census. It is not proof of current Modal network reachability or a
  successful live `modal volume get`.
- No Modal job was spawned, no paid work ran, no scorer slot was touched, no
  archive was evaluated, and no score or promotion claim was made.
- Existing unrelated working-tree edits were preserved. The EC2 dispatcher and
  preflight hook already contained uncommitted dependency-closure and receipt-
  wording edits when AC1 began; AC1's serializer handoff must use exact-hunk
  patch mode for those shared files rather than absorbing them.

Effective frontier remains CP135 at **S = 0.16195513827824176 @ 186,252 B
`[contest-CUDA T4, n600]`**. Own-vehicle frontier remains LC2 at
**S = 0.16959899569230852 @ 187,226 B `[contest-CUDA T4, n600]`**.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN / next MC35-BG2-family Modal dispatcher owner. Consumer store: that dispatch's named SSD `endpoint_closure/` and `retained/endpoint_payloads/` directories. Fire trigger: before the next data-bearing MC35/BG2 successor `.spawn()`, emit `modal_endpoint_closure_manifest.v1`, register a unique claim and call ID, and arm `tools/modal_endpoint_close.py` through `tools/launch_detached_process.py`; MAIN then adjudicates the typed receipt.**

## LIVE-HYPOTHESES

- A manifest emitted from the remote final object will eliminate the BG1-style
  skipped-harvest arm cycle because the closer can enumerate payloads without a
  human reconstructing paths. This is plausible from the R4 dogfood's complete
  5/5 reconstruction, but still needs one live provider-backed run.
- Closing the claim before the call ledger will eliminate the observed loud
  after-the-fact dual-ledger blocker on the normal path. Temporary-ledger tests
  prove ordering locally; only a live Modal endpoint tests provider timing.
- The typed memo handoff can eliminate hand commits for git-blocked endpoint
  memos when the final message supplies exact path, SHA, base SHA, and message.
  Its SHA/refusal and serializer invocation are tested, but no live arm declared
  such a handoff in this unit.

## DEAD-ENDS

- A second polling loop was rejected because `tools/modal_harvest_poller.py`
  already held the canonical timeout/remote-exception behavior; AC1 factors and
  calls it.
- A print-only recovery runbook was rejected because it would preserve the
  manual failure class. The closer executes canonical writes and downloads.
- Treating a local poll deadline as remote failure was closed because it would
  falsify provider terminality and prematurely close both ledgers.
- Trusting filenames, sizes, or a winner-only payload was closed by the complete
  manifest census plus per-payload SHA verification.
- Retrofiring BG2 or MC35 was closed because both searched units are already
  terminal; there is no live endpoint in those scopes for this arm to mutate.
