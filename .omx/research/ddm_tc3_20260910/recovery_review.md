# TC3 completed-public-receipt recovery review

Independent geometry-subarm review of
`experiments/ddm_tc3_recover_public_receipt.py`.
Final source SHA-256:
`fd88972a72812ca9611c2b0730c91e76ff73d57df63a7148e74fa634fae3bf08`.
Original receiver remains unchanged at
`b27900ebe52e6365844361f1931451461ac50dc4ef6536a70347dc4288561000`.
No job, scorer, rehash of either full raw, source/runtime edit, state mutation
or receipt recovery was executed by this reviewer. Parent executes the helper
after review. This review note is the only artifact written by the subarm.

## RECALL EVIDENCE

Read the new helper, original receiver and file-fact/atomic-JSON helpers.
Read the original rp1 `PARSEBACK_RESULT.json`, its `0.raw.MOVED.json`, the
successful literal-shell `PROCESS.json` and log, the public `RUN_BINDING.json`,
and the failed outer wrapper's `safe_run.json` and traceback. Earlier full
charter/common-contract and receiver-custody recall is recorded in sibling
`receiver_review.md`. No new scientific prior is introduced by this recovery.

The actual failure is confined to receipt construction: the inner literal
candidate shell returned zero, then the outer receiver wrapper tried to stat
the old source raw path after another arm had cold-stored it. The wrapper
receipt has `exit: 1`, `child_exit_nonzero: true` and
`receipt_status_disagrees_with_exit: true`, despite its `status: "ok"`.
The traceback confirms `FileNotFoundError` at the source raw stat. The helper
uses the separately recorded successful shell child and preserves the failed
outer wrapper fact; it does not reinterpret that misleading status string.

## Evidence inspected before review

The actual shell PROCESS records the literal candidate `bash inflate.sh`
command, returncode zero and 993.3946199170023 seconds. Its 5,872-byte child log
was independently rehashed and equals its recorded SHA:
`c6bf9d446e1608d1309421f8009c511dcc8bafee7100988630b4dbd76fed6c2b`.
The single complete report names candidate archive 180,107 B, SHA
`2ee6e292255a63d48391f852cde26b573fe3e8de378fabfa056632c24e7c6c13`,
600 pairs, full token SHA
`4aa519a25e4b02afb564498025b366cb9007ea663093c60bf8ce90d079dc8791`,
and raw 3,662,409,600 B, SHA
`fd4b08e6aa0e967cc1453b9363c5f5ad243cb3bb57417f0f383cdb4bbfbfb2e8`.
These are child-report observations until the recovery's fresh payload hashes
complete the comparison. This review does not claim to have rehashed the raw.

The source move record points to
`/Volumes/APDataStore/pact/ddm_rp1_round1_bulk/parseback_0.raw` and agrees with
the original parse-back receipt's raw size and SHA. Both source cold-store raw
and current candidate raw exist at 3,662,409,600 B; the candidate token field
exists at 117,964,800 B. Independently rehashed small custody records:
- rp1 `0.raw.MOVED.json`:
  `5728d89eff30a555f91ef298b81ecab88aa11dad7987b21af7c940dc05c0e3ac`.
- rp1 `PARSEBACK_RESULT.json`:
  `cd33a4d5dbaeb5cc8fcd4efca61b0fc2cdbf37caacb5767288dd2939549189ee`.

## Visible review pass 1: execution and byte custody — CLEAN

Read the complete final helper. It reuses the unchanged receiver's pointer,
source manifest, field and staged-runtime pin. The successful child's exact
argv and log fact must match, and its saved launch runtime/receiver binding
must equal the current original producer and staged runtime. Both retained
native-library files and sources are verified. Exactly one report must name
the current candidate archive and 600 pairs.

The helper reads the source archive identity from the original parse-back
receipt, derives the original raw path from that receipt, follows the adjacent
move record only into the expected APDataStore tier, and freshly hashes the
cold-store raw and candidate raw. Their bytes and SHA must equal the move
receipt, original parse-back raw fact and successful child report. It also
hashes the complete candidate token checkpoint and checks both the pinned
field SHA/size and the child report's decoded-token SHA. No unresolved blocker.

Shared assumption: the preserved process/log/launch receipts describe the
same successful child whose outputs remain on disk. Exact paths, log digest,
runtime identity and output hashes make that assumption reviewable; missing
or changed evidence refuses. No fresh decoding or scorer outcome is inferred.

## Visible review pass 2: recovery provenance and restart — CLEAN

Re-read the complete helper separately with the original public wrapper and
its failed traceback. The result keeps the original raw fact before the move,
the current cold-store fact, original parse-back receipt and move-receipt fact
separately. It records the helper source, successful child receipt and failed
outer wrapper receipt, with `rerendered: false`. The original receiver/runtime
remains frozen, so recovery does not invalidate its execution bindings.

All validation precedes output receipt creation. The recovered receipt and
normal consumer receipt are written with the existing atomic-JSON mechanism;
a crash between them can re-run the same read-only validation and complete
the second write. No decoder, renderer, compiler, subprocess, scorer or payload
mutation is called by the helper. The full raw objects remain in their existing
stores. The result remains `[macOS-CPU advisory]`, `score_claim: false`, and
explicitly leaves contest-CUDA identity and score to MAIN. No unresolved blocker.

Shared assumption: content hashes and the existing source parse-back receipt
remain the identity authority after relocation. The move changes location,
not the expected bytes. Parent reported Ruff green; no separate Ruff/test run
was performed by this subarm. Two source passes apply only to the hash above.

Disposition: FOLDED into parent ddm_tc3's active receipt recovery. Owner parent
ddm_tc3; consumer
`/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/move39/PUBLIC_IDENTITY.json`;
fire trigger completion of these two clean source passes and parent controls.

LIVE-HYPOTHESES: fresh raw hashing should complete the existing public identity
proof because the move receipt and successful child report already agree on
the expected sizes and hashes. This is a custody check, not new codec science.

DEAD-ENDS: rerendering solely to repair this missing parent receipt is
unnecessary when the complete successful child evidence and both raw objects
verify. The outer wrapper's `status: "ok"` cannot establish success because its
exit is 1; use the preserved, separately typed child PROCESS evidence.
