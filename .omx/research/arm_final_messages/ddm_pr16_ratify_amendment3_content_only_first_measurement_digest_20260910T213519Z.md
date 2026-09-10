Completed the pr16 adjudication. Full memo: [ddm_pr16 review](/Users/adpena/Projects/pact/.omx/research/ddm_pr16_ratify_amendment3_content_only_first_measurement_digest_20260910.md).

Five verdicts:

1. **AMEND:** content-only comparison is byte-strong and ratified, but completion must record and validate both content identity and the actual retained-root tree.
2. **RATIFY:** run4 registered exactly once in the canonical repo ledger with the matching lock; no snapshot ledger was written.
3. **RATIFY:** pr15’s authorize import-path amendment and foreign-cwd regression are correct.
4. **AMEND-PROSPECTIVE:** introduce typed `prefire_contract_consumer_fix.v1` rows, chained to the unchanged definition and exact latest implementation.
5. **AMEND / INCOMPLETE:** run3 is safely non-replayable but needs append-only terminal reconciliation; run4’s fresh pairing is clean but remains `SPAWNED`.

Literal amendment requirements:

- Completion must join the independently recomputed content digest across context, request, argv, result, and worker provenance.
- It must project the local manifest onto the recorded retained root and validate the actual worker tree.
- `candidate_seal.v3` and its `t4_direct` leg must contain identical runtime-custody objects.
- Consumer-fix rows must preserve exact latest-row refusal, definition-parent identity, ancestry, manifest identity, and typed tests.
- Run3 requires an immutable reconciliation receipt and linked terminal-failure ledger event; its existing records must not be edited.

Validation: 146 tests passed; expanded suite 181 passed with one deliberately excluded sandbox socket test. Unfiltered, that test alone failed during Unix-socket creation with `Operation not permitted`. Ruff and `git diff --check` passed.

Serializer: **rc 17** because Git object/index writes were denied. The verified fallback commit is `85b0f0432b5d66a8eed6c013250201e4f645730a`; [bundle](/Volumes/VertigoDataTier/pact/ddm_pr16/receipts/commit_serializer_fallbacks/20260910T213436.820760Z-14244/intended-commit.bundle). Shared HEAD remains `dce9174ca875852d693f15b6558171220c43e316`; nothing was staged.

Final frontier line:

`effective_frontier.score = 0.1372848557085275; archive_sha256 = 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e; archive_bytes = 180466; axis = contest_cuda`

## NEXT_IF_RESUMED

- **HARVEST-ONLY / owner MAIN / consumer store** `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run4/`; **fire trigger:** the existing call emits its terminal receipt. Do not refire or complete v3 until the amendments land.
- **AMEND / owner MAIN / consumer store** completion code, tests, implementation manifest, and typed freeze row; **fire trigger:** run4 is terminal and all literal requirements in the memo are implemented and reviewed.
- **RECONCILE / owner MAIN / consumer store** run3 terminal reconciliation receipt and ledger event; **fire trigger:** retained artifacts and existing rows rehash exactly.

## LIVE-HYPOTHESES

- Run4 should pass relocated-runtime content validation, but only its retained terminal receipt can confirm that.
- Explicit completion custody can close the gap without changing legacy `t4_direct`.
- A definition-parent plus exact implementation-head chain can batch consumer fixes without weakening freshness.

## DEAD-ENDS

- Equating normal-upload and retained-volume path hashes is closed.
- Dropping path-tree custody entirely is closed.
- Treating amendment `[3]` as a definition change is closed.
- Reusing or scoring run3 is closed.
- Treating run4 dispatch as completion is closed.