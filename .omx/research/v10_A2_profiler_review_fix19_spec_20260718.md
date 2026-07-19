# V10 A2 profiler FIX19 frozen implementation specification

Date: 2026-07-18 America/Chicago  
Lane: `v10_A2_profiler_20260718`  
Status: `FROZEN_BEFORE_IMPLEMENTATION`  
Authority: delegated V10 A2 profiler task plus FIX15-FIX18 custody contracts  
Pointer authority: unchanged; source-hardening only

## Trigger and exact rejected bytes

Three independent fresh reviews examined the exact lexicographic eight-file
bundle:

`c202200fec65b95c63627c75d57f92abafc39940ced60d13a8ef54ce2a1e6807`

The frozen-weight/source-authority lens was CLEAN.  The transaction lens
reproduced two defects and the whole-bundle lens reproduced one.  Review
counter rounds 27-29 record those verdicts.  The focused 298-test suite,
targeted tests, Ruff, format check, `py_compile`, `git diff --check`, static
destructive-operation scan, and no-Fourier scan were green, but they did not
cover the accepted states below.  The clean-pass seal is reset.  These bytes
authorize no source-freeze commit, claim, launch, score, promotion, or pointer
movement.

## Reproduced blockers

### F19.1 fresh publication mutates the target before durable authority

The generic atomic writer creates and fsyncs an empty target placeholder before
it writes a complete transaction generation.  A hard process exit at that
boundary leaves `target == b""`, a complete desired scratch file, and no
transaction.  Every retry preserves but permanently refuses that state.

### F19.2 consumer prior authorization is self-asserted on retry

Committed-target cleanup trusts a canonical transaction row whose
`prior_authorization` says `CONSUMER_AUTHORIZED_EXACT_PRIOR`, but it does not
recompare the recorded/displaced bytes with the current caller's
`expected_prior_payloads`.  A coordinated canonical record and valid-but-
foreign scratch can therefore be retained and accepted while the caller
authorizes a different exact prior.

### F19.3 retention integrity is treated as transaction provenance

Profiler grammar removes retained entries after checking only their decoded
original-name grammar and self-encoded byte count/SHA-256.  A separately
injected hash-valid retained recovery scratch containing `b"FOREIGN"` was
accepted by terminal custody; a nonempty retained object claiming a stage-
intent role was also accepted.  A retention filename proves integrity of the
retained bytes, not why they exist or which transaction/outcome owns them.

### Verdict scope

These are implementation-level crash recovery, exact-prior authorization, and
retained-role provenance findings.  The FIX18 byte-fed SegNet/source-custody
path reviewed clean.  The findings do not invalidate factor-2 feasible-set
mathematics, support union `S`, direct-logit/quotient separation, NoOp Pose
boundary, rate accounting, codec parse-back, or the representation family.

## Frozen FIX19 design

### 1. Fresh targets use a durable `FRESH_ABSENT` no-replace transaction

- Eliminate the empty target placeholder.  A fresh atomic write MUST NOT create
  or mutate the target pathname before complete durable transaction evidence.
- First recover or create and fsync one complete desired-payload generation.
  Then write a complete transaction generation with an explicit
  `FRESH_ABSENT_NOREPLACE` mode and fsync both that record and its parent
  directory.  The record binds target basename, exact parent custody, desired
  bytes/SHA-256, the one designated desired-source basename and stable
  descriptor identity, and every other admitted writer scratch row.
- Publish the designated desired source directly to the absent target with the
  descriptor-bound kernel `RENAME_NOREPLACE` primitive.  A pathname absence
  precheck is advisory only; the kernel no-replace result is the linearization
  authority.  Fsync the moved file and parent after publication.
- The only reachable retry states for a complete fresh transaction are:
  (a) designated source present and target absent, which retries the no-replace
  move; or (b) designated source absent and target equal to the exact desired
  bytes on the same bound inode, which is post-publication.  Both present, both
  absent, a mismatched target, source satisfied by retained/same-hash foreign
  custody, contradictory complete records, or partial-only evidence blocks
  without mutation.
- Source-to-target continuity is inode-bound, not byte-hash-only.  A
  same-byte/different-inode target is an ABA/late-destination state and blocks.
  Reconcile active transactions before any target-equals-desired early return.
- A crash before the post-move parent fsync may recover as either exact
  pre-publication or exact post-publication state.  Both must converge.  A
  clean steady-state target with no active transaction/scratch remains valid;
  an apparent interrupted fresh publication may not adopt an exact target
  without its complete transaction.
- Cleanup validates and losslessly retains every admitted non-source scratch
  before retiring transaction generations.  The last complete transaction is
  retired last.  No unlink, replace, truncate, or destructive rollback is
  introduced.

### 2. Existing-target cleanup replays caller authority

- Pass the immutable `expected_prior_payloads` through every committed-target
  cleanup and retry path.  A `CONSUMER_AUTHORIZED_EXACT_PRIOR` label is never
  authority by itself.
- The transaction's descriptor-admitted prior bytes/size/SHA-256 and stable
  identity must resolve bijectively to the active or retained displaced source.
  Return and compare the exact retained snapshot rather than a boolean match.
- Before the first cleanup/retention mutation, require those exact prior bytes
  to equal one current caller-authorized payload.  An empty authorization set,
  a different authorized payload, a missing/duplicate realization, or a
  self-consistent but foreign record blocks with a byte-identical tree.
- Transaction schema/mode, target desired bytes, source/prior roles, complete
  and partial record generations, active scratch, and retained scratch must
  form one non-contradictory bijection.  Every partial record is a strict prefix
  of the same complete record; extra rows or extra realizations block.
- Apply this exact replay to recovery manifest, progress, receipt, cache
  progress/certification/completion, and every other generic atomic consumer.
  A generic JSON reconciliation helper may not invoke cleanup with an empty or
  fabricated authorization; its caller must provide the target's exact prior
  transition contract.

### 3. Retained bytes require role-specific outcome provenance

- Keep `validate_retained_file` explicitly integrity-only.  No caller may
  subtract a retained name from a live or terminal grammar merely because its
  original basename, byte count, and SHA-256 are canonical.
- For generic atomic targets, validate a complete active or retained
  transaction group before excluding any retained scratch.  The final target
  must equal the transaction desired payload; each admitted row must resolve
  exactly once to its active/retained object (or, only for the fresh designated
  source, the exact moved target); displaced prior bytes must satisfy the
  caller's semantic authorization.  Orphans, duplicates, wrong targets,
  wrong roles, contradictory records, and independently injected retained
  objects block.
- For deterministic creation-prepared custody, a role-specific validator must
  bind the retained object to its certified creation record and exact expected
  target payload or an actual strict prefix of that payload.  An allowed-looking
  basename alone is insufficient.
- Before the first stage-intent or prepared-stage mutation, persist and fsync a
  canonical `profiler_stage_attempt_transaction.v1` record.  It may be an
  upgraded intent payload or a separate immutable record, but it must bind the
  identity SHA-256, frame, attempt, final/prepared/intent basenames, intended
  bytes/SHA-256, exact rebuild argv, and false-authority flags.
- A successful final stage receipt and a failed/short-attempt recovery manifest
  must each bind the exact stage-attempt transaction SHA-256.  Retained intent
  bytes must match the transaction's declared intent semantics; retained
  prepared bytes must match the corresponding recovery outcome.  Each attempt
  record and each retained object is consumed by exactly one final or recovery
  outcome, attempt ordinals remain contiguous, and the eventual successful
  receipt records its real attempt ordinal.
- Validation order is parse all active/retained objects, validate atomic
  transaction groups, validate stage-attempt/outcome bijections, validate
  recovery continuity and stage chain, and only then exclude proven retained
  false-authority custody.  Any failure precedes progress adoption, receipt
  emission, score/rate use, or filesystem mutation.
- Existing retained bytes remain preserved.  Unknown or unprovable bytes block;
  they are never deleted or silently promoted to authority.

## Required regressions

1. Through the production atomic writer, hard-exit after desired scratch fsync
   but before any transaction, during a partial transaction write, immediately
   after complete transaction fsync, immediately before/after no-replace
   publication, before post-move fsync, and before/after each scratch/record
   retention.  Repeated retries converge from every reachable state and keep
   unrelated bytes/inodes unchanged.
2. At the fresh publication boundary, introduce a late foreign target.  Both
   source and foreign target remain byte-identical and retry blocks.  Exercise
   both-present, both-absent, target-mismatch, same-byte/different-inode ABA,
   missing-source, partial-only, and contradictory-record states.
3. Produce a real post-exchange cut from exact prior `A`; retry once with only
   prior `B` authorized and require pre-mutation refusal, then with `A`
   authorized and require convergence.  Cover active and already-retained
   displaced sources and a forged canonical transaction that self-asserts a
   foreign prior.
4. Run the exact-prior cases through recovery manifest, progress, receipt,
   cache progress, certification, and completion consumers.
5. Inject independently retained arbitrary bytes and valid-but-foreign
   canonical JSON under allowed recovery atomic roles.  Also inject an extra,
   duplicate, missing, wrong-target, wrong-role, and partial-only transaction
   group.  Every case blocks byte-identically; one legitimate retained
   post-linearization group remains accepted.
6. Inject a nonempty retained stage intent; an empty but orphan retained intent;
   wrong frame/attempt/intended size/hash/transaction digest; and retained
   prepared bytes without their exact recovery outcome.  All block.  Two
   recovered attempts followed by a successful third attempt validates with
   contiguous outcome custody.
7. Inject hash-valid retained creation/output objects with allowed original
   names but foreign payloads.  Only exact transaction- or certified-prefix-
   backed objects are excluded from grammar; every orphan blocks before
   mutation.
8. Re-run all FIX18 directory, atomic, source-loader, multiple-attempt,
   typed-progress, scientific, support-union, codec, governor, storage,
   false-authority, I/O-taxonomy, and exact-argv regressions.

## Acceptance and launch boundary

Run the three owned test modules, the exact new transaction/retention tests,
Ruff check and format-check, bytecode compile over the exact eight delegated
Python files, `git diff --check`, explicit destructive-operation and
no-Fourier/DCT/rFFT scans, and a deterministic ordered component/bundle hash.
Reverify `uv.lock` and the sacred n600 source tree against their pre-task
hashes.

The replacement exact-byte bundle requires three new consecutive independent
zero-finding reviews.  Earlier clean lenses do not carry forward.  Only after
that seal may the serializer land a source-freeze commit.  A lane claim or
governed n600 launch remains a separate later action.  MAIN landing review is
mandatory.
