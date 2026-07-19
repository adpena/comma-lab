# V10 A2 profiler FIX20 frozen implementation specification

Date: 2026-07-19 America/Chicago  
Lane: `v10_A2_profiler_20260718`  
Status: `FROZEN_BEFORE_IMPLEMENTATION`  
Authority: delegated V10 A2 profiler task plus FIX15-FIX19 custody contracts  
Pointer authority: unchanged; source-hardening only

## Trigger and exact rejected bytes

Three independent fresh reviews examined the exact lexicographic eight-file
FIX19 bundle:

`0c1f7f625f11be8312d282d0da543929935276b6bccabe5c1ae73481606e9d5a`

Review-counter rounds 30-32 are all `NOT_CLEAN`.  The prior 322-test result,
Ruff, format, bytecode, diff, forbidden-operation, no-Fourier, lockfile, and
sacred-tree checks remain useful evidence but do not authorize these bytes.
The clean-pass seal is reset.  No source-freeze commit, lane claim, launch,
production-data write, score claim, promotion, or pointer movement is allowed
until a replacement exact-byte bundle receives three consecutive independent
zero-finding reviews.

## Reproduced FIX19 blockers

### F20.1 receipt restart authorizes the desired target as its own prior

For an existing receipt transition `A -> B`, the first process correctly gives
the atomic writer exact old receipt `A`.  A post-exchange crash leaves target
`B` and displaced scratch `A`.  Fresh-process resume reads current target `B`
in `_validate_resume_root`, `_terminal_custody`, and `run_profile`, then passes
`B` as the alleged prior.  Generic cleanup correctly refuses because the real
displaced prior is `A`; the profile is permanently wedged.

### F20.2 a partial stage-attempt intent occupies an authority pathname

`_atomic_stage` writes canonical attempt JSON directly to the semantic intent
pathname with `O_CREAT|O_EXCL`, then fills and fsyncs it.  A hard exit after any
strict prefix leaves malformed JSON under an intent-looking authoritative
name.  Both recovery validation and direct retry parse that partial file before
they can repair it, so the attempt cannot resume.

### F20.3 retained-orphan validation occurs after fresh publication

With an absent target and a pre-existing hash-valid retained atomic-role orphan,
the generic writer checks active scratch only, creates desired scratch and a
transaction, and publishes the fresh target.  Whole-chain validation then
finds the orphan and raises, but the tree has already mutated.  This violates
the pre-mutation refusal contract.

### F20.4 fully retired existing-target history lacks completion evidence

A fully retained `EXISTING_TARGET_EXCHANGE` record and displaced-prior row are
treated as completed history solely because their paths are retained.  The
validator therefore skips current exact-prior replay and accepts a foreign
caller prior.  That namespace is indistinguishable from a crash after all
retention moves but before the original operation returned.  Integrity and
retention are not completion provenance.

### Verdict scope

These are implementation-level transaction ordering, crash-resume authority,
and completion-provenance findings.  They do not invalidate the FIX18
byte-fed/frozen-SegNet path, factor-2 feasible-set mathematics, support-union
`S`, direct-logit/quotient separation, NoOp Pose boundary, rate accounting,
codec parse-back, or the representation family.

## Frozen FIX20 design

### 1. Generic atomic state has a durable completion proof

- Add an immutable versioned completion record for each canonical atomic
  transaction.  It binds the canonical transaction-record SHA-256, target
  basename, exact parent identity, mode, desired bytes/SHA-256 and target inode
  identity, displaced-prior bytes/SHA-256/inode when present, an admitted-row
  digest, and explicit false-authority flags.
- Construct and fsync the completion record through a generation/no-replace
  protocol.  Never create its semantic final pathname and then fill it.  A
  partial construction file is a strict-prefix, non-authority state.
- Safe ordering is: validate the entire active+retained group and terminal
  target; replay current exact prior for an unresolved existing-target group;
  durably publish the completion record; retain all non-record rows; retain the
  transaction record last; fsync the parent.  A completion record plus active
  rows is still in-flight.  `COMPLETED` requires the exact completion record and
  fully retired bijective group together.
- A fully retired group without its exact completion record is
  `RETIRED_UNFINALIZED`.  It must replay the current caller's exact displaced
  prior before any mutation and may then converge by publishing completion
  evidence.  Missing, partial-only, duplicate, foreign, or mismatched completion
  evidence blocks byte-identically.
- Sequential `A -> B -> C` transitions have distinct proofs.  A proof cannot
  authorize another transaction.  Historical completed groups remain
  non-authorizing and do not force a caller to remember old priors forever.
- Component-length preflight includes every completion construction, final,
  and retained basename before the first mutation.

### 2. Read-only prepublication custody validation is mandatory

- Immediately after immutable argument/name preflight and before target fsync,
  parent creation, scratch creation, transaction creation, retention, or
  publication, classify every active and retained target-scoped atomic role.
- Clean no-evidence state is allowed.  One valid target-absent
  `FRESH_ABSENT_NOREPLACE` prepublication group with its exact designated source
  is resumable.  Unknown, orphaned, duplicate, partial-only, contradictory,
  wrong-role, missing-source, or retained-only designated-source evidence
  refuses with a byte- and inode-identical tree.
- Revalidate this classification at the last authorization boundary before
  transaction creation or no-replace/exchange publication so a late retained
  sibling cannot cross the check.
- Retained naming remains integrity-only; the prepublication classifier grants
  no semantic authority from a basename or self-hash.

### 3. Receipt transitions persist semantic prior authorization

- Before an existing receipt transition `A -> B`, persist and fsync a canonical
  `profiler_receipt_transition_authorization.v1` record.  It binds exact prior
  receipt `A` (canonical payload or lossless exact encoding), its bytes/hash and
  stable identity, desired receipt `B` bytes/hash, profile identity, exact
  rebuild argv, validated stage-chain head/frame count, semantic-validation
  digest, target basename, and false-authority flags.
- The authorization record is written before the receipt atomic transaction.
  It is itself crash-resumable and may not be inferred from the receipt target.
  Absent-target creation uses an explicit `ABSENT` prior state.
- `_validate_resume_root`, `_terminal_custody`, and `run_profile` resolve an
  in-flight receipt exchange from this record and supply exact `A` to generic
  cleanup.  They never read current target `B` and call it the prior.
- The displaced realization must equal the authorization's exact `A`; current
  target must equal exact `B`; identity/argv/stage chain must match.  Missing,
  partial, stale, foreign, target-derived, or self-asserted authorization blocks
  before cleanup or terminal adoption.
- A completion outcome binds the authorization digest to the generic atomic
  completion proof.  A later transition may supersede but never overwrite or
  erase earlier custody.

### 4. Stage-attempt intent construction is non-authority and resumable

- Stop writing JSON directly to the semantic intent pathname.  Store each
  attempt in its existing short, identity/frame/attempt-bound recovery
  directory and atomically publish a canonical `attempt.json` (or an equivalent
  short stable name) through the generic fresh no-replace protocol.
- `INTENT_BUILDING` consists only of recognized atomic construction evidence
  and has no stage/rate/score authority.  The semantic attempt exists only after
  a complete canonical payload and parent fsync.  Prepared/final stage mutation
  is forbidden until that state.
- Recovery may keep one pending `INTENT_BUILDING` attempt and rederive its exact
  identity/frame/attempt/payload/argv.  Retry completes the same attempt; it
  does not allocate a new ordinal.  A non-prefix, wrong identity/frame/attempt,
  duplicate, or foreign construction blocks byte-identically.
- The attempt record, prepared/final payload, success/recovery outcome, and
  retained evidence remain a bijection.  Success or recovery binds the exact
  attempt digest.  Two recovered attempts followed by success remain contiguous
  attempts `0,1,2`.
- Complete legacy FIX19 intent evidence may be read only if it is fully
  canonical and outcome-bound.  Partial legacy semantic intents are unknown
  authority and block; no production profiler run was launched from this arm,
  so no live profile state may be silently migrated.

### 5. No scientific or execution expansion

- Limit implementation to the generic atomic helper, profiler transaction
  custody, and focused regressions in the exact delegated Python bundle.
- Do not change feasible-set math, selection, K/DOF, RD accounting, scorer
  construction, source loaders, governor/admission, storage waterfall, CLI,
  output authority labels, matrix/DAG/equation dispositions, or the sacred n600
  run tree.
- No Fourier/DCT/rFFT, destructive deletion, replace-capable foreign-target
  publication, hidden path loader, GPU dispatch, production cache/profile run,
  score, claim, or pointer movement is authorized by this fix.

## Required FIX20 regressions

1. Existing receipt `A -> B` through the real resume/root/terminal consumers:
   cut after exchange, discard process memory, prove fresh retry uses exact `A`
   and converges.  Missing/foreign/`B`-as-prior/partial authorization refuses
   pre-mutation with an identical tree.
2. Fully retired existing-target group without completion: prior `B` refuses;
   exact prior `A` converges and emits completion.  Exact completed history then
   validates with no prior.  Exercise cuts before/after completion fsync and
   before/after each row and final transaction retirement.
3. Completion corruption matrix: missing, duplicate, partial, wrong transaction
   digest, target, parent, desired inode/hash, prior inode/hash, admitted-row
   digest, and cross-transaction reuse all refuse byte-identically.
4. Absent target plus retained prepared/generation/transaction orphan refuses
   before any transaction-write or publication hook.  Cover valid fresh retry,
   extra/duplicate retained realization, partial-only record, missing source,
   and retained-only source.
5. Stage intent cuts at 0, 1, midpoint, and `N-1` bytes plus after complete
   scratch fsync, transaction fsync, no-replace publication, parent fsync,
   prepared-prefix write, final publication, outcome fsync, and retirement.
   Fresh restart converges on the same attempt ordinal.
6. Wrong pending attempt payload/hash/identity/argv/frame/ordinal, prepared or
   final without durable intent, duplicate outcome, and foreign retained
   construction all refuse before mutation.
7. Run the ambiguity cases through progress, receipt, cache progress,
   certification, completion, profiler recovery-manifest, and stage-chain
   consumers.  Preserve all FIX18/FIX19 directory, frozen-source, atomic,
   multiple-attempt, typed-progress, scientific, support-union, codec,
   governor, storage, false-authority, I/O-taxonomy, and exact-argv tests.

## Acceptance and landing boundary

Run the three owned test modules in one invocation, including every new
transaction/retention cut; Ruff check and format-check; `py_compile` over the
exact eight delegated Python files; `git diff --check`; explicit destructive
operation, hidden-path-loader, and Fourier/DCT/rFFT scans; and a deterministic
ordered component/bundle hash.  Reverify `uv.lock` and the sacred n600 source
tree against their pre-task hashes.

The replacement exact-byte bundle requires three new consecutive independent
zero-finding reviews.  A finding resets the seal.  Only after `CLEAN x3` may the
isolated branch receive a source-freeze commit.  A claim or governed n600 launch
is a separate later action.  MAIN landing review is mandatory.
