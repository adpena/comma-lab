# V10 A2 profiler review FIX15 specification — 2026-07-18

`research_only=true`  
`authority_axis=[macOS-CPU advisory]`  
`score_claim=false`  
`promotion_eligible=false`  
`pointer_delta=0`

## Trigger

Three fresh reviews of exact canonical-manifest bundle
`a5cacb5f5136f3ae6d8c003475f90bfc729aa9e7588041e206891c7429a3d353`
are `NOT_CLEAN`.  The mathematical factor-2 bounds, direct frozen-logit versus
rank-4 diagnostic separation, receiver parse-back, NoOp Pose boundary, and
false-authority labels remain clean.  Source freeze and every real n600 launch
remain blocked.

The replacement must close one cache-layout defect, executed-scorer byte
custody, repeat-interruption recovery, and the remaining read-to-commit
identity gaps as one fail-closed transaction design.  A narrow patch that only
makes the current tests pass is insufficient.

## Required implementation

### Extractor receipt must not invalidate its cache

- The extractor operational receipt and its stable prepared sibling MUST live
  outside the exact immutable feature-cache root.  Provide one deterministic
  helper for that sibling receipt path; do not add an unvalidated optional file
  to the cache grammar.
- The receipt must still bind the exact cache root, canonical rebuild argv,
  storage preflight, completion control, and false-authority state.
- A complete receipt, a strict-prefix interrupted receipt write, and a
  complete-write/pre-rename interruption must all reconcile deterministically
  without changing the cache-root entry set.
- After receipt emission or recovery, `validate_feature_cache`,
  `SegnetHeadFeatureCache.resume`, and the profiler feature-cache binding must
  continue to accept the same committed cache bytes.

### Executed SegNet bytes must equal recorded bytes

- Upgrade source-file custody reads to bind one non-symlink, link-count-one
  regular-file descriptor to its path identity for the entire hash/read.  A
  pathname replacement or content/identity change during the snapshot blocks.
- Extraction must take a complete stable source snapshot before scorer
  import/model load and another after load/factorization.  The snapshots must
  be byte-for-byte equal before cache creation/resume or any frame commit; the
  agreed snapshot is the one bound into cache identity.
- Profiling with `--score-segnet` must likewise take stable source snapshots
  before and after scorer load.  It must build the feature binding and profile
  identity only from equal snapshots and refuse before output initialization,
  stage creation, or receipt emission on drift.
- Cover `modules.py`, `frame_utils.py`, SegNet weights, the executed TAC scorer,
  and every other source already named by the existing binding.  Exact paths
  without stable byte equality are not sufficient custody.

### Descriptor-bound metadata and stage reads

- Cache JSON reads must consume bytes through a no-follow descriptor whose
  `fstat` identity matches the pre-open path identity and whose path still
  names that inode after the read.  Require canonical JSON bytes plus the
  canonical newline; read errors or identity changes preserve bytes and block.
- Stable prepared-prefix rewrite must reopen without following links, verify
  the descriptor is the exact previously read inode before truncation, write
  and fsync through that descriptor, revalidate the source pathname, then
  rename.  A pathname/identity change must not modify either the foreign file
  or the metadata target.
- Apply the same bound-read rule to profiler progress/receipt prepared files,
  stage intents, prepared/final stage payloads, and recovery payloads at every
  read-to-rename or read-to-unlink boundary.  Unknown names, links, read
  errors, and identity drift preserve all bytes and block.

### Progress snapshot is an authorization token, not an arbitrary mapping

- `_resume_from_stage_chain` may accept either no snapshot (one canonical
  direct load) or the exact `_ValidatedProgressSnapshot` type returned by
  `_validate_resume_root`.  Reject plain mappings supplied as snapshots.
- Recheck the typed snapshot's path identity after every potentially long
  semantic replay and immediately before ANY stage rename, intent unlink,
  recovery move, progress replacement, or receipt-affecting mutation.
- A progress-path change during semantic replay must leave progress, prepared
  stage, final stage, intent, and recovery custody byte-identical.

### Multiple interrupted attempts per frame

- One frame may be interrupted and recomputed arbitrarily many times.  Give
  each durable stage intent a deterministic attempt ordinal (or an equivalent
  collision-free persisted transaction identity) and bind that identity into
  its recovery transaction and canonical manifest.
- Preserve every positively identified missing/size-short attempt under the
  profile output; never overwrite, conflate, or delete a prior attempt.
  Terminal custody must validate and report every attempt in deterministic
  order while excluding all recovery bytes from stage/rate/score/promotion
  authority.
- Recovery must be idempotent before manifest commit, during the stable
  manifest write, before payload move, after payload move, and before/after
  intent unlink for both the first and later attempts.  A later recomputation
  must remain possible after each recovered attempt.

### Validate recovery grammar before mutation

- Resume-root validation must parse the complete recovery grammar before
  adopting an orphan/prepared stage or advancing progress.  It may recognize
  only the exact identity root, exact transaction names, and the explicitly
  recoverable in-progress states needed by the transaction state machine.
- An unrelated identity, malformed transaction, unknown entry, link, or
  inconsistent manifest/payload must block before any progress, stage, intent,
  or recovery mutation.  Terminal validation remains stricter and admits only
  completed transactions.

## Required regressions

- Minimal extraction-through-receipt: successful and interrupted receipt
  writes leave the cache canonical, resumable, and consumable by profiler
  binding; no `receipt.json` entry appears inside the cache root.
- Replace manifest/progress/certification/completion and stable prepared
  pathnames at controlled read/rewrite boundaries.  The operation blocks and
  preserves the original, foreign, target, stage, intent, and progress bytes.
- Change each frozen scorer source from a controlled loader hook between the
  two snapshots.  Extraction and profiling refuse before cache/stage/output
  authority is emitted; stable controls pass.
- A plain mapping passed as `progress_snapshot` is refused.  A real validated
  snapshot is read once and reused; changing its path during semantic replay
  produces zero mutation.
- Recover two consecutive size-short cuts of the same frame, then recompute it
  successfully and reconstruct a final receipt reporting both attempts.
  Inject second-attempt cuts before manifest, before move, after move, and near
  intent removal; every retry is idempotent and lossless.
- Unknown or incomplete unrelated recovery entries with an orphan/prepared
  stage block before progress or stage adoption.
- All FIX14, scientific, source-seed, codec, governor, storage, and I/O taxonomy
  regressions remain green.  `uv.lock` and the sacred witness directory remain
  unchanged.

## Acceptance

Run the three delegated test modules, Ruff check/format check and bytecode
compile over all eight delegated Python files, `git diff --check`, the explicit
no-Fourier scan, and a deterministic exact-component/bundle hash.  No
production extraction/profile launch, score, promotion, Pose, factor-10,
global-compression-minimum, or contest-axis authority is granted.

The replacement exact-byte bundle requires three new consecutive zero-finding
reviews.  Prior clean reviews do not carry forward.  MAIN landing review is
mandatory.
