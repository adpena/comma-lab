# V10 A2 profiler review FIX14 specification — 2026-07-18

`research_only=true`  
`authority_axis=[macOS-CPU advisory]`  
`score_claim=false`  
`promotion_eligible=false`  
`pointer_delta=0`

## Trigger

Independent review of exact ordered FIX13 bundle
`ce3fe48ab444ba49a8715c539bc579964b16d5dc19007c6b5fc789e015bfc931`
found one progress-read taxonomy gap and one consolidated P0 abrupt-write
resumability class.  `_resume_from_stage_chain` rereads a previously validated
progress pointer through raw `read_text/json.loads`; a cut during a stage
`.prepared` write can leave an unclassifiable truncated file that wedges every
resume; and PID-named `atomic_json` temporaries can survive SIGKILL and either
block exact profiler roots or contaminate feature-cache staging.

The scoped scientific FIX13 review is `CLEAN`; the I/O and operations reviews
are `NOT_CLEAN`.  No seal exists.

## Required implementation

### One progress snapshot

- `_validate_resume_root` must return its already canonical, regular,
  link-count-one validated progress object.
- Production resume must pass that immutable snapshot into
  `_resume_from_stage_chain`; the chain reconciler must not reopen progress.
- Direct unit-test callers may omit a snapshot, in which case the reconciler
  performs exactly one `_load_canonical_object` read.
- Recursive prepared-stage adoption reuses the same snapshot.  Only a successful
  explicit progress replacement creates a new pointer; path swaps, noncanonical
  equivalents, Unicode failures, and read `OSError` never leak a weaker error
  taxonomy or mutate state.

### Stable resumable metadata writes

- Replace PID-named atomic temporaries with one deterministic sibling name per
  target, exposed by a helper so exact root validators can recognize it.
- A stable temporary must be a non-symlink regular file with link count one.
  Read failures preserve it and fail closed.
- When the caller supplies the exact deterministic payload, a complete-equal
  temporary is adopted and a strict byte prefix is certified interrupted
  metadata scratch and may be rewritten.  Equal-length drift, non-prefix
  drift, competing names, hardlinks, and symlinks preserve bytes and block.
- Never remove the stable temporary in a `finally` path after a failed write;
  interruption evidence must survive for the next deterministic call.
- The profiler resume root may temporarily contain only the exact stable
  prepared names for `progress.json` and `receipt.json`.  Progress reconciliation
  and final receipt regeneration must consume them; legacy PID or unknown names
  remain forbidden.
- Feature-cache final validation may tolerate only the exact stable prepared
  runtime names needed for progress/completion recovery, never arbitrary
  debris.  Feature-cache creation must never promote a staging directory that
  contains a prepared temporary.  An identity-matched, exact-subset pre-final
  staging tree is certified rebuildable by its first scratch record and may be
  rebuilt; unknown names/links or scratch drift preserve all bytes and block.

### Stage write intent and lossless interrupted-byte custody

- Before opening `.frame_NNNN.bin.prepared`, `_atomic_stage` must atomically
  create and fsync one empty, regular, link-count-one intent marker.  Its exact
  filename binds final stage name, intended payload byte count, and intended
  SHA-256.  The marker is durable before payload creation.
- A complete prepared payload matching the intent size/hash follows the existing
  semantic replay, rename, fsync, and orphan-adoption path.  A complete hash
  mismatch, oversize file, malformed intent, duplicate intent, wrong frame,
  symlink, hardlink, or unknown entry blocks without mutation.
- A missing payload after a valid intent is zero-byte rebuildable stage scratch.
  A size-short payload is positively identified interrupted stage scratch.  Do
  not delete its bytes: atomically move it to an identity-bound recovery
  transaction under the profile output, with a canonical manifest recording
  original path, destination, actual bytes/SHA-256, intended bytes/SHA-256,
  identity hash, exact rebuild argv, rebuildability reason, and false-authority
  flags.  Fsync both directories.  Recovery must be idempotent across every
  transaction boundary.
- Terminal custody must validate and report recovery manifests/payload hashes;
  recovery bytes never enter the ordered stage chain, rate stream, score, or
  promotion authority.

## Required regressions

- The second resume-progress read is eliminated in production and direct calls
  use `_load_canonical_object` once.  Noncanonical JSON, Unicode decode failure,
  read `OSError`, and path/link substitution block before mutation.
- Stable metadata prepared files recover after a partial write and after a
  complete-write/pre-rename cut.  Equal-length/non-prefix drift and read errors
  preserve bytes and block.  No `.tmp-<PID>` name is emitted.
- A valid prepared stage still adopts durably and removes its intent.
- A size-short intent-bound stage is quarantined byte-identically, its recovery
  manifest hashes validate, progress remains at the old pointer, and the frame
  can be deterministically recomputed.  Interruptions before manifest, before
  move, and after move are idempotently recoverable.
- Same-size corrupt prepared bytes, missing/duplicate/wrong intent, unknown
  entries, symlink, and hardlink remain untouched and fail closed.
- Feature-cache creation recovers an interrupted first metadata write and a
  later prepared-metadata interruption without promoting debris.  Final cache
  validation rejects unknown/PID debris and allows only recognized stable
  runtime scratch until the deterministic writer reconciles it.
- All existing scientific, authority, source-seed, codec parse-back, governor,
  and FIX13 I/O-taxonomy tests remain green.

## Acceptance

Run:

```text
PYTHONPATH=src pytest -q \
  src/tac/tests/test_segnet_head_feature_cache.py \
  src/tac/tests/test_uint8_lattice_profile.py \
  src/tac/tests/test_uint8_lattice_seed_anchor.py
ruff check and ruff format --check on all eight delegated Python files
python3 -m py_compile on all eight delegated Python files
git diff --check
```

No production extraction/profile launch, score, promotion, Pose, factor-10,
global-compression-minimum, or contest-axis authority is granted.  Freeze and
launch remain blocked until the replacement exact-byte bundle passes three
consecutive zero-finding reviews and MAIN landing review.
