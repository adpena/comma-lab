# V10 A2 profiler FIX18 frozen implementation specification

Date: 2026-07-18 America/Chicago  
Lane: `v10_A2_profiler_20260718`  
Status: `FROZEN_BEFORE_IMPLEMENTATION`  
Authority: delegated V10 A2 profiler task plus FIX15-FIX17 custody contracts  
Pointer authority: unchanged; source-hardening only

## Trigger and exact rejected bytes

Three independent fresh reviews rejected the exact lexicographic eight-file
bundle:

`229c6136c31d349b616f59a19874617f9c544a3910a2791dc5784708371da1ad`

The focused suite, targeted crash/custody subset, Ruff, format check,
`py_compile`, `git diff --check`, and no-Fourier scan were green.  Those checks
did not cover the three implementation-custody classes below.  The clean-pass
seal is reset.  No source-freeze commit, dispatch claim, extraction/profile
launch, score, promotion, or pointer movement is authorized from these bytes.

## Re-derived blockers

### F18.1 directory retirement and publication remain destructive

`SegnetHeadFeatureCache.create` still uses `Path.rmdir`, `shutil.rmtree`, and
`os.replace` on cache/staging directories.  A late empty destination can be
overwritten, a substituted staging tree can be recursively deleted, and a long
valid cache basename can be removed before its longer staging name is proven
portable.

### F18.2 recorded weight bytes need not be executed weight bytes

The source snapshots hash `segnet.safetensors`, but the admitted scorer callable
reopens that pathname through `safetensors.load_file`.  A pathname replacement
during the loader read can construct the model from foreign bytes and restore
the recorded path before the second snapshot.  Pre/post pathname equality does
not prove model-state equality.

### F18.3 post-exchange scratch lacks displaced-target provenance

When a committed target and active atomic scratch coexist after an exchange
cut, generic cleanup retains the scratch without proving it is the descriptor-
admitted prior target.  Arbitrary or foreign scratch can therefore be mutated
into a canonical retained name and recovery can advance.

### Verdict scope

These are implementation-level cache/scorer/recovery custody findings only.
They do not invalidate the factor-2 feasible-set mathematics, exact support
union `S`, direct frozen-logit versus rank-4 diagnostic separation, NoOp Pose
boundary, rate accounting, or the V10 representation family.

## Frozen FIX18 design

### 1. Directory operations are lossless and no-replace

- Remove every production `rmdir`, recursive delete, and replace-capable
  directory publication from cache creation/recovery.
- Final staging publication MUST use the descriptor-bound, same-parent,
  no-replace directory primitive.  A destination that exists initially or
  appears at the final boundary blocks while preserving both directory trees.
- An existing cache root is never deleted by `create`.  A valid cache is opened
  only through the explicit resume path; an empty, foreign, or malformed root
  is preserved and refused.
- An empty or certified-rebuildable interrupted staging tree may be retired
  only by a descriptor-bound no-replace move to deterministic retained
  directory custody.  Its machine-readable receipt must bind original path,
  directory identity, recursively measured bytes/tree hash (or an equally
  strong certified-tree digest), rebuild argv/config/storage preflight,
  false-authority flags, reason, and retention destination.  If that proof
  cannot be completed, preserve the original tree and block.
- Future creation validates every recognized retained directory/receipt and
  excludes it from cache, score, rate, and promotion authority.  Unknown,
  malformed, linked, drifted, or role-inconsistent retained directories block.
- Prove the actual parent filesystem component limit before the first mkdir,
  move, receipt write, or root mutation.  Validate the cache basename, staging
  basename, directory-retention basename, and every required sidecar basename.
  `ENAMETOOLONG` after a mutation is forbidden.

### 2. Model construction consumes admitted weight payload bytes

- The first frozen-source snapshot MUST return the exact descriptor-read
  `segnet.safetensors` payload and derive its recorded bytes/SHA/path row from
  that same read.  Do not hash once and load through a second pathname read.
- Construct `SegNet` from the admitted `modules.py` module and deserialize the
  admitted weight payload in memory (for example the byte-oriented
  `safetensors.torch.load` API).  The model loader may not reopen the weight
  pathname.
- Preserve canonical CPU/eval/frozen-parameter semantics and prove parity to
  the stable canonical loader by state-dict bytes and a deterministic tiny
  forward control.
- Extraction and `--score-segnet` profiling both use this byte-fed loader.
  Their identity/source binding is built from the exact payload used to create
  the model, while the complete second source snapshot still detects lasting
  path/source drift before cache/output initialization.
- A controlled loader boundary that replaces the weight pathname during model
  construction must either be irrelevant because no pathname read occurs or
  refuse.  It may never execute foreign weights while recording admitted
  bytes.  Apply the same end-to-end assertion to extraction and profiling.

### 3. Atomic scratch recovery proves prior-target provenance

- A committed target plus active scratch is not authorized by filenames alone.
  Before any scratch retention, target rewrite, progress/stage adoption,
  receipt emission, or recovery mutation, prove each scratch payload is an
  exact reachable displaced prior target for that consumer.
- The generic atomic writer MUST NOT treat arbitrary bytes, or merely
  canonical JSON, as valid displaced custody.  Use a durable pre-exchange
  transaction record/generation, a consumer-supplied exact prior-state
  validator, or an equivalent fail-closed design that binds prior bytes/size/
  SHA and role across a crash.
- Partial transaction metadata is preserved and may be superseded only by a
  new complete generation; it is never truncated or deleted.  Unknown,
  malformed, missing, contradictory, or byte-drifted transaction evidence
  blocks with the entire tree byte-identical.
- For a fresh recovery manifest, the only reachable displaced placeholder is
  the descriptor-admitted empty target.  For progress, receipts, cache
  metadata, and completion records, validate the exact prior schema/identity/
  chain transition rather than a generic JSON shape.
- Hash-valid retention naming is integrity evidence, not provenance.  Unknown
  bytes may not become authorized merely by moving them to a self-identifying
  retained name.
- Post-linearization retry remains deterministic and idempotent at every cut:
  before transaction record, during stable record generation, before
  exchange, after exchange, before scratch retention, after retention, and
  before/after transaction-record retirement.

## Required regressions

1. Through real cache creation, introduce a late empty cache destination and a
   substituted staging directory at the final directory boundary.  Refuse and
   preserve both original and foreign trees/inodes.  Exercise fresh creation,
   exact interrupted staging recovery, and retry.
2. Exercise an existing empty cache root, empty staging root, invalid certified
   staging, and a basename whose staging/retention component would exceed the
   real filesystem limit.  Every refusal is pre-mutation; every recoverable
   case leaves validated retained-directory custody and completes on retry.
3. During actual extraction and profile scorer construction, make every
   pathname-based weight open fail and separately replace/restore the weights
   pathname at the loader boundary.  The byte-fed stable control succeeds,
   executed state matches admitted payload bytes, and no output/cache/stage/
   receipt/preflight mutation precedes refusal.
4. Inject arbitrary bytes and separately valid-but-foreign canonical JSON into
   post-exchange scratch for recovery manifest, progress, receipt, cache
   progress, certification, and completion consumers.  Retry blocks before
   any rename/retention/rewrite/adoption and the complete pre/post tree matches.
5. Exercise every reachable exact displaced-target state for fresh target,
   existing target, strict prefix, complete generation, exchange cut, cleanup
   cut, and repeated retry.  The target converges, prior custody is retained,
   and no active scratch/transaction name remains.
6. All FIX17 retention, same-frame multi-attempt, typed-progress, loader-source,
   scientific, source-seed/support-union, codec, governor, storage, I/O
   taxonomy, false-authority, and exact-argv regressions remain green.

## Acceptance and launch boundary

Run the three owned test modules, the exact new directory/weight/scratch
regressions, Ruff check and format-check, bytecode compile over the exact eight
delegated Python files, `git diff --check`, the explicit no-Fourier/DCT/rFFT
scan, and a deterministic ordered component/bundle hash.  Reverify `uv.lock`
and the sacred n600 source tree against their pre-task hashes.

The replacement exact-byte bundle requires three new consecutive independent
zero-finding reviews.  Prior reviews do not carry forward.  Only after that
seal may the serializer land a source-freeze commit.  A lane claim or governed
n600 launch remains a separate later action.  MAIN landing review is mandatory.
