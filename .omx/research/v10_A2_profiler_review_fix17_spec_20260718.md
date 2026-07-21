# V10 A2 profiler FIX17 frozen implementation specification

Date: 2026-07-18 America/Chicago  
Lane: `v10_A2_profiler_20260718`  
Status: `FROZEN_BEFORE_IMPLEMENTATION`  
Authority: delegated V10 A2 profiler task plus FIX16 review contract  
Pointer authority: unchanged; this is source-hardening only

## Trigger and exact rejected bytes

FIX16 completed its owned implementation and local test pass, but an independent
post-worker adversarial review rejected the exact eight-file bundle:

`e2c007a8eff3c10fe018f1f4e49e7bf05e989762b20877324bd2a435daec8c5a`

The ordered component hashes were:

1. `src/tac/governed_profile_admission.py` — `5a4a5e2cdfac7248d3a472b96ee324174961af1120fa2da1ef5a14173067aacd`
2. `src/tac/optimization/uint8_lattice_profile.py` — `18984102df8535596a3208888583ce4db4fedda99ed70b0c57ad19a8177358e8`
3. `src/tac/tests/test_segnet_head_feature_cache.py` — `53b25b465897a6cbe492fbcd6bc4f897bb10789c09346225e0b8f0d1eee633be`
4. `src/tac/tests/test_uint8_lattice_profile.py` — `16d96058272ae557e7526794a02442b312222a03eca9baf6c538a5f90ad8ed19`
5. `src/tac/tests/test_uint8_lattice_seed_anchor.py` — `0defd6e54b14dbb83764144ad633ebdb8d3193c40b418217dca96a66999a198d`
6. `src/tac/witness_control/segnet_head_feature_cache.py` — `a770e1c69a8b9ce13185d638bfbe7969c7ed713c3cd2dbf7fb8059d2de7cdbb6`
7. `tools/extract_segnet_head_features_n600.py` — `a714bde4fc40e1c4274b9426820168e144f70794fea296afcf7612b2f30f5a5c`
8. `tools/profile_v10_uint8_lattice_n600.py` — `54b70b0726a8cea2bd02af06f19a0aacd9ceaed2074bcb1edd29e6ea46472d27`

Review-counter round 23 records three accepted findings and verdict
`NOT_CLEAN`. The seal is reset. No source-freeze commit, dispatch claim, or
n600 launch is authorized from these bytes.

## Re-derived blockers

### F17.1 cache cleanup deletes a substituted foreign inode

`_unlink_bound_descriptor` validates a path against an open descriptor, calls
`os.unlink(path)`, and only then observes the descriptor link count. A controlled
substitution between validation and unlink produced:

```text
FeatureCacheError atomic scratch cleanup unlink did not detach the authorized inode
foreign_original_exists=False
candidate_exists=False
displaced_bytes='{"x":1}\n'
```

The error is not protective: foreign bytes are already deleted.

### F17.2 profiler intent cleanup has the same destructive interval

`_unlink_bound` validates a `BoundFileSnapshot` and then calls `Path.unlink()`
without a kernel identity predicate. The same substitution deletes the foreign
replacement and returns successfully. This path is used for stage intents and
partial/prepared creation cleanup.

### F17.3 profiler move overwrites a destination that appears late

`_replace_bound` validates only the source and calls `os.replace`. A foreign
destination appearing after an earlier absence check is overwritten. The
minimal reproduction leaves the destination containing the authorized source
and the prior foreign destination bytes lost.

### Verdict scope

These are implementation-custody negatives only. They do not invalidate the
factor-2 feasible-set mathematics, exact support-union `S`, direct-logit cache
contract, quotient diagnostic, or V10 representation family. FIX8's settled
support-union plus deterministic-fill receiver contract remains closed and is
not reopened.

## Frozen FIX17 design

### 1. No destructive pathname cleanup

Within the cache/extractor/profiler trust boundary, an authorized inode MUST NOT
be retired with `os.unlink`, `Path.unlink`, or an equivalent name-only unlink.
Retirement is lossless: move the exact descriptor-bound inode to a deterministic
retention sibling. Retained metadata is small historical custody, not score or
promotion authority.

### 2. Deterministic self-identifying retention names

For a source `P` and its descriptor-read payload `B`, the retention name MUST
encode all of:

- the original basename;
- `len(B)` as a fixed-width decimal field;
- `sha256(B)`;
- an eight-digit monotonically selected local ordinal.

The chosen name is the first absent canonical ordinal. Every reader/validator
that admits a retention name MUST reopen it no-follow, require one regular-file
link, and rederive its encoded byte count and SHA-256. Retention artifacts are
excluded from live scratch selection, stage-prefix counting, score authority,
and terminal result rows.

### 3. Descriptor-relative no-replace move

All source-to-destination moves use a platform syscall with no-replace
semantics relative to an already-open parent directory:

- Darwin: `renameatx_np(..., RENAME_EXCL)`;
- Linux: `renameat2(..., RENAME_NOREPLACE)`;
- unsupported hosts: fail closed before mutation.

The source is open and payload-verified before the syscall. Immediately after
the syscall, the destination pathname MUST name the same opened inode and its
descriptor bytes MUST still equal the admitted payload. A mismatch triggers a
no-replace rollback to the original name. Rollback uncertainty raises while
preserving every observed pathname; it MUST NOT delete or overwrite either
side.

The parent directory descriptor itself MUST be opened no-follow and proven to
name the same parent used for source and destination. Source and destination
for this primitive MUST share that exact parent.

### 4. Atomic metadata cleanup becomes retention

After a successful metadata exchange:

- the displaced prior target or empty placeholder at the scratch name is moved
  to a self-identifying retention sibling;
- active `.atomic-prepared` / `.atomic-generation-*` names are absent;
- retained siblings remain immutable and are ignored by future source
  selection;
- a post-linearization crash remains retryable and converges to that state.

Placeholder rollback after a failed pre-exchange attempt uses the same
retention primitive. It MUST NOT swallow an uncertain rollback/retirement
failure.

### 5. Profiler stage moves never overwrite

`_replace_bound` is replaced by the descriptor-relative no-replace primitive.
If the destination appears, the operation refuses and preserves source and
destination bytes. If the source pathname is substituted at the final boundary,
post-move identity verification and rollback preserve the authorized bytes and
the foreign bytes.

Stage-intent and prepared-creation cleanup uses retention rather than unlink.
Resume parsing MUST recognize and validate retained entries while excluding
them from the active intent/prepared/final state machine.

### 6. Reachable recovery grammar

The recovery grammar MUST accept every real cut point introduced by FIX16 and
FIX17, including:

- final manifest plus displaced retained placeholder after manifest exchange;
- complete manifest and payload plus retained metadata/intent custody;
- progress target committed plus retained prior progress;
- prepared or intent bytes moved to retention after their semantic role ends.

It MUST continue to reject unknown names, malformed retention names, hash/size
drift, payload without complete manifest, prepared manifest with payload,
manifest-only without exact live intent, multiple active intents, holes, and
nonterminal states that can advance progress without complete custody.

### 7. Required controlled-boundary tests

Tests MUST exercise actual production orchestration, not fabricated before/after
dictionaries:

1. cache cleanup substitution at the final move boundary preserves authorized,
   foreign, old-target, and target bytes;
2. profiler intent retirement substitution preserves both inodes and does not
   advance progress;
3. late foreign stage destination is not overwritten;
4. stage-source substitution rolls back without granting foreign authority;
5. retention filename byte/hash tampering blocks before replay or mutation;
6. post-linearization manifest and progress retries end with no active scratch
   and validated retained custody;
7. fresh and resume success paths remain deterministic;
8. all FIX16 controlled-loader source-role tests stay green.

Every refusal test records the complete pre/post file tree and proves no
unrelated pathname disappeared or changed.

## Acceptance and launch gate

Before a source-freeze commit:

- all three owned test modules pass;
- Ruff check and format-check pass on the exact eight-file surface;
- pycompile and `git diff --check` pass;
- the no-Fourier/DCT/rFFT scan is empty;
- `uv.lock` is byte-identical to its pre-task hash;
- the sacred n600 source tree rehashes to its pre-task hash;
- three new independent exact-byte reviews return `CLEAN` with zero findings;
- review-counter and review-tracker gates agree on those exact bytes.

Only after the serializer lands that reviewed source-freeze commit may the lane
be claimed and the governed resumable n600 extraction/profile run begin. MAIN
must perform a landing review. The frontier pointer remains unchanged.

