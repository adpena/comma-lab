# V10 A2 profiler review FIX16 specification — 2026-07-18

`research_only=true`  
`authority_axis=[macOS-CPU advisory]`  
`score_claim=false`  
`promotion_eligible=false`  
`pointer_delta=0`

## Trigger and verdict scope

Three fresh reviews rejected exact lexicographic eight-file bundle
`e19c776938da7a2919916fc16afa8e6d9b06d9c20f7b4dee5dff078daef83f8e`.
Source freeze, lane claim, and every real extraction/profile launch remain
blocked.

The accepted findings are limited to:

1. prepared metadata can be substituted after validation but before a
   pathname rename and foreign bytes can become the target;
2. a prepared-prefix inode can be displaced at the rewrite boundary and then
   mutated before the late path-identity refusal;
3. `source_file_row` follows a final-component symlink before its nominal
   no-follow open;
4. the extractor imports factorization code before the first source snapshot,
   allowing recorded stable bytes to differ from executed in-memory code;
5. the required controlled-loader extraction/profiling regressions exercise
   equality helpers only, not the real load-to-output ordering; and
6. nonterminal recovery admits a manifest-only transaction with neither
   payload nor matching intent, after which orphan adoption can advance
   progress before terminal validation rejects it.

One review observation is **DISMISSED / ALREADY SETTLED** and MUST NOT be
implemented: full camera-frame source reconstruction.  The later FIX8
contract explicitly supersedes the earlier source-anchor wording with exact
scorer-effective support-union `S` reconstruction plus declared deterministic
zero fill outside `S`.  Those outside-support bytes are not Seg receiver/rate
authority.  The factor-2 affine bounds, receiver closure on `S`, direct-logit
versus quotient separation, NoOp Pose boundary, rate accounting, and false-
authority labels remain clean and must not be reopened by FIX16.

## Required implementation

### Descriptor-derived metadata commit; no check-then-rename authority gap

- A trusted metadata commit MUST remain bound to the exact validated prepared
  inode through its linearization point.  Closing the descriptor and then
  performing an ordinary path-only replace is forbidden.
- Use either a descriptor-derived commit primitive or an atomic exchange plus
  validation/rollback transaction.  If the host cannot provide the required
  primitive, fail closed before changing the metadata target.  A plain
  post-rename hash check that leaves foreign bytes at the target is not
  sufficient.
- Preserve and revalidate any preexisting target.  On any source substitution,
  identity mismatch, exchange mismatch, or rollback uncertainty, the original
  target and all positively identified source/foreign bytes must remain
  non-authoritative and byte-preserved; do not report success.
- Keep all file descriptors open until the commit result is identity-checked,
  fsync the committed file and containing directory, and classify read,
  identity, commit, rollback, and fsync failures separately enough that no
  error is treated as rebuildable partial JSON.
- Apply the same primitive through `atomic_json` to cache metadata, extractor
  sibling receipts, profiler progress/receipt metadata, and recovery
  manifests.  Do not create a second weaker writer.

### Prepared-prefix recovery must not mutate a displaced inode

- A strict-prefix prepared file is recoverable only while its path still names
  the exact descriptor-bound inode.
- Provide an explicit pre-mutation authorization boundary, then revalidate the
  path-bound descriptor immediately before the first modifying syscall.  The
  mutation/commit sequence after that boundary must not dispatch through a
  monkeypatchable or user callback that can substitute the pathname.
- Prefer completing the verified prefix without destructive truncation, or
  write a separately certified replacement generation.  Do not truncate a
  previously read inode and discover displacement only afterward.
- If the pathname changes at the authorization boundary, preserve the
  original prefix inode, the foreign inode, and the metadata target byte-for-
  byte and block.

### Final-component no-follow source custody

- `source_file_row` must normalize the absolute parent path without resolving
  the final component.  Reject a final symlink, hardlink, non-regular file, or
  parent-chain symlink before hashing.
- Hash through one no-follow descriptor; require pre-open path identity,
  descriptor identity, and post-read pathname identity to agree.  Record the
  exact non-symlink pathname actually opened.
- Add final-component symlink, parent symlink, hardlink, pathname swap, and
  content-change regressions.  Every refusal preserves all bytes.

### Executed factorization/scorer bytes must be inside the snapshot window

- Remove eager import of the named factorization module and its callables from
  every import path used by extraction before `source_snapshot_before`.
  In particular, the feature-cache module must not indirectly import it before
  that snapshot.
- Import/load the factorization code, upstream modules/frame utilities, TAC
  scorer, weights, and SegNet model only after the complete first snapshot.
  Take the complete second snapshot after factorization/model construction and
  exact-module-path checks, before cache create/resume, storage initialization,
  or any frame/stage/receipt mutation.
- Production callables must be obtained from the module admitted inside that
  window; do not retain pre-snapshot aliases.
- Profiling with `--score-segnet` follows the same pre/post rule and builds its
  feature binding and immutable profile identity only from the agreed rows,
  before output initialization or resume mutation.

### End-to-end controlled-loader regressions

- Parameterize every named extraction source role and every named profiling
  scorer source role.  A controlled loader hook changes exactly one row/file
  between snapshots while the actual extraction/profile orchestration runs.
- Assert refusal identifies the changed role and occurs before storage
  preflight that can create output, cache create/resume, stage write, progress
  replacement, or receipt emission.  Assert the output/cache entry set and all
  sentinel bytes remain unchanged.
- Stable controls must traverse the same orchestration boundary successfully
  in tiny local-test scope.  Helper-only before/after dictionary tests may
  remain, but do not satisfy this requirement.
- Add a factorization pre-import regression: change its on-disk bytes after
  tool-module import would formerly have happened but before `run_extraction`;
  the new lazy path must either execute and record the same admitted bytes or
  refuse.  It may never execute old code while recording new stable bytes.

### Reachable recovery-state grammar only

- Encode the nonterminal recovery transaction as an explicit state matrix.
  Admit only crash states reachable in order from:
  durable intent -> transaction directories -> stable manifest write ->
  payload move/create -> intent unlink.
- Before a payload exists, the exact matching intent MUST still exist.  After
  the intent is absent, both final manifest and payload MUST exist and agree.
  A payload without a complete manifest, or a complete manifest without both
  payload and intent, is unreachable and must block.
- Empty recovery root/identity/transaction directories are recoverable only
  when the exact matching live intent proves the interrupted operation.  An
  empty orphan recovery structure without that intent blocks.
- Terminal grammar admits only complete manifest+payload transactions with no
  surviving matching intent.  Every attempt remains ordered, contiguous,
  false-authority, and excluded from stage/rate/score bytes.
- Validate this whole matrix before semantic replay, orphan adoption, prepared
  stage reconciliation, progress/receipt replacement, or any recovery
  mutation.

## Required regressions

- Substitute the prepared pathname immediately before the commit syscall.
  The operation blocks; foreign bytes never become the metadata target; the
  old target and both source inodes are preserved.
- Substitute a strict-prefix pathname at the last authorization boundary
  before mutation.  Original prefix, foreign file, and target are unchanged.
- Exercise fresh target, existing target, strict-prefix recovery, complete
  prepared recovery, pre-commit crash, post-linearization/pre-cleanup crash,
  and retry for every metadata consumer.
- Exercise all source symlink/hardlink/swap cases and every actual loader role.
- Reproduce the FIX15 manifest-only/no-payload/no-intent case with a valid
  orphan final stage.  Nonterminal validation must refuse and progress must
  remain byte-identical.  Add the other impossible permutations from the
  state matrix and every reachable first/second-attempt crash state.
- All prior scientific, affine-bound, source-seed/support-union, codec,
  governor, storage, exact-argv, false-authority, and I/O-taxonomy regressions
  remain green.  `uv.lock` and the sacred witness directory remain unchanged.

## Acceptance and launch boundary

Run the three delegated test modules, Ruff check/format check and bytecode
compile over all eight delegated Python files, `git diff --check`, explicit
no-Fourier scan, targeted race/state-machine reproductions, and a deterministic
exact-component/bundle hash.

The replacement exact-byte bundle requires three new consecutive zero-finding
reviews.  Prior reviews do not carry forward.  No source-freeze commit, lane
claim, production extraction/profile launch, score, promotion, Pose/factor-10
claim, or pointer movement is authorized until that seal.  MAIN landing review
remains mandatory.
