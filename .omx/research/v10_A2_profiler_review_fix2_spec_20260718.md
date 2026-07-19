# V10 A2 profiler second review-fix specification — 2026-07-18

`research_only=true`  
`verdict_scope=factor-2 uint8 affine feasible-set profiler implementation`  
`score_authority=false`  
`promotion_eligible=false`

## Why this correction is owed before n600

The corrected bounds path now counts all real n600 channel equations honestly,
but its certified lower bound is exactly one for every block.  That satisfies a
minimal existence certificate while leaving the requested K-handle nearly
content-free.  Fresh independent review also found two hardening gaps: uint64
inputs can wrap before the current int64 guard, and hash-chained stage receipts
are not checked for internal counter/aggregate/scope consistency.

This pass changes only the existing six A2 implementation/test files.  It does
not launch extraction, score, dispatch, mutate the sacred result tree, update a
frontier pointer, or claim factor 10.

## 1. Certified source-fiber lower bound

For a verified source witness `x` and a coefficient pair `(i,j)`, let

`g = gcd(c_i,c_j)`, `d_i = c_j/g`, and `d_j = c_i/g`.

Every integer `n` in the interval below yields another exact pair solution:

`u_i = x_i + n*d_i`, `u_j = x_j - n*d_j`.

Use exact integer floor/ceil arithmetic:

`lo = max(ceil(-x_i/d_i), ceil((x_j-255)/d_j))`  
`hi = min(floor((255-x_i)/d_i), floor(x_j/d_j))`  
`pair_count = max(0, hi-lo+1)`.

For four taps, the two moves in each perfect matching have disjoint support and
combine independently and injectively.  Therefore each matching product is a
certified feasible-set lower bound.  Emit the maximum of:

- `(0,1) | (2,3)`;
- `(0,2) | (1,3)`;
- `(0,3) | (1,2)`.

For three taps, use the maximum single-pair count.  For two taps, use the one
pair count.  For one tap, retain one.  This is a lower bound, never an exact
cardinality claim.  Persist an explicit method label such as
`MAX_DISJOINT_PAIR_NULL_FIBER_PRODUCT` in the returned bounds/receipts.

## 2. Integer safety before cast

Before any cast to int64:

- reject unsigned/signed values outside the int64 domain;
- reject source witnesses outside `[0,255]`;
- reject nonpositive coefficients;
- for arity `k`, reject any coefficient above
  `int64_max // (255*k)`, which makes coefficient products, their sum, and
  `255*sum(coefficients)` safe in int64;
- retain the post-cast guards and exact source-equation equality.

No wraparound may occur before validation.

## 3. Hash-chained stage receipt invariants

Add one strict validator used both before writing a stage and while rebuilding
resume state.  It must fail closed unless:

- mode is one of the two typed modes;
- global aggregate channel blocks equal `counters.total_blocks` and
  `scope.rgb_channel_blocks`;
- global aggregate scorer pixels equal `scope.scorer_pixels`;
- exact plus bounded aggregate blocks equal total blocks;
- counters are nonnegative integer (not bool);
- enumeration satisfies
  `selected = exhaustive_selected + bounded_selected` and
  `selected + omitted = total`;
- bounds mode has zero selections, all blocks bounded, all blocks omitted from
  a candidate stream, and an empty candidate payload;
- scorer mismatch/pixel arithmetic is valid;
- candidate payload byte count/hash agrees with the parsed bytes.

The mutable progress file remains only a pointer/index.

## 4. Narrow cache hardening

`commit_frame` must reject bool, float, string, and other coercive frame indices;
only Python/NumPy integers are admissible before contiguous-prefix comparison.

Cache creation must also be crash-recoverable.  Construct the marker,
manifest, memmaps, initial progress, and certification in a fixed same-filesystem
staging directory carrying a machine-readable rebuildable-scratch record.  Only
after validation and directory fsync may the staging directory be atomically
renamed to the final cache root.  A later fresh create must be able to re-enter
or safely rebuild that certified staging directory; it must never strand an
unidentified multi-GiB partial cache.

## 5. Profiler P0 launch/resume closure

The profiler must have an atomic identity-rooted `next_frame=0` progress pointer
before frame 0 starts.  Order initialization so a crash before the pointer leaves
at most an empty output directory, while a crash after it is admitted by
`--resume`.

Close the normal stage/pointer crash window losslessly:

1. fsync the stage;
2. if interrupted before pointer advance, resume validates the contiguous stage
   as the next identity/previous-hash-linked receipt;
3. rebuild aggregate/counters from it;
4. atomically adopt it by advancing the pointer.

Never delete a valid orphan stage.  The pointer's old chain head must match the
stage at `next_frame-1`; every additional contiguous stage must extend that
chain.  Reject holes, wrong names, chain drift, or a recovered prefix longer
than the new `--max-frames` request.

Bind every executed math module: exact-path check and hash both
`uint8_lattice_feasibility.py` and `uint8_lattice_profile.py` in profiler
identity, in addition to the tool itself.  This identity is frozen and committed
before any real extraction/profile launch.

## 6. Required tests

- Four-tap brute-force-small-target canary proving the pair-fiber lower is
  greater than or equal to one and no greater than exact truth.
- Canonical ordering canary using `2x2 -> 1x1`, three channels, exercising all
  four taps from `_source_block_geometry`.
- Pair-fiber source and every emitted lower candidate satisfy the equation by
  construction.
- uint64 overflow/coercion is refused before cast.
- Internally inconsistent but freshly hash-chained stage receipts are refused:
  counter/aggregate mismatch and mode-specific arithmetic mismatch.
- A frame-0 interruption is resumable from the initial pointer.
- A hash-valid stage written ahead of the pointer is adopted, not deleted; a
  malformed orphan is refused.
- Profiler identity includes exact path/bytes/SHA-256 for both executed
  optimization modules.
- Cache initialization uses an atomic, certified staging-to-final transition and
  a synthetic interrupted staging directory is safely recoverable/rebuildable.
- Non-integral `commit_frame(0.0, ...)` is refused.
- Existing focused tests, Ruff, format, `py_compile`, and `git diff --check`
  remain green.

## Stop rule

No full feature extraction or all-n600 profiler run until an independent
three-pass review finds no Critical/High authority blocker in this correction.
