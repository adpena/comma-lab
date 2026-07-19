# V10 A2 profiler review-fix spec — 2026-07-18

## Scope and evidence

This is the corrective implementation contract for the first A2 patch. Keep
the existing factor-2 solver untouched and edit only the six new code/test
files already owned by this lane.

Measured real-cache timing on 1,000 frame-0 channel blocks showed:

- node cap 1: 16,713 blocks/s, projected 5.88 h for 353,894,400 blocks;
- node cap 16: 3,578 blocks/s, projected 27.48 h;
- node cap 64: 1,186 blocks/s, projected 82.87 h;
- node cap 256: 528 blocks/s, projected 186.03 h.

Frame 0 has 587,773 unique equations among 589,824 channel blocks
(99.6523%), so the current 200k LRU does not make the all-n600 nested Python
loop feasible. The exact enumerator remains valuable for a small honest subset,
but it cannot be the full traversal engine.

## P0 authority fixes

1. For `--max-frames 600`, require a feature cache with status `complete` and
   the stored positive-control schema. A 600-slice
   `ready_for_completion_validation` cache must be refused. Partial prefix
   profiling may consume only a hash-valid explicit prefix and must label it.
2. Cross-bind profiler `--gt-cache` to the feature manifest's
   `source_files.gt_n600_npz` row by resolved path, bytes, and SHA-256. Refuse
   any mismatch.
3. Exact-source custody:
   - extractor must require imported `modules` and `frame_utils` paths to equal
     the exact files hashed in the manifest, not merely lie below the root;
   - include executed `tac.scorer`, factorization module, Python/Torch/NumPy
     versions, platform, and deterministic-thread configuration in identity;
   - when `--score-segnet` is used, profiler identity must bind exact upstream
     modules, frame-utils, weights, and scorer loader, and verify imported paths.
4. Make cache validation read-only by default. Provide an explicit writable
   validation/resume path used only by `SegnetHeadFeatureCache` writers.
5. Enforce canonical n600 camera geometry `(600,874,1164,3)` in the extractor.
6. Replace whole-cache `np.isfinite(...).all()` checks with frame/chunk bounded
   checks so completion cannot allocate >500 MB temporary boolean arrays.

## Resume/custody fixes

1. Do not restore unauthenticated aggregate/scorer counters from mutable
   progress JSON. At minimum, write one canonical per-frame receipt into the
   hashed stage payload, hash-chain stages from the immutable identity, and
   reconstruct cumulative aggregate/counters by parsing the preserved stage
   receipts on resume. Progress is a pointer/index only. Reject any stage-chain,
   frame-order, or progress-pointer mismatch. Do not merely trust the old
   `aggregate_state`, `selected_blocks`, `total_blocks`, `segnet_mismatches`, or
   `segnet_pixels` fields.
2. Fsync the stage directory after atomic replace.
3. Bind profile output identity to the feature cache manifest/progress identity
   and committed prefix used.

## Full-n600 feasible-set path

Add a default or explicit `bounds_only_source_witness` mode that is capable of
finishing all 600 frames:

- This mode is valid only for the no-op pose plug-in and targets derived by
  `DisjointResizeOperator.apply_numerators` from the same cross-bound GT frame.
- The original real uint8 2x2 channel block is an exact witness, so every block
  has a rigorously derived lower cardinality bound of at least 1. Verify the
  exact integer equation vectorially; a mismatch is fatal.
- Compute a sound finite root-state upper bound vectorially for every channel
  block, including range and gcd pruning. Do not call Python DFS per block.
- Aggregate lower/upper `log2(count)` histograms, exact/bounded fractions,
  classes, boundary/fragile/degenerate strata in NumPy batches. Every one of
  the 353,894,400 n600 channel blocks must contribute exactly once.
- This mode emits `DERIVED_BOUNDS_FROM_REAL_N600_SOURCE_WITNESS`, never an exact
  count unless a separate exhaustive certificate proves it. It emits no
  min-description or d_seg claim and no candidate stream.
- Preserve one frame-stage receipt and resume chain per frame, with measured
  wall-clock/RSS/timing counters.

Keep the general `profile_integer_block` exact-or-bounded enumerator and the
candidate-stream path as an explicit `enumerated_subset` mode. Its receipt must
state:

- exact frame/block scope;
- globally-exhaustive selection count versus cheapest-seen bounded selection;
- `MIN_DESCRIPTION_EXACT` only for exhaustively traversed intersections;
- `CHEAPEST_SEEN_NON_GLOBAL` for bounded candidates;
- scorer coverage frames/pixels separately from rate-stream coverage;
- omitted/fallback blocks and receiver non-closure. Never merge a partial
  scorer denominator with a wider byte denominator.

The parent will run multiple subset node caps for an honest exploratory RD
curve. The tool does not need to run all caps in one invocation, but each row
must carry its cap and exactness-insistence rule.

## Tests

Add focused regressions for:

1. full-600 request refuses `ready_for_completion_validation` cache;
2. GT/feature source mismatch refusal;
3. exact imported-path custody;
4. read-only validation memmaps and writable writer resume;
5. bounded-RSS completion scan;
6. resume rebuilds from stage receipts and refuses tampered progress/stage
   chain;
7. vectorized bounds match/enclose brute-force truth on small synthetic
   geometry and count each pixel/channel once;
8. a cost model where the strictly cheaper description winner (not a mere
   tie-break) differs from minimum norm;
9. malformed/bool candidate stream values and malformed/nonfinite pose results
   fail closed;
10. scored rows refuse mixed scorer/rate scope.

No heavy run, no commit, no docs/matrix/DAG edits in this corrective dispatch.
