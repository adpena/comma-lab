# v10 A2 source-witness RD anchor spec

Date: 2026-07-18  
Lane: `v10_A2_profiler_20260718`  
Authority: `codex_delegate:v10_A2_profiler_20260718:20260718T160947Z`  
Status: **DESIGN FROZEN / IMPLEMENT AFTER FIX5 SEMANTIC REPLAY**

## Purpose

The source uint8 block is already an exact witness of each factor-2 integer
resize equation.  Use that immutable witness to make the first real Seg-side
RD anchors tractable: every profiled block begins with one certified feasible
candidate, while deterministic bounded search may replace it only with a
cheaper feasible candidate.  This is a bounded cheapest-seen anchor, not an
exact minimum-description certificate.

## Solver contract

- `profile_integer_block` accepts an optional typed uint8 `seed_candidate`.
  Arity, range, and the exact integer equation are checked before search.
- The seed is evaluated through the same pose plug-in and description-cost
  surfaces as DFS leaves.  Pose rejection is not feasibility; a pose plug-in
  exception remains `PLUGIN_ERROR_UNKNOWN`.
- An accepted seed contributes one certified lower-bound member and initial
  selection.  If DFS later visits it, it is not counted or evaluated twice.
  A bounded upper count must subtract the already resolved seed while it is
  still represented in the unresolved search frontier.
- Exhaustive seeded and unseeded runs have identical exact cardinality and
  identical global min-description selection.
- Any reuse key includes the complete seed.  Real source-seed mode may bypass
  reuse to avoid retaining hundreds of thousands of near-unique blocks.

## Receiver closure

Add a strict inverse for the signed-residual candidate stream, including ULEB
termination, zigzag inverse, arity/count/header checks, exact trailing-byte
refusal, predictor reconstruction, and uint8 range checks.  The profiler must
parse back every stage row stream, reassemble the selected camera frame in the
canonical disjoint-support order, and compare it byte-for-byte with the
in-memory selected frame before frozen-SegNet scoring.  A scored RD row is
forbidden without this receiver closure.

## Real-profile mode

- Add identity-bound `--seed-source-witness`, valid only for
  `enumerated_subset` with deterministic node caps and no wall-clock cap.
- Extract the seed from the real source frame in the exact row-support,
  column-support, channel, and flattening order used by the integer equation.
- Node-cap 1 must preserve the source frame byte-for-byte after parse-back.
  Larger node caps may select cheaper exact-resize candidates; their realized
  frozen-SegNet mismatch is measured rather than assumed.
- Persist the selection label
  `KNOWN_SOURCE_WITNESS_SEEDED_CHEAPEST_SEEN_NON_GLOBAL` whenever any block is
  bounded.  Keep exact-count and global-min-description claims false.
- Pose bank remains unwired and factor 10 remains unsolved.

## Required tests and measurement

Test seed validation, double-count prevention, exhaustive parity, pose
reject/error behavior, cache-key separation, strict stream parse-back, and
camera-frame reassembly.  After the full feature cache is complete, run two
governed one-frame real anchors (node cap 1 and one larger bounded cap), both
through strict parse-back and the real frozen CPU-Torch SegNet.  Report actual
raw/zlib/Brotli bytes, wall time, selected/exact/bounded blocks, mismatches,
`d_seg`, axis, and explicit one-frame scope.  No subset row is promoted or
extrapolated to n600.
