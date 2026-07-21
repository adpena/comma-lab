# v10 A2 profiler review fix 4 — semantic-custody seal

verdict_scope: formulation — the NO-GO tokens herein grade the SPECIFIC profiler review-fix formulation this spec addresses (v10_A2 profiler fix round), not the profiler family or any measurement family.

Date: 2026-07-18  
Lane: `v10_A2_profiler_20260718`  
Authority: `codex_delegate:v10_A2_profiler_20260718:20260718T160947Z`  
Status: **PRE-LAUNCH / NO-GO UNTIL CLEAN RE-REVIEW**

## Why the fix3 seal reset

Both independent fix3 passes returned `NO-GO`.  The new validators reject the
original independent class/stratum edits and recover certified crash scratch,
but two freshly rehashed false receipts still survive: coordinated histogram
redistribution can falsify K-handle quantiles, and a self-consistent class-key
swap or stratum inflation is not bound to the real frozen-cache labels/masks.
No source freeze or n600 execution is permitted until both High findings close.

## Required corrections

### 1. Bind compact histograms to extrema and totals

For every nonempty `_CompactStats` state:

- find the first and last occupied canonical 0.25-wide bins;
- require `minimum` to inhabit the first occupied bin and `maximum` the last;
- use the factor-2 four-uint8 ceiling `log2(K) <= 32` for the clipped final bin;
- bound `total` by the count-weighted occupied-bin intervals, tightened by the
  exact stored minimum and maximum (handling one observation and same-bin
  min/max without double-counting);
- retain only the narrow deterministic float summation tolerance already used
  for class/global merges.

Add adversarial tests that move bins in global plus the owning class, and in
global plus class plus stratum, while keeping counts/totals/extrema unchanged.
Both must fail before reconstruction.

### 2. Bind class and stratum buckets to real immutable inputs

Each stage must carry exact `partition_custody` for its frame: canonical schema,
frame, uint8 class-label geometry/hash/counts, and packed-boolean geometry/hash/
counts for `boundary_annulus`, `fragile`, and `degenerate`.  The expected custody
must be re-derived from the source-bound live-logit cache and real source frame
on every production resume; it may not be trusted from the stage being checked.

`_validate_stage_receipt` must require an independently supplied expected
custody object, compare it exactly, then require every per-class scorer-pixel
count and every named-stratum scorer-pixel count to equal the independently
derived counts.  `_resume_from_stage_chain` must require a provider for this
custody and call it for every final or prepared stage.  The normal writer must
use the same derivation helper.  Bump the pre-execution stage schema because no
real A2 stage has yet been frozen.

Regression tests must reject a self-consistent class-key swap and a
self-consistent global-as-fragile substitution even after payload/chain hashes
are refreshed.  Resume tests must use a provider independent of stage bytes.

### 3. Finish cache and pointer schema hardening

- Validate the exact production storage-preflight schema in both scratch and
  certification: `PASS is True`; typed nonnegative free/required byte counts;
  `required <= free`; absolute selected/anchor paths; nonempty absolute
  waterfall roots; boolean local-test flag; and selected-root membership when
  local-test mode is false.  Scratch and certification remain exactly equal.
- Every committed-frame row has an exact key set, a non-boolean Python integer
  frame index, lowercase 64-hex slice hashes, and a finite JSON diagnostics
  mapping.  `0.0` and `false` must fail even for frame zero.
- `completion_positive_control` must be `null` in both non-complete cache
  states.
- Profiler progress uses the exact pointer schema and enforces `complete` iff
  the canonical 600-frame prefix exists; a partial prefix cannot claim complete.

These are fail-closed custody checks only.  They do not add score authority,
factor-10/Pose authority, execution authority, or permission to touch the
sacred c2 run or frontier pointer.

## Stop rule

Run focused pytest, fresh tamper probes, Ruff, format check, `py_compile`, and
`git diff --check`.  Then restart all three review passes from zero.  Any
Critical/High finding resets the seal again.  No real n600 extraction/profile
launch and no serializer source-freeze commit before all checks pass.
