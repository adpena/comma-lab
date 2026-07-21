# v10 A2 profiler review fix 3 — source-freeze gate

verdict_scope: formulation — the NO-GO tokens herein grade the SPECIFIC profiler review-fix formulation this spec addresses (v10_A2 profiler fix round), not the profiler family or any measurement family.

Date: 2026-07-18  
Lane: `v10_A2_profiler_20260718`  
Authority: `codex_delegate:v10_A2_profiler_20260718:20260718T160947Z`  
Status: **PRE-LAUNCH / NO-GO UNTIL CLEAN RE-REVIEW**

## Why this pass exists

Two independent post-fix2 reviews found no remaining P0 normal pointer/stage
window, but the operations review identified two P1/High defects that would
make the requested per-class/stratum receipts or certified cache recovery
untrustworthy.  Source bytes must not be frozen or executed until both are
closed and the seal is rebuilt from zero.

## Required corrections

### 1. Validate the entire aggregate state, not only `global`

`tools/profile_v10_uint8_lattice_n600.py::_validate_stage_receipt` must reject
freshly rehashed internal tampering in `per_class` and `strata`.

Required invariants:

- reconstruct only the exact typed aggregate schema; catch
  `LatticeProfileError` as a fail-closed `ProfilerError`;
- every bucket has nonnegative integral scorer/channel/exact/bounded counts,
  `exact + bounded == channel`, and internally coherent compact lower/upper
  stats;
- compact stats have the canonical 0.25/129-bin geometry, nonnegative integral
  counts/bins, `sum(bins) == count - zero_count`, finite totals/min/max, and
  consistent empty/nonempty extrema;
- the per-class buckets form an exact partition of global counts and histogram
  counts; their finite totals/extrema agree with global up to only deterministic
  floating summation tolerance;
- every named stratum is internally coherent and componentwise a subset of the
  global bucket (strata may overlap and therefore do not sum globally);
- the canonical class and named-stratum key sets are exact for this n600 tool.

Add adversarial regression tests for independently tampered class and stratum
buckets and for malformed aggregate reconstruction.

### 2. Recover only identity-certified cache creation scratch

`src/tac/witness_control/segnet_head_feature_cache.py` must wrap bounded memmap
parse failures (`OSError`/`ValueError`/`EOFError`) as `FeatureCacheError`, so
the existing creation recovery path may rebuild only after the staging scratch
identity and certification have already validated.  A truncated certified
`.npy` must be removed/recreated; unidentified or identity-drifted staging must
remain blocked.

Final-cache validation must also validate `certification.json`.  Tighten the
progress-state cross-invariants (`partial`, `ready_for_completion_validation`,
`complete`) and reject nonintegral completion-control frame indices without
coercion.  Add a synthetic interrupted-final-rename + truncated-array recovery
test and certification/progress tamper tests.

### 3. Preserve a fully written pre-rename profiler stage across SIGKILL

Replace the PID-named stage scratch with one deterministic prepared-stage name
per final frame.  Resume may reconcile it only when all of the following are
true: it is the sole prepared entry, the final prefix is contiguous, its frame
is exactly the next frame, its canonical receipt/payload validates, and its
identity/previous-stage link matches the validated chain.  Then atomically
rename it to the final stage, fsync the stage directory, and let normal orphan
adoption advance the pointer.  Unknown, malformed, duplicate, or conflicting
prepared bytes remain fail-closed and preserved.

Add a synthetic prepared-stage adoption test plus malformed/conflicting cases.

## Stop rule

Run focused pytest, Ruff, format check, `py_compile`, and `git diff --check`.
Then perform three fresh clean review passes.  Any Critical/High finding resets
the seal.  No real n600 extraction/profile launch and no source-freeze commit
before all checks pass.

