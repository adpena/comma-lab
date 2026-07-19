# v10 A2 profiler review fix 5 — independent semantic replay

Date: 2026-07-18  
Lane: `v10_A2_profiler_20260718`  
Authority: `codex_delegate:v10_A2_profiler_20260718:20260718T160947Z`  
Status: **PRE-LAUNCH / NO-GO UNTIL CLEAN RE-REVIEW**

## Why the fix4 seal reset

Two fresh review passes and the parent probe reproduced a false receipt that
fix4 still accepts.  Two real class masks can have the same scorer-pixel count
but distinct lower/upper K distributions.  Swapping their complete aggregate
buckets while leaving the independently re-derived label custody unchanged
preserves every count, global partition, compact-stat invariant, payload hash,
stage hash, and progress hash.  The validator accepts the wrong ownership of
the class-conditioned K statistics.  Equal-count named strata have the same
failure class.

## Required correction

Receipt validation must independently replay the semantic aggregate for the
stage from immutable production inputs and the identity-bound profiler
configuration.  A hash or digest computed from stage-provided bucket bytes is
not independent and is insufficient.

- Factor the per-frame bounds/enumeration calculation into one deterministic
  helper used by both the normal writer and replay validation.  The helper must
  take the source frame, live logits, exact resize operator, typed mode, node
  cap, selector identity, pose-plugin identity, and other result-affecting
  configuration explicitly.
- On every final or prepared stage resume, invoke that helper from the frozen
  source frame and source-bound live logits.  Compare the replayed per-class
  and named-stratum aggregate states exactly with the stored stage before
  merging it.  Global state remains covered by the existing exact partition
  equations; comparing the full aggregate state is preferred.
- Enumeration replay must not trust stored candidates or stored aggregate
  values.  If an enabled wall-clock cap prevents deterministic replay, fail
  closed before launch; production profiling uses the deterministic node cap.
- Do not recompute the same frame twice on the writer path: pass the freshly
  derived aggregate as the expected semantic state to the stage validator.
- Retain partition custody and all fix4 histogram/schema checks as independent
  defense layers.

Add regressions that reject, with unchanged independently derived partition
custody:

1. equal-count class A/B aggregate swaps;
2. equal-count named-stratum aggregate swaps;
3. the same attacks after a fresh stage payload and progress-chain rehash.

Add a positive regression proving a genuine stage replay validates and resume
continues from the preserved prefix.  Focused pytest, bounded randomized
profiles, Ruff, format, `py_compile`, and `git diff --check` must all pass.

## Stop rule

No source freeze, serializer commit, or real n600 work until three fresh review
passes return no Critical/High findings.  This correction adds no score,
promotion, factor-10/Pose, execution, sacred-run, or pointer authority.
