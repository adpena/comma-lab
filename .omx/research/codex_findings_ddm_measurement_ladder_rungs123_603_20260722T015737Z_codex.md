---
title: Codex round-1 findings — DDM measurement ladder rungs 1-3
utc: 2026-07-22T02:25:04Z
task: 603
verdict: CLEAN_AFTER_FIX_FOR_INVOCATION_STABILITY_AND_SCOPE
verdict_scope: local full-resolution n64/n256 measurement apparatus only
research_only: true
execution_allowed: false
---

# Round-1 outcome

The implementation is clean after one high-severity receipt correction. This is one reviewed landing,
not a three-clean-pass seal, n600 closure, scorer result, or PRIMARY admission.

## F1 — volatile receipt fields broke reproducible re-derivation — FIXED

The first committed-source receipt embedded exact wall time and current free-space bytes. Its archive
and checkpoints were deterministic, but an otherwise identical invocation could produce different
receipt bytes. Commit `918edb4453` removes both volatile values while retaining the fail-closed
free-space threshold and pass boolean. The final receipt is generated only from committed producer
bytes after this fix.

The output writer intentionally refuses any pre-existing checkpoint path. That is immutable-output
policy, not a determinism failure; MAIN must use a fresh reviewed output directory for a new proof.

## Re-derived invariants

- All target comparisons use exact 384x512 uint8 chunk bytes; no 8x8 projection enters the fitter or
  bridge.
- The payload is per pair-plane chart or chart stratum, never per pixel.
- Rung 3 hashes the exact rung-2 archive and records 256 ordered per-pair agreement rows.
- Pose debt zero is fully explained by a counted Pose6 stream and is not mislabeled as PoseNet.
- Six-stream sampled no-op honesty, final-byte unique homes, parse/re-encode, compiler x2, and disk
  resume identity all pass.
- `N600_SAME_ARTIFACT_ARCHIVE_CLOSURE` stays red because the measured prefix is n256.

## Verification

Ruff, formatting, compile, diff checks, 41 focused/predecessor tests, real n64/n256 low-level passes,
the full committed-source CLI, and a stopped-then-resumed n256 path are green. Measured full CLI time
was 17.34 seconds.

Pointer `0.1910828242 [contest-CPU]` unchanged. MAIN landing review required.
