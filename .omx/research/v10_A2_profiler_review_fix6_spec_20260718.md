# v10 A2 profiler review fix 6 — cache custody and governed admission

Date: 2026-07-18  
Lane: `v10_A2_profiler_20260718`  
Authority: `codex_delegate:v10_A2_profiler_20260718:20260718T160947Z`  
Status: **PRE-LAUNCH / NO-GO UNTIL CLEAN RE-REVIEW**

## Fresh-review findings

A post-fix5 adversarial audit reproduced two launch-blocking gaps in the
feature-cache half of the lane.

1. A non-control live-logit slice and its local row hash could be changed
   together and a completed cache still validated.  The progress rows were
   independently hashed but had no predecessor commitment or terminal root.
2. The extractor accepted RSS and timeout values into identity metadata, but
   an unarmed direct invocation could pass the repository admission helper in
   advisory mode.  The values were not an enforced process boundary.

The same audit confirmed that all stored logits come directly from the frozen
forward and that the rank-4 quotient remains diagnostic-only.  It also found
that the only independent fresh re-forward control is frame 195, so wording
that implies 600 independent replays would exceed the evidence.

## Required correction

- Chain every committed frame over the cache identity, predecessor chain
  digest, frame index, direct-logit and quotient-diagnostic slice hashes, and
  canonical per-frame diagnostics.  Resume must re-derive and verify the full
  predecessor chain before accepting a prefix.
- On completion, persist a separately structured terminal receipt binding the
  identity, committed frame count, and terminal chain root.  Complete-cache
  validation must require and verify that receipt.  A regression must alter a
  non-control frame and refresh its local slice hash while leaving the terminal
  commitment untouched; validation must reject.
- Describe this as deterministic corruption/tamper evidence.  It is not a
  digital signature and does not defend against an actor authorized to rewrite
  every artifact and every custody record.
- Refuse non-test extraction unless the real governed-admission marker is
  present.  `--rss-cap-mb` and `--timeout-seconds` are requested outer-governor
  limits and identity-bound provenance, never self-enforced claims.  Add
  regressions for unarmed refusal and governed acceptance.
- Keep the authority statement exact: all 600 frames, once complete, are
  direct frozen-forward captures serialized bit-for-bit into the cache; frame
  195 alone receives an independent fresh-forward bitwise control.  Do not
  claim an independent all-n600 replay.

Retain transactional flush/fsync-before-progress ordering, resumability,
positive-control semantics, false score/promotion authority, storage
preflight, direct-logit/rank-4 separation, and no-Fourier compliance.

## Stop rule

No serializer commit or real n600 launch until this correction, the seeded
receiver-closure integration, focused tests, and three fresh adversarial clean
passes all succeed.  This correction grants no score, contest-axis, promotion,
factor-10/Pose, sacred-run, or pointer authority.  MAIN landing review remains
mandatory.
