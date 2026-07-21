# Codex session summary — Predictor R4 tail race — 2026-07-21

`task=578` · `lane_id=predictor_r4_tailrace` · `MAIN_REVIEW_REQUIRED`

## Landed in this worktree

- A source/config-bound, resumable R4 implementation with typed `r4-n64` and `r4-final` stages.
- Exact R3 physical-tail reconstruction with global cross-stratum component subtraction.
- A counted three-way race: literal exceptions vs shared cellular generator plus its own exceptions
  vs eaten flips.
- Operator-routed prior reuse: exact n64 Task #208 detection, a numerically consumed openpilot
  degree-4 Lane polytope warm start, and a counted 4x3 rank-4 scorer-quotient form on every stream.
- Preserved per-stage SSD checkpoints and byte-identical repo/SSD final receipts.
- Regression tests, build spec, findings memo, DAG FEED, and reuse manifest.

## Empirical anchor

No n64 stream admitted the tested learned generator.  R3's physical union is 1,894,849 sites, not
its additive 1,898,681-site report, because 1,676 duplicate residual records and 2,156
residual/noncausal overlaps must be unioned once.  Admitted component overcredit is zero after
correct global site-level subtraction.  Curve-v4 therefore remains 216,207 bytes and corrects
description `d_seg` to 0.01606283399793837.

Task #208 selected Lane but did not select Movable on n64 because Movable static-IoU measured
0.33223226703755215 versus the sealed 0.20 cutoff.  The openpilot Lane prior round-tripped exactly
and was numerically consumed, but both Lane stream races still lost.  No rank-4 weights-bar form
cleared; admission would remain blocked regardless because the exact rank4-to-RGB uint8 pullback is
absent and #580 supplies only the spatial resize-kernel projector.

The subsequent upstream-weight directive was audited too.  This delegated worktree has no
`upstream/` directory, frozen scorer tensors, deeper-layer basis arrays, or activation receipt.
The existing encode-side skip-feature extractor is source-bound, but no scorer forward was run and
no weight-derived constants were invented.  This is a local-custody blocker only; the family stays
open.  Decode-side SegNet/PoseNet loading remains prohibited.

The ep725 83,838-byte witness at `d_seg=0.003457972208658854` dominates this formulation on both
custodied coordinates, with the explicit caveat that R4 is receiver-open description evidence.

## Verification and authority

R2/R3/R4 regressions: **22 passed**.  Ruff, py_compile, JSON/receipt invariants, and repeated
resume/final SHA checks: **PASS**.  Final receipt SHA-256:
`775d9228a638284de18b254357c47436192a302ea468a2db5cbd2af84f6687f2`.
Pointer **0.1910828242 [contest-CPU] UNMOVED**.  No scorer, archive realization, GPU/paid dispatch,
or pointer mutation occurred.

## Next action

MAIN must independently review and merge the serialized commit.  Do not dispatch n600 learned-tail
training from this result; the n64 strict gate failed.  The higher-EV continuation is the separate
receiver-realization line or a materially different grammar with a new measured entry premise.

## STORES CONSULTED

Canonical manuals/specs; Claude memory top-10; Task #578 workflow and seed doctrine; R2/R3 receipts
and SSD evidence; canonical equation sources; lane/subagent state; ep725 duty/yhat ledgers.

## HISTORICAL_PROVENANCE

Append-only TIER-0 summary for the delegated R4 worktree.  It does not authorize MAIN landing.
