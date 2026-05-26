# Receiver-Closed Correction Budget Wiring

Codex pass: 2026-05-26T01:50Z

## Verdict

Receiver/runtime closure now feeds the frontier rate-attack planner as a typed,
queue-visible correction-budget signal. The signal is deliberately local and
non-authoritative: it can prioritize targeted SegNet/PoseNet repair acquisition,
but it cannot claim score, promote, rank/kill, or dispatch exact eval.

## What Changed

- Added `frontier_rate_attack_receiver_closed_correction_budget.v1`.
- The feedback compiler discovers `materializer_submission_runtime_closure_report.v1`
  artifacts, pairs them with exact-readiness bridge reports, dedupes repeated
  closures by candidate/target/archive using the newest closure timestamp, and
  releases only receiver-closed saved bytes.
- Active-floor-only exact-readiness blockers are allowed for local correction
  planning; static runtime blockers such as `archive_manifest_missing`,
  `inflate_sh_missing`, runtime-tree gaps, or unknown uncleared blockers keep
  the budget locked.
- `targeted_correction_budget_summary` now separates total materializer rate
  signal from receiver-closed materializer bytes.
- Refresh artifact writers now emit `receiver_closed_correction_budget.json`,
  and DQS1 queue metadata carries the receiver-closed budget pointer.

## Live Artifact

Fresh bounded refresh:

`.omx/research/frontier_operation_portfolio_20260526T0150Z_receiver_closed_budget/`

Key result:

- receiver-closed candidate count: 1
- receiver-closed saved bytes: 156
- source target: `packet_member_zip_header_elide_v1`
- remaining gates: active-rate-floor override before exact dispatch,
  SegNet/PoseNet component eval before correction spend, exact auth eval before
  any score or promotion claim

## Verification

- `ruff` on touched scheduler/tool/test files passed.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_materializer_submission_closure.py -q` passed: 18 tests.
- Generated DQS1 follow-up queue validated cleanly: 4 experiments, 28 steps.
- Generated receiver-repair queue validated cleanly: 4 experiments, 10 steps.
- `tools/lane_maturity.py validate` passed: 1372 lanes.
- Scoped review-tracker policy check passed for the touched receiver-budget
  surfaces. A full-repo review-tracker sweep still reports pre-existing global
  review debt outside this tranche.

## Remaining Gap

The newly released 156-byte receiver-closed rate budget is only planning fuel.
The next queue-owned slice should consume it in the SegNet/PoseNet correction
acquisition layer, scoring candidate repairs by
`delta_segnet + delta_posenet + lambda * delta_bytes`, with exact authority
still fail-closed.
