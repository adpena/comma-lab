# Codex Findings: MLX Unit, Scope, And Authority Bug Class

Date: 2026-05-23T05:46:23Z
Agent: Codex
Scope: MLX scorer-response, DQS1 selective-window planning, learned sweep consumers, quality/speed calibration, and rebuildable artifact retention.

## Summary

The DQS1 issue was not isolated. It belonged to a broader bug class where a
locally valid scalar was interpreted under the wrong denominator, scope, cache
identity, or authority contract.

Fixed classes in this pass:

- Singleton/window MLX gains can no longer pass effective spend triage as if
  they were full-video gains. Selection now gates and sorts on normalized
  full-video gain, projected full-video delta, and normalized byte margin.
- Cross-family portfolio conversion now uses projected full-video MLX deltas,
  not raw selected-window deltas, when constructing predicted score means.
- Scorer-response summaries and next-probe economics now prefer normalized
  full-video MLX values for MLX rows and tag the planning scope.
- DQS1 window bridge now requires explicit full-video denominator and
  normalized-gain fields instead of silently defaulting denominator metadata.
- Pairset acquisition and dynamic learned sweep no longer treat an inherited
  source selector exact-CPU estimate as candidate-specific for child mutations.
- MLX quality/speed delta manifests now block spend-triage use unless the
  anchor and MLX row agree on full-contest sample count, pair window, archive
  SHA, inflated-output aggregate SHA, raw SHA, and candidate cache pair count.
- Local CPU advisory cache identity may unlock only local debug/speed uses in
  MLX scorer-response device contracts, not transfer-calibrated spend filters.
- Windowed MLX response dataset construction now requires auth-audited windows
  by default; unaudited datasets require an explicit debug opt-out.
- Rebuildable artifact retention now revalidates certificates before mutation,
  hashes payload files instead of trusting manifests, journals execution, routes
  real `experiments/results` deletion through the canonical GC helper, verifies
  cold-store copies before source deletion, and reports unknown raw surfaces as
  blocked instead of invisible.

## Evidence

- Local MLX CPU full-600 DQS1 score matched the local CPU advisory anchor within
  `+1.4618214861372714e-06`, making the prior apparent larger drift mostly a
  unit/scope bug rather than a fundamental CPU-vs-MLX gap for batch-1 CPU.
- Focused verification after fixes: `141 passed` across MLX spend triage,
  quality/speed delta, scorer-response dataset, cross-family portfolio, DQS1
  bridge, retention, and MLX scorer-response tests.
- Retention safely deleted 493.44 GiB of certified rebuildable scratch while
  preserving blocked failed-locality raw outputs. The post-cleanup plan reported
  zero certifiable reclaimable candidates and retained blocked candidates.

## Remaining Work

- Extend audit-stamp validation so MLX auth/local-advisory cache stamps load and
  verify referenced audit JSON by SHA and cache identity, rather than trusting
  embedded manifest copies.
- Harden `tools/materialize_mlx_scorer_cache_from_auth_eval.py --force` with
  owned-directory markers before deleting any preexisting work/cache directory.
- Broaden unknown raw-surface certifiers beyond direct `.raw` directories so
  common names like `auth_eval_work`, `eval_work`, and `contest_auth_eval_cpu_workdir`
  become either certifiably reclaimable or explicitly blocked with schema reason.
- Convert the scheduler worker from single-step synchronous execution to true
  resource-bounded batch dispatch, then rerun local MLX/CPU queues under live
  telemetry.
