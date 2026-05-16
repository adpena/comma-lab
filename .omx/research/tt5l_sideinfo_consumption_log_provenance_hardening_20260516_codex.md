# TT5L Side-Info Consumption Log Provenance Hardening - 2026-05-16

## Scope

Hardened the TT5L contest full-frame side-info consumption proof against
caller-supplied output-directory false positives. This is custody hardening
only: score_claim=false, promotion_eligible=false,
ready_for_exact_eval_dispatch=false, research_only=true.

## Fix

- `src/tac/substrates/time_traveler_l5_autonomy/consumption_proof.py` now
  refuses inflate provenance that lacks a bound log path and SHA-256, points to
  a missing or empty log, or has a stale log hash or byte count.
- Existing non-target payload identity and allowed `side_len` header-delta
  checks remain required by the contest side-info predicate.
- The contest proof CLI already exits nonzero when `predicate_passed=false`;
  the test surface continues to cover that behavior.

## Wire-In

Sensitivity map: N/A, no score signal.
Pareto constraint: N/A, custody hardening only.
Bit allocator: N/A.
Cathedral autopilot: N/A, this does not create dispatch-ready evidence.
Continual learning: N/A, no empirical anchor.
Probe disambiguator: N/A, no ambiguous design branch.

## Residual Blocker

Existing committed full-frame proof artifacts and L5 staircase consumer checks
do not yet require log-bound provenance. I did not edit
`src/tac/optimization/l5_staircase_v2.py` or rematerialize committed proof
artifacts in this slice because ownership was limited to the consumption proof
builder, its CLI, direct tests, and this ledger. The separate inflate
provenance CLI is also outside this slice's ownership; it must be extended to
record a log path/hash before it can mint proof-acceptable provenance after
this hardening.
