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

## Follow-Up Resolution

Resolved in the follow-up L5-v2 slice: `tools/build_tt5l_inflate_provenance.py`
now exposes `--log-path`, the committed contest full-frame proof was
rematerialized from log-bound CPU inflates, and
`src/tac/optimization/l5_staircase_v2.py` now pins the refreshed committed proof
SHA-256. This removed the side-info gate blocker without weakening the semantic
checks.
