# Codex Findings: False-Authority Paradox Closure

UTC: 2026-05-27T13:31:26Z

## Summary

Closed the automation paradox identified during the final-rate-attack sprint:
automation must move fast, but it must never treat its own schema booleans as
receiver, runtime, archive, or exact-readiness evidence.

Two dispatch-adjacent false-authority surfaces are now guarded by executable
code and regression tests:

1. Family-agnostic exact-readiness runtime proofs no longer pass from
   top-level `passed=true` / `runtime_consumption_proof_passed=true` alone.
   They require an actual runtime probe with evidence and, when a candidate
   archive path is supplied, a file-backed archive SHA/byte check.
2. Repair-budget materializer binding no longer marks a candidate archive
   materialized or receiver-consumed from manifest booleans and string paths.
   It validates candidate archive existence, archive SHA/bytes, proof file
   existence, proof JSON, and proof/receiver blockers before enabling local
   materialization execution audit.

## Automation Loop Closure

- Exact-dispatch path:
  `src/tac/optimizer/exact_readiness.py::validate_runtime_consumption_proof`
  now rejects boolean-only family-agnostic proofs and empty probes.
- Final-rate-attack materializer path:
  `src/comma_lab/scheduler/frontier_rate_attack_feedback.py` now verifies
  materializer archive/proof files before `candidate_archive_materialized`,
  `runtime_consumption_proof_present`, or `receiver_consumed` can become true.
- 5D follow-up path:
  the just-landed queue discovery hardening already routes expanded search
  roots through queue-owned input binding; this closure prevents those bound
  artifacts from becoming symbolic authority.

## Continual-Learning Loop Closure

This memo is paired with a Catalog #313 probe-outcomes ledger row:

- `probe_id`: `codex_false_authority_paradox_closure_20260527T133126Z`
- `substrate`: `automation_final_rate_attack_false_authority_gates`
- `probe_kind`: `schema_only_false_authority_paradox_regression`
- `verdict`: `PROCEED`
- `blocker_status`: `advisory`

The row is advisory rather than blocking because the bug class is now guarded;
future related work should preserve or extend these gates rather than bypass
them.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_exact_readiness_runtime_consumption_proof.py -q`
  passed: 3 tests.
- `.venv/bin/python -m pytest src/tac/tests/test_exact_readiness_runtime_consumption_proof.py src/tac/tests/test_repair_budget_materialization_execution.py -q`
  passed: 11 tests.
- Broad targeted suite passed:
  `src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py`,
  `test_pair_frame_5d_extended_operator_queue.py`,
  `test_frontier_rate_attack_refresh_pair_frame_cli.py`,
  `test_frontier_rate_attack_feedback.py`,
  `test_repair_budget_materialization_execution.py`,
  `test_exact_readiness_runtime_consumption_proof.py`,
  `test_mlx_cache_audit.py`,
  `test_mlx_scorer_response.py`,
  `test_mlx_production_contract.py`,
  `test_archive_grammar.py`
  passed: 271 tests in 57.15s.

## Remaining Follow-Ups

- Runtime adapter closure still has a related overstatement risk when an
  adapter directory exists but no expected runtime tree SHA is present.
- Materializer chain harvest still amplifies proof-present booleans before
  exact-readiness validates them.
- Experiment queue observer should revalidate false-authority fields instead of
  trusting `postcondition_passed` on authority-sensitive artifacts.

Those are follow-up hardening tasks, not blockers for this closure.
