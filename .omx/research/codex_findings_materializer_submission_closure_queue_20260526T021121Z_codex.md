# Materializer Submission Closure Queue Finding

Generated at: 2026-05-26T02:11:21Z

## Summary

The materializer exact-readiness followup is now a queue-owned custody chain:

1. harvest materializer manifests into an optimizer source queue;
2. build byte-closed submission/runtime closures;
3. run exact-readiness only on the closed source queue;
4. build the exact-eval dispatch plan from the bridge report.

This closes the previous parser-only gap where harvest could feed exact-readiness
before a runnable `archive.zip` + `inflate.sh` submission packet existed.

## Engineering Changes

- Split `build_materializer_execution_queue(..., include_exact_readiness_followup=True)`
  into explicit harvest, submission-closure, exact-readiness bridge, and dispatch-plan
  steps.
- Added multi-candidate submission closure support. A sweep source queue now closes
  every selected candidate into its own submission directory and writes one aggregate
  closed source queue for exact-readiness.
- Preserved `--require-ready` semantics on the standalone exact-readiness bridge CLI.
- Propagated inverse-scorer chain `inflate_runtime_dir` / `source_runtime_dir` into
  harvested optimizer rows so receiver/runtime closure no longer guesses.
- Preserved packet-member-merge generated receiver runtime fields in harvested rows,
  including `candidate_runtime_dir` and runtime tree SHA.

## Verification

- `ruff check` passed on the touched scheduler, materializer, closure, CLI, and test files.
- `pytest` passed for:
  - `src/tac/tests/test_family_agnostic_materializer_sweep.py`
  - `src/tac/tests/test_family_agnostic_materializers.py`
  - `src/tac/tests/test_byte_shaving_campaign_queue.py`
  - `src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
  - `src/tac/tests/test_frontier_rate_attack_bootstrap.py`
  - `src/tac/tests/test_materializer_submission_closure.py`
  - `src/tac/tests/test_optimizer_candidate_queue.py`

- Focused receiver/materializer closure slice: 40 passed.
- Broader queue/candidate/submission regression slice: 214 passed.
- Full touched materializer/queue regression slice: 266 passed in 13.01s.
- Lane registry validation: 1372 lanes validated cleanly.

## Authority Boundary

All outputs remain planning/custody artifacts only. The closed source queue and
bridge report do not claim score, promotion, rank/kill, or dispatch authority.
Exact CPU/CUDA auth eval and lane dispatch claims remain required before score
claims or paid dispatch.

## Next Bridge

The next high-EV integration is component-correction acquisition on top of the
receiver-closed rate budget: use SegNet/PoseNet component rows to spend newly
freed bytes only where `delta_segnet + delta_posenet + lambda * delta_bytes`
improves under exact-readiness and runtime closure constraints.
