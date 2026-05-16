# Result Review Reactivation Criteria Backstop - 2026-05-16

## Summary

`tools/build_result_review_packet.py` already required explicit reactivation
criteria for negative exact-CUDA regressions, but reviewed non-retirement
packets could still emit `"reactivation_criteria": []`. That weakens
no-signal-loss discipline because exact-CUDA-without-baseline, contest-CPU, and
non-CUDA/proxy reviews become machine-readable dead ends instead of naming the
next evidence that would change status.

## Fix

The tool now normalizes user-provided criteria and fills conservative default
criteria when none are provided:

- contest-CPU packets require same archive/runtime contest-CUDA replay and
  paired component deltas.
- exact-CUDA packets without a baseline require a matching contest-CUDA
  baseline and current-frontier comparison.
- non-CUDA/proxy packets require contest-CUDA exact eval with custody.
- indeterminate engineering/config regressions name audit blockers and require
  a rerun after custody is complete.

Explicit negative exact-CUDA retirements still require operator-provided
criteria; the backstop does not weaken that stricter gate.

## Verification

- `src/tac/tests/test_build_result_review_packet.py` asserts generated criteria
  for exact-CUDA-without-baseline, contest-CPU, and non-CUDA/proxy packets.

## Reactivation Criteria

Reopen this guard if a generated `tac_result_review_packet_v1` or derived
evidence row can contain an empty `reactivation_criteria` array without an
explicit operator-supplied reason.
