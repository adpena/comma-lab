# JCSP Cross-Paradigm Runtime Consumer Inventory Refresh

Date: 2026-05-16
Author: codex
Evidence grade: local source-and-test audit
Score claim: false
Dispatch attempted: false
Ready for exact eval dispatch: false

## Finding

The cross-paradigm frontier inventory still carried the older blocker:

`submission runtime detects but refuses jcsp.bin consumption`

That blocker is stale for the narrow real AQ rawvideo contract landed on
2026-05-11. `submissions/robust_current/inflate.sh` now detects `jcsp.bin`,
defaults `JCSP_RUNTIME_BRIDGE_MODE` to `consume-real-raw-outputs`, invokes
`submissions/robust_current/jcsp_runtime_bridge.py`, and exits successfully
only when the bridge emits real contest `.raw` outputs.

## Correction

The inventory row for `joint_admm_balle_arithmetic_stack` now routes to
`prove_jcsp_runtime_parity_and_charged_stack` instead of wiring a runtime
consumer that already exists. The remaining blockers are deliberately
fail-closed:

- candidate-specific raw-output parity proof is still missing for a stacked
  JCSP archive;
- Balle hyperprior stream codecs still need non-fixture charged stream
  instantiation;
- side information must be charged;
- no lane dispatch claim exists;
- no exact CUDA auth eval exists for a stacked JCSP archive;
- individual component readiness does not imply stack score composability.

## Non-Claims

This is not a score result, not a promotion, and not dispatch authorization.
It only prevents the operator queue from re-spending attention on an already
completed runtime-consumer step. The next frontier action is to build a
candidate-specific stacked JCSP packet, prove raw-output parity and charged
side-information closure, then claim the lane before any exact CUDA dispatch.

## Verification Plan

- `src/tac/tests/test_build_cross_paradigm_frontier_inventory.py` asserts the
  stale blocker is absent and the current parity/charging blockers remain.
- `src/tac/tests/test_build_frontier_roadmap_status.py` asserts the roadmap
  classifies JCSP as a byte-closed-candidate/parity task, not a missing
  runtime-consumer task.
- `src/tac/tests/test_jcsp_runtime_bridge.py` continues to cover the
  robust-current consume path from `inflate.sh`.
