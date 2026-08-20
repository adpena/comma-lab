Implemented and landed PO1 as commit `995bb49a14`, now an ancestor of current HEAD. No T4 scorer run was dispatched, so the effective frontier remains CP135 at `0.16195513827824176` and `186,252 B`.

Key artifacts:

- [Durable receipt and fire orders](/Users/adpena/Projects/pact/.omx/research/ddm_po1_t4_error_feedback_pose_compensation_20260813.md)
- [T4 feedback worker](/Users/adpena/Projects/pact/experiments/ddm_po1_t4_pose_feedback_worker.py)
- [Governed Modal dispatcher](/Users/adpena/Projects/pact/experiments/ddm_po1_modal_t4_pose_feedback.py)
- [Local compensation solver](/Users/adpena/Projects/pact/experiments/ddm_po1_t4_error_feedback_pose_compensation.py)
- [Focused tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_po1_t4_error_feedback_pose_compensation.py)

The apparatus retains scorer inputs, full outputs, first-six vectors, candidate payloads, checkpoints, and receipts. An initial temporary-directory retention defect in two local mutation fixtures was repaired by deterministic reconstruction; retained artifacts and matching hashes are recorded in the receipt.

Focused verification passed: Ruff, compilation, five PO1 tests, retention guard, review tracker, and lane-registry validation. The broader suite still has two pre-existing C1 custody-fixture failures, and the repository-wide preflight remains red on eight unrelated existing debt classes; no PO1-specific preflight failure was found.

## NEXT_IF_RESUMED

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135`. **Fire trigger:** the sole T4 component lane is clear and the CP135 archive pin verifies; execute the Round 1 command in the receipt.
- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** `ddm_po1` local solver. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_po1_20260813/solve/attempt1`. **Fire trigger:** Round 1 returns `FEEDBACK_USABLE` and every downloaded record passes its byte/SHA receipt.
- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_po1_20260813/dispatch/round2_candidate`. **Fire trigger:** the solver produces `SOLVE_RESULT`, the sole T4 lane is clear, and exact candidate archive bytes/SHA verify; execute the generated Round 2 order.

## LIVE-HYPOTHESES

- The CP135 pose residual has enough systematic structure for one T4 improvement because the update direction combines the actual T4 residual with a local Jacobian preconditioner.
- Most useful corrections will preserve packed CAP1; the canonical-variable fallback can bound the archive-shape risk when they do not.
- Candidate same-job repeat noise will remain below half the pose error on most pairs, making the feedback usable. This is untested until Round 1 runs.

## DEAD-ENDS

- Local CPU PoseNet cannot admit a candidate; prior evidence showed a roughly `21.3×` T4 inversion risk.
- Forcing changed codes into the historical fixed CAP1 length is closed: a retained mutation produced a `22,226 B` physical section.
- The five non-identity F0E1 selector pairs are frozen because the plain Jacobian does not model their selector discontinuity.
- Any candidate that changes an exact T4 SegNet field is rejected, even if its pose metric improves.

