Executed and committed as `fde2e4f823`.

The scorer-free census held all 200 JS6 proposals; 0 survived:

- Best lower-calibration net: `+5.25553385416666e-7 S`
- Best upper-bound net: `+6.291373697916667e-5 S`
- Both assume zero added bytes and optimistically credit every target pixel as a successful Seg flip.

Therefore no candidate or fire order was honestly admissible. The sealed [NO_FIRE_ORDER.json](/Volumes/VertigoDataTier/pact/ddm_js6b_pose_screened_compile_20260813/NO_FIRE_ORDER.json) records the `FOLDED` disposition. All 14 materialized Q3 diagnostic pairs were retained; deterministic resume reproduced the final hashes.

The T4 worker now optionally retains n600 SegNet fields plus official PoseNet vectors, inputs, full outputs, and repeat diagnostics from the same decoded dispatch. Flag-off behavior remains on the legacy path.

Full receipt: [ddm_js6b_pose_screened_compile_20260813.md](/Users/adpena/Projects/pact/.omx/research/ddm_js6b_pose_screened_compile_20260813.md)

Verification: 21 tests passed; Ruff, `py_compile`, two review passes, payload-retention gate, and commit hooks passed.

Not measured: candidate scorer outputs, candidate archive bytes, complete candidate S, or a new exact evaluation. Frontier remains **CP135 S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]**.

## NEXT_IF_RESUMED

- **FOLDED** — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_js6b_pose_screened_compile_20260813/NO_FIRE_ORDER.json`; fire trigger: reopen only when a receiver-realizable integer Q3 actuator or candidate-specific retained Pose evidence makes the upper screened net ΔS strictly negative.

## LIVE-HYPOTHESES

- A hard-rounded, relinearized Q3/CVP actuator may remain Seg-reachable while controlling integer Pose leakage.
- Joint whole-candidate selection using retained Pose vectors may escape the envelope governing fixed unprojected semantic-cell edits.

## DEAD-ENDS

- The sealed 200-row unprojected JS6 bank is closed at formulation scope.
- Firing its least-bad proposal is closed; its optimistic upper-bound net remains positive.
- Float Q3 energy cannot be treated as exact integer/receiver Pose nullity.
- Seg-only provisional admission is closed for this family.