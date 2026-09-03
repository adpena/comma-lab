MC1 is complete and committed with a typed **CEILING-REFUSED** verdict.

- Global translation: −136.861 SCREEN bits = −17.108 refusal-only bytes.
- Row-band translation: −504.767 bits = −63.096 bytes.
- Integer affine: −147.640 bits = −18.455 bytes.
- Required retraining gate: +5,000 B.
- All three also degraded alignment versus the co-located plane.

Per the charter, HPAC retraining, RC64 encoding, receiver construction, decode timing, scorer, Modal, and Metal were not run.

Artifacts:

- [Full verdict and per-class table](/Users/adpena/Projects/pact/.omx/research/ddm_mc1_motion_compensated_previous_plane_20260903.md:1)
- [Retained result](/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane/RESULT.json)
- [Payload manifest](/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane/MANIFEST.json)
- Commits: `2c32e2767b` runner/tests; `f9aa0b348e` verdict.
- Verification: 4 tests passed, Ruff and `py_compile` passed, two Python review passes recorded, payload gate found 0 issues, and all 34 retained manifest entries re-hashed cleanly.

Own-vehicle frontier remains **AFR1 — S 0.14797617125559104 @ 180,002 B [contest-CUDA T4, n600]**.

## LIVE-HYPOTHESES

- None within MC1’s adjudicated formulation. Non-constant-velocity flow or learned motion estimation would be a different family without a fire order from this arm.

## DEAD-ENDS

- Constant-velocity global, row-band, and integer-affine MC planes are closed at formulation scope: all missed the gate by over 5 KB and worsened alignment.
- The retrain/RC64/archive path is folded at the mandatory ceiling gate; rerunning it would violate closed-form-first.