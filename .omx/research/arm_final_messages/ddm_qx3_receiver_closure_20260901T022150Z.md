QX3 is complete with verdict `BLOCKED`.

Measured scorer-free:

- QX1 decoder parity: `0 / 117,964,800` mismatches against the retained QBT native field.
- QX1 versus QX2 baseline: `1,669,798 / 117,964,800` mismatches.
- Event sites: `9,619 / 17,926` mismatches.
- Cheapest exact closure: dense delta + QX2 events, LZMA-9e, `510,404 B` versus the `24,093 B` cap.
- Complete archive: `624,296 B`, `486,311 B` over the gate.
- Both receiver passes reconstructed all `17,926` events and the exact target SHA `36c6be71…8b68`.

Artifacts:

- [Verdict memo](/Users/adpena/Projects/pact/.omx/research/ddm_qx3_receiver_closure_20260831.md)
- [Result JSON](/Volumes/APDataStore/pact/ddm_qx3/RESULT.json)
- [Runner](/Users/adpena/Projects/pact/experiments/ddm_qx3_receiver_closure.py)
- [Verified fallback bundle](/Volumes/VertigoDataTier/pact/ddm_qx3/receipts/commit_serializer_fallbacks/20260901T022101.144818Z-90629/intended-commit.bundle), commit `f79c65996e1fd5fbb6f4eb81cc4f7ff8e6b43e97`, bundle SHA `1374538c…04cd7`

Validation passed: Ruff, compilation, two tests, payload-retention `0/2`, two review passes, zero policy violations, and rehashing of 25 QX3-owned facts totaling 862,378,336 bytes. Git writes were sandbox-denied; the serializer returned `rc=17`, retained the verified bundle, and left the index empty.

No scorer, evaluator, Modal, Metal, MPS, distortion, or score measurement ran. The frontier remains afr1: `S = 0.14797617125559104 @ 180,002 B [contest-CUDA T4, n600]`, archive `cbb8d92…5bf25`.

## NEXT_IF_RESUMED

- **BUNDLE_READY_MAIN_MUST_LAND** — owner: MAIN; consumer store: Git main plus the QX3 memo; fire trigger: import the verified bundle onto base `c3f2f4ad097c…`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN-assigned QX representation owner; consumer store: `/Volumes/APDataStore/pact/ddm_qx3/RESULT.json`; fire trigger: a structurally new receiver-available baseline statistic or direct QBT-to-target grammar plausibly fits within `24,093 B` without free-code GT/S2/scorer data.

## LIVE-HYPOTHESES

- A direct grammar from the decoded QBT field to the final target could exploit boundary continuity better than repairing C1 and then applying QXC1. Dense LZMA’s 510 KB result shows structure, though another 21.2× reduction is required.
- Retraining or re-valuing the counted QBT packet against the exact C1 field could make the baseline decoder-native. This requires a newly priced core and fresh distortion evidence.

## DEAD-ENDS

- Pure zero-byte derivation from this QX1 core is closed: the exact decoder misses 1,669,798 baseline sites.
- Event-site-only agreement is insufficient and already fails at 9,619 sites; QXC1 depends on the whole baseline.
- Dense-delta and sparse-uint32 corrections under Brotli, LZMA, and zlib are closed at formulation scope.
- Substituting the foreign QBT baseline without repricing QX2 is closed because QXC1 binds the exact C1 SHA.
- The exact 624,296-byte decode is not goal progress: it exceeds the byte gate and has no score row.