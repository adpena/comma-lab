Executed and committed as `da8176360b`.

The narrow verdict is `INCONCLUSIVE_AT_PREREGISTERED_THRESHOLDS` at `FORMULATION` scope. On `[macOS-CPU scorer-free retained-token n600]`:

- RC1 Lane agreement: 14.7291%.
- Full-Hamming reassignment: 23.1376% Lane agreement and 235,104 fewer total mismatches.
- Lane-only oracle: 60.3777% Lane agreement, with 9.73 collateral errors per recovered Lane token.
- NR1-K32 Lane agreement: 14.7117%.
- Source/codebook Lane-mask diversity: 15,348 versus 1,412 distinct masks.

This proves both assignment loss and incomplete Lane representation. It neither supports the registered `<60%` prediction nor reaches the `>90%` falsifier.

The immediate cure is global full-Hamming reassignment using the existing K=2,048 codebook: zero additional codewords, sections, or raw assignment bytes. Its compressed archive cost remains unknown because LQ1 correctly did not recut the live payload.

No scorer, archive recut, `d_seg`, or exact evaluation ran. All agreement figures are `PROXY-NOT-SCORE`; the frontier did not move.

Artifacts:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_lq1_lane_quotient_representability_20260822.md)
- [Reproducible runner](/Users/adpena/Projects/pact/experiments/ddm_lq1_lane_quotient_representability.py)
- [Primary retained result](/Volumes/APDataStore/pact/ddm_lq1_lane_quotient_representability/measurement_v2/RESULT.json)
- [Independent repeat](/Volumes/APDataStore/pact/ddm_lq1_lane_quotient_representability/verification_repeat_v2/RESULT.json)

The repeat used different chunking and matched both assignment fields, every oracle array, every confusion matrix, and the normalized scientific block byte-for-byte.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-FIRE-ORDER`; **owner:** MAIN-assigned RC1 successor; **consumer store:** `/Volumes/APDataStore/pact/ddm_lq1_lane_quotient_representability/successor_global_reassignment/`; **fire trigger:** AD2/RI1 release ownership and no duplicate RC1 recut is active; **action:** recut RC1 using the retained full-Hamming assignment, retain all candidates/repeats, and queue qualifying scorer work to MAIN.
- **Disposition:** `QUEUED-WITH-FIRE-ORDER`; **owner:** MAIN-assigned NR1 successor; **consumer store:** `/Volumes/APDataStore/pact/ddm_lq1_lane_quotient_representability/nr1_k32_oracle/`; **fire trigger:** NI1 releases ownership and the pinned K32 hashes remain valid; **action:** measure NR1-specific containment and assignment oracles before making any family-level verdict.

## LIVE-HYPOTHESES

- A Lane-mask or Lane-transition specialist conditioned by the shared quotient may outperform whole-program assignment because Lane’s 15,348 source masks are forced through only 1,412 codebook masks.
- Global full-Hamming reassignment plus a small topology residual may dominate either component alone: reassignment removes 235,104 mismatches, while the residual can target support absent from the codebook.
- NR1 may share RC1’s objective failure without proving a universal quotient wall; its own containment oracle can distinguish those explanations.

## DEAD-ENDS

- Simple class-balanced or flip-weighted refitting is closed by CB2’s measured falsifier.
- Global reassignment as a complete Lane cure is closed because it reaches only 23.1376% Lane agreement.
- Calling the existing codebook highly Lane-representable is closed because the Lane oracle reaches only 60.3777%, far below 90%.
- Closing the entire quotient/dictionary family is unsupported because the registered `<60%` threshold was narrowly missed and NR1 lacks its own oracle.
- Raising K along the retained RC1 route is closed because K=4,096 projects to 158,933 bytes, above the 137,986-byte ceiling.
- Promoting token agreement into evaluator improvement is closed because no scorer ran. **LQ1 own-vehicle frontier line: S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`; pointer unmoved.**