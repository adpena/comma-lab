The field-for-coder rate opening is real and receiver-closed, but distortion remains unmeasured and queued.

| Candidate | Edits | Real Δbytes vs jt21 |
|---|---:|---:|
| batch0 | 1,761 | −1,292 B |
| batch1 | 1,754 | −1,240 B |
| batch2 | 1,753 | −1,241 B |
| union | 5,268 | **−3,756 B** |

The union is **176,436 B**, rate-only ΔS **−0.0025009662**. Its public receiver decoded the exact retained token field and produced a 3,662,409,600 B uint8 raw in 1,740.7 seconds. The jt21 control also passed in 1,658.7 seconds.

Not measured: realized `d_seg`, fresh compensated `d_pose`, or net S. Therefore there is no admission, refusal, family closure, seal, Modal dispatch, or pointer move.

Artifacts:

- [Findings memo](/Users/adpena/Projects/pact/.omx/research/ddm_fcd1_field_for_coder_diagonal_20260829.md)
- [Machine result](/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/BYTE_ONLY_RESULT.json)
- Commit `6df3b4ea9b`
- Focused lint clean; 4/4 tests passed; two review-tracker passes completed.
- Existing unrelated dirty files were preserved.

Own-vehicle frontier: **S = 0.14811799921260607 @ 180,215 B `[contest-CUDA T4 n600]` — UNMOVED.**

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: qbt2b r10’s scorer claim becomes terminal. Run the union’s fresh Schur baseline/solve/close/repeat and publish only if `d_pose_after <= d_pose_base + band`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: repeat-identical Schur publication passes. Run both frozen scorers n600, recompute S from components, and apply the ±3.5e-6 admission band.
- **FOLDED** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: union refuses by ≤5×. Process batch0, batch2, then batch1 independently.
- **FOLDED** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: any candidate is admitted. Write its canonical seal and dual-axis fire order; MAIN retains dispatch ownership.

## LIVE-HYPOTHESES

- The union may lower realized Seg error: its 5,268 edits replace wrong labels with GT-correct coding argmax labels, and its 1.403 label-benefits/saved-byte exceeds the 0.785 realized-flip/B screen.
- Fresh Schur compensation may remove the dominant Pose tax without erasing the measured rate credit; qs5 proves reach, but not transfer to this object.
- A disjoint batch may win even if the union refuses because scorer costs need not compose additively.

## DEAD-ENDS

- Same dg2 moves plus perfect Pose compensation: Seg harm alone remains 45.18 rate credits net worse.
- Entropy estimates, average prices, or separately adding jt21 credit: replaced by exact joint re-encodes.
- Carrying old qs4/qs5 compensation: forbidden; only the fresh object-bound compile gate may publish.
- Treating B/H token labels as realized SegNet flips: wrong object until R and frozen scorers run.
- The stale native corrector: its older family generation correctly refuses jt21; the Python `FreeCorrector` decoded both objects under budget.

