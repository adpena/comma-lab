MST1 is complete and committed as `1c33f278920b91bf922e9620deb9ce20615135e8`.

| Earliest stage | Manufactured | Share | Final repairs | Gross break / repair | Ceiling |
|---|---:|---:|---:|---:|---:|
| Native render + frozen head, unseparated | 16,917 | 78.7093% | 6,154 | 28,602 / 6,281 | 21,537.2 B |
| Float bilinear round trip | 4,030 | 18.7503% | 695 | 4,841 / 11,821 | 5,130.6 B |
| uint8 round trip | 544 | 2.5311% | 69 | 853 / 1,624 | 692.6 B |
| CPU→CUDA terminal, unseparated | 2 | 0.0093% | 0 | 7 / 2 | 2.5 B |

The R-dominance prediction is falsified: the float round trip carries 18.7503%, not ≥50%. Manufacture instead concentrates at the native-render-plus-frozen-head observation, where changing the rendered input is legally addressable. R and uint8 are net repair stages, so “undo the round trip” would destroy valuable corrections.

The exact contest-CUDA DALI-GT gates reproduced without drift: 23,757 final errors, 21,493 manufactured errors, 2,264 surviving representation errors, and 6,918 repaired representation errors. Lane contains 5,285 manufactured pixels, 7,651.060 per million Lane pixels—41.99× the body density—but remains collateral-constrained.

Evidence boundaries:

- Intermediate observations: `[macOS-CPU advisory]`, frozen CPU-torch SegNet.
- Final support: retained `[contest-CUDA T4 component-only exact field replay]`.
- Verdict scope: `INSTANCE:DX2_T4_n600_WITH_MACOS_CPU_INTERMEDIATE_OBSERVATIONS`.
- No cure, changed archive, pose/rate response, or new exact score was measured.
- No new scorer fire was needed; the existing canonical-fire n600 capture was reused.
- Local custody contains 418 primary payloads totaling 20,834,592,704 bytes plus 33 attribution masks. No `/Volumes/*` writes occurred.

Artifacts: [memo](/Users/adpena/Projects/pact/.omx/research/ddm_mst1_manufactured_stage_split_20260822.md), [instrument](/Users/adpena/Projects/pact/experiments/ddm_mst1_manufactured_stage_split.py), [verification receipt](/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local/COMPLETED_VERIFICATION.json).

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN / next DX2 Seg-cure builder; consumer store: `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/renderer_joint_cure_r1/`; fire trigger: the sole n600 scorer lane is free and a retained byte-closed same-length candidate exists. Jointly optimize through native render, R, and uint8 while preserving pose and all 6,918 beneficial corrections.

## LIVE-HYPOTHESES

- A same-length joint renderer solve may reduce native-stage breaks without adding archive bytes; 78.7093% of final manufacture first appears on this controllable surface.
- Lane-aware admission may pay if frozen-head collateral constraints protect other classes; Lane debt is exceptionally dense, but unconstrained Lane repainting is known to spill.
- A shared CPU/CUDA margin objective may absorb the nine-pixel terminal disagreement as a validation fold, not a separate arm.

## DEAD-ENDS

- R-dominance is closed on DX2: 18.7503%, not ≥50%.
- Distributed loss is closed: the native observation alone carries 78.7093%.
- Generic deblur, blur, sharpen, antialias, and undirected dither are not justified follow-ons; R and uint8 are net repair stages here.
- A direct CPU/CUDA terminal cure is inadmissible and worth only a 2.5 B ceiling.
- Claiming score progress is closed: MST1 changed no archive. OWN-VEHICLE FRONTIER: UNMOVED — DX2 remains S=0.14821987563243377 @ 180,368 B `[contest-CUDA T4 n600]`.