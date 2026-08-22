Implemented and committed NI1 as `b34f3d6f6f`.

The exact candidate archive is 122,250 B, SHA `fe7fe805…ca534e`, leaving 15,736 B below the strict fixed-distortion cap. It independently decodes all four NR1 surfaces exactly once, retains identical full-token/archive repeats, all 36 coder-race payloads, and seven mutation controls.

No scorer was launched: the charter assigns the sole n600 slot to MAIN and explicitly says “do NOT fire.” Therefore d_seg, d_pose, Lane retention, S, and pass/fail against `0.00021731` remain unmeasured.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_ni1_nr1_k32_receiver_distortion_20260822.md)
- [Build result](</Volumes/APDataStore/pact/ddm_ni1_nr1_k32_receiver_distortion/build_r4/RESULT.json>)
- [Sealed MAIN fire order](</Volumes/APDataStore/pact/ddm_ni1_nr1_k32_receiver_distortion/build_r4/SEALED_FIRE_ORDER.json>)
- [Shipping adapter](/Users/adpena/Projects/pact/experiments/ddm_ni1_runtime_receiver.py)
- [Reproducible builder](/Users/adpena/Projects/pact/experiments/ddm_ni1_nr1_k32_receiver_distortion.py)

Verification passed: build and resume, archive integrity, parse/decode repeat identity, lint/compilation, targeted payload-retention audit with zero findings, and two review-tracker passes.

## NEXT_IF_RESUMED

- `QUEUED` — owner: MAIN; consumer store: `build_r4/harvest/advisory_r1/HARVEST.json`; fire trigger: MAIN holds the sole n600 slot, earlier scorer jobs are terminal, candidate pins revalidate, and Vertigo has ≥12 GiB free; action: canonical advisory repeat 1.
- `QUEUED` — owner: MAIN; consumer store: `build_r4/harvest/per_class_r1/RESULT.json`; fire trigger: advisory r1 retained its complete raw and exact d_seg, and MAIN holds the scorer slot; action: run the RI1 per-class method with Lane on its own row.
- `QUEUED` — owner: MAIN; consumer store: `build_r4/harvest/advisory_r2/HARVEST.json`; fire trigger: advisory r1 and per-class scoring are terminal; action: canonical advisory repeat 2.

## LIVE-HYPOTHESES

- K32 may pass because removing superseded HPAC widens the fixed-pose ceiling to `d_seg=0.00030617515`; this follows from exact bytes but is untested.
- Rare-class collapse may still make K32 fail; 1,558,833 tokens changed and RC1’s Lane collapse make it plausible, but not measured for K32.
- K64 may be the better fallback: it changes fewer tokens and projects to 133,122 B with the same honest receiver shape.

## DEAD-ENDS

- Token agreement cannot substitute for evaluator distortion.
- The July synthetic quotient ABI is not a shipping-receiver shortcut.
- Keeping learned HPAC beside NR1 would create paid, output-inert duplicate context.
- Mutation refusal proves integrity, not scorer effect.
- K128 remains rate-dead in this packet family.

OWN-VEHICLE FRONTIER: UNMOVED — NI1 S NOT MEASURED; DX2 remains S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600].