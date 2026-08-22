DX2’s Seg error is **90.4702% manufactured downstream** and **9.5298% representation-survived**:

| Component | Errors / 117,964,800 | Final-error share | Byte-equivalent ceiling |
|---|---:|---:|---:|
| Manufactured after correct transmitted label | 21,493 | 90.4702% | 27,362.9 B |
| Representation error surviving final argmax | 2,264 | 9.5298% | 2,882.3 B |
| Total | 23,757 | 100% | 30,245.2 B |

This exactly reproduces `d_seg=0.0002013905843`, rounding to the official `0.00020139`. The evidence is `[contest-CUDA T4 component-only exact field replay]`, not a new score: the retained FX5 argmax transfers because FX5 and DX2 have byte-identical T4 inflated raw output.

Key findings:

- RT1’s old `argmax != label` ratio is invalid here: it becomes 120.19% because rendering also corrects 6,918 representation errors.
- Lane holds 24.65% of errors despite only 0.586% area, but its 90.2493% manufactured fraction is slightly below the body average. The Lane-above-average prediction is falsified.
- The registered 60–90% band is narrowly falsified by 0.47 percentage points; the `<25%` representation-limited falsifier did not fire.
- RT1’s retained positive control reproduced 34,938 final errors, 1,717 representation errors, and 33,743 label-to-argmax changes exactly.
- Renderer, resize, uint8, logits, and argmax sub-fractions remain unmeasured because no progressive DX2 fields exist and ms9 did not own the scorer lane.

Artifacts:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_ms9_dx2_seg_manufactured_fraction_20260822.md)
- [Reproducible instrument](/Users/adpena/Projects/pact/experiments/ddm_ms9_dx2_seg_manufactured_fraction.py)
- [Retained receipt](/Volumes/VertigoDataTier/pact/ddm_ms9_dx2_seg_manufactured_fraction/MS9_FIELD_REPLAY.json)
- Retained 45 masks across 49 files, 663,632,396 B; deterministic repeat passed.
- Serializer commit: `aae19dd882`

OWN-VEHICLE FRONTIER: UNMOVED — DX2 remains `S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600]`.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-A-FIRE-ORDER`; **owner:** MAIN; **consumer store:** `/Volumes/VertigoDataTier/pact/ddm_ms9_dx2_seg_manufactured_fraction/fresh_stage_split_r1/`; **fire trigger:** MAIN grants the unique n600 scorer lane after admitted jobs are terminal, the store is fresh, storage/provenance preflight passes, and RT1/DX2 terminal controls reproduce exactly; **action:** run the progressive native-render → camera-uint8 → evaluator-resize → logits → argmax decomposition, retaining every field and mask.

## LIVE-HYPOTHESES

- Most manufactured loss lies in the learned render or resize path, because the transmitted label is already correct at 21,493 final-error pixels.
- A support-selective pre-R correction may work where generic deblur failed, because it can preserve the 6,918 beneficial downstream corrections rather than suppressing all renderer changes.
- Retained logits and margins may expose a cheap argmax-collapse subset; this remains plausible because the current exact payload retains only terminal classes.
- NI1’s K32 representation may still win through rate despite representation explaining only 9.53% of final errors; its shipped-receiver distortion remains unmeasured.

## DEAD-ENDS

- Transferring RT1/HV1’s 95–96.6% fraction directly to DX2 is closed: it is a different vehicle and used a non-additive diagnostic.
- Defining manufactured loss as every `argmax != label` pixel is closed: it yields 120.19% and double-counts beneficial corrections.
- Calling DX2 representation-limited is closed: only 2,264 of 23,757 final errors survive from representation error.
- Claiming Lane is more manufactured than average is closed: 90.2493% is below 90.4702%.
- Assigning render/R/uint8/argmax percentages from the retained fields is closed: those progressive fields and logits do not exist.