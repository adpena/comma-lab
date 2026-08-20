Implemented and landed jo1 in commit `3bc2cb557f`.

Measured result:

- Direct HP3/RC64 probability-object encoding made all 200 events cost only **+3 archive bytes**, with no explicit coordinate section.
- The 6-event calibrated candidate cost **+1 byte**: 186,253 B, SHA-256 `cbcbb9ec…`.
- All four objects receiver-closed through 117,964,800 symbols. The composed candidate repeated byte-identically.
- The JS7-44 exact-output reprice is `S=0.1632129625`; this is a derived reprice, not a new evaluator row, and does not beat cp135.
- No exact n600 scorer was run because jo1 does not own that lane. The effective frontier remains cp135 at `S=0.1619551383`; LC2 remains the own-vehicle row at `S=0.1695989957`. The pointer did not move, so this unit did not achieve goal progress.

Artifacts:

- [Research handoff](/Users/adpena/Projects/pact/.omx/research/ddm_jo1_joint_probability_object_20260812.md)
- [Implementation](/Users/adpena/Projects/pact/experiments/ddm_jo1_joint_probability_object.py)
- [T4 fire recipe](/Users/adpena/Projects/pact/.omx/research/ddm_jo1_joint_probability_object_20260812_t4_recipe.json)
- [Final retained result](/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/FINAL_RESULT.json)

All 4.6 GB of materialized payloads were retained. Five focused tests passed; review coverage reached 100%. Full dev preflight was not green: 17/25 gates passed and eight codebase-wide gates remained red, recorded without weakening or waiver.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN exact contest-CUDA scorer owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/main_t4`; fire trigger: MAIN owns the sole scorer lane, all recipe pins pass, and MAIN accepts the documented non-authority prefilter risk.

## LIVE-HYPOTHESES

- A broader n600-supported robust event set could yield enough Seg flips at nearly zero rate because 200 events already cost only three bytes.
- The six disjoint negative-pose events may preserve some pose benefit under exact scoring, although n32 additivity is visibly invalid.
- If exact scoring exposes positive pose debt, frame-0 Schur repair remains plausible because frame 0 is structurally Seg-free.

## DEAD-ENDS

- The existing JS7-44 stack is not a promotion candidate: its direct reprice remains worse than cp135.
- Bending-energy ordering is closed on this 200-event store: it produced 502 robust gains over 15 useful events versus 874 over 30, with incompatible preferred signs across strata.
- The current all-200 object is closed as a promotion candidate because its singleton pose diagnostic adds `0.000612` and its n32 projection is invalid, though its rate mechanism remains useful.
- The ps135 carrier is closed by its pass-4 exact result; jo1 correctly retained cp135’s carrier.
- Treating n32 event sums as compositor-authority evidence is closed because the composed pose sum crossed below the physical zero floor.