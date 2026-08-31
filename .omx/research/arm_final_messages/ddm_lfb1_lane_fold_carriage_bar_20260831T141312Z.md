D3 stays closed; the exact frontier did not move.

- Full-n600 control reproduced **1,325,033** mismatches exactly.
- Lane→Road fold created **690,874** exact corrections.
- At gf1’s measured **0.2909 B/correction**, restoration costs **200,975.25 B**—**9.262×** the **21,699 B** bar.
- All corrections are concentrated in `Lane 1 → Road 0`.
- No scorer, Modal dispatch, candidate archive, or new exact score was run.
- Memo committed as `1c9fbbf587`: [ddm_lfb1_lane_fold_carriage_bar_20260831.md](/Users/adpena/Projects/pact/.omx/research/ddm_lfb1_lane_fold_carriage_bar_20260831.md)
- Retained [result](/Volumes/VertigoDataTier/pact/ddm_lfb1_lane_fold_carriage_bar/LANE_FOLD_CARRIAGE_RESULT.json) and [custody manifest](/Volumes/VertigoDataTier/pact/ddm_lfb1_lane_fold_carriage_bar/CUSTODY_MANIFEST.json); all 58 listed artifacts re-hashed cleanly.

Own-vehicle frontier remains `[contest-CUDA T4 n600]` LB1, **S=0.14803010583079396**, 180,083 B.

## LIVE-HYPOTHESES

- None worth pursuing from this charter. A structurally different receiver-closed Lane carrier below 21,699 B remains a logical falsifier, but this measurement provides no positive lead for one.

## DEAD-ENDS

- D3 rate-only plus Lane restoration: closed at **9.262×** the carriage bar.
- Predicted 36–53 KB restoration range: falsified; derived price is 200,975.25 B.
- Using `ddm_ma2`’s 691,095 count as a control: invalid because it measures token field `cc10…`, not this charter’s `9ba2…` field.
- Pricing D3 with only its 52,531 B payload: invalid against a whole-archive bar; the comparable footprint includes 8 B framing, totaling 52,539 B.

