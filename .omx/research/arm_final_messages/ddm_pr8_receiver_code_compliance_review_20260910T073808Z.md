Completed the receiver-code compliance review.

The decisive finding is a **P0 Rule-118 blocker**: move 41 embeds video-selected Lane class `1` and rows `128–319` in free decoder code. TC4 inherits this and adds Movable class `3`. These parameters came from full-video census evidence and must be counted or replaced by an independently predeclared generic rule.

Other conclusions:

- Mixer weights and TC4’s selector mask are correctly stored in `archive.zip`.
- No new external dependency, scorer import, scorer weights, or derived scorer table was found.
- Move 41’s measured score remains `0.13758600733559048 [contest-CUDA T4 n600]`, but its 1,336.669-second inflate fails the live 1,260-second seal margin.
- TC4 timed out at 1,800 seconds and has no measured score or move 42.
- No contest-CPU receipt exists; the ordinary CPU entry point currently refuses to run.
- The shipped move-41 predictor uses float64 geometry, so integer cross-host determinism was not proved.

Artifacts:

- [Compliance review](/Users/adpena/Projects/pact/.omx/research/ddm_pr8_receiver_code_compliance_review_20260910.md)
- [Machine-readable evidence](/Users/adpena/Projects/pact/.omx/research/ddm_pr8_20260910/review_evidence.json)
- [Landing patch](/Users/adpena/Projects/pact/.omx/research/ddm_pr8_20260910/landing.patch)
- [Verified fallback bundle](/Users/adpena/Projects/pact/.omx/research/ddm_pr8_20260910/final_serializer/20260910T073547.155021Z-95456/intended-commit.bundle)

Git-object writes were sandbox-blocked, so nothing was committed. The verified fallback commit is `3606a65c66b2b33c9b1a5310c053871ac5f2b17b`; the shared index remained untouched.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: `.omx/research/ddm_pr8_20260910/`; fire trigger: harvest this review. Withdraw the current receiver runtimes from PR/submission packets while adjudicating the internal pointer separately.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `receiver-rule118-cure`; consumer store: `/Volumes/VertigoDataTier/pact/ddm_tc3_rule118_counted_config_cure/`; fire trigger: MAIN assigns the lane. Put content-selected class IDs and row bounds in the counted rider, then rebuild and revalidate.
- **FOLDED** — owner: `ddm_dwc1`; consumer store: `.omx/research/ddm_dwc1_20260910/`; fire trigger: a Rule-118-clean receiver is byte-closed. Optimize selected-map execution and require T4 inflate at or below 1,260 seconds.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: `/Volumes/APDataStore/pact/ddm_tc3_cpu_axis_cure/`; fire trigger: Rule-118 and T4-margin gates pass. Produce a conforming exact contest-CPU n600 receipt.

## LIVE-HYPOTHESES

- Counting the semantic IDs and row bounds should cost only a few bytes, preserving most of move 41’s 79-byte gain.
- Fixed-point geometry may remove the remaining cross-host coding risk without losing the rate improvement.
- TC4 may recover substantial time by computing only mask-selected maps, though move 41’s inherited margin deficit must also be removed.

## DEAD-ENDS

- “No per-frame fitted table” is not sufficient Rule-118 compliance; fitted scalar choices are content too.
- The shipped predictor is not integer-only; move 41 selects the float64 branch.
- macOS advisory identity cannot stand in for contest CPU.
- TC4’s projected `0.13749744809482524` is not a score.
- Adding more per-token context work before reducing the inherited timing wall is closed under the current seal policy.