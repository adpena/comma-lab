Verdict: **RERUN-REQUIRED.**

The completed review is [ddm_pr10_second_family_review_quiesced_timing_rule_20260910.md](/Users/adpena/Projects/pact/.omx/research/ddm_pr10_second_family_review_quiesced_timing_rule_20260910.md), SHA-256 `013a81df19836da670855a496fac5186f9bc04b16da6e5a8200e7f48398bef3d`.

The three strongest findings:

1. The sampling-time quarter-core rule refuses both attempts—counts 3 and 4—while the post-data hybrid rule changes attempt 1 to ADMIT count 0 and attempt 2 to REFUSE count 1. Any outside-stage threshold in `(95.9,124.2]`, including the selected 100%, creates that exact split.
2. The 5% stage-listing threshold sits above observed ordinary jitter, but the 1% own-median rule is not a contention bound. Only 62.00%/62.77% of the walls are checkpoint-covered, and sustained slowdown disappears into each run’s own median.
3. Attempt 2 is refused solely by ChatGPT at 124.2% in `during_0001`. Its two 100% Python processes, 98.3% `tar`, and 98.2% `git` overlap stage 75 and are admitted. The retained claim that those processes occupied native build is not supported.

The required replacement is frozen before launch: 25% individual threshold over the entire non-settle wall, 25% ancestor cap with `>=`, 200% aggregate cap derived from two spare P cores, three consecutive quiet 20-second settle samples, stage rates diagnostic-only, and no assembly-time threshold override. The raw receipt must bind `admission_rule`, its SHA-256, freeze time, and producer start time.

The serializer could not write the read-only Git object store. It produced and verified fallback commit `7fd96dabe821dc69f189c40f1569c78e5f1d4df2`; bundle SHA-256 `b2a19537d10b2a4ce2490ae45d2da850a8ca3f71ec63172984750b98c3e5a82c`. The shared index stayed empty and `HEAD` remained `675fcb53add0814cf2a29468bed39ed7cc80682a`.

Boundaries:

- Review-only: no source, test, timing receipt, checkpoint payload, sidecar, pointer, upstream file, or SSD payload changed.
- Only the memo and charter-mandated progress-log append were written.
- No decoder, producer, scorer, evaluator, dispatch, or timing rerun was launched.
- `%CPU` is a decaying `ps` average, not instantaneous P-core residency, frequency, I/O, memory-bandwidth, or thermal evidence.
- Timestamp reconstruction is second-granular; the instrument’s ±20-second overlap padding makes the cited large overlaps stable, but it is not nanosecond custody.
- Both runs share one host and day, with no independent clean replicate; jitter values are descriptive only.
- The 5% threshold is cleared only for diagnostic listing. The 1% admission threshold is not cleared.
- Attempt 2 is not classified as clean, and its shorter wall is not a clean-baseline estimate.
- Current RLC1 `g3/g4` receipts were outside the charter’s evidence set and were not used to tune this judgment.
- The validator census covered 330 tracked top-level validator symbols; it found no additional confirmed no-door case, but this is bounded absence—not a global result.
- Move 40’s exact score and directly measured 990.053829427-second T4 time remain intact. Only the local quiescence, calibration ratio, and inheritance authority are refused.

`composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40)` unchanged.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_pr10_second_family_review/final_serializer/20260910T110317.523503Z-28818/`; fire trigger: Git-writable MAIN context after bundle SHA verification. Land fallback commit `7fd96dabe821dc69f189c40f1569c78e5f1d4df2`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / ddm_dwc1; consumer store: `.omx/research/ddm_rlc1_20260910/TIMING_FIRE_ORDER.json`; fire trigger: ddm_pr10 landing before RLC1 `g5`. Implement the frozen-rule receipt binding and whole-window thresholds.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / ddm_dwc1; consumer store: `.omx/research/ddm_rlc1_20260910/TIMING_FIRE_ORDER.json` and retained timing receipts; fire trigger: cured instrument plus a quiet no-commit/no-spawn/no-pytest window. Establish RLC1 timing through a matched clean calibration or direct exact contest-T4 timing.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: RLC1 candidate seal and canonical evaluation ledger; fire trigger: exact runtime/archive binding and decode clearance at or below 1,260 seconds without the refused ratio. Authorize evaluation only after those gates clear.

## LIVE-HYPOTHESES

- A precommitted strict rule can obtain a usable clean calibration if the macOS activity jobs finish and all control planes remain idle; both attempts eventually observed one quiet sample, though never the required three-sample window.
- RLC1 may still clear 1,260 seconds because the unreviewed `g3/g4` local walls are near 830 seconds, but this remains unproved without a clean ratio or direct T4 result.
- Direct contest-T4 timing is likely the shortest sound route because it removes the contaminated local denominator entirely.

## DEAD-ENDS

- Reassembling attempt 1 under another threshold is closed; retrospective classification is the defect.
- Treating a run’s own median as a clean counterfactual is closed because uniform slowdown is normalized away.
- Using the 5% diagnostic cutoff to validate the 1% admission rule is closed.
- Calling attempt 1 quiesced is closed: its raw receipt says count 3 and `quiesced=false`.
- Placing attempt 2’s Python/git work in native build is closed by the checkpoint/sample epoch mapping.
- Using the 0.7 safety factor as evidence that the calibration is valid is closed.
- Withdrawing move 40’s exact score or actual T4 timing is closed; neither was falsified by this review.