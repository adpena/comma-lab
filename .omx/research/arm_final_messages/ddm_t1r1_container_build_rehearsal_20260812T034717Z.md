The T1 whole-container rehearsal is complete and committed as `45dfd8ed6d`.

Key results:

- Rehearsal archive: **187,046 B**, SHA-256 `12a5b181…ece80`
- Delta versus CP135: **+794 B**
- Real C1-on-HP3 RC64 stream: **115,237 B**, only **+6 B** versus CP135
- Repeat archive: byte-identical
- Receiver parse-back: **PASS**
- Shipped RC64 decode: **11.844 s**, 0.658% of the 30-minute budget
- All 117,964,800 event-order and spatial tokens restored exactly
- Corrupt-archive and mismatched-member controls both refused
- Retained custody: 1,398 files totaling 1,775,757,636 B

Prediction m38 was confirmed: no wired C1-on-CP135 whole-container adapter existed. The new [rehearsal adapter](/Users/adpena/Projects/pact/experiments/ddm_t1r1_container_build_rehearsal.py) closes that gap. The full evidence is in the [rehearsal memo](/Users/adpena/Projects/pact/.omx/research/ddm_t1r1_container_build_rehearsal_20260812.md) and [parse-back receipt](/Volumes/APDataStore/pact/ddm_t1r1/retained/70_PARSEBACK_RESULT.json).

Validation: 18 focused tests passed; Ruff, compilation, diff checks, review policy, and commit hooks passed. The broader developer preflight still has 8/25 pre-existing repo-wide failures, recorded in the memo.

No scorer, evaluator, GPU, or lane dispatch ran. This is not a candidate because it contains stale pass-4 pose bytes. The effective floor remains CP135 at **S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600], ours**. The own-vehicle frontier remains LC2 at **S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: HY1/js1 whole-container builder. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/`. Fire trigger: ps135 emits its terminal safe-run receipt and same-parent pose carrier; replace the sole stale CPR1 input and rerun `all` before requesting a scorer lane.**

## LIVE-HYPOTHESES

- The terminal pose section may fit at or below the stale 22,934-byte carrier section because it uses the same compact coefficient family.
- The composed object may clear the HY1 realization gate if the renderer preserves at least 82.824% of C1’s solved-plane Seg gain.
- Full inflate likely remains within 30 minutes given the 11.844-second shipped token decode, but the exact terminal archive still requires governed full rendering.

## DEAD-ENDS

- Adding the F26 `+11 B` proxy to CP135 is closed; the real HP3 result is `+6 B`.
- Gluing C1 without re-encoding is closed because CP135 and F26 use different probability objects.
- Treating the ExperimentBook RC64 file as the shipped backend is closed; only the shipped decoder’s full-symbol receipt is accepted.
- Re-running ANS on this exact HP3 state is closed at INSTANCE scope; it was already 9 B worse.
- Assigning this rehearsal a score or candidate status is closed because pose is stale and no scorer or full render ran.

