The full-n600 row is **blocked before scoring**. The exact hash-pinned public receiver was executed, but its F26 runtime rejected the authorized local host with `F26 inflation requires a CUDA-capable GPU`. No candidate score, d_seg, d_pose, or ΔS was inferred.

- Archive and runtime hashes matched the charter.
- Failure log and extracted payload were retained.
- Blocker receipt rehashed 28 records and matched its checkpoint byte-for-byte.
- No T4 confirmation was earned; `score_claim:false`, pointer unmoved.
- Scorer lane released; active scorer claims: 0.
- Tests: 7 passed; lint passed.
- Commit: `05d356447e282f475f29a6bec2039a5fe826dbd4`.

Artifacts: [memo](/Users/adpena/Projects/pact/.omx/research/ddm_re1x_round1_full_n600_eval_20260813.md:1), [blocker receipt](/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/ddm_re1_20260813/full_n600_exact/round_01_singleton_best/RE1X_FULL_N600_BLOCKER.json), [resumable evaluator](/Users/adpena/Projects/pact/experiments/ddm_re1x_round1_full_n600_eval.py:1).

Own-vehicle frontier remains `S=0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`; this attempt did not move it.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-A-FIRE-ORDER`. **Owner:** `MAIN sole scorer-lane router`. **Consumer store:** `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/ddm_re1_20260813/full_n600_exact/round_01_singleton_best`. **Fire trigger:** the operator explicitly authorizes an exact CUDA public-front-door decode, or supplies its retained candidate `0.raw` and receiver receipt; then resume the matched local n600 SegNet pass and two PoseNet passes.

## LIVE-HYPOTHESES

- Round 1 may render identically to CP135 because every downstream physical component parsed back identically at equal rate. Only an exact CUDA public decode can confirm this.
- The probability edit may be receiver-null because archive bytes changed while decoded tokens and downstream components did not.

## DEAD-ENDS

- The pinned runtime cannot inflate on local CPU or Metal.
- Patching or monkey-patching it would invalidate the required runtime-tree hash.
- Substituting CP135 raw would violate the public-front-door requirement.
- Token identity cannot replace retained PoseNet validation or complete-S recomputation.
- Round 2 remains instance-closed because it adds one byte and is dominated for this handoff.