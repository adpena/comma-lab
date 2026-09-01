AP reclaim succeeded. pfs1 is checksum-verified on Vertigo, its former AP path is a working symlink, and AP now has **52,713,881,600 bytes free**—stable across repeated samples. This meets the 50 GB decimal target and makes rxc1 gen-3 `READY`.

Key artifacts:

- [Execution memo](/Users/adpena/Projects/pact/.omx/research/ddm_vr2_ap_reclaim_round2_20260901.md)
- [MOVE_CERT](/Volumes/VertigoDataTier/pact/ddm_vr2_ap_reclaim_round2/MOVE_CERT_pfs1_20260901.json) — `6a5173ff…4fa7df`
- [Completion receipt](/Volumes/VertigoDataTier/pact/ddm_vr2_ap_reclaim_round2/MOVE_COMPLETE_pfs1_20260901.json) — `a5fdeb3e…0d40ae`
- Verified payload: 234 files, 51,417,783,170 logical bytes, full-checksum rc=0, zero differences.
- Reclaimed: 51,450,216,448 bytes.
- wwc1 and pk4 were folded as already-certified #1364 moves; neither was duplicated.
- No scorer, Modal job, evaluator, or `upstream/` mutation ran.

The serializer could not write the managed Git database. It produced a verified one-file [fallback bundle](/Volumes/VertigoDataTier/pact/anonymous/receipts/commit_serializer_fallbacks/20260901T160740.217556Z-4352/intended-commit.bundle), commit `4b1cf978d36e89a7e8fc777e544463da17027299`. The memo remains untracked in the shared worktree; the staged index is untouched.

Frontier unchanged: **AFR1 S=0.14797617125559104 at 180,002 B `[contest-CUDA T4 n600]`**.

## NEXT_IF_RESUMED

- `READY-TO-FIRE` — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`; fire trigger: AP free space rechecks at ≥1,400,000,000 bytes with no decline; run `.venv/bin/python experiments/ddm_rxc1_restartable_exact_coder.py --stage screen`.
- `BUNDLE_READY_MAIN_MUST_LAND` — owner: MAIN Git custodian; consumer store: Pact `main`; fire trigger: Git object writes become available or the operator imports the verified fallback bundle; land commit `4b1cf978d36e89a7e8fc777e544463da17027299`.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: storage apparatus maintainer; consumer store: `tools/vertigo_certify_move.py`; fire trigger: before its next source retirement; add a durable typed failure row around partial-retirement cleanup.

## LIVE-HYPOTHESES

- rxc1 can complete its remaining six rows without repeating the 26 sealed rows because its blocker records durable receipts and a frame-400 restart point.
- Current AP headroom should cover rxc1’s projected remainder and reserve, though concurrent fleet writers remain an external risk.
- The remaining certifier ambiguity can be closed locally around its unguarded final cleanup without altering successful-move semantics.

## DEAD-ENDS

- Recopying pfs1: current source/destination equality was fully checksummed and custody is complete.
- Re-moving wwc1 or pk4: both already have terminal hash-verified `MOVED_SYMLINKED` rows.
- Claiming 50 GiB: the exact result is 52,713,881,600 bytes, or 49.09 GiB; only the charter’s decimal target was met.
- Using pfs1’s old 307-row manifest as current equality proof: 73 rebuildable `.pyc` paths had disappeared before this move; equality was established between the current 234-file trees.
- Running rxc1 here: its fire belongs to MAIN’s scorer lane.

