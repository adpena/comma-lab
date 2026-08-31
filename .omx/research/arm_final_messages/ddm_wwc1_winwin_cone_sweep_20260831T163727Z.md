Executed and landed as commit `bd7c651b0a`.

Key results:

- FCD1 control reproduced exactly: selected union `B=5,268 / H=0 / W=0`; raw disagreement pool separately recorded as `5,268 / 221,862 / 541`.
- All five OE1 rungs share FCD1’s exact coding-argmax and benefit coordinates. Each saved `3,756–4,073 B`, repeated byte-identically, and decoded exactly.
- Joint FCD1∪JF2∪OE1 re-encode: `178,792 → 174,609 B`, `−4,183 B`, rate-only `ΔS=−0.002785288000910043`, archive SHA-256 `06cae23e…abe1f`.
- The joint archive is 27 B larger than JF2-only. Thus the extra 467 FCD1/OE1 cells do not compose additively.
- No scorer was run. `d_seg`, `d_pose`, and net score for the joint archive remain unmeasured. Existing FCD2/FCD3 evidence already refused this token-GT edit class on Pose and Seg, so no fire order or seal was issued.
- Frontier remains unchanged.

Artifacts: [research memo](/Users/adpena/Projects/pact/.omx/research/ddm_wwc1_winwin_cone_sweep_20260831.md), [resumable runner](/Users/adpena/Projects/pact/experiments/ddm_wwc1_winwin_cone_sweep.py). The Vertigo manifest covers 1,662 retained artifacts totaling 2,094,159,396 bytes. Two Python review passes, payload-retention checks, deterministic repeats, receiver checks, and commit hooks passed. Unrelated worktree changes were preserved.

## NEXT_IF_RESUMED

- `QUEUED-CONDITIONAL-NEW-FORMULATION` — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_wwc1_winwin_cone_sweep/followons/scorer_native/`; fire trigger: a chartered same-object scorer-native selector using realized Seg/Pose sensitivities and fresh in-compile compensation—not another token-GT cone sweep.

## LIVE-HYPOTHESES

- A scorer-native selector may retain the repeatedly measured RC64 savings while avoiding the token-label transfer failure. This is plausible because the favorable rate direction survived DX2, JF2, and five OE1 trajectories, whereas the failures arose at the token-GT versus realized-scorer boundary.

## DEAD-ENDS

- Re-scoring the unchanged 5,268-cell token-GT cone: FCD2 failed Pose at 42.961687× base; FCD3’s pose-safe subset worsened Seg and net score.
- Treating OE1 as independent semantic confirmation: its argmax and all selected coordinates are exactly FCD1’s.
- Adding FCD1/OE1-only cells to JF2 for rate: the joint archive is 27 B worse than JF2-only.
- Separate DG2 replay: its rows are contained in JF2.
- LD1 family-specific cone: 14 cells cost +1 B.
- AE1 substitution: no physical final RC64 candidate/coding-argmax object exists.

[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25.