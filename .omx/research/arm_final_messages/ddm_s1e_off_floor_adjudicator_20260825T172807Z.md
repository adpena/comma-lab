Implemented, executed, and committed the S1E adjudicator.

- Commit: `70009a6d0729077d042c2762b459578493c6a725`
- Instrument: [s1a_off_floor_adjudicator.py](/Users/adpena/Projects/pact/tools/s1a_off_floor_adjudicator.py)
- Evidence memo: [ddm_s1e_off_floor_adjudicator_20260825.md](/Users/adpena/Projects/pact/.omx/research/ddm_s1e_off_floor_adjudicator_20260825.md)
- Retained receipt: [seed_20260815_positive_control.json](/Volumes/APDataStore/pact/ddm_s1e_off_floor_adjudicator/seed_20260815_positive_control.json), 120,494 B, SHA-256 `3a6dae41b141e1fe6e41f3f5eba8f88072fd3b1a04123c87fbc6f043ecfa94da`
- Verification: 12 tests passed, lint passed, and both Python files have two consecutive clean review passes. The staged index is empty; unrelated worktree changes were preserved.

Measured `[Darwin-mps frozen-scorer advisory]` result:

- Seed 20260815 completed all 14 retained checkpoints through epoch 65.
- All 14 refused the renderer corner; zero crossed.
- Closest point: epoch 65, composed ΔS `+0.1554134085557307`.
- Real controller packet: 38,847 B, meaning signed bytes shed versus GB1’s 30,856 B renderer is `−7,991 B`, a `+0.005320878894399272 S` penalty.
- Pose damage exceeded segmentation damage at 14/14 checkpoints; the preregistered pose-binding prediction was not falsified.
- Global verdict is `INCOMPLETE_DATA` because seed 20260816 had not yet produced its controller result.
- `score_claim=false`; this is neither a population verdict nor a submission candidate.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_s1e_off_floor_adjudicator/both_off_endpoint.json`; fire trigger: seed 20260816 has complete `STAGE_CONTROLLER_RESULT.json`, `wd3_epoch_0065.pt`, and `epoch_0065_n60.json`, before authorizing ON seed 20260815.

## LIVE-HYPOTHESES

- The independent 20260816 seed may confirm or falsify the pose-binding ordering; seed variation is the remaining preregistered test.
- Stage A may still help Stage B despite refusing the standalone corner, because its useful output is trained distortion structure while Stage B must supply the missing rate-representation cut.
- An eventual exact n600 Stage-B candidate may differ from these n60 observations because the current rows are fixed evenly strided and MPS-advisory only.

## DEAD-ENDS

- Projected or hand-entered renderer bytes are closed: the real selected packet is 38,847 B.
- Seed 20260815 as a corner-crossing OFF-floor instance is closed: every measured checkpoint had positive composed ΔS.
- Segmentation-first compensation priority is closed for this seed: pose damage was larger at every checkpoint.
- Treating these n60 rows as prefixes or population evidence is closed: their pair IDs are `0,10,…,590`, and their axis remains non-authoritative.
- Treating Stage A as a submission candidate is closed: it is only an input to Stage B.

Own-vehicle frontier remains **gb1 — S 0.14811799921260607 @ 180,215 B `[contest-CUDA T4, n600]`**.