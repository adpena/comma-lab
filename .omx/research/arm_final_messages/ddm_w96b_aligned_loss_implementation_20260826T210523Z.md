Implementation is green; the aligned run remains blocked only by storage.

- Exact CE1 expected-flip margin law, tau `0.15→0.05`, 1% cosine floor, step-zero pose, strict resume identity, DSL selection, and lossless CAS retention landed in commit `3d9e021d07f6513be48ecbf768c40a5c4770957e`.
- Real 28-tree matched-window demand after dedup: `15,572,930,560 B` evaluations; `24,979,443,712 B` complete two-seed demand.
- This exceeds the recorded available denominator by `2,660,372,480 B`; with the 8 GiB reserve, required space is `33,569,378,304 B`.
- Tests: 49 passed. Ruff and developer preflight passed. Exhaustive preflight was environment-blocked because the sandbox forbids `ps`; no release or score claim was made.
- No training, scorer, Metal, Modal, n600, or contest evaluation ran. The unrelated `ddm_bs4` file remains untouched.

Artifacts: [implementation receipt](/Users/adpena/Projects/pact/.omx/research/ddm_w96b_aligned_loss_implementation_20260826.md), [storage receipt](/Volumes/APDataStore/pact/ddm_w96a_aligned_window/W96B_BUILD_AND_STORAGE_RECEIPT.json), [sealed fire order](/Volumes/APDataStore/pact/ddm_w96a_aligned_window/SEALED_FIRE_ORDER_W96B.json).

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN/operator; consumer store: `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`; fire trigger: #1165/pk4 certified reclaim raises AP free space to at least `33,569,378,304 B`.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store: same AP root; fire trigger: storage is green, fresh scorer/Metal claims exist, and the launch-disabled configs are resealed and validated; run seed `20260815`, then `20260816`.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store: same AP root; fire trigger: each checkpoint lands; run the matched S1E n60 screen.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store: same AP root; fire trigger: a checkpoint achieves `composed_delta <= matched OFF / 5` with sealed payload/config SHAs and an idle claimed n600 scorer lane.

## LIVE-HYPOTHESES

- The exact aligned loss may deliver the ≥5× screen improvement: CE1 showed strong gradient-alignment gains, but this remains untested on the W96 vehicle.
- Step-zero joint pose supervision may avoid the pose-dominated failures seen in prior W96 diagonals because pose shapes every update.
- Lossless compression inside CAS objects may reduce unique scorer-array demand further; the float-logit and repeated-frame structure makes this plausible, but it has not been measured.

## DEAD-ENDS

- Renaming WD3’s calibrated target-probability loss “aligned” is closed: it executes a different law.
- Evaluation-tree dedup alone clearing storage is closed: real demand remains `2,660,372,480 B` above the recorded denominator before reserve.
- Local fallback, symlink storage, or discarded payloads remain forbidden.
- The 35 OFF rows do not close the aligned formulation; they use a different loss.
- Higher LR as the ancestor cure and SVD-r32 remain closed by prior evidence/directive.

**Own-vehicle frontier: S = 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600], GB1 archive SHA `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4` — UNMOVED by ddm_w96b.**