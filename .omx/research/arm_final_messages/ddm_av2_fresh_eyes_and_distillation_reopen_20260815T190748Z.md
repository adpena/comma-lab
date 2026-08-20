Audit and artifacts are complete; only the Git landing is blocked.

Same-instrument results:

- WD2 `Δd_seg = +0.00074963`, or 7.0059× the allowed cap.
- Pose MSE is 623.7625× the hv1 base.
- Saving 17,372 B contributes `-0.0115673 S`, but total `ΔS = +0.9840885005`.
- The ep60 decode-MSE-only instance remains refused. The earlier 8.2×/634× cross-axis figures are superseded.
- The “160+ epochs” forecast is unsound: there is no measured mapping from decode loss to scorer quality.
- The advisory stamp is caused by `uv_group_not_declared`, not an observed package mismatch.
- Mirror attempt 2 used post-sweep SHA `fa7c4bf5…`, not birth SHA `d5bb36a2…`.

Artifacts:

- [AV2 audit memo](/Users/adpena/Projects/pact/.omx/research/ddm_av2_fresh_eyes_and_distillation_reopen_20260815.md)
- [WD3 charter](/Users/adpena/Projects/pact/.omx/research/charters/ddm_wd3_scorer_aware_width_distillation_20260815.md)
- [WD3 declarative config](/Users/adpena/Projects/pact/.omx/research/ddm_wd3_scorer_aware_width_distillation_20260815.json)

WD3 is explicitly code-not-built/no-launch. It routes scorer-aware W0 continuation before wider models, derives loss weights from score arithmetic, validates through the real receiver/scorers, and retains every payload.

The serializer was invoked but Git object creation failed with `Operation not permitted`. No commit exists; HEAD remains `3c32549295ab15540e71a4adef383ae4c4988083`. The shared index is empty and the three files remain intact and untracked.

Vehicle frontier unchanged: **S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]**.

## NEXT_IF_RESUMED

- **Disposition:** BLOCKED-READY-TO-LAND. **Owner:** MAIN in a Git-writable session. **Consumer store:** repository HEAD. **Fire trigger:** Git object writes are permitted and artifact SHAs remain `12609910…`, `1a81a012…`, `d556b682…`. **Action:** rerun `tools/commit_autosha.sh` on exactly the three files above.
- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN. **Consumer store:** `/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/`. **Fire trigger:** landing complete, r5 PID 63183 exited, scorer/Metal lanes claimed, and real WD3 code/cache/resume/retention gates pass. **Action:** build/cache, then W0 preserved-state → matched-reset control → D56 → F64 → conditional W96.

## LIVE-HYPOTHESES

- Scorer-aware W0 may recover most of the rate prize because capacity has never been tested under the correct task-space objective.
- Dense d4/w56 may dominate factorized d4/w64/r19 because it has the smallest packet and changes inherited computation least.
- Factorized w96 may be the only wider arm worth its rate erosion.
- Preserved optimizer/EMA state may beat a fresh optimizer once the known reset spike is controlled.

## DEAD-ENDS

- Blindly continue ep60 for “160+ epochs”: no scorer-transfer law exists.
- Attribute Pose 0.092 to the advisory environment stamp: known seams are far too small.
- Start with dense w96: projected larger than hv1 before distortion costs.
- Use pure or fixed-weight KL: prior evidence rejects it and its weight does not transfer.
- Bank a verdict on a prefix or strided n60 controller set.
- Claim post-sweep mirror equality with the birth snapshot.
- Repeat the old 8.2×/634× cross-axis figures.
- Create a temporary detached commit and call it landed: it would not update the governed repository.

