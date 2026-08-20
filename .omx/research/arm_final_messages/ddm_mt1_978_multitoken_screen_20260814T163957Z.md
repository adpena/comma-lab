Completed and landed as commit `af56d51c48`.

The `[macOS-CPU advisory, stratified-random n32 heldout]` screen was negative:

- MT1 matched CP135 total Seg exactly: `1,529 / 6,291,456` errors, `d_seg=0.0002430280`.
- Road→Lane errors improved `306 → 297`, but collateral errors erased the gain.
- Pose worsened by `+0.0001029022` MSE.
- The parsed counted model is `1,270 B`; whole-container pricing was not measured.
- HC1 direct-C1 remained substantially worse at `2,588` Seg errors and `d_pose=0.0057630776`.

The [memo](/Users/adpena/Projects/pact/.omx/research/ddm_mt1_978_multitoken_screen_20260814.md), implementation, tests, lane registry, and audit records are committed. All payloads—3.62 GB—are retained under `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/`.

A hash-sealed T4 sign gate is queued but was not fired. No joint-train order was created. Verification passed: 5 tests, Ruff, compilation, payload-retention census, dependency closure, registry validation, and two clean review passes.

No n600 or exact contest evaluation was performed. The pointer did not move. Effective frontier remains CP135 `S=0.16195513827824176 @ 186,252 B`; own-vehicle frontier remains LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`.

The shared active-claims ledger contains the lane’s active and terminal rows but remains uncommitted because that file already contained extensive unrelated concurrent edits.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN sole Modal scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/`; fire trigger: the T4 scorer lane is free, the local claim is terminal, and every sealed request/source hash matches; action: execute `SEALED_FIRE_ORDER.json` and harvest `FINAL_RESULT.json`.
- **QUEUED-CONTINGENT** — owner: MAIN/#978 successor; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/short_joint_train/`; fire trigger: harvested T4 result reports `positive_t4_sign=true`; action: seal a separately claimed, resumable n120-train/n120-heldout joint train. Fold this action if T4 is non-positive.

## LIVE-HYPOTHESES

- CPU/CUDA sign reversal remains possible because prior identical-byte evaluations showed component-level axis drift; only the sealed T4 gate can resolve it.
- Joint pose-conditioned support may preserve the nine Road→Lane gains without the observed pose damage, but it is a separate formulation requiring a new claim.
- The queued ps135/HY1 probability-object base may expose better support geometry than CP135; this screen’s numerical result does not transfer to that base.

## DEAD-ENDS

- BG1 bilinear gating: closed by BG2’s negative heldout evidence.
- HC1 direct-C1 substitution: closed by both its prior exact row and this heldout comparison.
- Explicit changed-site transmission: forbidden by the charter and not a valid #978 implementation.
- The sibling pre-correction retained store and its T4 seals: superseded after finding a reversed Road→Lane loss orientation; preserve but never fire.
- CP135 hidden-4, max-support-mass-0.25 local simplex: locally closed because total Seg did not improve and Pose worsened; do not repeat this local screen while its sealed T4 confirmation is pending.