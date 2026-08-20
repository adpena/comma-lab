The WD3 build apparatus is complete and committed as `effd8ff4ef`. No scorer, Metal, cache, training, candidate evaluation, or frontier-moving run was launched.

Implemented:

- Real adaptive sub-int16 `WD3Q` packet and receiver consumption.
- Paired receiver → uint8 chain → SegNet/PoseNet training path.
- Repeated retained n600 teacher/scorer cache builder.
- Adaptive dual objective, per-cell/edge selection, quantization re-score, and hard admission gates.
- Atomic ResumeRegistry checkpoints and scorer-free D56/F64 birth checkpoints.
- Typed fire-order compiler and retained-payload enforcement.

Verification:

- 31 WD2+WD3 tests passed.
- Ruff passed.
- Two review-tracker passes recorded for all three Python files.
- P0 payload-retention detector: 0 findings across the changed Python scope.
- Scorer-free verification retained 3,569 B uniform and 3,570 B adaptive test packets. These are apparatus packets, not candidate rates.

Artifacts: [handoff](/Users/adpena/Projects/pact/.omx/research/ddm_wd3_scorer_aware_width_distillation_20260815/HANDOFF.md), [fire order](/Users/adpena/Projects/pact/.omx/research/ddm_wd3_scorer_aware_width_distillation_20260815/FIRE_ORDER.json), [build receipt](/Users/adpena/Projects/pact/.omx/research/ddm_wd3_scorer_aware_width_distillation_20260815/build_v1/BUILD_RECEIPT.json).

## NEXT_IF_RESUMED

- Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/teacher_scorer_cache/`; fire trigger: re-verify PID 63183 absent, claim distinct live scorer and Metal lanes, compile the cache config, and explicitly authorize launch.
- Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/W0/`; fire trigger: byte-identical cache receipt lands, then run W0 preserved-state followed by its magnitude-matched reset control.
- Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/`; fire trigger: both W0 verdicts land, then materialize typed birth checkpoints and run D56 followed by F64.
- Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/`; fire trigger: enter W96 only on measured smaller-arm capacity pressure, or fresh only after both W0 arms are seeded-n120 instance negatives.
- Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/admission/`; fire trigger: a retained receiver-closed candidate passes the same-instrument admission gates, then run n600 and an exact contest axis.

## LIVE-HYPOTHESES

- Real-chain scorer-aware continuation may exchange WD2’s 17,372 B saving for an admissible distortion increase because the retained student’s original failure was trained against camera MSE rather than contest geometry.
- Adaptive per-cell packet depth may retain Road-hub boundary decisions more cheaply than uniform int4 because allocation is driven by actual score gradients and selected only after retained naïve re-scoring.
- The W0 reset control may separate objective failure from inherited optimizer-state failure because #816 measured a 3.16×–6.57× zero-moment excursion.
- A near-admission student may benefit from QS2/QS5 surgical finishing if its measured residual is localized on Road-hub edges and a receiver-consumed Schur compensation receipt holds pose below base.

## DEAD-ENDS

- Fixed ancestor `kd_w` or pose-distillation weights are closed: they are vehicle-specific and violate WD3’s adaptive-dual contract.
- Prefix or n60 negatives are closed: those populations are biased and lack negative authority.
- Projected packet/archive sizes cannot select an allocation; only retained, parsed, real-coder re-scores qualify.
- Dense W96 is closed absent a real-coder override because its current projection exceeds the base.
- Launching from this arm is closed: authorization and WD3-owned live lanes are absent. Vehicle frontier unchanged: **S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]**, archive SHA `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.