# NA5 retained fire order

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: pose arm / MAIN scorer-slot owner. Consumer store: `.omx/state/probe_outcomes.jsonl` and OD1 Stage-2 pose-recovery adjudication. Fire trigger:** a source-faithful `pose_carrier_arms` reconstruction reproduces store_nothing 1.995, real-f0+witness-f1 10.42, and warp-real-luma 37.4 on the original n8 calibrated-ξ path within a declared repeatability floor; then run both sample IDs in `SAMPLES.json`.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: pose depth-recovery arm. Consumer store: `.omx/state/probe_outcomes.jsonl` and OD1 Stage-2 pose-recovery adjudication. Fire trigger:** recover or parity-rebuild the original L2 depth cache/harness and reproduce HPLAN_REAL 0.878, L2_REAL 1.296, and L2_WITNESS 171.8 on n24 before requesting a scorer slot.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: pose solve arm. Consumer store: `.omx/state/probe_outcomes.jsonl` and OD1 Stage-2 pose-recovery adjudication. Fire trigger:** recover or parity-rebuild the original A0/A2/A2+ solver, calibrated-ξ initialization, and acceptance rule before requesting a scorer slot.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: pose texture arm. Consumer store: `.omx/state/probe_outcomes.jsonl` and OD1 Stage-2 pose-recovery adjudication. Fire trigger:** recover or parity-rebuild the original texture/aperture grids and reproduce the n24 A1T grid plus n8 self-pair diagnostic before requesting a scorer slot.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN shared-ledger owner. Consumer store: `.omx/state/probe_outcomes.jsonl`. Fire trigger:** the unrelated dirty rows 663–670 land or receive separate custody; then patch-commit only NA5 rows 671–674 without absorbing sibling work.

## LIVE-HYPOTHESES

- Absolute pose walls may weaken on representative pairs because the n24 source prefix is 2.535× harder than the D2 population, while both prepared n120 samples are population-matched.
- The carrier-arm ordering may survive because its original inter-arm gaps are much larger than the measured n8 selection distortion.
- Per-dimension pose errors may reveal a scalar-selection confound concentrated in dimension 0.

## DEAD-ENDS

- Do not fire UB1 as an unchanged rerun; its store-nothing and ξ mechanisms differ from source.
- Do not infer formulation d_pose from the D2 selection ratios.
- Do not treat source positive controls as a repeatability noise floor.
