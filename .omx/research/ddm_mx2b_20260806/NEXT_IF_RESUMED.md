# ddm_mx2b Next If Resumed

Resume from this state, not from the original blocker.

## Current State

- Target cache exists: `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/inputs/gt_pose_cache_600.pt`
- tq1c master cache exists: `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/master_cache/OUR_SURFACE_MASTERS.pt`
- Cache validation PASS: `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/inputs/cache_validation.receipt.json`
- Resumable wrapper exists: `src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py`
- Smoke PASS for n=4 scope: `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/smoke_tq1c_pose_resumable/resumed_step2.json`

## Fire Order

1. Re-read `.omx/state/main_hot_state.md` and confirm ET4/scorer-slot boundary.
2. Claim the lane with `tools/claim_lane_dispatch.py` before dispatch.
3. Verify cache shas against `RECEIPT.md`.
4. Run the Row-2 tq1c command from `LAUNCH_TICKET.json`.
5. If interrupted, rerun the same command with the `resume_command_addition`.
6. Do not use `--smoke-pairs` for the real tq1c fire.

## Held Boundary

The mx1-arm is not fireable until Row-1 emits a selected renderer surface. Build
an MX1 master cache first, then clone the tq1c command shape with the selected
MX1 surface path.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
