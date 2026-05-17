# Rudin Floor Remote Claim Lifecycle Regression - 2026-05-17

## Verdict

Current `main` already closes the stale Rudin remote-wrapper finding:
`scripts/remote_lane_substrate_rudin_floor_interpretable_ml.sh` refuses startup
without `RUDIN_FLOOR_DISPATCH_INSTANCE_JOB_ID` / `DISPATCH_INSTANCE_JOB_ID`,
verifies a live row through `tools/claim_lane_dispatch.py summary --live-only
--format json`, matches both `lane_id` and `instance_job_id`, and appends a
terminal claim row from the `EXIT` trap.

This landing preserves that fact as regression coverage rather than treating the
older audit note as still-live implementation work.

## Evidence

- Added `src/tac/tests/test_rudin_floor_remote_driver.py`.
- The test locks the following invariants:
  - startup requires an instance/job id and exits fail-closed when absent;
  - startup verifies an active dispatch claim before provider work proceeds;
  - the active-claim match uses both `lane_id` and `instance_job_id`;
  - successful and failed exits append terminal claim rows through the canonical
    `tools/claim_lane_dispatch.py claim --force` path.

## Score And Dispatch Status

- No provider dispatch was launched.
- No archive was built.
- No score claim, promotion claim, or lane retirement is made here.

## Next Use

Keep this as a guard against retreading the stale "Rudin only logs claim checks"
finding. The score-lowering path remains the L5/L5-v2 staircase: harvest paired
TT5L side-info cells, build the side-info effect curve, then only unlock
architecture lock / stack composition from paired CPU+CUDA evidence.
