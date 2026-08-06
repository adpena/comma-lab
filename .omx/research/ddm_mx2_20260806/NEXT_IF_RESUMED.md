# ddm_mx2 Next If Resumed

1. Re-read `RECEIPT.md`, `REPACK_RACE.md`, `PARITY.md`, and `LAUNCH_TICKET.md`.
2. Do not run n600 scorer work unless et4 has released or assigned the scorer slot.
3. Build the missing adapter for our master surface and target cache:
   - target cache with `seg` and `pose` keys;
   - master surface uint8 `(600, 3, 874, 1164)`;
   - provenance hashes for source archive/runtime/component selection.
4. Add a real resume path for `train_pose_carrier_full.py` before any long run. Latest/best result saves are not sufficient.
5. On MAIN, first run a tiny real-frame forward/parity smoke with batch shape pinned. Do not promote it to a verdict.
6. If MAIN smoke passes, run stratified n32 then n120 fit gates; never use a contiguous prefix.
7. Only after a real carrier fit exists, re-run `tac.pr130_lift.pose.repack_race` against the new carrier section and apply CPR1 only if the section is PR130 legacy-carrier-shaped.
8. Append a new receipt with measured/not-measured status and pointer-delta honesty.
