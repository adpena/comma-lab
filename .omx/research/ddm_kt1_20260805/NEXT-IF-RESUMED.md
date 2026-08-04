# KT1 next if resumed

1. Build the top transfer: wire terminal solvers to the scorer-free stop-policy surface in
   `src/tac/optimization/trajectory_stopping.py`.
2. Require terminal-solver receipts to persist consumed baseline/archive shas and a machine-readable
   `NEXT-IF-RESUMED` record before they can claim lane ownership.
3. Start from no scorer slot and no exact-eval claim; the first admissible follow-up is a code/test landing,
   not a measurement.

Current own-vehicle line at handoff: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.
Contest pointer remains borrowed/unmoved: `0.1910828242 [contest-CPU]`.
