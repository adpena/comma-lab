# DDM VO1 - Next If Resumed

## First Check

1. Re-read `.omx/state/main_hot_state.md` and the lane/owner state before any
   scorer or launch decision.
2. Confirm `ddm_et2` has released or explicitly yielded the scorer slot. If it
   has not, continue only scorer-free wiring or ledger work.
3. Re-read `.omx/research/ddm_vo1_20260806/REOPEN_LEDGER.jsonl`; do not route
   from memory alone.

## Fire Order

1. `et1_phase_field_q3_projection_and_pose_finish`
   - Scorer-free first: route the existing ET1 phase-field target through SW1
     solve-within and DK1 lattice-native realization.
   - Required local receipts: nullspace/constraint certificate, integer
     realizer leakage, exact input artifact hashes, and no cap-stopped solve.
   - First scorer use only after slot release: declared non-prefix n32 Seg/Pose
     check with eta denominator and pose ratio.
   - Byte-close only if eta clears the ET1 break-even after terminal pose
     repair or joint descent.

2. `q3x_q3_convergence_realizer`, `q31_q3_constrained_solve`, and
   `sq1_pose_null_constrained_paint`
   - Replace float-first/project-after realization with the DK1/SW1 instrument
     class before any new verdict.
   - Add cap-stop receipts. If the row still stops by iteration cap, label it
     as cap floor, not convergence.

3. `fd1_fd2_zero_accept_integer_near_margin`
   - Do not reopen by pose-null projection alone.
   - Build integer-aware near-margin proposals or a realized argmax-in-loop
     proposal path, then rerun a bounded acceptance window.

4. `lc1_pe3_label_filter_regrade`
   - Build a scorer-free per-record positive-net filter from LC1 confusion
     deltas.
   - Test filtered/conditioning PE3 only, not aggregate static-label
     substitution.

5. `rl1_road_lane_realization_half`
   - Do not reprice the coder first.
   - Build a receiver/parseback realization for the Lane-mask crop and then
     decide whether scorer spend is justified.

## Stop Rules

- Stop if scorer ownership is active elsewhere and the next action requires
  scorer.
- Stop if a candidate still depends on cap-stopped convergence.
- Stop if a proposed reopen is only a rerun of the same naive instrument.
- Stop and record a bounded absence if the needed source artifact cannot be
  found in the cited scope.

## Output Requirement

Any resumed unit should append a new receipt beside this directory or create a
successor directory. It must state pointer delta honesty and whether it produced
an exact score. VO1 itself produced no exact row.
