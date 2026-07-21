# Canonical-equations note — Task #610 wrong-levels describe sweep

UTC: 2026-07-21T22:09:16Z

## Decision

**No new empirical equation is registered.** All three requested candidate
byte fields are `null`, so promoting a law from this outcome would turn missing
receiver custody into a numerical anchor. Formalization remains pending until
a parser-consumed candidate passes full n64 receiver admission.

## Existing laws consumed

1. `cgauge_master_action_v1`

   `S = 100 D_seg + sqrt(10 D_pose) + 25 L_MDL / 37_545_489`.

   The rate break-even derivative is `25 / 37_545_489` score units per byte.
   Candidate residuals below this measured marginal value stop.

2. `fullstack_unique_home_assignment_v1` plus the #503 recursive-fractal law

   Store a generator once at its lowest sufficient home. A boundary is a
   class-generator tie locus; a pixel is an argmax readout; a frame is a warp;
   the same `xi` is stored once and read by frame/pair/pose consumers. Therefore
   independent description diagnostics cannot be summed unless a section
   registry proves non-overlap.

3. `ego_motion_cumulative_se3_bspline_v1`

   The frozen PNTG rows are PoseNet targets, not `xi`. First bind named
   `PoseTargetEgoEstimator` or `xi_from_pose_calibration` channels and
   calibration. Then, with translation-first twists and fixed root `T_0 = I`,

   `T_{p+1} = T_p exp(xi_p)`,

   and a knot decoder must recover pair twists as

   `xi_hat_p = log(T_hat_p^{-1} T_hat_{p+1})`.

   The 600 pair twists thereby define 601 absolute poses. This is representation
   math only. The requested `D_pose` becomes measured only after a shipped
   pure-knot XIP2 parser feeds the existing `xi -> H -> RGB` realizer and the
   frozen PoseNet runs on the realized pair.

4. `perclass_stratum_residual_carrier_taxonomy_v1` and
   `segnet_head_rank4_linear_flipdist_v1`

   Per-class/stratum residual weights prioritize the packet search; a local
   head-space flip distance is `|margin| / ||w_c-w_c'||`. Neither equation
   supplies bytes or receiver realization. Lane normals are the largest and
   Lane owns 77% of stride-2-skip ablation flips, so Lane edge precision is
   high-EV, not free.

5. `lane_band_ego_factorization_source_reparam_v1`

   Lane phase/ground transport consumes corrected `xi`; it does not authorize a
   pure static chart or a second temporal predictor. BEV-v2's identity-only
   stored rotations leave calibration-versus-geometry unidentifiable.

6. `realization_breakeven_bytes_v1`, `resize_exploit_flip_fix_frontier_v1`,
   and the `range(A)`/null-projector family

   A chart coefficient is admissible only after exact uint8/range realization,
   real receiver scoring, and positive marginal action. Invisible resize-null
   components are projected out before entropy coding.

7. `shearlet_nterm_upper_bounds_task_rate_v1` and
   `cgauge_curvelet_parabolic_bank_v1`

   Any future boundary residual uses a localized curvelet/shearlet family under
   equal-byte receiver measurement. Fourier remains a governed control, not the
   default residual basis.

## Formalization trigger

A successor may register a Task #610 empirical law only when one artifact binds:

- exact candidate section bytes and parser/re-encode identity;
- semantic self-detection and unique-home attribution;
- corrected-trajectory two-frame RGB realization through `R`;
- full n64 Seg/Pose admission and a preserved checkpoint chain;
- deterministic fresh archive bytes and SHA-256;
- exact command, source/runtime hashes, seed, and axis.

Until then, the canonical statement is a blocker, not an equation:
`NO_VERDICT_RECEIVER_RATE_CUSTODY`.
