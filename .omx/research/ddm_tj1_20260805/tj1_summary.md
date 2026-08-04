# ddm_tj1 trajectory-derived stopping replay

Schema: `ddm_tj1_trajectory_replay.v1`. Law: `trajectory_derived_stopping_law_v1`.
Scorer forwards executed by this replay: `0`.

## Positive controls

- `sq1_prefix_25_from_cw1_50_step_receipt`: step `25`, stop_reason `safety_bound_REPORTED`, eta `0.789509594883`, marginal S/step `0.000100479827695`.
- `sq1_full_50_cw1_receipt`: step `50`, stop_reason `safety_bound_REPORTED`, eta `0.862004264392`, marginal S/step `4.02278563717e-05`.

## Prefix projection

Step-25 projection to step 50 predicts objective `[6786.048593, 8371.130896]` and eta `[0.796753479934, 0.864347607988]`.
Measured step-50 eta `0.862004264392`; inside interval: `True`.

## SQ2 target

Target compute `100.0` predicts objective `[5335.612490, 6841.252767]` and eta `[0.861993485407, 0.926199893805]`.
Status: `PENDING_COMPLETE_RECEIPT`. Partial rows present: `21/32`.

## NG1 Class Map

- `sq1_25_step_solved_paint` -> `cap_bound_floor_not_converged`; recipient `sq1 solved-paint loop`; fire-order `extend or waterfill before any convergence/promotion wording`.
- `sq1_50_step_uncap_cw1` -> `cap_bound_floor_not_converged`; recipient `sq1 solved-paint loop`; fire-order `extend or waterfill before any convergence/promotion wording`.
- `q31_50_step_q3_constrained` -> `cap_bound_floor_not_converged`; recipient `q31 Q3 constrained paint`; fire-order `extend or waterfill before any convergence/promotion wording`.
- `gn_pose_solve_850` -> `genuine_stop_or_stale_off_chain`; recipient `terminal_pose_gn marginal floor`; fire-order `FOLDED_STALE_OFF_CHAIN`.
- `et1_budget_ladder_eta_floor` -> `cap_bound_floor_not_converged`; recipient `ET1 budget ladder`; fire-order `extend or waterfill before any convergence/promotion wording`.
- `na3_lr2_solved_paint_ladder` -> `cap_bound_floor_not_converged`; recipient `NA3 solved-paint ladder`; fire-order `extend or waterfill before any convergence/promotion wording`.

## Frontier

Own vehicle: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.
Contest pointer: `S = 0.1910828242 [contest-CPU] borrowed/unmoved`.
Pointer moved by TJ1: `False`.
