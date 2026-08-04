# ddm_tj1 recall inventory

Date: 2026-08-05. Arm: `tj1`. Scorer use by this arm: `0`.

Charter target: trajectory-derived stopping plus adaptive recursion depth, using
recorded trajectories only. Score claim: `false`. Pointer moved: `false`.

## RECALL EVIDENCE

| surface | bounded read | reuse decision |
|---|---|---|
| `#188` trajectory-dynamics early-stop | `src/tac/witness_dsl/campaign.py` implements stage-level `decide_next_stage`: ADVANCE/EXTEND/RERUN/ROLLBACK from trailing d_seg slope. | Reused the pure-trajectory, deterministic-decision shape. Not reused as-is: it is stage/curriculum policy, not solver-step stopping. |
| `#344` NCDE hit-to-solve detector | `src/tac/witness_control/ncde_trajectory.py`; `ddm_eg1_endgame_chain_20260728.md` says shadow-only and actuation `NONE`. | Not a stopping actuator. Kept as recall evidence only. |
| `#216/#475` saddle/grokking guards | `src/tac/witness_control/ddm_endgame_policy.py`; `src/tac/ddm_costate_organ.py`; EG1 says #475 is scoped to a fixed 31-feature chart and has no stage-advance authority. | Not a cap-to-convergence rule. Kept the guard discipline: labels can prioritize quotes, not declare convergence. |
| `#341/#342` terminal-head / GN solve economics | EG1 and NG1 say historical GN cap premise is stale; current terminal pose GN is cured/off live chain, with marginal-value floor as the live recipient. | Wired the shared law into `terminal_pose_gn` only when the marginal floor is positive. `marginal_value_floor=0.0` remains the no-gate limit. |
| `#302/#686` event continuation | `src/tac/optimization/ddm_event_continuation.py` treats budgets as safety caps, not stage lengths. | Adopted the same taxonomy: cap binding emits `safety_bound_REPORTED`, never convergence. |
| NG1 cap-artifact sweep | `.omx/research/ddm_ng1_20260805/cap_artifact_sweep.jsonl` has six matched cap rows. | Consumed for class map and fire-order routing. |

## Implemented Law

New executor: `src/tac/optimization/trajectory_stopping.py`.

Typed stop reasons:

- `converged_projected`
- `marginal_below_bar`
- `safety_bound_REPORTED`
- `continue_projected`

The law fits geometric and power-law tails when they pass the R2 floor; otherwise
it falls back to a local last-k slope. The caller supplies the exchange rate into
contest S units. For SQ1, the unit conversion is one SegNet flip over the 600-pair
384x512 lattice and the marginal bar is one counted archive byte per solver step.

## Positive Controls

Receipt: `.omx/research/ddm_tj1_20260805/trajectory_replay.json`.

| control | measured eta | stop reason | interpretation |
|---|---:|---|---|
| SQ1 prefix 25 | `0.7895095948827292` | `safety_bound_REPORTED` | floor, not convergence |
| SQ1 full 50 | `0.8620042643923241` | `safety_bound_REPORTED` | improved floor, still not convergence |

Prefix-25 projection to step 50 predicts eta
`[0.7967534799344995, 0.8643476079878595]`; measured step-50 eta
`0.8620042643923241` is inside the interval.

SQ2 validation target is persisted in the same JSON receipt. At inspection time
the SQ2 receipt was partial (`21/32` rows), so no complete SQ2 verdict is claimed.

## Canonical Equation

Registered equation id: `trajectory_derived_stopping_law_v1`.

Registration receipt:
`.omx/research/ddm_tj1_20260805/canonical_equation_registration.json`.

The registry row is research-signal, non-promoting, and points to
`tac.optimization.trajectory_stopping:evaluate_trajectory_stop`.

## Frontier Honesty

Own-vehicle frontier line remains
`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.
Contest pointer remains borrowed/unmoved at `S = 0.1910828242`.
