# TP1 birth A/B — matched-epoch read at ep946 (BUDGET-CONDITIONAL) + tail-slope adjudication [no-triality] [p0-ledger-ok]

Axis: [macOS-CPU/MLX advisory, realized training-gate d_seg — SAME gate subset both arms
so the A/B delta is clean; adoption numbers await composed n600]. score_claim=false.

## Matched-epoch endpoints (both 139 ep from window_02 parent ep807, seed 0)
| arm | ep945 gate d_seg | counted bytes | a1 @cap | window min |
|---|---|---|---|---|
| OFF | 0.004140218098958333 | 270,968 | FLAT | **0.003947 @ ep919** (then +1.9e-4 drift UP over final 26 ep) |
| ON (birth lane w0.35/a0.05/d1) | 0.004142478660300927 | 273,903 | FLAT | 0.004138 @ ep944 (endpoint IS min) |

Matched-budget verdict (BUDGET-CONDITIONAL per amendment 3): Δd_seg = +2.3e-6 (noise;
gate-metric σ ~1e-5–4.6e-5) at +2,935 B → ON ≈ OFF on distortion, ON +0.00195 S on rate.
The lever was ACTIVE (mid-window COUPLED_DESCENT + bytes present) — engaged, perturbed,
descended back to the same attractor. Scope: point-refutation of the UNDERIVED start
values ONLY (0.35/0.05/dilate-1); the birth family is NOT killed by this read.

## Tail-slope adjudication (amendment 3 — measured, not classifier labels)
Linear fits over the final 40/20 telemetry epochs:
- **ON: −6.3e-6 / −5.0e-6 per ep** (slope ≫ its σ 9.4e-6-tail noise floor over the fit
  window) → **CENSORED ENDPOINT, still descending**. Projected next-window gain −0.07 to
  −0.09 S if sustained (it won't be linear; the continuation MEASURES it).
- **OFF: +4.2e-6 / +9.7e-6 per ep** (ASCENDING past its own ep919 minimum) → its cap
  endpoint is PAST-plateau. OFF's best realized state = **intra_seg_trunk_tau_ep00919.npz
  (0.003947)** — preserved by the P0 intra-stage checkpoint rule, 0.019 S better than
  either endpoint. The late ascent is a finding (gate-metric oscillation or drift;
  the EMA shadow may differ — check at adoption).

## Actions taken
1. ON continuation w2 LAUNCHED (amendment-3 let-it-descend): sealed ticket
   `full_birth_lane_on_w2.json` hash aa2d7bbf693aabbaad6a00eb4d24057efa1025c0d4c50b440ac526b673e7f1f9,
   IDENTICAL lever stack (bi1 coverage-check guard makes the boundary re-seed a no-op;
   the amplify force continues the SAME descending system), ep946→1085, wall 130 min,
   warm optimizer state, resume ON stage final. pid 74163. Repeat while descending.
2. OFF: no extension owed (ascending), but adoption comparison MUST use its ep919
   checkpoint, not its endpoint. Converged-endpoint candidates at the routing boundary:
   {OFF@ep919 0.003947+270,968B-class} vs {ON@w2-plateau, TBD}.
3. jd1's joint window waits for the ON plateau; winning endpoint decided on CONVERGED
   states per amendment 3 (ticket regenerates against the winner).
