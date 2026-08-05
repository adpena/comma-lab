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

## ADDENDUM (MAIN, 2026-08-05 ~19:35Z) — w2 window COMPLETE, amendment-3 adjudication: EXTEND (w3 fired)

**w2 (ep946→1084, 138 eps, 83 min) endpoint = the BEST measured TP1 state:**
- **n600 full-confirm realized d_seg = 0.003883514404296875** (max-pair 0.011276) — below
  OFF@ep919's 0.003947 (the prior best) AND below the original burn plateau 0.0038892.
  The ON birth lineage, continued past its censored cap, now LEADS outright.
- Gate trajectory (36-pair gd1 HT design): first-10 mean 0.004152 → last-10 0.003997;
  window min = endpoint (0.003944 gate-basis) — monotone into the cap.
- **Tail-slope fit (the amendment-3 authority, overruling the a1 FLAT label):**
  tail-40 = −2.789e-6/ep at 6.2σ (REAL descent) · tail-20 = −1.048e-6/ep at 2.4σ
  (decelerating but above noise). Verdict: CENSORED again → one more warm window.
- One A1_REALIZATION_GAP_ALARM at ep1054 (smooth fell ≥2% while realized <0.5%) —
  logged, corroborates #824's η-transient surface; no decision rode on it.
- jd1 pose-finish machinery present but NOT engaged (engage predicate never fired —
  seg still descending, correct per #383).

**Action:** w3 ticket 97bc5306… compiled via sealed_ticket() from w2's OWN levers
(in-place window swap ep1085→1224), resume from w2's stage_seg_trunk_tau_final.npz,
governed launch OK (pid 80705, monitor armed). At the w3 boundary: same tail-slope
adjudication; if converged, the winner is ON-w3's endpoint (already dominating both
prior endpoints) and jd1's joint pose-finish ticket regenerates against it.
