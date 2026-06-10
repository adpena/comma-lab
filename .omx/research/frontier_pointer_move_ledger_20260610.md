# Frontier pointer-move ledger (the exact-score scoreboard) — 2026-06-10

The single durable record of EXACT frontier-pointer moves. One row per attempt that reaches the
adjudication layer (`tac.optimization.scorer_quotient_candidate_row`). Only a row with
`pointer_update_eligible=True` (contest-tier `exact_evaluate`, recomputed ΔS<0) actually moves
`.omx/state/canonical_frontier_pointer.json`. Advisory/proxy rows are recorded for prioritization but
NEVER promote (the sub-0.15 firewall + "Frontier scores are pointer-only"). Lead every session report
with the latest pointer + whether it moved.

## Current exact pointer
| axis | score | archive sha | bytes | as-of |
|---|---|---|---|---|
| contest-CPU | **0.19109982** | `b46897267ded…` | 177,169 | 2026-06-10 (recoded-R3, defensive bank) |
| contest-CUDA | 0.20533003 | `9cb989cef519…` | 186,876 | 2026-05-16 (pr106) |

## Moves
| # | lever | candidate_kind | authority | ΔS | new pointer | innovation | decision |
|---|---|---|---|---|---|---|---|
| 0 | recoded-R3 (baseline) | requant/recode | contest-CPU + CUDA | (baseline) | 0.19109982 | defensive_bank (R1+R2 borrowed PR#112; fails Innovation Gate) | banked, submission-blocked on `constriction` allowlist |
| 1 | #64 lossless stack | — | — | **0.0** | unmoved | n/a | NO-OP: R1+R2+R3 already in base, S12 inapplicable to procedural carrier (NO-FAKE refused a no-op masquerade) |
| 2 | #72 lever-D margin-conditional residual | margin_residual | exact_cpu_advisory / exact_pair_scorer | **0.0** | unmoved | reusable rate-win | DEFER: RATE side WINS (0.856 B/flip < 1.27, overturns #51's 1.525 unconditional floor) but DISTORTION side DIES on receptive-field collateral (467 fixed / 2823 new-bad, net −2356; waterfill admits 0). True floor = collateral, not rate. Reactivate on contiguous-residual base (lever C / #73). |
| 3 | #73 Dykstra legal-frame feasibility | dykstra_feasible_frame | exact_cpu_advisory / exact_pair_scorer | **0.0** | unmoved | sharp geometric finding | scorer_effect (holds BOTH terms at frontier: d_seg 0.00057, d_pose 2.40e-5 in-tube, 4/4) but cheap-feasible set EMPTY at low byte (≥625KB/pair generic basis; <400KB the pose tube breaks). PROVES feasibility≠generation + the 177KB learned HNeRV basis IS the cheap-feasible representation. Reactivation = Dykstra with C=learned manifold = subsumed by #71. |

## Convergent meta-finding (3 no-moves)
#64 (lossless exhausted) + #72 (residual codes cheap @0.856 B/flip but collateral kills application) + #73
(generic-basis feasibility needs ≥625KB) all confirm from different directions: **the 177KB learned HNeRV
nonlinear basis IS the cheap-feasible representation for holding pose+seg simultaneously.** The only
sub-frontier path that holds both is a SMALLER LEARNED basis → #71 (structural Q* compression) + lever-C.
#71 is the singular most-likely pointer-mover; #69 (requant) relaxes its target.

## Pending movers (will append a row on landing, via the schema firewall)
- #69 score-aware Q* re-quant (rate) · #71 Q* structural compression (rate, possibly sub-0.15 alone)
- #72 lever-D margin-conditional residual coder (d_seg) · #54 cross-pair waterfilled corrector (pose)
- #73 legal-frame Dykstra feasibility (realization) · #63 d_seg-loss hinge (gates the lever-C campaign)

## Innovation-status note (per the Innovation Gate)
The current 0.19109982 frontier is a **defensive bank** (`defensive_bank=true`, `class_shift=false`,
`borrowed_substrate=true`, `submission_recommendation=hold_not_final`) — banked for readiness, NOT the
innovative submission. The original sub-0.15 submission must come from a class-shift mover (#73
feasibility / #63→lever-C / #71 structural) with `class_shift=true`.
