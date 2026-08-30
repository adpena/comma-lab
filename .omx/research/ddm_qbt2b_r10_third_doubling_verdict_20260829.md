# ddm_qbt2b_r10 — third-doubling ENDPOINT VERDICT: the pre-registered STOP fires, and the campaign's shape resolves (counter 698, task #1318)

**Date:** 2026-08-29
**Status:** `CHASE_CLOSED_ON_MEASURED_CURVE` — the n32 constrained-margin chase STOPS per the rule
`ddm_qbt2b_r9_constrained_margin_verdict_20260829.md` §5 pre-registered BEFORE this run ended.
**Axis:** run rows are `[macOS-MPS governed n32 research row; not contest authority]`, HT-projected to
n600; `score_claim=false`, `promotable=false`. The lb1 comparison row is `[contest-CUDA T4 n600]`
authority. Every mixed-axis line below is labelled DERIVED-FEASIBILITY, never a score.
**Run:** rc=0, 10,010 steps, 5h56m, `governed_n32_r10`, RESULT.json 23,090,441 B;
`stage_04` byte-close + `stage_05` same-budget gate both completed. Lane rows closed
`completed_rc0` at 2026-08-29T16:55:53Z.

## 1. The pre-registered rule fired — exponent re-derived, not quoted

r9 §5 pre-registered: *e(30k→40k) ≤ −0.85 → chase stays credible; e > −0.85 → the flattening IS
the trend → STOP.* Computed by the §1 method (endpoint-to-endpoint on matched tail-25 means of
`seg_expected_flip_realized`), MAIN re-derived independently from the 10,000 in-run samples in
`run.log`:

| quantity | value |
|---|---|
| r10 tail-25 flip | **0.002968813** (tail-100 0.002965265; final 0.003112751; first 0.031892702) |
| r9 tail-25 flip @30,020 | 0.003530 |
| **segment exponent e(30,020→40,020)** | **−0.602178** |
| bar | −0.85 |
| verdict | **> bar ⇒ STOP** |

This reproduces the lane-ledger figure `−0.6022` to six significant figures from the raw samples.
The exponent series is now **−0.781 (r7) · −0.963 (r8) · −0.696 (r9) · −0.602 (r10)** — two
consecutive segments flatter than the r8 window-fit, exactly the STOP condition.

The flip also landed **+2.73% ABOVE** r9's own projection (0.002890 predicted, 0.002968813
measured) — the same over-promise the r9 window showed at +6.0%. The law is not merely flattening;
it has over-promised at both of the last two doublings.

## 2. The target RECEDES faster than the chase approaches it

Box-class flip is 0.00116. Projected cumulative steps to reach it, under each successive law:

| law | projected cumulative steps to box-class flip |
|---|---:|
| r8 segment (−0.9632) | 85,192 |
| r9 segment (−0.696) | 148,536 |
| **r10 segment (−0.602178)** | **190,557** |

40,020 steps of measurement moved the target from 85k to 191k. Each doubling buys less than the
prior law promised, so the remaining distance grows as it is measured. This is the honest reason
the chase closes: not a wall (the flip descends monotonically at segment scale, 2.559× still to
go), but a **decelerating trajectory whose target recedes** — the `caps=genus` trajectory-stopping
law, applied to a cap-free run on its own measured curve.

## 3. The structural blocker: the born object cannot reach 0.12 EVEN AT ZERO SEG ERROR

`stage_05_gate` (admitted=False, `REFUSED_MISSING_REAL_SAME_BUDGET_QBW1_CONTROL`), HT-projected
n600, S recomputed FROM COMPONENTS (#877) — reproduces the reported `S_hat` to 0.00e+00:

| term | value |
|---|---:|
| B_hat | 121,928 B |
| rate = 25·B/37,545,489 | 0.081186850 |
| d_seg_hat 0.002518335978 → seg | 0.251833598 |
| d_pose_hat 0.000575745612 → pose | 0.075877903 |
| **S_hat** | **0.408898351** |

**rate + pose ALONE = 0.157065 > 0.12.** Setting d_seg to exactly zero does not reach the target.
The pose budget arithmetic:

| assumed seg | pose budget left under 0.12 | implied d_pose ceiling | r10 is over by |
|---|---:|---:|---:|
| seg = 0 (unreachable best) | 0.038813 | 1.506461e-04 | **3.8×** |
| seg at the lb1 pointer's 0.020139 | 0.018674 | 3.487239e-05 | **16.5×** |

Against the absolute pose-budget law (memory `m110`, d_pose ≤ 1.25e-4), the born object sits
**4.6× over budget**. So the chase's remaining 2.559× on flip was never sufficient on its own.

## 4. The five-run trajectory, whole

| run | B_hat | d_seg_hat | d_pose_hat | rate | seg | pose | S_hat | admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| r6 | 122,330 | 0.009050369 | 0.002627342 | 0.081455 | 0.905037 | 0.162091 | 1.148582 | False |
| r7 | 122,574 | 0.013135274 | 0.001828622 | 0.081617 | 1.313527 | 0.135227 | 1.530371 | False |
| r8 | 122,325 | 0.004592896 | 0.000996511 | 0.081451 | 0.459290 | 0.099825 | 0.640566 | False |
| r9 | 122,171 | 0.003058751 | 0.000957764 | 0.081349 | 0.305875 | 0.097865 | 0.485089 | False |
| **r10** | **121,928** | **0.002518336** | **0.000575746** | **0.081187** | **0.251834** | **0.075878** | **0.408898** | **False** |

Rate is **flat to 0.5%** across the entire chase (121,928–122,574 B) while distortion falls 2.8×.
The born object holds its byte feasibility unconditionally: 121,928 B is **16,058 B UNDER** the
sub-0.12 cap of 137,986 B.

## 5. THE CROSS — the campaign's shape, stated exactly

This is the finding worth more than the verdict. Put the two live objects side by side:

| object | rate | seg+pose | S | vs 137,986 B cap |
|---|---:|---:|---:|---|
| born (qbt2b r10, advisory n32-HT) | 0.081187 | **0.327712** | 0.408898 | **16,058 B SPARE** |
| lb1 pointer (contest-CUDA T4 n600) | 0.119910 | **0.028120** | 0.148030 | 42,097 B OVER |

- lb1's distortion (0.028120) is **already inside** the born object's spare budget of 0.038813.
- The born object's rate (121,928 B) is **already inside** lb1's byte cap of 137,986 B.

**DERIVED-FEASIBILITY (mixed-axis, NOT a score claim, NOT an archive that exists):** an object
carrying the born rate and the lb1 distortion would score
`0.081186850 + 0.028120 = 0.109307078` — **sub-0.12 by 0.010693**, and 0.038723 below the live
pointer. Each of the two objects holds exactly one half of a sub-0.12 solution, and five runs of
measurement say neither can acquire the other's half: the born object's distortion decelerates
with a receding target (§2–§3), and the dx2/lb1 object's rate is defended on every coder axis
measured (`jt23` coder axis CLOSED at 0 B; `m144` lossless remainder ≈2,009 B = 4.8% of the
42,097 B demand; `ld1`/`ae1`/`oe1`/`rr9` all closed).

That is not a route. It is the sharpest available statement of **why** no route has been found:
the campaign has been asking each object to fix the half it is structurally worst at.

## 6. Disposition

- **n32 constrained-margin chase: CLOSED** on its measured curve. No r11. The λ-constrained margin
  mechanism WORKED (Lane held, births survived, flip descended monotonically 0.0319 → 0.00297 =
  10.7× over 40,020 steps) — the family is not refuted, the *chase to box-class on this object* is.
- **Scope:** INSTANCE — the qbt2b born object at 40,020 cumulative constrained-margin steps,
  n32 selection HT-projected to n600, advisory axis. It does not close the born-object FAMILY
  and it does not close constrained margin as a mechanism.
- **What survives, portable:** (a) the born object is the only measured body that is BYTE-FEASIBLE
  for sub-0.12 with room to spare — that property is what any successor object must inherit;
  (b) the constrained-margin mechanism with the Lane existence guard, measured across 40k steps;
  (c) the descent-law instrument with its own over-promise history recorded (r8→r9 +6.0%,
  r9→r10 +2.73%) — a law that must be re-fit, not extrapolated.
- **What does NOT survive:** the r8-law projection (85,192 steps) and the r9-law projection
  (148,536) are both superseded; do not cite either as the cost of reaching box-class.
- **Owed, not performed here:** `stage_05_gate.same_budget_qbw1_control` is still
  `REFUSED_MISSING_REAL_SAME_BUDGET_QBW1_CONTROL` — the gate has never had its matched control, so
  even a passing S_hat could not have admitted. That refusal is CORRECT and stays open.

**Own-vehicle frontier: lb1 — S 0.14803010583079396 @ 180,083 B [contest-CUDA T4, n600], UNMOVED.**
Sub-0.12 gap 0.028030; demand 42,097 B at current distortion.
