# qbt2b r10 endpoint adjudication — the born object is BYTE-FEASIBLE and DISTORTION-PRICED-OUT (task #1318)

**Axis:** `[local mechanism/advisory; not contest authority]` · `score_claim=false` · `promotable=false`.
Every S below is the trainer's own n32 Horvitz–Thompson `S_hat` (estimator `NO2_SECTION5_HT_COMPLETE`,
`selection_count: 32`), NOT an `upstream/evaluate.py` row. Three admission gates are unmet on every
run in the series (`d_pose_hat`, `s_hat`, `same_budget_qbw1_control`), and
`control_status = REFUSED_MISSING_REAL_SAME_BUDGET_QBW1_CONTROL` — so the LEVEL carries n32 +
no-matched-control uncertainty. The TRAJECTORY is what this memo adjudicates.

## 1. The run completed; nobody read it

`governed_n32_r10` exited **rc=0** at `2026-08-29T16:50:55Z` after **21,387.9 s** (5.94 h), peak RSS
2,453 MiB against a 118,784 MiB limit (`resource_safe_run_status.json`, schema
`safe_run_status_receipt.v1`, `receipt_status_disagrees_with_exit: false`). It ran stage_03a → 03 →
04 → 05 and wrote `stage_05_same_budget_admission/GATE.json`. It then sat unharvested ~3 h.

**Both r7 gates HOLD at the r10 endpoint** (task #1315's pre-registered pair):
- Lane `realized_within_class_error` = **0.11357** ≤ 0.12 ✓
- `seg_expected_flip_realized` = **0.0031128** ≤ 2 × 0.00972 = 0.01944 ✓ (6.2× under)

## 2. The four-point doubling series (MEASURED)

| run | cum. steps | `S_hat` | distortion | rate | `d_seg_hat` | `d_pose_hat` | `B_hat` |
|---|---:|---:|---:|---:|---:|---:|---:|
| r7  |  5,000 | 1.530371 | 1.448754 | 0.081617 | 0.013135 | 1.8286e-03 | 122,574 |
| r8  | 20,000 | 0.640566 | 0.559115 | 0.081451 | 0.004593 | 9.9651e-04 | 122,325 |
| r9  | 30,000 | 0.485089 | 0.403741 | 0.081349 | 0.003059 | 9.5776e-04 | 122,171 |
| r10 | 40,000 | 0.408898 | 0.327712 | 0.081187 | 0.002518 | 5.7575e-04 | 121,928 |

Cumulative steps are DERIVED from the ledger's own three statements (#1316 r8 = 15,020 steps;
#1317 r9 = 10,020 → 30,020; #1318 r10 → 40,020), which close consistently on r7 ending at ~5,000.
Only the first point depends on that inference; the two deltas that drive the fit do not.

Marginal `ΔS_hat` per 10k steps: **−0.5932 → −0.1555 → −0.0762** (decay ratios 0.262, 0.490).
Descent is monotone on all four columns. Rate is essentially flat and slightly falling.

## 3. THE FINDING: byte-feasible, distortion-priced-out

**This is the campaign's byte-feasible object.** `B_hat` = 121,928 B clears the sub-0.12 archive
ceiling of 137,986 B (m120) **with 16,058 B to spare**, and undercuts the gb1 pointer's 180,215 B by
**58,287 B = 0.038971 S of rate credit**. Every route that died on the rate axis (nr1, rc1's
successors, the whole #1193/#1198 alternative-representation swarm) died on the axis this object
already wins.

Its problem is entirely distortion, and the gap is large:

| target | distortion allowed | distortion now | reduction needed |
|---|---:|---:|---:|
| TIE gb1 (S 0.14811799921260607) | 0.066931 | 0.327712 | **4.90×** |
| sub-0.12 (T_4) | 0.038813 | 0.327712 | **8.44×** |

vs the gb1 body's measured distortion 0.028120, this object is **11.7× worse in distortion** while
being 32% cheaper in bytes.

## 4. Two extrapolations, both pricing it out of the near term

Fit on the four points (pure-python grid search; the unconstrained 3-parameter fit returns
`S_inf = −0.034`, unphysical, so it is reported only as the optimistic bracket):

- **Floor-pinned power law** (`S_inf` pinned to the physical floor = rate 0.081187):
  `S(n) = 0.081187 + 562.48·n^−0.70`, SSE 3.01e−4. → S(160k) = 0.2092 · **S(400k) = 0.1486** ·
  **S(1M) = 0.1167**.
- **Geometric marginal decay** (ratio 0.490 per 10k, the last measured ratio): asymptote
  **0.3357** — never reaches the pointer.

At 2.135 s/step (r10's own measured rate), the OPTIMISTIC model prices the milestones as:

| milestone | additional Metal |
|---|---:|
| 80k steps (one more doubling) | 23.7 h |
| 160k steps | 71.2 h (3.0 days) |
| **400k ≈ TIE the gb1 pointer** | **213.5 h (8.9 days)** |
| ~1M ≈ cross 0.12 | ~23 days |

**Both models agree the line cannot reach the current pointer in under ~9 days of exclusive Metal,
and the pessimistic one says never.** That is the adjudication: the born-object line is not CLOSED —
descent is real, monotone, and power-law with α = 0.70 — but it is **priced out of the near term**
and re-ranks below any route with a shorter path to an exact row.

## 5. What is NOT claimed

- No score claim. n32 HT estimate on an advisory axis with three unmet gates and no matched
  same-budget control. A real comparison to gb1 needs the qbw1 control the gate is refusing on.
- The extrapolation is DERIVED from 3 marginal deltas. The two models disagree by 0.037 at 50k
  steps — one 10k-step continuation (5.9 h) would discriminate them. That measurement is NOT
  ordered here: both branches price the line out of the near term, so discriminating the ETA
  (9 days vs never) does not change the near-term routing. Recorded as available, not owed.
- α = 0.70 is a fit on 4 points, not a law. It is not registered as a canonical equation.

## 6. NEXT_IF_RESUMED

- **CLOSED (adjudicated)** — task #1318's owed endpoint read. Both r7 gates hold; the series is
  monotone; the line is byte-feasible and distortion-priced-out at ≥9 days optimistic.
- **QUEUED-W-FIRE-ORDER** — owner MAIN; fire trigger: a Metal slot free for ≥24 h with no
  shorter-path candidate competing for it. Then r11 = 40k-step continuation to the 80k doubling,
  which both discriminates the two models AND banks real descent. Consumer: this memo's §4 table.
- **QUEUED-W-FIRE-ORDER** — owner MAIN; fire trigger: any successor that reduces this object's
  distortion by ≥4.9× at fixed bytes. The rate axis is already won by 58,287 B; the entire
  remaining question is distortion. Consumer: the #1308 route table, which this memo re-populates
  with one row that has a measured (if expensive) path.

## LIVE-HYPOTHESES

- The distortion descent is power-law, not asymptotic — the geometric read may be an artifact of
  fitting only 3 deltas across unequal step increments (15k, 10k, 10k).
- `d_pose_hat` fell 1.66× in the last interval (9.578e-4 → 5.757e-4) versus `d_seg_hat`'s 1.21× —
  pose is descending FASTER than seg on this object, the opposite of every other live line, where
  pose is the binding term (m110: pose budget ≤1.25e-4 absolute).

## DEAD-ENDS

- Reading this line's LEVEL as comparable to the pointer: the same-budget qbw1 control is missing
  and the gate says so. Only the trajectory is adjudicable from these receipts.
- Treating `B_hat` 121,928 B as a rate win to bank: it is a win only if the distortion is paid, and
  it is not. The bytes and the distortion belong to the same object.

Own-vehicle frontier UNMOVED this turn: **S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]**
(gb1, archive sha `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`).
