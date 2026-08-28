# ddm_qbt2b_r7 — lane-constrained margin ENDPOINT VERDICT (counter 695)

Date: 2026-08-28. Owner: MAIN. Axis: **[macOS-MPS governed n32 frozen-scorer advisory]**,
score_claim=false everywhere in this memo. Charter:
`.omx/research/charters/ddm_qbt2b_r7_lane_constrained_margin_20260828.md`. Arm build memo:
`ddm_qbt2b_r7_lane_constrained_margin_20260828.md` (§7 fire order executed verbatim).

## 1. Headline

**The constraint law works. The 2×2 is resolved: per-class primal-dual protection is the
right instrument for the born-Lane collateral — not weights.** Over the full 5,000-step
constrained margin window on the r5-born basis:

- **Lane was PRESERVED.** Final werr 0.098322, tail-25 mean 0.118574 at the 0.12 bound.
  Zero steps breached the 0.50 existence floor. r6's unconstrained margin stage erased the
  born Lane to 99.81% by comparison — the single defect this arm was chartered to cure.
- **Movable ended UNDER its bound**: final 0.004829, tail-25 0.007739 vs bound 0.009
  (r6 eroded Movable to 10.85%).
- **λ never approached the ceiling**: max λ_Lane 0.235807, max λ_Movable 0.297957,
  ceiling-contact steps 0/5,000 vs λ_max 5.0. The dual variables rose on violation and
  decayed on satisfaction — live, binding, and responsive (the #404 proof, in-telemetry
  every step).
- **Constraint cost bounded**: total realized expected-flip tail-25 0.012839 vs r6's
  unconstrained 0.00972 endpoint — 1.32×, well inside the pre-registered 2× allowance
  (bar 0.01944).
- **Pre-registered falsifier: NOT FIRED** (neither clause — no λ pinned at 5.0 with
  flip ≫ bar; no 0.50 breach under active λ). The constraint set is **FEASIBLE on the
  r5-born qbt basis**. Lane does NOT route to the m131/d3a analytic Lane-carrier leg.

## 2. Prediction adjudication (honest, both letter and substance)

The charter's PRIOR-LAW PREDICTION read: "holds Lane werr ≤ bound for the WHOLE
5,000-step window while total realized flip descends to within ~2× of r6's 0.00972."

- **Substance: CONFIRMED.** No erasure, bounded flip cost, no infeasibility signature.
- **Letter: NOT met pointwise.** 2,540/5,000 steps (50.8%) had instantaneous Lane werr
  above 0.12 (max 0.4139, early transient), and 1,899/5,000 had Movable above 0.009.
  This is the expected behavior of a simultaneous-discretization primal-dual scheme at
  an ACTIVE constraint: the trajectory chatters around the bound with mean at the bound;
  λ integrates the violation. A pointwise-hold prediction was too strong for the
  mechanism as built; future constraint predictions on this trainer should be stated in
  tail-mean/It-never-leaves-the-existence-region form. The tail-25 means (0.1186 Lane,
  0.0077 Movable) sit exactly at/under the bounds — the average-sense hold is clean.

## 3. Endpoint numbers (stage 03, constrained margin, steps 1–5,000)

| quantity | value | reference |
|---|---|---|
| Lane werr final / tail-25 / max | 0.098322 / 0.118574 / 0.413932 | bound 0.12; r5-born ~0.098–0.12; r6 endpoint 0.9981 |
| Movable werr final / tail-25 | 0.004829 / 0.007739 | bound 0.009; r6 endpoint 0.1085 |
| λ_Lane max / final | 0.235807 / 0.029433 | ceiling 5.0, contact 0 |
| λ_Movable max / final | 0.297957 / 0.081076 | ceiling 5.0, contact 0 |
| seg_expected_flip_realized final / tail-25 | 0.012975 / 0.012839 | r6 0.00972; 2× bar 0.01944 → WITHIN |
| pose_mse_realized final / tail-25 | 1.126e-3 / 1.188e-3 | m110 shipped-d_pose budget 1.25e-4 → ~9.5× ABOVE (open axis) |

Run: counter 695, pid 36491/36502, exit 0, elapsed 12,586.4 s, peak RSS 2,304.9 MiB.
Resumed from the counter-694 birth handoff (no margin optimizer step pre-crash; r6
precedent: resume bit-faithful across the 03a→03 boundary). all_payloads_retained=true;
per-step history embedded in every checkpoint (2,002 checkpoints, stage_03_end.pt +
periodic every 5 steps); reader `.omx/tmp/qbt2b_r7_endpoint_read.py` (checkpoint-embedded
history, quantities above).

## 4. Stages 04/05 (ran to completion after the margin window)

- **Stage 04 precision waterfill**: per-role sensitivity shortlist + real QBF1 byte-close
  emitted (`options/PRECISION_SENSITIVITY.json`), disposition honestly
  `NO_ADOPTION_WITHOUT_REALIZED_SCORER_AB` — no precision option adopted by citation.
- **Stage 05 admission gate (`GATE.json`): admitted=false, correctly.**
  B_hat **122,574 B** — the packet clears the sub-0.12 rate demand (≤137,986 B) with
  15,412 B to spare. But S_hat **1.5304**: d_seg_hat 0.013135 (seg ≈ 1.31 S) and
  d_pose_hat 1.829e-3 (pose ≈ 0.135) — HT-projected n32→n600, advisory. Failed gates:
  `s_hat`, `d_pose_hat`, `same_budget_qbw1_control`
  (REFUSED_MISSING_REAL_SAME_BUDGET_QBW1_CONTROL — the standing owed control).

**Consequence for the §7 "exact byte-closed row is the next consumer" rule:** NOT fired.
That rule was written before stage-05 numbers existed. Buying a T4 row on a candidate
whose own advisory gate reads S_hat 1.53 (10× the frontier) would spend the ~$1.39 Modal
headroom on a known-dominated packet — the honest reading of §7's intent is "the exact
row consumes the constraint result WHEN the vehicle's distortion axis closes," not "buy a
row on any surviving endpoint." No Modal dispatch fired from this adjudication.

## 5. What r7 settles and what it opens

SETTLED (family scope, n32 seeded-stratified cohort):
- The weight-family 2×2 is closed WITH its resolution: r4 unweighted-CE (no birth) ·
  r5 balanced-CE (birth + prior shift) · r6 unweighted-flip (erasure) · **r7 constrained
  flip (birth preserved, bounded cost)**. The instrument for thin-class protection under
  an area-priced objective is a CONSTRAINT on realized werr, not a weight.
- The QBFLOW vehicle is RATE-feasible for the sub-0.12 box at this size (122,574 B).

OPEN (the distortion axis — the whole remaining game for this vehicle):
- d_seg_hat 0.0131 vs the ~0.00116-class box need — 11× gap at 5,020 total steps, n32.
- d_pose_hat 1.83e-3 vs the m110 1.25e-4 budget — 14.6×.
- The same-budget QBW1 control (stage-05 gate hard-requires it) — Metal slot now FREE.

ROUTING:
1. **QBW1 same-budget control** (standing owed, unblocks the stage-05 gate) — next
   Metal-slot occupant.
2. **r8 budget/scale question**: whether the constrained-margin law at a longer budget
   (or n600 selection) closes toward the box, or plateaus — the r7 flip trajectory was
   still descending at step 5,000 (0.0136 tail-100 → 0.0128 tail-25); un-plateaued.
3. Pose stays in-window co-trained; no separate pose leg fires before the seg axis
   shows a closing trajectory (m110 ratio discipline).

## 6. Two-landing note — the device-gap class (counter-694 crash)

Counter 694 died at 61 s: `TypeError: Cannot convert a MPS Tensor to float64` in
`realized_within_class_error` — MPS has no fp64. **Class: a CPU-bounded smoke is
structurally blind to device-scoped dtype legality when the governed launch is
MPS-locked** (`validate_config` requires device=mps for action=train; the charter bounded
the arm's smoke to n1/CPU). Sibling of the #1306 Metal-blind-instruments genus. Fix
landed 8a10975b40 (CPU hop BEFORE the float64 cast — precision-identical: same ops
reordered on CPU runs; exact fp64 mean on every device). Landing 2 (class protection)
candidate: the charter-lint / smoke contract should require the smoke device to match the
launch device for any device-locked action, OR the trainer's own validate path should
carry a device-legality dry-forward. Filed as a two-landing candidate on the #1306 row
rather than a new catalog number (same genus: instruments/smokes on a different device
than the enforced runtime).

## 7. Lane closure

Terminal rows appended for `ddm_qbt2b_r7_scorer_20260828` + `ddm_qbt2b_r7_metal_20260828`
(completed). Own-vehicle frontier UNMOVED this unit: the exact pointer stands at
gb1 S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600] — this unit produced a
constrained-training LAW result + an honest advisory gate refusal, not an exact row.
