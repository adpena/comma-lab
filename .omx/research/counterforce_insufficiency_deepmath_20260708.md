# COUNTER-FORCE INSUFFICIENCY — deep-math investigation (where the v7.5 birth-stack counter-force falls short)

**Date:** 2026-07-08 · **Axis:** `[macOS advisory, run-1 existing telemetry n600]` NON-PROMOTABLE ·
**$0, read-only** (pid 63069 + run dirs UNTOUCHED) · **Pointer contest-CPU 0.19110 UNMOVED — MEANS.**
Operator question 2026-07-08: *"Where [is the] counter force measured as insufficient if anywhere — do
deep math investigation and against telemetry."*

## STORES CONSULTED
`v75_birth_counterforce_20260708.md` (§1 Chan-Vese Lever-1 derivation + §7 ramp; the ≥→~96% correction) ·
`road_anomaly_probe_20260708.md` (the recall-without-precision diagnosis; part_frac vs GT; mass
conservation) · `probe_PA_paintfloor_perclass_20260708.md` (oracle floor = 100% separatrix placement;
Road within-class flip 0.17%) · `p0_forces_derivation_20260708.md` (Force 3 tie-locus = the precision
placement counter-force; the interaction matrix) · `t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
(§2 sealed λ + the "~96% NOT ≥" R5 correction; §5 watch-list Road PRIMARY flip>0.30@ep200) ·
`chan_vese_area_constraint_birth_balance_20260708.py` (the registered equation module) ·
run-1 telemetry `experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z/run.log`
(verdict d_seg_by_class + handoff_readiness per_class part_frac/within_flip/gt_px, ep0→175) ·
CLAUDE.md · `docs/operating_manual_craft_handoff.md`. Canonical class order Road0/Lane1/Undriv2/Movable3/MyCar4.

## 0. THE MEASURED SUBSTRATE (run-1, n600, the birth-arm; ALL numbers re-derived from raw telemetry)

GT area fractions from `gt_px` (render grid, sum 117,964,800 px): **Road 0.23233 · Lane 0.00585 ·
Undriv 0.49518 · Movable 0.01238 · MyCar 0.25426** (match the equation module's `MEASURED_GT_AREA_N600`).

Per-class trajectory (`handoff_readiness.per_class`, `verdict.d_seg_by_class`):

| ep | Road part_frac | Lane pf | Movable pf | Road within_flip | verdict d_seg |
|---|---|---|---|---|---|
| 25 | 0.1388 | 0.0948 (16.2×) | 0.0897 (7.2×) | 0.4363 | 0.17685 |
| 125 | 0.1407 | 0.0805 (13.8×) | 0.0568 (4.6×) | 0.3978 | 0.13018 |
| 175 | 0.1440 | 0.0716 (12.2×) | 0.0564 (4.6×) | 0.3831 | 0.12548 |

**Apparatus-validity precondition met:** run-1 is LIVE and training (ep_loss>0, verdicts advancing
25→175); `d_seg = Σ_c within_flip_c · A_c^GT` reconstructs the logged verdict to **6 digits** at both ep125
(0.130182) and ep175 (0.125477) — the decomposition arithmetic below is faithful, not a reinterpretation.
Run-1 has NO counter-force (clock-mode birth-arm), so it is the correct instrument to PREDICT the
post-counter-force state. Lane self-corrects only from 16.2×→12.2× over ep25→175 and is DECELERATING
(loss-imposed floor, road_anomaly) → extrapolating this rate needs ~430 more epochs to reach 1.25×GT →
**the counter-force is NOT redundant; the CE precision pressure alone plateaus far above GT.**

---

## THE INSUFFICIENCY LEDGER — three quantified shortfalls

### (a) The δ-tolerance residual — MEASURED/DERIVED, closable by the δ/λ A/B

The equilibrium is `A_c* = (1+δ)·A_c^GT` with δ=0.25 — a **deliberate 25% residual overshoot for
stability**, so lane/movable settle at 1.25×GT rather than exactly GT.

- **Residual stolen area** = `δ·(A_GT_lane + A_GT_movable)` = 0.25·(0.00585+0.01238) = **0.00456**
  (equivalently: retracting lane 0.0805→0.00732 and movable 0.0568→0.01547 returns 0.11452; the majority
  deficit was 0.11892; residual not-returned = **0.00440**, matching 0.00456 up to the 0.00016 mass-conservation
  slack at ep125). Confirms the memo's "~0.0044 area."
- **As a fraction of the deficit:** 0.00456/0.11892 = **3.8% ≈ 4%** → the memo's "~4% of deficit" and
  "returns ~96%" (96.3%) are both CORRECT. The R5-corrected "~96%, NOT ≥ the deficit" framing holds exactly.
- **Implied residual Road d_seg:** the returned area splits deficit-proportionally (Road 77.1% / Undriv 22.9%
  of the 0.11892 deficit). Road's residual area-deficit = 0.771·0.00456 = 0.00351 abs = **0.0151 of GT-Road**.
  So the δ=0.25 area residual alone lifts Road within_flip from 0.394 (theft) down to ~0.015.

**Verdict (a):** the δ-band residual is a **real, quantified ~4% shortfall (~0.0046 area / ~0.015 residual
Road within_flip), by design, and closable** — it is monotone in δ (below), so it is a tuning knob, not a
wall. verdict_scope: instance (δ=0.25 on the v7.5 config).

### (b) The λ-scale assumption — DERIVED bound; safe with large margin

λ_c = W_birth/(δ·A_GT_c) with W_birth ≡ F_birth ≡ 1.0 ASSUMED (the argv amplify/recall weight). If the true
birth force is κ× the assumed (λ under-scaled by κ), the equilibrium moves to `(1+κδ)·A_GT` (more residual).
Two independent bounds from the ep125 runaway:

- **Dominance margin:** the retraction still CAPS the runaway iff κ < (r_obs−1)/δ = (13.76−1)/0.25 = **51×**.
  The birth force would have to be 51× stronger than the argv weight for the runaway to escape the cap.
- **Materiality margin:** the counter-force returns <50% of the stolen area only if κ > **13×**.
- At κ=1 (assumed): returns 96.3%.

Both the area penalty and the birth losses are boundary-localized on the SAME annulus (both ride the
softmax-Jacobian / δ(φ)), so their per-unit-weight gradient scales are comparable ⇒ κ ~ O(1). **Verdict (b):**
the W_birth=1.0 assumption is a MEASURED-ANCHOR config-conditional (the real argv weight), not a vibes
constant; material insufficiency needs κ>13 and cap-escape needs κ>51 — huge margin. The λ-scale A/B tunes
the *residual*, it is not needed to *prevent* insufficiency. verdict_scope: instance.

### (c) THE STRUCTURAL insufficiency — the honest architectural limit (area ≠ placement)

This is the deep-math heart. Decompose each class's per-class d_seg exactly:

> **within_flip_c ≈ (area_deficit_c / A_c^GT)  +  placement_residual_c**
> where area_deficit = GT−part_frac (the theft, gains≈0 for the victim classes), and placement = the
> codim-1 separatrix boundary-jitter residual.

Measured for **Road @ep125** (the floored class):
- Road within_flip (per-class d_seg) = **0.3978**
- area-theft component = deficit/GT = 0.09167/0.23233 = **0.3945 (99.2% of Road's d_seg)**
- placement/other component = 0.3978−0.3945 = **0.0033 (0.8%)**

So in the birth-arm regime, **99.2% of Road's floor is AREA THEFT — squarely the Chan-Vese counter-force's
domain.** The counter-force is the RIGHT, dominant fix here.

**But it has a hard floor it cannot cross.** The area constraint fixes AREA; it cannot place the SEPARATRIX.
FEED-PA measured that at the converged/oracle regime **100% of the achievable floor is boundary placement,
ZERO interior flips** (Road within-class flip 0.17% = 0.0017, the irreducible through-R placement floor).
As the counter-force returns the stolen area (δ→0), the area-theft term → 0 but the placement term is
UNTOUCHED:

- δ=0.25: Road within_flip ≈ 0.0151 (area residual) + placement → **~0.018** (the run-1-derived prediction below)
- δ→0: Road within_flip → **placement floor ~0.0017 (oracle) ≤ witness-actual (TBD)** — Chan-Vese CANNOT
  go below this.

**The counter-force is "insufficient" for placement BY CONSTRUCTION — and that is CORRECT.** Area theft
(birth-arm-specific over-paint) and separatrix placement (the converged-floor currency) are ORTHOGONAL
quantities. The placement residual is the **tie-locus normal-displacement force's (P0 Force 3) domain**, the
force `p0_forces_derivation` names as "the direct precision counter-force ... highest-EV of the four,"
explicitly SYNERGISTIC with the completion event ("completion fills birthed islands (recall); #3 places
their boundaries (precision)"). The Chan-Vese term returns the stolen area; Force 3 places the boundary; they
are different forces for different residuals. verdict_scope: **paradigm-safe** — this is not a negative on
the counter-force, it is the honest statement of what it does and does not address.

---

## PART 5 CROSS-CHECK — the pre-registered post-counter-force Road watch (from run-1, which has NO counter-force)

Run-1 (birth-arm, no counter-force) predicts the post-counter-force Road state:

**Predicted Road within_flip post-counter-force (δ=0.25) ≈ 0.015 (area residual) + placement ≈ 0.018–0.035**
(placement bounded below by the oracle 0.0017; the witness byte-limited placement floor is owed/TBD).

**Pre-registered watch for the v7.5 launch** (sharpens SPEC §5 "Road flip>0.30@ep200"):
- part_frac lane should retract toward **1.25×GT ≈ 0.0073**, movable toward **≈ 0.0155**, then HOLD (the
  Chan-Vese equilibrium); persisting overshoot ⇒ λ under-scaled ⇒ the λ A/B fires.
- Road within_flip should fall from ~0.38–0.40 toward **< 0.05** as the area returns; if it stalls above
  ~0.10 the counter-force is under-engaged (bound (b)); if it floors at ~0.015–0.035 and STOPS, that is the
  **placement floor** and the next lever is Force 3 (tie-locus), NOT more area constraint.
- Total d_seg reconstructs as Σ within_flip_c·A_c^GT (verified to 6 digits) — track the per-class split, not
  just the composite.

---

## CANDIDATE EQUATION — COUNCIL-FLAGGED (not registered here)

`area_theft_vs_placement_dseg_decomposition_v1`:
`within_flip_c = clip(GT_area_c − part_frac_c, 0)/GT_area_c + placement_residual_c`, where the first term is
Chan-Vese-addressable (area) and `placement_residual_c` is tie-locus-addressable (separatrix), the two
orthogonal d_seg reservoirs. **Anchored** on run-1 n600 (Road ep125: 0.3945 theft + 0.0033 placement =
0.3978 = measured; global identity d_seg=Σwithin_flip·A_GT verified to 6 digits at ep125/175). **Council-flagged
NOT registered** per triality discipline: the theft-half is measured, but the counter-force *efficacy* (how
much returns) and the witness placement floor are PREDICTED/owed to the v7.5 A/B — mirroring how
`road_anomaly` and the `chan_vese` λ-scale council-flag their predicted halves. Register after the v7.5
counter-force A/B lands the efficacy + placement-floor anchor.

## FINAL STATE
$0 static + existing-telemetry; n600 full; run-1 (pid 63069) untouched; NO launch. **Pointer 0.19110
UNMOVED — MEANS** (moves only through a byte-closed `upstream/evaluate.py` n600 exact row). Triality legs:
DAG FEED (owed on next append) · candidate equation (council-flagged) · no DSL change (investigation only).
