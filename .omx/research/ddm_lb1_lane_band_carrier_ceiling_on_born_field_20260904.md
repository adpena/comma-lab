---
title: "A perfect lane carrier is worth 0.159 of S and still leaves the born field at 8.94x the sub-0.12 accuracy corner: the landed analytic lane band recalls 93.5% of the lane at 2.5 KB but its precision tops out at 0.565 against a break-even of 0.909, so none of 162 measured configurations improves d_seg at all — the persistent set is a PRECISION wall, not a rate wall, and the representation the Lane sites demand is not a lane-shaped object"
arm: ddm_lb1
charter: .omx/research/charters/ddm_lb1_lane_band_carrier_ceiling_on_born_field_20260904.md
charter_commit: 29a303192
preregistration: .omx/research/ddm_lb1_prereg_20260904.md
preregistration_commit: ced026cdd
instrument_commit: db8fc4b64
utc: 2026-09-04T16:20:00Z
verdict_scope: "[macOS-CPU advisory . LABEL-SPACE CEILING (composition into the retained terminal argmax, NOT realized through R) . frozen CPU-torch SegNet . QBF1-born vehicle, cold control seed 20260902, terminal EMA shadow step 5000 . n32 sealed selection . DALI authority with PyAV beside it . formulation scope: the analytic lane ground-frame band as a LABEL-SPACE Lane authority on THIS born field . NON-PROMOTABLE . no score claim . 0 Metal / 0 Modal / 0 contest eval . $0]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_lb1 — pricing the lane-band carrier's ceiling on md1's persistent set

## The pre-registered prediction and falsifier, READ OUT BEFORE THE NUMBERS

Committed at `ced026cdd` before the measurement ran.

> **Prediction** (charter §4): rule (a) removes **≥ 50%** of the persistent set at **≤ 2 KB** with
> harm **< 20%** of the removal.
> **Falsifier**: **< 25% removed**, **or** harm ≥ removal, **or** bytes > 6 KB.

## Verdicts

| clause | measured | verdict |
|---|---:|---|
| removes ≥ 50% of the persistent set | **43.24%** (rule a, DALI) | prediction **FAILS** |
| at ≤ 2 KB | **2,832 B** coded (n32) | prediction **FAILS** |
| harm < 20% of the removal | harm is **409%** of the removal | prediction **FAILS** |
| falsifier: < 25% removed | 43.24% (a) · 32.86% (c) · **14.60% (b)** | **FIRES on rule (b)** |
| falsifier: harm ≥ removal | **4.09× (a) · 8.06× (b) · 3.77× (c)** | **FIRES on all three** |
| falsifier: bytes > 6 KB | 2,832 B; 2,079–2,832 B across the fit sweep | does **not** fire |

**The falsifier fires.** It fires on the harm clause under every composition rule, and on the
removal clause under rule (b). The byte clause never fires — bytes were never the problem.

## The finding, first

**A LANE-ONLY object cannot close this. Even a PERFECT one falls 8.94× short.**

Give the born field's terminal argmax exact Lane authority in both directions — an oracle no fitted
carrier can beat — and its `d_seg_hat` falls 0.0028066 → **0.0012197** (ΔS_seg **−0.15869**),
removing **63.12%** of md1's persistent set with **zero** broken sites. That is a large, real gain,
and it is **8.9375×** the sub-0.12 accuracy corner. Add optimization that removes every remaining
reachable site and the joint floor is **4.7036×** the corner. md1's 12.75× becomes 4.70× — a
**2.71×** improvement that still does not reach 1.

**And the landed carrier gets nowhere near its own oracle.** Across **162 measured configurations**
— 2 centerline degrees × 2 dash-gate settings × 3 AA softnesses × 6 coverage thresholds (72 fit
rows), plus 5 thresholds × 6 uncertainty-gate settings × 3 composition rules (90 composition rows) —
**not one improves `d_seg`.** The best of all 162 is **+0.008481 S**. The band is not short of
recall: at its best operating point it recalls **93.5%** of the lane, a false-negative `d_seg` of
**0.000387** that reproduces FEED-dj's own 0.00046. It is short of **precision**, and the shortfall
is exactly measurable.

**The wall is arithmetic, and it is a precision wall.** A UNION Lane claim heals a site only where
the born field is already wrong AND the ground truth is Lane; it breaks a site everywhere else it
paints. Measured on this field:

* `P(born wrong | GT = Lane)` = **0.09959** — the born field already gets **90%** of the lane right.
* `P(born correct | GT ≠ Lane)` = **0.99777** — the ground the band paints over is essentially all
  already correct.

A claim must therefore be right **90.92%** of the time merely to break even. The band's precision
ceiling across all 162 configurations is **0.5651** — short by **1.609×**. No threshold, no dash
gate, no uncertainty mask moves a 56% claim past a 91% bar.

**Bytes were never the constraint.** The whole 32-pair lane manifold codes to **2,832 B** through
the module's own bit-exact LBND2 coder, and quantization costs **0.73 recall points** (raw fit
0.5450 → dequantized 0.5377). At the naive n600 transfer the perfect oracle would pay
**4.49× the rate floor** — it is worth its bytes by a wide margin. It simply cannot reach the target.

## 1. Verified at source — every premise this arm adds

| premise | where | label |
|---|---|---|
| the terminal EMA shadow argmax and md1's six-class site partition are LOADED, never recomputed | `/Volumes/APDataStore/pact/ddm_md1_micro_macro/payloads/cold_control_seed_20260902/shadow_step_005000.npz`, `site_classes_cold_control_seed_20260902_shadow_dali.npz` | MEASURED (custody) |
| the terminal shadow state is `payload["ema"]["shadow"]` of the run's `stage_01_end.pt` | `ddm_md1_micro_to_macro.py:176`, `:456` | MEASURED (source) |
| integer HT scoring: `W = Σ_p w_p·n_wrong(p)`, `w ∈ {15,30}`, denominator `600·384·512` | `ddm_qbt1_qbflow_trainer.py:112`; `ddm_md1_micro_to_macro.py:655` | MEASURED (source) |
| GT authority DALI `gt_cache_dali.pt` sha `a91d9825…`, PyAV beside it (20,671 argmax sites apart) | `SWEEP_RECEIPT_cold_control_seed_20260902.json::gt_lineage` | MEASURED (custody) |
| **the carrier's own coder is `serialize_lane_band_rd` (LBND2) in `analytic_lane_render_band.py:979`, NOT `curve_relative_offset_coder.py`** | traced to the primary implementation; the charter and vr1 row 10 both name the wrong module (`curve_relative_offset_coder` codes a RESIDUAL sidecar, `analytic_lane_render_band.py:1-35`) | **CORRECTION** |
| **`0.00087` is the witness TARGET, not the band's achieved d_seg**; the band's measured false-NEGATIVE d_seg is 0.00046 and its false-POSITIVE d_seg **as full authority is 0.00396** | `src/tac/boundary_math/lane_sdf_component.py:17-18` | **CORRECTION** |
| the band-render gauge is recorded `measured=False` — "the NET-NEGATIVE through-R d_seg is realized by TRAINING WITH the band active (GPU-pending)" | `src/tac/witness_dsl/gauge.py:1578-1587` | MEASURED (source) |
| sub-0.12 accuracy corner `d_seg = 1.3646784205e-4` | `.omx/research/ddm_qn1_qbr1_n600_realization_ticket_20260903.md` | TRANSFERRED (DERIVED there, n600, at the falsifier pose, on the bound 106,626 B archive) |
| md1's persistent floor 0.0017403920 = 12.753× the corner; schedule-lever ceiling 1.61× | `.omx/research/ddm_md1_micro_to_macro_dynamics_20260904.md` §4 | TRANSFERRED (md1, same n32, same object) |
| rate term `25/37,545,489 = 6.658589531221714e-7` S/B | `upstream/evaluate.py:63`; CLAUDE.md banner | MEASURED (source) |

**Two headline corrections travel with this arm.** vr1 row 10's "Lane band d_seg 0.00087" is a
stale headline over a body that says the opposite: the primary implementation records 0.00087 as the
witness's *target*, the band's shape capture as 0.00046, and the band's own false-positive cost as
**0.00396 — 8.6× its false-negative cost**. The gauge for that same lever is flagged `measured=False`.
This arm's measurement CORROBORATES the module's recorded mechanism rather than contradicting it.
Genus: `[[m106]]` stale headlines survive corrected bodies.

## 2. What this prices, and what it does not

The composition is in **LABEL SPACE**. The real carrier composites lane appearance into the RGB
render *before* the contest R operator and the frozen SegNet then decides. Overwriting the argmax
assumes perfect label authority at every claimed pixel, so every number here is a strict **UPPER
BOUND** on what any realized composite could deliver. The module's own docstring records that the
naive realized composite *hurt* by +25% (0.00333 → 0.00415, n600). **A ceiling that already fails is
a stronger negative than a realized measurement would be**; a ceiling that passed would still have
owed a realized row.

## 3. Calibration gates — all three exact

| gate | declared in advance | measured |
|---|---|---|
| the re-run forward reproduces md1's retained terminal argmax | bit-for-bit | **0 differing sites** of 6,291,456 |
| the HT numerator reproduces md1's sealed `terminal_d_seg_hat` | 0.0028065999348958334 | **0.0028065999348958334**, gap **0.0** |
| B + H + W + unchanged-correct partitions every site, in integers | 6,291,456 | **exact** on every composition |

The lane class is **self-detected** — smallest area fraction (0.5965% of the frame) and highest
4-neighbour boundary-to-area ratio, which must agree or the instrument refuses. Both point to class
**1**, matching the carrier module's default and the CLAUDE.md canonical order. The row centroids
confirm the rest of that order independently: class 2 at 0.248 (top, Undrivable/sky), class 4 at
0.874 (bottom, MyCar/hood). No index is hardcoded anywhere in this arm.

## 4. The pre-registered rules at the module's own defaults

Fit: 158 lines over 32 pairs (4.94/pair), centerline degree 3, dash gate on, softness 1.0, coverage
threshold 0.5. Band recall 0.5377, precision 0.5628 (DALI). Coded **2,832 B**. Born field before:
`d_seg_hat` **0.0028065999**.

| rule | `d_seg` after | ΔS_seg | persistent removed | H (healed) | B (broken) | harm/removal |
|---|---|---:|---:|---:|---:|---:|
| **(a) REPLACE** | 0.0063536 | **+0.35470** | **43.24%** | 135,420 | 553,845 | **4.09×** |
| **(b) UNION** | 0.0047855 | +0.19789 | 14.60% | 33,045 | 266,490 | **8.06×** |
| **(c) BAND** (dilated 3 px) | 0.0051334 | +0.23268 | 32.86% | 99,150 | 373,635 | **3.77×** |

H / B / W are HT numerator units. The PyAV lineage agrees: 0.0063199 / 0.0047484 / 0.0050948, with
persistent removal 43.68% / 14.91% / 33.49% and harm ratios 4.12 / 8.20 / 3.79. **The verdict is
lineage-independent.**

**Where the harm lands — the mechanism, not a summary.** Rule (a)'s broken sites by ground-truth
class: Road 10,596 · **Lane 15,209** · Undrivable 19 · Movable 486 · MyCar 3,716. The largest single
casualty is **Lane itself**. The demotion clause destroys more correct lane than the carrier's claims
recover, because the born field already places 90% of the lane correctly and the carrier only recalls
54% of it. Replacing a 90%-correct predictor with a 54%-recall one is a loss before any false
positive is counted.

## 5. Optimal form — 162 configurations, none improves

The pre-registered pass ran the carrier at its module defaults, which leave `u_mask_enabled=False`:
the **naive** band the module names as its own failure mode. The optimal-form law makes that verdict
non-binding, so both knob families were tuned to their own optimum.

**Composition knobs** (5 coverage thresholds × 6 uncertainty-gate τ × 3 rules = 90 rows). The
uncertainty gate is the module's designed FP killer, driven here by the born field's OWN top1−top2
margin. Best row: `union, threshold 0.99, τ 0.25` → **+0.008481 S**, persistent removed **0.69%**,
harm/removal 3.76. The gate helps a great deal — it cuts the damage from +0.355 to +0.008 — by
claiming almost nothing (3,302 sites of 6.29 M). It converges on doing nothing, and does not reach it.

**Fit knobs** (2 degrees × 2 dash settings × 3 softnesses × 6 thresholds = 72 rows; coded bytes
2,079–2,832 B). Best by `d_seg`: **+0.15175 S**. Best by recall:

| configuration | recall | precision | FN `d_seg` | FP `d_seg` | ΔS_seg |
|---|---:|---:|---:|---:|---:|
| deg 3, dash gate OFF, softness 2.0, threshold 0.05 | **0.9351** | 0.3725 | **0.000387** | **0.009398** | +0.8166 |
| deg 3, dash gate ON, softness 1.0, threshold 0.75 | 0.5015 | **0.5651** | 0.002974 | 0.002302 | +0.1842 |

**This reproduces FEED-dj on an independent vehicle.** FEED-dj recorded FN `d_seg` 0.00046 (implying
~92% recall against a 0.6% lane fraction) and FP `d_seg` **0.00396** as full authority. Measured here:
FN **0.000387** at **93.5%** recall, FP **0.009398**. Same mechanism, same sign, FP 2.4× larger on
this selection. **The band captures lane SHAPE and pays for it 24× over in false positives.**

Precision and recall trade against each other along the whole sweep and never meet: the
precision-maximising row recalls 50%, the recall-maximising row is 37% precise, and the break-even
bar is 91%.

## 6. The perfect-Lane oracle — the ceiling of the representation class

Perfect Lane authority in both directions, every other class left to the born field. No lane-shaped
object can beat this, so it prices the **class**, not this fit.

| variant (demotion prior for born-Lane false positives) | `d_seg` after | ΔS_seg | persistent removed | B |
|---|---|---:|---:|---:|
| **born field's own runner-up** | **0.0012197** | **−0.15869** | **63.12%** | **0** |
| largest non-hood class (self-detected: Undrivable) | 0.0022128 | −0.05938 | 29.99% | 0 |

`B = 0` is structural, not luck: a site the oracle touches was either already wrong (demotion cannot
break it) or becomes correct. The oracle is a pure monotone improvement.

**It saturates exactly what a Lane-only object can reach.** 63.64% of the persistent HT numerator
touches Lane (130,665 of 205,305) — md1 measured 64.79% of persistent *sites*. The oracle removes
63.12%. **The remaining 36.4% of the persistent set has no Lane in it at all**, and no lane carrier,
however perfect, addresses one unit of it.

## 7. Rate — the axis that was never binding

| quantity | value |
|---|---:|
| LBND2 raw blob, 32 pairs | 8,887 B |
| coded (brotli q11), 32 pairs | **2,832 B** = 88.5 B/pair |
| coder cost in recall | 0.5450 → 0.5377 = **0.73 points** |
| ΔS_rate at n32 coded bytes | +0.0018857 |
| n600 bytes, naive per-pair transfer | 53,100 B — **TRANSFERRED**, conservative |
| ΔS_rate at that n600 cost | +0.035357 |
| oracle ΔS_seg + that ΔS_rate | **−0.12333 net** |
| oracle exchange rate | 2.9885e-6 S/B = **4.49× the rate floor** |

The n32 selection is **non-consecutive** (`SELECTION_IDS` spans 4…573), so LBND2's temporal delta has
almost no correlation to exploit. The per-pair byte cost measured here is an **upper bound** on the
consecutive-n600 cost, which makes the 4.49× a floor on the oracle's efficiency. **The lane manifold
is cheap. Cheapness is not the problem.**

## 8. A silent-truncation defect in the landed coder, found and closed

`_line_to_slot_vec` right-aligns the centerline into a fixed 4-coefficient slot and took `cc[-4:]`.
A `centerline_deg=4` fit was therefore **silently truncated** — its leading coefficient dropped.

MEASURED (n32, DALI, max over forward ∈ [5, 60] m, raw fit vs its own dequantized round trip):

| centerline degree | raw coeffs → shipped coeffs | max lateral error |
|---:|---|---:|
| 2 | 3 → 4 (leading-zero pad, polyval-identity) | 0.0433 m |
| 3 | 4 → 4 | 0.0511 m |
| **4** | **5 → 4 (leading coefficient dropped)** | **23.33 m** |

Two orders of magnitude past quantization, silently. The module already refuses NO-FAKE-style when
`K > _RD_MAX_SLOTS` on the **slot** axis; the same guard was missing on the **coefficient** axis.
Landed at `db8fc4b64`: `_line_to_slot_vec` now refuses a centerline of more than 4 coefficients or a
halfwidth of more than 2, with the measurement in the docstring, and three tests cover pad-is-lossless,
centerline-refuses, halfwidth-refuses. **Live count at landing: 0** —
`tools/measure_third_order_descent_filler.py:167` fits degree 4 for an analytic derivative bound only
and codes degree 2/3 pairs, so no caller's behaviour changes. The 36 degree-4 rows measured before the
guard are excluded from every table above; they were never the best rows.

Two pre-existing failures in `src/tac/tests/test_lane_groundframe_xi_transport_no_collapse.py` assert
a canonical-equation anchor count of 2 against a registry that holds 3 since commit `06c92b624`. They
are unrelated to this change; 87 tests across every direct consumer of the modified module pass.

## 9. GESTALT-DELTA

**Does the 12.75× remaining shrink to within the 1.61× the schedule levers can pay? NO — and the
answer is not close.**

| state | `d_seg` | × the sub-0.12 corner |
|---|---|---:|
| born field, terminal shadow | 0.0028066 | 20.566× |
| md1's persistent floor (perfect optimization) | 0.0017404 | 12.753× |
| **after a PERFECT Lane carrier** | **0.0012197** | **8.9375×** |
| after a perfect Lane carrier, then the 1.61× schedule ceiling | — | **5.5513×** |
| **perfect Lane carrier AND perfect optimization, jointly** | **0.00064189** | **4.7036×** |

A perfect Lane carrier improves the floor by **2.71×** (12.753 → 4.704). It is the largest single
lever this campaign has priced on the born vehicle's accuracy axis. It is also **4.70× short**, and
the shortfall is structural: the persistent set's non-Lane 36.4% is untouched by any lane-shaped
object, and the actually-landed carrier delivers **none** of the oracle's gain.

**Gestalt reading.** md1 closed optimization; gc1 closed capacity; gf2 closed form. This arm closes
the fourth door on the born vehicle's accuracy corner: **class-matched geometric carriers, priced at
their ceiling**. The lane is the persistent set's biggest single component and the corpus's best
lane object cannot pay for it — not for want of bytes (2.8 KB, 4.49× the rate floor) and not for want
of shape (93.5% recall), but because the born field is *already 90% right about the lane*, and a 56%-
precise claim over a 99.8%-correct background loses before it starts. The accuracy half of sub-0.12
on the small born body is now closed by **four** independent instruments, and the CLAUDE.md banner's
live demand — **−42,016 B at held distortion 0.028120** — stands unchallenged as the corner to work.
The durable transferable lesson is narrower than "carriers fail": **an authority-substitution lever
is priced by the incumbent's per-class accuracy, not by the lever's own fidelity.** Any future
carrier arm should compute its break-even precision from the incumbent BEFORE it is built. That is a
$0 arithmetic step this arm can hand forward.

## 10. NEXT_IF_RESUMED

The charter's own next step — "the born trainer with Lane held by the carrier in-loop" — is
**NOT authorized by this result** and should not be chartered. The ceiling it would chase is
8.94× the corner, and the carrier that would hold Lane in-loop delivers none of that ceiling.

What the measurement does authorize, in order:

1. **Nothing on the born vehicle's accuracy corner.** Four instruments now agree. Redirect to the
   rate corner the banner names.
2. **The break-even-precision screen as apparatus** (~1 hour, $0): a helper that, for any proposed
   class-authority substitution, computes `P(incumbent wrong | GT = c)` and
   `P(incumbent correct | GT ≠ c)` from a retained argmax and returns the required precision. This
   arm's whole negative is one line of that arithmetic — 0.909 against a 0.565 ceiling — and it was
   computable before any fitting. Wire it into the charter lint so a carrier charter must carry it.
3. **If a Lane lever is ever revisited**, the object to build is not a wider or better-fitted band.
   It is something that raises PRECISION on the 10% of lane the born field misses without claiming
   the 90% it already has — i.e. a *correction* keyed to the incumbent's own uncertainty, not an
   authority that overwrites it. The τ = 0.25 gate row is the seed of that shape: it cut the damage
   44× by claiming almost nothing. The question it leaves open, and this arm did not answer, is
   whether any claim rule restricted to the incumbent's low-margin sites can clear 0.909 precision.
4. **The vr1 row-10 headline needs correcting in place** so the next reader does not re-inherit
   "Lane band d_seg 0.00087" as an achieved number.

## Equations leg (`tac.canonical_equations`)

* **`v8_geometric_rate_decomposition_v1`** — this arm measures the Lane carrier of that
  decomposition on a new vehicle and confirms anchor 5's shape: the analytic composite's cost is
  dominated by class-authority false positives, not by rate. It adds the missing operating-point
  reading — the carrier's rate efficiency is real (4.49× the floor, oracle) while its realized
  authority is net-negative on every one of 162 configurations.
* **`checkpoint_trajectory_error_partition_v1`** — md1's partition is the object scored against; this
  arm reports what a class-matched carrier removes from each of its classes, and measures that
  63.64% of the PERSISTENT numerator touches Lane while a perfect Lane object removes 63.12% of it.

No new equation is registered: the finding is a ceiling price against two existing laws, and its one
generalizable form (the break-even-precision identity) is named as NEXT_IF_RESUMED item 2, to be
registered when it lands as code with a consumer.

## Custody

`/Volumes/APDataStore/pact/ddm_lb1_lane_band_ceiling/` — `CUSTODY_MANIFEST.json`
(schema `ddm_lb1_custody_manifest.v1`, 10 files, 22,343,362 B, every file sha256'd):
`terminal_shadow_top2.npz` (top-1, runner-up and margin per site, the payload the composition needs)
· `lane_band_lbnd2_dali.bin` / `.br` (the coded carrier, KEPT, not just its length) ·
`carrier_coverage_dali.npz` · `composed_argmax.npz` (all three composed fields) · `FORWARD.json` ·
`PRICE_dali.json` · `OPTIMAL.json` · `FITSWEEP.json` · `SUMMARY.json`.
0 Metal · 0 Modal · 0 contest eval · $0.
