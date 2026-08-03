---
arm: ddm_sq1
title: "η_seg through the real receiver (gp1 F1) + the sealed mg1 hinge A/B (bo1 §4)"
utc: 2026-08-03
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
slot: "scorer slot HELD (one full-n600 job at a time, chunk <=120)"
baseline_named: "live best S = 0.826496209256714 at 353,808 B (v4d_cx1_pj2ix2, pu2 base); d_seg 0.004311794704861111; d_pose 0.0025513987495742437; seg leg 0.4311790; pose leg 0.1597310"
floor_named: "PR130 = 0.172141 (the BAR; lessons-only lineage, never ours). gap = 0.654355209256714"
verdict_scope: "see per-result scoping; no family negative is licensed by this unit"
consumes:
  - ".omx/research/ddm_gp1_selective_gt_student_pricing_20260803.md (F1 pre-registration, honored verbatim)"
  - ".omx/research/ddm_bo1_seg_base_objective_menu_order_20260803.md §4 (sealed A/B spec, honored verbatim)"
  - "gc16 §9 P1 three revisions (noise floor bundled / P7 pose-null rider / edge-weighted branch)"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_sq1 — Job 1 (η_seg) and Job 2 (the sealed hinge A/B)

## §0 STATUS

- **§1 PRE-REGISTRATION — written and committed BEFORE any scorer forward.** Binding.
- §2 Job 1 (η_seg) — INTERIM verdict, committed the moment Job 1 completes.
- §3 Job 2 (hinge A/B) — filled after Job 1.

---

## §1 PRE-REGISTRATION (binding; committed before any measurement)

### 1.1 The quantity

`gp1` §5 defines **η = (flips actually fixed) / (flips described)** and states every net-ΔS in
its ladder scales by η. `gp1`'s entire table prices **descriptions**; nothing in it builds the
mechanism that turns *"band pixel p should be class c"* into an RGB change the frozen SegNet
actually argmaxes differently. This unit measures that mechanism.

**Primary (the number F1 tests) — NET, whole-frame accounting:**

```
eta_net = (flips_before - flips_after) / n_described
```

`flips_before` / `flips_after` are counted over the **whole 384x512 scorer field**, not only
inside the band. This is deliberate: a realization that fixes 900 band pixels while breaking
400 elsewhere has NOT delivered 900 flips of value, and `gp1`'s gross column assumed the
described flips simply vanish. Whole-frame accounting is the only definition that makes
`eta_net * gross` an honest estimate of the realized gain.

**Secondary (diagnostic only, never the F1 verdict):** `eta_inband_raw` = in-band flips
repaired / in-band flips described. The gap `eta_inband_raw - eta_net` **is** the collateral
damage and is reported as its own number.

### 1.2 Why a ladder of mechanisms, and what each rung licenses

η is **mechanism-scoped**: it is not a property of the band, it is a property of
(band, realization rule). A single number would be uninterpretable. Pre-registered rungs, all
edits applied at **camera resolution** (874x1164 uint8 — what the receiver actually ships),
all scored through the real path `edited camera RGB -> D (bilinear 874x1164 -> 384x512,
antialias=False) -> frozen CPU-torch SegNet -> argmax`:

| rung | edit | receiver-legal? | what its η licenses |
|---|---|---|---|
| **P0** | paste GT camera RGB over the **entire** frame_1 | no (positive control) | **must give η = 1.000 and flips_after = 0.** Validates decode alignment, D, scorer weights, GT cache. If P0 fails, NOTHING else in §2 is admissible. |
| **L0-r1** | paste GT camera RGB **only** in the r=1 band's camera blocks | no (content oracle) | the **locality ceiling** for truth-content band-local edits. If η(L0-r1) <= 0.583, A3 dies for the whole band-local-truth family, not just for one rule. |
| **L0-r2, L0-r3** | same at dilated bands | no | the **locality curve** — re-prices gp1's A4 (r=2, 408,244 B) with a real η, and says whether "widen the band" is the cure. |
| **L1-r1** | in-band camera pixels set to the **decoded frame's own** per-class anchor colour for the shipped target class | **yes** — computable from (decoded RGB, decoded L*, shipped label c); zero scorer weights | the **legal-today** realization. This is the η that decides whether row A3 is live *now*. |

**Address legality.** The band is computed from the **decoder's own argmax** `L*` (gp1's F4
concern: gp1 used a near-GT proxy render). This unit computes `L*` by running the frozen SegNet
on the **actual decoded frames** from the shipped archive, so F4 is answered as a by-product,
not assumed.

### 1.3 Pre-registered thresholds (gp1 F1/F2 verbatim)

| falsifier | threshold | kills | this unit's reading |
|---|---|---|---|
| **F1** | `eta_seg <= 0.583` on >= 32 pairs through the real receiver | gp1 A3 (free-band ALL, 367,523 B, bound -0.17466 S = 28.2% of gap) | **η_net(L1-r1)** decides A3-live-today; **η_net(L0-r1)** decides A3-live-ever-band-local |
| **F2** | `eta_seg <= 0.594` on the Road<->Lane subset | gp1 B2 (174,120 B, bound -0.07932 S) | same rungs restricted to the Road<->Lane edge |
| **F4** | decoder's **actual** `L*` band captures **< 90%** of flips at r=1 | A3, B2 and rz1's 93.53% jointly | measured directly here (gp1 proxy: 97.26%; rz1: 93.53%) |

**Routing pre-registered now, so the result cannot be re-narrated after the fact:**

- η_net(L1-r1) > 0.583 -> **A3 SURVIVES**; re-price A3 at `eta * gross` and hand MAIN a live row.
- η_net(L1-r1) <= 0.583 < η_net(L0-r1) -> **A3 dies for the built rule, survives for the
  family**; the gap between the two ηs is the realization headroom and NAMES the next build
  (a better label->RGB rule), verdict_scope FORMULATION.
- η_net(L0-r1) <= 0.583 -> **A3 dies for band-local realization entirely**; verdict_scope
  FORMULATION-BUT-WIDE (band-local truth-content), and the seg lane routes to bo1 §4's
  named fallback: the sub-cell realization instrument (dd1 Lane perpendicular offset), i.e.
  non-band-local carriers.
- Any rung > 0.583 only at r>=2 -> the cure is band width; re-price A4 and report the r at
  which the row is still net-negative after its own larger rate cost.

### 1.4 Denominators carried on every number

- total gap = **0.654355209256714** (live best 0.826496209256714 - PR130 0.172141).
  1% of gap = **0.0065436 S**.
- seg leg = 0.4311790; 1 flip = `100 / (600*384*512)` = **8.47800925926e-7 S**;
  1% of total gap = **7,718 flips**.
- pose: priced at the **CURRENT** `dS/d(d_pose_mean) = 5/sqrt(10*0.0025513987) = 31.3027`
  (K3: never a shelf price - `wf2` §5, shelf prices stale >= 2.22x).

### 1.5 Pair selection and the m88 control (MEASURED, pre-run)

32 pairs, **stratified systematic on the flips-sorted order** (32 equal strata, median pair of
each) - deliberately NOT a prefix.

| quantity | subset (n=32) | population (n=600) | ratio |
|---|---:|---:|---:|
| mean flips/pair (governing) | 845.4688 | 847.7333 | **0.997329** |
| median flips/pair | 805.5 | 807.0 | 0.998 |
| *contrast: prefix[0:32]* | 776.5625 | 847.7333 | *0.916046* |

Ratio 0.9973 => representative on the seg governing quantity. The prefix would have been 8.4%
off - `m88` in one line.

**Pose is a different population and is scoped accordingly.** `gp1` §1 measured the pose axis
at **4.6x** skew (vs 1.05x for seg); a subset representative on flips is **not** representative
on d_pose. Therefore all pose collateral below is reported **per pair and as a subset mean
with its own subset-vs-population ratio printed**, and is **never** extrapolated to n600.

Selected pairs: `[0, 20, 32, 48, 115, 154, 170, 179, 180, 195, 196, 211, 214, 242, 261, 288,
357, 365, 370, 394, 400, 420, 433, 439, 471, 474, 485, 501, 504, 514, 521, 533]`

### 1.6 Pose collateral is MANDATORY (not an add-on)

The corrections touch **frame_1**, which **both** scorers read - `pz1` measured that PoseNet
and SegNet make the *identical* `interpolate(..., segnet_model_input_size)` call
(`upstream/modules.py:73` and `:109`), so **the two scorers share the same `D`**. A seg-only
verdict is forbidden (`sf1`/`uv1` 3,019x law). For every rung, `d_pose` is recomputed with the
frozen PoseNet on `(decoded f0, edited f1)` against `(GT f0, GT f1)`, and any regression is
priced at the current `dS/d(d_pose)` before the rung's net is stated.

### 1.7 Positive controls that must pass before any verdict

| # | control | expected | source of truth |
|---|---|---|---|
| C1 | per-pair flip cache reproduces gp1 | 508,640 flips, d_seg 0.004311794704861111 | **PASS, pre-run** (gp1 + pu2 both exact) |
| C2 | recomputed `L*` argmax on decoded frames == `cx1_argmax_n600.npy` | exact on all 32 pairs | this unit |
| C3 | recomputed GT argmax == `gt_argmax_n600.npy` | exact on all 32 pairs | this unit |
| C4 | **P0 full-frame GT paste** | η = 1.000, flips_after = 0 | this unit |
| C5 | baseline d_pose on the 32 pairs vs `pz1` per-pair cache | agree | this unit |

**If C2/C3/C4 fail the harness is untrusted and no η is admissible** (the `m50` vacuity law:
an empty or misaligned scope must report VACUOUS, never PASS).

---

## §2 JOB 1 RESULT — pending

## §3 JOB 2 RESULT — pending

## §4 NEXT-IF-RESUMED — pending
