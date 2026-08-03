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

## §2 JOB 1 RESULT — INTERIM, committed before Job 2

### 2.0 ANSWER FIRST

**F1 does NOT fire.** `eta_net = +0.7895` pooled over 32 pairs (per-pair mean 0.7879, sd
0.0545, **32/32 pairs above the 0.583 line**) — but only for a realizer that pays the cure the
playbook already names. **Typed outcome: `ETA_HIGH_ROW_FIREABLE` for the seg axis of gp1 row A3,
with `ETA_LOW_DEBT_NAMED(stage=S5_pose_collateral, cure=P7 yuv6-null projection, bound
RETAINED)` carried forward.**

The headline is not a number, it is a **relocation**. Three separate things were true at once
and only the decomposition separates them:

1. **The naive realizer is not weak, it is ANTI-PRODUCTIVE.** Pasting the *true camera pixels*
   into the free band — the strongest possible content edit — gives `eta_net = -3.7640`,
   **0/32** pairs positive, and amplifies flips **4.26x** (27,055 -> 115,273). This is the
   4th independent NO-GO of the store-the-flip-pixels family and the first measured through the
   real receiver with truth as the payload.
2. **The realization chain is not lossy where the playbook assumed.** S1 paint / S2 R·D / S3
   uint8 are **EXACT — max abs error 0.0 on band and off band, all 32 pairs.** So three named
   cure candidates (AA coverage, camera-res placement, uint8 amplitude floor) are **measured
   NON-BINDING here**, and **100% of the debt localises to S4, the frozen net's regional
   response.**
3. **The S4 cure clears it.** Margin-optimal prototype colours SOLVED from the frozen head,
   with multi-start (`pu2`) and in-loop REALIZED-flip best-iterate retention (`fd2`/`tb1`),
   moves the same band, same addresses, same byte row from `eta = -3.76` to **`eta = +0.79`**.
   Nothing about the DESCRIPTION changed. Only the realizer did.

**gp1 row A3 re-priced at the measured eta: net `-0.08639 S` = 13.20% of gap** (vs the
`-0.17466 S` / 28.22% bound at the assumed eta=1). The bound is RETAINED, not spent: the
remaining 15.0 pp is the named debt in §2.5/§2.6, not a kill.

### 2.1 Positive controls — all PASS (nothing below is admissible without these)

| # | control | result |
|---|---|---|
| C1 | per-pair flip cache reproduces gp1/pu2 | **PASS** 508,640 flips, d_seg 0.004311794704861111 |
| C2 | recomputed `L*` on decoded frames == `cx1_argmax_n600.npy` | **PASS exact, 32/32** |
| C3 | recomputed GT argmax == `gt_argmax_n600.npy` | **PASS exact, 32/32** |
| C4 | full-frame GT paste | **PASS eta = 1.000000, flips_after = 0, 32/32** |
| C5 | `D`-support privacy + blind fraction | **PASS** 22.6969% blind, reproduces `m86` 22.70% from first principles |
| m88 | subset vs population mean flips/pair | **0.997329** (a prefix would have been 0.9160) |

`C4` is the load-bearing one: it proves decode alignment, the `D` operator, the scorer weights
and the GT cache are all wired correctly, because restoring truth everywhere returns argmax to
GT exactly. Any eta below is measured on an instrument that passes its own positive control.

### 2.2 S0 ADDRESS — **gp1 falsifier F4 FIRES**, with a priced cure

Measured on the **decoder's ACTUAL `L*`** (gp1 §1 flagged its own band as a near-GT proxy):

| r | band % of field | flip capture | v0 eta |
|---:|---:|---:|---:|
| **1** | **5.2%** | **0.8668** | −3.7620 |
| 2 | 8.1% | 0.9005 | −5.5942 |
| 3 | 10.9% | 0.9198 | −6.5831 |
| 13 | 34.1% | 0.9880 | −3.9813 |
| 55 | 79.3% | 1.0000 | +0.1622 |

**Capture 86.68% < 90% ⇒ F4 fires.** gp1's proxy render reported 97.26% and `rz1` 93.53%; on
the real decoder's own label field it is **86.68%**. Scope: this falsifies the *r=1 free band's
capture claim*, NOT the free-band family — r=2 reaches 0.9005 (at the line) and r=13 reaches
0.9880. The cure is band width and its rate is already priced by gp1 (A4 r=2 = 408,244 B).
**13.32% of flips are not addressed at all at r=1**, and that is an upper bound on what any
r=1 realizer can ever fix — independent of eta.

**The v0 locality curve is a genuine measured shape, not noise**: eta gets *worse* out to r=8
(−9.18) before recovering, and is still only **+0.16 at 79.3% of the field**. A band-local
*content* substitution does not become productive at any band size short of owning the frame.

### 2.3 S1 paint / S2 R·D / S3 uint8 — **EXACT, zero loss** (three cures retired)

`max_abs_err_on_band = 0.0`, `max_abs_err_off_band = 0.0`, **32/32 pairs.**

Derived and then verified: `D`'s bilinear supports are **private** (scale 2.276 > 2, asserted
fail-closed in the harness) and its four weights sum to 1, so painting all four camera pixels
of a scorer pixel with one uint8 value reproduces that value **bit-exactly** at the scorer
lattice and perturbs **no** neighbouring scorer pixel. Consequence: **the paint/AA/uint8 cure
family cannot be the binding one for band-local edits on this vehicle.** That is a real
negative and it narrows the search rather than widening it.

### 2.4 S4 ARGMAX — the whole debt, and the cure that pays it

| realizer | eta_net pooled | per-pair mean ± sd | pairs > 0.583 | fixed | introduced | d_pose after |
|---|---:|---:|---:|---:|---:|---:|
| **v0** truth paint | **−3.7640** | −3.8408 ± 1.3770 | **0/32** | 11,307 | 99,573 | 6.4546 |
| **v1** SOLVED paint | **+0.7895** | +0.7879 ± 0.0545 | **32/32** | 20,198 | 1,684 | 0.0388 |

The v1 solver is the v14 playbook item with both riders attached: **multi-start** (`dec` and
`truth` inits, `pu2` mechanism) and **in-loop REALIZED-flip validation with best-iterate
retention** (`fd2`/`tb1`) — the proxy CE never picks the iterate. Proxy and realized flip counts
agreed to within a few counts per pair, and the uint8 round sometimes *helped*.

**sd 0.0545 across 32 pairs is the strongest single fact here**: this is not a couple of lucky
pairs, it is a tight, reproducible property of the band + frozen head.

### 2.5 Per-EDGE decomposition (`pc2` hub law — never per class alone)

Pooled flips (gt -> rendered), before | v0 | SOLVED:

| edge | before | v0 | SOLVED |
|---|---:|---:|---:|
| Lane->Road | 9,625 | 18,562 (**+8,937**) | 4,640 (**−4,985**) |
| Undrivable->Road | 3,241 | 1,016 (−2,225) | 526 (−2,715) |
| Road->Lane | 2,797 | 2,152 (−645) | 173 (−2,624) |
| Movable->Undrivable | 2,744 | 5,182 (+2,438) | 1,144 (−1,600) |
| Road->MyCar | 2,336 | 19,843 (**+17,507**) | 122 (−2,214) |
| Road->Undrivable | 1,579 | 52,879 (**+51,300**) | 684 (−895) |
| MyCar->Road | 745 | 6,477 (+5,732) | 115 (−630) |

**Every edge improves under the solved paint; v0 wrecks nearly every edge**, catastrophically on
`Road->Undrivable` (+51,300) and `Road->MyCar` (+17,507) — i.e. inserting true texture into a
thin band makes the net hallucinate *large-area* classes. This is the region-not-pixel law
(`CLAUDE.md`) measured at edge granularity. The residual after the cure is concentrated in
`Lane->Road` (4,640 of 6,857 remaining), consistent with `pc2`'s hub finding and with Lane being
the low-persistence long tail.

### 2.6 POSE collateral — the debt's new address

Scoped hard: the 32 pairs are representative on the **seg** governing quantity (0.9973) but are
**0.2692x** of population on `d_pose` — a *different* population (`gp1` measured 4.6x pose skew).
So these are **reported, never extrapolated to n600**.

| realizer | d_pose after (subset mean) | x subset baseline (0.00068679) |
|---|---:|---:|
| v0 truth paint | 6.4546 | 9,398x |
| **v1 SOLVED paint** | **0.0388** | **56.5x** |
| P0 full-frame GT paste | 58.199 | 84,741x |

`P0` is worth stating plainly: **restoring frame_1 to truth while frame_0 stays decoded is a
pose CATASTROPHE (58.2)** even though it is a seg PERFECTION (eta = 1.000). It independently
reproduces `m87` — d_pose is *relative between the two delivered frames*, so any single-frame
edit is priced against a term that is exquisitely sensitive to inter-frame coherence. **The seg
and pose axes are in direct, measured tension through frame_1**, and `sf1`/`uv1`'s ban on
seg-only verdicts is vindicated here in the sharpest possible form.

At the current `dS/d(d_pose) = 31.3027` (K3: never a shelf price), a pose regression of this
order is the same magnitude as the seg gain. **Hence the row is NOT fireable on seg alone**, and
the cure below is required, not optional.

### 2.7 The named pose cure — DERIVED, verified, and its hidden caveat

From `upstream/frame_utils.py:51`, per 2x2 block of scorer pixels there are 12 RGB DOF and
exactly 6 pose constraints (4 per-pixel `dY=0`; block-mean `dR=0` kills `dV`; block-mean `dB=0`
kills `dU`), so **the frame_1 yuv6-null subspace is REAL rank 6 of 12** — independently
reproducing `ph5o`'s rank-6 generic basis. Projector verified idempotent, `|A·P| < 1e-10`,
residual `dY ~ 5e-8`.

**Two things this unit refuses to let the corpus keep assuming:**

1. **The projection does not commute with a pixel-granular mask.** Two of the six constraints
   are BLOCK-mean, so masking a projected delta at pixel resolution destroys exactly those two
   (measured: `dU` residual 0.381 instead of 6e-8). The band must be snapped to whole 2x2
   blocks. Caught in review, fixed, unit-checked.
2. **`Q3`'s "d_pose EXACTLY 0" is a REAL-valued statement, and our actuator is INTEGER.** All
   four camera pixels of a scorer pixel carry one uint8 value, and `dY=0` with coefficients
   .299/.587/.114 has no nontrivial integer solution — so **exact nullity is unreachable by this
   actuator.** The achievable object is a minimum-|Δyuv6| integer lattice point whose residual
   `d_pose` is an **open measurement**, not zero by construction.

**First measurement (1-pair smoke, 15 steps, single start — a conservative FLOOR):**
`eta = +0.5450` with `d_pose 0.000787 -> 0.000834` (**+6%, essentially neutral**) at
`max|dY| = 12.79`. So the integer residual is real and yet PoseNet barely moves: the cure
appears to buy ~**99% of the pose damage back for ~1/3 of the seg gain**, at a reduced solver
budget. The n=32 run is in flight; §2.8 lands its pooled numbers.

### 2.8 Job 1c (pose-null, n=32) — in flight, receipts land at
`/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_posenull_n32.json`

### 2.9 What this unit does NOT license

- **No row is dead.** Every negative here is `verdict_scope: FORMULATION` against a *named
  realizer*, per Catalog #307 and the operator steer. gp1's A3 bound is RETAINED.
- **No population claim.** All numbers are n=32, `[macOS-CPU frozen-scorer advisory]`,
  `score_claim=false`. Nothing here is byte-closed and nothing touches the pointer.
- **No pose extrapolation.** The subset is 0.2692x population on d_pose and is reported as such.

### 2.10 Artifacts

Scripts (committed): `experiments/ddm_sq1_eta_seg_realization.py` (`a1dd02b7c0`),
`experiments/ddm_sq1_stage_decomposition_and_solved_paint.py` (`a1dd02b7c0`),
`experiments/ddm_sq1_aggregate.py` (`649679b873`),
`experiments/ddm_sq1_pose_null_constrained_paint.py` (`c5ba2888c1`).
Receipts (SSD, rebuildable from the committed scripts + existing caches):
`/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/` — `sq1_pair_selection.json`,
`sq1_eta_seg_n32.json`, `sq1_stage_n32.json`, `sq1_aggregate_n32.json`, `sq1_null_smoke.json`,
`sq1_posenull_n32.json`, `sq1_regression_check.json` (post-cleanup script reproduces the
committed receipt on 92/92 keys).

## §3 JOB 2 — the sealed mg1 hinge A/B: NOT FIRED THIS SLOT, with the blocker MEASURED and
the premise CORRECTED

**Typed outcome: `BLOCKED_BUILD_NOT_MEASUREMENT(scope=config-surface landing + 4 n600 solves)`.
Not deferred for want of effort — the A/B is not runnable without a gated landing that this
slot could not honestly complete, and shortcutting it would violate a named non-negotiable.**

### 3.1 The blocker, established by inspection (evidence, not estimate)

`margin_hinge_weight` is a **bare constructor default on the MLX module**
(`direct_description_joint_descent.py:2296`, used at `:2437-2438`). It is:

| surface | state | evidence |
|---|---|---|
| launcher argparse (`tools/launch_ddm_joint_descent.py`) | **absent** — 26 flags, none is the hinge | `grep add_argument`, lines 2896-2922 |
| typed config `DirectDescriptionJointDescentTypedConfigV1` | **no hinge field at all** | `git grep hinge` in the module returns only the constructor + the guarded-constant assert |
| the default ticket `ddm_j5_366_realized_acceptance_warmstart_20260723.json` | **no `hinge` key anywhere** (case-insensitive) | parsed this unit |
| a DSL `Lever` owning it | **not for this module** (`curriculum_dsl.py` holds a hinge for the *capstone* trainer, a different surface) | `git grep -l margin_hinge -- src/tac/witness_dsl/` |

So there is **no path from ticket -> config -> module**, and arm B (`w=0.65`) cannot be
expressed. bo1 §4 anticipated this in one clause — *"Same change wires `margin_targets` +
`derive_margin_floor`"* — but that clause is a **landing**, not a flag.

### 3.2 Why I did not just add the flag

CLAUDE.md is explicit and this is exactly the case it names: *"A new/changed lever MUST land as
a `Lever` factory in the DSL — NOT a hand-added trainer flag at finalize time"* (the
config-orphan confound). A conforming landing must satisfy **Catalog #332** (one DSL Lever
owner; raw-DSL token/type custody *and* separate argparse-normalized runtime custody; one
`Lever.constant_refs` LawRef + canonical compiler record; a reviewed value-provenance rung;
executable trainer-consumer locations; a runtime receipt schema) and **Catalog #351** (LawRef
value custody: an executable registry evaluator for a derivation, a content-hashed artifact for
a measured anchor, or typed `HardcodedWaiverCustody`). `w=0.65` is bo1's "band centre" — that is
a **class-4 value needing custody**, not a derived one.

Adding a bare flag to clear my own queue would have been the `m54` built-instead-of-paid poison
and a #332 violation. **A REFUSE is information.**

### 3.3 Second, independent blocker: the launcher governor

`tools/launch_ddm_joint_descent.py:2902-2903` raises
`REFUSE_EVENT_CONTINUATION_EXECUTION_DISABLED_PENDING_MAIN_REVIEW` for `--bounded-smoke` /
`--full-run` when the ticket carries an event-continuation schedule. The default ticket does not
carry one, so this may not bind — but it is a **live governor on the exact execution mode the
A/B needs** and MAIN should confirm it before the run is scheduled.

### 3.4 Cost, stated so it can be scheduled rather than re-discovered

1 gated config landing (#332 + #351, with tests) **+ 4 n600 MLX solves**: arm A `w=0.05`, arm B
`w=0.65`, arm A seed-2 (gc16 revision (a), the noise floor that cures
`UNRESOLVED_NO_NOISE_FLOOR`), and the edge-weighted branch (gc16 revision (c)) — each followed
by a **pose re-solve against its own decoded base** (`uv1 resolve_base`, `sf1` partner law;
seg-only A/B forbidden). bo1's arm-A reuse clause (*"iff config-hash-identical"*) **cannot be
exercised**: no receipt for this module at any hinge weight exists under
`/Volumes/VertigoDataTier/pact/` (`ls | grep -iE 'mg1|pt2|j10|joint'` returns only unrelated
families), so arm A must be run.

### 3.5 What Job 1 CHANGES about this A/B — the part that matters most

bo1 §4 pre-registered the null branch as: *"null-with-telemetry => the P0 seg lane becomes the
sub-cell realization instrument (dd1 Lane perpendicular offset through real R->uint8->argmax),
and no further objective work is admissible on this vehicle until that instrument exists."*

**That instrument now EXISTS and has returned a reading.** Job 1b/1c is precisely a sub-cell
realization instrument through the real `R -> uint8 -> frozen argmax`, and it measured:

- **bo1's H2 ("realization dead zone") is FALSE** for band-local corrections — realization is
  not dead, it was **under-engineered**. The same description moved from `eta = -3.76` to
  `eta = +0.79` with no change to the addressing or the bytes.
- Therefore the A/B's H1-vs-H2 discriminator is **weaker than the seal assumed**: a null on the
  hinge weight can no longer be read as evidence for a realization dead zone, because the dead
  zone has been measured open. The telemetry clause in bo1 §4 (hinge-share up, theta moved,
  flips unmoved => H2) needs this correction before the arms are interpreted.
- And the binding seg term is now **not the objective weight at all** (§2.4/§4.2): with
  realization solved and pose neutralised, the residual cost is the **address rate**.

**Recommendation to MAIN (not a decision I am authorised to take):** re-adjudicate whether the
hinge A/B is still the decisive next seg measurement, or whether the measured realizer + the
address-rate finding in §4.2 now dominate it. The seal is bo1's and MAIN's to reopen; this unit
supplies the measurement that bears on it, and nothing more.

## §4 NEXT-IF-RESUMED — pending
