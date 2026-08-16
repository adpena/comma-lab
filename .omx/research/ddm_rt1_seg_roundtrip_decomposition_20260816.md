---
arm: ddm_rt1
title: "the seg axis is one pixel wide: 99.22% of hv1's scored flips sit ON the transmitted label boundary, the round trip is 33,743 flips (0.028604 S) MEASURED, flat paint is 35.4x worse, R supplies exactly zero, td1's r is 0.8492 -- and the free-band correction channel passes its coder gate at 32,270 real bytes then fails its realization bar (eta 0.6235 at n=9 vs 0.753, 0-of-n above it at every n) and is a NON-SUPPLIER (0.0183 S seg gain vs 0.0221 S rate cost; total +0.0025 S after correcting a pose-aggregation error in my own arithmetic), so the seg axis routes to the renderer"
utc: 2026-08-16
charter: ".omx/research/ddm_rt1_seg_roundtrip_decomposition_charter_20260816.md"
axis: "[macOS-CPU advisory] frozen CPU-torch SegNet -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "per-leg INSTANCE on the hv1 ep0634 vehicle; family verdicts only where named"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_rt1 — where the 0.0282 S round-trip loss is manufactured

STORES CONSULTED: the rt1 charter · td1 memo `ddm_td1_token_drop_schur_arithmetic_20260816.md`
+ `TD1_ATTRIBUTION.json` · v14 `ddm_v14_realization_fidelity_603_DAG_FEED_20260722.md` +
`..._canonical_equations_20260722.md` · fl1 `ddm_fl1_perclass_flicker_floors_20260731.md` ·
wc1 `ddm_wc1_decode_wallclock_verdict_20260816.md` (admission receipt
`ddm_wc1_advisory_fast_path_admission.v1`) · ns1
`ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md` · hv1 ep0634
`FINAL_RESULT.json` · sq1 `ddm_sq1_eta_seg_and_hinge_ab_20260803.md` §2.3/§2.4 · lr2
`ddm_lr2_legal_realization_ladder_20260804.md` · memories [[m88]] [[m96]] [[m91]] [[m95]].

## ANSWER FIRST

td1 said ~95% of our seg term is manufactured between the labels we ship and the argmax the
scorer reads back, and asked where. **It is manufactured on a curve one pixel wide.**

1. **The round trip is 33,743 flips = 0.028604 S — measured, not modelled** (td1 could only infer
   it). That is **96.6%** of the seg axis and **2.98×** the whole remaining sub-0.15 gap.
2. **99.22% of it sits exactly ON the transmitted label boundary.** The label interior — 88.4% of
   the field — carries **7 flips in 104 million pixels**. The boundary flips at 203,000× the
   interior rate.
3. **93.9% of flips already have the correct class one pixel away**, and only ~5.7% of the axis is
   a systematic area bias. The rest is symmetric sub-pixel edge jitter.
4. **The Road↔Lane edge alone is 43.4%** (0.0129 S). Three edges carry 80.4%.
5. **Every stage the charter proposed as a supplier is not one.** S2 flat-prototype paint reads
   back **35.4× worse** than the trained render, so it is a ceiling, not a floor — the v14 stage
   taxonomy does not nest on this vehicle and S1 is *negative*. S3 (the R operator) supplies
   **exactly zero** flips on piecewise-constant content. S4 (GT flicker) coincides with 27.7% of
   flips at 52× enrichment but is a bound on smoothing cures, not a hard floor.
6. **The residual is a TIE, not a wall.** At **98.3%** of flips the wanted class is already the
   runner-up, and the median logit deficit is **0.105** — half the axis needs less than 0.1
   logits, 84.5% less than 0.3, none more than 3. Correct pixels, by contrast, sit at margins of
   3–10. So the render put the scorer within a hair of the answer and lost on the last step.
7. **The one zero-byte post-hoc cure is CLOSED.** A flat-anchor repaint of the free boundary band
   costs **+1.3808 S** (47.6× worse) — and is worse than repainting the whole frame flat, because
   a local flat patch manufactures an edge the scorer believes.
8. **Free bonus: td1's H2 is answered.** `r_observational = 0.8492`, against td1's 0.8393
   admission bar — its token-drop refusal stands, by 1.2%.

**The lever** is sub-pixel edge placement on the Road↔Lane boundary. **The one byte-carrying
candidate** is a correction channel on that boundary — free support, priced in §3.3 and then
actually built and raced in **§5**.

**§5 (follow-on #1, executed): the coder gate PASSES at 32,270 real verified bytes vs the
35,117 B bar — but the family is bounded and the coder was never the binding constraint.** My
§3.3 guess that a structured coder could halve the i.i.d. floor is **REFUTED**: the flips are
isolated single pixels (mean run 1.110), so the best real coder beats i.i.d. by 2.5% and the
ceiling of all free conditioning is 12.2%.

**§6 (follow-on #2, executed): the channel fails its realization bar and cannot supply the gap.**
Realization efficiency under the pose constraint, measured on hv1 for the first time, is
**η = 0.6235** (n=9 seeded-random pairs, **0 of 9 above the bar**) against the required **0.753**.
Seg+rate net is **+0.00381 S**; the pose leg pays back **−0.00129 S**, for a **total of
+0.00252 S**. The channel is a **non-supplier**: its whole seg gain at η=0.62 is **0.0183 S against
a 0.0221 S rate cost**, so it cannot cover a −0.0096 gap whichever way the pose term settles.

η is the stable quantity across the landing sequence (0.644 → 0.624 from n=3 to n=9, sd ~0.07,
**0 of n above the bar at every n**). The pose leg is the noisy one: its payback shrank from
×0.431 (n=6) to ×0.713 (n=9), which moved the total from a momentary **+0.00029 break-even** back
to **+0.00252**. My own small-n caveat on that figure was correct and the later rows confirmed it
— §6.2b carries the trajectory.

Getting a trustworthy answer required finding and fixing **three defects in my own work** — two
in the instrument, caught by a positive control (§6.1: sq1's objective and sq1's edit support each
fail to transfer to a 15×-smaller residual, and a NO measured on either would have been an
artifact reported as physics), and one in my own arithmetic (§6.2b: I aggregated the pose leg as a
mean of per-pair ratios when the scorer averages d_pose itself — the two disagree in **sign**).

**Net: every post-hoc lever on the seg axis is now bounded — flat paint 35× worse, band repaint
+1.38 S, correction channel a non-supplier (+0.0025 S at n=9) — so the whole seg axis routes to
the renderer's own training.** lr2 closed this family at 14.6× this vehicle's flip count; §5–§6
re-priced it at *this* operating point, and it still cannot supply the gap — now with the reason
measured rather than inherited, and with the one number I got wrong corrected in §6.2b.
**Pointer UNMOVED.**

## §0 Prior-law prediction lines (stated BEFORE the measurement, per the anti-re-anchor law)

1. **td1's split** — 1,717 label errors vs 34,930.6 scored flips. PREDICTION: my advisory
   instrument must reproduce the 1,717 exactly (it is a scorer-free set comparison) and land
   near 34,930.6 on d_seg. Any large gap is CPU/CUDA drift and must be reported as such, never
   silently absorbed.
2. **v14's FORMULATION verdict** (`#603`) — "semantic-mask painting with fixed RGB class
   prototypes cannot recover the promised cell-space score under this receiver". PREDICTION:
   the flat-prototype paint leg will read back its own label field POORLY. v14 measured d_seg
   0.0274 for that family on the describe-line vehicle.
3. **sq1 §2.3/§2.4** — on the v4d vehicle, band-local *content* edits are catastrophic
   (truth paint `eta_net = -3.7640`, 0/32 pairs helped) while band paint *solved against the
   frozen head* works (`eta_net = +0.7895`, 32/32 pairs; pose-constrained variant +0.5406 at
   1.039x d_pose). sq1 also concluded "the paint/AA/uint8 cure family cannot be the binding one
   for band-local edits on this vehicle". PREDICTION: a flat class-anchor band repaint on hv1
   will HARM, not help.
4. **lr2's ladder** — every offset-CARRYING rung loses; the rungs with high eta are the solved
   ones that need no carrier. PREDICTION: any cure named here must be carrier-free or
   near-carrier-free to have a chance.
5. **fl1's floor scope** — the 0.005318 smooth-label flicker floor is FORMULATION-scoped and is
   pierced by phase-faithful renderers. hv1's d_seg 2.9611e-4 is already ~18x BELOW it.
   PREDICTION: S4 is not a floor that binds this vehicle; the GT-spike join will show only a
   small share of our flips sitting on GT-flicker pixels.
6. **m91 (pc2 hub law)** — seg is one graph with one hub; Road appears in 87.8% of flips.
   PREDICTION: the per-class charge will be Road-heavy, and per-EDGE reading beats per-class.

## §1 Instrument check — PASS

Gate paid before any decomposition row was claimed. Receipt
`RT1_INSTRUMENT_CHECK.json` (`ddm_rt1_instrument_check.v1`).

| control | measured | reference | verdict |
|---|---:|---:|---|
| raw custody sha256 | `e5539653…` | wc1 pin `e5539653` | PASS |
| advisory d_seg (n600) | **2.96173e-04** | contest-CUDA **2.9611e-04** | ratio **1.000213** |
| advisory scored flips | **34,938** | 34,930.6 | +7.4 flips |
| td1 label control (scorer-free) | **1,717** | td1 1,717 | EXACT |

The advisory CPU instrument sits **0.021% from the contest-CUDA seg term** on this vehicle. That
is a tighter CPU↔CUDA agreement than the seg axis has shown anywhere else in this lineage, and
it is what licenses every advisory row below. It is still `[macOS-CPU advisory]`: it is a
yardstick, never a score.

**Instrument pins** (et4 — batch shape is part of the forward instrument): frozen CPU-torch
SegNet, **batch = 1 pair**, `torch.set_num_threads(8)`, `SegNet.preprocess_input` verbatim
(`x[:, -1]` then bilinear to 384×512). Every leg below uses the identical instrument, so the
leg-to-leg differences carry no instrument term.

## §2 The per-stage ledger

All rows n600, full field, `[macOS-CPU advisory]`. `seg_dS_per_flip = 100/117,964,800 =
8.4771e-07` (td1's derived law, reused not re-derived). Receipt `RT1_LEDGER.json`.

### §2.1 The round trip is now EXACT, not modelled

td1 could only *infer* the round-trip share, because it never ran the scorer: it subtracted
1,717 label errors from 34,930.6 scored flips and carried a MODELED amplification `r`. This unit
measures the render's argmax directly, so the round trip is a counted set difference:

| row | flips | S units | note |
|---|---:|---:|---|
| scored seg term (advisory) | 34,938 | 0.029617 | the whole seg axis |
| transmitted labels vs GT | 1,717 | 0.001456 | what our label channel actually costs |
| **round trip: render argmax vs shipped labels** | **33,743** | **0.028604** | **EXACT, measured** |

td1's modelled 33,213.6 / 0.028156 S at r=1 lands **1.6% low**; the measured round-trip share is
**96.6%** of the seg term, not 95.1%. td1's headline survives its own measurement. The round trip
is **2.98× the entire remaining sub-0.15 gap** (−0.0095973).

### §2.2 S4 — the GT-flicker join (fl1)

| quantity | value |
|---|---:|
| GT spike pixels (differ from BOTH stride-2 neighbours) | 625,510 (0.53% of the field) |
| interior flips (598 pairs) | 34,805 |
| flips sitting on a spike pixel | **9,651 (27.73%)** |
| enrichment vs area | **52.2×** |
| S carried by spike-coincident flips | **0.008182** |

**My §0 prediction 5 was half wrong and I am recording the diff.** I predicted "only a small
share" of flips would sit on GT-flicker pixels. The floor-scope half held — the 0.005318
smooth-label floor does **not** bind, hv1 sits 18× below it. The share half did not: **27.7% of
our residual is coincident with GT label instability**, enriched **52×**. The reason is now
obvious in hindsight and worth stating: GT spikes and our flips live on the *same* object, the
codim-1 boundary, so they were never going to be independent.
That is not proof of irreducibility (a phase-faithful render can track a flickering GT), but it
is the honest ceiling on any temporally-smoothing cure: a cure that makes our field *smoother*
than GT walks into this 0.0082 S.

### §2.3 The geometry — the seg axis is one pixel wide

City-block ring index from the **transmitted** label boundary (the decoder's own, free):

| ring (px from label edge) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | ≥7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pixels in ring | 2,551,464 | 2,132,982 | 1,965,832 | 1,842,841 | 1,768,652 | 1,709,795 | 1,655,670 | 104,337,564 |
| flips vs GT | **34,666** | 212 | 37 | 14 | 1 | 1 | 0 | **7** |
| flips vs shipped label | 33,479 | 209 | 34 | 14 | 0 | 1 | 0 | 6 |

**99.22% of the whole seg axis sits exactly ON the label boundary.** The label interior —
104,337,564 pixels, 88.4% of the field — carries **7 flips in total**, a rate of 6.7e-08. The
boundary ring carries a flip rate of 1.359%, i.e. **203,000× the interior rate.**

This is the single most actionable number in the unit. The renderer does not mis-paint classes;
it places the class **edge** by about one pixel. The seg axis is a codim-1 edge-placement
problem, exactly as the unified level-set reading says — and now with a measured coefficient on
this vehicle rather than a borrowed one.

### §2.4 Per class — Lane is 43× over-represented

Charged to the GT class (the class the render should have produced):

| class | flips | S units | share of flips | share of area | enrichment |
|---|---:|---:|---:|---:|---:|
| 0 Road | 13,786 | 0.011686 | 39.5% | 23.2% | 1.70× |
| 1 Lane | 8,712 | 0.007385 | 24.9% | **0.59%** | **42.6×** |
| 2 Undrivable | 6,297 | 0.005338 | 18.0% | 49.5% | 0.36× |
| 3 Movable | 4,750 | 0.004027 | 13.6% | 1.24% | **11.0×** |
| 4 MyCar | 1,393 | 0.001181 | 4.0% | 25.4% | 0.16× |

Road carries the most absolute S (m91's hub law holds), but the *thin* classes — Lane at 42.6×
and Movable at 11.0× — are where the render is losing per unit of area. Lane alone is 0.0074 S,
77% of the whole remaining gap.

### §2.5 S1/S2/S3 — the flat-paint legs, and why the v14 stage taxonomy does not nest here

The palette is the receiver-legal per-class mean of the shipped decode's own camera RGB
(15 bytes, `RT1_PALETTE.json`): Road (53,66,88) · Lane (140,143,122) · Undrivable (109,146,172)
· Movable and MyCar as recorded. Class area shares recovered from the lifted label field —
Road 23.2%, Lane 0.59%, Undrivable 49.5%, Movable 1.24%, MyCar 25.4% — independently confirm the
canonical comma10k class order, so no index was assumed.

| leg | reads back its own labels with | scored d_seg vs GT | S units |
|---|---:|---:|---:|
| **shipped neural render** | **33,743 flips** | 2.96173e-04 | **0.029617** |
| flat prototype paint, camera-res through R | **1,195,663 flips** | 1.01407e-02 | **1.01407** |
| flat prototype paint, scorer-res (no lift, no resize) | **1,195,663 flips** | 1.01407e-02 | **1.01407** |

**S2 is not a floor this vehicle sits above — it is a ceiling this vehicle sits far below.** The
trained renderer reads its own label field back **35.4× more faithfully** than the best flat
prototype paint of the same labels. In S units the render is worth **−0.985 S** against the paint
baseline. So `S1 = RT − S2 = 33,743 − 1,195,663 = −1,161,920 flips`: **negative**. The v14
taxonomy's nesting (a paint-response floor plus a render deviation on top) is FALSE on the
trained-renderer vehicle; the two stages are alternatives, not layers, and the ledger is
explicitly NON-ADDITIVE here.

Consequence, stated plainly: **"find better class prototypes" is dominated by 35× on hv1.** v14's
FORMULATION verdict (#603) is not merely re-confirmed, it is re-confirmed against a much stronger
competitor than v14 had.

Where flat paint fails, charged to the label class it was painted with: Road **864,979** (72% of
its own failures), Movable 181,710, Lane 119,075, MyCar 25,938, **Undrivable only 3,961**. A flat
sky is read back almost perfectly; a flat road is not. Texture is load-bearing for Road — which is
precisely the hub class of §2.6.

### §2.5b S3 — the R operator supplies exactly zero

| leg | readback flips vs its own labels | scored flips vs GT |
|---|---:|---:|
| flat paint lifted to 874×1164, through R | 1,195,663 | 1,196,248 |
| flat paint straight at 384×512, R is identity | 1,195,663 | 1,196,248 |
| **S3 = difference** | **0** | **0** |

**S3 = 0 flips = 0.000000 S at n600.** Spot-checked further: on pairs 0 and 7 the two legs produce
**bit-identical argmax fields**, not merely equal counts. The nearest-neighbour lift followed by
the evaluator's bilinear downsample and uint8 is *transparent* to a piecewise-constant field.

⚠ **Scope.** This is measured on flat paint. It proves R adds nothing on piecewise-constant
content; it does **not** prove R is transparent to the textured neural render, and that leg is not
separable — the render exists only at camera resolution, so there is no "before R" version of it
to score. The honest statement is: **R is not a supplier for any paint-shaped candidate, and it
remains unmeasured for the render.** #149's pre-R placement precedent is consistent with this and
is not contradicted.

## §2.6 The edge shape — 94% of the axis is symmetric jitter, not bias

Scorer-free, from the retained argmax fields (`RT1_EDGESHAPE.json`).

| class | signed area error (pred − GT, n600) | Σ per-frame abs area error | systematic share |
|---|---:|---:|---:|
| 0 Road | +930 | 5,540 | 16.8% |
| 1 Lane | **−1,773** | 4,267 | **41.6%** |
| 2 Undrivable | +341 | 3,641 | 9.4% |
| 3 Movable | +718 | 3,194 | 22.5% |
| 4 MyCar | −216 | 1,238 | 17.4% |

The signed errors sum to zero by construction. Their total magnitude is 3,978 px, i.e. **at most
~1,989 flips (5.7% of the seg axis) are a systematic area bias**; the other **94.3% is symmetric
sub-pixel edge jitter**. Only Lane shows a strong one-directional signature: our render
**erodes lane markings** by 1,773 px net over n600.

**93.89% of all flips have the class the scorer wanted sitting in the 4-neighbourhood of our own
output.** The render is not omitting classes; it is placing the edge about one pixel off.

Per-EDGE (m91's hub law — never per class alone), pred→GT:

| edge | flips | share | S units |
|---|---:|---:|---:|
| **Road ↔ Lane** (0→1 8,491 · 1→0 6,687) | **15,178** | **43.4%** | **0.012866** |
| Road ↔ Undrivable (3,288 · 3,687) | 6,975 | 20.0% | 0.005913 |
| Undrivable ↔ Movable (2,936 · 3,000) | 5,936 | 17.0% | 0.005032 |
| Road ↔ Movable (1,777 · 2,420) | 4,197 | 12.0% | 0.003558 |
| Road ↔ MyCar (1,160 · 992) | 2,152 | 6.2% | 0.001824 |
| everything else | 500 | 1.4% | 0.000424 |

Three edges carry **80.4%** of the whole seg axis. Road touches **81.6%** of all flips — pc2's hub
law reproduced on hv1, on a vehicle pc2 never saw.

## §2.7 The zero-byte cure probe — flat-anchor band repaint is CLOSED on hv1

The one post-hoc cure that costs essentially nothing: the receiver already owns the label
boundary, so it can repaint that band with a 15-byte class palette using only bytes it has.
Radius 1 in gp1's own band operator (the anisotropic 11-pixel kernel, §code), α = 1.

| leg | scored flips vs GT | S | Δ vs base |
|---|---:|---:|---:|
| base render | 34,938 | 0.029617 | — |
| **+ flat-anchor band repaint r=1** | **1,663,803** | **1.410423** | **+1,628,865 flips = +1.3808 S** |

**47.6× worse.** The probe is not marginal; it is catastrophic, and it closes the flat-content
band-edit rung on hv1 with an n600 counted row rather than an inherited prediction.

It also reproduces sq1's v0 sign exactly (`eta_net = −3.7640`, 0/32 pairs helped) on a completely
different vehicle — and adds a mechanism worth keeping: **the band repaint (1,663,803) is worse
than repainting the WHOLE frame flat (1,196,248).** Painting only the band manufactures a hard
synthetic edge against surviving real texture, and the scorer reads that manufactured edge as a
boundary in the wrong place. That is the "SegNet sees REGIONS, not pixels" law showing up as a
sign: a local edit is not local to the scorer.

Consequence for the cure search: **no flat-content local edit will work on this vehicle.** Any
band-local cure must be SOLVED against the frozen head (sq1's v1), not painted from a palette.

## §2.8 The margin — the residual is a tie, not a capacity wall

For every flip, the deficit is `logit[what we produced] − logit[what the scorer wanted]`: exactly
how far a finisher must move the head to recover that pixel.

| quantity | value |
|---|---:|
| flips where the wanted class is already the **runner-up** | **34,341 / 34,938 = 98.29%** |
| mean deficit | **0.1645** |
| median deficit | **0.1051** |
| 10th / 25th / 75th / 90th percentile | 0.0143 / 0.0424 / 0.2228 / 0.3873 |

Cumulative share of flips by required logit movement:

| deficit < | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---:|---:|---:|---:|---:|---:|
| share of flips | 6.8% | 19.0% | **49.4%** | **84.5%** | 99.3% | 100.0% |

Against that, the pixels we get **right** are not close calls at all: 112.1M of 117.9M correct
pixels carry a top-1-vs-top-2 margin in [3, 10), and 359,722 exceed 10.

**So the seg axis is 34,938 razor-thin ties inside an otherwise extremely confident field.** Half
the axis needs less than one tenth of a logit; 84.5% needs less than 0.3; nothing needs more than
3. The wanted class is already second place 98.3% of the time.

This is the measurement that says the round trip is **not** a capacity or fidelity wall. It is a
tie-breaking problem on a one-pixel curve — which is precisely the regime where a solved,
margin-aimed edit has high realization (sq1's η 0.54–0.79) and where a blunt flat repaint
destroys everything (§2.7). It also explains the S1 sign: the trained render already put the
scorer within 0.1 logits of the answer at half the failure sites, which no flat palette can do.

## §3 The named lever and the cheapest cure per stage

### §3.1 The lever

**The Road↔Lane edge, one pixel wide, decided by a tenth of a logit.** 43.4% of the seg axis
(0.0129 S), inside a boundary ring that carries 99.2% of the axis, where 93.9% of flips have the
right class already adjacent, where the error is 94% symmetric jitter, and where the scorer needs
to move **less than 0.1 logits at half the sites and less than 0.3 at 84.5%** with the wanted
class already in second place **98.3%** of the time (§2.8). Everything else on the seg axis is a
smaller instance of the same object.

The four facts compose into one statement: **this is not a fidelity deficit, it is an unbroken
tie on a curve the decoder already knows for free.** That is the most favourable shape a residual
can have for a cheap cure — and it is exactly the shape that punishes blunt edits (§2.7) and
rewards solved ones.

### §3.2 Cheapest cure per stage

| stage | measured supply | cheapest cure | family precedent | verdict |
|---|---|---|---|---|
| **S1 render deviation** | **negative** — the render beats flat paint 35.4× | edge-weighted / margin-weighted objective in the render's own training | ns1 P1 train-for-editability · wd3 scorer-aware distillation (live) · CLAUDE.md lever 3 (R in-loop) | the only carrier-free supplier |
| **S2 paint→SegNet** | 1,195,663 readback flips — a ceiling, not a floor | none needed; the stage is dominated | v14 #603 FORMULATION verdict, re-confirmed against a 35× stronger competitor | CLOSED for hv1 |
| **S3 resize/uint8 (R)** | **0 flips, 0.000000 S** | none needed | #149 pre-R placement survives D | not a supplier |
| **the cure probe** | flat band repaint **+1.3808 S** | do not paint flat content locally | sq1 v0 η −3.7640 reproduced on a new vehicle | CLOSED on hv1 |
| **S4 GT flicker** | 27.7% coincidence, 52× enrichment, 0.0082 S | not fixable — it is a GT property | fl1 scope law: the floor binds only smoother-than-GT witnesses | a bound on smoothing cures, NOT a hard floor |

### §3.3 The one live byte-carrying candidate, priced

The decoder owns the label boundary **for free** — it is a deterministic function of the tokens it
already parses. So a seg-correction channel only has to name **which band pixels flip**, not where
the band is. That reframes gp1's "describe the flips" family at a support 46× smaller than the
frame.

MODELED, first order, and it credits no collateral and no class-disambiguation bits:

| quantity | value |
|---|---:|
| free band support (ring 0) | 2,551,464 px |
| flips on the band | 34,666 |
| band flip density | 1.3587% |
| i.i.d. binary entropy | 0.10373 bits/band px |
| **mask carrier at the i.i.d. floor** | **33,082 B** |
| break-even carrier at η = 1.0 | 44,481 B |
| break-even carrier at η = 0.7895 (sq1 free solve) | 35,118 B |
| break-even carrier at η = 0.5406 (sq1 pose-constrained) | 24,046 B |

Net ΔS at the i.i.d. floor: **−0.00759 at η = 1.0**, **−0.00136 at η = 0.79**, **+0.00601 at
η = 0.54**.

> ⚠ **SUPERSEDED BY §5 — read §5 before citing anything in this subsection.** This paragraph
> originally speculated that a structured coder could beat i.i.d. by 2× "because the flips are a
> contiguous 1-D signal along a curve", giving S ≈ 0.1472. **§5 built the coders and refuted
> that**: the flips are isolated single pixels (mean run 1.109), the best real coder beats i.i.d.
> by 2.5%, and the ceiling over all free conditioning is 12.2%. The paragraph is kept for
> provenance, not as a live claim.

**Why this is not a re-opening of a closed family.** lr2 closed the offset-CARRIER family on the
v4d vehicle at d_seg 0.004312 = **508,595 flips — 14.6× hv1's debt**. Carrier cost scales with the
flip count; the byte budget does not. At v4d's band density (19.9%) the same mask costs ~230 KB
and is hopeless; at hv1's 1.36% it costs 33 KB against a 44 KB budget. This is the
constants-are-poison law applied to a verdict: **lr2's closure was priced at an operating point we
no longer occupy, and this unit supplies the numbers to re-price it.** The re-price is owed before
any build.

## §3.4 td1's H2 is ANSWERED — for free, with no T4 row

td1 left `r` (scored flips produced per transmitted-label flip) UNMEASURED and wrote that "every
rung's admission collapses to it". The retained fields answer it observationally: at the **1,717**
pixels where our transmitted label already disagrees with GT, the render reproduced the wrong
label at **1,454** (84.7%) and **1,458** of them are scored flips.

**r_observational = 0.8492.**

td1's ≥8-bit rung needed **r < 0.8393**. Measured 0.8492 sits **1.2% above the bar** — the rung
still loses, so td1's "do not fire token drop" recommendation stands, but its margin is now known
to be razor-thin instead of unknown.

⚠ **Scope, honestly (m88).** This is OBSERVATIONAL and it is measured on the population of labels
that are *already wrong*, not on the population td1 would *deliberately drop* (chosen by bit
cost). Those are different populations and a prefix/selection bias between them is exactly the
error class m88 names. It bounds r for this population; it does not close the causal question.

## §4 Routing + queued follow-ons

**Pointer: UNMOVED.** hv1 ep0634 remains S 0.15959729295498598 @ 182,759 B [contest-CUDA T4
n600]. This unit measured; it did not move the score, and it was not permitted to.

| # | follow-on | fire condition | owner |
|---|---|---|---|
| 1 | **Mask-coder pricing on the retained flip mask** — code the 34,666 band flips against the 33,082 B i.i.d. floor with a real context/run-length coder along the boundary curve | **$0, scorer-free, fire now.** All inputs retained. If the mask does not code below ~35,118 B, the correction channel dies before any solver effort is spent | MAIN → a $0 desk arm |
| 2 | **η on hv1 under a pose constraint** — sq1's solved-paint rung re-measured on this vehicle's 1.36%-density band | only after #1 clears. At η ≤ 0.744 the channel loses even at the i.i.d. floor | successor charter (skeleton below) |
| 3 | **Edge-weighted render objective** — the carrier-free supplier; target the Road↔Lane 1-px band | route into the live wd3 / ns1-P1 line as a named objective term, not a new arm | wd3 line |
| 4 | **td1 H2 causal row** | DOWNGRADED by §3.4. Fire only bundled, and only if the drop-population's `r` is argued to differ from 0.8492 | not now |
| 5 | **td1 H3 — the 807 label-correction sites** | priced footnote: at r_obs 0.849 they recover ≈685 flips = 0.00058 S and also save bytes. Unchanged verdict — not worth a row alone | banked |

### Successor charter skeleton (only if #1 clears)

`ddm_<x>` — **the free-band seg-correction channel.** Measure, in order: (a) realized coded size
of the retained flip mask + the class-disambiguation bits where >2 classes meet; (b) η of a solved
paint restricted to flip sites on hv1, whole-frame accounted, with d_pose recomputed per sq1's
rider; (c) one byte-closed archive if and only if (a)+(b) clear
`net = -η·0.029617 + bytes·6.65834e-07 < 0` with margin. OPTIMAL FORM: sq1's v1 solver with both
riders (multi-start, in-loop realized-flip validation with best-iterate retention). Do NOT
inherit lr2's closure — it was priced at 14.6× this vehicle's flip count.

## §4.1 Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/` (APDataStore because
VertigoDataTier has 954 MiB free). Every leg's measured argmax field is persisted, not just its
scalar.

| artifact | bytes | sha256 (prefix) | what it is |
|---|---:|---|---|
| `argmax_base.npy` | 117,964,928 | — | the scored render's argmax field, n600 |
| `argmax_paint_cam.npy` | 117,964,928 | — | flat paint through R |
| `argmax_paint_scorer.npy` | 117,964,928 | — | flat paint at scorer res |
| `argmax_band_r1_a1.npy` | 117,964,928 | — | the cure probe |
| **`flip_mask_vs_gt.npy`** | 117,964,928 | `7910a29405f4…` | **the object the successor codes** |
| **`free_band_mask.npy`** | 117,964,928 | `649dd26f0843…` | the zero-byte support |
| **`flip_target_class.npy`** | 117,964,928 | `7fef0e87e27f…` | target class, 255 = no correction |
| `palette.npy` + `RT1_PALETTE.json` | 143 | — | the 15-byte receiver-legal palette |

Receipts: `RT1_INSTRUMENT_CHECK.json` · `RT1_GEOMETRY.json` · `RT1_EDGESHAPE.json` ·
`RT1_LEG_{paint_cam,paint_scorer,band_r1_a1}.json` · `RT1_MARGIN.json` · `RT1_LEDGER.json`.
Tool: `experiments/ddm_rt1_seg_roundtrip_decomposition.py` (stages `palette` / `instrument` /
`leg` / `geometry` / `margin` / `edgeshape` / `ledger`). Consumed, unmodified: the wc1 retained
decode `0.raw` (3,662,409,600 B, sha `e5539653…`, custody verified in-tool), the hv1 ep0634
`decoded_spatial_tokens.rc64.bin`, and the qs3 `gt_argmax_n600.npy` (sha `91d3ff11…`).

## §5 Follow-on #1 EXECUTED — the coder gate PASSES, and it was never the binding constraint

MAIN routed follow-on #1 back to me. It is done, at $0, scorer-free. Tool:
`experiments/ddm_rt1_flip_mask_coder_race.py` (`--mode race` / `--mode analysis`). Receipts
`RT1_CODER_RACE.json` + `RT1_MASK_ANALYSIS.json` and all eight coded payloads:
`/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/coder_race/` — M7 sha `4fb718bd01be…`,
M3 `00b46f253e86…`, M1 `75b7d95ee6f7…`, M2 `ff9617e78d00…`, M0 `36cf7f9229d0…`. The race is
deterministic: two independent runs produced byte-identical payloads.

**Every arithmetic payload was verified by decoding it back through the same online context
machine the encoder used**, so these are bytes a real receiver could parse — not entropy
estimates. Contexts read only the labels (free) and symbols already decoded.

### §5.1 The race (n600, 2,551,464 band symbols, 34,666 flips)

| coder | bytes | bits/flip | vs bar 35,117 B |
|---|---:|---:|---|
| **M7 CABAC boundary-walk (pair × run × temporal), 88 ctx** | **32,270** | 7.447 | **PASS −8.1%** |
| M6 CABAC boundary-walk (run × temporal), 8 ctx | 32,627 | 7.529 | PASS |
| M5 CABAC raster (pair × causal × temporal), 88 ctx | 32,699 | 7.546 | PASS |
| M3 static binary AC — the i.i.d. floor, realized | 33,087 | 7.636 | PASS |
| M4 adaptive binary AC, order-0 | 33,441 | 7.717 | PASS |
| M2 lzma(packed) preset 9\|EXTREME | 37,224 | 8.590 | **FAIL** |
| M1 brotli(packed) q11 | 37,632 | 8.684 | **FAIL** |
| M0 raw packed band bits | 318,933 | 73.601 | FAIL |

Two things worth keeping. First, **M3 lands at 33,087 B against my modelled 33,082 B floor — a
5-byte agreement**, which independently validates the §3.3 arithmetic. Second, **both
general-purpose compressors FAIL the bar**: brotli and lzma are 13–17% *worse* than a purpose-built
binary arithmetic coder on this object. A sparse boundary mask is not something a byte-oriented
LZ compressor can hold.

### §5.2 My §3.3 speculation is REFUTED, and here is why

§3.3 said "a structured coder that beats i.i.d. by 2× — plausible, because the flips are a
contiguous 1-D signal along a curve" would reach S ≈ 0.1472. **Measured: the best real coder beats
i.i.d. by 2.5%, not 50%.** The premise was wrong, and the diagnostic says exactly how:

| statistic (walk order, **n600 full field**) | value |
|---|---:|
| P(flip) | 0.013587 |
| P(flip \| previous walk symbol was a flip) | 0.0988 — a **7.27× lift** |
| **mean run length of consecutive flips** | **1.110** |
| runs of length 1 | **28,801 of 31,231 (92.2%)** |

The clustering lift is real but the mass is not in it: **the flips are isolated single pixels
scattered along the boundary, not contiguous displaced stretches.** So "the edge is shifted" is the
wrong picture; "the edge is salt-and-peppered" is the right one, and run-length structure — the
thing I bet on — barely exists.

Exact conditional entropy over every free conditioning variable (ideal-model limit, no learning
cost) confirms the ceiling:

| conditioning (all free to the receiver) | bytes | vs i.i.d. |
|---|---:|---:|
| i.i.d. | 33,082 | — |
| \| edge pair | 31,349 | +5.2% |
| \| pair × causal neighbours | 30,770 | +7.0% |
| \| pair × causal × temporal | 30,668 | +7.3% |
| \| pair × causal × temporal × row-band | 30,229 | +8.6% |
| **\| pair × causal × temporal × row × band-degree** | **29,058** | **+12.2%** |

**12.2% is the ceiling of free conditioning, not 50%.** My implemented M7 captured 2.5% of it;
a coder using the full context set would land near 29–30.5 KB after learning cost. That is
bounded headroom, and it does not change the verdict below.

### §5.3 The cost I had explicitly not credited: the target class

A flip tells the receiver *that* a pixel is wrong, not what it should become. Conditioned on
(our own label, edge pair) — both free — the target class costs **0.2226 bits/flip = 965 B**
(unconditional it would be 8,884 B). Small, but it is real and it is now counted.

### §5.4 The verdict — LIVE at the gate, BOUNDED as a route

| channel | bytes | η required | net ΔS at η=1.0 | net ΔS at η=0.7895 | net ΔS at η=0.5406 |
|---|---:|---:|---:|---:|---:|
| measured M7 mask only | 32,270 | 0.7312 | −0.007899 | −0.001714 | +0.005601 |
| **measured M7 + target class** | **33,235** | **0.7531** | **−0.007257** | **−0.001071** | **+0.006243** |
| best modelled context + target | 30,023 | 0.6803 | −0.009396 | −0.003210 | +0.004104 |

Resulting S from the hv1 base 0.15959729: **0.15234** at perfect realization, **0.15853** at sq1's
free-solve η, and a **loss** at sq1's pose-constrained η. Even the best modelled coder reaches only
**0.15020** at η = 1.0.

**So: the correction-channel family PASSES the coder gate and does NOT die there — but it is not a
sub-0.15 route by itself, and the coder was never the binding constraint. η is.** The gate that
decides it is now precise: **η must exceed 0.753** for the channel to break even at all. sq1's
free solve (0.7895) clears that by 5%; sq1's pose-constrained solve (0.5406) does not clear it at
all. Since any shippable version must pay the pose constraint, the honest prior is that this
channel is **more likely dead than alive**, and one measurement settles it.

⚠ Scope: 272 flips (0.78%) lie off the band and are unaddressable by this support. Every η here
is sq1's, measured on the v4d vehicle — nothing in §5 measures η on hv1.

## §5.5 Routing update (superseded by §6.4 — both follow-ons are now executed)

1. **Follow-on #1 — DONE.** Coder gate passed at 32,270 B. Do not re-run.
2. **Follow-on #2 — DONE (§6). η = 0.6461 (n=6) vs bar 0.753 → fails the bar; total S break-even.**
3. **Follow-on #3 (edge-weighted Road↔Lane objective into the wd3 / ns1-P1 training line) is
   promoted** — see §6.4.

## §6 The ETA GATE — pose-constrained realization on hv1 (follow-on #2)

Pre-registered bar, written into the tool before any solve: **η > 0.753 → LIVE; η ≤ 0.753 →
CLOSED**. Tool `experiments/ddm_rt1_eta_gate_pose_constrained.py`; receipts under
`/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/eta_gate_{null,free}/`.

Reference form imported, not reimplemented: sq1's `Scorer`, `decode_gt_frames`,
`pose_null_projector` / `project_null` / `snap_band_to_blocks`,
`realize_scorer_paint_to_camera`. Riders carried: multi-start (pu2), in-loop REALIZED-argmax
validation with best-iterate retention (fd2/tb1), whole-frame accounting, and **d_pose
recomputed against decoded GT** (qs4's stale-compensation lesson). Pairs drawn by seeded
RANDOM choice, never a prefix (m96).

### §6.1 Two instrument defects the positive control caught — both mine

**No verdict was admissible until an unconstrained control could realize anything at all.** It
could not, and the reason was my setup, twice over.

1. **sq1's objective does not transfer.** sq1 used a plain whole-frame cross-entropy. On v4d
   that was ~848 wrong pixels per pair; on hv1 it is ~50 wrong out of 196,608, so the target
   carries **0.025%** of the loss. Measured signature: with sq1's loss the best iterate is
   **step 0 for every pair in both modes** — the solver never moves. Cure: reweight the CE by
   `focus_weight` on the described set, keeping every other pixel at weight 1 so the collateral
   term survives.
2. **sq1's edit SUPPORT does not transfer, and this one was worth 0.65 η.** sq1 edits the whole
   label-boundary band. On hv1 that is **11,377 px to fix 37**. Measured what that does: the
   solver fixes **37/37 described pixels by step 5** — they are trivially fixable, exactly as
   §2.8's tiny deficits predict — while whole-frame flips go **35 → 343**. Collateral is **8×
   the gain**, so best-iterate retention correctly refuses every iterate and η pins at ~0.

   The correct actuator is the **described set dilated**, which is equally free (the receiver
   decodes the flip mask, so it knows the described set). Support ladder, 2 pairs, unconstrained:

   | support radius | pair 34 η | pair 82 η |
   |---|---:|---:|
   | r=0 (described only) | +0.5405 | +0.5098 |
   | **r=1** | **+0.6216** | **+0.6863** |
   | r=2 | +0.3243 | +0.5490 |

   r=1 is the optimum and is the configured default. **A NO measured on sq1's support would
   have been an instrument artifact reported as physics.**

### §6.2 The measured gate — **CLOSED**

Pose-null-constrained solve, described-set support r=1, 30 steps × 2 starts, best realized
iterate, whole-frame accounted, d_pose recomputed against decoded GT. Pairs drawn by seeded
random choice (seed 20260816).

| pair | described | flips before → after | **η_net** | d_pose ratio | realization err |
|---|---:|---:|---:|---:|---:|
| 33 | 49 | 49 → 17 | **+0.6531** | ×1.582 | 0.0 |
| 66 | 73 | 73 → 22 | **+0.6986** | ×0.218 | 0.0 |
| 81 | 41 | 42 → 20 | **+0.5366** | ×7.735 | 0.0 |
| 89 | 50 | 50 → 14 | **+0.7200** | ×0.902 | 0.0 |

| statistic | value |
|---|---:|
| **η pooled** | **0.6620** |
| η per-pair mean ± sd | 0.6521 ± 0.0819 |
| η min / max | 0.5366 / 0.7200 |
| **pairs above the 0.753 bar** | **0 of 4** |
| realization fidelity (max abs err on support) | **0.0 on every pair** |

**η = 0.6620 ≤ 0.753 → the family is CLOSED on arithmetic.**

**Stability as rows land.** The run was left going at `nice 10` behind the sister b2e arm and the
aggregator re-reads the incremental rows, so the live authority is
`ETA_GATE_VERDICT_AGGREGATE.json`, not any number frozen here. The verdict has not wobbled:
pooled η = **0.6442 (n=3) → 0.6620 (n=4) → 0.6448 (n=5) → 0.6461 (n=6) → 0.6235 (n=9)**, with
**0 of n above the bar at every n** and per-pair sd ~0.07. The table above is the n=4 snapshot; re-run
`--mode aggregate` for the current n.

The η shortfall itself is not marginal: on seg+rate the channel costs **+0.00314 S** at the
measured pooled η, and **even the best single pair (0.7200) does not clear break-even on that
leg.** The realization step is exact on every pair (`D` reproduces the solved paint to 0.0), so
nothing is lost between solve and score — the seg shortfall is entirely η. What *is* marginal is
the TOTAL, once the pose leg is aggregated correctly — see §6.2b, which corrects an error of mine
and moves the total to break-even.

### §6.2b ⚠ CORRECTION — I aggregated the pose leg wrongly, and it changes the total

**What I first reported:** "the pose leg makes it worse… charging the mean adds +0.0051 S."
**That was wrong, in sign.** I averaged per-pair d_pose *ratios*. `upstream/evaluate.py`
aggregates **d_pose itself** across pairs (`posenet_dists.sum() / batch_sizes`) and only then
takes √(10·d_pose). The correct aggregate is `mean(d_pose_after) / mean(d_pose_before)`, which
weights each pair by how much of the axis it actually carries. Mean-of-ratios weights a pair with
negligible d_pose the same as one carrying the axis, and here the two statistics **disagree in
sign**: mean-of-ratios ×1.809 vs scorer-convention **×0.4309**.

| statistic | ratio | ΔS pose |
|---|---:|---:|
| **scorer convention — mean of d_pose (CORRECT)** | **×0.4309** | **−0.00285** |
| mean of per-pair ratios (what I first used) | ×1.809 | +0.00286 |
| median of per-pair ratios | ×0.603 | −0.00185 |

Per-pair ratios span ×0.109 to ×7.735; 4 of 6 pairs improved. **The pose axis improves 2.3× in
aggregate**, so the pose leg *pays back* −0.00285 S rather than costing.

**Corrected total (n=6):** seg+rate **+0.00314** and pose **−0.00285** give a net of
**+0.00029 S — break-even at that n, not the clear loss I first reported.** (Later rows move it
to +0.00252 at n=9; see the trajectory below.) The tool now computes both and refuses to let one
stand for the other; `verdict` answers the pre-registered seg+rate bar only.

Two things I will not paper over. The pose gain is plausibly a **side effect of the `truth_dir`
solver start** biasing the uint8 rounding residual toward GT — encoder-side and legal, but not a
designed compensation, and it may not survive a different init or larger n. And an aggregate
d_pose over **6 pairs** whose ratios span 70× is not a reliable estimate of the n600 aggregate.

**That caveat was correct, and later rows confirmed it.** As the run continued the pose payback
shrank and the total drifted away from break-even:

| n | η pooled | pairs above bar | pose agg ratio | ΔS pose | **total ΔS** |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.6461 | 0 of 6 | ×0.431 | −0.00285 | **+0.00029** |
| 7 | 0.6382 | 0 of 7 | ×0.457 | −0.00260 | **+0.00078** |
| 9 | 0.6235 | 0 of 9 | ×0.713 | −0.00129 | **+0.00252** |

**η is the stable quantity** (0.644 → 0.624 across n=3…9, sd ~0.07, **0 of n above the bar at
every n**); the pose leg is the noisy one, and it is regressing toward no payback. The break-even
reading at n=6 was the optimistic end of the noise, not the answer. At n=9 the channel costs
**+0.0025 S**, and its seg gain (0.0183 S) still sits below its rate cost (0.0221 S) — the
non-supply conclusion is unchanged and is now measured at a larger n than the break-even claim it
replaces.

**verdict_scope: INSTANCE** — hv1 ep0634 base, ring-0 described set, r=1 described-set edit
support, pose-null-constrained realization, this solver budget, n=9 seeded-random pairs. Per
m96 a seeded-random subset **may** refute a bar (which is what happened: 0/4 clear it, and the
maximum measured value still loses); a subset that *cleared* the bar would not have licensed a
LIVE verdict without the fuller population — the aggregator enforces that asymmetry in code,
emitting `PASS_SUBSET_NOT_LICENSING_LIVE` rather than `LIVE` below n=120. The run was stopped
early to yield cores to the
sister b2e admission arm rather than contend; the aggregator reads the incrementally-written
rows, so the verdict is honest at the n it reached.

### §6.3 Why η lands at 0.64 and not 0.79 — the mechanism, not the budget

The described pixels are **trivially fixable**: §6.1 measured 37/37 fixed by step 5. What caps η
is **collateral** — every edit that moves a boundary pixel to its target also perturbs the region
evidence its neighbours depend on. That is the same law that killed the flat band repaint
(§2.7, +1.38 S) and sq1's truth paint (η −3.76): *a local edit is not local to the scorer.* The
solver's whole-frame accounting prices that honestly, and what is left after collateral is ~0.64
of the described flips — real, but a sixth short of what the byte arithmetic needs.

sq1's free-solve 0.7895 does not transfer here, and the unconstrained control on hv1 shows why
you cannot simply reach for it: on pair 33 the free solve returned η 0.5714 — *lower* than the
pose-constrained 0.6531 — while destroying pose by **×396**. On this vehicle the pose-null
projection is not only mandatory, it is also a **useful regularizer**: it restricts the edit to
directions the scorer's pose head cannot see, which happens to limit seg collateral too.

## §6.4 ROUTING — the post-hoc correction family is closed; everything goes to the renderer

**The free-band seg-correction channel cannot supply the remaining gap.** It passed its coder
gate (32,270 real bytes, §5) and then failed the pre-registered realization bar: **η = 0.6461
(n=6) against a required 0.753, with 0 of 6 pairs above it.** No fire-order is issued.

**The reason is weaker than I first wrote, and the honest one is stronger.** After the §6.2b
pose-aggregation correction the total is **+0.00029 S — break-even, not a clear loss.** So the
durable finding is *not* "this loses badly"; it is **"this lands on zero."** That conclusion is
robust to the pose noise in a way the loss claim was not: seg+rate contributes +0.00314 and the
pose leg −0.00285, and *whichever way the pose term moves at larger n, the channel is arithmetically
incapable of supplying a −0.0096 gap* — its entire seg gain at η=0.65 is 0.0190 S against a 0.0221 S
rate cost. A supplier has to clear the gap, not graze zero.

The closure is scoped (INSTANCE, n=9 seeded-random, this solver budget) and it is reopenable by
exactly two named things, neither of which is "try harder": a realization mechanism whose
collateral is structurally lower than a solved local edit (the whole family measured here —
flat paint, truth paint, band repaint, solved paint — is collateral-limited, §2.7/§6.3), **or**
a described set small enough that the byte cost falls faster than η does. The second is worth
one line of arithmetic: the channel needs `η·0.029387 > bytes·6.658e-7`, so at η = 0.6620 the
whole channel must cost **under 29,215 B** — 12% below the 33,235 B it actually codes at, and
§5.2 already measured the free-conditioning ceiling at 12.2%. **The two measured curves do not
cross.** That is why this is a closure and not a deferral.

**Everything routes to follow-on #3: the edge-weighted Road↔Lane objective in the renderer's own
training** (wd3 / ns1-P1 line). This unit has now bounded every post-hoc lever on the seg axis —
paint (35× worse), band repaint (+1.38 S), correction channel a non-supplier (+0.0025 S) — while the render
itself is measured to be the only actor that ever put the scorer within 0.1 logits of the answer
(§2.8). The seg axis is 99.22% a one-pixel edge-placement problem concentrated 43.4% on Road↔Lane;
that is a training target, and it is the one lever this unit did not bound away from sub-0.15.

## §7 What this unit did NOT establish

- **No causal `r`.** §3.4 is observational.
- **η on hv1 is now measured (§6) but only at n=9 seeded-random pairs**, one solver budget, one
  support radius. It refutes the bar (0/4 clear it, and the maximum loses); it does not
  characterize the η distribution across the full 600.
- **No LIVE verdict was ever available from this sample size** — by construction, per m96. Only
  the refutation direction is licensed.
- **No coded mask size.** 33,082 B is an i.i.d. entropy floor, not a coder result; a real coder
  may beat it (structure) or miss it (model cost). Both directions are open.
- **No claim that the 27.7% GT-flicker coincidence is irreducible.** fl1's scope law forbids that
  reading; it bounds smoothing cures only.
- **No score.** Every number here is `[macOS-CPU advisory]`.
