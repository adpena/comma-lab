# ddm_pd6 — the PRICE-FIRST generator on the move-52 field

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pd6_price_first_generator_20260916`. Base: **move 52**, S 0.13620226906030858 @ 179,332 B,
archive sha `ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`, re-derived from
`.omx/state/canonical_frontier_pointer.json` at bind time. Axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the distortion
legs; bytes EXACT through the shipped RLC1 coder and the shipped container. **No score is claimed.
Only `upstream/evaluate.py` on shipped bytes is a score, and MAIN fires.** No Modal, no authorization,
no fire, no completion, no packet.

Pose solver, stated once: every per-pair credit comes from
`ddm_jg5_pose_resolve_on_edited_renders.refine_pair` at PoseNet batch 1 on the moved render, with
`outer_rounds=40, max_gn_iterations=400` (pd4's cluster-search reference values); every n600 number
comes from `ddm_sj1_joint_admission`'s `pose` stage (`ddm_up2_shipping_pose_solve.measure_pose`,
batch 8) on the composed overlay this arm rendered.

---

## 1. What this arm inverted

Every pre-distortion pass on this object — sj1 passes 1–8, pd1 through pd5 — generated its proposals
by POSE SALIENCY and priced them afterwards. pd5 measured the consequence and closed the multi-token
formulation, and its §9 scoped that closure to exclude exactly one thing: the reverse order. This
arm runs that order and only that order.

The generator is the shipped coder's own model. `runtime.rlc1_mixer.LaneMixer` codes **every** one
of a frame's 196,608 positions (`end_frame` refuses a frame unless `self.seen.all()`), and every
coded position carries a full K = 5 probability row. So ONE real encode of move 52's own field yields
the first-order price of EVERY single-token change in the whole 600 × 384 × 512 plane:

```
delta_bits(position, symbol) = -log2 p_model(symbol) + log2 p_model(truth)
```

That is a RANKING, never a charge — it does not carry the group-causal geometry update or the
adaptive response the change itself causes. The CHARGE is a real 600-frame sheet encode, and no
credit is measured on anything the coder has not first charged.

## 2. The capture, and the two controls that make it credible

`experiments/ddm_pd6_price_first.py capture` wraps `LaneMixer.coding` and `.end_frame` as pure
observers — each calls the shipped method and returns its value unchanged — and runs the pricer's own
control encode.

| control | outcome |
|---|---|
| **the pricer re-packs move 52's own archive** | **179,332 B sha `ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`**, byte-identical to the live pointer, twins identical in-process, stream 119,097 B, decoded plane sha `117951bbd6800949…` — **with the capture wrappers installed** |
| **the captured rows ARE the charged rows** | per frame, the sum of −log2 p over the TRUE symbols must equal the pricer's own `per_frame_bits`. MEASURED over all 600 frames: **max absolute 0.00047 bits, max relative 3.216e−07** (float32 rounding) |
| the inherited decode | 3,662,409,600 B sha `ccb89e3e73bf61ac…eced`, re-hashed by this arm (pd3 and pd4 each parsed the same shipped bytes back cold and produced a bit-identical raw; pd4 measured max abs per-pair pose difference 0.000e+00 over 600 pairs) |

Capture wall: 1,213.6 s for the 600-frame encode plus the per-frame flush.

## 3. The cheap half, MEASURED

Per frame the capture keeps the 1,024 cheapest `(position, symbol)` pairs inside the argmax interior
(radius 3, which admits a median 178,453 of 196,608 positions — 90.8 %). Floor pairs are excluded;
they are pp1's twelve, and they are **exactly the twelve largest-d_pose pairs, holding 56.7 % of the
n600 pose mass**, so every number in this memo is drawn on the remaining 43.3 % over 588 pairs.

| quantity | MEASURED |
|---|---|
| pairs with candidates | **588** of 600 (the 12 floor pairs have none) |
| candidates emitted | **4,291** — 3,233 single-token, 1,058 two-token |
| held-fixed price-control pairs | **59** (every 10th pair contributes one candidate, repeated across all sheets) |
| pairs whose cheapest change is NEGATIVE-priced | **219 of 588 (37.2 %)** — the model would spend FEWER bits on the changed symbol |
| cheapest change per pair, first-order bits | median **0.771**, min **−31.258**, max **3.577** |
| the price ladder, ranks 0→7 (median) | **0.861 → 1.763 → 2.272 → 2.657 → 2.989 → 3.222 → 3.411 → 3.575** bits |
| population median candidate | **29.27** bits — the cheap tail is thin: the 1st percentile of all interior candidates is 11.9 bits |
| pairs with a cheap 8-neighbour of the head | **97.7 %** (the 2-token family is populated almost everywhere) |

**The ladder is the headline of the generator.** A pair does not have one cheap change; it has eight
well-separated ones, and the eighth still prices under four first-order bits. That is the breadth the
inversion was supposed to buy, and it is real.

### Where the cheap half lives

| SegNet class | share of the cheap half | area share of the frame |
|---|---:|---:|
| **Road** | **90.99 %** | 23.2 % |
| Undrivable | 6.40 % | 49.5 % |
| Movable | 2.46 % | 1.24 % |
| MyCar | 0.14 % | 25.4 % |
| Lane | 0.01 % | 0.59 % |

602,112 cells over 588 pairs; mean row 212.4 of 384, with 77.6 % of cells in rows 192–287. The cheap
half is the road surface just ahead of the car — where the ego-motion parallax lives — and the
dominant move is token 0 → token 1 (90 % of cells). Class indices are the canonical comma10k order,
self-checked by area and vertical centroid, never luma-sorted.

### The two pools are not the same object

pd4 and pd5 together priced **714** distinct cells. Of this arm's 588 pairs, the fraction whose
cheapest cells touch ANY of them:

| head | pairs touching the prior pool |
|---|---:|
| 8 cheapest cells | **3 of 588 (0.51 %)** |
| 64 cheapest | 5 (0.85 %) |
| 1,024 cheapest | 26 (4.42 %) |

The price-first generator nominates a part of the plane the saliency-directed passes never priced.
Whatever this arm measures is not pd5 under another name.

## 4. What the lever needs from credit — DERIVED before any credit existed

At move 52's operating point: S per archive byte 6.658589531221714e-07, S per seg cell
8.482752943113527e-07, S per unit of one pair's d_pose 1.2847092313591597 (DERIVED at this base
mean; it EXPIRES at the next pointer move).

A pair can credit at most its own d_pose. Walking the whole n600 base against that ceiling:

| recovered fraction of the pair's own d_pose | at 3 real bits/token | at 6 real bits/token |
|---|---|---|
| 5 % | 55 pairs pay, **−2.76e−05 S** | 26 pairs, −1.82e−05 S |
| 10 % | 87 pairs, −7.30e−05 S | 55 pairs, −5.53e−05 S |
| 20 % | 158 pairs, −2.09e−04 S | 108 pairs, −1.83e−04 S |

**The bar is −2e−05 S. Recovering 5 % of each pair's own d_pose at 3 real bits per token already
clears it.** That is the whole shape of the bet, written down before a single credit was realized
(`PREREGISTRATION.json`, `HEADROOM.json`). pd5's own rows say the break-even price for ITS median
credit is 7.255 bits/token — below pd5's own measured 12.300 median and inside this arm's cheap band.

## 5. The seg screen is a PER-PAIR budget, derived rather than picked

One flipped cell costs 8.482752943113527e-07 S. A pair can therefore only ever afford
`floor(base[pair] * pose_unit / seg_cell)` cells. MEASURED over the 588 non-floor pairs: **398 can
pay for none, 65 for one, 125 for two or more.** A flat screen is wrong in both directions — at 0
cells it discards the rows the heavy pairs can afford, at 2 cells it spends a 25-second refine on
light pairs that could never pay. Every refused row is written to disk with its own `d_cells`, so the
screen's cost is a measurement rather than an assumption.

## 6. THE CHARGE — eight sheets, 4,291 proposals, one real 600-frame encode each

Every candidate was charged before any credit was measured. Each sheet carries one proposal per
pair, so one encode charges the whole sheet frame-locally; twins byte-identical inside every encode,
every encode output-lossless.

| sheet | family | tokens | Δ bytes vs move 52 | **bits/token at sheet density** |
|---|---|---:|---:|---:|
| 00 | single, rank 0 | 588 | **+159** | **2.163** |
| 01 | single, rank 1 | 588 | +429 | 5.837 |
| 02 | single, rank 2 | 588 | +521 | 7.088 |
| 03 | single, rank 3 | 588 | +551 | 7.497 |
| 04 | single, rank 4 | 588 | +572 | 7.782 |
| 05 | single, rank 5 | 588 | +586 | 7.973 |
| **06** | **two-token, rank 6** | **1,117** | **+452** | **3.237** |
| **07** | **two-token, rank 7** | **1,117** | **+520** | **3.724** |

**pd5's eight sheets came back at 12.900 – 14.271 bits/token at the same kind of density. This arm's
cheapest sheet is 2.163, and its two-token sheets are 3.237 and 3.724.** pd4 DERIVED that 10.891
bits per changed token would clear the −2e−05 bar; the price lever is no longer the constraint —
it is beaten by a factor of five.

Per PROPOSAL, frame-locally:

| family | proposals | tokens | pooled bits/token | median | min |
|---|---:|---:|---:|---:|---:|
| single-token | 3,233 | 3,233 | 6.636 | 7.020 | −31.934 |
| **two-token** | 1,058 | 2,116 | **3.496** | **3.621** | −30.619 |

The context discount pd5 measured on the SECOND token survives being the generator rather than an
afterthought, and it is larger here: a two-token proposal costs about half as much per token as a
single one, because both members are cheap AND adjacent 97.7 % of the time.

**The cheap half is literally half.** Over all 4,291 charged proposals the median is 6.065
bits/token; 2,119 (49.4 %) price at ≤ 6. **389 proposals (9.1 %) carry a NEGATIVE real charge** —
the coder spends fewer bits on the changed field than on move 52's own.

| threshold | proposals | pairs | fraction of charged | median bits/token |
|---|---:|---:|---:|---:|
| ≤ 3 bits/token | 1,033 | 388 | 24.1 % | 0.943 |
| **≤ 5** | 1,728 | 506 | 40.3 % | 2.491 |
| **≤ 6 (the charter's threshold)** | **2,119** | **544** | **49.4 %** | **3.094** |
| ≤ 8 | 2,876 | 575 | 67.0 % | 4.244 |
| ≤ 12 | 3,830 | 585 | 89.3 % | 5.443 |

**The price's own noise floor, MEASURED and WORSE than pd5's.** 59 held-fixed control pairs are
repeated across all eight sheets: median spread **6.758 bits** (mean 6.851, max 14.867) against a
within-pair price range of median **12.424 bits** over 529 pairs. Signal is **1.84×** the noise,
where pd5 measured 3.3× (2.905 against 9.555). The reason is structural and worth saying plainly:
this arm's sheets edit **588** pairs where pd5's edited ~156, so every proposal's frame-local price
carries more spill from a crowd of neighbours. F3 does not fire, but the per-proposal price here is
a coarser instrument than pd5's, which is exactly why the SET re-price below is load-bearing rather
than a formality.

## 7. CREDIT AFTERWARDS — the wave

4,307 screens (render + frozen argmax), 3,104 refused by the per-pair seg budget, **1,203 realized
rows over 414 pairs**, seven shards, no prefix stop.

| quantity | MEASURED |
|---|---|
| credit improves the pair | **571 of 1,203 (47.5 %)** — the coin flip a pose-blind generator predicts (pre-registered 0.45–0.70) |
| best single credit | **−4.064e−06** d_pose |
| recovered fraction of the pair's own d_pose, per row | median −1.2 %, p75 **+17.5 %**, p90 **+56.5 %**, max **+98.5 %** |
| **best-of-n per pair** | **67.4 % of the 414 pairs improve**; recovered median **+10.9 %**, p75 **+46.2 %** |
| seg cells on realized rows | **51 repaid**, 687 neutral, 272 cost one, 193 cost two |
| seg screen pass rate | 27.9 % (single 30.8 %, two-token 19.2 %); refused rows' median d_cells 3 |
| **pairs with ANY positive resolved-pose credit** | **279** — F4's bar is 50 |
| **pairs whose best row pays at its OWN real price** | **128** |
| modelled net at the sheet price | **−1.2461e−04 S** |

**The carrier re-solve is doing almost all of the work, and this is the control that says so.**
Before the re-solve, a cheap token change makes the pair's pose **56.99× worse at the median** and
up to 2.4 million× worse; the re-solve recovers **98.0 %** of that damage. The credit this arm
reports is what survives AFTER the re-solve, never the stale number.

**The generator is pose-blind and the measurement says so.** Per row the median recovery is slightly
negative; the value is entirely in the tail and in being allowed eight shots per pair. That is the
honest shape of a price-first pool: it does not know where the pose is, it is simply cheap enough
that a coin flip can pay.

<!-- SET PRICE, LADDER, COMPOSITION AND THE THREE LEGS FOLLOW FROM THE SET ENCODES -->

<!-- # FORMALIZATION_PENDING: a measurement pass; the score arithmetic used throughout is the
registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->
