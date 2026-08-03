---
schema: ddm_hs1_heterogeneous_storage.v1
date_utc: 2026-08-03
arm: ddm_hs1 (find every axis on which "store it for one pair/frame, not all 600" pays)
lane_id: "lane_ddm_hs1_20260803"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false   # exact contest pointer 0.1910828242 [contest-CPU] UNMOVED. This arm fired no gate and no scorer pass.
axis: "[macOS-CPU advisory] NON-PROMOTABLE. $0 — NO scorer forward pass fired (slot held by ddm_pu2).
  Every number below is either read from an existing custody receipt or computed by reducing cached
  per-flip / per-pair arrays. Positive control on the reduction: pearson r = 0.999790 between two
  INDEPENDENT reductions of the per-pair seg axis (ru1 flip atlas vs sg1 dseg_per_pair) — §1."
verdict_scope:
  - claim: "the operator's selective-storage insight transfers from pose to seg along the PAIR index"
    verdict: REFUTED
    scope: "FAMILY - the per-pair index for the seg axis, over EVERY form tested:
      raw per-pair debt, per-pair cell selection, and static-mask + per-pair exception list"
    why_not_higher: "not PARADIGM: selective storage itself is VINDICATED on three other indices
      (space, edge, component). Only the PAIR index is dead for seg. And the measurement is at the
      tb1 ep399 endpoint (458,738 flips), 9.8% below the live base (508,639); the flatness is
      confirmed at two independent reductions of THAT endpoint, not at the live one (§7 R1-a)."
  - claim: "price-weighting the seg concentration relocates the optimal SPATIAL index"
    verdict: REFUTED
    scope: "FORMULATION - the m_def<0.25 tie-calibration band as the price proxy, on the 768-cell grid"
    why_not_higher: "one proxy, one grid. The EDGE index DOES re-rank under the same proxy (1.49x, §5),
      so price-dependence is real; it just does not show up spatially at c=16."
  - claim: "the useful index for seg is STATIC-SPATIAL, and per-pair cell selection is dominated"
    verdict: MEASURED (dominance, price-independent at equal payload mechanism)
    scope: "FORMULATION - cell-selection carriers on the 768-cell (16x16 px) grid"
verdict_scope_ladder: "INSTANCE < FORMULATION < FAMILY < PARADIGM."
consumes:
  - /Volumes/VertigoDataTier/pact/ddm_ru1_20260729/atlas_flat.npz        # 458,738 per-flip rows, tb1 ep399
  - /Volumes/VertigoDataTier/pact/ddm_sg1_20260731/dseg_per_pair.npy     # independent per-pair d_seg, same endpoint
  - /Volumes/VertigoDataTier/pact/ddm_sg1_20260731/cell_flip_mass.npy    # 24x32 cell mass (cross-check)
  - .omx/research/ddm_pz1_dpose_paired_n600_cx1_20260803.json            # per-pair d_pose at the cx1 base
  - .omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md              # the pose proof-of-concept
  - .omx/research/ddm_pc2_perclass_road_edges_20260802.md                # the edge decomposition + m_def semantics
  - .omx/research/ddm_sx2_g2_gate_and_displacement_carrier_20260803.md   # the 49 B static per-cell prior price
  - .omx/research/ddm_mf1_margin_morse_licence_20260803.md               # the per-component carrier price
  - /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/v4d_cx1_pj2ix2/report.txt
produces:
  - this memo (no code, no archive, no dispatch)
consumers: [MAIN, ddm_wf2 (price law), ddm_rd2 (live-base re-join)]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_hs1 — heterogeneous storage: which INDEX does the debt concentrate on?

## §0 ANSWER FIRST

**The operator is right, and the reason it works tells you exactly where it works.**

Selective storage pays **on whichever index the debt is concentrated on, and only there** — because
the address bits you spend are the price of *locating* the debt, and locating a debt that is
everywhere costs everything and buys nothing. So the question "does this pay?" has one answer per
index, not one answer overall. Measured, n600, all 600 pairs, no subset:

| index axis | Gini | top-1 slot | conc @ top-1 | top-1% of slots | verdict |
|---|---:|---:|---:|---:|---|
| **pose × PAIR** (time) | **0.8271** | 30.92% | **185.5×** | **62.01%** | **PAYS — banked** (`pu2`: ΔS −0.0354 at −3 B) |
| **seg × PAIR** (time) | **0.0858** | 0.29% | **1.7×** | **1.66%** | **DEAD — decisive negative** |
| **seg × CELL** (space, 768) | **0.8581** | 2.81% | **21.6×** | 15.51% | **PAYS — and it is STATIC** (`sx2`: 49 B → 430×) |
| **seg × EDGE** (9 present) | 0.6230 | 49.23% | 4.4× | — | pays, re-ranks under price (§5) |
| **seg × COMPONENT** (object) | — | — | — | — | **best measured price** (`mf1`: 81.4× better than `W`) |

1. **Pose and seg are opposite in the same vehicle.** Pose per-pair Gini **0.8271** — one pair holds
   30.92%, six hold 62.01%. Seg per-pair Gini **0.0858** — min 524 / median 746 / max 1333 flips,
   **max/min only 2.55×**, and **not one of the 600 pairs is flip-free**. Same n600, same vehicle
   family, same reduction. **The pose result does not transfer, and the reason is structural: seg
   flips live on the GT boundary (93.86% on-boundary, `ru1` receipt), and a dashcam frame's boundary
   length is nearly constant frame to frame.** There is no seg tail to find.

2. **For seg the concentration is SPATIAL, and that changes the economics by ~600×.** On the 768-cell
   (16×16 px) grid, Gini **0.8581**; **486 of 768 cells (63.3%) never flip in ANY of the 600 pairs**;
   the top 64 cells (8.3% of the frame) hold **65.93%** of all flips. A spatial index is **static** —
   you pay the address **once** and amortize it over all 600 pairs. That is precisely why `sx2`'s
   **49 B** counted static per-cell prior bought **21,048 B** of map term (**430×**) and beat every
   free generic extractor.

3. **Per-pair cell selection is DOMINATED — and this is price-independent.** At the same payload
   mechanism: a **full static 768-bit mask is 96 B raw / 49 B zlib and captures 100%**; per-pair
   top-64 cell selection captures only **88.71%** and costs **23,516 B** of address. Even the honest
   equal-*k* comparison loses: static top-128 = **91.22% at 62 B**, per-pair top-64 = **88.71% at
   23,516 B** — *more* capture for **379× less** address (§4).

4. **The hybrid does not rescue it either.** "Static mask + a per-pair exception list for the pairs
   the mask serves badly" is the natural repair. Measured: the static top-64 mask's per-pair capture
   spans only **45.60%–79.13%** (1.74×, std 6.24 pp), and the *missed* flips concentrate on the worst
   30 pairs at only **1.77×**. An exception list of 30 pairs reaches **3.02% of all flips** (§6).
   Per-pair heterogeneity in seg is weak at every form tested.

5. **The verdict is price-dependent, and I found one place where the ranking flips and one where it
   does not** (§5). By **mass** Road↔Lane is #1 (49.23%). By **cheapness density** (`m_def < 0.25`,
   `pc2`'s tie-calibration band) Road↔MyCar (**45.72%** cheap-within) and Road↔Undriv (**45.35%**) beat
   Road↔Lane (**30.62%**) by **1.49×** — an independent corroboration of `pc2` §"tie calibration",
   reached from the flip atlas rather than from its edge table. **But spatially the price proxy does
   NOT relocate the mask**: cheap-flip cell Gini 0.8547 vs 0.8581 all-flip, top-64 cell-set overlap
   **89.1%**. So: **price re-ranks WHICH EDGE to buy; it does not re-rank WHERE.**

6. **`mf1` is already an instance of the operator's directive, and it is the best-priced mechanism we
   have.** Its carrier is **1,483 components across 600 frames = 2.47 per frame** — storage for the
   frames that have a Movable object, nothing for the frames that do not — at **0.01564 B/flip,
   81.4× better than `W`**, for −0.0490 S (**7.92% of the current gap**). The directive is not a
   hypothesis on this axis; it is a shipped, priced win on the **COMPONENT** index.

**Pointer honesty.** The exact contest pointer `0.1910828242` [contest-CPU] is **UNMOVED**. This arm
measured no score, built no archive, and fired no gate. It is a measurement of *where to point the
next carrier*, not a carrier.

**Baseline named on every Δ below.** Live best **S = 0.7910689** (`pu2`, 353,805 B), seg leg
**0.4311790**, gap to PR130 (0.172141) **= 0.6189279**, live flips **508,639**,
**1% of gap = 9,295 B**, **`W` = 1.273108215332031 B/flip** = `4·DEN/PX`, exactly invariant.

---

## §1 Apparatus validity — checked before anything was read off it

The whole memo rests on one substrate: `ddm_ru1_20260729/atlas_flat.npz`, 458,738 per-flip rows with
`pair`, `y`, `x`, `gt_class`, `realized_class`, `m_def`, `gt_margin`, `dist_bin`, `gt_flicker`.

**Positive control on the axis this memo's headline negative depends on.** The per-pair seg
distribution is computed two independent ways and compared:

| quantity | source | independent? |
|---|---|---|
| per-pair flip count | `np.bincount` over `ru1.pair` | reduction of the flat atlas |
| per-pair `d_seg` | `sg1/dseg_per_pair.npy` | produced by the `sg1` exact-solve pipeline, not by `ru1` |

**pearson r = 0.999790.** Gini 0.0858 (sg1) vs 0.0861 (ru1); top-6 share 1.66% vs 1.65%. The
flatness is not an artifact of one reduction.

**Denominator control.** `ru1.pair` covers all 600 indices; `F.sum() == 458738` exactly; the
768-cell histogram sums to the same total. **No empty scope anywhere in this memo** — every
"top-k" is reported against its denominator and against the uniform share.

**Endpoint honesty, stated before use.** `ru1` and `sg1` are the **tb1 ep399** endpoint
(458,738 flips = 0.38888 S). The **live base is cx1/pu2 at 508,639 flips = 0.43118 S** — the atlas
holds **90.19%** of the live flip count. Every *structural* claim below (Gini, concentration,
dominance) is measured at ep399 and labelled as such. The live-base re-join is owed and belongs to
`ddm_rd2`. What makes the transfer plausible but *not proven*: the mechanism behind seg's per-pair
flatness (boundary length is frame-invariant) is endpoint-independent, while the mechanism behind
pose's per-pair concentration (a solved axis with a six-pair residual tail) is endpoint-*specific*.
See §7 R1-a for the attack on exactly this.

---

## §2 The pose side — what "it works" actually looked like

Read from `pz1` at the cx1 shipped base (n600, all 600 pairs):

| statistic | value |
|---|---:|
| Σ `d_pose` | 1.530839 (mean 0.00255140 — reproduces the live `report.txt` 0.00255143) |
| min / median / max | 1.195e-05 / 5.498e-04 / **0.473295** |
| max / min | **39,599×** |
| **Gini** | **0.8271** |
| top-1 pair | **30.92%** (conc **185.5×**) |
| top-6 pairs | **62.01%** (conc **62.0×**) |

`pu2` fixed six of these and landed **ΔS −0.0354283 at −3 archive bytes**, end-to-end
`upstream/evaluate.py` n600. The break-even question dissolved because the byte cost was ≤ 0 — but
the *reason* it could dissolve is this table: **when 1% of the slots hold 62% of the debt, the
address bits are negligible against the payload you skip.**

That is the general form of the operator's insight, and §3 is the test of whether seg has it.

---

## §3 The seg side — the decisive negative on the PAIR index

Per-pair flip count, `ru1`, all 600 pairs:

| statistic | seg × PAIR | pose × PAIR | ratio |
|---|---:|---:|---:|
| **Gini** | **0.0858** | 0.8271 | **9.6× flatter** |
| max / min | **2.55×** | 39,599× | 15,500× flatter |
| top-1 slot share | 0.29% (**1.7×** uniform) | 30.92% (185.5×) | **107× weaker** |
| top-6 slot share | 1.66% (**1.66×** uniform) | 62.01% (62.0×) | **37.4× weaker** |
| slots with **zero** debt | **0 of 600** | 0 of 600 | — |
| min / median / max flips | 524 / 746 / 1333 | — | — |

**Read plainly: there is no seg tail.** The worst pair carries 1.74× the average; the best carries
0.69×. Six pairs hold 1.66% of the seg debt where six pairs hold 62.01% of the pose debt.

**Why** — and the *why* is what makes this a law rather than a data point. `ru1`'s own receipt:
**93.86% of flips are exactly on the GT boundary**, 6.08% within 3 px, **0.058% interior**. `d_seg`
is therefore a measurement of *boundary length that is mis-placed*, and a forward dashcam's total
inter-class boundary length is nearly frame-invariant. Pose is the opposite: `d_pose` is a
**relative** 6-vector between two delivered frames, so it is dominated by the handful of pairs where
the ego-motion model is locally wrong.

**Consequence, binding on the next carrier design:** *do not build a per-pair seg correction
carrier.* Any byte spent on a per-pair seg address is spent locating a debt that is uniformly
distributed in that coordinate. Six pairs of seg attention is worth 1.66% of the seg leg = **0.00716 S
= 1.16% of gap** even if it were *free and perfect* — versus the same six pairs of pose attention,
which `pu2` actually banked at 5.41% of gap.

---

## §4 Where seg *does* concentrate — space, and why static beats per-pair by ~600×

768-cell grid (16×16 px, matching `sg1`'s own `cell_flip_mass` geometry):

| k (cells) | static capture | uniform | conc | static address (**once**) |
|---:|---:|---:|---:|---:|
| 1 | 2.81% | 0.13% | **21.6×** | 1 B |
| 8 | 15.51% | 1.04% | 14.9× | 8 B |
| 32 | 41.16% | 4.17% | 9.9× | 24 B |
| 64 | **65.93%** | 8.33% | 7.9× | 39 B |
| 128 | **91.22%** | 16.67% | 5.5× | 62 B |
| 192 | 98.45% | 25.00% | 3.9× | 77 B |
| **768 (full mask)** | **100.00%** | 100% | 1.0× | **96 B raw / 49 B zlib** |

Gini **0.8581**; **486 of 768 cells (63.3%) hold zero flips across all 600 pairs.**

### 4.1 The dominance (price-independent)

The operator's phrasing — "only for a particular frame or pair, instead of for all 600" — has a
per-pair reading on this index too: let each pair name **its own** best-k cells. Measured, at equal k:

| k | static capture | **per-pair** capture | static address | **per-pair** address |
|---:|---:|---:|---:|---:|
| 8 | 15.51% | 24.97% | 8 B | 4,600 B |
| 32 | 41.16% | 62.63% | 24 B | 14,108 B |
| 64 | 65.93% | **88.71%** | 39 B | **23,516 B** |
| 128 | **91.22%** | 100.00% | **62 B** | 37,089 B |

At equal *k* the per-pair index genuinely captures more — up to **+22.8 pp at k=64**. But *k* is not
the budget; **bytes are**, and the comparison at equal bytes is not close:

> **static top-128 captures 91.22% for 62 B. Per-pair top-64 captures 88.71% for 23,516 B.**
> **More capture, 379× less address.** And the full static mask reaches **100% for 96 B raw / 49 B
> zlib**, a ceiling the per-pair index can equal but never beat, at 1/245th the cost.

**This dominance does not depend on any mechanism price**, because both sides carry the same payload
through the same mechanism; only the address differs, and static's is strictly smaller for strictly
larger capture. It is the one conclusion in this memo that survives the entire 65× price spread.

### 4.2 Why the ~600× — the amortization identity

A static index is written **once** and read by all 600 pairs; a per-pair index is written **600
times**. So for any per-cell object of address cost `A` and payload `P`:

```
static:    A + P                    (P shared across 600 pairs)
per-pair:  600·A + 600·P'           (nothing shared)
```

**Selective storage pays exactly when the index it selects on is the index the debt varies along.**
Pose varies along `pair` → per-pair storage. Seg varies along `(y,x)` and is nearly constant along
`pair` → static storage, and paying 600× for a per-pair copy of a constant is the definition of
waste. `sx2` measured this end of it directly: **49 B → 21,048 B of map term, 430× leverage**, and
its own note that *"Canny's marginal contribution on top of 49 B of stored table is negative"* is the
same statement from the other side — the counted static table already holds what the generic
extractor was trying to re-derive per frame.

---

## §5 Price-dependence — where the ranking flips and where it does not

Per the `wf2` correction: **`W` is what a flip is WORTH (score-level, exactly invariant), not what
any mechanism COSTS.** Measured mechanism prices span **65×**, so a concentration curve alone cannot
rank work. Prices I am *citing*, not re-deriving (canonical home: `ddm_wf2`):

| mechanism | measured price | vs `W` | index it acts on |
|---|---:|---:|---|
| `sx2` static per-cell prior | **49 B → 21,048 B** (430×) | — | **CELL (static)** |
| `sx2` lossless `L*` description | **0.4981 B/flip** | 2.6× better | whole field |
| `mf1` Movable displacement carrier | **0.01564 B/flip** | **81.4× better** | **COMPONENT** |
| `qd1` / `gr1_cell_drop50` | **32.53 B/flip** | **25.5× worse** | CELL (drop) |
| `rz1` separatrix addressing | 0.60 bits/band-pixel | *different unit — not interconvertible* | band |

Note the two `sx2` rows and the two CELL rows disagree in sign: a **static per-cell prior** is 430×
leverage while a **per-cell drop** is 25.5× *worse* than `W`. **Same index, opposite verdicts,
because one ADDS a shared description and the other REMOVES per-pair payload.** That is the
coordinator's "two mechanisms, opposite verdicts on the same concentration" case, and it is already
in the receipts.

### 5.1 The test I ran: does price re-rank my curve?

Price proxy: **`m_def < 0.25`** — `pc2`'s tie-calibration band, where the flip is a runner-up by
< 0.25 logits on a boundary already in the right place, i.e. `SPEC_v8` §1's ~0-byte per-class bias
`b_c`. **This is a PROXY for cheapness, not a measured mechanism price**; treat every §5 number as
INFERRED, not MEASURED-as-a-price. 162,549 of 458,738 flips (35.43%) fall in the band.

**Spatially — NO re-rank.**

| | all flips | cheap flips (`m_def<0.25`) |
|---|---:|---:|
| cell Gini | 0.8581 | **0.8547** |
| top-32 cells | 41.16% | 40.38% |
| top-64 cells | 65.93% | 64.09% |
| nonzero cells | 282/768 | 281/768 |
| **top-64 cell-set overlap** | — | **57/64 = 89.1%** (Jaccard 0.803) |

Per-pair, likewise: cheap-flip Gini **0.0728**, min 183 / med 268 / max 456, **zero pairs with zero
cheap flips**. Price-weighting does not resurrect the pair index either.

**Per EDGE — a real re-rank, 1.49×.**

| edge | flips | % all | cheap-within | share of all cheap |
|---|---:|---:|---:|---:|
| Road↔Lane | 225,840 | **49.23%** | 30.62% | 42.54% |
| Road↔Undriv | 74,586 | 16.26% | **45.35%** | 20.81% |
| Undriv↔Movable | 54,347 | 11.85% | 32.83% | 10.98% |
| Road↔Movable | 52,607 | 11.47% | 35.21% | 11.40% |
| Road↔MyCar | 49,958 | 10.89% | **45.72%** | 14.05% |
| Lane↔MyCar | 681 | 0.15% | 44.20% | 0.19% |
| Lane↔Movable | 512 | 0.11% | **3.52%** | 0.01% |
| Lane↔Undriv | 130 | 0.03% | 23.08% | 0.02% |
| Movable↔MyCar | 77 | 0.02% | 33.77% | 0.02% |

**Mass ranks Road↔Lane #1; cheapness-density ranks Road↔MyCar and Road↔Undriv #1, by 1.49×.** This
reproduces `pc2`'s independent reading (*"Road↔Undriv + Road↔MyCar … nearly half these flips are
runner-up by <0.25 logits on a boundary that is already in the right place … tie calibration"*) from
a different reduction of the same atlas. **Road↔Lane is the biggest pile and simultaneously the
*hardest* pile: it is 1.49× less near-tie than the two edges ranked below it by mass.**

### 5.2 The compounding correction — bytes in the invisible complement

`wf2`'s dimension correction bounds every number above from a direction the concentration curve
cannot see: the head is **rank-4 ⇒ 140 of 144 dims provably invisible to `d_seg`**; `D`'s null
dimension is **80.6742%**, **22.70% blind to both scorers**; pose is `rank(J) ≤ 6` with 6 of 12
outputs unscored. **Bytes landed in the invisible complement buy zero flips at infinite cost.** My
768-cell mask indexes the *image* lattice, which is not the same object as the visible complement of
`D` — a cell can be "hot" in flips and still be reached only through directions `D` annihilates.
**This is a real gap in my curve and I am naming it, not papering it:** the composition
`cell-mass × visible-fraction-per-cell` is **OWED**, and until it is measured, every §4 capture
figure is an **upper bound** on what a carrier aimed by that mask can realize.

---

## §6 The hybrid — static mask plus a per-pair exception list

The natural repair for §3 is: keep the static mask, and pay per-pair only for the pairs it serves
badly. Measured at the static top-64 mask:

| k | global capture | per-pair capture: min | p05 | med | p95 | max | pairs below 0.8× global |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 25.09% | 12.89% | 16.82% | 25.09% | 35.04% | 43.34% | 101 |
| 32 | 41.16% | 26.96% | 31.98% | 41.52% | 52.26% | 58.74% | 44 |
| **64** | **65.93%** | **45.60%** | 53.83% | 66.97% | 75.25% | **79.13%** | **23** |
| 128 | 91.22% | 62.13% | 78.66% | 93.64% | 97.74% | 99.23% | 12 |
| 192 | 98.45% | 75.37% | 93.39% | 99.59% | 100% | 100% | 1 |
| 256 | 99.83% | 91.18% | 99.15% | 100% | 100% | 100% | 0 |

At k=64 the per-pair capture spread is only **1.74×** (std 6.24 pp) and the **missed** flips
concentrate on the worst 30 pairs at just **1.77×** (8.85% of the miss vs 5.00% uniform). An
exception list of 30 pairs — address `log2 C(600,30)/8 = 21 B` once — reaches **3.02% of all
flips = 0.01302 S = 2.10% of gap** *if the correction were free and perfect*, which it is not.

**The exception-list hybrid is not refuted as an idea; it is refuted as a *seg* idea, because the
quantity it would exploit (per-pair heterogeneity) is the quantity §3 measured to be absent.** The
same 21 B address on the **pose** axis buys the 6-pair tail worth **62.01%** of that leg.

---

## §7 Round-1 adversarial self-review

**R1-a — "the flatness is an endpoint artifact." PARTLY OPEN; the strongest attack.**
The atlas is tb1 ep399 (458,738 flips), the live base is cx1/pu2 (508,639) — 9.8% apart. If the live
base's *extra* 49,901 flips were concentrated on a few pairs, the live per-pair Gini could be higher
than 0.0858. I cannot rule this out without a scorer pass, which I do not have. Two things bound it:
(i) 49,901 flips is 9.8% of the total, so even if **all** of them landed on 6 pairs, those 6 pairs
would hold ≤ 11.4% of the live seg debt — still **5.4× below** pose's 62.01%; (ii) the mechanism
(boundary length is frame-invariant) is endpoint-independent. **Verdict: the FAMILY-level negative on
the seg pair-index survives its own worst case with a 5.4× margin, but the exact live Gini is OWED
to `ddm_rd2`'s live-base re-join.**

**R1-b — "you proved the skew you assumed."** This was the pre-registered failure mode. I checked it
two ways. First, the headline result is a **NEGATIVE** on the axis I was pointed at (pair), which is
the opposite of confirming an assumed skew. Second, my *positive* result (spatial) is confirmed by an
artifact I did not produce and could not have tuned: `sx2`'s 49 B static per-cell prior was measured
**before** this arm existed and pays **430×**. Third: every top-k figure is reported **with its
uniform share and denominator**, so the reader can see the concentration ratio rather than a bare
share. **Residual risk:** the 768-cell grid is one granularity; at c=32 (192 cells) the Gini is 0.821
and at c=16 it is 0.858, so the spatial finding is stable across the two grids I tried, but a
boundary-conforming (non-rectangular) partition is untested.

**R1-c — "you priced the gain at `W`, which is worth not cost."** Caught by the coordinator mid-run
and **corrected**: the §4 dominance is now stated **without any price**, as capture-vs-address, and
it holds across the full 65× price spread. §5 carries mechanism prices as **citations to `wf2`**, and
the one price I generate myself (`m_def<0.25`) is labelled a **PROXY**, not a price. **No number in
this memo converts flips to bytes via `W` as a cost.**

**R1-d — "the cheap/all subset comparison is a prefix trap."** The `m_def<0.25` subset is selected by
the governing quantity, so its mean `m_def` (0.1182) differs from the population's (0.5894) **by
construction** — I report the ratio (0.201) and explicitly disclaim it as the selection, not a
control. The claim I actually make is **distributional over ALL flips** (cell Gini of the cheap
subset vs of the population; cell-set overlap), which is not a prefix and not a subset mean. **No
contiguous-block or prefix sampling appears anywhere in this memo** — all 600 pairs and all 458,738
flips enter every reduction.

**R1-e — "static-dominates is trivially true because the full mask is only 96 B."** Correct, and that
*is* the finding: the address is cheap **because the debt is static**. The non-trivial content is
that per-pair selection cannot buy back the difference at any k (the +22.8 pp peak at k=64 costs
23,516 B and is beaten outright by 62 B of static top-128), and that the same argument run on pose
gives the opposite answer. **The dominance is trivial only in hindsight, which is the correct shape
for a law.**

**R1-f — "capture ≠ repair."** Stated in §5.2 and restated here: every capture figure is an **upper
bound**. A carrier aimed at a cell fixes some fraction of that cell's flips at some mechanism price,
and the `cell-mass × visible-fraction` composition is **OWED**. Nothing in this memo licenses a ΔS
claim; the only ΔS quoted is `pu2`'s and `mf1`'s, both from their own receipts.

**R1-g — negative-existence honesty.** "No per-pair seg carrier should be built" is a *forward*
recommendation, not an existence claim. Where I make existence claims I scope them: **I did not find
any per-pair seg concentration in the three forms I tested** — raw per-pair debt, per-pair cell
selection, static+exception hybrid — **within the scope of `ru1`'s tb1-ep399 atlas and `sg1`'s
per-pair d_seg**. Forms not tested: per-pair *edge* selection, per-pair *component* selection,
per-pair payload at fixed static address (which is what tokens already are).

---

## §8 What this hands the next arm

1. **Kill the per-pair seg carrier before it is designed.** (§3, FAMILY-scoped, 5.4× margin under its
   own worst case.) Redirect that design budget to the **static-spatial** and **component** indices.
2. **The seg index to build on is STATIC-SPATIAL + EDGE-RESOLVED.** Road↔Lane is 49.23% of flips and
   is *more* spatially concentrated than the population (cell Gini **0.9035** vs 0.8581; top-64 cells
   hold **81.51%** of its flips vs 65.93% of all). A 49 B-class static object, edge-resolved to
   Road↔Lane, is aimed at 22.1% of the entire gap (`pc2`) with the best-measured static price (430×,
   `sx2`).
3. **But buy Road↔MyCar / Road↔Undriv FIRST if the carrier is a per-class bias.** They are 1.49×
   richer in near-tie flips (45.7% / 45.4% vs 30.6%) at 27.2% of flips combined — i.e. Road↔Lane is
   the biggest pile *and* the hardest. (§5.1, INFERRED from the `m_def` proxy; a measured price for
   an edge-resolved `b_c` is **OWED** to `wf2`.)
4. **The COMPONENT index is under-exploited and best-priced.** `mf1` runs 2.47 components/frame at
   81.4× better than `W`. Movable-touching edges are 23.5% of flips; **no equivalent
   component-indexed carrier exists for Lane**, whose 0.59% area / IoU 0.263 orbit is the classic
   long-tail. **Owed: is Lane component-decomposable the way Movable is?**
5. **OWED to `wf2`:** (a) `cell-mass × visible-fraction-per-cell` under `D`'s 80.67% null — turns my
   capture upper bounds into realizable ones; (b) a measured price for edge-resolved `b_c`;
   (c) prices for the 8 edges that have none.
6. **OWED to `ddm_rd2`:** the live-base (508,639-flip) re-join of the per-pair Gini, which closes
   R1-a.
