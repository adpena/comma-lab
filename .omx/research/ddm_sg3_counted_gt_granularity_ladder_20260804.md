# ddm_sg3 — Counted-GT artifacts: the granularity ladder, measured

**Date** 2026-08-04 · **Arm** `ddm_sg3` · **Axis** `[macOS-CPU advisory]`, `score_claim=false`
**Instrument** `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/`
(`gt_argmax_n600.npy`, `cx1_argmax_n600.npy`, 600×384×512 uint8) — **$0, no scorer forward.**
**Scripts** `.omx/research/ddm_sg3_scripts/{census,ladder,static_and_semantic}.py`
**Evidence** `.omx/research/ddm_sg3_ladder_addr.json`, `.omx/research/ddm_sg3_static_semantic.json`

Operator ask (2026-08-03): *"ship in the archive artifacts from GT that would be surgically
targeted for only particular features and particular frames. At the class level, but also below,
like for each island / road-lane interface."*

**Verdict in one line: the mass is where the operator said it is, but the ENCODING intuition
inverts — going *below* class costs 2.2–3.5× MORE, not less, and "particular frames" is dead
(Gini 0.109). The only counted-GT granularity that pays is COARSER and STATIC, not finer.**

---

## §0 THE LOGIC THAT MAKES THIS BANKABLE (bz1 mirage law)

Every number here is **DESCRIPTION-ONLY**: the bytes to *describe* a GT-derived set at a given
granularity. No realizer is priced, no scorer was consulted, nothing is scorer-guided. A realizer
can only **add** bytes. Therefore this table is a **lower bound**, and:

| if | then |
|---|---|
| `address_cost ≥ achievable_seg` | **DEAD for every realizer** — bankable KILL |
| `address_cost < achievable_seg` | says **NOTHING** about profitability — never a win |

I emit no wins. I emit **kills, specs, and a break-even survival per rung** that `lr2` can price
for free when its legal-realization ladder lands. Realizer class of every row below:
**DESCRIPTION-ONLY**. No ORACLE row appears; no row is banked as a gain.

---

## §1 COMPLIANCE FRAME + THE ARITHMETIC THAT KILLS BULK

**Compliance (settled, per `video_specific_not_gt_derived_is_the_counted_test_three_way_20260803`).**
GT label maps are **this-clip ⇒ VIDEO-SPECIFIC ⇒ COUNTED**. That is legal and is exactly what the
operator proposed: pay for them in `archive.zip`. The rule-118 fake is smuggling video-derived
content into `inflate.py` as "code". **Every byte in this memo is typed COUNTED.** No row proposes
a free-rider. (The three-way taxonomy's *operator-property* tier — `D` nullity, the 22.70% blind
fraction, openpilot geometry — is GENERIC/free but is **not** what a GT label map is.)

**The arithmetic, recomputed from scratch (do not inherit):**

```
DEN = 37,545,489      PX = 600·512·384 = 117,964,800
W   = 4·DEN/PX = 1.2731082153320312 B/flip          (break-even exchange rate)
MEASURED cx1 flips = 508,640  →  d_seg 0.004311794  →  seg leg 0.4311795 S   ✓ matches as1 exactly
```

Baseline stated explicitly (`qd1` discipline — a ΔS without its baseline is unanchored):
live best **S = 0.7910689**, bar (PR130) **0.172141**, **gap = 0.6189279**; **1% of gap = 9,295 B.**
Seg is **69.67% of the live gap**. (Memory `m66`'s "1% = 10,907 B" was computed against the older
0.7262358 gap; at today's frontier it is 9,295 B. Same equation, moved baseline.)

**Bulk GT storage is structurally dead, and here is the number:**
the seg gap to the bar is 473,629 flips; at par `W` that is **602,981 B = 1.70× the ENTIRE live
archive (353,805 B)**. There is no version of "ship the GT masks" that fits. Only surgical
storage can pay — which is precisely the operator's framing, and it is correct.

---

## §2 THE GRANULARITY LADDER — MEASURED, real coders, both cost axes

Address bytes = **best of zlib-9 / lzma-9e / brotli-11** on the packed indicator. This is a
*measured* coder result, not an i.i.d. entropy bound. **That distinction is load-bearing:** the
i.i.d. bound said the exact per-slot address costs 591,108 B; the real coder achieves **343,592 B**
— the bound was **1.72× pessimistic** and would have produced a false kill.

| rung | object | px in set | flips | **bytes** | B/flip | vs W | coder |
|---|---|---:|---:|---:|---:|---:|---|
| **L0 static** | static 2D risk map | 43,798 | 508,640 | **3,997** | 0.0079 | **0.01×** | brotli |
| L4 cell | 32×32 per-frame occupancy | — | 508,640 | **2,844** | 0.0056 | 0.004× | brotli |
| L4 cell | 16×16 per-frame occupancy | — | 508,640 | **11,356** | 0.0223 | 0.02× | brotli |
| L1 class | GT mask MyCar | 29,993,509 | 64,065 | 28,732 | 0.4485 | 0.35× | lzma |
| L1 class | GT mask Movable | 1,460,325 | 119,933 | 57,929 | 0.4830 | 0.38× | brotli |
| L1 class | GT mask Undriv | 58,413,281 | 151,521 | 83,966 | 0.5542 | 0.44× | brotli |
| L1 class | GT mask Road | 27,407,046 | 444,945 | 293,050 | 0.6586 | 0.52× | brotli |
| L1 class | GT mask Lane | 690,639 | 236,816 | 189,460 | 0.8000 | 0.63× | lzma |
| L2 iface | Road↔MyCar | 63,027 | 63,027 | 31,658 | 0.5023 | 0.39× | brotli |
| L2 iface | Undriv↔Movable | 61,892 | 61,892 | 39,769 | 0.6426 | 0.50× | brotli |
| L2 iface | Road↔Movable | 57,225 | 57,225 | 37,128 | 0.6488 | 0.51× | brotli |
| L2 iface | Road↔Undriv | 89,545 | 89,545 | 58,936 | 0.6582 | 0.52× | lzma |
| **L2 iface** | **Road↔Lane** | 235,148 | 235,148 | **198,468** | 0.8440 | 0.66× | lzma |
| L2 iface | Lane↔MyCar | 903 | 903 | 2,044 | 2.2636 | **1.78×** | brotli |
| L2 iface | Lane↔Movable | 681 | 681 | 2,403 | 3.5286 | **2.77×** | brotli |
| **L6 slot** | exact flip set (all) | 508,640 | 508,640 | **343,592** | 0.6755 | 0.53× | lzma |
| **L3 crop** | Road↔Lane per-island | 235,148 | 235,148 | **700,394** | 2.9785 | **2.34×** | crop+coded |

### §2.1 REFUTATION — "below class, per island" costs MORE

This is the operator's hypothesis, and the measurement inverts it.

| interface | islands | median px | bbox hdr B | mask B | crop total | full-vol | **crop/full** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Road↔Lane | **112,077** | **1** | 658,452 | 41,942 | 700,394 | 198,468 | **3.53×** |
| Road↔Undriv | 28,768 | 2 | 169,012 | 12,295 | 181,307 | 58,936 | **3.08×** |
| Road↔MyCar | 12,914 | 4 | 75,870 | 6,265 | 82,135 | 31,658 | **2.59×** |
| Undriv↔Movable | 13,650 | 2 | 80,194 | 12,260 | 92,454 | 39,769 | **2.32×** |
| Road↔Movable | 11,754 | 2 | 69,055 | 11,892 | 80,947 | 37,128 | **2.18×** |

**Two independent mechanisms, both measured:**

1. **The flip set is DUST, not islands.** Road↔Lane is **112,077 connected components of median
   1 pixel**. A 47-bit bbox header per island costs **658,452 B — 15.7× the 41,942 B of actual
   mask payload it wraps.** You cannot address dust one grain at a time.
2. **Cropping destroys the context the coder was exploiting.** Even ignoring headers, the summed
   crop masks (41,942 B) plus fragmentation lose to whole-volume coding, because lzma/brotli
   exploit cross-frame and cross-island redundancy that cropping severs.

This gives `as1`'s "Lane's deficit is DIFFUSE" its **mechanism**: Lane error is *boundary dust*
distributed along the separatrix, not lost islands. Consistent with `pc2` (93.89% of flips within
3 px of separatrix; interiors 0.058%).

**The charitable reading — GT *semantic* islands (a lane dash IS an island) — also shows no
concentration:**

| GT class | islands | per frame | median px | p90 px | flips carried | **top 1% of islands carry** |
|---|---:|---:|---:|---:|---:|---:|
| Lane | 16,581 | 27.6 | 7 | 147 | 185,801 | **10.4%** |
| Movable | 2,207 | 3.7 | 104 | 2,130 | 78,833 | **7.6%** |

Top 1% of islands carry ~10% of flips — i.e. **flips are spread across islands roughly in
proportion to island count.** There is no head to select. Island identity is not a selection axis.

### §2.2 REFUTATION — "particular frames" is dead

**per-frame Gini = 0.1089** (nearly uniform). Top-10% of frames carry **14.7%** of flips against a
uniform 10%; max/min frame ratio only 3.72×. This independently reproduces `hs1`'s "seg × PAIR
Gini 0.0858 = DEAD" on a different lineage (`cx1` argmax vs `hs1`'s cache). **Frame selection buys
a 1.47× concentration — nothing.** The `hs1` asymmetry stands and is now confirmed twice:
**cells concentrate (Gini 0.8581), frames do not (0.086–0.109).**

---

## §3 STATIC vs PER-FRAME — the crossover, quantified

**The single most important structural fact I measured: only 43,798 of 196,608 pixels (22.28%)
EVER flip, across all 600 frames.** Max flips for any one pixel: 143/600.

| | address bytes | capture | precision | B/flip |
|---|---:|---:|---:|---:|
| (a) **STATIC 2D risk map** | **3,997** | 100% of flips | **1.936%** | 0.0079 |
| (b) **EXACT per-slot set** | **343,592** | 100% of flips | 100% | 0.6755 |

The static map addresses 43,798 px × 600 frames = 26,278,800 slots to catch 508,640 flips.
**The precision premium — what it costs to go from 1.94% to 100% precision — is 339,595 B
= 0.22612 S of rate.** That premium buys avoiding collateral damage on 25,770,160 addressed
non-flip slots. Converting the premium to flips at `W`: 266,745 flips.

> ### THE CROSSOVER
> **Static addressing beats exact addressing iff the realizer's collateral flip rate on
> already-correct pixels is below `266,745 / 25,770,160` = 1.035%.**
> Below 1.035% collateral → ship the 3,997 B map. Above → pay 339,595 B for precision.

This is the operator's crossover, stated as an exchange rate rather than a binary. It is a
**mechanism** question (`lr2`/`wf2` own it), and it is now a single number they can test against.

---

## §4 ARTIFACT SPEC — the one counted-GT object worth building

### L0-A: static risk map + modal label — **4,266 B**

| component | bytes | content |
|---|---:|---|
| 2D risk bitmap (384×512) | **3,997** | which of 196,608 pixels ever flip (43,798 = 22.28%) |
| modal GT label per risky px | **269** | 3-bit class, brotli'd (43,798 values → 269 B) |
| **TOTAL** | **4,266** | = **0.002841 S** of rate = **1.13% of the live archive** |

- **Type:** VIDEO-SPECIFIC ⇒ **COUNTED**, shipped in `archive.zip`. No rule-118 exposure.
- **Receiver consumption (the #417 bijection):** the decoder reads the bitmap into a static
  384×512 per-pixel class prior and conditions rendering at those pixels toward the modal class.
  Consumption is provable by byte-mutation: flip any bitmap byte → the conditioned pixel set
  changes → decoded frames change. **Not a marker; a consumed field.**
- **Label purity (MEASURED):** mean 88.4%, median 94.2%; 60.7% of risky pixels are >90% pure in
  their GT class over the 600 frames. The static label is a *real* signal, not noise.
- **INFORMATION CEILING (MEASURED, honest cap):** flips whose GT class equals their pixel's static
  modal label = **141,893 / 508,640 = 27.90%**. So this artifact can express the correct target
  for at most 27.90% of flips → **ceiling 0.12028 S** of the 0.43118 S seg leg.
- **Break-even survival = 0.002841 / 0.12028 = 2.36%.**
- **lr2 rung:** *receiver-context paint* / static prior (also usable as *micro edit-head*
  conditioning). It is a 2D static field — the cheapest rung to realize.

**Why this is the pick:** it needs **2.36% survival** where the Road↔Lane explicit mask needs
**66.30%** — a **28× easier bar** — and its prize (0.12028 S ceiling) is 19.4% of the live gap.

### Break-even survival, the whole ladder

| rung | bytes | flips addressable | rate S | seg S @100% | **break-even survival** | lr2 rung |
|---|---:|---:|---:|---:|---:|---|
| L4 cell 32×32/frame | 2,844 | 508,640 | 0.00189 | 0.43118 | **0.44%** | token-translate **gate only, no label** |
| **L0 static map+label** | **4,266** | 141,893 | 0.00284 | **0.12028** | **2.36%** | **receiver-context paint** |
| L4 cell 16×16/frame | 11,356 | 508,640 | 0.00756 | 0.43118 | **1.75%** | token-translate **gate only, no label** |
| L2 Road↔Undriv | 58,936 | 89,545 | 0.03924 | 0.07591 | **51.70%** | warp+solved-residual |
| L6 slot exact | 343,592 | 508,640 | 0.22878 | 0.43118 | **53.06%** | warp+solved-residual |
| L1 GT Lane mask | 189,460 | 236,816 | 0.12615 | 0.20075 | **62.84%** | warp+solved-residual |
| L2 Road↔Lane | 198,468 | 235,148 | 0.13215 | 0.19934 | **66.30%** | warp+solved-residual |
| L3 crop Road↔Lane | 700,394 | 235,148 | 0.46636 | 0.19934 | **233.96%** | — refuted |

**Caveat on the L4 cell rungs, stated plainly:** they are the cheapest addresses on the ladder but
they carry **no label** — they say *where* errors are, never *what* is correct. They are gates, not
correctors, and `as1` already MEASURED a static 16×16 gate at only **−0.001678 S** realized. Cheap
address ≠ strong mechanism. I do not recommend them as standalone artifacts and I do not claim
their break-even is achievable; they are listed for completeness of the ladder.

---

## §5 HONEST KILLS — each stated as a SPEC with its byte-cut factor

### KILL-1 (bankable) — Road↔Lane as an explicitly shipped per-slot mask

This is the operator's named interface. It is **dead for every realizer whose survival is ≤ 66.30%**
— and the only survival ever measured on it is `cg3`'s **55.5%**. The kill is *conditional on that
survival*, and I state the condition rather than hiding it: the address floor does not kill the row
at survival > 66.30%, it kills it at the survival we have actually observed.

```
address floor            198,468 B = 0.132152 S rate      (MEASURED, real coder, lower bound)
seg at cg3 survival .555              0.110632 S
NET with a FREE realizer             +0.021519 S          <-- LOSS before any realizer exists
break-even survival      0.132152 / 0.199337 = 66.30%
cg3 MEASURED survival                       55.50%        short by 1.195x
```

**This independently reproduces `cg3`'s net loss (+0.130073 S) from a different instrument** —
`cg3` priced a camera-paint *realization*; I priced only the *address*, and the address alone
already exceeds the achievable seg **at cg3's measured survival**. Two arms, disjoint methods, same
verdict. This matters for attribution: `cg3`'s loss is **not** merely an artifact of its chosen
paint mechanism — swapping realizers rescues this row only by moving survival past 66.30%, which is
a **1.195× improvement on the best survival anyone has measured here**, not a free re-roll.

**SPEC (not a kill of the family):** needs the **survival × address-efficiency product improved
1.195×**. Either survival 0.555 → **>0.663**, or address 198,468 → **<166,100 B**, or any product
of the two. `lr2`'s receiver-local realization is the named escape and is exactly the right shape:
it attacks survival, which is the cheaper of the two factors to move.

### KILL-2 — per-island / per-crop addressing at ANY interface

Needs a **2.18×–3.53× byte cut** merely to reach parity with the whole-object mask it replaces,
and the whole-object mask is itself already dead at Road↔Lane. Root cause is structural, not
implementational: **median island = 1–4 px, and the header is 15.7× the payload.** A cleverer
header (say 24 bits instead of 47) still leaves Road↔Lane crops at 336,231 B + 41,942 B = 1.9× the
full-volume mask. **Retire per-island addressing of the FLIP set.** (GT *semantic* islands remain
alive as a **generator** target — a lane dash is 7 px median and 27.6/frame, which is the
~8-dim-polyline territory CLAUDE.md already routes to `gt2`; that is a grammar, not an address.)

### KILL-3 — "particular frames" as a selection axis

Gini 0.1089; top-10% frames carry 14.7% vs uniform 10%. Needs a **~6× concentration increase** to
be worth a per-frame address. **Retire frame selection.** Per-frame *refinement* is a different
question and is answered by the §3 crossover (1.035% collateral), not by selection.

### KILL-4 — Lane↔Movable and Lane↔MyCar interfaces

Address costs **2.77×** and **1.78×** of `W` respectively — these two interfaces cost more to
address than their flips are worth, before any realizer. Combined they are 1,584 flips (0.31% of
seg). Needs a 2.77×/1.78× byte cut. **Not worth revisiting**; correctly ignored by every arm.

---

## §6 WHAT I REFUTE IN MY OWN CHARTER

1. **My charter's framing "below class → cheaper" is wrong, and I measured it wrong-way-round.**
   Below-class addressing is 2.2–3.5× *more* expensive. The charter's own instruction to report
   the denominator is what caught it (112,077 islands, median 1 px).
2. **I nearly emitted a false kill from an i.i.d. entropy bound.** My first pass computed exact
   per-slot addressing at 591,108 B (91.3% of budget, "regime B nearly dead"). The real coder does
   it in **343,592 B (53.1%)** — the bound was **1.72× pessimistic**. Recorded because it is the
   generic failure mode of this whole ladder: *entropy bounds kill rows that real coders save.*
3. **`as1`'s "Road node participation 89.62%" does not reproduce on my count.** Flips with Road on
   either side = 153,242 + 291,703 = 444,945 / 508,640 = **87.48%**, not 89.62%. Same cache, so
   this is a definitional difference (probably undirected-edge-graph node participation vs my
   slot count), not a vehicle difference. Low stakes; flagged, not resolved.
4. **The charter told me `pc2` measured "Road↔Lane = 49.2% of flips"; on `cx1` I measure 46.23%**
   — which matches `as1`'s `cx1` figure exactly. The 49.2% is the `ru1`/`tb1` vehicle. Vehicle
   difference, correctly predicted by `as1`; I confirm `as1`, not `pc2`, for the live vehicle.
5. **Not measured, and I will not assert it:** whether the static map's 27.90% ceiling can actually
   be realized. That is `lr2`'s ladder. My number is a *ceiling*, never a gain.

---

## §7 HANDOFFS

- **`lr2`** — three rungs are pre-priced for you: static map+label (2.36% break-even),
  cell gates (0.44%/1.75%, label-free), exact slot (53.06%). And **one number decides the
  static-vs-exact fork: collateral flip rate ≷ 1.035%** (§3).
- **`wf2`** — the address floors above are additive to your mechanism prices; a row is dead when
  `address + mechanism ≥ achievable seg`. Road↔Lane is already dead on address alone.
- **`gt2`** — GT *semantic* islands are the live handoff: Lane = 27.6 components/frame, median 7 px,
  p90 147 px. That is a grammar/generator object, not an address, and it is yours.
- **`cg3`** — your camera-paint loss is confirmed structurally: the Road↔Lane address floor exceeds
  achievable seg at your measured survival. Your queued receiver-local escape is the right move and
  needs **1.195×**.

**No scorer slot consumed. `fz1`'s n600 slot untouched.**
