# ddm_pd7 — the PRICE-FIRST generator, pass 2, on the move-53 field

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pd7_price_first_pass2_20260916`. Base: **move 53**, S 0.1361014714463198 @ 179,286 B,
archive sha `aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957`, re-derived from
`.omx/state/canonical_frontier_pointer.json` at bind time. Axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the
distortion legs; bytes EXACT through the shipped RLC1 coder and the shipped container. **No
score is claimed. Only `upstream/evaluate.py` on shipped bytes is a score, and MAIN fires.** No
Modal, no authorization, no fire, no completion, no packet.

Pose solver, stated once: every per-pair credit comes from
`ddm_jg5_pose_resolve_on_edited_renders.refine_pair` at PoseNet batch 1 on the moved render,
with `outer_rounds=40, max_gn_iterations=400`; every n600 number comes from
`ddm_up2_shipping_pose_solve.measure_pose` at batch 8.

---

## 1. Why a second pass on a new field is a different unit

pd6 inverted the generation order and it worked. This arm runs the same order on the field pd6
left behind, because two things about that field are MEASURED to have changed:

1. **101 of the 600 pairs are re-rendered.** The repair family re-opens on a re-rendered pair
   (`repair_family_exhausts_per_object_20260910`: 375 of 489 new rows came from re-rendered
   pairs and none from unchanged ones).
2. **Every price expired.** The cheap half is a property of the prior's own context and move
   53's 118 new tokens changed that context (pc2/pc3). §3 measures the expiry rather than
   assuming it.

This arm also widens the enumeration to **≤ 8 bits/token** (pd6 measured ≤ 6 and never tested
the credit above it) and reports the 6/8 split.

## 2. The binding

`pd4` binds move 52 in module globals evaluated at import, and every pd6 stage calls
`pd4.bind_move52`. This arm re-points those constants at move 53 PROCESS-LOCALLY
(`ddm_pd7_price_first_pass2.rebind_pd4_to_move53`) and then lets pd4's own verification path do
the work: archive bytes and sha, argmax sha, field sha, raw bytes and sha, the pointer's score
re-derived from its three components, and a re-assertion THROUGH pp1/sj1/pd1/pd2/pd3 rather
than through this module's copy of the facts. No pd4, pd6 or shipped-runtime source was edited.

| bound object | value |
|---|---|
| archive | `aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957`, 179,286 B |
| token field | pd6's `setprice/set_02.npz`, sha `f3ab56ed0996ed1e…` — 118 tokens over 101 pairs vs move 52 |
| decoded token plane | `4cb0b147bdae8ce0618938453bb6ccc5b7b5a4927d6ad4e1b0bd9dd9ff46119c` — reproduced INDEPENDENTLY here: the pricer's own `field_to_u8` of that npz hashes to the same value pd6's parse-back reported, so the field bound here IS the plane the receiver decodes |
| SegNet argmax | move 53's OWN cold decode, sha `bed84af6ec58a320…`, 12,128 flips |
| per-pair pose base | move 53's OWN cold decode at batch 8, sha `1c75d6539d4dc5a4…`, mean 4.13644120761059e−06 |
| score re-derived | 100·0.00010288 + √(10·4.14e−06) + 25·179,286/37,545,489 = **0.1361014714463198**, equal to the pointer's field |

At move 53's operating point: S per archive byte **6.658589531221714e−07**, S per seg cell
**8.482849604221636e−07** (through move 53's own instrument ratio 1.0006776569920846), S per
unit of one pair's d_pose **1.2957025872757242**. All three EXPIRE at the next pointer move.

## 5. What the lever needed, DERIVED before any credit existed

`HEADROOM.json`, written before the wave. A pair can credit at most its own d_pose; the gain is
`base[pair]·frac·pose_unit` and the cost of one token at a given price is `(bits/8)·S_per_byte`.

| recovered fraction of the pair's own d_pose | 3 bits/token | 6 bits/token | 8 bits/token |
|---|---|---|---|
| 5 % | 68 pairs pay, **−3.396e−05 S** | 34 pairs, −2.217e−05 | 24 pairs, −1.731e−05 |
| 10 % | 105 pairs, **−8.903e−05** | 68 pairs, −6.791e−05 | 53 pairs, −5.785e−05 |
| 20 % | 150 pairs, −2.085e−04 | 105 pairs, −1.781e−04 | 93 pairs, −1.616e−04 |

**The bar is −2e−05 S, and it is TIGHTER here than it was for pd6.** Cell for cell against
pd6's own `HEADROOM.json` (the artifact, not its memo table — the two disagree, and the artifact
is what this arm cites):

| recovered / price | pd6 (move 52) | pd7 (move 53) | shift |
|---|---:|---:|---:|
| 5 % @ 3 bits | 70 pairs, −3.484e−05 | 68 pairs, −3.396e−05 | −2.5 % |
| 5 % @ 6 bits | 35 pairs, −2.249e−05 | 34 pairs, −2.217e−05 | −1.4 % |
| 10 % @ 3 bits | 108 pairs, −9.170e−05 | 105 pairs, −8.903e−05 | −2.9 % |
| 10 % @ 6 bits | 70 pairs, −6.967e−05 | 68 pairs, −6.791e−05 | −2.5 % |
| 20 % @ 3 bits | 158 pairs, −2.148e−04 | 150 pairs, −2.085e−04 | −2.9 % |
| 20 % @ 6 bits | 108 pairs, −1.834e−04 | 105 pairs, −1.781e−04 | −2.9 % |

**The headroom shrinks 2–3 % per move**, because each pass lowers the base d_pose its successor
would have to recover from (4.2075e−06 → 4.1364e−06, −1.7 %), only partly offset by the pose
unit rising (1.2847 → 1.2957, +0.9 %). That is a horizon on this family: the price-first
generator makes its own successor harder by exactly the amount it succeeds. And the 8-bit column
pd6 never computed does NOT clear the bar at 5 % recovery (24 pairs, −1.731e−05) — the widened
threshold has to earn its place through the pairs it ADDS, not through its price.

The per-pair seg budget (`floor(base[pair]·pose_unit / seg_cell)`) over the 588 non-floor pairs:
**407 pairs can pay for no cells at all, 58 for one, 123 for two or more** (pd6: 398 / 65 / 125).

### The controls that make the base credible

| control | outcome |
|---|---|
| **move 53's cold decode, re-run by this arm** | **3,662,409,600 B sha `8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b`** — BIT-IDENTICAL to pd6's, 1,201.3 s on CPU at the contest thread count (4), `decoded_field_matches_admitted: true`. pd6 hashed this decode and then removed the payload under its retention cap; this arm rebuilt the same bytes rather than inheriting a number |
| the field IS the decoded plane | the pricer's `field_to_u8` of `field_move53.npz` hashes to `4cb0b147bdae8ce0…`, the plane sha pd6's own parse-back reported — two independent paths to the same bytes |
| **the batch-1 / n600 pose-base band, MEASURED over ALL 600 pairs** | max gap **5.464e−09** (0.13 % of the base mean), median **2.994e−10**, p99 3.873e−09; the wave's gate is 2× the non-floor max = **1.0929e−08**. batch-1 mean 4.136432e−06 against the n600 batch-8 mean 4.136441e−06 — the two instruments agree to six figures. pd6 carried a band of 2.264e−08; this one is MEASURED on this field rather than inherited |
## 3. The capture, and the two controls that make it credible

`ddm_pd7_price_first_pass2 capture` wraps `LaneMixer.coding` and `.end_frame` as pure observers
— each calls the shipped method and returns its value unchanged — and runs the pricer's own
control encode of move 53's field. Capture wall **1,324.5 s**.

| control | outcome |
|---|---|
| **the pricer re-packs move 53's own archive** | **179,286 B sha `aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957`**, byte-identical to the live pointer, **twins identical in-process**, stream 119,038 B (ideal 119,037.33), decoded plane sha `4cb0b147bdae8ce0…` — **with the capture wrappers installed**. **F1 does not fire** |
| **the captured rows ARE the charged rows** | per frame, the sum of −log2 p over the TRUE symbols must equal the pricer's own `per_frame_bits`. MEASURED over all 600 frames: **max absolute 0.000346 bits, max relative 2.436e−07** against the charter's 1e−06 bar. **F2 does not fire** |

## 4. THE PRICES EXPIRED — and the expiry is LOCAL

pd6 retained its own capture rows on move 52's field, so the two are comparable cell for cell.
The charter's premise was that the cheap half expires per move. It does, but not uniformly.

| head-8 overlap (the 8 cheapest (position, symbol) pairs, move 52 vs move 53) | median | mean | min | max |
|---|---:|---:|---:|---:|
| all 588 pairs | 8 | 7.27 | 1 | 8 |
| **the 101 RE-RENDERED pairs** | **6** | **5.35** | 1 | **7** |
| the 487 unchanged pairs | 8 | 7.67 | 5 | 8 |

**Not one of the 101 edited pairs keeps all eight of its cheapest changes, and the median edited
pair loses two of eight; most unchanged pairs lose none.** 64 of 588 pairs lost their move-52
rank-0 cell out of the move-53 head-1024 entirely. Deeper in the list the pool is stable: the
head-1024 cell overlap is 947/1024 at the mean, and it is the same for edited (942) and
unchanged (948) pairs. So the edit re-ranks the TOP of a pair's price list without moving the
list itself.

This is the charter's premise measured rather than asserted, and it is sharper than the premise:
the expiry is CONCENTRATED where the field moved.

### The honest consequence, and where this pool is NOT new

pd6 could say its pool was a different part of the plane from pd4's and pd5's saliency pool.
That still holds here — the head-8 overlap with the pd4+pd5 pool is **0 pairs of 588** (head-64:
2; head-1024: 24) — but it does **not** hold against pd6 itself: **587 of 588 pairs' cheapest
eight cells touch cells pd6 priced.** The price-first generator on this object nominates
essentially the same CELLS twice.

What makes the wave new is different, and it is measured:

| | rows |
|---|---:|
| this arm's proposals | **4,291** |
| of those, proposals pd6 actually RENDERED and re-solved | **959 (22.3 %)** |
| on the 101 edited pairs | 213 of 794 (26.8 %) |
| on the 487 unchanged pairs | 746 of 3,497 (21.3 %) |
| of the 959 repeats, ones pd6 measured as IMPROVING their pair | 430 |

**77.7 % of this pool is ground pd6 never rendered** — not because the cells are different but
because pd6's own per-pair seg budget refused 3,104 of its 4,291 proposals before they were ever
realized. The repeats are not waste either: the field moved under them, so the same edit renders
differently, prices differently and is screened against a different per-pair budget.

### Where the cheap half lives

| SegNet class | share of the cheap half (pd7) | pd6 on move 52 | area share |
|---|---:|---:|---:|
| **Road** | **90.81 %** | 90.99 % | 23.2 % |
| Undrivable | 6.55 % | 6.40 % | 49.5 % |
| Movable | 2.49 % | 2.46 % | 1.24 % |
| MyCar | 0.13 % | 0.14 % | 25.4 % |
| Lane | 0.014 % | 0.01 % | 0.59 % |

602,112 cells over 588 pairs, mean row 212.3 of 384; the dominant move is token 0 → token 1.
The cheap half is the road surface just ahead of the car on both fields — the geometry of the
cheap half is a property of the coder and the scene, not of the field's 118 edited tokens.

## 4b. The cheap half, MEASURED (first-order, a RANKING never a charge)

| quantity | pd7 on move 53 | pd6 on move 52 |
|---|---:|---:|
| pairs with candidates | **588** of 600 | 588 |
| candidates emitted | **4,291** (3,233 single, 1,058 two-token) | 4,291 |
| pairs whose cheapest change is NEGATIVE-priced | **209 of 588 (35.5 %)** | 219 (37.2 %) |
| cheapest change per pair, first-order bits | median **0.793**, min **−21.565**, max **3.550** | 0.771 / −31.258 / 3.577 |
| two-token partner is a neighbour | **98.20 %** | 97.7 % |
| held-fixed price-control pairs | **59** | 59 |

The twelve pairs without candidates are pp1's floor pairs, excluded by construction; they are
the twelve largest-d_pose pairs and hold 56.7 % of the n600 pose mass, so every number in this
memo is drawn on the remaining 43.3 % over 588 pairs.
### Where the new ground actually is — the pool decomposed against pd6's own logs

pd6 retained every screened proposal, not only the realized ones, so this arm's 4,291 proposals
split exactly three ways against it:

| this arm's proposals | rows | what pd6 knew about them |
|---|---:|---|
| pd6 **REFUSED** them on its per-pair seg budget | **2,850 (66.4 %)** | pd6 measured their `d_cells` on move 52's render and never rendered a pose; median 3 cells, min 1, max 28, and **1,302 of them were at ≤ 2 cells** — refused because the PAIR's budget was 0 or 1, not because the edit was expensive |
| pd6 **REALIZED** them | 959 (22.3 %) | rendered, seg-censused and carrier-re-solved on move 52; 430 improved their pair there |
| pd6 **never saw** them | **498 (11.6 %)** | new to the price list entirely |

**And the re-rendered pairs are where the genuinely new proposals concentrate.** 307 of those
498 sit on the 101 edited pairs — **3.04 new proposals per edited pair against 0.39 per
unchanged pair, a 7.8× density**. That is the repair-family law
(`repair_family_exhausts_per_object_20260910`) showing up in the PRICE pool, having been
measured before only in the saliency pool.

So the honest shape of this pass is: it is not a new part of the plane, it is **pd6's own
refused pool re-screened against a moved render and a moved per-pair budget, plus a small
genuinely-new head that lives on the pairs the last move re-rendered.** A pd6 refusal is a
PREDICTION here, not a verdict, because both the render and the budget moved under it.
## 6. THE CHARGE — eight sheets, 4,291 proposals, one real 600-frame encode each

Every candidate was charged before any credit was read. Each sheet carries one proposal per
pair, so one encode charges the whole sheet frame-locally; twins byte-identical inside every
encode, every encode output-lossless.

| sheet | family | tokens | Δ bytes vs move 53 | **bits/token** | pd6 on move 52 |
|---|---|---:|---:|---:|---:|
| **00** | **single, rank 0** | 588 | **+226** | **3.075** | **2.163** |
| 01 | single, rank 1 | 588 | +431 | 5.864 | 5.837 |
| 02 | single, rank 2 | 588 | +519 | 7.061 | 7.088 |
| 03 | single, rank 3 | 588 | +545 | 7.415 | 7.497 |
| 04 | single, rank 4 | 588 | +564 | 7.673 | 7.782 |
| 05 | single, rank 5 | 588 | +579 | 7.878 | 7.973 |
| **06** | **two-token, rank 6** | **1,117** | **+497** | **3.560** | 3.237 |
| **07** | **two-token, rank 7** | **1,117** | **+565** | **4.047** | 3.724 |

**This table is the arm's cleanest result.** Ranks 1 through 5 reproduce pd6's prices to within
1.5 %. The rank-0 sheet does not: it is **3.075 bits/token against pd6's 2.163, +42 %**. The
two-token sheets are up 10 %. **Move 53 spent the 118 cheapest tokens on the object, and the
charge says so — the head of the price list got dearer and the body did not move.** That is the
same statement the head-8 / head-1024 expiry census makes, arrived at through a completely
different instrument: one is a comparison of two capture tables, the other is sixteen real
600-frame encodes.

### The cheap half, CHARGED, and the 6/8 split the charter asked for

| threshold | proposals | pairs | fraction of charged | median bits/token |
|---|---:|---:|---:|---:|
| ≤ 3 bits/token | 990 | 399 | 23.1 % | 1.052 |
| ≤ 5 | 1,728 | 511 | 40.3 % | 2.629 |
| **≤ 6 (pd6's threshold)** | **2,141** | **544** | **49.9 %** | **3.283** |
| **≤ 8 (this arm's)** | **2,897** | **574** | **67.5 %** | **4.332** |
| ≤ 12 | 3,861 | 585 | 90.0 % | 5.497 |

**Widening 6 → 8 adds 756 proposals and 30 pairs.** Over all 4,291 charged proposals the median
is 6.025 bits/token (pd6: 6.065), the range is −24.751 to +24.887, and **341 proposals (7.9 %)
carry a NEGATIVE real charge** — the coder spends fewer bits on the changed field than on move
53's own (pd6: 389, 9.1 %).

## 7. CREDIT AFTERWARDS — the wave, all 4,291, no prefix stop

7 shards, **4,291 screened, 3,141 refused by the per-pair seg budget, 1,150 realized rows over
406 pairs**.

| quantity | pd7 on move 53 | pd6 on move 52 |
|---|---|---|
| credit improves the pair | **514 of 1,150 (44.7 %)** — inside the pre-registered 0.40–0.70 | 47.5 % |
| best single credit | **−8.509e−06** d_pose | −4.064e−06 |
| recovered fraction per row | median −2.8 %, p75 **+11.9 %**, p90 **+45.6 %**, max **+98.5 %** | −1.2 / +17.5 / +56.5 / +98.5 |
| **best-of-n per pair** | **65.0 % of the 406 pairs improve**; recovered median **+8.1 %**, p75 **+41.3 %** | 67.4 % / +10.9 % / +46.2 % |
| seg cells on realized rows | **32 repaid**, 645 neutral, 267 cost one, 206 cost two | 51 / 687 / 272 / 193 |
| seg screen pass rate | **26.8 %** (single 29.7 %, two-token 17.9 %); refused rows' median d_cells 3 | 27.9 % |
| **pairs with ANY positive resolved credit** | **264** — F4's bar is 40 | 279 |
| **pairs whose best row pays at its OWN real price** | **72** | 128 |
| modelled net at the sheet price | **−5.172e−05 S** | −1.246e−04 |

**The carrier re-solve is doing the work, and this is the control that says so.** Before the
re-solve a cheap token change makes the pair's pose **53.0× worse at the median** and up to
160,540× worse; the re-solve recovers **97.8 %** of that damage. Every credit reported here is
what survives AFTER the re-solve, never the stale number.

**The harvest is thinner than pd6's and the headroom predicted it.** 72 pairs pay at their own
price where pd6 had 128, and the modelled net is −5.2e−05 where pd6's was −1.2e−04. The generator
is the same and the credit shape is the same; what changed is that the cheapest rung costs 42 %
more and the pose left to recover is 1.7 % smaller.

## 8. THE SET PRICE — and a defect in this arm's own chain

| iteration | pairs / tokens | ledger bits | real frame-local bits | **residual bias** | measured pairs used |
|---:|---:|---:|---:|---:|---:|
| 0 | 72 / 85 | −210.34 | **−85.01** | **−59.58 %** | 0 |
| 1 | 72 / 85 | −210.34 | — | — | **0 — a NO-OP** |
| **2** | **57 / 71** | **−204.45** | pending | pending | **50** |

**Iteration 1 was a no-op, and the cause was mine.** pd5's set-price reads its measured bits from
`STATE.json`, which a SEPARATE `setabsorb` call populates; my chain ran `setprice` twice without
it, so iteration 1 re-ranked on the same ledger and reproduced iteration 0's set byte for byte
(`set_repeats_previous: true`, `measured_pairs_used: 0`, identical field sha). Recorded rather
than hidden: the fix was to absorb iteration 0's own encode and run a real iteration 2, which
folds 50 measured pairs and shrinks the set from 72 pairs to 57 — the same direction pd6's
iteration took (128 → 110).

The no-op was not free and not wasted: its encode ran to completion on a field byte-identical to
iteration 0's, in a **separate process with its own natively-built backend**, and produced a
byte-identical archive (`3ea59adf27225ab8…`, 179,273 B). That is the twin check at its strongest
— and the CLOSED archives built from those two streams, in two separate directories, are
byte-identical too.

**The ledger over-credits by 59.6 % on this field**, against pd6's −23.2 %. The spill onto
unedited frames is −19.34 bits, 18.5 % of the signed whole-field total. pd4's warning that a
per-bit rule misprices the subset it chooses is not merely reproduced here, it is larger — which
is why the SET re-encode, not the ledger, is what any byte claim below rests on.

## 9. THE COMPOSITION — re-verified, and cleaner than pd6's

A 600-pair overlay rendered from the 72-pair set-00 field, a carrier re-solve over all 600 pairs
starting from the SHIPPED codes, and an n600 pose leg on the result.

| leg | value |
|---|---:|
| per-pair sum of credits | **−2.705648e−05** |
| **composed, re-verified (n600 mean shift × 600)** | **−2.705721e−05** |
| **realized fraction** | **1.0000** |
| **spill onto the 528 unkept pairs** | **0.000000e+00** |
| stale pose (candidate renders, shipped carrier) | 7.488e−05 = **18.1× base** |
| carrier coordinates changed | **482** over 72 spliced pairs |

The composed n600 resolved pose is **4.0913458560e−06** against the base **4.1364412076e−06**.
pd6 measured a realization fraction of 0.9989 with real spill; this arm measures **1.0000 with
none**, so the per-pair estimator is neither optimistic nor pessimistic on this vehicle.
## 11. THE COLD PARSE-BACK — and the census that EXPLAINS pd6's open question

The candidate tree was decoded cold on CPU at the contest thread count (4), through the tree's
own `inflate.py` and `runtime.f26_inflate`, **1,213.9 s**, raw 3,662,409,600 B sha
`ff43a9c97c72d0917ac4c2b856315648eddf3c0d3f69717eca132bfecf37a324`.

| control | outcome |
|---|---|
| the decoded TOKEN PLANE is the admitted field | **PASSES** — `decoded_field_matches_admitted: true`, 66 tokens over 53 pairs |
| **the decode reproduces the OVERLAY's own odd frame, pair by pair** | **53 of 53 IDENTICAL, max \|Δ\| = 0 grey levels.** pd6 measured 9 of 110 disagreeing by 114–154 |
| **d_seg on the DECODE** (`step0 --raw`, the same jg1 instrument move 53's leg used) | **12,127 flips against move 53's 12,128 — 1 cell REPAIRED**, exactly what the per-pair census predicted (`d_cells_total = −1`); 28 argmax cells moved in all |
| **d_pose on the DECODE** (`up2.measure_pose` batch 8 on the candidate's own `0.raw` and its own carrier) | **4.099227789357173e−06 — IDENTICAL TO ALL DIGITS to the admission; max per-pair \|Δ\| 0.0, ratio 1.0** |

### pd6's nine unexplained pairs, EXPLAINED

pd6 left its sharpest open question in §15.5: *"The nine bad pairs have no identified mechanism.
They do not differ from the 101 in token count, token move, or seg cost. What is measured is
that their decoded odd frame differs from the overlay's and their pose credit does not survive;
what is NOT measured is why."* It recorded the consequence as a law about the RENDERER: that
`render-edits` reproduces the receiver's odd frame for most edited pairs and not for all.

**That law is wrong, and this arm can say so because it hit the same genus from a different
direction and then went and checked pd6's own retained fields.**

pd6 rendered its overlay from `set_00` (128 pairs) and then closed `set_01` (110 pairs). Its
set-price iteration does not only DROP pairs — it re-ranks, so a surviving pair's chosen EDIT
can change between rungs. Comparing pd6's two retained fields, plane by plane:

| MEASURED on pd6's own artifacts | result |
|---|---|
| pairs whose plane differs between `set_00` (the overlay's field) and `set_01` (the closed candidate) | **9: 82, 154, 167, 238, 268, 305, 487, 508, 532** |
| pd6's nine pairs whose decode disagreed with the overlay | **82, 154, 167, 238, 268, 305, 487, 508, 532** |
| **set equality** | **EXACT** |
| pairs whose plane differs between `set_00` and `set_02` (pd6's SHIPPED 101-pair candidate) | **0** — which is precisely why its decode reproduced the admission to all digits |

**The renderer was faithful the whole time. The bookkeeping was stale: the overlay showed a
different edit from the one the archive carried, for exactly those nine pairs.** pd6's cure
worked for the right reason by accident — dropping the nine removed exactly the pairs whose
composition was stale.

**This arm reproduced the genus independently and caught it BEFORE closing.** Its ladder winner
`set_03` differs from the overlay field `set_00` on **2 pairs (238 and 419)**. Rather than close
on a stale pose leg, the overlay was re-rendered from `set_03`'s own field, the carrier
re-solved over all 600 pairs against that overlay, and the pose re-measured. The corrected
composition is 4.0992277894e−06 where the stale one read 4.0971030257e−06 — the stale number was
**optimistic by 2.12e−09** in the n600 mean. Small on this rung; it was 4.9× on pd6's.

**LAW, corrected and cheap: a composed overlay is valid only for the pairs whose token plane in
the SHIPPED field equals the plane the overlay was rendered from. When a size ladder's rungs are
set-price ITERATIONS, one composition does NOT serve the whole ladder, because an iteration
re-ranks as well as drops. Compare the planes — it costs a few seconds — and re-render when they
differ.** pd6's decode-and-drop cycle (two full parse-backs, ~40 minutes) is the expensive
diagnosis of a defect a per-pair `array_equal` finds for free.

## 12. THE CANDIDATE — every leg on the shipped bytes' own decode

| leg | value | how |
|---|---:|---|
| rate | **−1.331718e−05** | **−20 B EXACT** — the closed archive, re-solved carrier included |
| seg | **−8.482850e−07** | **1 cell REPAIRED** (12,127 against 12,128), MEASURED by `step0 --raw` on the candidate's own `0.raw`; carried to T4 by move 53's own same-instrument ratio 1.0006776569920846 |
| pose | **−3.176196e−05** | **4.099227789357173e−06** MEASURED by `up2.measure_pose` batch 8 on the candidate's own `0.raw` with its own carrier |
| **S projected** | **0.13605554401927** | 100·0.00010287151715039579 + √(10·4.099227789357173e−06) + 25·179,266/37,545,489 |
| **net vs move 53** | **−4.592743e−05** | **2.296 bars** · **1.98× the 34.8 B container-break sd** |

**All three legs are negative and all three are measured on the decode.** The candidate is 20
bytes SMALLER than move 53, repairs a SegNet cell, and lowers pose by 0.9 %. It sits inside the
pre-registered band [−8e−05, −2e−05].

### The SIZE ladder, on real CLOSED archives

The ladder is not monotone in set size, and it did not pick the converged ledger's favourite.

| candidate | pairs / tokens | **exact closed archive** | rate | seg | pose | **net** | **bars** |
|---|---:|---|---:|---:|---:|---:|---:|
| set 00 (iter 0) | 72 / 85 | 179,278 B `1ba72b3b…` (−8 B) | −5.327e−06 | +8.483e−07 | −3.792e−05 | −4.239885e−05 | 2.120 |
| set 02 (iter 2) | 57 / 71 | 179,270 B `7f67b1f3…` (−16 B) | −1.065e−05 | +1.697e−06 | −3.342e−05 | −4.237867e−05 | 2.119 |
| **set 03 (iter 3) — THE CANDIDATE** | **53 / 66** | **179,266 B `5c6bf403…` (−20 B)** | **−1.332e−05** | **−8.483e−07** | **−3.176e−05** | **−4.592743e−05** | **2.296** |

set 02 is a smaller archive than set 00 and scores WORSE, because it costs two SegNet cells where
set 00 costs one; set 03 is smaller still and scores BEST, because it REPAIRS one. **The ladder
on closed archives chose the row, exactly as pd6's handoff said it must.**

### Twins, at both levels

| level | outcome |
|---|---|
| in-process | each encode runs two arithmetic encoders; every one produced byte-identical members |
| **cross-process, encoder member** | `pd7set03` was encoded in two separate PROCESSES, each with its own natively-built rc64 backend: both members **179,262 B, identical sha** |
| **cross-process, CLOSED archive** | the candidate was closed from each of those two streams, staged in separate directories: both **179,266 B sha `5c6bf403b4cb4554fe24a46bdf5b46d62876854764a10a22d8c90d76a7292ee6`** |
| (bonus) | the no-op iteration-1 field is byte-identical to iteration 0's, so its independent encode is a third process agreeing on `3ea59adf27225ab8…`, 179,273 B |

## 13. The chartered chain, closed

| step | outcome |
|---|---|
| byte-close on move 53 | **179,266 B sha `5c6bf403b4cb4554fe24a46bdf5b46d62876854764a10a22d8c90d76a7292ee6`**, identity control passed, every frame-1 section byte-identical except the carrier (+4 B for 365 moved coordinates over 53 pairs) |
| twins | byte-identical across two processes at BOTH the encoder member and the closed archive |
| cold n600 public parse-back | **1,213.9 s** CPU at the contest thread count, `decoded_field_matches_admitted: true`, raw sha `ff43a9c97c72d091…` |
| manifest (Catalog #420) | **51 rows**, all hashes re-verified from OUTSIDE the tree, all runtime dependencies listed, derived listing validated |
| literal census (rule 118) | **CLEAR** — the only files differing from move 53 are `archive.zip`, the two archive pins inside `inflate.py`, and the derived `MANIFEST.sha256` that restates them. The RLC1 rider is present: `stage-tail` was handed the tail SUFFIX from the start, so no archive here is the 64-byte-optimistic undecodable kind |
| candidate + frontier public smokes | **identical behaviour**: both `REACHED_CUDA_GATE` on `inflate.sh` (1.97 s) and both `REACHED_TOKEN_DECODE` on the 300 s public-path probe |
| decode wall clock | **INHERITED** from move 53's own `t4_direct` leg, 1,185.9 s against the 1,260 s limit; `behavior_digests_equal: true` (`9f6e71680a13d859…` on both). The legacy raw-identity digest differs, as it must: it includes the two archive pins |
| **seal** | **`SEAL_ddm_pd7_price_first_pass2_contest_cuda.json`, seal sha `54c6ae9de90a32cf76110bf34e3d0064b95bff7bb371e8fc9374bf52e5d5d3b2`**, axis `contest_cuda`, admit bar −2e−05, **SEAL_VALID** |

**MAIN fires.** This arm ran no Modal call, wrote no authorization, no completion and no packet.

## 14. Falsifiers, as pre-registered

| falsifier | outcome |
|---|---|
| F1 the pricer does not reproduce move 53's own archive | **does not fire** — 179,286 B sha `aab908d3…`, twins identical, **with the capture wrappers installed** |
| F2 the captured rows are not the charged rows | **does not fire** — max relative **2.436e−07** against the 1e−06 bar (max absolute 0.000346 bits) |
| F3 the per-proposal price does not resolve above its noise | **does not fire, but barely** — noise floor **6.349 bits** (59 held-fixed controls) against a 12.298-bit within-pair range over 529 pairs: **1.94×**, where pd6 had 1.84× and pd5 3.3× |
| F4 fewer than 40 pairs carry a cheap proposal with positive resolved-pose credit | **does not fire** — **264** pairs do, and 72 pay at their own real price |
| F5 the admitted set nets worse than −2e−05 S | **does not fire** — **−4.592743e−05**, 2.296 bars, on legs measured on the shipped bytes' own decode |
| F6 the set re-price differs from the ledger by >10 % after iteration | **does not fire** — **−59.58 % → −19.05 % → −1.35 %** over three real iterations |
| F7 the projection's pose leg and the decode's differ by more than 1.5× | **does not fire** — **ratio 1.0, identical to all digits**, max per-pair \|Δ\| 0.0 |

**No falsifier fires.** The band was [−8e−05, −2e−05] and the measured net is −4.59e−05.

## 15. What this does NOT claim

1. **No score of any kind.** Every S here is a PROJECTION on measured legs. Only
   `upstream/evaluate.py` on shipped bytes is a score, and MAIN fires. No Modal call, no
   authorization, no completion, no packet, no pointer write.
2. **The distortion legs are `[macOS-CPU advisory]`** — frozen CPU-torch PoseNet and SegNet
   against DALI-lineage GT. The seg leg is the jg1 instrument's count carried to T4 by move 53's
   own same-instrument ratio 1.0006776569920846, the same carry move 53's own leg uses, not an
   independent T4 measurement.
3. **The decode wall clock is INHERITED, not measured on this candidate.** Its scope is stated in
   the seal: identical normalized receiver code; the candidate's payload-dependent time was not
   re-measured on T4. The CPU decode did run, in 1,213.9 s.
4. **pp1's twelve floor pairs stay excluded** on pp1's measurement. They are the twelve
   largest-d_pose pairs and hold 56.7 % of the n600 pose mass, so every number here is drawn on
   the remaining 43.3 % over 588 pairs.
5. **The first-order price is a RANKING, never a charge.** Every candidate was charged by a real
   600-frame sheet encode before any credit was read, and the selected SET was re-encoded as one
   field before any byte claim.
6. **The bits/token numbers are THIS pool at THIS edit shape on THIS field**, measured once.
   None of it is a law. In particular the +42 % on the rank-0 sheet is one comparison of two
   fields, not a rate at which prices decay.
7. **The explanation of pd6's nine pairs is a set identity, not a re-decode.** What is MEASURED
   is that pd6's nine disagreeing pairs are exactly the nine whose plane differs between its
   overlay field and its closed candidate, and that the field it shipped differs on none. This
   arm did not re-run pd6's retracted decode to confirm the mechanism end to end; it reproduced
   the genus on its own object and measured 53/53 agreement after re-rendering.

## 16. Custody

Store **`/Volumes/APDataStore/pact/ddm_pd7/`**. **Vertigo was never opened for writing.**
Retained **1.5 GiB** against the charter's 3 GiB cap — 2,261 files, 777,293,422 B hashed in
`RETENTION_MANIFEST.json`. APDataStore free space MEASURED at every heavy step: 23 GiB at start,
17 GiB through the capture, 15 GiB through the wave, 12 GiB through the composition, 8 GiB at
the tightest point (which is why the candidate's decode was certified and freed before the twin
encode was allowed to start), **24 GiB after the certified prune**.

Every bulk payload removed was hashed first, with the exact command that rebuilds it, in
`BULK_CERTIFICATE.json`: move 53's own decode (3.66 GB, sha `8a14f55a…`), the candidate's decode
(3.66 GB, sha `ff43a9c9…`), the superseded set-00 overlay (1.83 GB, sha `80cf7d10…`), the
shipped field's overlay (1.83 GB, sha `5f14966b…`), and the pricer's 12 per-field u8 planes and
encoder states (1.46 GB, each naming its source npz and its own recorded sha). **Nothing was
deleted before its identity was recorded.**

| path | what |
|---|---|
| `PREREGISTRATION.json` · `HEADROOM.json` · `BASEBAND.json` | the prediction and its falsifiers and the DERIVED headroom, all written before any credit; and the wave's own gate, MEASURED over all 600 pairs |
| `capture/frames/frame_????.npz` · `capture/CAPTURE.json` | the coder's own price rows, 600 frames, and the byte-identity control |
| `plan/` · `CHEAP_GEOMETRY.json` · `DISJOINTNESS.json` · `PRICE_EXPIRY.json` · `PD6_OVERLAP.json` · `PD6_REFUSED.json` | the cheap half, where it lives, and the four censuses that say what is new about it |
| `sheets/` · `PRICE_MERGE.json` · `CHEAP_HALF.json` | the eight sheets, the 4,291 real-encode charges, the noise floor and the threshold sensitivity |
| `search/wave/realized_*.jsonl` · `screen_*.jsonl` · `WAVE_ANALYSIS.json` · `CREDIT.json` | **every realized row and every screened proposal** — 1,150 realized, 4,291 screened, winners and losers |
| `setprice/STATE.json` · `set_0?.npz` · `LADDER.json` · `close/*/CLOSE.json` · `close/*/candidate_archive.zip` | the four set-price iterations with their absorbed residuals, the size ladder, and every closed archive including the two twins |
| `pose/pose_stale.npy` · `pose_resolved.npy` · `pose_resolved3.npy` · `POSE_ON_DECODE_set03c.npy` · `refine/` · `refine3/` | the pose vectors, including the decode's own, and both carrier re-solves |
| `parseback*/PARSEBACK_RESULT.json` · `RENDER_AGREEMENT_set03c.json` · `seg_cand/argmax_n600.npy` · `PUBLIC_SMOKE.json` · `LITERAL_CENSUS.json` · `seal_inputs/` | the decode receipts, the render-agreement census, the decode's own SegNet argmax, the smokes, the rule-118 census and the seal inputs |
| `BULK_CERTIFICATE.json` · `RETENTION_MANIFEST.json` | the custody record |

Producer: `experiments/ddm_pd7_price_first_pass2.py` — a thin process-local rebinding of pd4's
move-52 constants onto move 53, plus a MEASURED base-band gate, a move-53 headroom derivation, a
`run` passthrough and a `price-merge` that reads this arm's own sheet encodes. Every stage is
pd6's, imported and called unchanged; downstream, pd4's merge/carry/assemble, pd5's
setprice/setabsorb, `ddm_sj1_rlc1_price` and `ddm_sj1_joint_admission` are all unchanged.
Nothing under `ddm_pd1`–`ddm_pd6`, `ddm_sj1`, `ddm_jr*`, `ddm_psa*`, `ddm_mrs*` or
`/Volumes/VertigoDataTier/` was written.

## 17. Two defects this arm found in its own chain, recorded rather than hidden

**Defect 1 — the missing absorb.** pd5's set-price reads its measured bits from `STATE.json`,
which a SEPARATE `setabsorb` call populates. This arm's chain ran `setprice` twice without it, so
iteration 1 re-ranked on the same ledger and reproduced iteration 0's set byte for byte. Caught
by reading `measured_pairs_used: 0` rather than by trusting `set_repeats_previous: true` — a
fixed point and a no-op look identical from the outside. **LAW: `set_repeats_previous` is only
evidence of convergence when `measured_pairs_used` is non-zero.**

**Defect 2 — the premature prune.** The chain was written to certify and delete the pricer's u8
planes as soon as the wave ended. `ddm_sj1_rlc1_price.guard()` re-verifies EVERY registered
field's u8 by sha on every call, so that deletion would have refused the next `add-field` with
`field custody changed` and blocked the whole set-price ladder. Caught by reading `guard()`
before the chain fired. **LAW: the pricer's bulk is prunable only after its LAST call, never
between.**

Both are recorded because a chain that is only correct when nobody reads it is not correct.

## 18. What this hands the next arm

1. **pd6's biggest open question is closed, and it was bookkeeping.** The nine pairs whose decode
   disagreed with the overlay are exactly the nine whose plane differs between the field the
   overlay was rendered from and the field the archive carried. `render-edits` is faithful. The
   cheap structural check — compare the shipped field's plane against the overlay's field, pair by
   pair — replaces a decode-and-drop cycle that cost pd6 two full parse-backs. **Any arm that
   composes a pose leg across a size ladder must run it, because a set-price iteration re-ranks
   as well as drops.**
2. **The price-first generator still pays on a field it has already worked, but the cheapest rung
   is measurably dearer.** Rank-0 went 2.163 → 3.075 bits/token (+42 %) while ranks 1–5 moved
   under 1.5 %. The head of the price list is what a pass spends; the body is not.
3. **The expiry is LOCAL and TOP-ONLY.** Head-8 overlap with the prior's rows is median 6/8 on
   the re-rendered pairs and 8/8 on the unchanged ones, while the head-1024 pool is stable at
   947/1024 for both. An edit re-ranks the top of a pair's price list without moving the list.
4. **The real new ground is the PRIOR's seg-refused pool, not a new part of the plane.** 2,850 of
   4,291 proposals are rows pd6 refused on its per-pair seg budget and never rendered; only 498
   are new to the price list, and 307 of those sit on the 101 re-rendered pairs — 7.8× denser per
   pair than the unchanged ones. **The seg screen, not the price, is the binding gate on this
   generator.** An actuator that paid the seg debt elsewhere (psa1/psa2's pair-selective renderer
   bias is the live candidate) would roughly triple this pool's yield without touching the price.
5. **The family has a measured horizon.** The headroom shrinks 2–3 % per move because each pass
   lowers the base d_pose its successor must recover from. At move 53, 5 % recovery at 8 bits per
   token no longer clears the bar on its own. The generator makes its own successor harder by
   exactly the amount it succeeds.
6. **What is still unmeasured:** whether the 3,141 proposals this arm's seg budget refused would
   pay if the seg debt were paid elsewhere; and whether a fourth set-price iteration would move
   the ladder again (iteration 3's residual is −1.35 %, so the estimates have converged, but the
   ladder was still moving at iteration 3).

## 19. verdict_scope

**verdict_scope: n/a for the pass** — no negative verdict is drawn. Every pre-registered
falsifier was evaluated and none fired; the candidate is sealed and MAIN fires.

**verdict_scope: FORMULATION, for the corrected instrument law.** What is MEASURED and now
binding is that on THIS object, with THIS renderer, a composed overlay is valid exactly for the
pairs whose token plane in the shipped field equals the plane the overlay was rendered from —
and that pd6's nine disagreeing pairs are, as a set identity on its own retained fields, exactly
those pairs. That **SUPERSEDES** pd6's §18 formulation-scope law, which attributed the
disagreement to `render-edits` not reproducing the receiver's odd frame for every edited pair.
The renderer is not implicated by any measurement either arm has taken.

**Nothing here closes or reopens pd5's multi-token formulation.** This arm changed the FIELD, not
the run length; its admitted set is 40 single-token and 13 two-token proposals.

<!-- # FORMALIZATION_PENDING: a measurement pass and a byte-closed candidate; the score arithmetic
used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->

Own-vehicle frontier (unchanged by this arm — MAIN fires):
**S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600]** (move 53).
