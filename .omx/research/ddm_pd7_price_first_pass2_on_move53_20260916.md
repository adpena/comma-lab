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

---

## STILL OWED at the time of this commit (the arm is mid-flight)

This memo is committed early so its censuses are durable, not because the pass is finished.
Owed, in order: the eight sheet CHARGES (real 600-frame encodes) and the noise floor; the
credit wave (7 shards over all 4,291 proposals, no prefix stop); the SET price and its
iteration residual; the SIZE ladder on real closed archives; the three legs on the candidate's
own cold decode; the render-agreement census against that decode; and the seal. Every falsifier
except F1 and F2 is still open. **No candidate, no byte claim and no verdict is made here.**

<!-- # FORMALIZATION_PENDING: a measurement pass; the score arithmetic used throughout is the
registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->

Own-vehicle frontier (unchanged by this arm — MAIN fires):
**S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600]** (move 53).
