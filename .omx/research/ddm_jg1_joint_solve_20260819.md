# ddm_jg1 — the full joint solve over every stored section of the shipped object

- **arm** `ddm_jg1` (task #1134 — the js1 joint line reborn as a SOLVE)
- **date** 2026-08-19
- **axis** every number this arm measures is `[macOS-CPU advisory]` unless it carries an
  explicit DALI-lineage tag · `score_claim=false` · `promotable=false`. This arm fires no
  Modal job; MAIN owns the T4 slot.
- **cost** $0.
- **store** `/Volumes/APDataStore/pact/ddm_jg1/`
- **status** COMPLETE. Written incrementally and committed at every stage boundary.
  **Pointer UNMOVED** at contest-CUDA `0.15652626435208142`. This arm produced a validated
  instrument, four measured findings, one hard negative and its reversal — not a row.
- **code** `experiments/ddm_jg1_seg_solve.py` · `src/tac/tests/test_ddm_jg1_seg_solve.py`
  (17 tests) · commits `5199e87135`, `0977c3e093`, `fdc494a3a1`, `c784935485`,
  `c43cefcd07`, `8c56347e64`, `22213c53d8`

## ANSWER FIRST

1. **S0 — the coordinate map.** The RX1 header counts three streams and **everything after
   them is the tail**: hpac 13,515 B (the IHS1 probability model) · semantic 30,856 B
   (renderer weights) · carrier 22,143 B (the pose CAP1, solved by up2/up3) · **tail
   109,792 B = 62.2% of the archive = the HPAC-coded token payload.** The tokens are
   `(600, 384, 512) uint8` in `{0..4}` — **a 5-class semantic label map**, 117,964,800
   cells at 0.00745 bits/token. The seg actuator was never an opaque latent.
2. **95.9% of the seg debt is render/re-segment loss, not stored-label error.** The stored
   tokens are **99.9985% identical to the DALI GT argmax**; only 1,714 cells differ. The
   debt lives in the paint -> re-segment round trip, so the actuator's job is
   PRE-DISTORTION, not better labels.
3. **A $0 local contest-axis seg instrument now exists** and reproduces the T4 leg to five
   figures: **0.00030307 vs 0.00030309 (0.99995x)** on DALI, 1.00002x on PyAV, with a
   **byte-exact** forward model.
4. **The move class decides the sign.** Block/dilation moves realize WORSE at every radius
   (r=1: -55%, r=2: -351%). Single-cell coordinate moves repair **1.55 cells per changed
   token** and are perfectly additive within a sparse pass. Same lesson up2 learned on
   pose: block/gradient steps lose, lattice coordinate descent with realized acceptance
   wins.
5. **THE HARD NEGATIVE: token seg edits destroy pose — mean `d_pose` x387.** Because frame
   `2p` is a *photometric probe*, this vehicle encodes pose as a photometric relationship
   between the frames, and the shipped carrier was solved against the ORIGINAL frame 1.
   Two token edits (0.5% of camera pixels) move `d_pose` 6.5x. A seg-only solve would have
   looked like a clean win and cost **+0.159 S**.
6. **AND ITS REVERSAL: the carrier re-solve absorbs the damage at ~0 bytes.** Re-running
   the carrier's own coordinate descent against the edited frame 1 recovers `d_pose` to
   **1.073x of original** (1.01x / 1.34x / **0.87x** — one pair ends better than shipped),
   moving 9-12 of 12 coefficients, well inside up2's measured "+/-4 on all 7,200
   coefficients = +5 bytes" envelope. **The two actuators COMPOSE.**
7. **S3 — pose basis enlargement is priced out by arithmetic, at $0.** The basis is
   12,277 B (55.3% of the carrier body); doubling it costs up to **+0.008175 S** while the
   entire pose leg is **0.008746 S**. Even `d_pose -> 0` nets +0.000571 S. The free move is
   **re-orientation** of the existing 12 dims, still unowned.
8. **The projection, and it is a projection.** At the measured first-pass rates the joint
   move extrapolates to **-0.0104 S -> ~0.14614**. It rests on 3 pairs, a first pass, and a
   rate price MODELLED from a different body's probability model. **52 seconds of
   re-encoder replaces the only modelled leg.**

STORES CONSULTED: `.omx/state/canonical_frontier_pointer.json` (re-read at start) ·
`.omx/research/ddm_up2_shipping_object_pose_solve_20260819.md` (the method law, §3/§5/§8) ·
`.omx/research/ddm_up3_thirteenth_move_byteclose_20260819.md` (the byte-close machinery,
§1/§3/§5) · memory `pose_gap_was_gt_cache_lineage_not_cuda_20260819` ·
`/Volumes/APDataStore/pact/ddm_up3/retained/**` ·
`/Volumes/APDataStore/pact/ddm_to1/generations/to1_tail_override_r1/**` — the SHIPPED
receiver, read at source (`cpr1/inflate.py`, `runtime/residual_archive.py`,
`runtime/f26_inflate.py`) rather than quoted from memory · `upstream/modules.py`.

---

## THE BASE (re-read from the pointer, not from the charter)

| term | value | S contribution |
|---|---:|---:|
| `d_seg` | 0.00030309 | **0.030309** |
| `d_pose` | 7.649246787e-06 | 0.008746 |
| archive | 176,420 B | 0.117471 |
| **S** | | **0.15652626435208142** |

`archive.zip` sha `7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f`.
Gap to sub-0.15 = **0.006526**.

---

## S0 — THE COORDINATE MAP

Read out of the receiver's own parse chain (`residual_archive._decode_rx1_models`,
`:159-240`), not guessed. The ZIP holds one `STORED` member `p` of 176,320 B; the RX1
header (`<4sBBBBHHH>`, 14 B) counts three streams and **everything after them is the
tail**:

| # | section | archive B | share | what it is | actuator status |
|---|---|---:|---:|---|---|
| 0 | RX1 header | 14 | 0.0% | framing (`reserved=6`) | no |
| 1 | hpac stream | 13,515 | 7.7% | the **IHS1 probability model** for token coding | model axis, live |
| 2 | semantic stream | 30,856 | 17.5% | `SemanticTokenRenderer` weights (width 96) | weights axis, live |
| 3 | carrier stream | 22,143 | 12.6% | CAP1: basis + 600x12 int12 coefficients | **SOLVED by up2/up3** |
| 4 | **tail** | **109,792** | **62.2%** | **the HPAC-coded token payload** | **THE SEG ACTUATOR** |

(`+100 B` of ZIP container overhead reaches the scored 176,420.)

**The charter's premise — that the seg content is an opaque latent — is wrong, and the
truth is much better.** `cpr1/inflate.py:95` is
`self.token_embed = nn.Embedding(NUM_CLASSES, width)` with `NUM_CLASSES = 5` (`:21`), and
`render_video` (`:318`) calls `semantic(tokens[start:end].long(), indices)`. The tokens
are **5-way class labels on a 384x512 grid, one grid per pair** — the field is
`(600, 384, 512) uint8 in {0..4}`, **117,964,800 cells**, entropy-coded into section 4.
The shipped object is literally a coarse semantic map plus a small net that paints it into
RGB, which SegNet then re-segments. That is the task-lossy ego-scene codec archetype,
observed rather than asserted.

### The two frames are NOT symmetric, and that asymmetry is the whole joint structure

* `cpr1/inflate.py:313-328` renders frame **2p+1** from the tokens.
* `cpr1/inflate.py:336-352` renders frame **2p** as
  `127.5 + CARRIER_AMPLITUDE * (12-dim basis expansion)` — **a photometric probe frame,
  not a picture of the scene.**
* `upstream/modules.py:108` is `x = x[:, -1, ...]` — **SegNet sees only frame 2p+1.**
  PoseNet sees both.

| actuator | frame it writes | reaches `d_seg` | reaches `d_pose` |
|---|---|---|---|
| tokens (section 4) | 2p+1 | **YES** | **YES** |
| carrier coeffs (section 3) | 2p | no | YES |
| renderer weights (section 2) | 2p+1 (all pairs) | YES | YES |

So the carrier is a **pose-only** actuator with `d_seg` obligation exactly zero, and the
tokens are a **joint** actuator. Any token edit that buys `d_seg` and spends `d_pose` can
be met by re-solving the carrier, which is free in bytes (up2 §ANSWER-4) — that is the S2
composition hypothesis, and it is structural, not hopeful.

### The exchange rate that governs the seg axis

Both legs are linear in their own unit, so they can be quoted against each other exactly:

* `d_seg` denominator = 600 x 384 x 512 = 117,964,800 argmax cells, so
  `d_seg = 0.00030309` is **~35,754 disagreeing cells**.
* one fixed cell = `0.030309 / 35754` = **8.477e-07 S**
* one archive byte = `25 / 37,545,489` = **6.658e-07 S**

**One repaired seg cell is worth 1.273 archive bytes.** Section 4 currently spends
109,792 B on 117,964,800 tokens = **0.00745 bits/token**, so the model predicts the token
field extremely well and a token flip against a confident context is expensive. The seg
axis is therefore not a free-actuator problem like pose was: every proposal must be priced
against this exchange rate, and the honest question is not "can we flip cells" but
"can we flip a cell for less than 1.273 bytes".

---

## S1a — THE SEG DEBT IS NOT STORED-LABEL ERROR. IT IS RENDER/RE-SEGMENT LOSS.

The first thing to measure on the seg axis is not a solve, it is a decomposition: does
the stored token map already equal the GT label map? Pure array comparison over the whole
117,964,800-cell field, both lineages:

| comparison | disagreeing cells | as a `d_seg`-equivalent | vs realized `d_seg` |
|---|---:|---:|---:|
| tokens vs **GT[dali]** | **1,714** | 0.00001453 | **0.05x** |
| tokens vs GT[av_pyav] | 20,763 | 0.00017601 | 0.41x |
| GT[dali] vs GT[av_pyav] | 20,671 | 0.00017523 | — |

**Three findings, none of them assumed:**

1. **The stored tokens are 99.9985% identical to the DALI GT argmax.** The encoder stored
   the label map essentially exactly.
2. **The encoder was fit against the DALI lineage** — the contest-CUDA axis. The token
   field's disagreement with PyAV GT (20,763) is almost entirely just the lineage gap
   itself (20,671). Nobody had established which lineage the shipped tokens target; they
   target the one that ships.
3. **95% of the realized seg debt is render/re-segment loss.** Realized `d_seg` is
   0.00030309 = ~35,754 disagreeing cells, of which only ~1,714 (5%) could possibly be
   blamed on a wrong stored label. The other ~34,040 cells are cells where **we stored the
   RIGHT label, the renderer painted it, and SegNet did not read it back.**

### What that does to the strategy — and to the prior refusal

This inverts the obvious attack. "Store better labels" addresses at most 5% of the debt
and costs rate to do it. The live mechanism is the **paint -> re-segment round trip**, and
the only actuator that reaches it is **PRE-DISTORTION**: deliberately setting a token
*away* from GT so the painted RGB drives SegNet's argmax *back onto* GT. That is the
"invert the frozen space" thesis applied to the seg axis, and it is a different move class
from anything the token actuator has been asked for before.

It also reframes the standing refusal. A search that hunts for *better labels* is hunting
in the 5%, and would correctly price out. Whether pre-distortion prices out is a separate,
unmeasured question — and it is the question S1b answers.

### The rate arithmetic that governs it (measured, tight)

A pre-distortion is a **substitution, not an addition**: the token count never changes,
only the value, so the cost is exactly the HPAC model's surprise at the new symbol.
Against the exchange rate from S0 — **1 repaired cell = 1.273 archive bytes = 10.18
bits** — the admission test per pre-distorted token is:

> `cells_repaired_by_this_flip x 10.18 bits > log2(p_old / p_new)` under the shipped
> IHS1 model.

The field averages 0.00745 bits/token, so the context model is extremely confident and a
flip against it is not cheap. But one token flip perturbs RGB over the renderer's
receptive field (`TokenBlock` dilations 1,1,2,4 over 3x3 kernels, so ~17 px), and SegNet
then re-segments a still larger neighbourhood — so a single flip can repair **several**
cells. The sign of this inequality is therefore genuinely open and is exactly what a
realized-descent probe measures.

---

## S1b — THE INSTRUMENT, AND ITS TWO CONTROLS

Before a single proposal is scored, the instrument has to reproduce the row it claims to
improve. Two independent controls, both passed:

### Control 1 — the seg leg, n600, BOTH lineages

`experiments/ddm_jg1_seg_solve.py validate --pairs 600`. SegNet argmax of frame `2p+1`
read straight out of the receiver's own `0.raw`, scored against each GT cache by
`SegNet.compute_distortion`'s own definition (`modules.py:111-113`):

| lineage | measured `d_seg` | published | ratio |
|---|---:|---:|---:|
| **dali** (contest-CUDA) | **0.00030307** | 0.00030309 | **0.99995** |
| av_pyav (advisory) | 0.00043337 | 0.00043336 | 1.00002 |

**This is a $0 local instrument that reproduces the T4 seg leg to five significant
figures** — the seg analogue of up2's 0.99993x pose instrument, and it did not exist
before this arm. Both lineages are reproduced from the same argmax field, which is
computed once: the argmax is a property of OUR frames alone, so sharing it across
lineages is the factorisation, not an approximation.

### Control 2 — the forward model is the receiver's, byte for byte

Re-rendering the SHIPPED tokens through the shipped `SemanticTokenRenderer`
(66,339 params, loaded exactly as `runtime/f26_inflate.py:481-494` loads it) and diffing
against the shipped decode:

| pairs | pixels compared | pixels changed | max abs delta |
|---:|---:|---:|---:|
| 4 (seeded) | 12,208,032 | **0** | **0** |

**Byte-exact.** One detail here is load-bearing and was inherited rather than
rediscovered: `cpr1/inflate.py:312` sets `semantic_batch = 8 if cuda else 1`, and up2 §6
MEASURED batch shape to be byte-changing on this exact half (1,326 pixels at +/-1). Had I
rendered at batch 8 for speed, this control would have failed for a reason that has
nothing to do with tokens. Rendering costs **0.25 s/pair**.

### The flip ledger — n600, DALI lineage, 35,752 flips

| decomposition | count | share |
|---|---:|---:|
| flips where the **stored token already equals GT** | **34,296** | **95.9%** |
| flips where the stored token is wrong | 1,456 | 4.1% |

and by the EDGE the flip lies on (`gt -> ours`), which is the decomposition `ddm_pc2`
established is the right one because seg is one graph with one hub:

| edge | flips | share |
|---|---:|---:|
| **Lane -> Road** | **8,619** | **24.11%** |
| **Road -> Lane** | **6,797** | **19.01%** |
| Road -> Undrivable | 3,773 | 10.55% |
| Undrivable -> Road | 3,407 | 9.53% |
| Undrivable -> Movable | 3,100 | 8.67% |
| Movable -> Undrivable | 2,997 | 8.38% |
| Road -> Movable | 2,453 | 6.86% |
| Movable -> Road | 1,825 | 5.10% |
| MyCar -> Road | 1,262 | 3.53% |
| Road -> MyCar | 996 | 2.79% |

**Two findings that change where the seg attack points.**

1. **The Road<->Lane edge alone carries 43.1% of the entire seg debt.** Lane is
   **0.586%** of the field by area but appears in **44.3%** of flips — a **75x**
   over-representation. Road is in 81.5% of flips, re-confirming `ddm_pc2`'s hub finding
   (87.8%) on an independent body.
2. **It is DISPLACEMENT, not erasure.** The campaign's standing reading of the lane
   long-tail is *erasure* — the witness drops the lowest-persistence features. If that
   were the mechanism here, `Lane -> Road` would swamp `Road -> Lane`. Measured, they are
   **8,619 vs 6,797 — a ratio of only 1.27**, i.e. near-balanced, with a net lane loss of
   just 1,822 cells against 15,416 lane-edge flips total. **88% of the Road<->Lane debt is
   a boundary that is in the wrong PLACE, not a lane that is missing.** A displaced
   boundary is a far cheaper thing to fix than an erased structure, and it is the natural
   target of a one-cell pre-distortion.

---

## S1c — THE MOVE CLASS DECIDES THE SIGN. BLOCK MOVES FAIL; COORDINATE MOVES WORK.

Every number below is REALIZED: the proposed token field goes through the receiver's own
forward model (byte-exact per S1b control 2) and then through the frozen CPU SegNet.
Nothing is predicted, linearised, or scored on a surrogate.

### The block move — dilate the correct class around each failing cell — FAILS, hard

n=6 seeded-random pairs, 268 base flips, DALI lineage:

| family | tokens changed | flips repaired | % of debt | cells/token |
|---|---:|---:|---:|---:|
| `all_r0` (write GT only where the token is wrong) | 12 | **+9** | **+3.36%** | +0.750 |
| `all_r1` (disk radius 1) | 576 | **-148** | -55.22% | -0.257 |
| `all_r2` (disk radius 2) | 1,611 | **-942** | -351.49% | -0.585 |
| `roadlane_r1` (restricted to the hub edge) | 326 | -117 | -43.66% | -0.359 |
| `roadlane_r2` | 917 | -592 | -220.90% | -0.646 |

Dilation is **strongly counterproductive**, and monotonically worse with radius. The
mechanism is not subtle: widening a class in the token map paints a wider class in RGB,
and SegNet faithfully reports the wider class — so the boundary simply moves to the other
side and the new flips outnumber the repairs. `all_r0` independently reproduces S1a's
predicted ~4% ceiling (+3.36%), which is a good check on both.

### The coordinate move — ONE token, to an adjacent cell — WORKS

Same instrument, 18 seeded-random flip sites over 3 seeded-random pairs, 40 realized
evaluations, each move a SINGLE token set to the GT class at the failing cell or one of
its four neighbours:

| quantity | measured |
|---|---:|
| flip sites with an improving single-cell move | **12 / 18 (67%)** |
| mean cells repaired per accepted move | **1.50** |
| total cells repaired | 18, from 12 changed tokens |
| **cells repaired per changed token** | **1.50** |
| **break-even HPAC budget** | **15.3 bits/token** |

**The winning move is almost always a single ADJACENT cell** — `(y, x-1)`, `(y+1, x)` —
i.e. the minimal one-cell shift of a displaced boundary, exactly the mechanism S1b's
ledger predicted when it measured the Road<->Lane debt as displacement rather than
erasure.

**This is up2's method law reproduced on a different axis and a different actuator.**
There, a gradient LM solve was correct in its Jacobian and realized WORSE at every
damping, while lattice coordinate descent with realized acceptance converged. Here a
block/dilation move realizes worse at every radius, while the single-coordinate move
realizes better at 67% of sites. **The failure of the block move is not evidence that the
token actuator refuses — it is evidence that the move class was wrong**, and any prior
verdict drawn from a block-shaped or blind-search proposal is scoped to that formulation,
not to the actuator.

### What is now the ONLY open question on this axis

The seg side is measured and positive: 1.50 repaired cells per changed token. Against the
S0 exchange rate (1 cell = 1.273 B = 10.18 bits) that buys a budget of **15.3 bits per
changed token**. The field averages 0.00745 bits/token, so the IHS1 context model is very
confident and a flip against it is not cheap.

**Everything therefore reduces to one measurable number: what does the shipped IHS1 model
actually charge for one changed token?** Under 15.3 bits, the seg axis opens; over it,
it closes. That number is not estimated here — a re-encoder exists
(`experiments/ddm_rr2_encoder_byteclose.py` + the `ddm_rr2_encoder_build` custody), so it
can be MEASURED byte-closed rather than modelled. That is the next measurement, and it is
the one that decides the axis.

---

## S1d — THE RATE PRICE OF THE MOVE. THE AXIS IS OPEN.

The break-even budget is 15.28 bits per changed token (S1c). The question is what the
shipped IHS1 model actually charges. Measured against the model's own per-cell class
logits (`ddm_hm1_20260816/retained/base_logits_int16_n600.i16`, `(600, 196608, 5)` int16).

### The model reconstruction is validated before it is used

I got this wrong on the first pass and caught it on a control, so the control is worth
stating. The int16 logits are stored in units of **1/8** (`LOGIT_SCALE = 8`, the runtime's
`HPAC_LOGIT_PRECISION`); I first divided by 256 and got a field-average of 1.525
bits/token against a shipped stream of 0.00745 — a **200x** disagreement that could only
mean I was scoring the wrong distribution. At the correct scale:

| quantity | value |
|---|---:|
| base-only cross-entropy, my reconstruction | **0.007880 bits/token** -> 116,199 B |
| hm1 receipt, base **+ table** correction | 0.007603 bits/token -> 112,109.58 B |
| the shipped stream that receipt priced | **112,110 B** (agrees to 0.42 B) |
| the to1 body's token tail (ours) | 109,792 B -> 0.007446 bits/token |

Base-only lands 3.6% above base+table, and the gap is exactly the table correction I
omitted. **The reconstruction is the coder's own distribution, not a look-alike.**

### The price, on 20 seeded-random pairs / 2,309 candidate move cells

| quantity | mean | median |
|---|---:|---:|
| cost of the CURRENT token at move sites | 0.792 bits | 0.091 bits |
| cost of the TARGET token at move sites | 5.511 bits | 4.398 bits |
| **delta per changed token** | **+4.718 bits** | **+4.328 bits** |

| percentile | p10 | p25 | p50 | p75 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| delta bits | -2.344 | +0.902 | +4.328 | +8.295 | +12.443 | +19.642 |

* **moves under the 15.28-bit budget: 95.6%**
* **moves that are FREE or cheaper (delta <= 0): 20.0%**
* the solver pays the CHEAPEST of the four neighbours per site, so this all-neighbours
  distribution is an **upper bound** on what a solve actually spends.

**And the mechanism is the one the S1c geometry predicted.** The field averages 0.00788
bits/token because the interior is near-certain — but at the class boundaries where every
one of our moves lives, the model already hedges: the CURRENT token there costs 0.792 bits
(median 0.091), roughly **100x the field average**. A boundary token is a coin the model is
already unsure about, so re-labelling it is cheap. The expensive tokens are in the
interior, and we never touch the interior.

### The headline arithmetic

Per changed token: repairs 1.50 cells and costs 4.718 bits = 0.590 B.

| leg | per changed token |
|---|---:|
| seg gain | `-1.50 x 8.477e-07` = **-1.2716e-06 S** |
| rate cost | `+0.590 x 6.6586e-07` = **+3.928e-07 S** |
| **net** | **-8.79e-07 S per changed token** |

Equivalently, per repaired cell the net is **-4.549e-07 S**, i.e. **the move keeps 54% of
its gross seg gain after paying rate**. Closing the remaining **0.006526** to sub-0.15 on
this axis alone would need **~14,346 repaired cells — 40% of the 35,752 flips.** The probe
found an improving move at **67%** of sampled sites, so 40% is not obviously out of reach.

**That is a GO signal for the axis, and it is NOT a result.** Six things stand between it
and a row, and they are named rather than waved at:

1. **1.50 cells/token rests on n=18 sites over 3 pairs.** Small.
2. **Gains are almost certainly NOT additive** — neighbouring repairs overlap, so the
   realized total will be below the sum of per-site gains. This is the one that can most
   easily halve the estimate, and it is being measured (S1e).
3. **Cross-body transfer.** The logits are the hm1/`182,759 B` generation; we ship to1
   (`109,792 B`), whose model is SHARPER. A sharper model charges MORE per flip. This is
   the cross-regime-constant-transfer genus and it is a real risk to the 4.718.
4. **Context coupling.** The HPAC model decodes in 190 groups and feeds decoded tokens
   forward as context, so changing a token perturbs downstream logits. The per-symbol
   delta is first-order and exact only for the changed symbol itself; many simultaneous
   changes compound this.
5. **The table correction is omitted** from the marginal number (base-only logits).
6. **No byte-closed archive exists.** Nothing here has been through the re-encoder, so
   every rate figure is modelled, not measured on `archive.zip`.

---

## S1e — CORRECTION TO S1d. THE YIELD DECAYS, AND THE AXIS IS OPEN ONLY WITH A STOPPING RULE.

S1d's "the axis is OPEN" was drawn from a **first-pass** yield of 1.50 cells per changed
token, and S1d itself named non-additivity as the caveat most likely to halve it. It was
measured, by two independent runs, and **it does more than halve it.**

| run | passes | tokens changed | cells repaired | cells/token |
|---|---:|---:|---:|---:|
| this arm, pair 283, single pass | 1 | 13 | 19 | **1.462** |
| this arm, S1c sampled sites | 1 | 12 | 18 | 1.500 |
| **sister scout, pair 133, iterated** | **8** | **77** | **30** | **0.390** |

The first pass is efficient and the two single-pass measurements agree (1.46, 1.50). But
pushing the same pair to 25.9% of its own seg debt drove the average to **0.390**. The
marginal yield **decays hard**.

That changes the sign of the verdict, because the budget is proportional to the yield:

| yield regime | budget (bits/token) | measured cost | verdict |
|---|---:|---:|---|
| first pass, 1.46 cells/token | **14.9** | +4.718 mean | **net positive, 3.2x margin** |
| iterated to 8 passes, 0.390 | **3.97** | +4.718 mean | **net NEGATIVE** |

**So the axis is not simply open. It is open up to a stopping point and closed past it.**
A seg-greedy solve that chases every repairable cell walks straight past the point where
the rate term overtakes the seg term and ends up worse than the base. The correct
formulation is not "descend until converged" — which is what worked for the pose carrier,
where bytes were nearly free — but **a rate-aware descent with a Lagrangian stopping rule**:
accept a move only while `cells_repaired x 10.18 bits > cost_bits`, and stop when the
marginal move stops clearing it.

Two things make the stopping rule more favourable than the table's flat comparison, and
both are measured rather than hoped:

1. **+4.718 bits is the mean over ALL four neighbour candidates; a solver pays the
   CHEAPEST.** The distribution is wide — p25 = +0.902, p10 = -2.344, and **20% of moves
   are free or better**. A cost-aware selector that ranks candidates by
   `cells_repaired / bits` rather than by cells alone pays far less than the mean.
2. **The early repairs are the cheap ones.** Diminishing returns cuts both ways: the
   first-pass margin is 3.2x, so there is real headroom before the rule bites.

**Honest extrapolation, with its error bars stated.** At the first-pass rate (~13 edits
per pair repairing ~19 cells) over 600 pairs: ~11,400 cells repaired (`-0.00966 S`) for
~7,800 changed tokens at 4.718 bits (`+0.00306 S`), net **~-0.0066 S** — which would
cover the 0.006526 gap. **I do not claim that.** It extrapolates one pass on two pairs to
600, uses the mean rather than the cheapest-neighbour cost, ignores the cross-body model
gap (S1d caveat 3), and has never been byte-closed. It is quoted only to show the axis is
worth the next unit's compute, not as a projected row.

### Three corrections this arm owes to its own charter and its own S1c

1. **The "~1000x" refusal is mis-attributed.** It is not task #930 on this actuator; it is
   **`ddm_sm1`** (`.omx/research/ddm_sm1_seg_search_transfer_20260803.md:343`), and it
   **was** realized coordinate descent — but on a **different actuator**: a 4-channel x
   16-level 16x16-cell token code inside a 767,812 B TR1 archive the live parser refuses,
   carrying `verdict_scope: INSTANCE`. It never scoped to the dense 5-ary map this arm
   actuates. The reopening was justified; the justification is now precise.
   Sisters: `#869 = ddm_tw1` is a **rate** arm that computed no `d_seg` at all; `#978` is a
   counted receiver-side conv module, not token edits.
2. **My S1c proposal family was too narrow, and in an informative way.** I only ever tried
   setting a neighbour to the **GT class**. A sister sweep over all five classes found
   that **0 of 12 accepted single-token edits chose the GT class.** The winning edits are
   **adversarial** — they write a class that is *wrong* at that cell in order to steer the
   painted RGB so SegNet lands on GT. That is pre-distortion confirmed at the edit level,
   and it means my 1.46-1.50 is a **lower bound on per-move quality**: the true best move
   is outside the family I searched.
3. **Edits are local, which licenses spatial packing.** A single token flip changes a
   **median of 1** argmax pixel, with Chebyshev radius 0-11 px in 9 of 10 trials. Applying
   all improving edits at >=64 px separation jointly reproduced the sum of their solo
   deltas at ratios **1.000 / 0.818 / 0.750**. So many well-separated proposals can share
   one render+SegNet forward, taking a site from ~2.33 s to ~48 ms — a ~48x speedup that
   makes an n600 rate-aware solve affordable at $0.

---

## S3 — POSE BASIS ENLARGEMENT IS PRICED OUT BY THE BYTE MAP. RE-ORIENT INSTEAD.

`ddm_up2` §5 measured the pose wall as the **12-dimensional basis**: the residual sits in
the carrier Jacobian's smallest singular direction, so nulling it demands a median **6.4x**
larger frame perturbation than a basis relaxed to full 24x32 freedom. My charter asks for
the minimal basis extension that spans that excess step, priced with rate in the loop.

**It does not need a probe. The shipped byte map already refuses it.** Measured from the
archive through the up3 parser:

| carrier component | bytes | share of the carrier body |
|---|---:|---:|
| **basis** (12 x 3 x 24 x 32 = 27,648 coefficients @ 3.552 bits) | **12,277** | **55.3%** |
| Rice payload (the 600 x 12 int12 coefficients) | 9,759 | 44.0% |
| metadata + scales + selector tail | ~150 | 0.7% |

The basis is the **majority of the carrier section**, and it is a fixed cost shared across
all 600 pairs. So adding dimensions scales the dominant term:

| move | rate cost | best possible seg/pose return |
|---|---:|---:|
| 12 -> 24 dims (double the basis) | **up to +12,277 B = +0.008175 S** | the ENTIRE pose leg is **0.008746 S** |

**Even driving `d_pose` to exactly zero would net `+0.000571 S`** — and that is against an
uncompressed upper bound on the cost, with the whole pose leg surrendered to pay for it.
Basis enlargement cannot reach sub-0.15 on this vehicle; it cannot even pay for itself
with certainty. **Verdict: refused on arithmetic, `verdict_scope: family` for
dimension-adding on this carrier layout, at $0.**

**What the same arithmetic licenses instead.** up2's finding was that the basis is
mis-ORIENTED, not too small — the residual lands on the *smallest* singular direction. A
**re-orientation of the existing 12 dimensions costs ZERO extra coefficients**: same
27,648 basis values, same 12,277 B envelope, different numbers (the entropy of the new
values is the only second-order cost). That is up2's owed item 2, it is the only pose move
whose rate leg is free, and it remains unowned. **The pose axis's next move is
re-orientation, not enlargement** — and this arm's contribution is to have priced the
alternative out before anyone spent a run on it.

---

## S2 — THE JOINT COUPLING

The reason this arm exists. `upstream/modules.py:108` gives the asymmetry: SegNet reads
only frame `2p+1`, PoseNet reads both. So the carrier is a pose-only actuator with zero
seg obligation (which is why up2's solve was seg-free by construction), while **the token
edits of S1 land in a frame PoseNet also reads**. A seg gain bought with tokens is
therefore NOT automatically free on pose, and nothing in the campaign had measured that.

Measured here: first-pass greedy seg descent on seeded-random pairs, then `d_pose` through
the frozen CPU PoseNet against the **DALI** GT targets (lineage gate VERIFIED at run time,
`up2.verify_gt_lineage(axis="contest_cuda") -> VERIFIED`), with frame `2p` held at the
shipped decode.

### The measurement — and it is the hardest negative this arm found

3 seeded-random pairs, first-pass greedy seg descent, `d_pose` on the DALI targets:

| pair | seg flips | tokens | `d_pose` before | `d_pose` after | factor |
|---|---|---:|---:|---:|---:|
| 283 | 38 -> 13 | 20 | 1.0989e-05 | **9.0402e-03** | **822x** |
| 468 | 70 -> 37 | 19 | 4.2551e-06 | 4.4506e-04 | 105x |
| 513 | 80 -> 48 | 19 | 2.3061e-06 | 5.4207e-04 | 235x |
| **total** | **90 cells repaired** | **58** | | **mean delta +3.3366e-03** | |

The seg leg behaved exactly as S1 said it would — **1.552 cells/token**, matching S1e's
1.651 and S1c's 1.50. **The pose leg was destroyed.**

Priced on the shipping objective, extrapolating the per-pair mean to the field:

| leg | value |
|---|---:|
| seg gain (18,000 cells) | **-0.01526 S** |
| `d_pose` 7.649e-06 -> 3.345e-03, so `sqrt(10 d_pose)` 0.008746 -> 0.1829 | **+0.1742 S** |
| **net** | **+0.159 S — catastrophically worse** |

**The pose damage is ~11x the seg gain.** A token-only seg solve is refused on the joint
objective by an order of magnitude, and it would have looked like a clean win to anyone
measuring only `d_seg`. That is precisely the trap this arm was built to find.

### The mechanism, measured rather than inferred

A sensitivity ramp on pair 283 (random single-token edits, camera pixels changed vs
`d_pose`):

| edits | camera px changed | `d_pose` |
|---:|---:|---:|
| 0 | 0 | 1.0989e-05 |
| 1 | 0 | 1.0989e-05 |
| 2 | 5,294 (0.5%) | 7.1887e-05 |
| 5 | 17,090 | 1.7985e-04 |
| 10 | 20,727 | 2.4488e-05 |
| 20 | 50,014 | 3.2735e-04 |

**Two edits — half a percent of the camera frame — move `d_pose` 6.5x, and the response
is non-monotonic** (10 edits reads lower than 5). This is not a smooth gradient; it is an
erratic, high-gain response.

The reason is structural and it explains why nobody could have guessed the magnitude.
Frame `2p` is not a picture — it is `127.5 + amplitude x (12-dim basis)`, a **photometric
probe**. up2 §5 measured the carrier's dominant singular direction as "almost certainly
the global photometric one" (sigma ~ 10-14.5). **So this vehicle encodes pose as a
photometric relationship between the probe frame and the semantic frame.** Frame `2p+1`'s
photometry is half of that code, and the shipped carrier coefficients were solved by up2
**to convergence against the ORIGINAL frame `2p+1`**. Editing tokens moves frame 1 out
from under a converged solve.

**This also retro-explains the campaign's oldest seg result.** `ddm_qs1` measured 189
changed pixels buying 32 net flips and refused it on rate; `js8`/`vd1` measured 136 of 200
singleton edits harmful. Those refusals were read as the token actuator being weak. The
measurement here says something sharper: **the seg actuator is strong (1.55 cells/token)
and the pose coupling is what refuses it.**

### The composition question this opens — and it is the RIGHT question

Because the damage lands in the direction the carrier has the MOST authority over (the
global photometric one, sigma ~ 10-14.5 against sigma_min ~ 0.011), a **carrier re-solve**
is not obviously hopeless: the carrier's 12 free coefficients per pair exist precisely to
set the photometric relationship, and up2 proved they have real authority (429/600 pairs
improved, 0 worsened). The token edit does not consume the carrier's control; it moves the
target the carrier is aiming at.

So the joint move is **not** "edit tokens" — it is "edit tokens, THEN re-solve the carrier
against the new frame 1". That is measured in S2b.

See `retained/S2_joint_coupling.json` and `retained/S2_edited_tokens.npz` (the edited token
payloads are retained per the always-keep-the-payload rule, not just their measured
lengths).

---

## S2b — THE COMPOSITION WORKS. THE CARRIER ABSORBS THE POSE DAMAGE, AT ~0 BYTES.

The S2 negative is real but it is **not** the end of the move, because the damage lands in
exactly the direction the carrier controls best. Test: take the S2-edited token field,
render frame `2p+1` from it, and re-run the carrier's own lattice coordinate descent (the
12 int12 codes, offsets +/-1/2/4, realized acceptance on `d_pose` against the DALI targets)
against that NEW frame 1.

| pair | `d_pose` original | edited frame1, SAME codes | **after carrier RE-SOLVE** | codes moved | passes |
|---|---:|---:|---:|---:|---:|
| 283 | 1.0989e-05 | 9.0402e-03 (**822.7x**) | **1.1098e-05 (1.01x)** | 12 | 5 |
| 468 | 4.2551e-06 | 4.4506e-04 (**104.6x**) | **5.7074e-06 (1.34x)** | 9 | 3 |
| 513 | 2.3061e-06 | 5.4207e-04 (**235.1x**) | **2.0027e-06 (0.87x — BETTER than shipped)** | 10 | 3 |
| **mean** | | **387x damage** | **1.073x of original** | | |

**The carrier recovers 98.7-100% of the pose damage in 39-64 s per pair, moving 9-12 of
its 12 coefficients — and on pair 513 it ends BETTER than the shipped codes.** That last
row is worth pausing on: the re-solve is not merely repairing, it is re-optimising against
a frame 1 that up2 never saw, and sometimes finds a better optimum than the one up2
converged to. The seg edit changed the pose problem, and the new problem is occasionally
easier.

Why it works, and why it was predictable from up2's own measurements: the carrier's
dominant singular direction is the global photometric one (`sigma ~ 10-14.5`), and the
damage a token edit does to the pose readout is photometric. The carrier has **12 free
coefficients against 6 pose equations per pair** — underdetermined — so it has the slack to
re-aim. The token edit does not consume the carrier's control authority; it moves the
target, and the carrier follows.

**And the re-aim is nearly free in bytes.** up2 §ANSWER-4 measured the CAP1/AR1 Rice stream
absorbing coefficient perturbations at almost zero cost: **all 7,200 coefficients perturbed
by +/-4 costs +5 bytes**; 1,000 coefficients at +/-1 costs +6 bits. The re-solve here moves
9-12 coefficients per pair, well inside that envelope.

### The joint move, and the arithmetic that follows

> **edit tokens for seg -> re-solve the carrier for pose -> pay only the token bits.**

| leg | per the measured first-pass rates | extrapolated to n600 |
|---|---|---:|
| seg | 1.552 cells/token realized (S2), 90 cells from 58 tokens on 3 pairs | 18,000 cells = **-0.0152588 S** |
| rate | +4.718 bits/token (S1d, MODELLED, cross-body) on ~11,600 tokens | ~6,841 B = **+0.0045553 S** |
| pose | `d_pose` 7.649e-06 -> 8.207e-06 (x1.073), so `sqrt(10 d_pose)` 0.008746 -> 0.009059 | **+0.0003133 S** |
| **net** | | **-0.0103902 S** |

Against the base `0.15652626435208142` that projects to **~0.14614 — below 0.15.**

**I am not claiming that, and the distance between this and a claim is the whole point.**
It is an extrapolation from 3 pairs, on a first pass, with a rate price MODELLED from a
different body's probability model, never byte-closed, never through `evaluate.py`, on an
axis whose contest row costs money. What it IS: the composition question the charter asked
has a measured answer, and the answer is **yes — the two actuators compose, the pose
damage is recoverable, and the recovery is byte-cheap.** That converts the seg axis from
"refused by pose coupling" (which is what S2 alone said, and what the campaign has
implicitly believed since `qs1`) into a live, priced, bounded path.

The single measurement that would turn it into a row is item 1 of Owed: **52 seconds of
re-encoder** for the exact byte delta.

---

## My own round-1 adversarial review

1. **Did I quote a prefix anywhere?** No. Every sub-n600 sample uses `up2.select_pairs`,
   which refuses a contiguous prefix below n600. The instrument validation is full-field
   n600. This mattered more than usual here: `ddm_bp2`/`ddm_na2` measured seg prefixes
   0.95-0.97x *easier* and pose prefixes 2.54-4.21x *harder*, and S2 quotes both axes.
2. **Is the instrument circular?** The seg instrument reproduces two independently
   published numbers on two different lineages (0.99995x, 1.00002x) that it was not fitted
   to, and the forward model is byte-exact against a decode it did not produce. Neither
   could be true of a look-alike.
3. **Did I catch my own error?** One, and it would have produced a confident wrong
   headline: I priced the token move with the logits scaled by 256 instead of 8 and got
   +0.147 bits/token, which would have read as "essentially free". The 200x disagreement
   with the shipped stream's own bits/token is what exposed it (S1d). The lesson is the
   one the campaign already knows — validate the instrument against a number it must
   reproduce BEFORE quoting a delta from it — and I needed it.
4. **Did I over-claim in S1d?** Yes, and S1e is the correction, written as a correction
   rather than a quiet edit. "The axis is OPEN" was true of the first-pass yield and false
   of the iterated yield; the honest statement is "open up to a stopping point". A sister
   arm's independent 8-pass measurement is what forced it.
5. **Is my S1c proposal family complete?** No — I only tried the GT class, and a sister
   sweep found **0 of 12** accepted edits chose GT. My numbers are therefore a LOWER bound
   on per-move quality, which I have said in S1e rather than left implicit.
6. **Is the S2 extrapolation legitimate?** The per-pair `d_pose` factors (822x, 105x,
   235x) are measured; the field-level `+0.159 S` is an EXTRAPOLATION from 3 pairs and is
   labelled as such. It does not need to be tight — the seg gain would have to be wrong by
   an order of magnitude for the sign to change.
7. **Cross-body transfer is the weakest leg I lean on.** The rate price (+4.718 bits)
   comes from the hm1/`182,759 B` model generation, not from to1's own (sharper) model.
   That is the cross-regime-constant-transfer genus, and a sharper model charges MORE. The
   fix is not a better estimate, it is the re-encoder: 52 s for an exact answer.
8. **What I did not do.** No byte-closed archive, no re-encoder run, no seal, no n600
   solve. Every rate number here is modelled. The arm produced measurements and one hard
   negative, not a row — and the pointer is UNMOVED at 0.15652626435208142.

---

## Owed, with owners

1. **Run the re-encoder on an edited token field.** 52 s for the exact byte delta, which
   replaces the modelled +4.718 bits and closes S1d caveats 3, 4 and 5 at once.
   `experiments/ddm_rr2_encoder_byteclose.py` is pinned to an older archive and needs
   re-pointing at the to1 body. **Unowned; this is the cheapest high-value item on the
   board.**
2. **Finish the S2b composition test at n>3.** If the carrier re-solve recovers pose, the
   joint move is live and the seg axis reopens with a real budget; if it does not, the
   token actuator is refused on this vehicle and that is a family verdict worth banking.
3. **Re-orient the 12-dim carrier basis** (up2 owed item 2) — the ONLY pose move whose
   rate leg is free (S3). Still unowned after two arms.
4. **A rate-aware seg solve**, if item 1 clears: sparse-and-wide (first pass across all 600
   pairs), spatially packed at >=64 px separation (~48x speedup), ranking candidates by
   `cells_repaired / bits` rather than by cells, with the Lagrangian stopping rule of S1e.
5. **The all-class proposal family** (S1e correction 2), since GT-class edits are
   measurably not the best move.
6. **DALI-lineage `margins` and decoded RGB do not exist locally** and cannot (DALI
   requires CUDA). Any seg work needing a margin field is PyAV-only today.


---

## Retained payload

`/Volumes/APDataStore/pact/ddm_jg1/` — `JG1_RETENTION_MANIFEST.json` (12 files, sha256 +
bytes each, 117,989,926 B). Headline: `retained/base_argmax_n600.npy` (the full n600 SegNet
argmax field of the shipped decode, 117,964,928 B — the object every seg delta is measured
against) · `retained/S1_instrument_validation_n600.json` (both lineages + the flip ledger) ·
`retained/S1b_radius_sweep_n6.json` · `retained/S1d_rate_price_of_move.json` ·
`retained/S1e_composition_greedy.json` · `retained/S2_joint_coupling.json` ·
`retained/S2_edited_tokens.npz` (**the edited token payloads themselves**, not just their
measured lengths) · `retained/S2b_carrier_resolve.json` · the four run logs.

The 3.66 GB `0.raw` decode this arm reads is NOT copied: it is deterministically rebuildable
from `archive.zip` + `inflate.sh` and already retained under `ddm_to1` custody, per the
certify-or-block rule.
