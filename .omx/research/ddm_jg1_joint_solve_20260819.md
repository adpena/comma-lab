# ddm_jg1 — the full joint solve over every stored section of the shipped object

- **arm** `ddm_jg1` (task #1134 — the js1 joint line reborn as a SOLVE)
- **date** 2026-08-19
- **axis** every number this arm measures is `[macOS-CPU advisory]` unless it carries an
  explicit DALI-lineage tag · `score_claim=false` · `promotable=false`. This arm fires no
  Modal job; MAIN owns the T4 slot.
- **cost** $0.
- **store** `/Volumes/APDataStore/pact/ddm_jg1/`
- **status** IN PROGRESS — this memo is written incrementally, stage by stage, and
  committed at every stage boundary.

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

*(S2/S3 sections follow as they are measured.)*
