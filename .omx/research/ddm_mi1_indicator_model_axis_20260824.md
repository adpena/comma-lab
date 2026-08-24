# ddm_mi1 — the shipped model already conditions on every axis the charter named but one; that one is worth **211 B**, and a PAID probability model misses break-even by **47×**

**Date:** 2026-08-24 · **Arm:** `ddm_mi1` · **Pointer:** UNMOVED · **No Modal job, no Metal job, no
scorer, no training, no frame rendered. $0.**
**Axis:** `[macOS-CPU advisory / scorer-free shipped-receiver instrumentation]`.
`score_claim=false` · `promotable=false` · no archive built, none promoted.
**verdict_scope:** two verdicts at two different scopes — see §9.

---

## 0. Result first

**1. The recall gate fired and it removed most of the charter's candidate space before any build.**
Read at source from the shipped DX2 r7 runtime, the adaptive corrector's context is exactly
`5 (predicted class) × 64 (confidence bin) × 2 (t−1 agreement) × 2 (t−2 agreement) × 8 (temporal
run length) × 5 (boundary bucket) = 51,200 cells`
(`runtime/rr4_free_corrector.py:95,231-232`, sha `96fd35aa…`). **t−2 agreement and temporal run
length are already in the shipped stack.** `ddm_tba1`'s D6 said so; I verified it in the code rather
than trusting the memo, and it is right.

**2. Of the five causally-legal axes I enumerated, one is a conditioning site nowhere in the shipped
stack: ABSOLUTE POSITION IN THE FRAME.** *(Scope: absent from the surfaces enumerated in §2, which are
every conditioning site in the shipped runtime. Not a claim that no other causal axis exists — I
searched the shipped code exhaustively, not the space of possible features.)* The shipped network is
built with `patch = HPAC_PATCH = 64` (`cpr1/inflate.py:33,257`), so it tiles the frame into **6×8 = 48
tiles**, gives every tile the **same** tile-relative coordinate grid
(`cpr1/hpac_integer.py:287-294, 393`) and the same per-frame shift; no corrector carries a position
feature at all. **Which tile a pixel is in is structurally invisible to the whole stack.** MEASURED,
cross-fitted over all 117,964,800 positions, conditioning on which tile is worth **+112.94 B held-out
gross**, and on a 4× refinement of it (`patch192`) **+211.13 B**.

**And the negative control failed, which is the more useful half.** `subtile4` — quadrant *within* the
tile, which the network **does** receive on its coordinate channels — was carried as a control that
should return ~0. It returned **+56.51 B from four cells**. Chasing that led to the mechanism:
**the shipped model's confidence does not fully account for how much of its own causal context has
been decoded yet.** Eight bins of the within-tile decode-group index (`groupbin8`) return
**+64.20 B from 8 cells** with the smallest fitted offsets on the ladder — the model is over-cautious
early in the scan and over-confident late (§5). **A feature being an INPUT to a trained network does
not mean the network consumed it**, and that corrects the assumption my own ladder was designed on.

**3. The charter's question answers NO, by 47×.** A paid 10K-int8 model costs **10,000 B**
(`ddm_eu2`, 0.006658589531 S ÷ 6.658590e-07 S/B = 9,999.9993 B). Break-even is **8.9867% of the
111,275.62 B indicator**; closing the 42,381.16 B demand alone needs **47.073%**. The richest
unconsumed context on the whole stream returns **0.19%**. **47.4× short of break-even, 248× short of
mattering.**

**4. A new bound on the whole conditioning family, and it is the number I would carry forward.**
The shipped model's realised indicator cost exceeds *its own self-predicted cost* by
**2,162.13 B, at z = +11.89** against the perfectly-calibrated null. That excess is systematic, not
sampling noise — but it is **4.63× smaller than the paid model's own byte cost** and **24.2× smaller
than the 52,381 B needed to close the demand while carrying that model**. Even a model that captured
*all* of it would not pay for itself.

**5. `ddm_cx3` had already answered the charter's question on this exact body, two days ago, in the
replacement formulation: the model axis supplies 0 B** — best challenger 125,210 B against the
shipped 113,777 B (**11,433 B worse**), hindsight-ideal data term **3,447 B worse before model cost**.
I did not re-run it. My ladder is the complementary *additive* formulation (β = 0 nests the shipped
model, so it can only help), which is the strictly more favourable test — and it agrees.

**The one positive is real and I am nominating it:** position conditioning at **+211.13 B** is
**14.1× the 1e-5 naming bar** (ΔS 1.406e-04) and **2.0× larger than `ddm_ma1`'s −104.58 B**, which the
campaign built and adopted. It costs **zero stored bytes**. **But build the 8-cell `groupbin8` rung
first** — 64.20 B, the best-conditioned row on the ladder, against a decode wall with 160.2 s of
margin. §7 is the fire order and the three blockers.

---

## 1. Corrections to the charter — eight, and the first three change what the arm is

**(a) The 76,600 B was already measured, and it is not what the charter called it.** MAIN's relay
corrected the subtraction and is right: **76,601.5389368755 B** (re-derived here, matching `hc1`'s
76,601.54) is the **"argmax is wrong" BRANCH INSIDE the indicator**, not the indicator minus `df1`'s
zero mode. It is the surprisal of the flip events under the model, `Σ_flips −log2(1−pmax)`. It is
**not** "the bytes spent saying WHERE the model is wrong" — nothing addresses anything. The two
branches are not separable levers; they are the two outcomes of one binary code, and a change to the
model moves both, in opposite directions.

**(b) The exchange ratio of a lossless recode is 0, not undefined — and this family is not new.**
Damage ÷ credit with zero damage is **0** whenever credit is positive; undefined only if the credit is
also zero. So a lossless recode is not "the family the kill-mechanism cannot reach" — it sits at the
best possible point of a ladder that never applied to it. And the campaign has been working it since
2026-08-17: `fx1`, `fx2`, `ma1`, `r012`, `to2`, `cx3`, `oe1`, `tba1`, `ad2`, `hc1` are all
zero-distortion. **The operative bar here was never the exchange ratio. It is plain byte arithmetic:
did the stream shrink by more than the machinery cost.** That is the bar §5 applies, and it is the
bar the paid model fails.

**(c) 111,275.62 B is the realised CODE LENGTH, not the entropy — and the difference is this arm's
actual target.** The entropy is `Σ_i H_b(1−pmax_i) = 109,113.49589719012 B`, which is the *same
object* as `df1`'s `address_bound` (872,907.9671775216 bits; I re-derive 872,907.967177521 — agreement
to 13 significant figures across two independent implementations). The 2,162.13 B gap is
**not** "the calibration slack hc1 priced at 8.44 B": `hc1`'s recalibration recovers **40.91 B gross
held-out of it, i.e. 1.9%**. The other **98.1% is conditional structure that monotone repricing
cannot see** — which is exactly why conditioning was the right thing to test. My ladder measures
211.13 B of that gap, **9.8% of it**, on the one unconsumed axis I found.

**(d) `ddm_eu2`'s 10K candidate is not a model of this stream.** It is a *video-invariant
comma10k/openpilot-trained semantic prior* (`EU2_RECEIPT.md`), priced as a context/orderer against the
**TR1/qo1** vehicle whose token stream was **346,478 B — 3.05× the DX2 stream's 113,777 B**. Its
"2.89% of the stream" figure is stale by that factor; on DX2 the same packet is **8.79%** of the token
block and **74.0%** of the entire 13,515 B HPAC model block. Its own GO bar (recover ≥30% of a
106,954 B ordering gap, or save ≥30,000 token bytes) is 3× above break-even, and nothing measured on
this stream is within 100× of it.

**(e) There are TWO never-fired queued experiments on this question, not one.** Besides `#938`/`eu2`,
`ddm_cl1` (2026-08-09, `.omx/research/ddm_cl1_capacity_20260809/PREREGISTRATION.md`) preregistered
*"whether allocating more serialized bits to the HPAC prior saves more serialized token bytes than it
costs"* with break-even `Δtoken / Δmodel < −1` — **QUEUED-WITH-A-FIRE-ORDER, sandbox fire REFUSED,
never run.** Both were written against pre-DX2 bodies. §5 prices the question they share on the live
body, at $0, and the answer removes the reason to fire either.

**(f) The `ma1`/`fx1` miss count and the `hc1`/`df1` flip count are different objects.** `fx1` and
`ma1` count **223,694** misses against the *pre-corrector* HPAC argmax; `df1`/`hc1`/this arm count
**227,671** flips against the *coding-row* argmax. The corrector moves the argmax at **3,977**
positions net. Anyone joining those two literatures must pick one.

**(g) `ddm_ma2`'s structural fact, confirmed independently at source: the coding alphabet IS the
output partition.** `NUM_CLASSES = 5` (`runtime/residual_archive.py:42`) and `#define F26_ALPHABET 5`
(`runtime/f26_hpac_native.c:25`) — one symbol per pixel, and the five symbols are the SegNet classes.
So this arm's "is the argmax right?" indicator is literally *"is the shipped partition's class correct
at this pixel?"*, and every context I propose is conditioning on class labels. It also means the
expensive positions of my indicator and the seg-damaging positions are the **same** positions: `tba1`
measured the top 1% of positions carrying 96.32% of bits and 90.96× enriched in manufactured seg
error, and `ma2` measured GT-disagreement positions costing **10.17× more per position** than agreeing
ones. **That co-location is why the indicator is payload and not overhead** — MAIN's §6 correction,
with the mechanism attached.

**(h) Numbers I was given and deliberately did not use.** Neither the 21.62× figure (a refusal
magnitude, seg-leg-only, never built), nor `af1` §9's nomination (void per its own §2.4), nor the
17,957 B `ad2` figure (NR1's vehicle, not DX2) appears anywhere in this arm's arithmetic. §6 is
built only from measurements taken on the DX2 body, plus two `fx1`-era ceilings that are labelled as
such.

---

## 2. What the shipped model conditions on — the source-verified inventory

This is the load-bearing read. Runtime pinned at
`/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2` (the body behind archive
`976f706d…`, S 0.14822 @ 180,368 B).

| stage | conditioning variable | levels / form | source |
|---|---|---|---|
| `IntegerHPAC` (13,515 B, **counted**) | decoded causal cone in the CURRENT frame, diagonal-masked wavefront | 7×7 mask-A + 5×5 dil-2 mask-B + 3×3 dil-4 mask-B | `cpr1/hpac_integer.py:234-261` |
| | previous decoded frame f−1, 3×3, then propagated by the masked convs | 5-class one-hot → 64 ch | `:262-264, 335-340` |
| | **tile-relative** coordinates | 64×64, identical in every tile | `:287-294, 393` |
| | frame index | embedding, 600 × 8 | `:217, 318` |
| static table (**counted**) | boundary bucket × predicted class | 5 × 5 | `runtime/residual_archive.py:672-676` |
| `rr4` corrector (**free**) | predicted class | 5 | `runtime/rr4_free_corrector.py:231` |
| | confidence `⌊−log2(1−pmax)/0.5⌋` | 64 | `:83, 219-226` |
| | **t−1 agreement** with argmax | 2 | `:224` |
| | **t−2 agreement** with argmax | 2 | `:225` |
| | **temporal run length**, saturated | 8 | `:84, 229, 311` |
| | boundary bucket `d(f−1)` | 5 | `:86, 232` |
| `fx2` mixer (**free**) | 4 causal neighbours (L, U, UR, UL) + local homogeneity | 6 / 5 levels | `runtime/fx2_model_axis_corrector.py:147,156,160` |
| `ma1` within-miss (**free**) | (neighbour classes, prev1) | 1,296 cells | `runtime/free_corrector.py:112-143` |
| — | **WHICH 64px TILE (6×8 = 48)** | — | **ABSENT EVERYWHERE** |

Source digests: `rr4_free_corrector.py` `96fd35aa…` · `hpac_integer.py` `cea40a9b…` ·
`free_corrector.py` `dd337159…` · `fx2_model_axis_corrector.py` `77e81ac8…` ·
`residual_archive.py` `aca361f3…` (the same digest `hc1` pinned).

**This is a heavily engineered multi-stage adaptive context model, not a static table.** Three of the
five contexts I had planned to test are inside it. The one that is not is absolute position — and it
is absent for a legible reason: the model tiles the frame into 192 patches and gives every patch the
*same* coordinate grid and the *same* frame shift, so it cannot distinguish patch 3 from patch 180
except through content.

---

## 3. Re-derivation — every control reproduces, independently

Computed from the retained fields, in my own code, before any analysis (`VERIFY.json`).

| quantity | this arm (MEASURED) | prior arm | agreement |
|---|---:|---:|---|
| positions | 117,964,800 | — | — |
| flips (`coding argmax ≠ transmitted`) | **227,671** | hc1 227,671 | exact |
| float32-saturated positions | **67,955,679** | df1 67,955,679 | exact |
| flips among saturated | **0** | df1 0 | exact |
| indicator code length | **111,275.62229665744 B** | hc1 111,275.62 | exact |
|  · "argmax is right" branch | **34,674.083359781944 B** | hc1 34,674.08 | exact |
|  · "argmax is wrong" branch | **76,601.5389368755 B** | hc1 76,601.54 | exact |
| bits per flip, wrong branch | **2.6916573103074346** | hc1 2.6917 | exact |
| `Σ H_b(1−pmax)` | **872,907.967177521 bits** | df1 872,907.9671775216 | 13 s.f. |

**One number that is new and that I would put in front of any future model-axis charter: when the
shipped model is wrong, it had already assigned an average probability of `2^(−2.69166) = 0.154786` to
being wrong.** The best *free enumerable* feature anyone has found concentrates flips at only
**4.769%** (`hc1`'s causal `d = 0` cell, 2.16% of positions, 53.4% of all flips). **The shipped model
is already 3.246× sharper than the densest cell of the best free variable.** Any new context has to
beat 15.48%, not 4.77% — and the campaign has repeatedly priced levers against the latter.

---

## 4. The realised excess, and whether it is noise — NEW

Cross-entropy = entropy + divergence. The shipped model pays **890,204.9783732595 bits**; its own
stated probabilities predict **872,907.967177521 bits**. The excess is
**17,297.011195738567 bits = 2,162.126399467321 B**.

Is that a systematic gap or one unlucky draw? Under the null *"the flip field is an independent
Bernoulli draw with the model's stated per-position probabilities"*, the realised code length has mean
equal to the entropy and variance `Σ_i q_i(1−q_i)·log2(q_i/(1−q_i))²`. MEASURED:

| | value |
|---|---:|
| realised − entropy | **17,297.011 bits** = 2,162.126 B |
| null σ | **1,454.878 bits** |
| **z** | **+11.889** |

**The null is rejected at 11.9σ. There IS residual structure the model's own probabilities do not
capture, and it is worth ~2,162 B.** That is the honest size of the whole conditioning target on this
stream, and it is the number the next model-axis charter should be written against.

**What this is NOT.** It is not a hard ceiling. A better-conditioned model has sharper probabilities,
so *its* entropy term is lower too, and it can in principle go below 872,908 bits. The bound that
covers that case is not mine — it is `cx3`'s, which measured the replacement formulation's
hindsight-ideal data term at **3,447 B WORSE** than the shipped model. Between them: repricing the
shipped model's output can win at most ~2,162 B (this arm), and replacing it with the best named
causal summaries *loses* 3,447 B before model cost (`cx3`). Both are far below break-even.

---

## 5. The conditioning ladder — MEASURED

Map family `q' = σ(logit(1−pmax) + β_c)`, one offset per cell of the context, **which nests the
shipped model exactly at β = 0**, so every delta is the new mechanism and never the plumbing. Newton
fit, 2-fold cross-fitted over a seeded random split of all live positions (**never a prefix** —
`[[m88]]`/`[[m96]]`). Float32-saturated positions are excluded *exactly*: their cost is 0 for any
finite offset, so they are immovable by this family.

Live positions **50,009,121**; base indicator **111,275.6223 B**. Seed 20260824:

| context | in shipped stack? | cells | **held-out B** | in-sample B | max\|β\| |
|---|:--:|---:|---:|---:|---:|
| `none` (pure recalibration) | yes | 1 | 2.77 | 2.77 | 0.01 |
| `boundary_d` | yes | 5 | **5.27** | 5.80 | 0.04 |
| `agree2` (t−2 agreement) | yes | 2 | 17.28 | 17.29 | 0.05 |
| `run8` (temporal run length) | yes | 8 | 15.43 | 16.73 | 0.05 |
| `cls_ubin` (the rr4 head) | yes | 320 | −71.57 | 61.04 | 26.68 |
| **`tile48` (WHICH 64px tile)** | **no** | 48 | **+112.94** | 122.34 | 24.38 |
| **`subtile4` (quadrant IN the tile)** | **input present** | 4 | **+56.51** | 57.31 | 0.10 |
| **`groupbin8` (SCAN POSITION in the tile)** | **input present** | 8 | **+64.20** | 65.77 | 0.11 |
| `group190` (exact decode-group index) | input present | 190 | +51.25 | 104.37 | 0.29 |
| **`patch192` = `tile48 × subtile4`** | **no** | 192 | **+211.13** | 234.54 | 24.96 |
| `row384` (absolute row) | no | 384 | +71.11 | 126.23 | 24.60 |
| `patch192 × ubin` | no | 12,288 | −2,120.17 | 969.59 | 26.68 |
| `frame_band × row` | no | 9,216 | −484.95 | 561.02 | 24.70 |
| `row384 × ubin` | no | 24,576 | −5,874.29 | 802.88 | 27.27 |

**The instrument is validated by an independent positive control.** `hc1` measured the per-`d` offset
rung at **5.43 held-out / 5.92 in-sample**; my independently written fit gives **5.27 / 5.80** —
**0.16 B and 0.12 B apart** on a 5-parameter map, across two implementations that share no code.

**Three things this table says.**

**`tba1`'s D6 is confirmed by measurement as well as by source.** The two features D6 names — t−2
agreement and run length — return **17.28 B and 15.43 B**. The shipped stack has consumed them to
within ~17 B. So has the rr4 head, which goes *negative* held-out.

**The oracle mirage reproduces exactly.** In-sample gain climbs monotonically with cell count
(2.77 → 234.54 → 802.88 → 969.59 B) while held-out peaks at 211.13 B and then collapses to
−5,874.29 B. This is `hc1`'s and `pk3`/`pk4`'s signature on a third independent axis. **Anyone quoting
a ~970 B "oracle" for position conditioning is quoting noise.**

**The position gain SPLITS, and the split is the most useful thing this arm found.** The shipped
network is built with `patch = HPAC_PATCH = 64`, so its own tessellation is 6×8 = **48 tiles**, and my
192-cell grid is a 4× refinement of it. Decomposing:

| cell | what it is | held-out B | B per cell |
|---|---|---:|---:|
| `tile48` | WHICH tile — structurally invisible to the network | **112.94** | 2.353 |
| `subtile4` | quadrant WITHIN the tile — the coordinate channels DO carry it | **56.51** | **14.127** |
| `patch192` | their join | **211.13** | 1.100 |
| — | interaction (211.13 − 112.94 − 56.51) | **41.68** | — |

**Two findings, and the second one corrects my own instrument.**

**(i) The invisible axis pays 112.94 B**, as designed: the network gives every tile the same
coordinate grid and the same frame shift, so which tile a pixel is in cannot reach it.

**And the retained per-cell table says WHY `subtile4` pays, which turns a curiosity into a mechanism.**
The decode order within a tile is `columns + 2·rows` (`cpr1/inflate.py:275-281`), so quadrant 0 is
decoded FIRST — with the least decoded context — and quadrant 3 LAST. The fitted offsets run monotonically
along exactly that axis:

| quadrant | scan order | live positions | flips | flip rate | fitted β | gain |
|---|---|---:|---:|---:|---:|---:|
| 0 (top-left) | first | 13,134,258 | 64,629 | 0.492% | **−0.0631** | 17.80 B |
| 1 (top-right) | — | 11,969,567 | 60,078 | 0.502% | +0.0019 | 0.01 B |
| 2 (bottom-left) | — | 12,587,644 | 52,713 | 0.419% | +0.0479 | 7.93 B |
| 3 (bottom-right) | last | 12,317,652 | 50,251 | 0.408% | **+0.0988** | 31.55 B |

**The model is over-cautious where it has least context and over-confident where it has most.** β is
negative at the start of the scan and positive at the end, and the true flip rate falls 0.492% →
0.408% along it. The network partly tracks this — its confidence does rise — but not enough.

**So I tested the mechanism instead of asserting it**, by adding the scan variable itself:
`groupbin8`, eight bins of the 190-group decode index within the tile.

| cell | cells | held-out B | B/cell | max\|β\| |
|---|---:|---:|---:|---:|
| `subtile4` (quadrant proxy) | 4 | 56.51 | 14.13 | 0.099 |
| **`groupbin8` (scan position)** | **8** | **64.20** | **8.03** | **0.112** |
| `group190` (exact group index) | 190 | 51.25 | 0.27 | 0.291 |

**The hypothesis holds: the scan variable beats its quadrant proxy by 13.6%** (64.20 vs 56.51 B), and
the *exact* 190-level index is WORSE held-out than eight bins of it (51.25 vs 64.20) while its
in-sample doubles (104.37) — the optimum sits near eight bins and the rest is variance. So this is
**not a position effect in the image**; it is a **decode-order effect**: the shipped model's confidence
does not fully account for how much of its own causal context has been decoded yet.

**(ii) `subtile4` was supposed to be a NEGATIVE control and it returned 56.51 B.** The network
*receives* the tile-relative coordinates as input channels on `conv_a`; a 4-cell offset table on the
final coding row still recovers 56.51 B from them. **So "in the shipped stack" cannot mean "present as
an input" — it has to mean "measurably consumed."** The rr4 cells satisfy the stronger reading
(`boundary_d` 5.27, `agree2` 17.28, `run8` 15.43, `cls_ubin` negative) because adaptive counts
converge on their own context by construction. A trained network carries no such guarantee, and this
row is the receipt. I had the weaker definition when I designed the ladder; the measurement corrected
it.

**And it reorders the fire order, for two reasons.** In a mixer whose binding constraint is decode
wall-clock and whose cost scales with cells, `groupbin8` returns **64.20 B from 8 cells** and
`subtile4` **56.51 B from 4** — against `tile48`'s 2.35 B/cell and `patch192`'s 1.10.

**And they are the only well-conditioned rows.** `groupbin8`'s largest fitted offset is **0.112** in
log-odds and `subtile4`'s **0.099** — small nudges on densely populated cells. `tile48`'s is **24.38**
and `patch192`'s **24.96**, meaning cells ran to the Newton clip, which happens only when a cell
records **zero flips in its training fold** (23 of `tile48`'s 48 tiles have zero flips overall). That
is a real saving on this video but the *unregularised* form of the bet, and the KT-smoothed online
estimator that would actually ship never assigns zero. **So `tile48`'s 112.94 B and `patch192`'s
211.13 B are upper bounds on their shippable form in a way that `groupbin8`'s 64.20 B is not.**
*(Scoping that honestly: the clipped cells carry only 4.29 B of `tile48`'s 122.34 B in-sample total —
3.5% — so the caveat is about the mechanism's form, not about most of its magnitude.)*

### One defect of mine, and two controls I did not plan

My first `run8` row was measured with the run counter as **uint8**, which WRAPS at `RUN_CAP = 255`;
`rr4` declares it **int64** (`rr4_free_corrector.py:176`), which SATURATES. Mine therefore relabelled
every position whose class had been stable for 256+ frames — most of the static ego-hood and sky — as
run 0 instead of run 7. I found it by re-reading the shipped `__init__` rather than the line I had
already transcribed. Corrected, `run8` moves **15.23 → 15.43 B held-out** (16.58 → 16.73 in-sample);
no other row touches `run_state`, and the verdict is unchanged. **The defect was real, its effect was
0.20 B, and both facts belong in the record** — a control built on a lookalike is not a control, even
when it happens to land within 0.2 B of the right answer.

**Cross-path control.** `stage_verify` computes the base cost through `log`/`log1p`; `stage_ladder`
computes it through `logaddexp`. They return **111,275.62229665744** and **111,275.62229665765 B** —
agreement to **2.1e-10 B**, so the ladder's zero point is the verify stage's number and not a second,
differently-rounded object.

**Nesting control.** `none` returns 2.77 B in-sample and 2.77 B held-out, identical to 2 decimal
places, as a 1-parameter map on 50M samples must. A ladder whose 1-cell rung disagreed with itself
across folds would be reporting split noise, not structure.

### Seed stability

Seeded random 2-fold splits, all 50,009,121 live positions, held-out bytes:

| seed | `tile48` | `subtile4` | `patch192` | `row384` | `boundary_d` | `none` |
|---|---:|---:|---:|---:|---:|---:|
| 20260824 | 112.94 | 56.51 | 211.13 | 71.11 | 5.27 | 2.77 |
| 777 | 111.81 | 55.70 | 202.51 | 77.10 | 4.23 | 2.07 |
| 31337 | 111.56 | 54.75 | 204.79 | 78.53 | 4.63 | 2.41 |
| **spread** | **1.38 (1.2%)** | **1.76 (3.2%)** | 8.62 (4.2%) | 7.42 (9.8%) | 1.04 | 0.70 |

**The two load-bearing rows are the two most stable rows in the table** — `tile48` at 1.2% and
`subtile4` at 3.2%, against 9.8% for `row384` and 47.7% for the degenerate `cls_ubin`. The answer is
not a lucky split, and the stability tracks cell population exactly as it should.

**Scope on the seed table:** `groupbin8` and `group190` were added *after* the three-seed sweep, once
the retained cell table pointed at scan position, so they carry **one seed only** (20260824). Their
cells are as densely populated as `subtile4`'s and their fitted offsets are as small, so I expect
comparable stability — but that is INFERRED, not measured, and the implementing arm should re-run the
sweep before quoting `groupbin8`'s 64.20 B as a ±number.

---

## 6. The break-even arithmetic — the charter's question, priced

| quantity | value | source |
|---|---:|---|
| indicator code length | 111,275.62229665744 B | MEASURED, §3 |
| exchange rate | 6.658590e-07 S/B | `ddm_tx1` §0, cited |
| demand at fixed distortion | 42,381.16120555642 B | `ddm_tx1` / `df1` |
| paid 10K-int8 model | 10,000 B (0.006658589531 S) | `ddm_eu2` |
| **break-even: gross saving needed** | **10,000 B = 8.9867% of the indicator** | DERIVED |
| **to close the demand alone** | **52,381.16 B = 47.073% of the indicator** | DERIVED |

MAIN's 9.0% and 47.1% are **confirmed**.

Every ceiling and win I found in recall on this stream, against those two bars (scope: the arms named
below; I did not search exhaustively for a twelfth):

| arm | what it priced | measured | vs 10,000 B |
|---|---|---:|---:|
| `hc1` | recalibration of the coding row | +8.44 net / +40.91 gross held-out | 1,185× short (244× on gross) |
| **`mi1`** | **conditioning, richest unconsumed axis** | **+211.13 held-out** | **47.4× short** |
| `fx2` | hit-event remainder under raced contexts | ≈315.3 B (hindsight-optimal, fx1-era body) | 31.7× short |
| `fx1` | within-miss perfect-model ceiling (`ma1` since harvested 104.58 B of it) | 1,247.19 B (fx1-era body) | 8.0× short |
| `mi1` | realised excess over the model's own entropy | 2,162.13 B (z = 11.89) | 4.63× short |
| `r012` | composable coder ceiling | 88 B | 114× short |
| `tba1` D1 | explicit selector at any threshold | +9.45 B (colex +9.90) | 1,058× short |
| `ad2` | RC1 addressing, NR1 QCTX/QEVENT | 0 B | ∞ |
| `cx3` | model axis, replacement formulation | **0 B** (+11,433 B worse) | fails |
| `oe1` | zero-stored causal escape member | **+10,818 B worse** | fails |

**The sharpest line in the table is the cumulative one.** Every zero-byte model-axis win the campaign
has produced on this stream — `fx1` **−560.07 B** (byte-closed and fired), `fx2` D1 **−151 B**
(projection), `ma1` **−104.58 B** (measured, not byte-closed), plus this arm's **−211.13 B**
(held-out code length, not byte-closed) — totals **≈1,027 B**, and only 560 B of that is byte-closed.
**A paid 10K model must beat the entire history of the model axis by 9.7× just to break even, and by
51× to matter.** Nothing in the record suggests it can; `cx3` measured the closest formulation at
*worse than nothing*.

**DERIVED:** since the conditioning target is the 2,162.13 B excess, a paid model would have to be
**4.63× more wrong than the shipped model actually is** before this family cleared its own byte cost.

---

## 7. The one positive — nominated, with its blockers

**Position conditioning: `patch192` +211.13 B gross held-out, zero stored bytes, ΔS 1.406e-04 =
14.1× the 1e-5 naming bar.** For scale, `ddm_ma1` was built and adopted on **−104.58 B / ΔS
−6.99e-05 / 20.0× bar**; this is **2.0× larger**. It is receiver-derivable at zero cost — the decoder
always knows which pixel it is decoding — and rule-118 clean: nothing video-derived enters the
runtime.

**Build `groupbin8` FIRST, not `patch192`.** The decode wall is the binding constraint (`fx1`:
160.2 s of margin, and members are the expensive dial), and members cost table updates per group, i.e.
cells. `groupbin8` is **8 cells for 64.20 B**, it is the best-conditioned row on the ladder
(max\|β\| 0.112, no clipped cell), and it is **6.4× the naming bar** on its own. Ladder the build:
`groupbin8` (8 cells) → `tile48` (48, different axis, so it should compose) → `patch192` (192), and
stop at the first rung that fails the decode gate. **That ordering did not exist before §5's
decomposition** — the flat "+211 B" number would have sent the build straight to the 192-cell rung,
which is the worst-conditioned and most expensive one on the board.

**It is NOT a rate claim, and three things must be measured before it is one.**

1. **Static versus online.** My instrument is a static 2-fold cross-fit. The shipped mechanism would
   be an online KT cell like `rr4`/`ma1`. `fx2` measured online beating static by ~5,500 B on a
   *temporal* split; my *random* split removes non-stationarity but charges no warm-up. **The sign of
   the difference is not known and must be raced, not projected.**
2. **Code length is not bytes.** `ddm_fs2` measured `−log2 p` mispricing the real re-encode by
   **0.93× for moves away from the argmax and 0.09× for moves toward it**. The 211.13 B is a model-ledger delta; **the verdict is a physical RC64
   re-encode**, which this arm did not run.
3. **Composability.** My offset sits on the final coding row. Once the `fx2` mixer is refitted with a
   patch member, part of the 211 B may already be inside it. `fx2` §3 measured six SSE formulations
   losing precisely because their context duplicated the mixer's own.

**Fire order (queued, not fired):** add `groupbin8` as a `Fx2ModelAxisMixer` member context and race
it against the shipped 13-member D1 build under `fx2`'s own harness, with the frozen
`FreeCorrector(plane)` identity control at delta +0.000000 bits; then `tile48`, then `patch192`.
Verdict = physical RC64 stream bytes, not the model ledger. Falsifier = the `groupbin8` rung does not
clear **+20 B** of real stream on full n600 (under a third of its measured code-length gain, allowing
for `fs2` mispricing and for the mixer having already absorbed part of it). **Not yet measured and
worth one row:** `tile48 × groupbin8` (384 cells) — the two axes are independent, so their join is the
natural composite, and nobody has priced it.

---

## 7b. R3 adjudicated — MAIN's open question, resolved at source

**MAIN's fork:** either `rr9` (`#1244`) already closed `tri1` §3.5's "last live cell" R3 — *lossless
traversal reorder WITH model refit* — or it closed only the composition surface and left the fused
single move open.

**Resolved: `rr9` closed R3 ITSELF, and `tri1` §3.5 is stale.** `ddm_rr9_reorder_refit_20260824.md`
is titled *"R3 adjudicated"* and does three things in order:

1. **It verified `tri1`'s premise before building.** Measured over the four named artifacts: **0
   refit tokens** {refit, retrain, fine-tune} against **91 generic-coder tokens** {brotli, lzma, zlib,
   zstd}. So `to2` and `ad2` genuinely *replaced* rather than *refitted*, R3 was genuinely unmeasured,
   and `tri1` was right to name it — **as of the moment `tri1` wrote it.**
2. **Case (a), within-group reorder** — the only permutation the coder admits without breaking
   decodability. **MEASURED byte-neutral at full n600: 113,777 B → 113,777 B, 0 B, 0.000000%,
   ΔS 0.0**, round trip proven lossless by digest, with a negative control that detects **one** flipped
   token in 117,964,800.
3. **Case (b), cross-group reorder** — ruled **out of R3's scope by construction**: the group-index
   expression is simultaneously the coding partition *and* the causal mask baked into the trained
   convolution weights, so changing it means training a different model.

**I verified case (b)'s load-bearing claim at source, in a file `rr9` did not cite.** The group plan
is `grid = columns + HPAC_DELTA * rows` with `HPAC_PATCH = 64`, `HPAC_DELTA = 2`
(`cpr1/inflate.py:33-34, 275-287`). The network's causal masks come from
`patch_group_mask(kernel, delta, type_)`, `offset = column − center + delta·(row − center)`, and the
shipped model is constructed with **`delta = 2`** (`cpr1/hpac_integer.py:73-83, 186, 234-261`).
**Same expression, same delta, two files.** Traversal order and trained receptive field are one
object. `rr9` is right.

**Do not spend a slot on R3.**

**And the two arms compose into a single sentence.** `rr9` case (b) — *"changing the traversal means
training a different model"* — is precisely the cell **this** arm prices. `rr9` declared it out of
R3's scope; §6 says what it would have to clear: **10,000 B to break even, 52,381 B to matter**,
against a stream whose richest unconsumed context returns **211 B** and whose closest measured
formulation (`cx3`) returns worse than nothing. **R3 case (b) is not merely out of scope — it is
priced, and it does not pay.**

**On `#1201` — "reordering is a SUBSTITUTE for a context model."** If that law holds, R3 and this
arm's conditioning are the same lever measured two ways, and the two measurements settle which
formulation dominates on this body: the **reorder** formulation returns **0 B** (`rr9` case (a),
exact, byte-closed); the **conditioning** formulation returns **+211 B** (this arm, code length, not
byte-closed). **Conditioning dominates by measurement — fire §7's member, not a reorder.** Scope: this
compares the only two formulations either arm actually measured; it does not prove the substitution
law.

---

## 8. Honest limits

- **This arm measured no `d_seg`, no `d_pose`, no `S`, and built no archive.** `dD = 0` is an identity
  for this family (§0), not a measurement: a probability model feeds the coder, the coder emits the
  transmitted symbol, so the decoded field is bit-identical and every SegNet cell and PoseNet input is
  unchanged. Same argument as `hc1` §7.
- **The ladder's rich rows are contaminated by unregularised MLE.** Cells with zero training-fold
  flips drive β to the ±4-per-step clip; the resulting held-out numbers are honest but they measure
  overfitting, not absence of information. **The load-bearing rows are the low-cell ones**
  (`patch192`, `row384`, and the four shipped controls), where every cell holds ≥130,000 live
  positions.
- **The 211.13 B is one context family on one split geometry.** A learned position feature with a
  smoother parameterisation could find more; nothing here bounds that.
- **`tile48`'s 112.94 B includes at least one Newton-clipped cell** (max\|β\| 24.38, reached only by a
  cell with zero training-fold flips). That is an honest saving on this video but the *unregularised*
  form of it; the KT-smoothed online estimator that would ship never assigns zero, so 112.94 B is an
  upper bound on its shippable form. `subtile4` (max\|β\| 0.099) carries no such caveat.
- **My "in shipped stack" column originally meant "present as an input", and that was wrong.**
  `subtile4` falsified it in §5. Every row in that column should be read as "measurably consumed by
  the shipped stack", and only the rr4-fitted rows are known to satisfy the stronger reading.
- **`fx1`/`fx2` ceilings quoted in §6 were measured on the fx1-era body** (110,512 B token stream),
  not DX2 (113,777 B). They transfer as an order of magnitude, not as numbers. Labelled INFERRED where
  used.
- **I did not test a genuinely richer *spatial* neural receptive field, ego-motion warping, or a
  retrained network.** Those remain open and are exactly `cx3`'s stated scope carve-out. What this arm
  shows is that the *enumerable free* part of that space is exhausted to within ~211 B, so a retrained
  network would have to find its 10,000 B somewhere no enumerable summary reaches.

---

## 9. Verdict

**(1) The PAID probability-model family is CLOSED on this object, by arithmetic on measured
quantities.** `verdict_scope:` **FAMILY — a counted probability-model packet added to the DX2 token
stage** (archive `976f706d…`, n600, 117,964,800 positions). A 10,000 B packet needs 8.9867% of the
indicator to break even and 47.073% to matter. The richest causally-legal context absent from the
shipped stack returns **0.19%**; the model's entire systematic excess over its own self-predicted
cost is **1.94%**; the closest measured formulation (`cx3`) returns **worse than nothing**. **Do not fire
`#938`/`eu2` or `ddm_cl1` on this body.** Both were priced against vehicles whose token stream was
3.05× larger; on DX2 their own break-even is 47× beyond anything measured.

**(2) The FREE conditioning axis is NOT closed, and it has one live cell.** `verdict_scope:`
**FORMULATION — static per-cell log-odds offsets over enumerable causal contexts.** Absolute patch
index is worth **+211.13 B held-out gross** and is absent from every shipped surface. §7 is its fire
order. This does not reopen the paid family: 211 B is 47.4× short of paying for a model packet, and
it is free precisely because it stores nothing.

**(3) The number to carry forward instead of the charter's framing.** The conditioning target on this
stream is **2,162.13 B at z = +11.89**, not "181% of the demand". MAIN's arithmetic was true and its
implication was not: that mass is payload the decoder genuinely needs, not overhead awaiting a
cleverer scheme. Of it, `hc1` recovered 1.9% by recalibration and this arm 9.8% by conditioning on
the one axis nobody had used.

**A negative here was worth as much as a positive, and this is mostly a negative.** The pointer did
not move.

---

## 9b. Routing — one proposed memory line, and where each finding goes

**Proposed durable memory (MAIN owns the index; I am not editing it):**
`an_input_the_network_receives_is_not_a_feature_it_consumed_20260824` — *`subtile4` was designed as a
NEGATIVE control because the shipped network receives tile-relative coordinates on `conv_a`. It
returned **+56.51 B from four cells**, the best bytes-per-cell on the ladder. An adaptive counter
converges on its own context by construction; a trained network carries no such guarantee. When
auditing "is this already conditioned on?", the test is **measurably consumed**, not **present as an
input**.* Sister of `[[available-field-vs-authoritative-field]]` and
`[[measured_object_vs_named_object_20260816]]`, one level up: those govern which field you read; this
one governs whether reading it counts as consuming it.

- **To whoever builds §7:** the ladder `subtile4` → `tile48` → `patch192`, the +20 B falsifier, and the
  three blockers. Start from `RETAIN_cell_tables.json`, which says which cells carry the gain.
- **To the boundary/annulus work:** nothing new from me; `hc1`'s causal/acausal correction still binds.
- **To MAIN's ledger:** `#938`/`eu2` and `ddm_cl1` should be closed on this body, not fired (§9.1).
  `tri1` §3.5's R3 should be marked superseded by `rr9` (§7b).

---

## 10. Custody

Inputs, all retained fields from prior arms, consumed by digest and never re-derived:

| input | bytes | sha256 |
|---|---:|---|
| TO2 decoded token field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| `df1` coding argmax | 117,964,800 | `db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e` |
| `df1` coding pmax (f32le) | 471,859,200 | `f37e3d8a21d02647437bf950d7a8a75b751c2a9644c7b8ad48aca2833be4794b` |
| `hc1` causal boundary `d` | 117,964,800 | `a6ffb6fe75190ef9ec956f961f470a5c6a7251dd46f6ff2a774b355673161324` |

The token digest matches the shipped DX2 body's pin; the run **fails closed** if it does not, and also
if the flip count, the saturated count, the re-derived entropy, or the re-derived indicator cost
disagree with `hc1`/`df1`.

Outputs at `/Volumes/APDataStore/pact/ddm_mi1_indicator_model_axis/measurement_v1/`:

| artifact | bytes | sha256 |
|---|---:|---|
| `VERIFY.json` | 2,914 | `5c7b65bc1d690ff2…` |
| `LADDER_seed20260824.json` (14 contexts) | 6,922 | `7f297981bdd508a1…` |
| `LADDER_seed777.json` (12 contexts) | 5,952 | `0036444782fbb97e…` |
| `LADDER_seed31337.json` (12 contexts) | 5,948 | `57ba60b8de1860fb…` |
| **`RETAIN_cell_tables.json`** | **95,366** | `070fe1024c3e920d…` |
| `MANIFEST.json` | 2,262 | `f1f5ce9c02a84288…` |

Total 119,364 B. *(`._*` entries in the manifest are ExFAT AppleDouble metadata, not artifacts.)*
APDataStore had 124 GiB free; **Vertigo has 8.4 GiB free and was read-only.**

**`RETAIN_cell_tables.json` is the payload, not a scalar summary.** For each of seven contexts it
carries, per cell: live positions, observed flips, the model's own stated flip mass, the fitted
log-odds offset, and the base and fitted bits. That table IS the mechanism a downstream arm would
ship, and it is what located the scan-position effect (§5) after the aggregate had already been
computed. The **per-position** cost field is deliberately not retained: it is exactly reconstructible
from this table plus the retained `df1` `pmax`/`argmax` fields plus this script, so it is certified
rebuildable rather than discarded.

**Reproduce:** `.venv/bin/python experiments/ddm_mi1_indicator_model_axis.py --stage
{verify,ladder,retain,manifest}`. Verify 22 s; ladder ~190 s per seed; retain ~200 s. Every stage
fails closed on a custody or internal-consistency mismatch and writes its own receipt.
