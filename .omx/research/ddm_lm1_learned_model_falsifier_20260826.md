# ddm_lm1 — a learned parametric model cannot replace the 13,515 B HPAC network: the model axis is not underpriced, the STREAM is, by ~79,000 B

**Date:** 2026-08-26 · **Arm:** `ddm_lm1` · **Pointer:** UNMOVED · **No Modal job, no Metal job, no
scorer, no renderer, no frame rendered, no archive built or mutated. $0.**
**Axis:** `[macOS-CPU advisory / scorer-free lossless coding measurement]`.
`score_claim=false` · `promotable=false` · `promotion_eligible=false` · `rank_or_kill_eligible=false`.

**verdict_scope:** `FORMULATION` — see §9 for the exact boundary, and for the one sub-family this
does **not** close. §7 is the one section that is **not** lossless and **not** a `d_seg`-neutral
measurement; it is labelled rate-side-only throughout.

---

## 0. Result first

**1. Row 2 answers NO, and the reason is the opposite of the one the row was written to test.**
`ddm_no1` row 2 asked whether a `W`-byte learned parametric model could **replace** the 13,515 B
HPAC network and net bytes, and priced the question as a recovery fraction `r` of the 113,777 B
stream (29.6% / 34.0% / 42.8% at `W` = 5K/10K/20K). The measured answer is that **`r` is negative at
every capacity tested**: the best replacement I can build makes the stream **larger**, not smaller.
Reclaiming the whole 13,515 B model is irrelevant when the stream grows by ~79,000 B to do it.

**2. The decisive number is exact and needs no split, no seed and no training.** The best
zero-stored-byte adaptive discrete-context model over the true decode causality costs
**193,065.0 B** (Krichevsky–Trofimov prequential, `k` = 11 causal neighbours), against a shipped
token subsystem of **127,292 B** total. It misses break-even by **65,773.0 B** and is **1.517× the
entire shipped stream + model**. The KT curve has a **measured interior minimum** — 198,543.3 B at
`k` = 9, **193,065.0 B at `k` = 11**, 196,544.6 at 13, 199,303.6 at 15 — so this is not an
under-tried result: past the knee, more context makes it **worse**.

**3. A rigorous lower bound closes the whole discrete-context class, not just my implementation.**
The hindsight-ideal conditional entropy at the richest rung (17 causal neighbours, 86,063 occupied
contexts) is **150,903.2 B**. That is the code length an oracle with that context, **unlimited
parameters, zero generalisation gap and zero model cost** would achieve, so it lower-bounds *any*
model — table, mixture, MLP, transformer — whose input is those neighbours. It is still
**23,611.2 B above the `W` = 0 break-even bar** of 127,292 B, and **1.77×** the demand bar.

**4. Two unrelated model classes land within 0.5% of each other.** A multinomial logistic model over
144 causal slots (4,320 parameters, continuous parameter sharing — the thing a count table
structurally cannot do) converges to **1.6886×** the shipped bits/symbol = **192,118.1 B** held-out.
The discrete KT minimum is **193,065.0 B**. The two differ by **946.9 B = 0.49%**, and they share no
mechanism: one is a count table with temporal adaptation and no parameter sharing, the other is a
fitted linear model with parameter sharing and no temporal adaptation. Agreement that close between
complementary blind spots is the strongest evidence in this memo that **~192–193 KB is a property of
the representation, not of either implementation.**

**5. What separates them is NONLINEARITY, not receptive-field area — and that is measured, not
assumed.** The obvious explanation for the shipped network's 113,776.16 B is its large causal field
(read at source, `cpr1/hpac_integer.py` stacks a masked 7×7, a dilated-2 5×5 and a dilated-4 3×3
depthwise pair, plus a 3×3 convolution over the whole previous frame, 64 channels wide — order 25×25
effective, ~9k int8 parameters). **My own sweep refutes that explanation.** Expanding the linear
model's field 2.08× (144 → 299 slots) returned 1,437.7 B of stream for 1,666 B of model — a *net
loss*, so the linear family's own optimum is at SMALL field. Area is available to a linear model and
does not pay. What a linear model cannot buy at any area is the **nonlinearity**, and that is the
whole remaining gap. **The 13,515 B are not overpriced; they are the cheapest part of the subsystem.**

**6. `ddm_hm1` already measured the same thing from the other side and I reproduce its sign.** hm1
measured the *shrink* margin at **−1.15 token bytes per counted model byte** — the shipped model is
marginally worth its bytes. My replacement measures the *far* end of the same curve and finds it far
worse than linear: giving back all 13,515 B costs ~79,000 B, an effective slope of **−5.87**.

---

## 1. The bar, and a correction to the charter's restatement of it

The shipped token subsystem is two counted objects (`ddm_cx3`, reproduced here):

| counted object | bytes |
|---|---:|
| RC64 token stream | 113,777 |
| HPAC model blob (Brotli-coded; decoded object 17,952 B) | 13,515 |
| **counted token subsystem** | **127,292** |

A replacement ships `stream' + W`. So:

    break-even (net > 0)          stream'  <  127,292 - W
    demand-closing (net >= D)     stream' <=  127,292 - W - D,   D = 42,228 B

**The bar I was handed in the charter was wrong, it is now fixed upstream, and this memo cites the
fix.** The charter restated `no1`'s kill number as `stream' < 113,777 - 13,515 + W`, which is
inconsistent with `no1`'s own `r`-table. I derived the disagreement before measuring and raised it;
**MAIN verified it independently at source and landed the correction in commit `4257fa1006`** —
in-place fixes at all three sites plus an append-only `§CORRECTION` quoting the originals verbatim
(Catalog #110/#113). The settled arithmetic:

    Net saving = 13,515 - W + r * 113,777
    r >= (42,228 + W - 13,515) / 113,777   <=>   stream' <= 85,064 - W     (demand-closing)
                                                 stream' <  127,292 - W    (break-even)

which reproduces the published 29.6% / 34.0% / 42.8% at `W` = 5K/10K/20K to four figures. The prose
form differs from the correct bar by **27,030 - 2W** — *stricter* for `W` < 13,515 and *looser* above
it — and the sister line `Net = 13,515 - W - r*113,777` carried the same `r`-sign slip. **Cite `no1`'s
`§CORRECTION`, not the charter's inherited prose form.**

MAIN confirmed my stage-2 accounting already used the corrected bar
(`-67,059.1 = 192,118.1 - (127,292 - 2,233)`), so no verdict here moves. It cost nothing because every
measured value misses **both** bars by a wide margin — but it would have decided the arm had the
result been close, and a falsifier whose kill number is ambiguous is not a falsifier. Every row below
is reported against **both** bars explicitly.

## 2. Recall and non-re-entry — what was already measured, and why row 2 was still live

Four arms bracket this question. I read all four at source before building; two of them changed what
I built.

**`ddm_mi1` does NOT cover table replacement — confirmed at source, and this was the charter's
decisive pre-build check.** mi1's own verdict line reads `verdict_scope:` **"FAMILY — a counted
probability-model packet ADDED to the DX2 token stage"**, and its arithmetic is additive throughout:
*"A paid 10K-int8 model costs 10,000 B … Break-even is 8.9867% of the 111,275.62 B indicator"* — the
packet must **save its own 10,000 B**, with no reclaim of the incumbent. mi1 says so itself: *"My
ladder is the complementary additive formulation (β = 0 nests the shipped model, so it can only
help)."* Its conditioning ceiling (the 2,162.13 B systematic excess at z = +11.89) is likewise a
bound on *offsets applied to the shipped model's output*, which is undefined once that model is
removed. **So mi1's 47.4× row prices a conditioning-adder at fixed model, not a table-replacer, and
row 2 was correctly live.**

**But mi1's conclusion transfers even though its arithmetic does not, and my result is why.** mi1
concluded the model axis supplies nothing. I reach the same place by a different route: not because
the model is too expensive, but because removing it costs ~5.9× more stream than it returns in model
bytes. Both point at the same object; only the mechanism differs.

**`ddm_cx3` is the nearest replacement bracket and it explicitly left this open.** cx3 measured the
replacement formulation at **0 B** (best challenger 125,210 B, 11,433 B worse; hindsight-ideal data
term 117,224 B, 3,447 B worse before model cost). Its verdict_scope, verbatim: *"This is a
`FORMULATION`-scoped negative: the tested causal token summaries and retained learned-predictor
summaries on the pinned DX2 field. It does not kill a differently trained HPAC network, a new
learned representation, or a model that consumes the continuous five-class probability vector rather
than the named summaries."* My arm occupies exactly the gap cx3 named — **raw causal neighbourhoods
and a fitted parametric model, not named summaries** — and finds the same sign, far more strongly:
cx3's best replacement was 1.10× the shipped stream; mine is 1.70×. cx3's challengers were better
than mine because they were allowed to consume the shipped predictor's retained trace; mine are not,
which is what "replacement" means.

**`ddm_hm1` supplies the one quantitative bracket the charter did not name**, and it is the most
useful row in the recall: *"removing 420 model bytes costs 484 token bytes, slope −1.15 — just past
break-even, i.e. the shipped model is marginally worth its bytes at the shrink margin."* hm1 also
states what it does not close: *"a retrained or widened HPAC network. That branch stays open and
`ddm_cl1`'s built-but-never-fired ladder is still the right instrument for it."* That branch is still
open after this arm too (§8), and `cl1` is still its instrument.

**`ddm_ef1`** raced generic estimators (PPMd/ZPAQ, +251,545 B). Those carry no domain prior and no
causal structure; my models carry both, which is why they land at +79,000 B instead of +251,545 B —
the same sign, three times closer, still nowhere near the bar.

**`ddm_rsf1` set the instrument constraint and I honoured it.** rsf1 measured the available `entropy`
rate surrogate **anti-correlated** with shipped bytes (ρ = −0.7235). No number in this memo comes
from it. §3 states what the numbers do come from.

## 3. The instrument is a code length, not a surrogate — and `tb2` proves that is the same thing here

Every byte figure below is `-Σ log2 p(token | causal context)` under an **explicit** probability
model over the exact five-symbol decode alphabet. That is not an entropy estimate of the field; it is
the quantity an arithmetic coder emits.

**`ddm_tb2` measured the size of that identity on this exact stream:** physical RC64 stream 910,216
bits vs the sum of all 117,964,800 selected integer-frequency costs 910,209.280609 bits — an explicit
finite-interval/final-padding residual of **6.719391 bits = 0.839924 B**, inside the allowed `<9`-bit
bound. So on this object code length **is** the physical byte count to better than 1 B in 113,777.

I re-verified both pins before measuring rather than citing them:

| pin | expected | measured this run |
|---|---|---|
| token field sha256 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | **MATCH** |
| token field bytes | 117,964,800 | **MATCH** |
| cost-field bit sum | 910,209.280609 | **910,209.280609** |

and the class histogram reproduces the canonical n600 order and areas independently:
Road 23.2331% · Lane 0.5858% · Undrivable 49.5175% · Movable 1.2380% · MyCar 25.4255%.

**Independent real-coder control.** `real_coder_control` encodes a slice through an actual carry-less
range coder with an adaptive model, then **decodes it back and asserts byte-identity with the source
tokens**. It reports emitted payload bytes against the model's code length, so the coder overhead is
measured on my own coder and not merely inherited from tb2. Result in §5.

## 4. Decode causality — read at source, not assumed

A replacement is only legal if it consumes what the decoder actually has. From
`submissions/robust_current/jg5_sub015_runtime/runtime/cpr1/`:

- `inflate.py:33` — `HPAC_PATCH = 64`, `HPAC_DELTA = 2`.
- `inflate.py:275-284` — `group_masks` iterates `(1 + DELTA) * PATCH - DELTA = 190` groups over a
  grid `column + DELTA * row`.
- `hpac_integer.py:73-84` — `patch_group_mask` keeps a neighbour iff
  `offset = dx + DELTA*dy < 0` (type A), or `<= 0` (type B, on features).

So a position at patch-local `(r, c)` decodes in group **`g = c + 2r`**, `g ∈ [0, 190)`, and the
causally-available set of the current frame is every position in a **strictly smaller group**. I
verified the skew this implies rather than assuming a raster: `(+1,−3)` is **legal** (offset −1) while
`(+1,−2)` is **not** (offset 0, same group). All previous frames are fully available.

**What the incumbent consumes, and therefore what a replacement is allowed.** `IntegerHPAC.forward`
takes `(current, idx, previous_raw)` and nothing else: the partially-decoded current frame, the frame
index, and the previous frame's raw token field, one-hot'd in `prepare_frame_context`. **It never sees
the rendered RGB.** So this is a self-contained autoregressive model over the token field, and a
replacement competes on exactly the same information — which is what makes row 2 a clean coding
question rather than a rendering one.

**One channel the incumbent has that my stage-2 model does not**, stated because it cuts against me:
`frame_embed` is a learned 600×8 per-frame code, so the network carries per-frame parameters while my
logistic model carries none. That channel is **not** unmeasured, though — the stage-1 adaptive models
update their counts every frame and therefore adapt temporally by construction, and they land within
0.5% of the logistic model anyway. The two stages have **complementary blind spots** — stage 1 has
temporal adaptation but no parameter sharing, stage 2 has parameter sharing but no temporal
adaptation — and they agree. That agreement is worth more than either row alone.

**One asymmetry, in my favour, stated openly.** Group order is *global* across all 6×8 patches, so
cross-patch neighbours in earlier groups are legal — but the shipped network convolves inside
**zero-padded patches** (`_to_patches`), so it is structurally blind to them. My models use exact
per-position global group comparison and therefore see **more** legal context than the incumbent
(96.9%–98.4% availability at the boundary offsets, vs 0% cross-patch for the network). The negative
below is measured **with that advantage granted**, which strengthens it.

## 5. Stage 1 — the exact ladder (no split, no seed, no training)

For each rung, `hindsight` is `Σ_c Σ_t N_ct log2(N_c/N_ct)`; `KT` is the exact Krichevsky–Trofimov
prequential code length, achievable with **zero stored bytes** (the decoder rebuilds identical counts
from the prefix it has already decoded, so under contest rule 118 the model is generic algorithm in
`inflate.py` and only the stream is counted).

| k | occupied contexts | hindsight-ideal (B) | KT prequential, W=0 (B) | KT ÷ shipped stream |
|---:|---:|---:|---:|---:|
| 1 | 6 | 1,272,194.5 | 1,272,226.5 | 11.18× |
| 3 | 187 | 243,535.7 | 243,936.1 | 2.144× |
| 5 | 1,178 | 218,653.1 | 219,976.4 | 1.933× |
| 7 | 4,080 | 213,730.3 | 217,226.6 | 1.909× |
| 9 | 9,253 | 191,786.5 | 198,543.3 | 1.745× |
| **11** | **20,388** | **179,605.4** | **193,065.0** | **1.697×** |
| 13 | 32,809 | 176,499.6 | 196,544.6 | 1.727× |
| 15 | 46,252 | 172,556.8 | 199,303.6 | 1.752× |
| 17 | 86,063 | **150,903.2** | 197,929.6 | 1.740× |

**Read the two columns differently, because they mean different things.**

- **KT is the honest, achievable, `W` = 0 number** and it has a **measured interior minimum at
  `k` = 11 (193,065.0 B)**. Past the knee the implicit model cost — the KT-minus-hindsight gap, which
  is exactly the price of *learning* the table — outruns what extra context returns. At `k` = 17 that
  gap is 47,026.4 B.
- **Hindsight keeps falling** and will reach 0 by memorisation as `k` grows, so a low hindsight at
  large `k` proves nothing on its own. Its use is as a **bound**: at any fixed `k` it lower-bounds
  every model with that input.

**Why the hindsight column is a bound and not just a small number.** For a code length
`Σ_i -log2 q(y_i | c_i)` where `q` depends only on the context, the sum splits by context, and within
one context `-Σ_t N_ct log2 q(t|c)` is minimised at `q(t|c) = N_ct/N_c`, giving exactly
`Σ_t N_ct log2(N_c/N_ct)`. So no assignment — fitted, oracle, or hand-chosen — can beat it. It also
covers ADAPTIVE models on the same context, because a prequential code length is never below the
hindsight-optimal one for its own model class (the excess *is* the learning cost, visible here as the
KT-minus-hindsight gap: 47,026.4 B at `k` = 17). The bound binds models whose input is those
neighbours; a model given strictly more information is outside it, which is exactly why §6 exists.

Both miss both bars at every `W`:

| | best KT (W=0) | hindsight-ideal k=17 (W=0) | break-even bar | demand bar |
|---|---:|---:|---:|---:|
| stream bytes | 193,065.0 | 150,903.2 | < 127,292 | ≤ 85,064 |
| shortfall | **+65,773.0** | **+23,611.2** | — | — |

An independently written adaptive context-mixing coder (6 count models, online logistic mixing,
per-frame decode-legal updates) reached **303,233.7 B** over the full 600 frames with its marginal
ratio flattening at **~1.59×** by frame 599 — worse than the best single KT rung because the mixer was
crude, but the same sign and the same order.

**Real-coder control (MEASURED).** 20,000 positions encoded through the carry-less range coder with
an adaptive model, then decoded back: **`decode_byte_identical = True`**, code length **38.734189 B**,
emitted payload **43 B**, difference **4.265811 B**, payload sha256 `324aa001…`. The gap is the
coder's fixed 4-byte final flush plus sub-byte rounding — an **additive constant, not a multiplicative
factor** — which is the property the instrument needs. On the full stream the same identity is tb2's
measured **0.839924 B in 113,777 B**. Scope: the slice is contiguous and therefore locally
low-entropy, so it samples the coder's *correctness*, not the field's diversity; the field-wide
identity is tb2's, not mine.

## 6. Stage 2 — a learned parametric model with a large receptive field

Stage 1 cannot close the family by itself, and saying it did would be a negative-existence overclaim:
a count table cannot represent a large receptive field at all (contexts are `6**k`), and that is
precisely where the shipped network's power lives. Stage 2 supplies the missing class — a multinomial
logistic model over one-hot causal slots,

    logit[k] = bias[k] + Σ_j Wt[slot_j, value_j, k]

whose parameter count is **linear** in receptive-field area rather than exponential, trained on a
temporally-stratified split and serialised so `W` is a real coded byte count.

**Split:** 20 contiguous blocks of 30 frames; test = every 4th block (blocks 1, 5, 9, 13, 17) = 150
frames = 25%. Contiguous blocks, **never a prefix** — `ddm_bp2` measured that a prefix of this
population is a different population. Blocks keep adjacent-frame correlation on one side of the split
instead of leaking it across, which an interleaved split would not do. Deterministic, no RNG in the
split itself; sampling seed 20260826 recorded.

**One measured implementation defect, fixed and recorded.** The first fit used dense Adam and
**diverged** (train 0.032574 → 0.048659 → 0.098408 bits/sym). The features are one-hot, so a weight
row receives a gradient only on steps where its `(slot, value)` occurs, while Adam's moment buffers
decay on *every* step — rare rows accumulate a near-zero `v` and then take an enormous `lr/√v` step.
Switching to **Adagrad**, whose accumulator grows only where the gradient is nonzero (so the dense
update is exactly the sparse one), restored monotone convergence. The objective is convex, so an
undertrained fit would give only an *upper* bound and a negative drawn from it would be weak; the full
per-epoch trajectory is recorded in every stage-2 receipt so the plateau is auditable rather than
asserted.

### 6.1 The sweep

| config | slots | params | wd | `W` (B) | final train b/s | held-out b/s | ratio | stream (B) | total (B) | net (B) | gen gap | convergence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `rf144` | 144 | 4,320 | 1e-06 | 2,233 | 0.012289 | 0.013029 | 1.6886× | 192,118.1 | 194,351.1 | -67,059.1 | -0.000110 | converged |
| `rf299` | 299 | 8,970 | 1e-06 | 3,899 | 0.012130 | 0.012931 | 1.6759× | 190,680.4 | 194,579.4 | -67,287.4 | +0.000413 | converged |
| `rf618` | 618 | 18,540 | 1e-06 | 7,322 | 0.012508 | 0.016182 | 2.0972× | 238,610.2 | 245,932.2 | -118,640.2 | +0.002372 | **UNDER-OPTIMIZED (provable)** |
| `rf1052` | 1052 | 31,560 | 1e-06 | 11,857 | 0.014724 | 0.023005 | 2.9815× | 339,222.7 | 351,079.7 | -223,787.7 | -0.002038 | **UNDER-OPTIMIZED (provable)** |
| `rf618wd0.0003` | 618 | 18,540 | 3e-04 | 5,402 | 0.013550 | 0.014177 | 1.8373× | 209,043.0 | 214,445.0 | -87,153.0 | +0.000190 | **NOT TESTABLE** |

- `rf618`: UNDER-OPTIMIZED (provable) -- final train 0.012508 > 0.012130 of a strictly nested smaller model at the same weight decay
- `rf1052`: UNDER-OPTIMIZED (provable) -- final train 0.014724 > 0.012130 of a strictly nested smaller model at the same weight decay
- `rf618wd0.0003`: NOT TESTABLE -- sole member of its regularisation group; nested monotonicity does not apply and train loss is not comparable across weight decays

**Convergence is tested, not asserted, and the test is rigorous.** The slot sets are strictly nested
(`rf144 ⊂ rf299 ⊂ rf618 ⊂ rf1052`), so at the optimum the final TRAIN loss must be non-increasing in
model size: a larger model can always zero its extra weights and reproduce a smaller one exactly. Two
rows violate that and are therefore **provably not at their optimum** — their held-out numbers are
UPPER BOUNDS and are **not** evidence about receptive-field saturation in either direction. The
`wd=3e-4` row is the sole member of its regularisation group, so the test cannot be applied to it
(train loss is not comparable across weight decays, since regularisation legitimately raises it).

**Therefore the only load-bearing rows are `rf144` and `rf299`, which both pass**, and they say:
expanding the receptive field 2.08× bought **0.75%** of held-out bits/symbol (0.013029 → 0.012931 =
1,437.7 B of stream) while `W` grew by 1,666 B — **a net loss of 228 B**. The linear family's optimum
is at SMALL receptive field.

**A `wd=3e-3` rung was launched and deliberately stopped**, not lost: it would again have been the
sole member of its regularisation group and so untestable by the monotonicity criterion, and the RF
question is not load-bearing — closing the gap from `rf299` would require a **36%** improvement from
doubling the area when doubling it actually bought **0.75%**. Recorded here rather than silently
dropped.

### 6.2 The per-`W` table, and why the `W` grid is moot

The charter pre-registered `W` ∈ {5K, 10K, 20K}. Both bars **tighten** as `W` grows, so **`W` = 0 is
the most favourable capacity in the whole family** — and the zero-stored-byte adaptive model is a
real, legal member of it (contest rule 118: counts rebuilt at decode time from the already-decoded
prefix are generic algorithm in `inflate.py`, not counted payload). **The family therefore fails at
every `W` because it fails at `W` = 0**, and no larger capacity can rescue it.

| `W` (B) | break-even: `stream' <` | demand-closing: `stream' ≤` | best measured stream at ≈ this `W` | clears break-even | clears demand |
|---:|---:|---:|---|---|---|
| 0 | 127,292 | 85,064 | **193,065.0** — KT `k`=11, achievable | **No**, +65,773.0 | No |
| 0 *(oracle, unachievable)* | 127,292 | 85,064 | **150,903.2** — hindsight `k`=17 | **No**, +23,611.2 | No |
| 2,233 | 125,059 | 82,831 | 192,118.1 — logistic rf144 | **No**, +67,059.1 | No |
| 3,899 | 123,393 | 81,165 | 190,680.4 — logistic rf299 | **No**, +67,287.4 | No |
| 5,000 | 122,292 | 80,064 | no config lands here; bar is stricter than `W`=0 | No | No |
| 10,000 | 117,292 | 75,064 | no config lands here; bar is stricter than `W`=0 | No | No |
| 20,000 | 107,292 | 65,064 | no config lands here; bar is stricter than `W`=0 | No | No |

The **loosest bar anywhere in the table is 127,292 B** (break-even at `W` = 0). The **best number any
model of either class reached, at any capacity, with zero model cost and zero generalisation gap, is
150,903.2 B.** The gap is 23,611.2 B and it is a lower bound, so it cannot be closed by better
fitting, more capacity, or more training.

## 7. The bound is ALPHABET-CONDITIONAL — and which merge you pick matters 2.6× (MAIN's question (b), measured not argued)

MAIN asked whether the 150,903.2 B lower bound says anything about a field-side alphabet change, or
whether it is alphabet-conditional. **It is alphabet-conditional, and strongly.** Rather than reason
about it, I measured it: the instrument takes any field definition, so a merged alphabet costs ~10 s.
The whole field is relabelled before contexts are built, because a decoder only ever sees merged
tokens; the KT alphabet size drops to the live one, so no cost is charged for a symbol that cannot
occur.

| field | class merged | % of field | contexts (k=11) | hindsight (B) | **KT, W=0 (B)** |
|---|---|---:|---:|---:|---:|
| none — the 5-class field §5 measured | — | — | 20,388 | 179,605.4 | 193,065.0 |
| **Lane → Road** | Lane | 0.5858% | 8,943 | **67,865.3** | **73,061.0** |
| Lane → Undrivable | Lane | 0.5858% | 15,379 | 182,546.7 | 191,438.3 |
| Movable → Undrivable | Movable | 1.2380% | 12,565 | 166,887.9 | 174,223.6 |

**Two findings, and the second is the one worth carrying.**

**(1) The §9 closure does not transfer across an alphabet change.** The same crude discrete-context
model that costs 193,065.0 B on the 5-class field costs **73,061.0 B** on the Lane→Road field — a
**2.64× reduction**, and on the corrected bars it **clears break-even by 54,231 B and the
demand-closing bar by 12,003 B at `W` = 0**. The hindsight bound moves 150,903.2 → 58,092.5 B
(at `k` = 17). So §9's negative is scoped to the **5-class field** and says nothing about a merged one.

**(2) The driver is SPATIAL EMBEDDING, not class rarity and not class difficulty.** Removing the
*same* class gives 73,061.0 B (into Road) or 191,438.3 B (into Undrivable) — **2.62× apart, from an
identical reduction in alphabet size and an identical 0.5858% of field relabelled.** And merging a
*larger*, also-difficult class (Movable, 1.2380%) buys only 9.8%. Lane markings sit **on** the road,
so Lane→Road collapses an enormous amount of codim-1 boundary and leaves a spatially homogeneous
region; Lane→Undrivable puts the relabelled pixels in maximal disagreement with everything around
them and buys essentially nothing. **The selection criterion for a merge is which class the target is
spatially EMBEDDED IN — a wrong pick costs 2.6× and looks identical on every summary statistic the
campaign currently tracks (area, IoU, flip share, bit share).**

This also explains the gestalt's [[m131]] line mechanically: Lane holds 33.56% of model bits at 0.59%
of area **because of its boundary with Road**, and the merge that deletes that boundary is the one
that returns the bits.

**Reconciliation with `ddm_ld1`, which measured Lane→Road and got the opposite sign.** `ld1` tested
Lane→Road as a **field edit under the FROZEN model** and every rung *enlarged* the archive
(+21…+1,528 B). There is no conflict: a frozen 5-class model fed a relabelled field predicts worse,
so the stream grows. My rows are the code length under a model **refit to the merged field**. This is
exactly why `no1` row 3 states that D3 **requires a retrain, which is an object change** — and it is
now measured rather than asserted.

**Scope, stated hard, because this row is the one most likely to be over-read:**

- **RATE SIDE ONLY. No scorer ran. `d_seg` is NOT measured here and no row above is admissible as a
  distortion measurement.** A merge is **not lossless** — unlike everything else in this memo.
- The merge relabels **691,095 positions = 0.5858%** of the field. That is a **field-level
  disagreement count**, not a `d_seg`, and the mapping from one to the other is the binding risk of
  row 3 and is unmeasured by this arm. Whether the rate win survives it is **exactly the open
  question**, and nothing here should be read as saying it does.
- The merged-field numbers come from my **crude** model class. Do **not** rescale them by my 1.697×
  deficit to project a shipped-quality model on the merged field — that is a cross-regime constant
  transfer ([[m143]]) across a changed object, and it is not licensed by anything measured here.

## 8. Payload custody

All persisted under `/Volumes/APDataStore/pact/ddm_lm1_learned_model_falsifier/` (Vertigo is at 8.4
GiB / 100% capacity and was not used):

| artifact | bytes | sha256 (first 16) |
|---|---:|---|
| `RESULT_stage2_rf1052.json` | 2,774 | `5d833ebbaafd8ddf…` |
| `RESULT_stage2_rf144.json` | 2,924 | `b097d49291328c83…` |
| `RESULT_stage2_rf299.json` | 2,871 | `d3b7fd17499b1028…` |
| `RESULT_stage2_rf618.json` | 2,821 | `7a88b321b219ddd9…` |
| `RESULT_stage2_rf618wd0.0003.json` | 3,027 | `6de862cca9e7ae0e…` |
| `ladder_insurance/RESULT.json` | 10,604 | `7f150817e33e9125…` |
| `ladder_insurance/retained/real_coder_control.bin` | 43 | `324aa001175905dd…` |
| `merge_1_2/RESULT.json` | 6,601 | `03dc0c6c6c0a8b8e…` |
| `merge_3_2/RESULT.json` | 6,597 | `30df0633afb41c22…` |
| `merge_lane_into_road/RESULT.json` | 7,184 | `9f1d8889b28d1d80…` |
| `retained/model_rf1052.bin` | 11,857 | `32c86337fca14e40…` |
| `retained/model_rf144.bin` | 2,233 | `440fe67a74132517…` |
| `retained/model_rf299.bin` | 3,899 | `0947278dedbfec1d…` |
| `retained/model_rf618.bin` | 7,322 | `f52e2f2f007274a3…` |
| `retained/model_rf618wd0.0003.bin` | 5,402 | `628c30748eb62757…` |
| `retained/weights_rf1052.npz` | 16,068 | `8983f60db0e028ef…` |
| `retained/weights_rf144.npz` | 3,187 | `b146647e5b4063c9…` |
| `retained/weights_rf299.npz` | 5,417 | `b547bea720268dcd…` |
| `retained/weights_rf618.npz` | 9,936 | `4bb0b4090807ff87…` |
| `retained/weights_rf618wd0.0003.npz` | 8,218 | `45246cfcda9a9b24…` |

Working copies and run logs (not custody, but re-derivable provenance):
`.omx/tmp/ddm_lm1/{full,sweep2,insurance,rf618fix}/run.log`. The `full` log records the complete
sha256 verification of the token field (`token sha OK`, run WITHOUT `--skip-sha`) before that run was
deliberately stopped; `ladder_insurance/RESULT.json` is the canonical stage-1 receipt.

Inputs read-only and re-hashed this run:
`/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/retained/fields/decoded_tokens_instrumented.u8`
— 117,964,800 B, sha256 `cc10a7b0…` (**matches the `ddm_tba1` / `ddm_no1` charter pin**) ·
`position_rc64_frequency_cost_bits.f64le.bin` — 943,718,400 B, summing to 910,209.280609 bits.

## 9. Verdict scope — what this closes and what it does not

**VERDICT: REFUSED at measured scope.** `verdict_scope:` **FORMULATION — replacement of the DX2
13,515 B HPAC network by (a) adaptive or static DISCRETE-CONTEXT models over causal token
neighbourhoods, and (b) LINEAR/logistic parametric models with continuous parameter sharing over
causal receptive fields, on the pinned field `cc10a7b0…`, n600, 117,964,800 positions.**

**Closed by this arm, and the strength of each:**

1. **Discrete-context models — closed by a rigorous bound, not by my implementation.** The
   hindsight-ideal at `k` = 17 (150,903.2 B) lower-bounds *every* model taking those neighbours as
   input, at unlimited capacity and zero model cost. It exceeds the `W` = 0 break-even bar by
   23,611.2 B. The achievable `W` = 0 number (193,065.0 B) additionally has a measured interior
   minimum, so the family is closed from both directions.
2. **Linear parametric models — closed by measurement, with an interior optimum.** Expanding the
   receptive field 2.08× (144 → 299 slots) returned 1,437.7 B of stream for 1,666 B of model: a **net
   loss**. The family's own optimum sits at small receptive field and still misses break-even by
   ~67,000 B.

**NOT closed, and I will not claim it is:**

- **A NONLINEAR learned network** — an MLP, a conv stack, or a differently-architected HPAC. This is
  the *incumbent's own class*, and nothing here bounds it. It is precisely the branch `ddm_hm1` named
  as open (*"a retrained or widened HPAC network. That branch stays open and `ddm_cl1`'s
  built-but-never-fired ladder is still the right instrument for it"*) and `ddm_cx3` named as open
  (*"a differently trained HPAC network, a new learned representation"*). It stays open after this
  arm too. **What this arm does is change its prior, hard:** at a receptive field matched to the
  incumbent's and roughly double its parameters, the best linear model is still ~1.6× its code
  length, so the entire remaining gap is attributable to **nonlinearity**, not to capacity, not to
  context, and not to the incumbent being overpriced.
- **A model consuming something other than the token field** — e.g. the rendered RGB, or the
  continuous five-class probability vector. `cx3` named that too. Out of scope here.
- **Anything about `d_seg`, `d_pose` or the pointer.** Lossless by construction; no scorer ran; no
  row here is admissible as a distortion measurement.

**Negative-existence scope (m53).** I did not search the space of possible models. I searched two
named classes and measured them. "No learned model can do this" is **not** what this memo says.

## 10. GESTALT-DELTA

**Before:** the gestalt ([[m144]]) held that dx2's lossless remainder is ≈2,009 B = 4.8% of demand,
**all of it on the model axis**, with the coder axis closed at 0 B (`jt23`) — and therefore that
sub-0.12 needs a different object.

**After, and this sharpens rather than contradicts it:** the 2,009 B model-axis remainder is a
**refinement-at-fixed-model** quantity — it is what conditioning or recalibration can scrape off the
incumbent's output. It is **not** a representation-replacement budget. Measured from the replacement
side, the model axis does not hold +2,009 B of headroom; it holds **−79,288 B**. Removing the model
costs 5.87× more stream than the model itself is worth.

**The new law this arm adds: the token/model exchange curve is CONVEX, and the shipped model sits
just past its own knee.** `ddm_hm1` measured the local slope at the shrink margin as **−1.15** token
bytes per counted model byte. This arm measures the global slope all the way to zero as **−5.87**.
The curve steepens **5.1×** between the margin and the origin, so there is no capacity at which
giving model bytes back pays. Both ends of that curve are now measured, by two arms, in agreement.

**The consequence for where the campaign points: change the FIELD, not the MODEL — and §7 measures
that the field is where the compressibility actually lives.** The 113,777 B stream is not
compressible by a better probability model over the same 117,964,800 five-class symbols; two model
classes and a rigorous oracle bound all say so. But the **same crude model** that costs 193,065.0 B
on the 5-class field costs **73,061.0 B** on a Lane→Road field — it goes from missing break-even by
65,773 B to clearing the **demand** bar by 12,003 B, without becoming one byte smarter. **The
compressibility was never in the model. It was in the alphabet.**

**And the new selection law §7 adds: the merge criterion is SPATIAL EMBEDDING, and a wrong pick costs
2.6×.** Lane→Road returns 73,061.0 B; Lane→Undrivable returns 191,438.3 B — same class, same 0.5858%
relabelled, same alphabet reduction, **2.62× apart**. Merging a larger and equally difficult class
(Movable, 1.2380%) buys only 9.8%. None of area, IoU, flip share or bit share — the four statistics
the campaign currently tracks — distinguishes the good merge from the useless one. **What
distinguishes them is which class the target is spatially embedded in.** That is a new, cheap,
transferable selection rule for `no1` row 3, and it did not exist before this arm.

**The caveat that governs all of it:** §7 is rate-side only. The merge relabels 691,095 positions and
its `d_seg` consequence is **unmeasured here**. The rate half is now priced; the distortion half is
row 3's binding risk and remains open.

## 11. NEXT_IF_RESUMED

- **CLOSED — row 2, the learned probability model, at the two classes measured.** Owner: none. Do not
  charter a discrete-context or linear replacement for the DX2 token stream; refuse any such charter
  with §5's ladder and §6's sweep. Reactivation only via a **nonlinear** architecture, and only with
  the §8 prior stated in the charter.
- **REUSABLE SCREEN — the apparatus contribution, and the thing that would have pre-empted this
  build.** `experiments/ddm_lm1_learned_model_falsifier.py::score_rung` takes **any** context
  definition and returns the exact hindsight-ideal and the exact KT prequential code length in
  **~35 s** for the whole ladder, with no training, no split and no seed. Any future arm proposing
  "a better model on the token field" should run it FIRST: it prices the proposal's ceiling before a
  line of model code is written. Owner: next arm touching the token stream. Fire condition: immediate,
  it is free.
- **QUEUED-WITH-A-FIRE-ORDER, and now RATE-PRICED — `no1` row 3 (`tba1`-D3 alphabet merge).** Owner:
  MAIN / next slot. This arm re-aims the queue at it and supplies two things its charter did not have:
  (i) a **measured rate side** — a Lane→Road merge takes the same crude model from 193,065.0 B to
  **73,061.0 B**, clearing the demand bar by 12,003 B at `W` = 0, with the hindsight bound moving
  150,903.2 → 58,092.5 B; (ii) a **selection rule** — merge into the class the target is spatially
  EMBEDDED in (Lane→Road 73,061.0 B vs Lane→Undrivable 191,438.3 B, **2.62× apart on an identical
  alphabet reduction**), which no area/IoU/flip-share statistic predicts.
  **The charter must carry the distortion half as its binding risk**: the merge relabels 691,095
  positions (0.5858%), no scorer ran here, and whether the rate win survives `d_seg` is exactly the
  unmeasured question. It must also state that `ld1`'s opposite-signed result is not a contradiction —
  `ld1` edited the field under a FROZEN model (archive grew +21…+1,528 B) whereas the win requires a
  REFIT, which is why `no1` calls D3 an object change. Per MAIN's #1236 recall, the alphabet is not a
  parameter anywhere in the HPAC stack, so D3 is a **mechanism build**, not a config change.
- **DONE — `no1` row 2's kill-number correction.** Raised by this arm, verified independently at
  source by MAIN, landed as commit **`4257fa1006`**: three in-place bar fixes plus an append-only
  `§CORRECTION` quoting the originals. Discrepancy characterised as `27,030 - 2W`. No verdict anywhere
  changed. Nothing further owed; future arms cite `no1 §CORRECTION`.
- **DEFER, with a worsened prior — `ddm_cl1`'s nonlinear retrain ladder.** Owner: unchanged (`cl1`,
  currently BLOCKED, never fired). It remains the right instrument for the one open sub-family, but
  this arm plus `hm1` now bracket it from both ends of a convex curve whose knee is already occupied.
  A charter that fires it should state the §10 slope law and say what it expects to beat.

## 12. Own-vehicle frontier

**gb1 — S 0.14811799921260607 @ 180,215 B `[contest-CUDA T4, n600]`, archive sha `ba1f3830…` —
UNMOVED by this arm.** Gap to 0.12 = 0.028118 ⇒ shed **42,228 B** at fixed distortion (target archive
**≤ 137,986 B**), or 150 B at zero distortion.

---

## ADDENDUM (pf2x r88, 2026-08-27) — formalization-track disposition

# FORMALIZATION_PENDING:token/model exchange-curve convexity law (slope −1.15 at the shrink margin per ddm_hm1, −5.87 to the origin per this arm, 5.1× steepening ⇒ shipped model past its own knee) is a genuine two-anchor law owed a canonical_equations registration; flagged at the r88 Catalog #344 re-baseline per the 2026-07-09 precedent's direct-footer discipline
