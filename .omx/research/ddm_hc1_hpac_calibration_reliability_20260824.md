# ddm_hc1 — the HPAC model is OVER-confident, not under-confident; perfect recalibration releases **8.44 heldout net bytes** at zero distortion, and the token stream is 97.80% one binary question

**Date:** 2026-08-24 · **Arm:** `ddm_hc1` · **Pointer:** UNMOVED · **No Modal job fired. No scorer fired. $0.**
**Axis:** `[macOS-CPU advisory / scorer-free shipped-receiver instrumentation]`.
`score_claim=false` · `promotable=false` · no archive built, none promoted.
**verdict_scope:** `FAMILY: recalibration of the shipped DX2 HPAC coding row` — see §10 for the
exact boundary of what that does and does not close.

---

## 0. Result first — the three lines the charter asked for

1. **ECE sign: OVER-CONFIDENT.** Signed ECE (empirical accuracy − stated `pmax`) =
   **−1.9289131e-05**, identical at 10 / 15 / 32 / 64 / 256 equal-mass bins. Aggregate stated
   confidence **0.9980892982**, aggregate empirical accuracy **0.9980700090**. The model claims
   slightly more than it delivers.
2. **Heldout net bytes: +8.44 B** (best rung, 3-seed spread 8.376 / 8.436 / 8.497). The best
   *domain-native* rung nets **−14.57 B**. Best gross before the map's own bytes: **+40.91 B**.
3. **Falsifier outcome: FAMILY CLOSED.** 8.44 B against the pre-registered 2,000 B threshold —
   **237× below it.**

**MAIN's premise is refuted outright, and its size estimate by 2,055×.** The charter hypothesised
under-confidence on already-correct positions worth **17,337 B**. The model is over-confident, and
the whole recalibration family is worth **8.44 B = 0.0199% of the 42,381.16 B demand**
(ΔS = 5.617e-06). The charter said to say so early if I found MAIN wrong. I did, and this is it.

---

## 1. The correction MAIN needs most — its `d` statistics are ACAUSAL

The charter's motivating numbers are **`d=0` is 2.17% of positions carrying 94.53% of the bits**,
`d≥4` is 92.80% / ~1.86%, annulus 5.04% / 3.61%. I reproduce **all four to 4 significant figures** —
but only when `d` is taken from **the frame being coded**. That field is not available to the
receiver, which has not decoded that frame yet.

The shipped decoder's own feature is `_boundary_buckets(frame f−1)`
(`runtime/residual_archive.py:531`, fed to the corrector as `feature = d*5 + predicted`). On that
causally-legal field the concentration is far weaker (MEASURED, `DCOMPARE.json`):

| `d` | positions (causal) | **bits (causal)** | bytes (causal) | positions (acausal) | **bits (acausal)** | bytes (acausal) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2.1629% | **49.8485%** | 56,715.7 | 2.1665% | **94.5323%** | 107,555.2 |
| 1 | 1.8052% | 15.6849% | 17,845.7 | 1.8082% | 2.1880% | 2,489.4 |
| 2 | 1.6635% | 7.0626% | 8,035.6 | 1.6664% | 0.8557% | 973.6 |
| 3 | 1.5595% | 4.0903% | 4,653.8 | 1.5622% | 0.5689% | 647.3 |
| 4 | 92.8090% | **23.3136%** | 26,525.3 | 92.7967% | **1.8550%** | 2,110.6 |

**The position geometry is the same to 3 significant figures; the join to COST is not.** `d=0`'s bit
share falls **1.896×** (94.53% → 49.85%) and `d≥4`'s rises **12.6×** (1.86% → 23.31%) the moment the
variable is made receiver-computable. A quarter of the stream's bytes sit four or more pixels from
any boundary the receiver can see.

This is not a quibble about which is "right" — both are real fields. It is that **any lever priced
on the acausal concentration is priced on information the decoder does not have.** The campaign's
boundary/annulus work should state which field it means. Retained as
`retained/per_d_reliability_table.json` and `analysis/DCOMPARE.json` for that purpose.

---

## 2. What the token stream actually spends its bytes on

The df1 decomposition is lossless and exact:

```
-log2(p_sel) = -log2(pmax)                            if the receiver's argmax is right
             = -log2(1-pmax) + -log2(p_sel/(1-pmax))  if it is wrong
```

MEASURED over all 117,964,800 positions of the DX2 body:

| term | bytes | share of the 113,776.16 B tail |
|---|---:|---:|
| **INDICATOR — "is my argmax right?"** | **111,275.62** | **97.80%** |
|  · "yes" branch — confirmation entropy | 34,674.08 | 30.48% |
|  · "no" branch — wrongness | 76,601.54 | 67.33% |
| **CONDITIONAL — "then which of the other four?"** | **2,500.54** | **2.20%** |

Per flip: **2.6917 bits** to say *the argmax is wrong* and **0.0879 bits** to say *which class
instead* — 2.7796 bits total, matching df1's independently-measured 2.7795.

**The token half of this archive is, to 97.80%, one binary question asked 117.96 million times.**
That is why calibration was the right thing to test, and it is also why the answer is so small: a
binary sub-code is exactly the object a reliability diagram governs, and this one is nearly right.

---

## 3. The reliability diagram

32 equal-mass bins over the non-saturated positions, plus the float32-saturated cell as its own
terminal bin (it is a mass point holding 57.6% of positions and cannot be split by any binning).
Gap = empirical accuracy − stated `pmax`; negative is over-confident.

| bin | positions | flips | stated `pmax` | empirical | **gap** | indicator B |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 352,969 | 126,290 | 0.641670051 | 0.642206539 | **+5.365e-04** | 40,541.75 |
| 1 | 350,227 | 52,097 | 0.853059852 | 0.851247905 | **−1.812e-03** | 26,182.39 |
| 2 | 349,730 | 21,005 | 0.941079097 | 0.939939382 | −1.140e-03 | 14,215.41 |
| 3 | 351,411 | 9,790 | 0.972777576 | 0.972140883 | −6.367e-04 | 8,026.28 |
| 4 | 350,764 | 4,913 | 0.986122857 | 0.985993431 | −1.294e-04 | 4,646.38 |
| 5 | 350,133 | 2,785 | 0.991901472 | 0.992045880 | +1.444e-04 | 2,921.75 |
| 6 | 350,813 | 1,843 | 0.994633546 | 0.994746489 | +1.129e-04 | 2,075.14 |
| 7 | 349,548 | 1,373 | 0.996139049 | 0.996072070 | −6.698e-05 | 1,617.86 |
| 8 | 351,243 | 1,068 | 0.997098238 | 0.996959370 | −1.389e-04 | 1,308.86 |
| 9 | 351,694 | 846 | 0.997758356 | 0.997594500 | −1.639e-04 | 1,072.49 |
| 10 | 352,491 | 733 | 0.998231081 | 0.997920514 | −3.106e-04 | 950.07 |
| 11 | 350,690 | 585 | 0.998588371 | 0.998331860 | −2.565e-04 | 780.82 |
| 12 | 352,106 | 533 | 0.998851477 | 0.998486251 | −3.652e-04 | 723.50 |
| 13 | 350,776 | 453 | 0.999047590 | 0.998708578 | −3.390e-04 | 628.47 |
| 14 | 350,050 | 397 | 0.999204909 | 0.998865876 | −3.390e-04 | 560.99 |
| 15 | 351,562 | 311 | 0.999328724 | 0.999115377 | −2.133e-04 | 452.17 |
| 16 | 348,954 | 240 | 0.999424715 | 0.999312230 | −1.125e-04 | 359.23 |
| 17 | 351,728 | 229 | 0.999501711 | 0.999348929 | −1.528e-04 | 345.62 |
| 18 | 351,023 | 187 | 0.999566977 | 0.999467271 | −9.971e-05 | 288.56 |
| 19 | 349,495 | 192 | 0.999622709 | 0.999450636 | −1.721e-04 | 296.76 |
| 20 | 350,269 | 140 | 0.999668478 | 0.999600307 | −6.817e-05 | 223.16 |
| 21 | 351,314 | 114 | 0.999706338 | 0.999675504 | −3.083e-05 | 185.73 |
| 22 | 354,766 | 99 | 0.999739211 | 0.999720943 | −1.827e-05 | 164.00 |
| 23 | 376,038 | 120 | 0.999768905 | 0.999680883 | −8.802e-05 | 196.85 |
| 24 | 452,470 | 124 | 0.999796415 | 0.999725949 | −7.047e-05 | 206.57 |
| 25 | 562,443 | 127 | 0.999823532 | 0.999774199 | −4.933e-05 | 215.80 |
| 26 | 703,037 | 118 | 0.999850826 | 0.999832157 | −1.867e-05 | 206.57 |
| 27 | 1,015,145 | 142 | 0.999878382 | 0.999860119 | −1.826e-05 | 253.33 |
| 28 | 1,789,377 | 195 | 0.999905885 | 0.999891024 | −1.486e-05 | 356.64 |
| 29 | 2,316,967 | 193 | 0.999931710 | 0.999916701 | −1.501e-05 | 362.35 |
| 30 | 4,117,240 | 172 | 0.999961498 | 0.999958224 | −3.274e-06 | 344.02 |
| 31 | 30,602,648 | 257 | 0.999993356 | 0.999991602 | −1.753e-06 | 566.07 |
| **32 (saturated)** | **67,955,679** | **0** | 1.000000000 | 1.000000000 | **+0.000e+00** | **0.00** |

**29 of 32 finite bins are over-confident.** Only bins 0, 5 and 6 run the other way. The
float32-saturated cell asserts certainty across 67,955,679 positions and is **never once wrong** —
perfectly calibrated as far as 118 million samples can tell, and it already costs 0 bits.

ECE by bin count (MEASURED): absolute **1.929e-05** (10 bins) → 2.089e-05 (15) → 2.403e-05 (32) →
2.652e-05 (64) → 3.640e-05 (256). The signed value is **−1.9289131e-05 at every bin count**, because
it is just the aggregate gap and does not depend on the binning.

**Why such small gaps cannot become bytes.** The excess cost of miscalibration is a KL divergence,
which is *quadratic* in the gap: `≈ gap² / (2·p(1−p)·ln2)` bits per position. Bin 1 carries the
largest gap in the table, 1.812e-03, over 350,227 positions — and yields under a byte, because
`p(1−p) = 0.1253` sits in the denominator. The measured total works out to **8.4e-07 bits per
position**, which is where the whole answer comes from.

---

## 4. Does the model already see `d`? — the load-bearing question

Per-`d` aggregate (MEASURED, causal field):

| `d` | positions | share | flip rate | stated `pmax` | empirical | **gap** | bytes | bit share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,551,462 | 2.1629% | 4.769e-02 | 0.95253495 | 0.95230617 | **−2.288e-04** | 56,715.7 | 49.85% |
| 1 | 2,129,461 | 1.8052% | 1.607e-02 | 0.98408334 | 0.98392786 | −1.555e-04 | 17,845.7 | 15.68% |
| 2 | 1,962,333 | 1.6635% | 7.481e-03 | 0.99255806 | 0.99251860 | −3.946e-05 | 8,035.6 | 7.06% |
| 3 | 1,839,635 | 1.5595% | 4.606e-03 | 0.99533877 | 0.99539420 | **+5.542e-05** | 4,653.8 | 4.09% |
| 4 | 109,481,909 | 92.8090% | 4.439e-04 | 0.99956872 | 0.99955606 | −1.265e-05 | 26,525.3 | 23.31% |

**The curves do separate — and neither of MAIN's two branches is the right description.** The gaps
differ by an order of magnitude across `d` and `d=3` flips sign. Per-`(d, confidence)` cells separate
harder still: at the lowest confidence bin the gap runs from **−1.380e-03** at `d=0` to **+5.128e-03**
at `d=3` — over-confident against the boundary, under-confident in the annulus. So the shipped
corrector's `d*5 + predicted` table has **not** fully absorbed the variable; residual structure is
there and it is measurable.

**It is also economically worthless.** Conditioning the map on `d` buys, heldout and gross:
**+16.83 / +12.07 / +13.41 B** across three seeds at 8 confidence bins, **+0.79 / −2.62 / +0.58 B** at
32, and **−41.95 / −130.84 B** beyond that. Conditioning the 2-parameter Platt map on `d` buys
**+7.26 / +5.85 / +7.16 B**. Every one of those is smaller than the bytes needed to ship the extra
parameters.

The honest verdict is neither "flat" nor "separates, therefore headroom": **the curves separate, the
separation is real, and it is worth single-digit bytes.** MAIN's gate for the next conditioning
variable — class-pair identity, decoder margin — was "only if `d` separates". It separated by sign
but not by size, so I did **not** sweep them. Sweeping a second free variable to chase ~15 gross bytes
against a 20 B parameter cost would be the same trade one level down.

---

## 5. The ladder — heldout, in-sample, and what the map itself costs

Map family `q = sigmoid(a_c · logit(pmax) + b_c)`, which **nests the identity** at `(a,b) = (1,0)`.
2-fold cross-fitted over a seeded random split of all 117,964,800 positions (**never a prefix** —
`[[m88]]`/`[[m96]]`). Map bytes = 4 B per stored parameter plus 4 B per stored bin edge.

| rung | kind | params | heldout B | in-sample B | map B | **net B** |
|---|---|---:|---:|---:|---:|---:|
| global temperature | **generic CONTROL** | 1 | 12.44 | 12.54 | 4 | **+8.44** |
| global Platt | **generic CONTROL** | 2 | 14.21 | 14.32 | 8 | +6.21 |
| per-`d` offset | domain-native | 5 | 5.43 | 5.92 | 20 | −14.57 |
| per-`d` Platt | domain-native | 10 | 21.47 | 23.58 | 40 | −18.53 |
| per-bin(8) offset | pmax only | 8 | 24.08 | 25.87 | 60 | −35.92 |
| **per-`d` × bin(8) offset** | **domain-native** | 40 | **40.91** | 50.52 | 188 | −147.09 |
| per-bin(32) offset | pmax only | 32 | 22.65 | 31.58 | 252 | −229.35 |
| per-`d` × bin(32) offset | domain-native | 160 | 23.44 | 75.65 | 764 | −740.56 |
| per-bin(128) offset | pmax only | 128 | 7.42 | 54.66 | 1,020 | −1,012.58 |
| per-`d` × bin(128) offset | domain-native | 640 | −34.52 | 159.69 | 3,068 | −3,102.52 |
| per-bin(512) offset | pmax only | 512 | −44.80 | 128.88 | 4,092 | −4,136.80 |
| per-`d` × bin(512) offset | domain-native | 2,560 | −175.64 | 403.46 | 12,284 | −12,459.64 |

Three things this table says.

**The oracle is 403 B and it is a mirage.** In-sample gain climbs monotonically with cell count —
12.54 → 50.52 → 159.69 → **403.46 B** — while heldout gain peaks at 40.91 B and then goes *negative*.
That is the `ddm_pk3`/`pk4` signature reproduced exactly: the 403 B "oracle" is the map fitting the
noise in 2,560 cells, and out of sample it costs 175 B instead of saving 403.

**The generic control wins, and I have to say so.** Per the charter amendment the generic form is
carried only to price how much of a gain is generic. Here it is **all of it and more**: no
domain-native rung nets positive, because none of them pays for its own parameters. Charging the
edges as free (fixed dyadic bins in `inflate.py`, rule-118 legal) does not rescue them either —
per-bin(8) still nets −7.92 B and per-`d` × bin(8) still nets −119.09 B.

**Split variance is negligible.** Best net across seeds 20260824 / 777 / 31337: **8.436 / 8.376 /
8.497 B**, spread 0.12 B. The answer is not a lucky split.

### The address tax, and why it is small here but still decides

`d` costs **zero stored bytes**: the shipped decoder already computes `_boundary_buckets` for its
corrector, so the conditioning variable is rule-118 free and adds no decode work. This map is not
`ddm_tba1`/`ddm_df1` addressing — it names no positions, it reprices all of them through a function
of quantities the receiver already holds. There is no `3.1468×` address bound here.

**And it still decides the ladder.** At 4 B per parameter, a map needs to clear its parameter count
in bytes, and the miscalibration is so small that only the 1-parameter map does. The tax that killed
the previous three families killed this one too — just at a different scale.

---

## 6. The 4-way conditional pool, closed by measurement

The indicator decomposition leaves `−log2(p_sel/(1−pmax))` untouched, and that residual is
**2,500.54 B — above this arm's own 2,000 B falsifier.** It cannot be waved past.

The retained fields do not carry the runner-up's *index*, so the model's conditional row cannot be
reconstructed and no map can be fitted to it. What IS computable is the best a receiver-derivable
empirical table could do. MEASURED, cross-fitted over the 227,671 flips:

| | bits | bytes |
|---|---:|---:|
| shipped conditional | **20,004.07** | **2,500.51** |
| best empirical `P(transmitted \| argmax, d)` table | 212,645.14 | 26,580.64 |
| empirical `P(transmitted \| argmax)` table | 226,679.36 | 28,334.92 |

*(Handicap disclosed: the table's Laplace smoothing spreads mass over all 5 classes including the
argmax, which the flips cannot be. That wastes at most ~125 pseudo-counts over 227,671 flips, i.e.
**≤ ~23 B** — it can only make the table look worse, and the measured gap is 192,641 bits, so the
verdict is safe by three orders of magnitude.)*

**The shipped conditional is 10.63× cheaper than the best table**, so replacing it would *cost*
24,080 B. At 0.0879 bits per flip the model is already ~94% certain which class it should have
picked whenever it is wrong. The pool is closed from below by measurement, not by argument.
`verdict_scope` on this row is `FORMULATION: receiver-derivable (argmax, d) empirical table` — it
bounds the pool; it does not close the conditional as a family.

---

## 7. Distortion is zero — structurally, and the coder still works

**`dD = 0` EXACTLY, and this is an identity, not a measurement.** A recalibration is a deterministic
map applied to the coding row by both encoder and decoder. The RC64 decoder emits the transmitted
symbol; the autoregressive context, the corrector state and the render are functions of decoded
symbols alone. Changing the coding row changes the BITS, never the SYMBOLS. The decoded field is
bit-identical, so every SegNet cell and PoseNet input is bit-identical. There is no argmax to verify
invariant, because nothing downstream reads the coding row's argmax.

What *could* break is codeability, so that was checked (MEASURED):

| | value |
|---|---:|
| RC64 probability floor `2⁻³¹` | 4.6566129e-10 |
| runner-up cells below floor, **SHIPPED** | 44,290,888 |
| runner-up cells below floor, **RECALIBRATED** | 44,290,888 |
| **runner-up cells NEWLY below floor** | **0** |
| row scale factor `(1−q)/(1−pmax)` | min 0.6081, max 12.7644 |

Below-floor is not uncodeable — `bl1.rc64_costs` clamps every frequency to ≥ 1, and the shipped
stream already clamps *exactly* this many cells. The invariant that would actually break the coder is
`0 < freq[w] + balance < 2³¹`. Since `freq[w] + balance = 2³¹ − Σ_{j≠w} freq_j` and every non-winner
frequency is clamped to ≥ 1, that quantity is ≤ 2³¹−4 and > 0 for any row with five positive entries.
**The invariant is structural and holds for any fitted map.**

*(SCOPE: only `psecond` was retained, so the below-floor counts bound the RUNNER-UP cell, not the row
minimum. The scale factor is exact and is what multiplies every non-argmax cell.)*

---

## 8. Two errors of mine, recorded

**(a) My first estimator was the wrong family and it inverted the sign of the answer.** A reliability
map that REPLACES `pmax` with its bin's empirical accuracy discards every distinction the model draws
*inside* the bin. At 64 equal-mass bins that reported a **−6,987 B "gain"** — a loss dressed as a
measurement. The cure was to require the map family to nest the identity. The discarded number is
itself a finding: **the model's `pmax` carries at least 6,987 B of resolution beyond a 64-cell
quantization of itself**, which is a lower bound on how much a coarse confidence proxy would destroy.
Superseded receipt retained as
`analysis/SUPERSEDED_ANALYZE_fine65536_pre_codeability_fix_and_pre_conditional_bound.json` and
`analysis/ANALYZE_bins64.json`.

**(b) I wrote a codeability rationale the field contradicts.** My first receipt claimed the map moves
`q` strictly toward 0.5 so the winner frequency only decreases. The measured `max(q − pmax)` is
**+0.1306** — positive, because the under-confident bins get *sharpened*. The claim was wrong; the
conclusion survives on the structural invariant in §7, which never needed it. Corrected in the
committed source.

**(c) A near-miss worth naming.** `ddm_df1`'s `flip_flags.npy` is `packbits` with
`bitorder='little'`. Read with numpy's default (`'big'`) it yields the **exactly correct 227,671
flips and 117,737,129 non-flips** while joining to the wrong positions — the zero-mode bytes come out
107,823.54 instead of 34,674.11. Correct counts, wrong join. I caught it only because I re-derived
the flags definitionally from `argmax != transmitted` against the sha-verified TO2 token field; the
two agree at **all 117,964,800 positions** (`VERIFY.json`). Anyone else consuming that field should
pass `bitorder='little'` explicitly.

---

## 9. Controls and custody

- TO2 decoded token field sha256 `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`,
  117,964,800 B — matches the shipped digest.
- All six df1 headline figures reproduced from the retained fields before any analysis ran: total
  910,209.2806090622 bits; zero mode 117,737,129 positions / 34,674.11291334549 B; positive mode
  227,671 / 79,102.04716278728 B; float32-saturated 67,955,679 positions / 0.04860804728106061 B.
- **RC64 coder faithfulness:** realized 910,209.2806 bits vs the ideal model ledger's 910,209.4321
  bits — the coder is **0.0189 B cheaper** than the model over the whole stream. Coder quantization
  is not a byte source and was not pursued.
- **The boundary feature is the shipped one:** this module's `boundary_buckets` was proven equal to
  `runtime/residual_archive.py::_boundary_buckets` (sha `aca361f3e949…`) on 12 seeded random frames,
  0 mismatched.
- **Representation fidelity:** the binned identity cost is 890,206.98 bits against an exact
  890,204.98 — a **0.250 B** residual. Every reported gain is a difference taken on that same
  representation, so the residual cancels to first order. Gains below ~1 B (the per-`d` offset rung's
  5.43 B is the closest) should be read with that in mind.
- Denominator on every share in this memo is 117,964,800 positions unless stated.

---

## 10. Verdict

**The recalibration family is CLOSED on this object.** The shipped HPAC coding row is over-confident
by 1.93e-05 in probability, and the best receiver-derivable recalibration of it — evaluated out of
sample over all 117,964,800 positions, net of the bytes needed to ship the map — releases **8.44
bytes**. That is 0.0199% of the 42,381.16 B the campaign needs at fixed distortion, and 237× below
the pre-registered 2,000 B falsifier. Conditioning on boundary distance `d`, the one domain-native
variable the receiver gets for free, adds at most **16.83 gross bytes** and never pays for its own
parameters. The 4-way conditional pool is closed separately by measurement: the shipped model beats
the best empirical table by 10.63×. **DERIVED:** since the gain scales as the square of the
calibration gap, the model would have to be **12.7× more miscalibrated** — ECE ≈ 2.4e-04 rather than
1.9e-05 — before this family cleared 2,000 net bytes.

**verdict_scope:** `FAMILY: recalibration of the shipped DX2 HPAC coding row`
(archive `976f706d…`, n600, 117,964,800 token positions). This closes *recalibration* — monotone
repricing of the existing model's output. It does **not** close the model axis: a model that is
better *conditioned*, not merely better *calibrated*, is `ddm_ds1`'s lever and is untouched here. It
also does not close the 4-way conditional as a family (§6 scope), only the receiver-derivable-table
formulation of it.

### Routing, per the charter amendment

- **CONFIRMS the sharp-optimum law (`#1214`) on a sixth independent axis.** Five arms measured the
  HPAC optimum as sharp in every direction they pushed. This one pushes in the *probability* direction
  — the one place a sharp optimum in parameter space does not automatically imply a sharp optimum in
  output space — and finds it sharp there too. The 13,515 B HPAC model block stays closed.
- **The exchange-ratio ladder (`#1245`) is not escaped.** A zero-distortion gain does have infinite
  damage-to-credit ratio, so 8.44 B is free. It is also 8.44 B. The ladder needs a ≥21.6× improvement
  in a mechanism that moves real bytes; this is not one.
- **Durable artifact for the boundary/annulus work:** `retained/per_d_reliability_table.json`
  (35,246 B, sha256 `4c462384f24d51e5ae75bfd087a1424783bbadce6c145cb3e8130528a1cbc2ba`)
  and `analysis/DCOMPARE.json`, carrying the causal/acausal split in §1. **Any boundary lever priced
  on "94.53% of bits at `d=0`" is priced on an acausal field and should be re-priced at 49.85%.**

### Retained payloads (`/Volumes/APDataStore/pact/ddm_hc1_hpac_calibration/measurement_v1/`)

| artifact | bytes | sha256 |
|---|---:|---|
| `retained/boundary_distance_d.u8.bin` | 117,964,800 | `a6ffb6fe75190ef9ec956f961f470a5c6a7251dd46f6ff2a774b355673161324` |
| `retained/flip_flags_derived.npy` | 14,745,728 | `baa377e7f6ea190f2953a19b1bcd363fc5c2b09ea398728a6e85c1023b09a3a5` |
| `retained/per_d_reliability_table.json` | 35,246 | `4c462384f24d51e5ae75bfd087a1424783bbadce6c145cb3e8130528a1cbc2ba` |
| `analysis/ANALYZE_fine65536_seed20260824.json` | 48,442 | `bc732a71fa9c9d87c331ef975bf033467a9c2760bfda3dae5533cec9781e87be` |
| `analysis/ANALYZE_fine65536_seed777.json` | 47,180 | `a933a5d9ce779a43e2369e5098fc228f35f7f3dc942eb182593895e8d1cfaa7e` |
| `analysis/ANALYZE_fine65536_seed31337.json` | 47,166 | `3bbc5aebe15c9259e63428d89e07972b31ffa43f4a9c32d32b7a5844adce899a` |
| `analysis/DCOMPARE.json` | 1,481 | `00e954f477d8d49dea7cd9558c3724fe50d3b7f64863203dfaa044affeac9b9a` |
| `analysis/VERIFY.json` | 3,559 | `dd9c4f9254abeefb37ec35b4541c7837083550805ddebd730a0297525fd17cce` |
| `analysis/DFIELD.json` | 825 | `d348f4b0c0456b8e0698f2459f2b22397edd4b2c240cbaecdd142fcfcce24d60` |
| `analysis/ANALYZE_bins64.json` (superseded, sec.8a) | 31,543 | `0a47f578f76503fb1a8141b7fdee62db81fcf391c3fb84ac1023a40288c25a7d` |
| `analysis/SUPERSEDED_ANALYZE_fine65536_pre_codeability_fix_and_pre_conditional_bound.json` | 46,391 | `b83e7501ab5d89fa371899eb15525b40455c5ef61c156bf51d7b2e1574c0543a` |
| `analysis/RETENTION.json` | 2,797 | `8927b68761d80913744da4dc7f3cd7a7730a408474ae8f573f99388f11056f89` |

Total retained: 132,975,158 B. Vertigo has 8.4 GiB free and was not written to.

**Reproduce:** `.venv/bin/python experiments/ddm_hc1_hpac_calibration.py --stage
{verify,dfield,analyze,dcompare,manifest}`. Every stage is resumable from disk and writes its own
receipt. Total wall clock for all five stages: **under 70 s**.
