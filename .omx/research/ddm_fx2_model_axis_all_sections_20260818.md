# ddm_fx2 — the probability-MODEL axis: causal geometry, the SSE stage, and where the model axis ends

Date: 2026-08-18 · Arm: `ddm_fx2_probability_model_axis_all_sections_20260818` · Authority: exact
decode-identical code-length measurement, full n600 · **Score claim: false** · **Pointer moved: false**

## Conclusion first

**The model axis is still live on the token stream, and the thing that pays is GEOMETRY, not a
second stage.** The best measured architecture takes **−797.42 B** off the n600 token field against
the live rr4 law — **−237.35 B beyond ddm_fx1's −560.07 B**, a 42% extension of the inherited win, at
a decode-identical token field.

**But the row I RECOMMEND is not that one.** Serial decode timing says the 19-member build leaves
only 29 s of margin against the contest budget, so the recommended candidate is the 13-member build
at **−710.84 B** — projected archive **180,450 B**, projected **S = 0.158059824632**, 10× the 1e-5
naming bar, with 118 s of margin against a row that already passed on T4 with 160 s. The bigger row
is banked, not thrown away: it becomes the pick the moment somebody measures real T4 headroom
instead of projecting it. §8 is that arithmetic.

Four findings, in the order they change what the next arm should do:

1. **ddm_fx1's causal template was under-used, and the probe that shows it costs one minute.** The
   shipped 190-group wavefront is exactly `group(x,y) = (x&63) + 2*(y&63)`, and under it the
   **UP-RIGHT neighbour is 98.6945% causal — identical to UP, and higher than LEFT**, which fx1 did
   consult. Widening the template to four neighbours and adding a *local homogeneity* feature (do my
   decoded neighbours agree with EACH OTHER, rather than with the prediction) is free receiver code
   and it dominates every other dial measured here.
2. **The SSE/APM second stage LOSES, in 6 of 6 formulations (+9.91 B to +139.56 B).** This is the
   classic PAQ stage ddm_fx1 named as its rank-3 next step. It is measured dead on this stream, and
   the mechanism is legible: it costs most where its context duplicates the mixer's own.
3. **The other archive sections are not model-axis targets, and the semantic blob is the case that
   proves it.** It carries the archive's only other adaptive coder — an order-0 rANS with no context
   at all — and every richer context makes it WORSE (order-1 +300.6 B, order-2 +769.2 B). Mixing
   recovers 73% of that dilution but never crosses into profit.

4. **And the biggest single number I measured is not on my axis at all.** MAIN's ddm_pd1 relay
   pointed at weight SERIALIZATION rather than another coder race. Measured on our payload: the
   semantic blob carries **8,284 B of fp16 metadata that nothing entropy-codes**, and byte-splitting
   it before the container's own Brotli is worth **−515 B** — with a control that reproduced the
   shipped 34,763 B section at delta +0. That is larger than this arm's entire token-stream win, and
   it belongs to whoever builds the receiver-side un-split. §10.

**Own-vehicle frontier: `S = 0.15816036933414834 @ 180,601 B [contest-CUDA T4, n600]`, archive sha
`65c75d7f…` — verified at source in ddm_fx1's own fire receipt. This arm did NOT move it** — only
MAIN's fire can. No Modal spend.

## 1. Controls — the instrument before any verdict

No row below is admissible unless these hold. All are full n600, all re-run in my own hands.

| control | target | measured | verdict |
|---|---:|---:|---|
| fx1 winner reproduced through MY harness | −560.07 B | −560.07 B | **delta −0.00 B** |
| fx2 with the SSE stage OFF ≡ fx1, on a driven replay | bit-identical | 18545.41735241344 bits both | **bit-identical, same payload sha** |
| a COLD SSE bin returns exactly 1.0 | exactly | exactly | pass (unit + replay) |
| R3 ranking instrument vs fx1's measured k1 order | reproduce the order | top-3 exact, 1 adjacent swap of 7 | pass, scoped in §4 |
| R5 order-0 replay vs the shipped semantic ANS bytes | 26,430 B | 26,385.7 B + 52 B of rANS state flush | **within 0.03%** |
| §10 re-Brotli of the exact F12 semantic body | 34,763 B | 34,763 B | **delta +0** |
| **frozen `FreeCorrector(plane)` vs the raced D1 row** | 878,403.5143494605 bits | 878,403.5143494605 bits | **delta +0.000000 bits** |

Two of these are load-bearing. The second says my class collapses onto the law it generalises
bit-for-bit, so every delta below is the new mechanism and never the plumbing. The last says the
FROZEN shipping surface — `FreeCorrector(plane)`, which is what the encoder and the receiver actually
build, with no arguments — reproduces the raced configuration exactly. Without it the raced number
would describe a parameterised object rather than the thing that ships.

**The test suite is not vacuous.** Three defects were injected and all three were caught: a
reintroduced transcendental on the decision path (the `ddm_rr2` refusal class), an inert SSE stage,
and the causal template narrowed back to fx1's left+up. 17 fx2 tests plus fx1's 107 all pass.

## 2. R1 — the scan-order probe, and the finding fx1 left on the table

`ddm_fx1._spatial_level` reads LEFT and UP. Measuring the real `group_index.u8` recovers the decode
order exactly — the rule is `(x & 63) + 2 * (y & 63)`, 190 groups — and a neighbour is already
decoded exactly when its group index is smaller. Measured, in-bounds:

| offset | causal | offset | causal |
|---|---:|---|---:|
| up `(0,−1)` | **98.6945%** | up-left `(−1,−1)` | 97.3425% |
| **up-right `(+1,−1)`** | **98.6945%** | `(+2,−2)` | 97.3822% |
| left `(−1,0)` | 98.6301% | down `(0,+1)` | 1.3055% |

Up-right is causal whenever `y&63 ≠ 0`, which is *exactly* UP's condition — and the `x` wrap makes
it causal even at `x&63 = 63`, where the neighbour's group is `g−65`. **It was always as available
as up.** The whole upper-right quadrant follows. Correctness never rests on that arithmetic: every
read is gated by the runtime `known` mask, so the arithmetic explains why the wider template is
worth its cost while the mask is what makes it safe.

## 3. R2 — the SSE / APM second stage is measured DEAD on this stream

The PAQ answer to "the mixture is biased in a regime" is a second stage indexed by the mixture's own
output. Built exactly, reusing the shipped `2^(−k/2)` ladder on `1 − q` so the bins are log-spaced in
the tail with no logarithm, and the shipped KT count ratio with `q_mix` as the accumulated
expectation, so a calibrated bin and a cold bin both return exactly 1.0.

| SSE context | cells | Δ vs its own base |
|---|---:|---:|
| `qbin` | 64 | **+9.91** |
| `cls_homog_qbin` | 1,600 | +24.26 |
| `bnd_qbin` | 320 | +42.69 |
| `cls_qbin` | 320 | +44.98 |
| `cls_bnd_qbin` | 1,600 | +139.56 |
| `qbin` + learned exponent, on the 13-member base | 64 | +15.62 |

**Six formulations, six losses.** The mechanism I believe (INFERRED, not measured): the cost tracks
how much the SSE context DUPLICATES the mixer's own. `cls` and `boundary` are already inside the
51,200-bin shipped member context, so `cls_bnd_qbin` re-estimates what is already estimated and pays
pure variance — the worst row. `cls_homog_qbin` brings `homog`, which no member had, and is the least
bad of the crossed contexts. `qbin` alone is coarsest and cheapest.

The honest scope: **this is a family-level negative for SSE-on-top-of-a-converged-mixer on THIS
stream**, not for SSE in general. The mixer's weights are fitted by descent on exactly the log-loss
the SSE would repair, so there is little miscalibration left and the second stage buys noise. The
learned exponent was the pre-registered rescue and it made things worse, not better, which is the
result that closes the family rather than leaving it ambiguous.

## 4. R3 — statistical context discovery, and its validated scope

Racing a context costs 3–4 minutes. Ranking one need not. The instrument scores a candidate partition
by the code length achievable if every cell were given its own best exponent on the odds multiplier —
and that bound is EXACT for the single-member case, because `m` comes from KT counts of `(hit, p_max)`
and neither depends on the coding probability. **Sweeping the exponent cannot perturb the data the
sweep is measured on.** Overfitting is measured rather than assumed: the exponent is fitted on frames
`[0,300)` and scored on `[300,600)`.

| context | cells | in-sample B | held-out B |
|---|---:|---:|---:|
| `cls_boundary_agree_homog_ubin8` | 4,000 | 608.3 | **175.0** |
| `cls_boundary_homog_ubin8` | 1,000 | 509.4 | 168.4 |
| `cls_homog_ubin8` | 200 | 417.3 | 148.6 |
| `cls_boundary_agree_ubin8` *(the fx1 winner)* | 800 | 459.6 | 66.1 |
| `cls_boundary_agree_spatial4_ubin8` | 4,800 | 609.3 | 35.8 |
| `cls_spatial4_ubin8` | 240 | 363.3 | 22.3 |
| `cls_boundary` | 25 | 304.7 | −14.2 |
| `none` | 1 | 17.3 | −363.7 |

**Positive control: it reproduces fx1's measured k1 ordering — top 3 exact, one adjacent swap out of
7.** So it may rank; it may not be quoted as a magnitude.

**Pre-registered prediction, then the race.** The instrument said all three `homog` contexts beat the
fx1 winner. Measured, on a matched 13-member base:

| context | R3 rank | measured Δ vs live | verdict |
|---|---:|---:|---|
| `cls_boundary_agree_homog_ubin8` | 1 | **−710.84** | predicted |
| `cls_boundary_homog_ubin8` | 2 | −641.26 | predicted |
| `cls_boundary_agree_ubin8` *(incumbent)* | 4 | −600.18 | — |
| `cls_homog_ubin8` | 3 | −542.94 | **missed — it LOST** |

**Ranks 1 and 2 predicted; ranks 3 and 4 inverted.** The honest reading of the miss: R3 scores a
context in isolation, holding `m` at the shipped law, so it measures standalone recalibration power.
In the race the context selects weights over 13 members, and `cls_homog_ubin8` drops `boundary` and
`agree`, which those members depend on. **The instrument ranks contexts as recalibrators, not as
weight-selectors over a rich member set** — usable for shortlisting, not for the final call.

## 5. R4 — the member pool, the compose, and the axis that is nearly free

ddm_fx1 measured its pool still climbing at 11 members (8 → 11 bought 23 B). It is still climbing at
19. Every row is full n600 against the live rr4 law; `K` is the member count.

| id | K | mixer context | Δ vs live | Δ vs fx1 |
|---|---:|---|---:|---:|
| **E1 compose** | **19** | `cls_boundary_agree_homog_ubin8` | **−797.42** | **−237.35** |
| E6 | 19 | `cls_boundary_agree_spatial4_ubin8` | −770.93 | −210.86 |
| E3 | 15 | `cls_boundary_agree_homog_ubin8` | −755.93 | −195.86 |
| D1 | 13 | `cls_boundary_agree_homog_ubin8` | −710.84 | −150.77 |
| C5 | 19 | `cls_boundary_agree_ubin8` | −708.11 | −148.04 |
| E4 | 12 | `cls_boundary_agree_homog_ubin8` | −674.28 | −114.21 |
| **E5 cheap** | **11** | `cls_boundary_agree_homog_ubin8` | **−664.08** | **−104.01** |
| C4 | 17 | `cls_boundary_agree_ubin8` | −662.06 | −101.99 |
| C3 | 15 | `cls_boundary_agree_ubin8` | −657.92 | −97.85 |
| B3 | 13 | `cls_boundary_agree_ubin8` | −600.18 | −40.11 |
| B2 (homog only) | 12 | `cls_boundary_agree_ubin8` | −586.67 | −26.60 |
| B1 (spatial4 only) | 12 | `cls_boundary_agree_ubin8` | −578.46 | −18.39 |
| fx1 winner (control) | 11 | `cls_boundary_agree_ubin8` | −560.07 | −0.00 |
| E7 | 19 | `cls_boundary_agree_ubin16` | −629.34 | −69.27 |
| E8 (learning rate 2⁻²) | 19 | `cls_boundary_agree_homog_ubin8` | −151.94 | **+408.13** |

Three things this table says that the inherited prior did not.

**The two axes compose.** The member axis alone (C5, −148.04) and the context axis alone (D1, −150.77)
each buy about 150 B, and together they buy 237.35 B — 79% additive. The individual new members are
also near-additive: `spatial4_surprise` −18.39 and `homog_surprise` −26.60 sum to −44.99 against a
measured −40.11 together (89%).

**The context axis is nearly FREE and the member axis is not — and that inverts the fx1 ordering
under a decode budget.** E5 takes fx1's *exact eleven members* and changes only the mixer context:
−664.08 B for one extra pass over four already-decoded neighbours. That beats C3's fifteen members on
the old context (−657.92) while running faster, because a mixer context adds no per-group table
update and each member adds one. ddm_fx1's dial ordering (context > members > lr) survives; what is
new is that under a *cost* constraint the gap is much wider than the byte table alone suggests.

**The learning-rate plateau NARROWS as the model grows.** ddm_fx1 measured 2⁻² and 2⁻⁴ within 3 B at
8 members over 800 weight cells. At 19 members over 4,000 cells the same 2⁻² costs **+408 B against
its own base** — a swing of more than 550 B on a dial the inherited prior recorded as flat. The
plateau was a property of that operating point, not of the learner. Any future arm that grows this
model must re-measure the rate rather than inherit it.

## 6. R6 — the conditional-entropy floor, and how much is actually left

The R3 bound answers "what is the best exponent per cell". This answers the stronger question: what is
the best hit PROBABILITY per cell — which bounds every model that sees that context, mixing and SSE
and anything else. KT-smoothed, so a one-observation cell cannot claim zero entropy.

The decomposition closes exactly against ddm_fx1 §5, which is the control: hit-event + within-miss =
total, `110,862.39 + 1,247.19 = 112,109.58` uncorrected and `109,264.09 + 1,247.19 = 110,511.28`
shipped. Both match to the cent.

| context | cells | used | obs/used cell | in-sample floor | held-out |
|---|---:|---:|---:|---:|---:|
| `cls` only | 5 | 5 | 23,592,960 | 242,646.9 | 247,297.1 |
| shipped joint | 51,200 | 15,386 | 7,667 | 108,522.2 | 114,788.9 |
| shipped × homog | 256,000 | 28,264 | 4,174 | 108,305.4 | 117,001.0 |
| shipped × homog × spatial4 | 1,536,000 | 57,349 | 2,057 | **108,151.4** | 121,690.7 |

**Where this arm sits.** E1's total 109,713.85 B minus the untouched within-miss 1,247.19 B leaves a
hit-event code length of **108,466.66 B**. Against the richest floor measured, **315.3 B remain** —
and that floor is hindsight-optimal, so the achievable remainder is smaller than 315 B. The
token-stream hit event is close to done.

**The held-out column is the finding, and it is not the floor.** A static per-cell probability fitted
on frames [0,300) and applied to [300,600) is WORSE than the shipped online estimator in every row
(114,788.9 vs 109,264.09). The field is strongly non-stationary: online adaptation is worth more than
5,500 B, which is 7× everything this arm and ddm_fx1 won together. That also explains why the R3
held-out magnitudes under-predicted the races by ~5× — a static bound cannot see what an online
learner earns by tracking drift — and it is the reason R3 is quoted as a ranker only.

**So the honest close on the token stream: about 315 B of hit-event headroom under contexts of this
richness, plus ddm_fx1's untouched 1,247 B within-miss ceiling.** Those two numbers are what a future
arm is playing for, and they are now both priced.

## 7. R5 — the other sections: mapped, and one of them priced

The archive does not have eight sections. It has **three disjoint byte regions**, and the eight names
in the build receipt are a decoded VIEW:

| region | bytes | model today |
|---|---:|---|
| `token_stream` | 110,512 | neural HPAC logits + the shipped adaptive KT corrector — **this arm's target** |
| `compressed_models` (RX1M, Brotli) | 70,453 | container; `semantic` 34,763 · `carrier` 22,161 · `hpac` 13,515 |
| residual table | 96 | fixed 6-bit packing |
| zip framing | 100 | — |

Order-0 byte entropy of the three Brotli streams is 7.98–7.995 bits/byte, so **outer recompression is
dead** and any gain must come from the inner bodies, discounted ~13% on the way back out (`ddm_ra2`,
measured).

**The semantic blob was the one worth pricing**, because it is the archive's only other *adaptive*
coder and its model is order-0 with no context whatsoever (`_update` increments a count and stops).
Measured on the real symbol sequences, 13 ANS streams, 26,430 shipped bytes:

| model | raw bytes | vs shipped order-0 |
|---|---:|---:|
| order-0 adaptive *(the shipped model — control)* | 26,385.7 | — |
| order-1 (previous symbol) | 26,686.3 | **+300.6** |
| order-1 MIXED (geometric blend with order-0) | 26,465.7 | **+80.0** |
| order-2 | 27,154.9 | **+769.2** |

**Every context loses.** The W4-quantised weights are close to i.i.d., so there is no order-1
structure to find, and splitting the counts only splits the evidence. This independently reproduces
`ddm_ra2`'s carrier finding (`ctx_dim × activity` cost +82 B where `ctx_dim` alone saved 259.8 B) on
a different section — so **context dilution is a property of these payloads, not of one codec.**

The mixed row is the part worth banking: **mixing recovers 73% of the dilution (300.6 → 80.0 B)
without being told which model is right.** That is the fx2/fx1 mechanism working exactly as designed
on a payload it was not built for — and it still does not cross into profit, because hedging can
protect you from a bad context but cannot manufacture structure that is not there.

**Priced ceiling, not a byte claim.** No container round-trip, no exact-decoder proof, and the 13%
Brotli discount is unapplied. What it buys the next arm is the right to skip this section.

## 8. The candidates, projected — and the constraint that picks among them

Archive bytes follow from the code length by `archive = 181,161 − (110,512 − ceil(code_bytes))`: the
token section is `ceil(code_bytes)` and every other section is carried through unchanged.

**That formula has a control, and it is exact.** ddm_fx1's raced code length was 109,951.21 B; the
formula gives token 109,952 B and archive 180,601 B; its byte-close **measured** token 109,952 B and
archive 180,601 B, with `token_delta_vs_target: 0`. So the projection below is calibrated on a
measured round trip rather than assumed — though it is still a projection.

Projected against the **fx1 frontier, fired and confirmed at 180,601 B / S 0.15816036933414834
[contest-CUDA T4 n600], archive sha `65c75d7f…`**:

| id | K | mixer context | token B | archive B | projected S | Δ B vs fx1 | bar × | decode verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|
| E1 | 19 | `cls_boundary_agree_homog_ubin8` | 109,714 | 180,363 | 0.158001894903 | −238 | 15.8× | **29 s margin — refused** |
| E6 | 19 | `cls_boundary_agree_spatial4_ubin8` | 109,741 | 180,390 | 0.158019873095 | −211 | 14.0× | untimed, ≈E1 |
| E3 | 15 | `cls_boundary_agree_homog_ubin8` | 109,756 | 180,405 | 0.158029860979 | −196 | 13.0× | untimed, between |
| **D1** | **13** | **`cls_boundary_agree_homog_ubin8`** | **109,801** | **180,450** | **0.158059824632** | **−151** | **10.0×** | **118 s — RECOMMENDED** |
| **E5** | 11 | `cls_boundary_agree_homog_ubin8` | 109,848 | **180,497** | **0.158091120003** | **−104** | 6.9× | 142 s — fallback |

**These are PROJECTIONS, not byte-closed measurements.** Uncertainty is ±1 B on the ceiling. The
byte-close is the first step of the fire-order, not a claim this arm makes.

**Why there is a menu rather than one row.** ddm_fx1 measured the decode budget at **160.2 s of margin
(8.9%) against the 1,800 s contest limit** with 11 members, and that margin is the thinnest number in
the whole lineage. Members are the expensive dial — each one adds a table update per group — while a
mixer context adds one pass over four already-decoded neighbours and no table at all. That is why E5
exists: **it takes fx1's exact eleven members and changes only the context**, so its decode cost is
close to the row already proven to fit, and it still banks 6.9× the naming bar.

**The decode cost, measured SERIALLY — one row at a time, nothing else running.** The race elapsed
times above ran 8–9 jobs at once and are not a cost. The baseline is the shipped rr4 law through the
same harness, so the marginal is the corrector and nothing else. The conversion to a real parse-back
is a **one-point calibration**: ddm_fx1 measured +137.49 s on the real parse-back for what is a 109 s
marginal on this harness, giving a scale of 1.260. The *marginal* carries because it is additive
host-CPU numpy work; the 1,502.29 s absolute base is a `[macOS-CPU advisory]` local replay and does
**not** carry to T4.

| candidate | K | harness s | marginal | projected parse-back | projected margin | verdict |
|---|---:|---:|---:|---:|---:|---|
| shipped rr4 law | 1 | 34 | — | 1,502 | 298 s (16.5%) | — |
| fx1 winner *(fired, passed)* | 11 | 143 | 109 | 1,640 | 160 s (8.9%) | the proven reference |
| **E5** | 11 | 158 | 124 | 1,658 | **142 s (7.9%)** | safest row that beats fx1 |
| **D1** | 13 | 177 | 143 | 1,682 | **118 s (6.5%)** | **recommended** |
| E1 | 19 | 247 | 213 | 1,771 | **29 s (1.6%)** | **refused — that is not a margin** |

**This is why the largest row is not the recommended one.** E1 buys 87 B more than D1 and spends 89 s
more of a margin that fx1 already drew down to 160 s. Twenty-nine seconds against a one-point
calibration with an advisory-local base is indistinguishable from zero. D1 keeps 74% of the margin
that already passed on T4 and still banks 10× the naming bar, so **`SHIPPED_CONFIG` is frozen to D1**
and E1 stays on the order as the upside if MAIN measures real T4 headroom first.

## 9. Honest limits

* **Selection on the scored clip.** The member set, the mixer context and the learning rate were
  chosen by racing on the scored video, exactly as ddm_fx1 §7 disclosed for its own selection. The
  *family* is robustly negative — every widened-template row measured negative — and the cheapest
  candidate (E5) involves the fewest selected degrees of freedom: fx1's own members with one context
  swapped. What is **not** claimed is that the winning configuration is optimal on another clip.
* **Code length, not archive bytes. NOT BYTE-CLOSED, and I hit a real blocker.** Every number here is
  a measured code length from the exact replay, and §8's archive figures are projections on
  ddm_fx1's calibration (its own projection landed at `token_delta_vs_target: 0`). I attempted the
  byte-close and it is blocked at input verification, not at the algorithm:
  `ddm_pq2_compress_e2e.py`'s default recipe pins `rc64_source_sha256 =
  5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6`, and **no file on either SSD
  matches it** — I hashed all 252 `.c` files under the pact tree and every `rc64_backend.c` copy is
  `05839d1416e68a49…`. The other three inputs resolve cleanly (prepared dir `80d9c8c6…` ✓, tokens
  `9ba2e52b…` ✓, hm1 dir ✓). The fix is a `--recipe-json` carrying the correct pin, or restoring the
  pinned file; I did **not** bypass the fail-closed check, because that check is the thing standing
  between a rebuild and silently different bytes.
* **Cross-platform exactness is argued, not demonstrated.** Every operation is IEEE correctly rounded
  and the AST gate refuses transcendentals, but `ddm_rr2` is the receipt that a correct local proof is
  not a cross-platform one. The parse-back is a hard gate, not a formality.
* **`_causal_neighbours` runs unconditionally**, even for a configuration whose members and context
  read none of its output. That makes the T1 timing below an over-statement of fx1's own cost, so
  every marginal quoted against it is conservative — but it is also ~10 s of avoidable work in any
  shipped configuration and a future arm should gate it on the declared feature set.
* **E2 (19 members × 8 count buckets) was killed, not measured.** 32,000 weight sets × 19 members made
  the per-group learner allocation dominate; ddm_fx1 had already measured `cb=1` above `cb=8`, so it
  was the cheapest row to drop. It is an unmeasured cell, not a negative.

## 10. MAIN's ddm_pd1 relay — four rows, each checked against our own payloads

The relay arrived mid-arm. Each row was verified rather than adopted, and two of the four change
what happens next.

**Row 1 — LZMA `lc=0 lp=1 pb=0 dict=64KiB`, "worth −1,092 B on LZMA'd array sections". NOT
APPLICABLE HERE, for two independent reasons, both verified at source.** First, our runtime *already*
declares exactly that filter (`runtime/residual_archive.py:57-68`: `FILTER_LZMA2`, `dict_size 1<<16`,
`lc 0`, `lp 1`, `pb 0`) — the tuning is not an available delta, it is already ours. Second, our
shipped archive declares `RX1M codec=2`, which is **Brotli**: no section is LZMA'd, so the delta has
nothing to land on. Measured anyway, on our three real inner bodies through the real codecs:

| section | raw | shipped (Brotli) | LZMA at pd1's params | Δ |
|---|---:|---:|---:|---:|
| hpac | 17,952 | 13,515 | 13,603 | **+88** |
| semantic | 36,040 | 34,763 | 34,796 | **+33** |
| carrier | 22,219 | 22,161 | 22,223 | **+62** |

LZMA at those parameters is *worse* than the Brotli the container chose, on every section.

**Row 2 — weight SERIALIZATION, not another coder race. CONFIRMED, and it is the biggest single
number this arm produced: −515 B.** The semantic blob carries **8,284 B of fixed metadata (fp16
scales and fp16 tensors) that is entropy-coded by nothing** — it rides the outer Brotli as raw
interleaved fp16, where the exponent and mantissa bytes alternate and destroy the match distance.
Splitting the high and low bytes before compression is a pure serialization change:

| | bytes | Δ |
|---|---:|---:|
| shipped semantic section | 34,763 | — |
| **control**: re-Brotli of the exact F12-ordered body | **34,763** | **+0 — the container's parameters reproduced exactly** |
| candidate: same ordering, fp16 metadata byte-split | **34,248** | **−515** |

The control is what makes it readable: my instrument reproduces the shipped section byte-for-byte
before it is asked to beat it, so the −515 B is the split and nothing else. **This is a section-body
measurement, not yet an archive byte** — the receiver must un-split at decode time, which is an exact
integer permutation but still needs implementing and round-tripping. Unlike the token-stream work it
is not a drop-in. pd1's aim was right and it points away from where I spent this arm.

**Row 3 — re-key the ceiling to miss BIT-SHARE, not the 0.19% miss fraction. Taken, and my numbers
survive because they were already bit-keyed.** Nothing in §6 is keyed to a position fraction: the
decomposition is `hit-event bits + within-miss bits = total bits`, and it closes exactly
(`110,862.39 + 1,247.19 = 112,109.58` uncorrected, `109,264.09 + 1,247.19 = 110,511.28` shipped).
I have **not** re-derived the 0.19% figure and do not rely on it. The relay's companion point does
apply to us: `q` carries 96.2–98.4% of miss cost only under a STRONG prior, and ours is strong
(99.81% hit rate), so the mixer is on the right sub-axis for this vehicle specifically — a weaker
prior would move the target.

**Row 4 — "context policy is worth 0.4 bits/token" as a ranking prior. DOES NOT TRANSFER, and the
reason is a 388× operating-point gap.** That prior is measured going from 2.908 → 2.4993 bits/token.
Our token stream, after the HPAC neural prior, sits at **0.007495 bits/token** — 388× lower. My
measured context upgrade bought 104 B over 117,964,800 tokens, i.e. **7.05e-6 bits/token**, five
orders of magnitude below the prior's scale. A bits-per-token prior taken from an unmodelled stream
cannot rank members on a stream this heavily pre-modelled; the ranking here has to come from an
instrument on our own field, which is what §4 built.

**pd1's own caution, honoured.** Every number in this memo is exact bits through the real codec or
the real replay. The one place I report a model comparison rather than a codec output (§7's order-1
and order-2 rows) is labelled a priced ceiling, and its order-0 control was checked against the
shipped bytes first.

## STORES CONSULTED

`.omx/research/ddm_fx1_fixed_point_logistic_mixer_20260817.md` §4–§5 + §9 (the 33-row inherited race
table, the miss-sector decomposition, the 1,247 B ceiling, the sealed order) · memory
`[[probability-model-axis-live-fx1-sweep-prior]]` (the inherited dial ordering; never re-measured) ·
`.omx/research/ddm_fx1_t4_sealed_fire_order_20260817.json` (the seal schema this arm matches) ·
`experiments/ddm_fx1_logistic_mixer_corrector.py` (the mixer this extends, read at source) ·
`experiments/ddm_rr4_free_corrector_v2.py` (the live law, the exactness argument, the frozen
`2^(−k/2)` ladder reused by the SSE bins) · `src/tac/micro_edit/coder_replay.py` (the instrument) ·
`/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/RESULT_build.json` (the section digests and the
181,161 B incumbent) · `…/candidate_runtime/runtime/entropy/adaptive_ans.py` (**verified at source**:
the semantic rANS model is order-0, `_update` increments and stops) ·
`…/runtime/entropy/renderer_weight_codec.py` + `…/runtime/residual_archive.py` (the WANS1 framing and
the real three-region member split) · `/Volumes/APDataStore/pact/ddm_ra2_cpr1_headroom/` (the carrier
dilution result and the 13% Brotli discount) · `.omx/research/ddm_hx1_pr_wave_harvest_20260817.md`
(the scan-order lead this arm measured).

## Artifacts

Payloads → `/Volumes/APDataStore/pact/ddm_fx2/`, 84 files / 14.5 MB, every one hashed in
`RETENTION_MANIFEST.json`. **Per-variant bit payloads are retained for EVERY raced architecture, not
only the winner** (the ddm_fx1 compounding rule), so the next arm reads the family shape at $0:

* `race/*.json` — all 25 n600 rows with their final learned weights and payload shas
* `retained/bits_*.npy` — the per-frame bit vector behind each row
* `probe/r1_scan_order.json` — the causality of all 48 offsets
* `probe/r3_context_rank.json` — the context ranking + its positive control
* `probe/r6_floor.json` — the conditional-entropy floors
* `probe/r5_semantic_headroom.json` — the semantic-blob model race
* `probe/r5b_main_relay_rows.json`, `probe/r5c_scale_split_f12.json` — MAIN's relay rows, measured
* `probe/frozen_dropin_control.json` — the frozen surface vs the raced row
* `probe/mutation_check.py` — the three injected defects and their catches
* `race.py`, `timing_serial.sh`, `emit_fire_order.py`, the batch scripts

Code: `experiments/ddm_fx2_model_axis_corrector.py` · `src/tac/micro_edit/tests/test_fx2_model_axis.py`
(commits `53d9a13d61` build+tests, `85880c77a6` frozen config).
Fire-order: `.omx/research/ddm_fx2_t4_sealed_fire_order_20260818.json`.

## NEXT_IF_RESUMED, ranked

1. **Build the semantic fp16 metadata byte-split — it is the biggest measured number here, −515 B,
   and it is not mine to compose.** Bigger than this arm's whole token-stream win (−151 B on the
   recommended candidate), measured through the real Brotli with a control that reproduced the
   shipped section at delta +0. It needs a receiver-side un-split (an exact integer permutation) and
   its own round-trip, which is why it is a separate build rather than part of the candidate. §10
   row 2.
2. **MAIN byte-closes and fires the candidate on the decode-cost menu in §8** — after clearing the
   `rc64_source_sha256` pin blocker in §9. The measurement is done; what converts it into a pointer
   move is the byte-close, the parse-back and one T4 row.
3. **The within-miss relative law — still untouched, still the largest priced target on the token
   stream.**
   ddm_fx1 measured its ceiling at **1,247 B** and nobody has attacked it. Every arm so far, this one
   included, has worked the hit event, which §6 now shows is within ~315 B of its floor. The ratio
   has inverted: the un-adapted sector is now 4× the remaining adapted headroom.
4. **Re-derive the learning rate whenever the model grows.** §5's E8 row is the receipt: a dial the
   inherited prior recorded as a flat plateau swung 550 B at 19 members. Do not inherit it.
5. **Do NOT re-run the SSE/APM stage as formulated** (6 of 6 negative, §3) and **do not model the
   semantic, carrier or hpac blobs with richer contexts** (§7 measured, and `ddm_ra2` measured the
   same on a different section). A mixture is the right hedge against dilution and it still does not
   turn a structureless payload into a compressible one.
6. **If a future arm wants more from the hit event, it needs a richer CONTEXT, not a second stage.**
   §6 shows the floor keeps falling as the context is enriched (108,522 → 108,151 B), but cell
   occupancy falls with it (7,667 → 2,057 observations per used cell), so the next context must buy
   its richness from a feature with real structure rather than from a cross product.
