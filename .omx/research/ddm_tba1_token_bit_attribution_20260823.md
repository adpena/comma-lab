# ddm_tba1 — the token stream is concentrated past the point where any explicit selector can pay for itself: the expensive set is too small to name and too task-relevant to drop

`date_utc: 2026-08-24` · `arm: ddm_tba1` ·
`axis: [macOS-CPU advisory / scorer-free re-derivation over retained coder output]` ·
`score_claim: false` · `promotion_eligible: false` · `frontier_moved: false` ·
`verdict_scope: INSTANCE:DX2_archive_976f706d_n600_shipped_HPAC_RC64_stream` — with the
selector-cost leg at `FORMULATION` scope (explicit position-selected subsets), stated per row.

## ANSWER FIRST

**SUM residual: 6.719391 bits = 0.839924 B**, or 7.382e-6 of the 910,216-bit physical stream,
inside the `<9`-bit arithmetic-coder bound (final-interval flush `<2` bits + partial-byte padding
`≤7` bits). The gate PASSES; the attribution is sound.

**Concentration: Gini 0.9951593787014772.** The top 1% of positions carries **96.323842%** of the
bits (109,593.6 B) and the top 10% carries **99.900879%** (113,663.4 B). The charter's
pre-registered `>70% in the top 10%` is **CONFIRMED** at 99.90%.

**The charter's premise was stale and I did not re-run the encoder.** `ddm_tb2` already produced
this exact measurement on 2026-08-23, and tb2's own recall found `ddm_bl1` had produced the field
before that. I verified both retained copies by SHA-256 (`99d7833d…`), confirmed the decoded-symbol
field is `cc10a7b0…` — **this arm's charter-pinned categorical field sha** — and re-derived every
headline from the primary artifact instead. A third byte-identical replay would have been
rediscovery, which CLAUDE.md names the cardinal signal-loss sin.

**The new result is the SELECTOR-COST CEILING, and it closes a family by arithmetic.** Any lever
that treats a chosen subset of positions differently must tell the receiver *which* positions.
Priced at the i.i.d. indicator reference `N·H(m/N)`, that cost **exceeds the entire bit mass of the
subset at every cost threshold from `>0.01` to `>16` bits** — even under the maximally generous
assumption that the subset could then be coded for exactly zero. The best net over all thresholds
is **+9.45 B** (25 positions, `>24` bits): **0.0222% of the 42,382 B demand, ΔS +6.29e-06**. The
reference is calibrated, not asserted: it predicts `ddm_ae1`'s **measured** flag cost to **94.81%**
and ae1's measured net to within **6.9%**. I then ran the obvious counter-attack — PR101's
canonical colex-rank position encoding (CLAUDE.md L31) — and **it fails**: it moves the family's
best net only to **+9.90 B**, still short by **4,280×**.

**This is a CONFIRMED concentration with NO actionable direction, and that is the decision-relevant
answer.** I am saying it in the charter's words rather than manufacturing a lever. The bit mass and
the task-relevant mass occupy the *same* 0.2% of positions: naming that set costs more than it
holds, and dropping it is the whole-body-lossy family `ni1`/`nr1` already closed at 247.69×/349×
over bar.

The pointer did not move. No scorer, Modal job, Metal job, archive mutation, or `upstream/` write
occurred.

---

## 1. SUM VERIFICATION — the gate

| quantity | bits | bytes | source |
|---|---:|---:|---|
| physical RC64 stream | **910,216** | **113,777** | ar1b census, span `[66522,180299)`, sha `e2af55e6…` |
| sum of all 117,964,800 per-symbol costs | **910,209.280609** | **113,776.160076** | MEASURED this run |
| residual | **6.719391** | **0.839924** | physical − attributed |
| allowed bound | `<9` | `<1.125` | interval flush `<2` + padding `≤7` |

Residual as a fraction of the stream: **7.382194e-06**. `residual_within_bound: true`.

This reproduces tb2's residual to all printed digits from an independent code path. Internal
closure also holds: the five-class partition sums to 910,209.2806218 bits against a total of
910,209.2806091 (residual −1.27e-05 bits, float64 accumulation order) and the class counts sum to
exactly 117,964,800.

## 2. SELF-DETECTED CLASS ORDER — verified, never hardcoded, never luma-sorted

Assigned from spatial/static signature only: Undrivable = smallest vertical centroid, MyCar =
largest, Road = largest remaining area, Lane = lowest temporal IoU, Movable = remainder. The
area-rank rule was deliberately *not* used to separate MyCar from Road, because those two differ by
only ~2 pp of area and a rank rule would decide the canonical order on a near-tie.

| idx | self-detected | area | row centroid | temporal IoU | bytes | bit share | enrichment | ΔS if zeroed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | **Road** | 23.2331% | 239.3 | 0.9522 | 44,297.0 | 38.9334% | 1.68× | 0.029496 |
| 1 | **Lane** | 0.5858% | 226.4 | **0.2526** | 38,649.8 | 33.9700% | **57.98×** | 0.025735 |
| 2 | **Undrivable** | 49.5175% | **95.0** | 0.9942 | 12,893.7 | 11.3325% | 0.23× | 0.008585 |
| 3 | **Movable** | 1.2380% | 199.1 | 0.8528 | 11,876.8 | 10.4388% | 8.43× | 0.007908 |
| 4 | **MyCar** | 25.4255% | **334.6** | 0.9930 | 6,058.8 | 5.3252% | 0.21× | 0.004034 |

`class_order_selfcheck_all_match_canonical: true` — all five match the canonical comma10k order
`[Road, Lane, Undrivable, Movable, MyCar]`. The areas independently reproduce CLAUDE.md's n600
figures (23.2%/0.59%/49.5%/1.24%/25.4%), including the n600 Movable correction. The byte column
reproduces tb2's decoded-symbol table exactly.

Note the axis: this is the **decoded coded field**, not the DALI GT. tb2's class table is keyed on
GT; the two differ by the field/GT disagreements tb2 reports, not by rounding.

## 3. BIT MASS BY STRATUM — and the bulk-lever ceiling

| stratum | positions | bytes | share | **cheap complement** | complement bytes | **% of demand** | ΔS |
|---|---:|---:|---:|---:|---:|---:|---:|
| top 0.01% | 11,796 | 14,924.0 | 13.1170% | bottom 99.99% | 98,852.1 | 233.24% | 0.065818 |
| top 0.1% | 117,965 | 60,245.3 | 52.9507% | bottom 99.9% | 53,530.9 | 126.31% | 0.035644 |
| **top 1%** | **1,179,648** | **109,593.6** | **96.3238%** | **bottom 99%** | **4,182.6** | **9.87%** | **0.002785** |
| top 5% | 5,898,240 | 113,414.9 | 99.6824% | bottom 95% | 361.3 | 0.85% | 0.000241 |
| top 10% | 11,796,480 | 113,663.4 | 99.9009% | bottom 90% | 112.8 | 0.27% | 0.000075 |
| top 25% | 29,491,200 | 113,767.9 | 99.9927% | bottom 75% | 8.3 | 0.02% | 0.000006 |
| top 50% | 58,982,400 | 113,776.1 | 100.0000% | bottom 50% | 0.02 | 0.0001% | ~0 |

**The bulk-lever ceiling.** The 116,785,152 cheapest positions — 99% of the field — hold
**4,182.59 B in total**. Any lever acting on that bulk is capped there: **9.87% of the demand,
ΔS ≤ 0.002785**, short by 10.1×. The cheapest 90% hold **112.8 B**; the cheapest half hold
**0.02 B**. This number is not reported anywhere in the corpus and it is what makes the aggregate
refusals of `ad2` (addressing already free), `to2` (reordering, +196%) and `cx3` (named contexts,
+11,433 B) mechanistically unsurprising: those levers redistribute structure across all positions,
and 96.3% of the money is in 1%.

Cost distribution, read off the same sorted curve:

| threshold | positions above | frac | bytes above | share | mass at-or-below |
|---:|---:|---:|---:|---:|---:|
| `>1e-6` | 39,694,997 | 33.6499% | 113,775.8 | 99.9997% | 0.4 B |
| `>0.01` | 2,055,360 | 1.7424% | 111,977.9 | 98.4195% | 1,798.3 B |
| `>0.1` | 851,496 | 0.7218% | 106,809.9 | 93.8772% | 6,966.3 B |
| `>1` | 228,949 | **0.1941%** | **79,276.3** | **69.6774%** | 34,499.9 B |
| `>2` | 112,569 | 0.0954% | 58,923.7 | 51.7892% | 54,852.4 B |
| `>4` | 41,051 | 0.0348% | 34,044.7 | 29.9225% | 79,731.5 B |
| `>8` | 9,698 | 0.0082% | 12,905.3 | 11.3427% | 100,870.8 B |
| `>16` | 420 | 0.00036% | 974.1 | 0.8562% | 112,802.0 B |

min 2.687e-09 · median **1.008e-08** · p90 1.413e-04 · p99 4.536e-02 · **max exactly 31.0 bits**
(the `1/2^31` integer-frequency floor) · mean 0.007716 · zero-cost positions: **0**.

**The demand restated on the concentrated set:** 42,382 B = 339,056 bits = **37.25% of the entire
stream**. Sourced from the top 1% alone it requires cutting their cost by **38.67%**. Those
positions already code at **0.7432 bits/position** against a memoryless `log2(5) = 2.3219` — the
most expensive 1% is already 3.12× better than uniform.

## 4. COST × HARM — reproduced exactly, on a third independent computation

Recomputed from wj1's retained gross manufactured mask (28,602 positions, 0.024246% of the field,
carrying 6,846.84 B = 6.018% of stream bits) with this arm's own top-k selection. The packbits
bit-order was **sourced** from wj1's retained writer (`bitorder="little"`, lines 167/500/502), not
guessed — popcount is bit-order invariant and cannot validate it; the order-sensitive validator is
the intersection count below.

| stratum | observed positions | expected | **count ×** | observed bytes | **bit × (baseline A)** | **bit × (baseline B)** |
|---|---:|---:|---:|---:|---:|---:|
| top 0.1% | 9,480 | 28.60 | **331.4448** | 45.75 | **366.0268** | 780.8886 |
| **top 1%** | **26,016** | 286.02 | **90.9587** | **6,841.83** | **257.4799** | 99.9269 |
| top 5% | 28,355 | 1,430.10 | 19.8273 | 6,846.42 | 248.9863 | 20.0000 |
| top 10% | 28,523 | 2,860.20 | 9.9724 | 6,846.83 | 248.4422 | 10.0000 |

**wj1 reproduction check: EXACT.** positions 26,016 vs 26,016 (`positions_exact_match: true`);
bits relative error **3.99e-16**; bit-enrichment relative error **1.32e-15** — machine precision.
The pattern-of-patterns claim is **CONFIRMED at token-bit granularity**, now on three independent
computations rather than one plus two citations.

**Two baselines, because the floor you divide by decides the answer.** Baseline A (wj1/tb2
convention): randomly-placed manufactured positions each carry the *stratum's mean* cost.
Baseline B (mass-share): the manufactured bit mass spread in proportion to position share. They
disagree by 2.58× at top-1% (257.48 vs 99.93) and by 24.8× at top-10% (248.44 vs 10.00 — where B
is 10.0000 by construction, since the stratum captures ~100% of the manufactured mass). Both are
legitimate answers to different questions. **The "257×" headline is baseline-dependent and must
travel with its baseline.** This is the `pc2` genus firing again, not a new defect.

Boundary tie exposure is negligible: 1 / 1 / 3 / 23 field-wide ties at the four cut values, so the
top-k selection is unambiguous.

**Precision on the charter's `>10×` clause:** confirmed at top-1% (90.96× count, 257.48× bits) and
on bits at top-10% (248.44×). On *count* at top-10% it is **9.9724×**, marginally below 10×. Stated
plainly rather than rounded up.

## 5. THE SELECTOR-COST CEILING — the new result

Any lever that treats a chosen subset differently must identify it. For a subset of size `m` out of
`N = 117,964,800` the i.i.d. indicator reference cost is `N·H(m/N)` bits. **Prize** below is the
maximally generous assumption: every bit the subset holds, i.e. the subset coded for exactly zero.

| threshold | set positions | prize (B) | prize % demand | selector ref (B) | **net ceiling (B)** | prize/selector |
|---:|---:|---:|---:|---:|---:|---:|
| `>0.01` | 2,055,360 | 111,977.9 | 264.21% | 1,868,546.9 | **−1,756,569.0** | 0.060 |
| `>0.1` | 851,496 | 106,809.9 | 252.02% | 910,208.4 | **−803,398.5** | 0.117 |
| `>0.5` | 381,118 | 92,944.6 | 219.30% | 462,785.2 | **−369,840.6** | 0.201 |
| `>1` | 228,949 | 79,276.3 | 187.05% | 299,076.3 | **−219,800.1** | 0.265 |
| `>2` | 112,569 | 58,923.7 | 139.03% | 161,470.9 | **−102,547.2** | 0.365 |
| `>4` | 41,051 | 34,044.7 | 80.33% | 66,354.3 | **−32,309.6** | 0.513 |
| `>8` | 9,698 | 12,905.3 | 30.45% | 18,199.4 | **−5,294.1** | 0.709 |
| `>16` | 420 | 974.1 | 2.30% | 1,026.0 | **−51.8** | 0.949 |
| `>24` | **25** | 83.2 | 0.196% | 73.8 | **+9.45** | 1.128 |
| `>30` | 3 | 11.6 | 0.027% | 10.0 | **+1.60** | 1.160 |

**The family's maximum net is +9.45 B — 0.0222% of the demand, ΔS +6.289e-06, short by 4,487×.**
The ratio column shows why: as the set shrinks the prize falls faster than the selector cost until
`m ≈ 25`, where break-even arrives on a set too small to matter.

**Calibration against a MEASURED anchor.** `ddm_ae1` built real explicit flags for its 93,580
anti-predicted positions and measured **130,228 B**.

| quantity | value |
|---|---:|
| i.i.d. reference for m = 93,580 | **137,351.94 B** |
| ae1 MEASURED flag cost | **130,228 B** |
| measured / reference | **0.9481** |
| reference-predicted net | −110,706.64 B |
| ae1 REPORTED net | **−103,582.70 B** |

The reference lands within **5.2%** of the measured selector cost and **6.9%** of the measured net.
So this indicator is only ~5% compressible — the expensive set is close to i.i.d. as an indicator,
which is *why* the tax is nearly irreducible. The reference is therefore an **empirically
calibrated estimate**, not a strict bound: a correlated indicator can be coded below `N·H(p)`, and
ae1's was, by 5.2%. A future arm that beat it by 5.2% at `>8` bits would still net −4,347 B.

### The colex-rank counter-attack, run and failed

The obvious objection is that a *combinatorial* encoding beats the i.i.d. indicator — this is
CLAUDE.md L31, PR101's canonical `SIDECAR_NOOP_INFER_RANK_LEN = 3 B` colex-rank trick, and a reader
should raise it. I priced it exactly, as `log2 C(N, m)` via `lgamma`:

| threshold | m | prize (B) | i.i.d. `N·H` (B) | **colex `log2 C(N,m)`** (B) | net i.i.d. | **net colex** |
|---:|---:|---:|---:|---:|---:|---:|
| `>1` | 228,949 | 79,276.3 | 299,076.3 | 299,075.0 | −219,800.0 | **−219,798.7** |
| `>8` | 9,698 | 12,905.3 | 18,199.4 | 18,198.4 | −5,294.1 | **−5,293.1** |
| `>16` | 420 | 974.1 | 1,026.0 | 1,025.3 | −51.9 | **−51.2** |
| `>24` | **25** | 83.2 | 73.8 | 73.3 | +9.4 | **+9.90** |

Colex is cheaper by exactly `0.5·log2(2π·m·p(1−p))` bits — **10.2 bits at m = 228,949**, against a
2.39-million-bit selector. It moves the family's best net from **+9.45 B to +9.90 B**: still
**0.0234% of the demand, ΔS +6.59e-06, short by 4,280×**. The counter-attack was run and it fails.
The verdict in §5 stands under the strongest position-set encoding this campaign knows.

## 6. THE SHARPNESS QUALIFIER — what the five arms did not test

**The invariant across every prior arm that produced a byte number: `N` was held at exactly
117,964,800.** Not one changed how many positions are coded.
*SEARCH SCOPE: the nine memos `ddm_{oe1,ld1,ae1,ni1,cx3,ad2,to2,ef1,jf1}` (heads read in full plus
the sections cited here), plus `ddm_ar1b`, `ddm_wj1` and `ddm_tb2` read in full. I did not read all
154 dated memos; tb2 did, and reported no additional context-aligned per-position map.*

| arm | direction tested | field | model | order | measured |
|---|---|---|---|---|---|
| oe1 | causal escape MEMBER added to alphabet | fixed | extended | fixed | +10,818 … +12,305 B |
| ld1 | lossy Lane→Road edits, 6 rungs | **changed** | fixed | fixed | +21 … +1,528 B |
| ae1 | explicit stored flags / static overlays | fixed | +side info | fixed | net −103,583 / −14 / −34.5 B |
| cx3 | named causal conditional-entropy contexts | fixed | **changed** | fixed | +11,433 B |
| ef1 | generic estimators (PPMd, ZPAQ) | fixed | **replaced** | fixed | +251,545 B |
| to2 | 9 lossless orderings × 3 generic coders | fixed | replaced | **changed** | +223,087 B |
| ad2 | addressing / assignment layouts | — | — | changed | 0 B on dx2 (already free) |
| jf1 | joint field+model refit (epoch 2 of 60) | **changed** | **refit** | fixed | +2,715 … +7,103 B |
| ni1/nr1 | K32 whole-body representation | **changed** | changed | — | 122,250 B archive; distortion 247.69×/349× over bar |

Untested directions, each with the map's prediction and its reason:

| # | untested direction | map's prediction | ceiling (B) | ΔS | reason from the bit map |
|---|---|---|---:|---:|---|
| **D1** | explicit position-selected treatment at *any* cost threshold (ae1 generalized) | **SHARP — closed by arithmetic** | **+9.45** | +6.3e-06 | §5: selector cost exceeds the prize at every threshold; the set is too small to name |
| **D2a** | reduce coded support `N` by dropping CHEAP positions | **SHARP — insufficient by 10.1×** | 4,182.59 | 0.002785 | 99% of positions hold 3.68% of bits |
| **D2b** | reduce coded support `N` by dropping EXPENSIVE positions | **SHARP — task collapse** | >79,276 | >0.052 | that set is 90.96× enriched in manufactured seg error and holds 99.93% of its bit mass; = ni1/nr1, closed 247.69×/349× over bar |
| **D3** | alphabet reduction **in the model** (retrain on 4 symbols after a class merge) | **the one ceiling that approaches demand** | **38,649.8** | **0.025735** | Lane holds 33.97% of bits at 0.59% area; ld1 only tested this as a FIELD edit under a FIXED model |
| **D4** | probability-model precision (the `1/2^31` frequency floor) | **SHARP and wrong-signed** | 974.1 | 0.000649 | the entire `>16`-bit tail is 0.86% of the stream; coarser frequencies raise cost |
| **D5** | pair-selective coding | **not sharp — EMPTY** | — | — | tb2 measured no pair concentration: top pair 0.329%, top 100 pairs 20.82% |
| **D6** | free (receiver-regenerable) selectors | **SHARP — budget already spent** | — | — | the only family escaping D1's tax; the shipped HPAC context already consumes the best free selector (predicted class, t−1/t−2 agreement, boundary bucket). oe1 and cx3 tested it and lost |

D5 deserves its own label: it fails for **lack of structure**, not for a steep penalty. That is a
different verdict from "sharp" and should not be recorded as one.

**D3 is the only untested direction whose ceiling is not arithmetically excluded**, and it is the
one this arm most wants a successor to take. It is not a recommendation and carries no projection.
**Falsifier for any D3 charter:** a model retrained on a reduced-alphabet field must (a) beat
113,777 B on a real re-encode by >0 B with its own model cost counted, and (b) have its `d_seg`
measured on MAIN's authority lane — Lane is the worst distortion class (IoU 0.2526, ~19% of flips),
so a byte win there is not a score win and must never be called one.

## 7. WHY EVERY DIRECTION IS SHARP — the one-sentence mechanism

The stream's bit mass and its task-relevant mass occupy **the same 0.2% of positions**. That set
can be escaped in exactly two ways, and the map closes both: **name it** and pay a selector tax that
exceeds what it holds (§5, calibrated on ae1's measurement), or **drop it** and lose the argmax the
score is computed on (ni1/nr1, 247.69×/349× over bar). The sharp-optimum law five arms measured in
five directions is, on this evidence, one geometric fact seen five times.

## 8. WHAT IS NOT CLAIMED

- **The bit map is a MAP, not a PRICE.** `ddm_fs2` measured `−log2 p` prices 0.77–0.88× wrong away
  from argmax and 0.09× toward it. Every ceiling here is an upper bound read off the coder's own
  costs. **No rate claim is made from the model; a rate claim is a measured re-encode.**
- No lever is nominated, no coder is raced, no bytes are projected onto any archive.
- The selector-cost leg is `FORMULATION`-scoped to explicit position-selected subsets priced at the
  i.i.d. reference. It does **not** close free/receiver-regenerable selectors (D6) — those are
  closed, separately and only at `FORMULATION` scope, by oe1's and cx3's measurements.
- The `N`-invariance claim in §6 carries the search scope stated there. It is not a global
  nonexistence claim.
- No `d_seg`, `d_pose`, or `S` was measured. The manufactured-error mask is wj1's inherited gross
  native-render support, and "not manufactured" is not "render-correct."

## 9. PRIOR-LAW ADJUDICATION

| pre-registered statement | measured | verdict |
|---|---:|---|
| bit mass concentrated: top 10% carries **>70%** of bits | **99.900879%** | **CONFIRMED** |
| expensive mass enriched in manufactured-error positions **>10×** over independence | top-1%: **90.9587× count / 257.4799× bits** | **CONFIRMED** |
| falsifier: near-uniform (top-10% < 25% of mass) | 99.90% | **NOT FIRED** |
| falsifier: enrichment < 2× | 90.96× / 257.48× | **NOT FIRED** |

Both legs confirmed. Per the charter's own instruction, the honest reading is stated in its words:
**this is a CONFIRMED concentration with no actionable direction.** The map does not open an escape
from the sharp optimum; it explains the optimum, and it converts three untested directions
(D1, D2a, D4) from "unknown" to "closed or insufficient" at $0, which is the value delivered.

## 10. CUSTODY

Inputs, all read-only and re-hashed this run:

| object | bytes | SHA-256 |
|---|---:|---|
| tb2 per-symbol cost field | 943,718,400 | `99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86` |
| wj1 copy of the same field | 943,718,400 | `99d7833d…` — **byte-identical, verified** |
| decoded categorical field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` (charter pin) |
| wj1 gross manufactured mask | 14,745,600 | `b756ca948f5db3dd…` |

Retained payloads, this arm's store
`/Volumes/APDataStore/pact/ddm_tba1_token_bit_attribution/`:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `derivation_v1/retained/cost_bits_cumsum_asc.f64le.bin` | 943,718,400 | `7d142ed575bf6482d77310694733edd9142bb81cd32c8c077675e9fad7a58205` |
| `derivation_v1/retained/cost_rank_ascending.i32le.bin` | 471,859,200 | `348de943abde66f32b9925349083baa24aaf9dec7a78ec3da6ea6df8e4192b73` |
| `derivation_v1/RESULT.json` | — | `d826064fedcd4e67abb2f2947563fae68df7ee498656c32b1fd95dcc1d39321c` |
| `sharpness_v1/retained/top1pct_expensive_mask.n600.packbits` | 14,745,600 | `e4fc14ae657d7148e0e8d3a33064bb047a27a18882cf55e83fb96af01ac976bb` |
| `sharpness_v1/retained/top1pct_expensive_and_manufactured.n600.packbits` | 14,745,600 | `fe0020736d8a1dc8fc299d2bdc748030905697daffec267786c28b920b4195f1` |
| `sharpness_v1/RESULT.json` | — | `a197484714226f2848fd29259decaab8d9133f04ce3d9663a1af812916d686c3` |

**P0 — ALWAYS KEEP THE PAYLOAD, and the one judgement call made against it.** The per-symbol
`−log2 p` vector is retained **byte-identically in two independent prior stores** (`ddm_bl1` and
`ddm_tb2`, sha `99d7833d…`), both re-verified by this run. A third identical copy would add 944 MB
and zero signal, so it was not duplicated — the bytes are not lost, which is what the rule
protects. What this arm *materialised* is persisted in full: the Lorenz cumsum (every quantile in
§3 is read off it), the ascending cost rank (the join key a successor needs to intersect this
ordering with any other per-position field without re-sorting 118 M positions), and both
expensive-set masks. `cost_bits_sorted_asc` is **not** re-persisted and is certified rebuildable
exactly as `cost[argsort(cost)]` from the pinned field plus the retained rank. This reasoning is
recorded here and in both `RESULT.json` files so it is auditable rather than silent.

Exchange rate **6.658590e-07 S/B** is CITED from `ddm_tx1_toolbox_crosswalk_20260819.md` §0, not
re-derived. Cross-check: 42,382 B × 6.658590e-07 = **0.0282204**, reproducing fb1's 0.028220 gap
from the opposite direction.

Instruments: `experiments/ddm_tba1_bulk_ceiling_derivation.py`,
`experiments/ddm_tba1_sharpness_arithmetic.py`. Both ruff-clean, two review passes each; pass 1
found the class-detector leaning on the MyCar/Road area near-tie, twelve 944 MB copies in the
strata loop, and a redundant persist; pass 2 found a missing storage preflight and a missing
partition cross-check; the first run then **refused** on a bit-order probe that could not
discriminate (popcount is bit-order invariant), which forced sourcing the convention from wj1's
retained writer instead of guessing. Each of those was a real defect caught before it produced a
number.

`STORES CONSULTED: /Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1 (read-only, cost + symbol fields); /Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join/measurement_v1 (read-only, gross manufactured mask + JOIN_RESULT.json + retained provenance source); .omx/research/ddm_{fb1,ar1b,tb2,oe1,ld1,ae1,ni1,cx3,ad2,to2,ef1,jf1,wj1,tx1,tl1,sy2,w72}*.md; .omx/research/charters/ddm_{tba1,tb2}_token_bit_attribution_20260823.md; git objects 9c137a91ed / e864cb4ab4 / fe2ba12dc2 / 637af0c8c1 (pin verification); ddm_mst1 receipts: not read (wj1's derived mask used instead); ddm_bl1 store: not read (its field is byte-identical to tb2's, verified by sha); scorer store: none; Modal: none; Metal: none; ddm_jf1 receipts: NOT TOUCHED (sacred).`

## 11. NEXT_IF_RESUMED

- **D3 is the only untested direction whose ceiling (38,649.8 B, ΔS 0.025735) approaches the
  demand.** A successor charter must retrain the model on a reduced-alphabet field — ld1 only
  tested the field edit under the frozen model — and must carry the §6 falsifier: a real re-encode
  beating 113,777 B with model cost counted, AND an authority-lane `d_seg`. Lane is the worst
  distortion class; a byte win there is not a score win.
- **Do not charter D1, D2a, or D4.** §5 and §3 close or bound them at $0. Any charter proposing an
  explicit per-position selector on this stream should be refused with the §5 table.
- The retained `cost_rank_ascending.i32le.bin` is the join key for any future per-position
  intersection on this object; it removes an 8-second 118 M-element sort from every successor.

**Own-vehicle frontier: UNMOVED — dx2, S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`,
archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.** This arm fired
no scorer and built no candidate; it spent $0 and closed three directions by arithmetic.
