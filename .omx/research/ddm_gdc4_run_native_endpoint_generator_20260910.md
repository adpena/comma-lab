# ddm_gdc4 — the run-native endpoint generator, priced at its own oracle — n600 receipt, 2026-09-10

`research_only=true` · `score_claim=false` · `promotable=false` · `frontier_moved=false`
Axis: `[macOS-CPU byte-only n600]` throughout. No scorer, no training, no Modal, no candidate
archive, no MPS number, no burn.

Frontier line unchanged by this arm:

`composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`

## Verdict

**FORMULATION-NO-GO for the learned run-native endpoint generator, refused before the burn by its own
arithmetic.** All three of the charter's early-stop screens — packet `K <= 60,000 B`, mismatches
`M <= 65,000`, `>= 90 %` of errors long-and-boundary-adjacent — are falsified by measurement on the
real n600 field, and the family's *best possible* member misses the `94,010 B` door by **2.50x**.

The burn was not launched. Launching it would have spent hours to land inside a region an oracle
already bounds, using renders already sitting on the SSD.

Three measured facts carry the verdict.

1. **gdc1 was already the perfect-accuracy instance of this vehicle.** This arm computed from scratch
   the optimal bounded-endpoint approximation `M(E)` of every one of the 230,400 rows — the fewest
   cells any generator emitting at most `E` `(x_stop, class)` symbols per row can get wrong. It
   reproduces gdc1's retained scanline family **to the unit** on gdc1's own field:
   `E=8 -> 88,304`, `E=12 -> 9,311`, `E=16 -> 561`, `E=24 -> 0`. gdc1's `K` *is* the endpoint budget
   `E`, and gdc1 sat on the oracle at every one. No training run can be more accurate.
2. **The family's whole `K + R_exact` curve is above the door, and it is monotone.** Its minimum is
   at `E=24` — the exact field, zero residual — at **235,087 B = 2.50x the door**. There is no
   interior minimum: with a real coder, every byte spent buying accuracy saves more residual than it
   costs, which inverts the ordering gdc1's explicit per-row packet produced.
3. **A second, independent factorization lands in the same place.** A dense 2-D causal-context model
   on the identical field costs **229,528 B**. Run-native and dense agree to **2.4 %**. Runs are not
   a better description of this field; they are the same description plus an addressing tax.

## A provenance defect found and corrected: the gdc chain has been measuring the wrong field

The charter pins the target as `…/ddm_sj1_pass6/retained/fields/pass6.u8`, sha256 `78e57545…`, and
gdc3 states "Move 44 inherits this field byte-identically". **It does not.** MEASURED here,
first-hand:

| object | sha256 | relation |
|---|---|---|
| `…/ddm_sj1_pass6/retained/fields/pass6.u8` (the charter's pin, and gdc1/gdc2/gdc3's) | `78e57545439515eb29f806cc5a5f7d8b14acf658955cdd7561debb4edf3b7db6` | the full 251-pair pass-6 edit, **never shipped** |
| `…/ddm_sj1_pass6/retained/fields/subset6.u8` | `a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8` | **the shipped field** |
| `…/ddm_rlc5_cure_on_move43/move42/field.u8` (what the move-44 encoder bound) | `a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8` | byte-identical to `subset6.u8` |

`pass6.u8` and the shipped field differ in **154 cells across 90 frames**, first at `[12, 181, 253]`.
The shipped archive is the Lagrange-admitted *subset* (161 pairs), not the full pass-6 edit.

An independent confirmation fell out of this arm's own instrument: gdc1's `K=24` render, which is
*exact* against `pass6.u8` (`M = 0`), measures `M = 154` against the shipped field — the same 154
cells, found by a different route.

**Every load-bearing number in this memo was re-measured on the shipped field.** The verdict does not
move: 154 cells are 1.3e-6 of the field, and every quantity shifts by under 0.16 %.

| quantity | on `pass6.u8` | on the shipped field | delta |
|---|---:|---:|---:|
| total runs | 584,354 | 584,128 | −226 |
| run-native adaptive packet (exact) | 235,449 | **235,087** | −362 |
| dense context floor | 229,663 | **229,528** | −135 |
| oracle `M(8)` | 88,304 | 88,235 | −69 |
| oracle `M(12)` | 9,311 | 9,279 | −32 |

The correction matters for the chain, not for this verdict: gdc1's door derivation, gdc2's `R(M)`
roster, and gdc3's geometry table were all computed against `78e57545…`. Their conclusions are large
multiples of the door and are not at risk from 154 cells, but any successor that byte-closes against
the shipped archive must pin `a92e7d90…`. Genus: `[[available-field-vs-authoritative-field]]`.

## Authority, source and provenance

- Target field (corrected): `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8`,
  117,964,800 B, sha256 `a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8`.
  Charter-pinned field also measured throughout for continuity with the chain.
- Oracle renders (read-only): `/Volumes/VertigoDataTier/pact/ddm_gdc1_generator_door/scanline_v1/k{04,06,08,12,16,24}/receiver_render.u8`,
  117,964,800 B each; `k08` sha256 `abd130921beec23255112e8b56148d55c49cf4fede03e99e0e99b55378b16ea2`,
  matching gdc2/gdc3's pin. All six shas are in this arm's `MANIFEST.json`.
- Charter `.omx/research/ddm_gdc4_run_native_endpoint_generator_charter_20260910.md`; governing memos
  gdc1 `c924bb6a…`, gdc2 `2f534b1d2ec91038…`, gdc3 `db4510fdae22e343…`.
- Custody: `/Volumes/VertigoDataTier/pact/ddm_gdc4_run_native_endpoint_generator/`, every raw stream
  beside every coder output, `MANIFEST.json` with a sha256 per row. 83 GiB free at close. Nothing
  deleted; no predecessor tree written.
- Lane `ddm_gdc4_run_native_endpoint_generator_20260910`. Seed `20260910`. Code landed through the
  serializer with two review passes per Python file; 23 focused tests; ruff and py_compile clean.

## The door, restated at move 44

Read from the shipped archive sections, not assumed. Non-tail is unchanged from move 43, so gdc1's
gate stands.

| section | move 43 | move 44 |
|---|---:|---:|
| non-tail (ZIP overhead + RX1M header + HPAC model + semantic renderer + carrier) | 60,497 | 60,497 |
| **replaceable tail** | **119,969** | **119,909** |
| archive | 180,466 | 180,406 |
| **gate** `P_program + R_exact` | — | **<= 94,010** |

The demand is therefore `119,909 - 94,010 = ` **25,899 B, or 21.599 %** of the tail.
(gdc1's memo quotes the move-43 tail, 119,969 B, and decomposes it loosely as
`119,833 + 35 + 100 + 1`; the true split is `96 B` RCF1 residual-table prefix + `40 B` mixer rider +
`119,833 B` RC64 stream. Same total, and the door is unaffected.)

## What the field actually is (MEASURED, full n600, shipped field)

| quantity | value |
|---|---:|
| horizontal runs | 584,128 |
| transitions (non-trivial endpoints) | 353,728 |
| runs per row: mean / median / p90 / p99 / max | 2.5353 / 1 / 7 / 13 / 23 |
| rows byte-identical to the row above | **70.79 %** |
| rows byte-identical to the previous frame's same row | **68.19 %** |

The field is far simpler in run terms than the vehicle assumed: a median row is a *single run*. That
was the encouraging part, and it is why the language itself never binds.

### The ORACLE bounded-endpoint floor

`M(E)` is computed per row by exact dynamic programming over run boundaries (breakpoints may be
restricted to run boundaries without loss, since the target is constant inside a run). It is
cross-checked against exhaustive enumeration in the test suite.

| `E` | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M(E)`, shipped field | 3,173,386 | 673,524 | 234,221 | 88,235 | 30,855 | 9,279 | 558 | 17 | 0 |
| `M(E)`, `pass6.u8` | 3,173,411 | 673,602 | 234,295 | 88,304 | 30,907 | 9,311 | 561 | 18 | 0 |
| gdc1's measured family `M` at `K=E` on `pass6.u8` | — | 673,602 | 234,295 | 88,304 | — | 9,311 | 561 | — | 0 |

Identity at every shared point. **gdc1's scanline family was this vehicle at its accuracy ceiling.**
Its 475,002 B exact packet was a coder problem, not a language problem — this arm codes the same
exact field in 235,087 B, **2.02x smaller**.

## The complete family curve (MEASURED, full n600, every row exact)

`K(E)` is this arm's free online context-adaptive code length on the oracle render (target-independent).
`R_exact(E)` is the residual gdc2 measured with exact field closure on those same retained renders —
cited, not re-derived.

| `E` | `M` | gdc1 explicit packet | **`K(E)` this arm** | `R_exact(E)` | **`K + R_exact`** | x door |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 673,602 | 133,426 | 94,962 | 174,680 | 269,642 | 2.868 |
| 6 | 234,295 | 223,494 | 144,220 | 105,628 | 249,848 | 2.658 |
| 8 | 88,304 | 306,042 | 184,964 | 60,520 | 245,484 | 2.611 |
| 12 | 9,311 | 421,886 | 227,361 | 11,772 | 239,133 | 2.544 |
| 16 | 561 | 459,394 | 234,566 | 1,238 | 235,804 | 2.508 |
| **24** | **0** | 475,002 | **235,449** | **0** | **235,449** | **2.505** |
| **24, shipped field, this arm end-to-end** | **0** | — | **235,087** | **0** | **235,087** | **2.501** |

Two readings worth keeping.

- **The curve inverts gdc1's.** Under gdc1's explicit per-row packet, `packet + R` rose monotonically
  with `K` and the family's best point was `K=4 at 308,106 B`. Under a real coder it *falls*
  monotonically and the best point is exactness at 235,087 B. gdc1's ordering was an artefact of its
  coder, the same way its family winner was once an artefact of a transferred residual rate (gdc2
  Finding 3). This is the third coder-or-constant artefact corrected in this chain.
- **The accuracy axis is closed absolutely.** Every row is an upper bound on what a trained generator
  could achieve at that budget, because the oracle caps accuracy by construction.
  (The `R_exact` column is gdc2's, measured against `pass6.u8`; against the shipped field each entry
  shifts by at most the cost of 154 cells. The bottom row avoids the caveat entirely: it is this
  arm's own end-to-end exact number on the shipped field, with no residual at all.)

## The three charter screens, each falsified

| screen | charter value | MEASURED | verdict |
|---|---:|---:|---|
| packet `K` | `<= 60,000 B` | 184,964 B at `E=8`, the coarsest budget near the `M` screen | **3.08x over** |
| mismatches `M` | `<= 65,000` | reachable only at `E >= 9`, where `K >= 184,964 B` | unreachable jointly |
| error geometry | `>= 90 %` long-and-boundary-adjacent | **6.44 %** long-boundary; **81.42 %** short-boundary | **inverted** |
| projected residual at `M = 65,000` | 32,174.14 B | 44,548.7 B at the realized 0.685364 B/mismatch | **1.384x optimistic** |
| total | `<= 94,010 B` | 235,087 B family minimum | **2.50x over** |

The geometry row deserves its own sentence, because it is the fourth borrowed rate constant in this
chain that did not survive contact with its own vehicle. The charter screened the residual at
0.494987 B/mismatch — the *long-boundary* rate — on the premise that an endpoint generator's errors
would be long boundary runs. gdc3's own table, on the `E=8` oracle, which **is** this vehicle at that
budget, shows 5,685 long-boundary errors out of 88,304 (6.44 %) against 71,893 short-boundary
(81.42 %). A bounded-endpoint approximation errs in *short* runs, because truncating a row's run list
drops short runs first. The premise was inverted on evidence that already existed when the screen was
written. Same genus as `[[cross-regime constant transfer]]` and
`[[binding-instruction-numbers-expire-and-nobody-rederives-them]]`.

## Why the representation does not help: two factorizations, one wall

Four exact constructions, each decoding byte-identically to the source in about a second on the
declared host — far inside the 1,260 s receiver budget.

| construction | counted bytes | x door | x move-44 tail |
|---|---:|---:|---:|
| v1 run-mode packet (copy/delta/explicit modes; best of brotli-q11 / LZMA2-extreme / zlib-9 per stream) | 286,527 | 3.048 | 2.390 |
| v2 transition-edit packet (MATCH/SUBST/INSERT/DELETE against the two free causal references) | 259,668 | 2.762 | 2.166 |
| v2 script under a free online context-adaptive coder | **235,087** | **2.501** | **1.961** |
| dense 2-D causal-context model on the same field (left, up, up-left, up-right, previous frame) | **229,528** | 2.442 | 1.914 |
| *the incumbent shipped tail, for reference* | *119,909* | *1.275* | *1.000* |

The two factorizations agree to 2.4 %. That is the finding: **runs are not a better description of
this field.** The dense coder never has to say *which* transition maps to which; the run coder pays
68,576 B for exactly that (44,497 B of op script + 24,079 B of inserts) and does not earn it back on
the positions. This is gs3's gestalt made quantitative on a new object: every construction so far
pays for the surprise the coder cannot remove by ADDRESSING it.

### The endpoint jitter is near-irreducible

`dx`, the endpoint displacement against the causal reference row, costs 157,422 B — **1.674x the
entire door on its own**, from 330,003 symbols. Eight predictors were raced against it:

| scheme | bits/symbol | vs order-0 |
|---|---:|---:|
| order-0 | 4.391 | — |
| conditioned on the previous row's `dx` (±3 buckets) | **4.255** | **−3.1 %** |
| conditioned on the previous row's `dx` (±8 buckets) | 4.463 | +1.6 % |
| second difference `dx − dx_prev` | 4.363 | −0.6 % |
| second difference, conditioned | 4.468 | +1.8 % |
| second difference, conditioned on transition index | 4.409 | +0.4 % |
| conditioned on the previous frame's same-row `dx` | 4.509 | +2.7 % |
| predicted from the previous frame's same-row `dx` | 4.816 | +9.7 % |

Every scheme lands within 3.1 % of doing nothing, and both slope extrapolation and temporal
prediction are *worse*. That is the signature of an irreducible quantity, not of a weak
implementation: the argmax boundary of a segmentation is jagged at the pixel scale, so row-to-row
endpoint displacement carries no exploitable slope. (This eight-way race is a 120-frame contiguous
diagnostic — the codec is causal over frames, so a prefix is the only coherent subset; the
load-bearing `dx` number, 3.284 bits/symbol plus escape, is full n600.)

### Where the bytes are: Lane

| | symbols | share | bytes | share |
|---|---:|---:|---:|---:|
| `dx` at Lane-adjacent transitions | 214,800 | 65.09 % | **93,995** | 59.20 % |
| `dx` everywhere else | 115,203 | 34.91 % | 64,790 | 40.80 % |

**Lane boundary jitter alone costs 93,995 B — 99.98 % of the whole 94,010 B door — while Lane is
0.59 % of the field's area.** Inserts, by contrast, are mostly not Lane (17,288 of 23,951): new
boundaries are born elsewhere, but Lane's existing boundaries wander every row. Independent
agreement: the incumbent's own per-class ideal-bits table prices class 1 at 327,892 bits = 40,987 B,
**34.1 %** of its stream, at 0.473823 bits/symbol against 0.0018 for classes 2 and 4
(`.omx/research/ddm_tc1_token_tail_bound_and_shared_mixer_pricing_20260909.md`). Two unrelated
factorizations, one concentration — the same one MD1–MD4 and `[[LANE 0.59% area, 33.56% bits]]`
record.

## Verdict scope, and what would reopen it

- **Accuracy axis — FAMILY scope, closed.** The oracle DP caps accuracy for *any* generator emitting
  at most `E` endpoints per row, learned or not. It is exact, computed independently here, verified
  against exhaustive enumeration, and it agrees with gdc1's six retained renders to the unit. Nothing
  reopens this short of leaving the bounded-endpoint output language.
- **Rate axis — FORMULATION scope, under this arm's coder class.** The family's minimum is 235,087 B
  with a free online model. A learned generator must beat that by 2.50x *and* pay for weights the
  online model gets for nothing. What would reopen it: a coder that takes the run factorization from
  235,087 B to 94,010 B. The evidence against is that the dense factorization under a matched
  instrument lands within 2.4 % of the same number, and the incumbent — which already runs a learned
  neural prior plus free adaptive mixing on the dense side — still only reaches 119,909 B, 1.275x
  over the door, before a single additional counted weight.
- **Not measured:** `d_seg`, `d_pose`, contest score, candidate archive size, contest receiver time.
  No scorer, burn, Modal, upstream edit, sealed-tree edit, or predecessor-tree write occurred.

## What this hands the successor

The door has not moved, but it is now stated in the only terms that matter — bits per token cell,
against the incumbent's own architecture.

| | bytes | bits per cell | vs door |
|---|---:|---:|---:|
| the door | 94,010 | 0.006375 | 1.000 |
| the incumbent move-44 tail | 119,909 | 0.008132 | 1.275 |
| best dense single-context model built here | 229,528 | 0.015568 | 2.442 |
| best run-native model built here | 235,087 | 0.015943 | 2.501 |

**The remaining 21.599 % must come from the MODEL, on the dense token factorization where the
incumbent already lives — not from a representation change to runs.** That is measured, not argued:
the two representations agree to 2.4 % under matched instruments, and the incumbent's ~1.9x advantage
over both comes from mixing, not from its symbol space.

But the successor must not inherit an optimistic version of that pointer, so the honest ceiling
belongs next to it. The incumbent is **not** a naive coder waiting to be improved: its tail is an
RC64 five-symbol range coder driven by a sparse HPAC neural prior (190 groups, intra-frame cascade
plus inter-frame conditioning), a counted 25x5 fixed boundary-residual table, a **free** adaptive
corrector over 23 predictor families, and a counted mixer rider (40 B at move 43; 64 B at move 44,
`RLC1` + 40 int8 weights + 19 B geometry). And tc1 already priced the obvious next lever: its
35-weight counted mixer realized **−589 B**, which is 81.46 % of that family's numerical ceiling but
only **6.09 %** of the Miller–Madow joint oracle over its five added contexts — and that joint
oracle's own optimistic net is **−9,011 B**, i.e. about **35 %** of the 25,899 B demand, before the
1,561,144 B a dense table for it would cost. So "mix more contexts" is measured to fall short by
roughly 3x on its own.

That leaves the sharpest form of the remaining question, which no arm in this chain has yet answered:
**where does 25,899 B of Lane-conditioned surprise live that the HPAC prior plus a 23-family free
mixer does not already remove?** Lane is 0.59 % of the area and 34.1 % of the incumbent's stream
bits; a successor that buys 21.6 % almost certainly buys it there, and it should be charged against
tc1's measured mixing ceiling from the first line of its charter rather than against a fresh
projection.

## RECALL EVIDENCE

Searched before and during the work: the charter and the gdc1/gdc2/gdc3 memos in full; gs3 Addenda
24–27; `.omx/state/main_hot_state.md` and the active lane-dispatch ledger (no duplicate gdc4 owner;
lane claimed before any work); `.omx/research/` by content for `119,969 | token tail | mixer |
horizontal run | scanline | endpoint | boundary-adjacent | residual geometry`; the shipped runtime
under `submissions/_staging_move44_pr140_swap/`; and the retained gdc1/gdc2/sj1/rlc5 SSD trees
(read-only). Findings that changed the work:

- The shipped tail's real architecture (RC64 + sparse HPAC neural prior + counted fixed table + free
  23-family corrector + counted mixer rider) and tc1's measured mixing ceiling turned the successor
  pointer from "improve the model" into "improve the model, charged against a measured −9,011 B joint
  oracle that is only 35 % of the demand". Without that recall this memo would have handed MAIN an
  optimistic lever.
- gdc2 Finding 2's `R_exact` roster, measured with exact closure on the same retained renders, let
  this arm price the family curve without re-measuring a settled quantity.
- gdc3's per-geometry table on the `E=8` oracle supplied the 6.44 % long-boundary share that
  falsifies the charter's own geometry screen — the evidence was already in the predecessor memo.
- The `pass6.u8` / `subset6.u8` split surfaced only because the successor question forced a read of
  what the shipped encoder actually binds. It was then verified first-hand by byte comparison, not
  taken on report.

I did not find a prior byte-closed pricing of the *optimal* bounded-endpoint approximation of this
field. That scoped absence is why the oracle curve, not another training run, was the decisive
instrument.

## Verification and boundaries

- Full population: 600/600 frames for every number reported as n600 — the run census, the oracle floor
  over all 230,400 rows, all four exact packets, the dense floor, all six curve points, and the Lane
  split. Every one re-run on the shipped field after the provenance correction.
- Exactness: every packet decodes byte-identically to its source field; re-encode is byte-identical;
  decoders fail closed on stream residue and on a truncated mode stream (both tested).
- Cross-validation: the oracle DP is checked against exhaustive enumeration in the test suite, and
  independently against gdc1's six retained renders.
- Coders: brotli q11, raw LZMA2 extreme, zlib 9, raced per stream; plus a sequential online model
  whose cost is a real code length, not a compressed file.
- Tests: 23 focused tests pass; ruff and py_compile clean; two review passes per Python file; one
  defect found and fixed in the second pass — the `dx` escape model clamped magnitudes at 63, which
  *understated this arm's own cost*; corrected before any number was quoted (230,035 -> 235,449 B on
  `pass6.u8`).
- Retention: every raw stream beside every coder output, sha256 per row in `MANIFEST.json`.

## LIVE-HYPOTHESES

- The door is a **model** problem on the dense token factorization, localized to Lane. Lane is 0.59 %
  of the area, 34.1 % of the incumbent's stream bits, and 59–65 % of the boundary cost in this arm's
  unrelated factorization. A successor that buys 21.6 % buys it there.
- Whether 25,899 B of Lane-conditioned surprise survives the HPAC prior plus the free 23-family mixer
  is genuinely open. tc1's joint oracle says the *specific* five added contexts hold at most ~9,011 B
  of it; that bounds one family, not the question.

## DEAD-ENDS

- **Learned run-native endpoint generators** (this charter): the accuracy axis is capped by an oracle
  already on disk, and the family's best `K + R_exact` is 235,087 B = 2.50x the door.
- **Bounded-endpoint accuracy tuning in general**: `M(E)` is exact and gdc1 already sits on it at six
  budgets. There is nothing to win by training toward it.
- **Endpoint-displacement prediction**: eight predictors, all within 3.1 % of order-0; slope
  extrapolation and previous-frame prediction are worse. The argmax boundary carries no exploitable
  row-to-row slope.
- **Runs as a better factorization than cells**: measured 2.4 % *worse*, and the gap is an explicit
  68,576 B addressing tax the dense coder never pays.
- **Screening a residual with a borrowed geometry rate**: the charter's `>= 90 %` long-boundary
  premise was inverted (6.44 % measured) by a table its own predecessor had already published.
- **Pinning `pass6.u8` as the shipped field**: it is the never-shipped full pass-6 edit and differs
  from the shipped field in 154 cells.

<!-- # FORMALIZATION_PENDING: arm process memo; the oracle bounded-endpoint floor lands in the equations leg with the first construction that passes the door -->
