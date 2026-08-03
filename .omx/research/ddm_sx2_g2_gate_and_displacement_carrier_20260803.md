# `ddm_sx2` — G2 run, and the "displacement" hole measured instead of named

**arm:** `ddm_sx2` · **date:** 2026-08-03 · **axis:** `[macOS-CPU advisory]` — **NO scorer run.**
`score_claim=false` · `promotion_eligible=false` · `rank_or_kill_eligible=false` ·
`ready_for_exact_eval_dispatch=false`. Zero contest-scorer forwards. Substrate is the cached GT
SegNet argmax (`lstars`, n600, full population) and GT `frame_1` pushed through the exact
`SegNet.preprocess_input` resize. `ddm_pu2` holds the scorer slot; every claim here that needs one
is queued in §7 with its command.

---

## §0 ANSWER FIRST

**G2 PASSES on its literal threshold and its CONCLUSION IS REFUTED.** Both halves are measured.

> `ddm_sx1` §10: *"report `rho` = fraction of the 2,551,382 boundary px located within ±1 px.
> `rho > 0.5` makes S1's occupancy map effectively free."*

The best generic contour extractor (Canny) reaches **`rho` = 0.6210** on the full n600 population —
above the threshold. But the occupancy map is **not** free: at the extractor's *best* budget it
removes only **28.1 %** of the 36,798 B cell-map term. And above that budget **`rho` and the
map's free-fraction move in OPPOSITE directions** — at 16× budget `rho` climbs to 0.716 while the
free-fraction collapses to 6.3 %. **`rho` is budget-inflatable; it does not gate what it was
asked to gate.**

**The control is the finding.** A **static per-cell boundary map, held out (built on even frames,
scored on odd), costing 49 B zlib-compressed, removes 57.2 % of the map term — 2.0× the best free
generic extractor.** Every blend of the two is *worse* than the static prior alone (union 49.1 %;
weighted blends 20.7 / 25.2 / 47.4 %). **The free generic algorithm has negative marginal value
once the 49 B table is present.** Counted-but-tiny beats free-but-generic.

**S2 is refuted as formulated, by direct measurement of the quantity it costed.** `sx1` priced S2
as `253,341 B · (1 − rho)`. That formula multiplies a *code length* by a *recall on boundary
location*; the two are not proportional. Measured with an order-4 context model on the same field:

| | bytes |
|---|---:|
| unconditional description of `L*` (my independent re-derivation) | **238,945** |
| + free Canny indicator admitted as side information | 222,114 |
| + free Canny indicator, dilated ±1 px | **216,395** |
| **bits the free predictor actually buys** | **22,550 B = 9.44 %** |
| what `253,341·(1−rho)` predicts it buys | **62.10 %** |

**Overstated 6.58×.**

**And the hole `sx1` named is 2.5–3.4× smaller than it said, and is the harder half, not the more
valuable one.** `sx1` split the seg residual boundary-PRECISION 76.4 % / object-DISPLACEMENT 23.3 %
from an edge-level proxy (`flips/len`). Joining `pc2`'s own per-edge ">3 px off" shares directly:

> **93.89 % of seg flips lie WITHIN 3 px of the GT separatrix. MODEL-FREE — a flip-share-weighted
> sum of `pc2`'s own numbers, no mixture, no reference, no direction assumption.**

The modelled object-level component is **6.89 – 9.17 %**. `sx1` conflated *the Movable edge is 2×
denser per unit length* (true, and it survives) with *the errors on it are not boundary-shaped*
(false — 76–78 % of them are). **The separatrix program reaches 91–94 % of the seg axis, not
76.4 %.** That is the single most useful number here and it is good news for `sx1`'s own carrier.

**Rigid translation is REFUTED as the displacement mechanism (FORMULATION level).** Two independent
inversions of the same unknown disagree **5.8×**: the distance profile says `d ≈ 4.93–5.14 px`, the
flip count says `d ≈ 0.860 px`. A single-magnitude rigid offset cannot produce both. The surviving
decomposition is a **mixture: ~24 % deep/object-level + ~76 % SUB-PIXEL boundary error at
`d ≈ 0.65 px`.** Sub-pixel displacement is a **phase** error, which converges with the independently
recorded deficit *"phase-faithfulness = per-pair POSITIONAL DOF; `tokens_delta` gives AMPLITUDE
only."* Two instruments, one missing DOF.

---

## §1 CONSTANTS — recomputed from source in this session, never re-typed

| quantity | value | derivation |
|---|---:|---|
| `DEN` | 37,545,489 | `upstream/evaluate.py` rate denominator |
| `PX` | 117,964,800 | `600 × 384 × 512` |
| `d_seg` (cx1) | 0.004311794704861111 | `ddm_pz1_dseg_n600_cx1_20260803.json` → `d_seg_base_mean` |
| seg TERM | 0.4311795 | `100 · d_seg` |
| **flips (cx1)** | **508,640** | `d_seg · PX` |
| `W` | **1.2731082153320312** B/flip | `4·DEN/PX` |
| gap to PR130 floor | 0.6543562 | charter |

**Defect found in `sx1`'s headline (favourable direction).** `sx1` reports the description floor as
**0.5349 B/flip = 42 % of `W`**. That does not reproduce from its own artifact:
`253,341 B ÷ 508,640 flips = 0.4981 B/flip = 39.1 % of W`. The quoted figure implies a denominator
of 473,623 flips, which is neither `cx1`'s 508,640 nor `pc2`/`tb1`'s 458,738. The corrected number
is **7.4 % cheaper** than claimed, so realization headroom is **6.20 bits/flip, not 5.90** — the
error was pessimistic and its correction *strengthens* `sx1`'s conclusion.

Everything else in `sx1` §5 reproduces exactly from `ddm_sx1_label_field_mdl_n600.json`:
`1,781,833.41 / 1,986,727.72 = 89.686 %` of bits on the separatrix; `0.6984` vs `0.001775`
bits/px; ratio `393.5×`. **`sx1`'s most-defended claim survives independent re-derivation.**

---

## §2 G2 — the gate, run (artifacts `ddm_sx2_g2_contour_hitrate_{all,odd}.json`)

`experiments/ddm_sx2_g2_contour_hitrate.py`. Substrate: **GT `frame_1`** through the exact
`SegNet.preprocess_input` path (`F.interpolate(size=(384,512), mode='bilinear')`, `align_corners`
default False — *not* a cv2 resize; the camera→scorer stride is non-integer, 1164/512 = 2.2734 and
874/384 = 2.2760, so the two operators do not agree). Every operator gets the **same budget**
(equal-count at the GT boundary density 0.021628) so precision and recall are comparable.

**This is the CEILING, not the decoded number.** The premise of S2 is that `inflate.py` runs the
extractor on *its own decoded base*, which is strictly degraded. A ceiling that fails is a decisive
kill; a ceiling that passes does not establish the decoded value. No `cx1` decode exists in this
tree — the newest full `inflated/0.raw` is 2026-07-15, old lineage. §7 queues it.

### n600, full population

| extractor | `rho` (±1) | precision (±1) | c16 cell recall | c16 residual map B | free % |
|---|---:|---:|---:|---:|---:|
| canny | **0.6210** | 0.3839 | 0.7978 | 26,446 | **28.1** |
| sobel_luma | 0.4726 | 0.4903 | 0.6690 | 27,799 | 24.5 |
| morph_gradient | 0.4250 | 0.5085 | 0.6071 | 28,206 | 23.4 |
| laplacian | 0.4266 | 0.3460 | 0.6939 | 34,296 | 6.8 |
| sobel_rgbmax | 0.3865 | 0.3743 | 0.6190 | 33,283 | 9.6 |
| **random_control** | 0.1791 | 0.0438 | **0.9962** | **36,798** | **0.0** |
| **static_prior_control** | 0.5948 | **0.6842** | 0.7545 | **18,540** | **49.6** |

*(This row is a per-**pixel** static prior aggregated up to cells, because that is what the shared
harness computes. The carrier itself is a per-**cell** static map, which is both cheaper and
better — 49 B for 57.2 %. The two are measured separately and must not be conflated; the
per-cell number appears in the round-2 table and in "priced, honestly" below.)*

Held-out (`--subset odd`, prior built on even frames only): static prior `rho` 0.5962,
precision 0.6849, residual 9,293 B of an 18,401 B marginal = **49.5 % free**. It generalizes.

**Two sanity anchors that make the instrument trustworthy.** (1) The `random_control` residual is
**36,798 B — exactly `sx1`'s independently derived iid upper bound**, to the byte. My entropy
accounting and `sx1`'s agree. (2) `random_control` reaches **0.9962 cell recall while carrying
provably zero information**: at a 2.16 % pixel budget, `P(a 16×16 cell contains no predicted px)
= (1−0.0216)^256 = 0.4 %`, so almost every cell is "predicted boundary". **Cell recall is a vacuous
metric at this budget** — the residual-entropy column is the one that reports anything.

### Round-1 self-review — is the verdict an artifact of the budget I chose?

Equal-count was my choice. A bigger budget trivially raises `rho`. Swept (n=120 stride-5 sample —
a *stride*, not a prefix, because a prefix of this population is a contiguous scene block):

| budget × | density | `rho` | precision | c16 free % |
|---:|---:|---:|---:|---:|
| 0.5 | 0.01092 | 0.4565 | 0.5280 | 24.7 |
| **1.0** | 0.02184 | 0.6194 | 0.3809 | **27.0** |
| 2.0 | 0.04368 | 0.6450 | 0.2044 | 21.7 |
| 4.0 | 0.08735 | 0.6460 | 0.1026 | 17.0 |
| 8.0 | 0.17470 | 0.6862 | 0.0667 | 15.4 |
| 16.0 | 0.34940 | **0.7161** | 0.0439 | **6.3** |

The free-fraction **peaks at exactly the budget I chose** — my headline is the extractor's best
case. And `rho` and free-fraction **anti-correlate** above it. This is the cleanest possible
refutation of the gate's threshold logic: **you can buy `rho > 0.7` and simultaneously destroy the
thing `rho` was supposed to certify.**

### Round-2 self-review — is my negative an artifact of a permissive aggregator?

"Cell is predicted iff ANY predicted pixel lands in it" is the *most permissive* rule and is why
random scores 0.9962. A count-ranked aggregator could be strictly better and would weaken my
negative. Tested, held out on odd frames:

| cell predictor | recall | precision | residual B | free % |
|---|---:|---:|---:|---:|
| canny, ANY-pixel aggregator *(my headline)* | 0.7967 | 0.4853 | 13,235 | 28.1 |
| canny, COUNT-ranked top-N | 0.5515 | 0.5531 | 14,877 | 19.1 |
| **STATIC per-cell prior (49 B)** | **0.8358** | **0.8383** | **7,867** | **57.2** |
| union(static prior, canny top-N) | 0.9199 | 0.6193 | 9,369 | 49.1 |
| canny + 2× static prior | 0.5672 | 0.5689 | 14,600 | 20.7 |
| canny + 8× static prior | 0.6109 | 0.6127 | 13,771 | 25.2 |
| canny + 32× static prior | 0.7788 | 0.7811 | 9,674 | 47.4 |

The count-ranked aggregator is **worse** — the ANY rule was already generous to the extractor, so
the negative survives its own strongest attack. And **every blend is below the static prior alone.**
Canny's marginal contribution on top of 49 B of stored table is **negative**.

### The static prior priced, honestly

It is **video-derived ⇒ COUNTED**, not free. Priced by construction, not asserted:
`c=16` grid is 24×32 = 768 cells ⇒ 96 B raw, **49 B zlib-9**, 100 B lzma. `c=32`: 24 B raw,
**23 B zlib**.

| | cost | buys | exchange rate |
|---|---:|---:|---:|
| static per-cell prior, c=16 | **49 B** | 57.2 % of the 36,798 B map term = **21,048 B** | **430×** |
| best free generic extractor | 0 B | 28.1 % = 10,340 B | ∞ but dominated |

**Scope, welded on:** this works because the contest scores **one clip** from a **rigidly mounted
camera** — the horizon and the ego-hood separatrix are in almost the same cells every frame. It is
a contest-native exploit, legal (counted, in `archive.zip`, no scorer weights), and it would not
transfer to a different clip. That is not a defect; the contest is one clip.

### G2 verdict

| claim | verdict |
|---|---|
| `rho > 0.5` for a generic extractor | **TRUE** — canny 0.6210, n600 |
| ⇒ "S1's occupancy map effectively free" | **REFUTED** — 28.1 % at best budget; `rho` and free-fraction anti-correlate |
| S2's cost model `253,341·(1−rho)` | **REFUTED** — measured 9.44 % vs predicted 62.10 %, **6.58× over** |
| S4 (the composite) | **SURVIVES, with a different mechanism** — the map term is best bought by a 49 B counted static table, not by a free generic algorithm |

**Never prematurely kill.** S2 is refuted at the **FORMULATION** level (this cost model, this
extractor family, this substrate). The **FAMILY** — "let the decoder derive what it can and count
only the residual" — is *vindicated*, not killed: it is exactly what the 49 B static map does. The
correction is about *which* derivation is cheap, not about whether derivation-plus-residual works.

---

## §3 The conditional-MDL measurement (artifact `ddm_sx2_conditional_mdl.json`)

`experiments/ddm_sx2_conditional_mdl.py`. Order-4 causal context `(W, N, NW, NE)` over
`{5 classes + outside}`, two-part MDL with a Krichevsky–Trofimov model charge on used contexts.

**Independent re-derivation cross-check:** my unconditional code length is **238,945 B** against
`sx1`'s **253,341 B** — **5.7 % apart** from a different context definition and a different model
charge. Two instruments, one magnitude. `sx1`'s floor is corroborated.

Admitting the free Canny indicator as an extra context symbol drops it to **216,395 B**. That
**22,550 B = 9.44 %** is what a free generic separatrix predictor is worth as *description* side
information. The naive formula claims 62.10 %.

**Why the formula fails, structurally:** 89.7 % of the field's bits sit on the separatrix, but
those bits are not "where is the boundary" — they are *which of five labels is on each side, and
exactly which pixel the crack falls on*. A ±1 px edge indicator answers neither question. Recall on
a location is not a fraction of a code length, and nothing licenses treating it as one.

---

## §4 The displacement hole, measured (artifact `ddm_sx2_displacement_magnitude.json`)

`experiments/ddm_sx2_displacement_magnitude.py`, `lstars` only — no decode, no vehicle, no scorer.

A carrier cannot be designed against an adjective. If the residual were a rigid per-object
translation of magnitude `d`, it would leave two independent signatures, each invertible for `d`:

- **route A** — the changed band's fraction lying >3 px from the separatrix, matched to `pc2`'s
  observed 17.9–19.7 % on the Movable edges;
- **route B** — the changed-pixel count, matched to `pc2`'s displacement-class flip count.

Measured on the GT label field, 8 directions, n600:

```
Movable: 3.66 components/frame · mean area 665 px · median 105 px · perimeter 237 px/frame
         median inradius 4.00 px
DROPOUT signature (whole object missing): 2,434 px/frame, 67.36 % of them >3 px off boundary

 d   changed px/frame   frac >3px   frac >5px
 1        207.2           0.0233      0.0216
 2        408.6           0.0235      0.0217
 3        601.0           0.0371      0.0222
 4        781.1           0.0849      0.0227
 5        947.4           0.1858      0.0410
10       1616.2           0.4662      0.2949
```

> **route A: `d` = 4.93 – 5.14 px.  route B: `d` = 0.860 px.  DISAGREEMENT 5.8×.**

**Single-magnitude rigid translation is REFUTED** — it can produce the observed spread or the
observed count, not both. Correcting route B for the deep component (×0.76) drives it to
**`d ≈ 0.654 px`**, i.e. *more* disagreement. On the verdict ladder this is **FORMULATION**-level:
the *family* "positional error" survives, as a mixture.

### The mixture, and its honest status

Two components, endpoints both measured here: shallow = the `d=1` translation band
(`frac>3px = 0.0233`), deep = the full-dropout profile (`0.6736`). Solving on the Movable edges:
**deep 23.9–26.7 %, shallow at `d ≈ 0.65 px` — SUB-PIXEL.**

**Two parameters fitted to two observations is exactly determined — a decomposition, not a
validated model.** Its only corroboration is that the same model applied to the three non-Movable
edges returns small, sensible deep fractions (6.3 % / 2.3 % / 0.9 %) where we expect them.
`pc2` reports no `>5 px` column, so the over-determining test is unavailable. Stated, not hidden.

### The model-free number, which is the one to quote

The per-edge ">3 px off" shares are already a direct measurement. Weighted by flip share and
renormalised over the 99.70 % of flip mass these five edges cover:

> **6.11 % of seg flips lie >3 px from the GT separatrix. 93.89 % lie within 3 px.**

No mixture, no reference class, no direction assumption. The model-based deep fraction under the
three direction-split conventions:

| edge | flip % | `off3` | deep, mean-ref | deep, max-ref |
|---|---:|---:|---:|---:|
| Road↔Lane | 49.23 | 0.026 | 0.063 | 0.032 |
| Road↔Undriv | 16.26 | 0.021 | 0.023 | 0.021 |
| Undriv↔Movable | 11.85 | 0.197 | 0.239 | 0.202 |
| Road↔Movable | 11.47 | 0.179 | 0.240 | 0.218 |
| Road↔MyCar | 10.89 | 0.008 | 0.009 | 0.008 |
| **weighted total** | | | **9.17 %** | **6.89 %** |

*(A third convention, `min`-ref, returns 56 % and is **self-refuting**: it is driven entirely by
assigning Road↔Lane the Lane-side reference of 0.0007. Lane components are ~1 px thin, so a flip on
a Lane pixel can never be >3 px from a boundary — the `off3` mass on that edge is necessarily
Road-side. Named and discarded on physical grounds, not on convenience.)*

**Model-free 6.11 %, modelled 6.89–9.17 %. All three cluster at 6–9 %.**

### What this does to `sx1`'s split, and to my own charter

| | `sx1` §2.5 | measured here |
|---|---:|---:|
| boundary-shaped | 76.4 % | **90.8 – 93.9 %** |
| object-level / not separatrix-shaped | 23.3 % | **6.1 – 9.2 %** |

`sx1` inferred the split from `flips/len` — the Movable edges run 1.9–2.1× denser per unit of
interface. **That density fact is true and survives.** What does not follow from it is that the
extra flips are off the boundary; 76–78 % of them are on it. **Denser ≠ differently shaped.**

**My charter told me this hole was "the more valuable half." The measurement says the opposite,
and I report that rather than the framing.** In gap terms:

| class | flips | ΔS if fully killed | share of the 0.6543562 gap | rate-equivalent |
|---|---:|---:|---:|---:|
| boundary-shaped | 461,980 | **0.39163** | **59.85 %** | 588,151 B |
| object-level (mean-ref) | 46,660 | 0.03955 | 6.04 % | 59,403 B |
| object-level (model-free) | 31,101 | 0.02636 | 4.03 % | 39,595 B |

The charter priced the hole at 23.3 % × 61.4 % = 14.3 % of the gap. It is **4.0–6.0 %** — **2.4–3.5×
smaller.** And it is the half whose realization is *harder*: a boundary-shaped flip needs a
sub-pixel edge nudge, which an amplitude carrier can plausibly deliver; an object-level flip needs
the decoder to **synthesize a car-shaped blob that SegNet's argmax reads as Movable** — squarely the
§6 realization wall, at its most acute. **Priority ordering: the boundary-shaped 90–94 % is both
the larger prize and the easier realization. The hole is real, smaller than named, and last.**

### The carrier the measurement actually points at

Not a per-object integer offset — that is what the 5.8× disagreement refutes. The shallow component
is **sub-pixel**, and a sub-pixel displacement is a **phase** shift. This lands on the same missing
DOF that was already recorded independently: *"phase-faithfulness = per-pair POSITIONAL DOF;
`tokens_delta` gives AMPLITUDE only — ONE deficit."* Two instruments, arrived at from opposite
directions, name the same slot.

Priced as a description (**realization unmeasured — this is a budget, not a gain**):

| carrier | count | description cost | flips addressed | exchange rate vs `W` |
|---|---:|---:|---:|---:|
| sub-pixel phase, 2 bits/axis, per Movable component | 2,197 comps | ~1,098 B | shallow Movable ≈ 90,200 | ~105× |
| object existence+position, 3 B/object, ~1–2 objects/frame | ~900 | ~2,700 B | deep ≈ 31,101–46,660 | ~15–22× |

Both ratios are large **because description is cheap and realization is the wall** — which is the
same conclusion `sx1` reached, `ddm_br1` reached independently from the rate side (264 lossless
re-expressions of the token lattice; `IDENT` within 0.83 % of the incumbent ⇒ the lattice *shape*,
not its coding, is where rate lives), and this arm now reaches from the description side. **Three
arms, three axes, one wall.** Every candidate above must be judged against the
**6.20 bits/flip** realization budget (§1, corrected), not against its description cost.

---

## §5 ASSUMPTION LEDGER

| # | assumption | class | status |
|---|---|---|---|
| B1 | `SegNet.preprocess_input` = `x[:,-1]` then bilinear to (384,512) | VERIFIED_VIA_SOURCE_INSPECTION | `upstream/modules.py` quoted; replicated with `F.interpolate`, not cv2 |
| B2 | `lstars` == live `argmax SegNet(GT)` | VERIFIED_VIA_EMPIRICAL_ANCHOR | inherited from `sx1` A3 (cross-cache class-area agreement); **not re-verified here** |
| B3 | class order = canonical comma10k, not luma-sort | VERIFIED_VIA_EMPIRICAL_ANCHOR | re-derived here: class 1 interior-fraction 0.0007 (thin ⇒ Lane), class 4 interior 0.9589 + bottom rows (⇒ MyCar) |
| B4 | GT-RGB extractor performance ≥ decoded-RGB performance | INFERRED_FROM_DOMAIN_LITERATURE | a lossy decode cannot add true edges; supports the **negative** direction only |
| B5 | `pc2` per-edge `off3` + flip shares transfer `tb1 ep399` → `cx1` | ASSUMED_AWAITING_VERIFICATION | `sx1` A4, +10.87 % flips, still untested; **§4's shares inherit it** |
| B6 | order-4 context entropy ≈ achievable code length | INFERRED_FROM_DOMAIN_LITERATURE | bracketed by `sx1`'s MEASURED lzma-9e at 1.69× |
| B7 | deep/shallow mixture is a 2-param fit to 2 observations | **exactly determined, NOT tested** | stated in §4; no `>5 px` column exists in `pc2` to over-determine it |
| B8 | static prior is legal counted content | VERIFIED_VIA_SOURCE_INSPECTION | this-clip-derived ⇒ COUNTED; 49 B priced by construction, ships in `archive.zip`, no scorer weights |

**PROVISIONAL:** everything in §4 that quotes a *share* (rests on B5). The G2 verdict (§2) and the
conditional-MDL measurement (§3) rest only on B1/B2/B4/B6 and are the solid claims here.

### Review rounds — counter stands at 0 clean passes

| round | findings | reset? |
|---|---|---|
| **R1** | (a) G2's `rho` threshold could be an artifact of the budget I chose → swept; verdict *strengthened*, free-fraction peaks at my budget and anti-correlates above it; (b) `sx1`'s 0.5349 B/flip does not reproduce from its own artifact (0.4981 correct) | **YES** |
| **R2** | (c) my ANY-pixel cell aggregator is the most permissive rule and could be understating the extractor → count-ranked tested, it is *worse*, negative survives; (d) the static prior was priced per-pixel when the carrier is per-cell → repriced at 49 B | **YES** |
| **R3** | (e) the 9.17 % deep fraction rests on a direction-split convention → model-free 6.11 % derived and promoted to the headline, `min`-ref degenerate case named and physically dismissed; (f) my n=10/n=24 smokes were **prefixes** and disagreed sharply with n600 (static-prior `rho` 0.856 → 0.595) → every number in this memo is full-population or stride-sampled | **YES** |

**Three rounds, three finding-sets. By the project's own rule this is round-3 output, not sealed
work.** Most likely still wrong, in order: (1) §4's shares, which inherit `pc2`'s untested vehicle
transfer (B5); (2) the deep/shallow mixture, which is exactly determined and not validated (B7);
(3) the GT→decoded gap in §2, entirely unmeasured (B4).

---

## §6 WHAT I DID NOT DO

- **No scorer pass.** `ddm_pu2` holds the slot. G1 remains owed and unrun.
- **No decoded-frame measurement.** No `cx1` decode exists in this tree (newest full
  `inflated/0.raw` is 2026-07-15, banned lineage). §2 is the GT ceiling; the decoded `rho` and the
  decoded map free-fraction are both unmeasured. Queued as G4.
- **Did NOT use `src/tac/boundary_math/contour_codec.py`** and nothing here rests on it — it
  serialises every uint8 label in raster order and LZMA-compresses the dense array (task #913), so
  it is a dense-raster baseline, not a 1-D wire representation.
- **Did not re-verify B2.** Inherited from `sx1`'s cross-cache anchor.
- **Did not diff against `ddm_de1`** (codex, independent derivation) — `sx1`'s NEXT-IF-RESUMED item 4
  is still open.
- **Did not measure realization for any carrier in §4.** Both rows are descriptions with a budget
  attached, not gains. That is the wall, and it needs a decode.
- **The R1 budget sweep and R2 aggregator race were computed inline**, importing
  `experiments/ddm_sx2_g2_contour_hitrate.py` and reusing its `ex_canny` / `_cells_any` /
  `label_boundary_4conn` / `_resize_to_scorer_grid` on the same `lstars` and `gt_f1` inputs (R1 on
  a stride-5 n=120 sample, R2 on the full odd n=300 hold-out). They are reproducible from the
  committed module but are not themselves committed as JSON artifacts.

---

## §7 QUEUED GATES — exact commands owed

**G1 — `sx1`'s S1 falsifier, one n600 scorer pass. Still owed, unchanged.** Note that this memo
*improves* its predicted ΔS: with the map term bought by a 49 B static table instead of transmitted
at 36,798 B, S1's saving at `c=16, q=0.25` moves from 177,730 B to ~198,700 B.
```
.venv/bin/python -m tac.contest_score --archive <cand>.zip --n-pairs 600 --device cpu
```
PASS iff `Δflips < 156,000` (rescaled from `sx1`'s 139,584 at the larger saving) **and**
`Δd_pose ≈ 0`.

**G4 — G2 on DECODED frames (scorer-free, needs one decode).** Everything in §2 is the GT ceiling.
Decode a live `cx1` archive and rerun:
```
bash <submission_dir>/inflate.sh <archive_dir> <out_dir> upstream/public_test_video_names.txt
.venv/bin/python experiments/ddm_sx2_g2_contour_hitrate.py --n-pairs 600 --subset odd \
    --frames <decoded frames npz> --out .omx/research/ddm_sx2_g2_decoded.json
```
The static prior's 57.2 % is computed from `lstars` and is **decode-independent**; only the generic
extractors' column moves. Expect them to fall.

**G5 — over-determine the deep/shallow mixture (B7).** Re-derive `pc2`'s per-edge off-boundary
table with a `>5 px` column as well as `>3 px`, on `cx1`. The mixture fitted to `off3` then
*predicts* `off5`; agreement validates it, disagreement kills it. This is the test that does not
currently exist, and it is scorer-free given the flip atlas.

**G6 — `sx1`'s G3, still owed.** Re-derive the per-edge flip table on `cx1` rather than `tb1 ep399`.
Every share in §4 is PROVISIONAL until this lands (B5).

---

## NEXT-IF-RESUMED

1. **G5 before any displacement carrier is built.** The mixture is the whole basis for "sub-pixel
   phase, not integer offset," and it is exactly determined, not tested. G5 is scorer-free.
2. **Aim at the boundary-shaped 90–94 %, not at the hole.** It is 59.85 % of the gap against the
   hole's 4.0–6.0 %, and its realization is the easier one. The charter's priority was inverted;
   the measurement, not the framing, should set the order.
3. **Adopt the 49 B static per-cell map wherever a boundary-cell map is needed** — 430× exchange
   rate, held out, decode-independent. Retire "free generic contour extractor" from the S2/S4
   design: it is dominated, and blending it in makes things worse.
4. **Stop quoting `rho`.** It is budget-inflatable and anti-correlates with the thing it was meant
   to certify above the natural budget. The residual conditional entropy is the metric that reports
   something; cell recall reaches 0.9962 on pure noise.
5. **G4** whenever a `cx1` decode exists — it is the only unmeasured leg of §2.
6. Diff against `ddm_de1` (still open from `sx1`).
