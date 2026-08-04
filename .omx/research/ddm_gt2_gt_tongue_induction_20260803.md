---
schema: ddm_gt2_gt_tongue_induction.v1
date_utc: 2026-08-03
arm: ddm_gt2 (re-run the grammar-induction line against the real n600 GT argmax corpus)
lane_id: "lane_ddm_gt2_20260803"
research_only: true
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
pointer_moved: false   # exact contest pointer 0.1910828242 [contest-CPU] UNMOVED. This arm fired no gate.
scorer_forwards_run: 0
axis: "[macOS-CPU advisory] NON-PROMOTABLE. Zero scorer forwards: every number is read off the
  argmax corpus ddm_pu2 kept, or produced by a REAL coder (lzma/brotli/zlib) on that corpus.
  No training, no dispatch, no archive rebuilt, no pointer mutation."
baseline_named: "live best S = 0.7910689 at 353,805 B [macOS-CPU advisory]; seg leg 0.4311790;
  gap to PR130 = 0.6189279. Every S-delta below is stated against THAT baseline."
sample_scope: "n = 600 pairs, selection_mode = FULL_POPULATION (no subset). The only subsets in this
  arm are a 3-frame and a 6-frame contiguous-prefix SMOKE, labelled at both use sites and cited as
  NOT evidence (prefix bias, m88/m96)."
premise_corrected_at_source: "This arm was chartered on the premise that the prior induction line
  'was built against sampled or reconstructed corpora and has never been re-run against this'.
  THAT PREMISE IS FALSE for #620 g1 and #651 dv2 and is retracted. Verified at source, not from
  memory: gt_argmax_n600.npy is BIT-IDENTICAL to g1's corpus gt_n600.npz::lstars -- 0 of
  117,964,800 px differ, sha256 f2c8be94774780bd on both. It IS true for #650 dv1 (a v12
  reconstruction as predictor, stride-8x16 sampled) and #664 pf1 (induced no label field at all)."
verdict_scope:
  - claim: "re-running the induction on the real GT corpus will move the prior two-part code lengths"
    verdict: REFUTED
    scope: "FAMILY - all label-field vocabulary induction, because the corpus is bit-identical"
    why_this_high: "identity is proven at source by SHA over the full array, not inferred"
    consequence: "the charter's directive (a) explicitly anticipated this: 'If it did not, that is
      a result.' It is this arm's first result, and it is why the arm pivoted to F7/phase."
  - claim: "a REGION production structurally cannot serve the Lane class"
    verdict: REFUTED
    scope: "this arm's OWN prior draft of this claim, at FORMULATION level"
    why_not_higher: "MEASURED here: the contour production beats the raster production for EVERY
      class (Lane 0.691x, Road 0.672x, Movable 0.613x), so area/perimeter does NOT make region
      productions impossible. What area/perimeter DOES predict is confirmed: cost per DELIVERED
      pixel, Lane 0.1896 B/px vs Road 0.00725 B/px = 26.2x. A cost ratio is not an impossibility."
  - claim: "sx1's 253,341 B is the lossless cost of the label field L*"
    verdict: REFUTED_AS_A_CODER_COST
    scope: "FORMULATION - the number is a context-model ENTROPY ESTIMATE (H1+model)/8, not a coder
      output. The real coder costs 410,584 B, 1.6207x more."
    why_not_higher: "sx1's estimate is correct AS an estimate, and its own 60-frame lzma row
      extrapolates to within 4.1% of the measured n600 coder cost. Only the reading of the
      estimate as an achievable cost is refuted."
  - claim: "an explicit per-pair PHASE production (whole-frame integer translation) pays"
    verdict: REFUTED
    scope: "FORMULATION - whole-frame INTEGER translation, radius +-10 px, on the argmax lattice"
    why_not_higher: "does NOT refute the phase axis. Sub-pixel, per-region, projective, and
      scale/expansion phase productions are untested and are the natural next form: 586 of 599
      pairs prefer zero shift precisely because forward ego-motion is an expansion field, not a
      translation. argmax also quantises position to the lattice, so sub-pixel phase is invisible
      to this corpus BY CONSTRUCTION."
  - claim: "explicit temporal carry-forward beats implicit whole-corpus coding"
    verdict: REFUTED
    scope: "FORMULATION - carry-forward residual coding with explicit addressing, this corpus"
    why_not_higher: "the mechanism (address dominance) is measured, not assumed, and it points at
      a specific cure (implicit-address / generative productions) rather than killing temporal
      productions as a family."
verdict_scope_ladder: "INSTANCE < FORMULATION < FAMILY < PARADIGM."
consumes:
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/{gt,cx1}_argmax_n600.npy
  - experiments/results/mlx_fleet_gt_cache/gt_n600.npz   (g1/dv2's corpus, for the identity proof)
  - .omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md   (corpus + fail-closed control row)
  - .omx/research/ddm_dd1_lane_component_census_n600.json     (area/perimeter - CONSUMED, not re-measured)
  - .omx/research/ddm_sx1_label_field_mdl_n600.json           (the H1 estimate priced against)
  - .omx/research/direct_description_g1_grammar_induction_20260722*        (#620)
  - .omx/research/ddm_dv2_sdwl1_*_20260723*                               (#651)
produces:
  - experiments/ddm_gt2_gt_tongue_induction.py
  - /Volumes/VertigoDataTier/pact/ddm_gt2_20260803/gt2_{control,verbs,mdl,split,lane,phase}.json
consumers: [MAIN, ddm_cg1 (task #809 force ledger - the VERB LEXICON is its column headers), ddm_dd1]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_gt2 — the tongue we are learning from GT

## §0 ANSWER FIRST

**The charter's premise was false, and correcting it produced a better result than the charter asked
for.** The corpus did not just become free — `g1` (#620) and `dv2` (#651) already induced from a
**bit-identical** field (`sha f2c8be94774780bd`, 0 of 117,964,800 px differ, verified at source).
So directive (a)'s honest answer is: **nothing moved, because nothing changed.**

What this arm adds instead, on axes `g1`/`dv2` did not induce:

**1. THE ADDRESS-DOMINANCE LAW — measured three times independently.** Every explicit residual
production spends ~78% of its bytes saying **WHERE**, not **WHAT**:

| production | address B | payload B | **address share** |
|---|---:|---:|---:|
| static-lexicon + per-pair residual | 331,236 | 102,496 | **76.37%** |
| temporal carry-forward (P1) | 442,568 | 121,232 | **78.50%** |
| carry-forward + phase (P2) | 441,768 | 120,784 | **78.53%** |

This is *why* correction streams cannot pay — and the mechanism is new. `pu2` priced Road↔Lane at
299,369 B break-even and concluded a correction stream cannot buy it; this says the money goes to
**addressing**, not to content. Payload is cheap. Location is expensive.

**2. THE CURE IS GENERATIVE, AND IT IS MEASURED.** A production whose address is *implicit* escapes
the tax entirely. The lossless contour production for Lane costs **130,960 B** — it beats raster
(0.691×), beats `g1`'s best lossless Lane row-runs (180,701 B) by **27.5%**, and beats buying Lane's
own 236,816 flips at the exchange rate (301,477 B at `W`) by **2.30×**. A polyline says *where* by
construction; a residual must name it.

**3. LANE'S FAILURE IS NOT RATE, AND NOT VOCABULARY.** `g1` already induced the vocabulary; this arm
prices it **2.30× under break-even**. Yet the shipped tongue still annihilates **58.23% of all Lane
words** (9,655 of 16,581 components retain <5% of their own pixels) against Road's 5.45% and MyCar's
**0.00%**. Lane loses **26.903% of its entire population**; Road loses **0.559%** — a **48.1×**
reliability gap across the separatrix they share. **So Lane is a REALIZATION failure** (`m11`'s
standing crux), not a description-length one, and the verb is `ANNIHILATE`: words are *dropped*, not
mis-sized.

**4. THE SCORER CANNOT HEAR IT.** `upstream/modules.py` computes SegNet distortion as a uniform
per-pixel mean. Lane is 0.5855% of pixels, so destroying a quarter of the class costs **0.199 S** —
comparable to Road's boundary wobble, a far less destructive act. **The objective has no term that
can express which word class it lost.**

**5. PHASE, AS A WHOLE-FRAME INTEGER TRANSLATION, DOES NOT PAY** — 50 B on 563,800 (**0.009%**), and
**586 of 599 pairs prefer zero shift**. *(NEGATIVE, scope: FORMULATION — whole-frame integer
translation only; forward ego-motion is an expansion field, not a translation, and argmax quantises
sub-pixel phase away by construction. The phase FAMILY is untested and open.)*

---

## §1 WHAT IS NEW HERE vs `g1`/`dv2` — and what is only a re-anchor

Per the anti-re-anchoring discipline (`m38`), the prior-law prediction line comes **first**:

| prior law | predicted | measured outcome | new? |
|---|---|---|---|
| `sx1`: 89.69% of bits on 2.16% separatrix px, 393× | boundary dominates cost | **reproduced exactly** from sx1's own JSON | **re-anchor** |
| `sx1`: L\* = 253,341 B lossless | a coder should land near it | **refuted as a coder cost: 410,584 B, 1.6207×** | **NEW** |
| `hs1`: static lexicon amortises (379×) | whole-corpus should beat per-frame | **confirmed, 26.5%**, independent direction | corroboration |
| `dd1`: Lane minor axis 2.51 px, a/p 1.407 | region-paint should fail on Lane | **partly refuted** (§3) — a/p predicts cost/px (26.2×), not impossibility | **NEW + self-correction** |
| `rz1`: boundary band recovers 93.53% | address should be ~free | **does NOT transfer to a static address object: 28.0–62.4%** | **NEW negative** |
| `pu2`: Lane −26.50% / Road +27.22% | net erosion | **reproduced exactly; denominator identified** (% of total flips) | clarification |
| `m91`: Road is the hub at 87.8% | Road dominates the edge table | **confirmed at 87.48%** on a different corpus | corroboration |

**Honest accounting: 3 of 7 rows are re-anchors or corroboration, not discoveries.**

### 1.1 The corpus identity proof (directive (a))

```
gt_n600.npz::lstars  (600,384,512) int64   sha256 f2c8be94774780bd…   <- g1 (#620), dv2 (#651)
gt_argmax_n600.npy   (600,384,512) uint8   sha256 f2c8be94774780bd…   <- ddm_pu2, this arm
IDENTICAL: True      differing px: 0 of 117,964,800
```

**Independent cross-validation of both arms:** my lzma bitplane coder and `g1`'s row-run coder, on
this bit-identical corpus, agree within 5% — Lane 189,460 vs 180,701 B (4.8%), Movable 62,288 vs
59,481 B (4.7%). Two independent codecs, one corpus, same answer.

### 1.2 The denominator that was ambiguous, now pinned

`pu2` quotes "Lane −26.50%, Road +27.22%" without its denominator. It is **net change as % of TOTAL
FLIPS**; I reproduce it exactly (−26.499 / +27.222). Three denominators, all true, very different:

| quantity | Road | Lane | Undrivable | Movable | MyCar |
|---|---:|---:|---:|---:|---:|
| net px | +138,461 | −134,786 | +2,127 | −37,733 | +31,931 |
| net as % of **total flips** (`pu2`'s) | +27.222 | **−26.499** | +0.418 | −7.418 | +6.278 |
| net as % of **own GT population** | +0.505 | **−19.516** | +0.004 | −2.584 | +0.106 |
| **% of own population misclassified** | 0.559 | **26.903** | 0.128 | 5.398 | **0.054** |

**The third row was missing and is the sharpest.** A percentage without its denominator is a ΔS
without its baseline, one level down.

---

## §2 THE EDGE TABLE AND THE VERB DECOMPOSITION

n600, **508,640 flips**, `d_seg` 0.004311794704861, control **rel 1.09e-06** vs `pu2` (fail-closed).
Decomposed per **EDGE** per `m91` — charging by GT class splits one separatrix across two rows.

| edge | flips | % flips | **DISPLACE** | **TRANSFER** | disp% | dominant | S | % of gap |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| **Road↔Lane** | **235,148** | **46.231** | 101,070 | **134,078** | 42.98 | **Lane→Road** | 0.19934 | **32.207** |
| Road↔Undrivable | 89,545 | 17.605 | 47,498 | 42,047 | 53.04 | Undrivable→Road | 0.07591 | 12.264 |
| Road↔MyCar | 63,027 | 12.391 | 26,020 | 37,007 | 41.28 | Road→MyCar | 0.05343 | 8.632 |
| Undrivable↔Movable | 61,892 | 12.168 | 31,370 | 30,522 | 50.69 | Movable→Undrivable | 0.05247 | 8.477 |
| Road↔Movable | 57,225 | 11.251 | 34,922 | 22,303 | 61.03 | Movable→Road | 0.04851 | 7.838 |
| Lane↔MyCar | 903 | 0.178 | 120 | 783 | 13.29 | Lane→MyCar | 0.00077 | 0.124 |
| Lane↔Movable | 681 | 0.134 | 30 | 651 | 4.41 | Lane→Movable | 0.00058 | 0.093 |
| Movable↔MyCar | 135 | 0.027 | 16 | 119 | 11.85 | Movable→MyCar | 0.00011 | 0.018 |
| Lane↔Undrivable | 84 | 0.017 | 8 | 76 | 9.52 | Undrivable→Lane | 0.00007 | 0.012 |

**Road participates in 87.48% of flips** (444,945/508,640) — `m91`'s hub confirmed on an independent
corpus (87.8% of 458,738 on `pc2`'s different base; **the two flip counts are different populations
and must not be pooled**, but the hub fraction agrees to 0.3 pp).

**The gap is not one verb.** Road↔Lane is TRANSFER-dominated (57.0% — genuine area handover);
Road↔Movable is DISPLACE-dominated (61.0% — boundary wobble). **Every Lane minor edge is
TRANSFER-dominated (86.7 / 95.6 / 90.5%): Lane never wobbles, it only loses.**

### 2.1 Frame targeting pays on the small edges, not the big ones

Share of an edge's TRANSFER carried by its worst 10 frames (uniform would be 10/600 = 1.67%):

| edge | frames touched | top-10 share | concentration |
|---|---:|---:|---:|
| Road↔Lane | 600 | 3.64% | 2.18× |
| Road↔Undrivable | 600 | 5.76% | 3.45× |
| Undrivable↔Movable | 600 | 15.39% | 9.22× |
| Lane↔Undrivable | 20 | 84.21% | 50.4× |
| Movable↔MyCar | **10** | **100.00%** | 60.0× |

**Aiming a force at individual frames is a small-edge instrument, not a big-edge one.** Guarding
against the binary reading: this does not make frame-targeting worthless — it makes it *precisely
targeted*, and the four smallest edges are exactly where it is decisive.

### 2.2 Per-class verbs (n600)

| class | GT comps | **ANNIHILATE** | annih % | BIRTH | ERODE px | GOUGE px | FRAGMENT Δ | flip% own | depth≤1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Road | 1,266 | 69 | 5.45 | 16 | 141,180 | 11,747 | −533 | 0.559 | 4.57% |
| **Lane** | **16,581** | **9,655** | **58.23** | 591 | 135,683 | 2,926 | **−6,038** | **26.903** | **75.04%** |
| Undrivable | 650 | 39 | 6.00 | 3 | 68,608 | 5,892 | −38 | 0.128 | 0.57% |
| Movable | 2,207 | 361 | 16.36 | 5 | 53,940 | 16,718 | −492 | 5.398 | 9.73% |
| MyCar | 600 | **0** | **0.00** | 1 | 14,827 | 1,240 | +2 | 0.054 | 1.03% |

**58.23% of Lane words are annihilated outright.** MyCar — one rigid component per frame, 25.4% of
area — loses **zero**. The tongue speaks the big static words perfectly and cannot pronounce the
small moving one.

---

## §3 THE LANE PRODUCTION, AND A CORRECTION TO MY OWN CLAIM

*(directive (c))*

**I drafted a FAMILY-level claim that region productions structurally cannot serve Lane, resting on
`area/perimeter` = 1.407 vs Undrivable's 176.2. The measurement refutes it and I withdraw it.**
*(REFUTED, scope: FORMULATION — my own draft claim; see frontmatter.)* Lossless contour beats
lossless raster for **every** class:

| class | G_raster (lzma) | **G_contour lossless** | ratio | B per delivered px | vs Road |
|---|---:|---:|---:|---:|---:|
| **Lane** | 189,460 | **130,960** | **0.691** | **0.18962** | **26.2×** |
| Road | 295,576 | 198,580 | 0.672 | 0.00725 | 1.0× |
| Movable | 62,288 | 38,152 | 0.613 | 0.02613 | 3.6× |

What `area/perimeter` **does** predict is confirmed: **cost per delivered pixel** (Lane 26.2× Road;
the a/p ratio predicts 15.6× — same order, same direction). **A cost ratio is not an impossibility
claim**, and conflating them was the error.

Simplification tolerance (`approxPolyDP` ε), Lane, error stated in the unit the score charges:

| ε px | vertices | program B | error px | S if all became flips | B saved per error px |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 582,281 | **130,960** | **0** | 0.00000 | — |
| 0.5 | 190,317 | 129,864 | 24,048 | 0.02039 | 0.046 |
| 1.0 | 40,540 | 75,532 | 144,884 | 0.12282 | 0.383 |
| 2.0 | 27,085 | 54,172 | 292,904 | 0.24830 | 0.262 |

**Every simplification sells error below `W` = 1.273 B/flip** (best 0.383 B/error-px at ε=1.0), so
simplifying and buying the errors back is a **loss at every tolerance**. **Lossless is the operating
point** — convenient, because lossless is already 2.30× under break-even:

> Lane lossless contour description: **130,960 B**.
> Buying Lane's 236,816 flips at `W` = 1.273108215332031: **301,477 B**.
> **Ratio 0.434 — describing Lane exactly is 2.30× cheaper than correcting it.**

`W` is `ddm_wf2`'s exchange rate, cited not re-derived; it is a rate, not a cost.

### 3.1 The production that cannot make the erasure error

A production is **erasure-immune for class *c*** iff the existence of a *c*-instance is determined by
a term the other classes cannot outvote. The contour/stroke production qualifies: the instance exists
because the program says so — there is no area contest. A dropped stroke is a **missing symbol in the
program string**, detectable at encode time; a dropped region is silently absorbed by its neighbour
and is not. `mf1`'s component-native Movable result (2.47 components/frame at **81.4× better than
`W`**) is the same shape, on the class with the next-lowest `area/perimeter` (11.07).

---

## §4 THE STATIC/PER-PAIR SPLIT AND THE PHASE AXIS

*(directive (b) and the coordinator's F7 pointer)*

| coding of the GT label field | real coder | bytes | bits/px |
|---|---|---:|---:|
| whole corpus as one object | lzma9e | **410,584** | 0.02784 |
| whole corpus as one object | brotli11 | 424,728 | 0.02880 |
| whole corpus as one object | zlib9 | 581,266 | 0.03942 |
| each frame independently | lzma9e | 558,364 | 0.03787 |
| static lexicon (424 B) + explicit per-pair residual | lzma9e | 434,156 | — |
| temporal carry-forward P1 | lzma9e | 563,800 | — |
| carry-forward + integer phase P2 | lzma9e | 563,750 | — |

**Sharing context across the 600 pairs is worth 26.5%** (410,584 vs 558,364) — `hs1`'s static-lexicon
finding from an independent measurement, applied to the grammar's design for the first time. The
static lexicon itself is nearly free: **424 B** for the whole modal field.

**But every EXPLICIT split loses to the implicit coder** — 434,156 and 563,800 vs 410,584 — for the
one reason in §0: address dominance. And `rz1`'s free-address result does **not** transfer to a
*static* address object:

| band radius | band % of frame | residual covered |
|---:|---:|---:|
| 1 | 2.288 | 28.02% |
| 3 | 4.580 | 42.11% |
| 8 | 10.225 | 62.38% |

*(NEGATIVE, scope: FORMULATION — a band around the STATIC modal field's boundary. `rz1`'s 93.53% was
measured around a PER-FRAME decoder label field, a different object. The lesson is precise: **the
address must be re-derived from a per-frame already-transmitted object, not a static one** — the
scene moves, so the static boundary does not predict where a given frame differs.)*

**The phase axis (F7), whole-frame integer translation, radius ±10:**

| production | bytes | residual px/pair | nonzero shifts |
|---|---:|---:|---:|
| P1 carry | 563,800 | 2,449.0 | — |
| P2 carry + phase | **563,750** | 2,443.8 | **13 / 599 (2.2%)** |

Gain **50 B = 0.009%**; residual reduction 0.21%. Shift histogram: `(0,0)`×586, `(1,0)`×8,
`(−1,0)`×4, `(−2,0)`×1 — **every nonzero shift is purely vertical and ≤2 px**. *(NEGATIVE, scope:
FORMULATION — whole-frame INTEGER translation only. Forward ego-motion is an expansion field, not a
translation, and argmax quantises sub-pixel phase away by construction. The phase FAMILY is untested
and open; see §7 FIRE ORDER 2.)*

---

## §5 THE SYMBOL TABLE — every symbol typed before it is priced

*(the hard NO-FAKE gate. `#913` is the live instance of the failure: a dense raster LZMA baseline
wearing a boundary-edge codec's name. Typing precedes pricing so it cannot recur one layer up.)*

**Three-way test** (operator ruling 2026-08-03): fixed-**operator** properties are GENERIC and free;
**scorer weights** are video-invariant but 73 MB (economic, not banned); **this clip's content** is
COUNTED.

| symbol | type | why | counted? |
|---|---|---|---|
| 5-class semantics + canonical index order | GENERIC | frozen-SegNet property, clip-invariant | no |
| image lattice; `D`'s 2×2 disjoint sampling; 22.70% blind fraction | GENERIC | operator property (`m86`) | no |
| `fillPoly` / stroke rasteriser; dilation; distance transform | GENERIC | algorithms, not content | no |
| zigzag+varint integer code | GENERIC | a standard code | no |
| the shift OPERATOR that applies a phase symbol | GENERIC | an algorithm | no |
| band = `dilate(boundary(X))`, X already transmitted | **GENERIC** | deterministic function of held bytes — re-derived, not paid twice | no |
| static per-pixel modal lexicon (424 B) | **FITTED** | induced from THIS clip | **yes, exactly** |
| contour control points; widths | **FITTED** | this clip's geometry | **yes, exactly** |
| per-pair residual payload; per-pair phase symbol (2 B) | **FITTED** | this clip's frames | **yes, exactly** |
| pre-argmax margin / logit fields | **OUT OF SCOPE** | not measurable from argmax; `hg1` owns it | n/a |

### 5.1 The causality constraint that keeps "free address" honest

`rz1`'s free-address result is sound **only under a constraint that must be stated or it becomes a
fake**: the receiver has **no SegNet** (73 MB, not in the archive). It can re-derive a boundary only
from a label field **it has already been sent**, in an order it can actually execute. So:

> **ADDRESS is free iff it is a deterministic function of payload already transmitted, in an
> executable order** — not "free because computable in principle."

`gt1`'s law (GT never ships; basis free, coefficients counted) and `gt3`'s free-address circularity
law govern the same boundary; nothing in §4 claims a free address that survives them — which is
precisely why §4's explicit splits are reported as **losses**.

### 5.2 Free is not an advantage

`sx2` measured a **49 B counted** static map removing **57.2%** — **2.0×** the best free generic
extractor — with **every blend worse than the counted prior alone**. So GENERIC is not treated here
as budget to spend. `rs2`'s correction is honoured: the 80.67% invisible fraction is `D`'s null
fraction on the **camera** plane and is structurally unreachable from the token lattice (`M = D∘U`
gain range [0.6866, 1.0283], **0.0%** attenuated below 1e-3). **No symbol above is typed GENERIC on
scorer-invisibility grounds** — every one is generic because it is an *operator or algorithm*, which
is a different and much stronger argument.

---

## §6 THE VERB LEXICON — column headers for `ddm_cg1` (#809)

Emitted machine-readably at `gt2_verbs.json:verb_lexicon`, each with its operational rule. A force
reaches a class **through a production**, so the verb is the channel.

| VERB | gloss | measured by | sign |
|---|---|---|---|
| `DISPLACE` | separatrix moved; mass locally conserved | per frame per edge: `2·min(a→b, b→a)` | symmetric |
| `TRANSFER` | one side genuinely lost area | per frame per edge: `abs(a→b − b→a)`, signed to winner | antisymmetric |
| `ERODE` | surviving component lost a shallow rim | patch on component retaining ≥5%, max GT-depth ≤1 | negative |
| `GOUGE` | surviving component lost interior | same, max GT-depth >1 | negative |
| `ANNIHILATE` | **an entire component gone — a word dropped** | GT component retaining <5% of its px | negative |
| `BIRTH` | component with no GT counterpart | cx1 component with <5% overlap | positive, usually harmful |
| `FRAGMENT` | component count rises without area change | `comps(cx1) − comps(GT)` per class per frame | **area-neutral ⇒ invisible to `d_seg`** |
| `AMPLITUDE` | pre-argmax margin moved without crossing | **OUT OF SCOPE** — invisible after argmax; `hg1` owns it | — |
| `PHASE` | sub-pixel position of the structure | **OUT OF SCOPE** — argmax quantises to the lattice; `pc2` owns it | — |

**Two of nine are declared out of scope rather than guessed.** `AMPLITUDE` and `PHASE` are real and
load-bearing — `hg1`'s 11.4× barrier disparity, `pc2`'s positional DOF — but are *by construction*
unmeasurable from an argmax corpus, and this arm ran zero scorer forwards. `cg1` should carry the
columns and source them from `hg1`/`pc2`.

`DISPLACE`/`TRANSFER` are computed **per frame then summed**: computing them globally lets one
frame's forward flow cancel another's reverse flow and manufacture false conservation.

**What the verb split buys, per `hg1`:** `d_seg` prices Lane **erasure** and Road **boundary-nudge**
identically while their barriers differ **11.4×**, and `Lane→Road` has the steepest margin recovery
of all 14 sides. In verb terms: **Lane's `ANNIHILATE` is expensive to prevent and its `DISPLACE` is
cheap to command — one class, two verbs, opposite signs.** A single Lane production collapses them
into one number and no force can be aimed at either.

---

## §7 FOLLOW-ONS — every row FIRED / FOLDED / QUEUED-WITH-FIRE-ORDER

*"Noted" is not a disposition.*

| # | item | disposition |
|---|---|---|
| 1 | VERB LEXICON → `cg1` (#809) force-ledger columns | **FIRED** — machine-readable at `gt2_verbs.json:verb_lexicon`, 7 in-scope + 2 out-of-scope with owners named |
| 2 | Charter premise "prior line used sampled corpora" | **FOLDED** — retracted in frontmatter + source docstring; identity proven at source |
| 3 | My own draft FAMILY claim "region-paint cannot serve Lane" | **FOLDED** — withdrawn, replaced by the measured cost-per-px ratio (§3) |
| 4 | `sx1`'s 253,341 B cited as a lossless cost | **FOLDED** — reclassified as an entropy estimate; real coder 410,584 B (1.6207×) |
| 5 | Lane lossless contour 130,960 B, 2.30× under `W` | **QUEUED — FIRE ORDER 1.** Fire condition: any arm building a Lane carrier. Blocker: realization through R (`m11`), NOT description length. Owner: unclaimed — nearest is `dd1` |
| 6 | Phase as **expansion/scale** or per-region, not whole-frame translation | **QUEUED — FIRE ORDER 2.** Fire condition: F7 gets an owner. Needs a sub-pixel substrate; argmax cannot resolve it, so measure pre-argmax |
| 7 | Address-dominance law (76.4 / 78.5 / 78.5%) → generative-over-corrective | **QUEUED — FIRE ORDER 3.** Fire condition: next correction-stream proposal. Predicts ~78% address; re-form as a generative production |
| 8 | Frame-targeting is a small-edge instrument (§2.1) | **QUEUED — FIRE ORDER 4.** Fire condition: any per-frame force. Movable↔MyCar is 100% in 10 frames; Road↔Lane is 2.18× uniform |
| 9 | `d_seg` has no per-class term (§0.4) | **QUEUED — no fire order.** Scoring is frozen and contest-fixed; recorded as a permanent property of the objective, not an actionable item |

## §8 NEXT-IF-RESUMED

All six stages landed; `--stage all` reproduces every number from the corpus in ~3 min. Highest-value
resume, in order: (a) price a **stroke** production (centreline + width) against the contour
production for Lane — the contour double-counts each side of a 2.5 px structure, so a centreline
should be ~2× cheaper than 130,960 B; (b) re-run the phase stage with an **expansion** parameter
(scale about a focus of expansion) instead of translation; (c) hand `cg1` the per-frame edge×verb
rows for its force ledger.
