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
baseline_named: "live best S = 0.7910689 at 353,805 B; seg leg 0.4311790; gap to PR130 = 0.6189279.
  Every S-delta in this memo is stated against THAT baseline and against the PR130 seg leg 0.02966."
verdict_scope:
  - claim: "a REGION production can serve the Lane class"
    verdict: REFUTED
    scope: "FAMILY - region-paint productions (any grammar whose cost scales with boundary and whose
      delivered pixels scale with area), on the Lane class, on this corpus"
    why_this_high: "the refutation is a ratio of two MEASURED geometric quantities (area/perimeter
      = 1.407 for Lane vs 21.93/97.4/176.2 for the region-like classes), not the failure of one
      implementation. Any member of the family inherits the ratio."
    why_not_higher: "it does NOT refute region productions generally - they are 125x more efficient
      on Undrivable. The verdict is class-scoped, not paradigm-scoped."
  - claim: "sx1's 253,341 B is the lossless cost of the label field L*"
    verdict: REFUTED_AS_A_CODER_COST
    scope: "FORMULATION - the number is a context-model ENTROPY ESTIMATE (H1+model)/8, not a coder
      output. The real coder costs 410,584 B, 1.6207x more."
    why_not_higher: "sx1's estimate is correct AS an estimate and its own 60-frame lzma row
      extrapolates to within 4.1% of the measured n600 coder cost. Only the reading of the estimate
      as an achievable cost is refuted."
  - claim: "the seg gap is a boundary-displacement problem"
    verdict: REFUTED_AS_THE_DOMINANT_MODE
    scope: "FORMULATION - on the Road<->Lane edge, which is 46.23% of flips. Measured per frame,
      TRANSFER (net area handover, 57.0%) exceeds DISPLACE (conserved wobble, 43.0%)."
    why_not_higher: "displacement IS the dominant verb on other edges (Road<->Movable 61.0%
      DISPLACE); see the edge table. The gap is not one verb."
  - claim: "this arm is the first grammar-induction run on a real corpus (the predecessor
      docstring's premise)"
    verdict: REFUTED
    scope: "INSTANCE - the predecessor's own dying catch, now verified at source: g1 (#620)
      consumed experiments/results/mlx_fleet_gt_cache/gt_n600.npz on its command line and dv2
      (#651) consumed the same cache with SHA-256 verification. That cache's lstars is
      BIT-IDENTICAL to this arm's corpus: 0 of 117,964,800 px differ vs ddm_pu2's
      gt_argmax_n600.npy (measured 2026-08-03). gt2 is the UPGRADE (shipped-receiver side +
      real-coder pricing + verb decomposition), not the first run."
  - claim: "an explicit two-part factoring of L* (lexicon+residual OR keyframe+delta) saves bytes"
    verdict: REFUTED
    scope: "FORMULATION - mask-addressed explicit factorings under general-purpose coders, on
      this corpus. Static mode+residual = 434,156 B (+5.7% vs implicit 410,584); temporal
      keyframe+churn = 564,784 B (+37.6%, worse even than per-frame-independent 558,364). Both
      die the same death: the ADDRESS is 76.4% / 78.5% of the explicit cost."
    why_not_higher: "does NOT refute factored GRAMMARS generally - it refutes transmitting the
      address as a mask. A receiver-generated (boundary-conditioned) address is untested and is
      the named next measurement."
verdict_scope_ladder: "INSTANCE < FORMULATION < FAMILY < PARADIGM."
consumes:
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy   (600x384x512 uint8)
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy  (600x384x512 uint8)
  - .omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md      (the corpus + its fail-closed control row)
  - .omx/research/ddm_dd1_lane_component_census_n600.json        (area/perimeter census - CONSUMED, not re-measured)
  - .omx/research/ddm_dd1_contour_coherence_n600.json            (orientation coherence length)
  - .omx/research/ddm_sx1_label_field_mdl_n600.json              (the H1 estimate this arm prices against)
  - .omx/research/direct_description_g1_grammar_induction_20260722*     (#620 prior art)
  - .omx/research/ddm_dv1_description_vocabulary_*_20260723*            (#650 prior art)
  - .omx/research/ddm_dv2_sdwl1_*_20260723*                             (#651 prior art)
  - .omx/research/codex_findings_ddm_pf1_pointfree_program_description_20260723_codex.md  (#664 prior art)
produces:
  - experiments/ddm_gt2_gt_tongue_induction.py                   (the probe; 5 stages, re-runnable)
  - /Volumes/VertigoDataTier/pact/ddm_gt2_20260803/gt2_control.json
  - /Volumes/VertigoDataTier/pact/ddm_gt2_20260803/gt2_verbs.json
  - /Volumes/VertigoDataTier/pact/ddm_gt2_20260803/gt2_mdl.json
  - /Volumes/VertigoDataTier/pact/ddm_gt2_20260803/gt2_split.json
  - /Volumes/VertigoDataTier/pact/ddm_gt2_20260803/gt2_temporal.json
  - /Volumes/VertigoDataTier/pact/ddm_gt2_20260803/gt2_lane.json
consumers: [MAIN, ddm_cg1r (task #809 force ledger - gt2_verbs.json:verb_lexicon + the VERB_* keys are its column headers, machine-readable), ddm_dd1]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_gt2 — the tongue we are learning from GT, re-run on the real corpus

## §0 ANSWER FIRST

**We are not learning a tongue from GT. We are learning a tongue that cannot pronounce one of its
five words — and the scorer we optimise against cannot hear the difference.**

Four measured statements, all on the real n600 GT argmax corpus, zero scorer forwards:

1. **The dropped word class is dropped at the level of whole WORDS, not pixels.** At n600,
   **9,655 of 16,581 Lane connected components — 58.23% of all Lane words — are entirely
   annihilated** (retaining <5% of their own pixels), against **5.45% for Road and 0.00% for
   MyCar**. Lane loses **26.903% of its entire population** to misclassification; Road loses
   **0.559%** — a **48.1× disparity in per-pixel reliability** between two classes whose
   separatrix they share.

2. **The scorer is structurally blind to it.** `upstream/modules.py` computes SegNet distortion as a
   uniform per-pixel mean. Lane is **0.5855%** of pixels, so misclassifying 26.9% of the entire
   Lane class costs only **0.158 S** (185,801 px of d_seg) — the same order as Road's boundary
   wobble (0.130 S), which is a far less destructive act. **The objective cannot express which
   word class it lost.**

3. **Storing the tongue is not affordable, so the grammar must GENERATE.** A REAL coder on the full
   GT label field costs **410,584 B (lzma9e)** — **1.1605× the entire 353,805 B live archive**.
   `sx1`'s published **253,341 B** is a context-model entropy ESTIMATE, not a coder output; the
   real-coder gap is **1.6207×**. Any plan that budgets against 253,341 B is budgeting against a
   number no coder has produced.

4. **The lexicon should be static and the sentences per-pair — but the factoring must stay
   IMPLICIT, because the ADDRESS is the whole rate problem.** Coding the corpus as one object
   costs **410,584 B**; each frame independently **558,364 B** (shared context worth **26.5%**).
   Yet **every EXPLICIT factoring loses**: static mode+residual = **434,156 B (+5.7%)**, temporal
   keyframe+churn = **564,784 B (+37.6%, worse than per-frame-independent)** — and both die the
   same death: **the address (WHERE the residual/churn is) is 76.4% / 78.5% of their cost**,
   while the payload (WHICH class) is cheap and the static lexicon itself is nearly free
   (**424 B** for the whole modal field). The tongue's rate problem is not the words — it is
   saying WHERE. The design consequence: **the address must be GENERATED by the receiver from
   geometry it already holds, never transmitted as a mask** (§4, §8).

**The one-line design consequence.** Lane needs **two productions with opposite signs**, not one:
it is the most expensive class per pixel to HOLD (`area/perimeter = 1.407`, **125× worse than
Undrivable** under a region production) and — per `hg1` — the cheapest per unit of margin to MOVE.
A lexicon that gives Lane a single production is structurally unable to say that, and every force
aimed through that single production will trade one against the other without being able to see it.

---

## §1 WHAT WAS ACTUALLY NEW HERE (and what I only re-anchored)

### 1.0 The premise correction — this arm is the UPGRADE, not the first real-corpus run

The predecessor's first docstring claimed the prior grammar line (#620/#650/#651/#664) "was built
against sampled or reconstructed corpora." **That premise is FALSE and is corrected here, verified
at source:** `g1` (#620) consumed `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` on its own
recorded command line, and `dv2` (#651) consumed the same cache with SHA-256 verification. That
cache's `lstars` array is **BIT-IDENTICAL to this arm's corpus — 0 of 117,964,800 px differ**
against `ddm_pu2`'s `gt_argmax_n600.npy` (measured 2026-08-03, all 600 frames). The prior line
already spoke to the real GT. What it did NOT have, and what this arm adds:

1. **the SHIPPED-RECEIVER side** (`cx1_argmax_n600.npy`) — the tongue as SPOKEN vs as written,
   which is what makes flips, verbs and the conjugation diagnosis measurable at all;
2. **REAL-coder pricing** — every grammar priced by an actual lzma/brotli/zlib output, with the
   estimate-vs-coder gap itself reported (it is 1.62×; §3 of the prior line budgeted against it);
3. **the VERB decomposition** — the force-attribution channel `ddm_cg1r` consumes.

Per the anti-re-anchoring discipline (`m38`), the prior-law prediction line comes FIRST:

| Prior law | What it predicted for this arm | Measured outcome |
|---|---|---|
| `sx1`: 89.69% of bits on 2.16% separatrix px, 393× concentration | boundary dominates any grammar's cost | **REPRODUCED exactly** from sx1's own JSON (bnd_bits/H1_bits = 89.69%, bnd_px/PX = 2.163%) — re-anchor, not discovery |
| `sx1`: L* = 253,341 B lossless | a coder should land near it | **REFUTED as a coder cost: 410,584 B, 1.6207×** — NEW |
| `hs1`: static lexicon amortises, per-pair addressing does not (379×) | whole-corpus coding should beat per-frame | **CONFIRMED from an independent direction: 26.5%** — NEW corroboration |
| `dd1`: Lane median minor axis 2.51 px, area/perimeter 1.407 | region-paint should fail on Lane | **CONFIRMED and given a mechanism** — NEW |
| `pu2`: Lane −26.50% / Road +27.22% | net erosion | **REPRODUCED exactly, and the denominator identified** (% of total flips) — NEW clarification |
| `m91`: Road is the hub at 87.8% | Road should dominate the edge table | **CONFIRMED at 87.48% on a different corpus** — NEW corroboration |
| `hs1`: the ADDRESS is the cost (379× static-vs-per-pair) | explicit factorings should die on address | **CONFIRMED ×2 with real coders: address = 76.4% (static) / 78.5% (temporal) of the explicit two-part cost; both factorings LOSE to implicit joint coding** — NEW law |
| `dd1`: component census (27.64 Lane comps/frame etc.) | this arm's independent component pass should agree | **REPRODUCED exactly, all 5 classes** (27.64 / 2.11 / 1.08 / 3.68 / 1.00 comps/frame) — fail-closed cross-check |
| charter (`pu2`): the Lane drop is "erosion" | one verb should describe it | **REFINED: one verb cannot.** By WORD count the drop is `ANNIHILATE` (58.23% of Lane words gone whole); by PIXEL count it is `ERODE` (73.0% of Lane flip px are rim-peel of survivors). Two different forces, two different cures — a single "erosion" column would aim both at neither |

**Honest accounting: two of the nine rows above are re-anchors, not discoveries.** They are
reported as such.

### A denominator that was ambiguous, now pinned

`pu2` quotes "Lane −26.50%, Road +27.22%" without its denominator. It is **net change as a
percentage of TOTAL FLIPS**. I reproduce it exactly: **−26.499% / +27.222%**. Two other
denominators give very different-looking numbers for the same fact, and all three are true:

| Quantity | Road | Lane | Undrivable | Movable | MyCar |
|---|---:|---:|---:|---:|---:|
| net px | +138,461 | −134,786 | +2,127 | −37,733 | +31,931 |
| net as % of **total flips** (`pu2`'s) | +27.222% | **−26.499%** | +0.418% | −7.418% | +6.278% |
| net as % of **own GT population** | +0.505% | **−19.516%** | +0.004% | −2.584% | +0.106% |
| **% of own population misclassified** | 0.559% | **26.903%** | 0.128% | 5.398% | 0.054% |

**The third row is the one that was missing, and it is the sharpest.** A ΔS without its baseline is
unanchored; a percentage without its denominator is the same failure one level down.

---

## §2 THE CONJUGATION DIAGNOSIS — why the tongue erases Lane and repaints Road

*(directive (d); coordinator's "semantics and verbs")*

### 2.1 The edge table — EDGE-indexed, per `m91`, never class-indexed

Corpus: cx1 vs GT, n600, **508,640 flips**, `d_seg` 0.004311794704861 (control rel **1.09e-06**).

| edge | flips | % of flips | asym | dominant direction | S contrib | % of gap |
|---|---:|---:|---:|---|---:|---:|
| **Road↔Lane** | **235,148** | **46.231** | 3.65 | **Lane→Road** | 0.19934 | **32.207** |
| Road↔Undrivable | 89,545 | 17.605 | 1.68 | Undrivable→Road | 0.07591 | 12.264 |
| Road↔MyCar | 63,027 | 12.391 | 3.02 | Road→MyCar | 0.05343 | 8.632 |
| Undrivable↔Movable | 61,892 | 12.168 | 2.34 | Movable→Undrivable | 0.05247 | 8.477 |
| Road↔Movable | 57,225 | 11.251 | 1.61 | Movable→Road | 0.04851 | 7.838 |
| Lane↔MyCar | 903 | 0.178 | 1.36 | Lane→MyCar | 0.00077 | 0.124 |
| Lane↔Movable | 681 | 0.134 | 12.35 | Lane→Movable | 0.00058 | 0.093 |
| Movable↔MyCar | 135 | 0.027 | 15.88 | Movable→MyCar | 0.00011 | 0.018 |
| Lane↔Undrivable | 84 | 0.017 | 1.27 | Undrivable→Lane | 0.00007 | 0.012 |

**Road participates in 87.48% of all flips** (444,945 of 508,640) — `m91`'s hub structure confirmed
on an independent corpus (it measured 87.8% of 458,738 on `pc2`'s different base; **the two flip
counts are different populations and must not be pooled**, but the hub fraction agrees to 0.3 pp).

**The bottom four edges together are 0.356% of flips.** Nine of the ten possible edges carry
non-zero mass, but the grammar only has to be good at five.

### 2.2 The mechanism — erasure is selective for THINNESS, and Lane has no interior

`dd1`'s census (consumed, not re-measured):

| class | area/perimeter | median minor axis | components/frame |
|---|---:|---:|---:|
| Undrivable | 176.20 | 192.1 px | 1.08 |
| MyCar | 97.41 | 97.8 px | 1.00 |
| Road | 21.93 | 94.0 px | 2.11 |
| Movable | 11.07 | 9.0 px | 3.68 |
| **Lane** | **1.41** | **2.51 px** | **27.64** |

**`area/perimeter` IS the efficiency of a region production** — pixels delivered per unit of
boundary description. Lane delivers **1.41 px per boundary px**; Undrivable delivers **176.2**.
**A region production is 125× less efficient on Lane than on Undrivable**, and that ratio is a
property of the class's geometry, not of any implementation. This is why the refutation in the
frontmatter is scoped to FAMILY rather than to one codec.

A structure 2.51 px wide has essentially no interior. `rz1` measured 6.92% of Lane at depth ≥2 px
against Road's 63.64%. **Every Lane pixel is a boundary pixel**, so every Lane pixel is contested,
so a production that resolves contests by area loses all of them at once.

### 2.3 The verb decomposition (n600, landed — §7 carries the full tables)

The n600 component pass confirms the smoke's shape at 200× the scale: **Lane loses 9,655 of its
16,581 GT words whole — a 58.23% word-annihilation rate — against Road's 5.45% and MyCar's 0.00%.**
By pixel count the same drop reads differently: 73.0% of Lane's flip pixels are `ERODE` (rim-peel
of components that survive), 25.4% are `ANNIHILATE`. **Both readings are true and they are
different failures**: annihilation is the recall catastrophe (the word is gone), erosion is the
fidelity tax (the word is thinner). A force ledger that carries only "erosion" aims at neither.

Two corpus-level asymmetries the verb pass surfaced that no prior arm had measured:

- **The failure is RECALL, not PRECISION.** `ANNIHILATE` totals **10,124 words** across all
  classes; `BIRTH` (hallucinated words) totals **616** — a **16.4×** asymmetry. The receiver
  under-generates the tongue; it almost never invents it.
- **`FRAGMENT` measured NEGATIVE everywhere but MyCar (+2): the receiver CONSOLIDATES.** cx1
  carries 6,038 FEWER Lane components than GT (−533 Road, −492 Movable, −38 Undrivable). One word
  split into several essentially never happens; several words merged (or dropped) into fewer is
  the norm. `cg1r`'s column should be signed `FRAGMENT(+)/CONSOLIDATE(−)`.

**In grammar terms the answer to "why erase Lane and repaint Road" is:**

> The tongue has **one production** — *paint a region and let the classes compete for pixels by
> area*. Under that production Lane is not a word the grammar can hold: it is 0.5855% of the area
> and 20.3% of the perimeter, so in every contest it is the loser, and the winner is whichever
> class it borders — which is Road **87.48%** of the time. **Erasure-then-repaint is not two events.
> It is one event seen from both sides**, which is exactly why `m91` says decompose per EDGE.

### 2.4 Is there a production rule that CANNOT make this error?

**Yes, and its defining property is that it does not resolve pixel contests by area.**

A production is **erasure-immune for class c** iff the existence of a c-instance in its output is
determined by a term the other classes cannot outvote. Two concrete forms:

- **Curve-native / stroke production.** `paint_stroke(polyline, width) -> c`. The instance exists
  because the program says so; there is no competition. Its cost scales with **length**, which for
  Lane is the right measure (aspect ratio 25.48). Priced in §5.
- **Component-native production with an explicit existence symbol.** `emit(component_id,
  transform) -> c`, per `mf1`'s Movable result (2.47 components/frame at **81.4× better than W**).
  A dropped component is then a *missing symbol in the program string*, which is detectable at
  encode time — where a dropped region is silently absorbed by a neighbour and is not.

**The general rule the corpus supports: a class whose `area/perimeter` is below ~2 cannot be held by
an area-competitive production and needs an existence-carrying one.** Lane (1.41) is the only class
of the five below that line; Movable (11.07) is the nearest other candidate and is exactly where
`mf1` already found component-native productions paying 81.4×.

---

## §3 THE SYMBOL TABLE — every symbol typed before it is priced

*(§3 of the charter, the hard NO-FAKE gate. `#913` is the live instance of the failure: a dense
raster LZMA baseline wearing a boundary-edge codec's name. Typing precedes pricing here so the same
thing cannot happen one abstraction layer up.)*

**The three-way test** (operator ruling 2026-08-03): a property of the **fixed operators** is
GENERIC and free; **scorer weights** are video-invariant but 73 MB (economic, not banned); **this
clip's content** is COUNTED.

| symbol | type | why | counted? |
|---|---|---|---|
| the 5-class semantics + their canonical index order | GENERIC | property of the frozen SegNet, invariant across clips | no |
| the image lattice; `D`'s 2×2 disjoint sampling; the 22.70% blind fraction | GENERIC | operator property (`m86`) | no |
| `fillPoly` / stroke rasteriser; dilation; distance transform | GENERIC | algorithms, not content | no |
| zigzag+varint integer code | GENERIC | a standard code | no |
| **boundary band = `dilate(boundary(X))` where X is already-transmitted payload** | **GENERIC** | a deterministic function of bytes the receiver already holds — it re-derives it, so it is not paid for twice | **no** |
| static per-pixel modal lexicon | **FITTED** | induced from THIS clip | **yes, exactly** |
| polyline control points, widths | **FITTED** | this clip's geometry | **yes, exactly** |
| per-pair residual payload | **FITTED** | this clip's frames | **yes, exactly** |
| pre-argmax margin / logit fields | **OUT OF SCOPE** | not measurable from an argmax corpus; `hg1` owns it | n/a |

### 3.1 The causality constraint on "the address is free"

`rz1`'s result — the dilated label boundary, computable from the decoder's own `L*`, recovers
**93.53%** of the margin separatrix — is sound **only under a causality constraint that must be
stated or it becomes a fake**: the receiver has no SegNet (73 MB, not in the archive). It can only
re-derive a boundary from a label field **it has already been sent**. So:

> **ADDRESS is free iff it is a deterministic function of payload already transmitted, in an order
> the receiver can actually execute.** Not "free because it is computable in principle."

Under that constraint the free-address claim survives, and §4 prices it.

### 3.2 Free is not an advantage — the counted rival must be priced

`sx2` measured a **49 B counted** static map removing **57.2%**, which is **2.0× the best free
generic extractor**, and **every blend was worse than the counted prior alone**. So this arm does
not treat GENERIC as a budget to spend. `rs2`'s correction is honoured too: the 80.67% invisible
fraction is `D`'s null fraction on the **camera** plane and is **structurally unreachable from the
token lattice** (`M = D∘U` gain range [0.6866, 1.0283], **0.0%** attenuated below 1e-3). **No symbol
in this memo is typed GENERIC on the grounds of scorer-invisibility.** Every GENERIC row above is
generic because it is an *operator or algorithm*, which is a different and much stronger argument.

---

## §4 THE STATIC / PER-PAIR SPLIT — address priced apart from payload

*(directive (b))*

**The split landed, and it sharpens the direction into a LAW: shared context wins, explicit
factoring loses, and the address is the whole problem.** Three measured factorings of the same
field, all real lzma9e outputs (`gt2_split.json`, `gt2_temporal.json`; full tables §7.4–7.5):

| factoring of L* | bytes | vs implicit | address share |
|---|---:|---:|---:|
| **implicit** (whole corpus, one coder context) | **410,584** | 1.000× | — |
| explicit STATIC (mode field 424 B + residual) | 434,156 | +5.7% | **76.4%** |
| per-frame independent (no sharing at all) | 558,364 | +36.0% | — |
| explicit TEMPORAL (keyframe 984 B + churn masks) | 564,784 | +37.6% | **78.5%** |

Three design facts fall out, each measured:

1. **The static lexicon is nearly FREE — 424 B for the entire per-pixel modal field.** The words
   cost nothing to hold. (The payload of the static residual — WHICH class where the mode is
   wrong — is also cheap: 102,496 B.)
2. **The address is 76–79% of any explicit factoring's cost, and it kills both.** Saying WHERE
   this pair differs from the template (static) or from the previous pair (temporal) through a
   transmitted mask costs more than the difference is worth. `hs1`'s 379× address finding, now
   confirmed with real coders from two more directions.
3. **The residual does not live near the STATIC boundary — because the separatrix MOVES.** Only
   **28.0%** of residual px fall within r=1 of the static field's boundary (62.4% at r=8), while
   per `pc2` **93.89%** of receiver flips sit within 3 px of the TRUE per-frame separatrix. The
   boundary language is real but DYNAMIC: a static band cannot address it; the receiver's own
   current field can (`rz1`: 93.53% of the separatrix is re-derivable from payload already held).

**The design consequence, stated once:** the grammar's address must be a deterministic function of
already-transmitted payload (§3.1's causality constraint) — a boundary-conditioned context model
where "near the receiver's current separatrix" is the prior, not a transmitted mask. That is the
only address route left standing after this arm, and it is the named next measurement (§8).
(Sharing context across the 600 pairs is worth 147,780 B = 26.5% — `hs1`'s finding arriving from
an independent direction; full coder table §7.3.)

---

## §5 THE LANE PRODUCTION — region-paint vs curve-native at matched fidelity

*(directive (c); landed from `gt2_lane.json`, full table §7.6)*

The structural argument is §2.2. The price is measured with a real coder at six simplification
tolerances, with the reconstruction error reported **in the unit the score charges** (pixels that
would become flips), so the two grammars are compared at matched fidelity rather than by assertion.
Three measured outcomes:

1. **Contour programs beat bitplane rasters lossless, for every class tested:** at ε=0 (zero
   reconstruction error) G_poly costs 0.691× raster on Lane, 0.672× on Road, 0.613× on Movable —
   a real 1.4–1.6× win with the rasteriser GENERIC (uncounted) and only the vertex program counted.
2. **Every lossy operating point (ε>0) is DOMINATED at the live exchange rate.** Valuing the
   introduced error pixels at W = 1.2731 B/flip: Lane ε=0.5 saves 1,096 B while introducing error
   worth 30,615 B (**27.9× against**); Lane ε=1.0 is 3.3× against; Road ε=1.0 is 6.3× against;
   Movable ε=1.0 is 3.9× against. Road ε=0.5 even costs MORE bytes AND adds error. **The
   simplification knob buys nothing anywhere on this corpus — lossless programs or nothing.**
3. **No storage grammar makes Lane affordable — the census's 125× prediction confirmed by a real
   coder.** Per pixel HELD, Lane costs 0.2743 B/px under raster vs Road's 0.0108 (**25.4×**), and
   0.190 vs 0.0072 B/px under lossless polygons (**26.2×**) — the disparity is grammar-invariant
   because it is the class's geometry (`area/perimeter` 1.41 vs 21.93). Lane's lossless program is
   130,960 B (≈218 B/frame amortised) for 0.586% of the pixels — 37% of the whole live archive.
   **The existence-carrying Lane production of §2.4 must therefore be temporally GENERATED
   (tracked words: birth/death/motion of ~27.6 persistent components per frame), not stored
   per-frame in any chart.**

---

## §6 THE VERB LEXICON — column headers for `ddm_cg1r`'s force ledger

*(coordinator's directive; `ddm_cg1r` = task #809, row shape `(class|edge) × force × VERB →
helps/harms/neutral + protection`)*

**Machine-readable emission:** `gt2_verbs.json` carries `verb_lexicon` (each verb with gloss,
operational measurement rule, sign convention, in-scope flag) and the measured masses under
`edge_indexed[].VERB_DISPLACE_px / VERB_TRANSFER_px` and `class_indexed_secondary[].VERB_ANNIHILATE_* /
VERB_BIRTH_* / VERB_ERODE_px / VERB_GOUGE_px / VERB_FRAGMENT_component_delta` — exactly the column
shapes `cg1r` consumes.

A force acts on a class **through a production**, so the verb is the channel by which any training
force, loss term, guard or carrier reaches a class. These names are emitted machine-readably at
`gt2_verbs.json:verb_lexicon`, each with its operational measurement rule.

| VERB | gloss | measured by | sign |
|---|---|---|---|
| `DISPLACE` | the separatrix moved; mass locally conserved | per frame per edge: `2·min(a→b, b→a)` | symmetric |
| `TRANSFER` | one side genuinely lost area to the other | per frame per edge: `|a→b − b→a|`, signed to the winner | antisymmetric |
| `ERODE` | surviving component lost a shallow rim | patch on a component retaining ≥5%, max GT-depth ≤1 | negative |
| `GOUGE` | surviving component lost interior | same, but max GT-depth >1 | negative |
| `ANNIHILATE` | an entire component is gone — **a whole word dropped** | GT component retaining <5% of its pixels | negative |
| `BIRTH` | a component with no GT counterpart | cx1 component with <5% overlap | positive, usually harmful |
| `FRAGMENT` | component count rises without proportional area change | `components(cx1) − components(GT)` per class per frame | **can be area-neutral ⇒ invisible to `d_seg`** |
| `AMPLITUDE` | pre-argmax margin moved without crossing | **OUT OF SCOPE** — invisible after argmax by construction; `hg1` owns it | — |
| `PHASE` | sub-pixel position of the structure | **OUT OF SCOPE** — argmax quantises position to the lattice; `pc2` owns it | — |

**Two of the nine verbs are declared out of scope rather than guessed.** `AMPLITUDE` and `PHASE`
are real and load-bearing — `hg1`'s 11.4× barrier disparity and `pc2`'s positional DOF both live
there — but they are *by construction* unmeasurable from an argmax corpus, and this arm ran zero
scorer forwards. `cg1r` should carry the columns and source them from `hg1`/`pc2`.

**Why `DISPLACE`/`TRANSFER` are computed per frame and then summed:** doing it globally lets one
frame's forward flow cancel another frame's reverse flow and manufacture false conservation. The
per-frame form also gives `cg1r` and any future carrier the **individual frames** where a verb
concentrates — and the measured answer is a REFUTATION of the home-frame hope at edge level:
**on all five big edges the top-10 frames carry only 3.6–15.4% of the TRANSFER mass and every one
of the 600 frames is touched.** The verbs are DIFFUSE where the mass is; only the four tiny edges
(0.36% of flips) concentrate (Movable↔MyCar: 100% in 10 frames). A force aimed per-frame buys
almost nothing on the edges that matter; it must be aimed per-EDGE, through the production.

**The sharpest thing the verb split buys, per `hg1`:** `d_seg` prices **Lane erasure** and **Road
boundary-nudge** identically while their barriers differ **11.4×**, and `Lane→Road` has the steepest
margin recovery of all 14 sides. In verb terms that is: **Lane's `ANNIHILATE`/`ERODE` is expensive
to prevent and its `DISPLACE` is cheap to command — one class, two verbs, opposite signs.** A single
Lane production collapses them into one number and no force can be aimed at either.

---

## §7 MEASURED TABLES

All numbers are real n600 measurements on the bit-verified corpus, `[macOS-CPU advisory]`,
zero scorer forwards. Coders are actual lzma/brotli/zlib outputs. Control: 508,640 flips,
`d_seg` 0.004311794704861 (rel 1.09e-06 vs `pu2`'s fail-closed row). The verb decomposition
sums EXACTLY to the control (per-class residual ≤ 34 px, the surviving-pixel remainder inside
annihilated components).

### 7.1 Edge × verb (`gt2_verbs.json:edge_indexed`)

| edge | flips | % flips | DISPLACE px | disp% | TRANSFER px | xfer% | direction | top-10-frame share of TRANSFER | frames touched |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Road↔Lane | 235,148 | 46.23 | 101,070 | 43.0 | 134,078 | **57.0** | Lane→Road | 3.6% | 600 |
| Road↔Undrivable | 89,545 | 17.60 | 47,498 | 53.0 | 42,047 | 47.0 | Undrivable→Road | 5.8% | 600 |
| Road↔MyCar | 63,027 | 12.39 | 26,020 | 41.3 | 37,007 | 58.7 | Road→MyCar | 6.7% | 600 |
| Undrivable↔Movable | 61,892 | 12.17 | 31,370 | 50.7 | 30,522 | 49.3 | Movable→Undrivable | 15.4% | 600 |
| Road↔Movable | 57,225 | 11.25 | 34,922 | **61.0** | 22,303 | 39.0 | Movable→Road | 9.5% | 600 |
| Lane↔MyCar | 903 | 0.18 | 120 | 13.3 | 783 | 86.7 | Lane→MyCar | 22.0% | 241 |
| Lane↔Movable | 681 | 0.13 | 30 | 4.4 | 651 | 95.6 | Lane→Movable | 26.4% | 270 |
| Movable↔MyCar | 135 | 0.03 | 16 | 11.9 | 119 | 88.1 | Movable→MyCar | 100% | 10 |
| Lane↔Undrivable | 84 | 0.02 | 8 | 9.5 | 76 | 90.5 | Undrivable→Lane | 84.2% | 20 |

The verb MIX is per-edge: the head edge (Road↔Lane) is TRANSFER-dominant (the boundary-displacement
reading is refuted there), Road↔Movable is DISPLACE-dominant, two edges sit at ~50/50. And the
mass is DIFFUSE: every big edge touches all 600 frames with ≤15.4% of transfer in its worst 10.

### 7.2 Class × component verbs (`gt2_verbs.json:class_indexed_secondary`)

| class | flips | % of own pop misclassified | ANNIHILATE words (rate) | ANNIH px | BIRTH words / px | ERODE px | GOUGE px | FRAGMENT Δcomps | erasure selectivity (flip-rate d≤1 / d>1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Road | 153,242 | 0.559 | 69 (5.45%) | 315 | 16 / 36 | 141,180 | 11,747 | −533 | 250× |
| **Lane** | **185,801** | **26.903** | **9,655 (58.23%)** | 47,226 | 591 / 1,264 | 135,683 | 2,926 | **−6,038** | **16×** |
| Undrivable | 74,697 | 0.128 | 39 (6.00%) | 197 | 3 / 6 | 68,608 | 5,892 | −38 | 2,034× |
| Movable | 78,833 | 5.398 | 361 (16.36%) | 8,180 | 5 / 762 | 53,940 | 16,718 | −492 | 27× |
| MyCar | 16,067 | 0.054 | 0 (0.00%) | 0 | 1 / 45 | 14,827 | 1,240 | +2 | 1,153× |

- `ANNIHILATE` total 10,124 words vs `BIRTH` 616 — **16.4× recall-over-precision asymmetry**.
- `FRAGMENT` is negative for 4 of 5 classes: the receiver **CONSOLIDATES** (column should be
  signed `FRAGMENT(+)/CONSOLIDATE(−)`).
- Lane's LOW selectivity (16×) is not robustness — it is the absence of an interior: 75.0% of
  Lane's own pixels sit at depth ≤1, so there is no deep refuge for flips to spare.
- GT components/frame reproduce `dd1`'s census exactly: 27.64 / 2.11 / 1.08 / 3.68 / 1.00.

### 7.3 Real-coder MDL of L* (`gt2_mdl.json`)

| grammar | codec | bytes | bits/px |
|---|---|---:|---:|
| whole corpus, one object | **lzma9e** | **410,584** | 0.02784 |
| whole corpus, one object | brotli11 | 424,728 | 0.02880 |
| whole corpus, one object | zlib9 | 581,266 | 0.03942 |
| per-frame independent | lzma9e | 558,364 | 0.03787 |

vs `sx1`: H1 estimate 253,341 B → **coder/estimate = 1.6207×**; `sx1`'s own 60-frame lzma row
extrapolates to 428,120 B → measured/extrapolated = 0.959 (the 60-frame row was honest; the
ESTIMATE read as a cost was not).

### 7.4 Static split (`gt2_split.json`)

| part | type | bytes |
|---|---|---:|
| static lexicon (per-pixel mode, whole field) | FITTED | **424** |
| residual address (explicit masks, 3,126,195 px = 2.65% of field) | FITTED | 331,236 |
| residual payload (class symbols) | FITTED | 102,496 |
| **two-part total** | | **434,156** (+5.7% vs implicit 410,584) |

Address share **76.4%**. Band coverage of residual around the STATIC boundary: r=1 28.0%,
r=2 35.9%, r=3 42.1%, r=5 51.9%, r=8 62.4% — against `pc2`'s 93.89% within 3 px of the TRUE
per-frame separatrix. **The boundary language is dynamic; a static band cannot address it.**

### 7.5 Temporal split (`gt2_temporal.json`)

| part | bytes |
|---|---:|
| keyframe (frame 0, lzma9e) | 984 |
| churn address (599 masks; churn 2,449 px/step = 1.25% of frame; min/med/max 1,409/2,328/9,184) | 442,568 |
| churn payload | 121,232 |
| **two-part total** | **564,784** (+37.6% vs implicit; worse than per-frame-independent 558,364) |

Address share **78.5%**. The churn is small and steady; SAYING WHERE it is costs everything.
(Corpus geometry: consecutive frame_1 fields are 2 video frames apart — this is the un-warped,
zero-motion-model baseline for the GENERATE production; the pair screw should shrink it, §8.)

### 7.6 Lane production pricing (`gt2_lane.json`; W = 1.2731 B/flip, owned by `ddm_wf2`)

| class | grammar | bytes | err px | B per held px | vs raster |
|---|---|---:|---:|---:|---:|
| Lane (690,639 px) | raster bitplane | 189,460 | 0 | 0.2743 | 1.000 |
| Lane | polygons ε=0 | **130,960** | 0 | 0.1896 | **0.691** |
| Lane | polygons ε=1.0 | 75,532 | 144,884 | — | 0.399 |
| Road (27,407,046 px) | raster bitplane | 295,576 | 0 | 0.0108 | 1.000 |
| Road | polygons ε=0 | **198,580** | 0 | 0.0072 | **0.672** |
| Movable (1,460,325 px) | raster bitplane | 62,288 | 0 | 0.0427 | 1.000 |
| Movable | polygons ε=0 | **38,152** | 0 | 0.0261 | **0.613** |

Every ε>0 row is dominated at W (error value / byte saving: Lane ε=0.5 **27.9× against**, ε=1.0
3.3×; Road ε=1.0 6.3×; Movable ε=1.0 3.9×; Road ε=0.5 costs MORE bytes AND adds error). Per held
pixel Lane is 25.4× (raster) / 26.2× (polygons) more expensive than Road — grammar-invariant,
because it is the geometry (`area/perimeter` 1.41 vs 21.93).

---

## §8 NEXT-IF-RESUMED

Pointer honesty first: **the exact pointer 0.1910828242 [contest-CPU] is UNMOVED; this arm fired
no gate and claims no score.** What it leaves behind is a measured design law and four named
follow-ons, in EV order:

1. **The boundary-conditioned address (the only address route left standing).** Both explicit
   factorings died on a 76–79% address share, while the residual/churn provably lives near the
   receiver-known DYNAMIC boundary (`pc2` 93.89% ≤3 px; `rz1` 93.53% re-derivable). Next $0
   measurement: an arithmetic coder over context classes {distance-to-receiver-boundary ×
   local-edge-pair}, pricing the SAME residual streams of §7.4/§7.5 with the address absorbed
   into per-pixel priors. The win condition is beating 410,584 B implicit; every input is already
   on the SSD.
2. **Motion-compensated churn (the GENERATE baseline with the screw in).** §7.5's 2,449 px/step
   churn is un-warped. Warp `L*_{t−1}` by the pair screw (`tac.lie`, banked pose targets) before
   the delta; the churn that survives is the true irreducible sentence content. Same two-part
   pricing, same corpus, $0.
3. **The Lane tracked-word grammar (existence symbols over time).** 16,581 Lane words at 27.64
   per frame are mostly the SAME dashes persisting; §5 shows no per-frame chart is affordable.
   Code births/deaths/motion of tracked components; `ANNIHILATE` becomes a detectable missing
   symbol at encode time (§2.4), which is the erasure-immunity property the class needs.
4. **`cg1r` integration (live consumer).** Columns from `gt2_verbs.json` as specified in §6;
   `AMPLITUDE` sourced from `hg1`, `PHASE` from `pc2`; `FRAGMENT` signed as consolidate-negative.

What a resumer must NOT redo: the corpus identity check (bit-exact, §1.0), `dd1`'s census (twice
reproduced now), the estimate-vs-coder gap (1.6207×, measured), or any ε>0 polygon operating
point (all dominated at W).
