# ddm_cg1 — actuator granularity + the per-class reach of the free margin key (task #809)

**Axis:** `[macOS-CPU scorer-free advisory]` · `score_claim=false` · `promotion_eligible=false` ·
`rank_or_kill_eligible=false` · **scorer forwards run: 0**
**Own-vehicle frontier: S = 0.7910689 @ 353,805 B `[macOS-CPU advisory]` — UNMOVED by this unit.**

---

## 1. Answer first

Two measured findings and one schema hole closed.

**(a) Ranking classes by margin-key ENRICHMENT sends bytes where they buy the FEWEST flips.**
Spearman(enrichment, precision) = **−0.900, p = 0.037**; ordering stable at all five global budgets.
Enrichment picks MyCar (precision 0.2084); precision picks Lane (0.6852). **3.29× fewer flips per
site spent.** Eighth confirmation of the standing law, in a coordinate nobody had checked.

**(b) Lane is the CHEAPEST class in which to buy a flip, not the hardest.** Lane's precision 0.6852
is the highest of five at every budget. What is hard about Lane is *choosing which* Lane pixels
(enrichment 2.5×, worst of five) — not affording them.

**(c) `actuator_granularity` now exists as its own axis** over `ddm_fl2`'s ledger, and on its first
run it flagged an apparent counterexample to its own law. Adjudicating that anomaly produced the most
useful artifact of the unit: **the canonical template for a per-edge force that works.**

---

## 2. The structural fact (verified at source)

`upstream/modules.py:112-113`:

```python
diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float()
return diff.mean(dim=tuple(range(1, diff.ndim)))
```

A uniform per-pixel mean — **no per-class term, no per-verb term**. It weights classes purely by
AREA. Measured consequence: per-class base flip rates span **502×** (Lane 0.269028 → MyCar 0.000536),
and Undrivable at 49.52% of the plane contributes 14.69% of flips while Lane at 0.59% contributes
36.53%. **Any population mean over this mixture is a large-area statistic and is silent about Lane.**

---

## 3. What I measured

`experiments/ddm_cg1_perclass_margin_reach.py` → `.omx/research/ddm_cg1_perclass_margin_reach_n600.json`

`ddm_mg1` measured the frozen head's GT self-margin as a per-site key at 98.1× enrichment / 42.51%
precision — **population** numbers. Per `m88`, I conditioned that key on GT class.

**Controls, both exact.** Flip mask reproduces the evaluator seg leg `d_seg = 0.004311794704861111`
at `rel_err = 0.0`. `gt_margin_negative_sites = 0` (confirms GT *self*-margin, an encode-side prior).
Cross-instrument: my population bin reproduces mg1 on a different pipeline (42.5138% vs 42.51%,
98.6× vs 98.1×).

| class | area % | share of flips | base flip rate | precision `[0,0.096)` | enrichment |
|---|---:|---:|---:|---:|---:|
| Road | 23.23% | 30.13% | 0.005591 | 0.3289 | 58.8× |
| **Lane** | **0.59%** | **36.53%** | **0.269028** | **0.6852** | **2.5×** |
| Undrivable | 49.52% | 14.69% | 0.001279 | 0.4434 | 346.7× |
| Movable | 1.24% | 15.50% | 0.053983 | 0.5646 | 10.5× |
| MyCar | 25.43% | 3.16% | 0.000536 | 0.2084 | **389.0×** |

**Enrichment spans 156×. The population's 98.6× describes no class.** Enrichment is a ratio to a
class's *own* base rate — it reports how unusual the bin is *for that class*, not what a byte buys.

**Mechanism for Lane's 2.5×** (from `ddm_dd1`, not re-derived): Lane's area/perimeter is **1.407 px**
vs Movable 11.07, Road 21.93, MyCar 97.41, Undrivable 176.20. A class with no confident interior gives
a margin key nothing to contrast against. Lane isn't a rare-fragile-site problem — it is *uniformly
broken* (26.9% of every Lane pixel flips).

**Reach ≠ discrimination — do not collapse these.** At a 1,000,000-site global budget, recall is near
parity (Road 69.6%, Lane 60.1%, Undrivable 67.8%, Movable 53.5%, MyCar 82.1%). The key **addresses**
Lane fine — Lane's sites are low-margin in absolute terms so they rank high globally — while failing
to **discriminate** within it. Two dimensions, opposite signs.

---

## 4. The schema hole, and what closing it found

`src/tac/force_actuator_granularity.py` — an **annotation layer over `fl2`'s `LEDGER`**, not a
parallel ledger. 48/48 forces classified, 0 unclassified.

**Why the axis was needed.** `fl2`'s own governing law is *"asymmetry is a PRIOR on a per-site
actuator, never the actuator"* — but `fl2` encodes granularity as a sentinel **inside the verb axis**
(`verb == "AGGREGATE"`). That works only for forces with no verb. It cannot express the case the law
is actually about: `as1.grow_lane.harms` has `verb="TRANSFER"` and a measured **+0.2459 S harm**, and
being an aggregate actuator is *precisely why it failed*. `fl2` records the harm but not the reason.
Splitting granularity out frees `verb` to mean only "which production" and makes the prediction
queryable.

**The law's scoreboard, now computable: of 11 measured aggregate-actuator rows, 0 improved.**
9 rows remain standing predictions. Falsifier: land one aggregate actuator with a realized-through-R
improvement and the law breaks.

### The adjudicated near-counterexample — the unit's most useful output

First run flagged `pc2.tie_calibration` as an aggregate actuator that IMPROVES — the only measured,
realized-through-R improvement among all per-scope forces (**dS −0.046 at ~0 B**). I had classified
`per_edge_tie_calibration` as `PER_SIDE_AGGREGATE` **on the strength of its name**. Reading `ru1`'s
evidence inverted it:

> *"the **per-atlas** version: 11.9% of flips, dS −0.046 at ~0 B, **+yield in 17/18 cells** … required
> RGB direction differs **per edge**: edge-resolve the existing sign rule."*

**The actuator is per-atlas-cell; the EDGE only supplies the SIGN.** Edge = prior, cell = actuator. It
*confirms* the law rather than breaking it — and it is the **template every other per-edge force should
copy**: −0.046 S at ~0 bytes is **61% of the entire Road↔Undrivable edge** (0.075909 S).

The sharpest live consequence: `road_undriv_bulk_field` is a **`BUILT_UNFIRED` per-side signed scalar
on the very same edge** where the per-cell version already works. It is predicted dead in its current
construction; re-specced per-cell it inherits a template with a realized receipt.

---

## 5. What I refuted — including three things in my own charter and one of my own builds

1. **My charter said `ddm_rt2` refuted `ddm_lg1`'s clause 1. It does not.** `rt2` adjudicates
   `#888`/`er1`/`rz1` and *cites* `lg1`'s `derive_margin_floor` as the **corroborating** source.
2. **My charter framed the cure as "hinge weight, not floor," as if the floor were mis-tuned.** `mg1`
   measured it stronger: the floor **cannot be the lever at any value** — flip coverage is already
   100% at the shipped 0.1. The live lever is the **weight**, `w* ∈ [0.50, 0.80]` = 10–16× shipped.
3. **My charter said the per-FRAME axis was unmeasured. It is measured and CLOSED** —
   `ddm_dd1_edge_frame_concentration_n600.json`, and `fl2` records it refuted 3× (top-10 frames ≤15.4%
   of TRANSFER, all 600 frames touched on every big edge). Not duplicated.
4. **I retracted a result of my own mid-flight.** I fitted `err_rate ∝ (area/perimeter)^−1.29`
   (r² 0.9095) and briefly read it as subsuming `pc2`'s `area^−1.22` (r² 0.8896). It does not: at n=5 a
   1.22× residual cut is not decisive, and **MyCar stays anomalous under both** (0.273× / 0.398×).
   `pc2`'s MyCar finding stands; mine does not explain it. Dropped from the deliverable.
5. **I deleted my own first build.** I shipped a `force_verb_ledger` module with verb + protection +
   granularity axes, then a harvest of `fl2` (commit `c4bc5ee042`, **69 rows** — not the ~35 I was
   briefed) showed `fl2` **already owns `verb`** (richer: ERODE/GOUGE split, TRANSFER, ANNIHILATE,
   FRAGMENT) **and `protection`** (5 values incl. `BUILT_UNFIRED` that mine lacked). I owned **one of
   three** claimed axes. Shipping it would have been the parallel-registry anti-pattern, so the module,
   its JSONL, and its 24 tests were removed and rebuilt as the annotation layer above.

---

## 6. Rows handed to `fl2`, not forked

`.omx/research/ddm_cg1_proposed_fl2_rows_20260804.jsonl` — 4 rows in `fl2`'s exact 15-field schema,
**validated against its real `ForceLedgerRow` contract, 0 `row_id` collisions**: the enrichment
inversion (HARMS, `ABSENT` protection), Lane no-discrimination (NEUTRAL), Lane cheapest-flips
(IMPROVES), and the structural 502× base-rate spread. They are proposals for `fl2` to fold; I did not
edit another arm's module.

**Gaps in `fl2` I found but did not fix** (owner's call): `magnitude_kind` — its headline addition —
has **zero test coverage**; the `FRAGMENT` split/merge sign exists only in prose, not as a field; the
provenance rung is free text inside `protection_ref`, not queryable; and 8 of 9 edge rows carry
`verb="DISPLACE"` as an explicitly-labelled placeholder for an UNMEASURED verb split.

---

## 7. The decisive next measurement

**Re-run the `#766` waterfill with the key swapped from enrichment-like to precision-ranked, and
measure matched-byte damage.** `mg1` already built the harness (it measured the barrier re-rank at
0.0000% advantage with a shuffled-key control proving the rig has power). This is a key substitution
in a working, controlled instrument — **$0 and scorer-free**.

Pre-registered falsifier: **< 1% matched-byte damage advantage** ⇒ the −0.900 inversion is real but
already priced by the current allocator, and the harm row downgrades `HARMS → NEUTRAL` at
`FORMULATION` scope. I did not run it: the scorer slot is owned by `fz1`, and the waterfill and its
prices belong to `wf2`.

Second, $0: **per-site depth as the Lane key.** `as1` derived *depth-weight, never margin-weight* for
Lane; `cg1` measured *why* margin fails there (no interior). These agree and are **not** in tension —
margin is the right **site selector**, depth the right **force weighting once a site is selected**.
Flagged explicitly because collapsing them would later read as a contradiction.

---

## 8. Wire-in (Catalog #125)

1. **Sensitivity map** — ACTIVE: per-class precision is a byte→flip sensitivity for any allocator.
2. **Pareto constraint** — ACTIVE: `as1`'s 55.27% symmetric-jitter share is a hard ceiling — no
   directed force can address undirected jitter, so the whole directed-force programme competes for at
   most ~44.7% of the seg residual.
3. **Bit allocator** — ACTIVE: *rank by precision, never enrichment* (§7).
4. **Cathedral autopilot** — N/A: advisory, no archive-deployable artifact.
5. **Continual-learning posterior** — ACTIVE: `predicted_dead()` + `coverage()` are queryable; both
   report denominators so vacuity cannot read as pass (`m50`).
6. **Probe disambiguator** — ACTIVE: `predicted_dead()` is a standing falsifiable prediction over
   every future per-side actuator; 9 rows live.

`council_predicted_mission_contribution: frontier_breaking` — it changes the allocation key for the
seg leg, **64.9% of the remaining gap**, unmoved across eight frontier revisions.

**STORES CONSULTED:** `upstream/modules.py` (source) · `fl2` `force_class_edge_ledger` (imported and
executed) · `ddm_pu2` directed-flip receipt · `ddm_mg1` (both JSONs, read directly) · `ddm_dd1`
(component census, contour coherence, edge-frame concentration) · `ddm_hg1` (signed depth profile) ·
`ddm_as1`, `ddm_sx2`, `ddm_rt2`, `ddm_lg1`, `ddm_ru1` (relayed, labelled in-row).

**Honest limits.** Relayed numbers were **not re-derived by `cg1`** and carry their source path. My
probe is n600 with both controls exact, but it is `[macOS-CPU advisory]` and moves no pointer. Per
`bz1`'s mirage law, nothing here is priced through the legal deterministic receiver, so **no row in
this unit is bankable as a score**. Granularity classifications record HOW a force actuates (readable
from construction), never WHETHER it works — and the `per_edge_tie_calibration` episode shows a
name-based classification can be wrong; each is a judgment open to the same adjudication.

---

# APPENDIX A — the pre-registered measurement, FIRED (2026-08-04)

`experiments/ddm_cg1_precision_vs_enrichment_waterfill_key.py` →
`.omx/research/ddm_cg1_waterfill_key_swap_n600.json`

Turf resolved (`wf2` not live; scorer slot remains `fz1`'s and this touches **0 scorer forwards**),
so §7's specified measurement was run rather than left as a named-$0-never-run row.

## A.1 Verdict: FALSIFIER FIRED — the harm row downgrades, as pre-registered

| bytes freed | shipped `flip_count` damage | precision adv. | **enrichment adv.** | shuffled control |
|---:|---:|---:|---:|---:|
| 10% | 0 | — | — | — |
| 30% | 0 | — | — | — |
| 50% | 246 | **+0.000000%** | −4.88% | −120,345.5% |
| 70% | 23,414 | **+0.000000%** | **−126.04%** | −1,404.8% |
| 90% | 232,467 | **+0.000000%** | −40.73% | −99.0% |

*(positive = less damage admitted = better. The 10%/30% budgets admit zero damage under the shipped
key, so no key can separate there; the verdict uses only the three budgets with discriminating power.)*

**Precision-ranking gives 0.000000% advantage and a BIT-IDENTICAL drop order.** The reason is
structural and I had not seen it until I read the harness: **all 768 `wr1` cells hold an identical
site count** (16×16×600), so flips-per-site is `flip_count` divided by a constant. **The shipped `#766`
key already IS the precision key.** The class-level inversion cannot reach this allocator. Per the
pre-registration (`<1%` ⇒ downgrade), `cg1.enrichment_key.inverts` moves **HARMS → NEUTRAL at
FORMULATION scope**. Honored without renegotiation.

## A.2 But the harm is real where it *can* reach — so the row becomes a guard, not a corpse

The pre-registration anticipated only "real but already priced." The run measured something it did
not anticipate: the **true enrichment analogue** — flips normalized by the flips *expected* from each
cell's own GT class composition at population base rates — is up to **−126.04% worse**, more than
doubling admitted damage at the 70% budget. Instrument power is not in doubt (shuffled control
−120,345%).

So the correct disposition is neither "confirmed" nor "dead" (the never-binary rule): the row converts
into a **STANDING GUARD with a measured price**. `#766` is protected *by construction* — equal-site
cells make normalization impossible. The guard binds any **future** allocator whose units differ in
site count (per-component, per-class, or variable-grain carriers), where normalizing by a per-class
base rate costs up to **126% more damage at matched bytes**. `protection: ABSENT → BUILT`.

## A.3 Composition with `sg3` (commits `e7a15a6577` / `cf20d4efa1`)

`sg3` measured Lane error as boundary **DUST** for explicit addressing: 112,077 components, median
**1 px** — address floors kill explicit Lane masks. `cg1` measured Lane as having the **highest
per-site precision of any class** (0.6852) for a margin-keyed selector.

**These compose; they do not conflict.** They are statements about two different costs:

- *Addressing* Lane is unaffordable — 112,077 median-1px components cannot carry per-component addresses.
- *Fixing* a Lane site is the cheapest of any class once you are already looking at it.

The form that satisfies both is therefore forced: **a decoder-derived key (free context, zero address
bytes) × precision ranking.** A.1 supplies the missing half of that statement — **the precision-ranking
half is already shipped and already optimal at `#766`'s grain.** So the open work is entirely the
*other* half.

And that sharpens the next rung to one question. The key `cg1` measured is the **GT self-margin**, which
is encode-side only — a decoder cannot compute it. The live question is therefore:

> **Is there a decoder-derivable proxy for the GT margin — computable at decode from the render the
> decoder already holds, costing zero address bytes — that preserves the per-site ranking?**

That is $0 and scorer-free to test (correlate a render-derived margin against the cached GT margin on
the same n600 sites), and it is the single measurement that would turn the Lane result from a
diagnosis into a carrier. It is **not** claimed here and is left as the named next rung with its
fire-condition, not as a promise.

## A.4 Honest limits on the appendix

Drop rule imported verbatim from `ddm_mg1_barrier_rerank_probe` so the comparison is apples-to-apples
with the barrier result. Control: `d_seg` rel_err **0.0**, tiling covers the plane exactly once. The
`class_sites @ base_rate` matmul emits divide-by-zero/overflow/invalid `RuntimeWarning`s on this BLAS
**even for finite inputs** — reproduced on clean random data, so spurious (stale FP flags leaking into
its error check). Rather than silence it, finiteness of both the inputs and the result is **asserted
and recorded** in the receipt, so a real fault cannot later hide behind a warning we learned to ignore.
`[macOS-CPU advisory]`, **0 scorer forwards**, no pointer moved, nothing bankable as a score.
