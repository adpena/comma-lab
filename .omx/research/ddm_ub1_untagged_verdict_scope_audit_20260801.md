# ddm_ub1 — untagged-verdict scope audit (task #846)

`date_utc: 2026-08-01` · axis `[macOS-CPU advisory]` · `score_claim: false` ·
`promotion_eligible: false` · **$0, scorer-free, 0 scorer forwards, 0 launches.**
**Pointer 0.1910828242 `[contest-CPU]` UNMOVED.** This audit moved no number and does not claim to.

Operator #846: *"audit all of those negative results, especially anything simple and binary in black
and white because the implementations were likely not optimal… naive and toy."*

---

## 0. THE HEADLINE — the corpus's scope-grading is good; its scope **transit** is not

**MEASURED.** The three-day audit (ba29/ba30/ba31) covered **2026-07-29 → 07-31 only**. The 07-19 →
07-28 window is **not covered by it** — 664 documents, 71 carrying verdict tokens, 226 token lines.
That is where I went.

What I found there is not a corpus that fails to grade its negatives. `codex_findings_ddm_a2_…_20260724`
grades **19 rows** with an evidence class, an explicit `[naive-*]` token, a named resolving measurement,
and a proper status ladder (`STANDS` / `STANDS-AS-BOUND` / `RE-SCOPED` / `SUSPEND-PENDING-x`). That is
better than what this audit was commissioned to produce.

**The defect is one layer up: a correctly-scoped verdict loses its scope as it is cited forward.**
Two independent instances, both source-verified, both currently load-bearing:

| # | claim | scope where MINTED | scope where CITED NOW | de-scope |
|---|---|---|---|---|
| **UB1-A** | post-hoc stored corrections are dead | `pose`-value storage · *without a compact code-to-photometry inverse* · *on this witness vehicle* (CLAUDE.md: **formulation**, 5 formulations) | `ddm_gc14…20260731.md:332` — **"RULED OUT by law"**, unqualified, applied to a **seg/phase** carrier on the **TR1** vehicle | formulation→**law** · pose→**seg** · witness→**TR1** |
| **UB1-B** | plane storage is rate-dead | `a2` **RE-SCOPED** it to `FORMULATION:EXACT_REVERSIBLE_L3_RASTER_RESIDUAL_RATE_DEAD`, *"cannot close rehomed, layer/five-type, generative, or score-quotient representations"* | registry line 750 anchor `verdict_scope: "family (exact-plane storage under ANY lossless entropy stage)"`, `VERIFIED_VIA_EMPIRICAL_ANCHOR`; MAIN 08-01 roadmap: *"Plane-storage family measured RATE-DEAD"* | formulation→**family** · storage→**family** |

**NEW LAW (proposed):** *a verdict's scope is not a property of the verdict, it is a property of the
citation.* A re-scope that does not reach the **registry row and the roadmap sentence** has not landed.
This is the exact sibling of ba31's *"an arithmetic composition inherits the weakest premise of its
inputs"* — here, **a citation inherits the strongest wording of its ancestors** unless something stops it.
Sister law, already in the corpus: cn3's *"a gate's LIVE-COUNT-0 is meaningless until its DENOMINATOR is
asserted."*

---

## 1. UB1-A — "post-hoc store-apply RULED OUT by law" · **INSTANCE→FORMULATION, mis-applied**

**MEASURED, exhaustive over the named scope** (recursive over the repo, `*.md` + `*.py` + `*.jsonl`):
the identifier `post_hoc_stored_corrections_dead_joint_descent_required_law_20260718` occurs in exactly
**two** files — `tools/measure_ddm_pa1_posenet_amplitude_twin.py` and
`.omx/research/ddm_gc14_first_descent_20260731.md`. **It is NOT in the canonical equations registry**
(0 matches over 864 rows). A claim cited as *"law"* that the equations leg cannot arbitrate is a
triality break, not a law.

The only code reference states the scope verbatim (`:1317-1325`):

```
"prior_application": "post-hoc pose-value storage without a compact
                      code-to-photometry inverse is dead on this witness vehicle"
```

Three qualifiers — **pose-value** · **absent an inverse** · **this witness vehicle**. `gc14:332` applies
it, stripped of all three, to kill **#425's STORE leg** (dash δ(s) phase coherence — a **seg** carrier,
on **TR1**). The pa1 tool itself models the correct handling: it explicitly separates its own mechanism
from the prior (*"it does not store target Pose6 values or claim an inverse"*). **The corpus contains
both the right practice and the wrong transit, eight days apart.**

**GRADE: FORMULATION** (5 formulations, pose, witness vehicle) — **not a law, and not binding on seg
carriers or on TR1.**

**Relative significance.** #425 STORE leg costs 37,158 B = **0.024742 S** of rate. W1-COH's reach on the
same axis is `≤0.00167` d_seg (all-flicker) / `0.00037` (deep tail).

| if the reach were realized | net S | % of gap (0.7918468) | % of inventory (0.097465) |
|---|---:|---:|---:|
| all-flicker `0.00167` | **+0.14226** | **17.97%** | **146.0%** |
| deep tail `0.00037` | +0.01226 | 1.55% | 12.6% |

The band **straddles the adopted Contrarian SKIP bound (0.05 S)** — 0.0123 is below it, 0.1423 is 2.8×
above it. **That is exactly why the $0 preflight is decisive: it collapses a band that currently spans
the decision threshold.** Note also that gc14's own dominance argument is *"negative-expectation against
a **free alternative**"* — and gc14 §13 declares that free alternative (T·continuation) **DRAINED**
(0.00946 S, SKIP), with MAIN 08-01 adding that window_03 has reversed and window_04 must not run.
**A dominance verdict is a relative claim; when the dominating alternative is retired the verdict does
not survive by inertia.**

**NAMED REACTIVATION MEASUREMENT.** The `[REQUIRED BEFORE BUILD]` **Fisher-weighted Jacobian spectrum**
for #535 — **$0**, from existing W1-COH receipts, and **gc14 itself already says it "should be queued
regardless of branch."** It is the cheapest conversion of a never-fired design into a priced one.
Second, free, and independent: **register the post-hoc law in the equations registry with its three
qualifiers**, which mechanically prevents the seg/TR1 mis-application.

---

## 2. UB1-B — "plane-storage family RATE-DEAD" · **the object is SOUND; the NOUN over-reaches**

**HONEST NON-REACTIVATION on the tested object.** DERIVED here from the registry's own fields, not
recalled: n600 realized endpoint **409,526,925 B** against a box of **264,320 B** = **1,549× over**.
Cross-check: `25 · 409,526,925 / 37,545,489 = 272.69` vs the recorded `S = 272.73` ✓. The largest coder
gain ever measured in this campaign is **1.381×** (ba29 Surface C, 19 coordinates). **No entropy stage
closes 1,549×.** Do **not** reactivate exact reversible plane storage. The `n24/n48-extrapolated` flag on
the codec sweep and the stale `S<0.19108` box reference are both real, and both **immaterial** — the
current bar (0.172141) is *tighter*, and a 1,549× margin is immune to a 1.381× instrument.

**The defect is the noun, and it is not cosmetic.** a2 named precisely what is *not* closed: **rehomed /
layer-typed / generative / score-quotient** plane representations. Those are not *storage* — they are
lossy and derived, cheaper by construction by orders of magnitude, and they are is1's **#1 and #2 ranked
prospective families**. A 1,549× margin on storing a thing says nothing about generating it. Yet:

- registry line 750: `verdict_scope: "family (exact-plane storage under ANY lossless entropy stage)"`,
  `VERIFIED_VIA_EMPIRICAL_ANCHOR`, consumed by three named consumers;
- MAIN's 08-01 roadmap (`:61`): *"Plane-storage family measured RATE-DEAD"* — untagged, family-worded;
- `generator_description_crux_synthesis_20260719.md:7`: *"plane-storage is RATE-DEAD **as a family**"* —
  the pre-re-scope ancestor, still on disk and still the wording that propagated.

**GRADE: FORMULATION** — sound, un-reactivatable on its object, **mis-scoped by one noun**, and that noun
sits directly upstream of the top-ranked open representation family.

**NAMED REACTIVATION MEASUREMENT.** None for the tested object — it is correctly dead. What is owed is a
**scope correction, not an experiment**: amend the registry anchor's `verdict_scope` to a2's surviving
token and record the four named non-closed representation classes. The a2-named family-scope resolvers
(`#669c` exact rehoming/layer/type rate + `DC1` receiver-closed fit) remain the measurements that would
license any *family*-level wording.

---

## 3. UB1-C — gr1 verdict 3, "QA07 nested-rung DOMINATED at BOTH granularities" · **INSTANCE**

`ddm_gr1_granularity_rerace_20260730.md` is otherwise a **model artifact**: it tags verdicts 1 and 2
`INSTANCE/FORMULATION` explicitly, carries its pose caveat forward, and says of its own headline *"this
refines the knee, it does not open a new mechanism."* **Verdict 3 is the one that carries no scope tag**,
and it is the most binary sentence in the document: *"No middle ground pays."*

**The confound is at source, and it is the arm's own finding turned against itself.** In
`experiments/ddm_gr1_granularity_rerace.py`:

- `:217` `order = np.argsort(g_abs.reshape(-1))` and `:238` `corder = np.argsort(cell_sens)` — **one
  `|g|` ordering**, and both the drop family (`:247`) and the rung family (`:253`) index into it.
- The rung candidates are **three hand-picked fraction triples** (`(0.35,0.55,0.75)`, `(0.50,0.70,0.85)`,
  `(0.633,0.80,0.92)`), not an allocation.

But gr1's **own verdict 2(a)** is: *"first-order |g| is a poor proxy for the finite drop-to-base flip
cost (zero-gradient tokens flip pixels when dropped)."* **The document measures the ordering to be a poor
proxy, then uses that same proxy to allocate the rung arm, then declares the rung family dominated.**
A rung ladder's optimum is the **lower convex hull of each cell's 4-point RD curve**, which is a
different ordering from a drop ranking by construction — the rung arm was never given its own optimum.
This is the *"implementations were likely not optimal… naive and toy"* signature the operator named.

**GRADE: INSTANCE** — three guessed fraction triples under a self-declared poor proxy. Not a family kill,
and **not** the structural statement *"no middle ground pays."*

**Relative significance.** At near-matched bytes the published rows are `cell_rung_a` 354,946 B @ d_seg
0.004681 vs `cell_drop50` 359,221 B @ 0.003947. The rung arm was **4,275 B cheaper** (0.002847 S of rate)
and **0.000734 d_seg worse** (**0.0734 S** of seg) — it paid **25.8× more seg than the rate it bought**.
That misallocation alone is **9.27% of the gap** and **75.3% of total known inventory**. Whether an
optimally-allocated rung beats `cell_drop50` is **UNMEASURED**; I do not claim it does.

**NAMED REACTIVATION MEASUREMENT.** Two, both cheap, using the existing harness:
1. **Convexity test ($0 on bytes, no scorer):** for a sample of cells, re-encode the 4 rung points
   `{L16, L8, L4, base}` through the real SMEVR coder and test whether any cell's `L8` or `L4` point
   lies **below its `L16`↔`base` chord**. If none does for any cell, verdict 3 **promotes to FAMILY**
   for this alphabet — a genuine strengthening. If some do, it is refuted and a hull allocation is owed.
2. **Budget-matched rung (one n48 eval):** build the rung candidate at *exactly* `cell_drop50`'s
   359,221 B under a hull-based allocation and evaluate once. gr1's comparison was neither
   budget-matched nor hull-allocated.

---

## 4. HONEST NON-REACTIVATIONS — negatives that deserve to stand (the control group)

A sweep that re-opens everything is as useless as one that re-opens nothing. These I attacked and
**failed to break**:

| row | why it stands |
|---|---|
| **is1 `training_necessary_residual = EMPTY_ON_CURRENT_EVIDENCE`** | The *strongest* row I read. It ships its own scope welded on: *"an evidence claim, not a universal mathematical claim"* and *"does not denote an empty trainable representation family."* It is a fail-closed proof-state over 9 named predicates, and a2 graded it `STANDS`. **This is the model of how a negative should be written.** |
| **A2-08 on its tested object** | 1,549× over box at n600 (§2). Un-closable by any entropy stage. |
| **gr1 verdicts 1, 2, 4** | Explicitly tagged `INSTANCE/FORMULATION`, strict domination with large margins, pose caveat travels, n600 confirm on the winner. |
| **SEG-CONTINUATION "CLOSED"** | Already audited 2026-08-01 (`ddm_bs1_margin_density_preflight_20260801.md`); survives. Not re-examined here. |
| **pj1 `f`/fp1 `f′` capacity floors** | Already correctly graded by ba30/MAIN as majority-class collapse (`f = 0.50482448154026` = the constant-Undrivable predictor, abs diff `0.00e+00` on all five classes) with **#833** minted as a two-landing and the scope explicitly held (`BR-D` unaffected; fp1's receiver floor `0.008305` is on GT argmax and collapse-immune). **Confirmed, not redone.** |
| **rg-ladder 25-row residue / class-birth 0/10** | is1 Directive 5 already reframed it as *"the residue is the DEMAND, not debris"* and registered the interface-placement law. Correctly scoped at mint. |

**The encouraging pattern ba31 noted holds here too:** in every case where a negative was well-scoped,
the scoping was done by the arm on *its own* headline.

---

## 5. What I did NOT reach

Stated plainly, because coverage claims are the dominant false-claim class:

- **~215 of the 226 verdict-token lines in the 07-19 → 07-28 window were not individually adjudicated.**
  I graded 3 rows to depth and confirmed 6 more, against a denominator of 71 documents. Depth was chosen
  over coverage per the task; this is the cost.
- **`codex_findings_ddm_ra1_…20260724` (32 untagged lines, the densest single document)** was read only
  at its header. It is *itself* a blocker-dissolution re-grade with a typed verdict
  (`FIVE_REACTIVATE_NOW_NINE_EVENT_GATED_FOUR_STILL_BLOCKED_THIRTY_SUPERSEDED`), so it is likely
  already-graded — **but I did not verify that, and it is the highest-value single next target.**
- `SPEC_v10_capstone_RECONCILED_20260719` (6), `optimal_start_card_366_refoundation_20260725` (5),
  `ddm_rv1_conditional_validity_regrade_20260728` (4) — enumerated, not opened.
- I did **not** re-audit the 07-29 → 07-31 window (ba29/30/31's denominator) and make no claim about it.
- The **transit-decay scan was run against two claims**, not against the corpus. I do **not** claim these
  are the only two — I claim I found two by following the named suspects, and that the mechanism
  generalizes. **A full transit-decay census over all registry anchors is the obvious next unit** and is
  $0.
- No negative-existence claim here is unbounded. Where I state absence — the post-hoc law is not in the
  registry; the FORMULATION token does not appear in MAIN's roadmap — the search was exhaustive over the
  stated scope and the scope is named inline.

## 6. Triality

- **DAG:** this file; FEED block owed by MAIN on consumption.
- **equations:** two registry actions proposed, **neither executed here** (this arm has no registry
  authority): (i) amend anchor `exact_plane_storage_rate_dead_family_20260719`'s `verdict_scope` to a2's
  surviving token; (ii) register `post_hoc_stored_corrections_dead_joint_descent_required_law_20260718`
  **with its three qualifiers**, or stop citing it as *"law."*
- **DSL:** none. No lever, no flag, no config.
- **tasks:** #846. No new arm spawned. #833 confirmed, not duplicated.
