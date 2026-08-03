# ddm_hg1 — the negatives as a shape: separatrix asymmetry is EXTENT, not density; and the holonomy is identically zero

**Arm:** `ddm_hg1` · **Date:** 2026-08-03 · **Cost:** $0, scorer-free, cached artifacts only
**Axis:** `[macOS-numpy advisory · NON-PROMOTABLE]` — **pointer `0.1910828242` UNMOVED.**
**Operator directive:** *"These negative signals might point to a more optimal representation or
topology or homography or holonomy or geometry or other deep math."* + *"We also have past research
regarding the asymmetry of the depth on either side of the separatrix."*

---

## 0. Answer first

1. **Lead A's falsifier was NOT un-run.** It ran on **2026-07-08** and landed
   `experiments/results/t5_probe_waveB_20260708/q1_signed_asymmetry.json`, verdict
   `fire=false, robust_dead=false` — an **INDETERMINATE**. It has sat unread for 26 days. So the
   texture/UNIWARD family is neither reopened nor legitimately closed; it is in limbo, and the
   negatives review's own kill-scope shrink was never followed by a close.

2. **That artifact's own declared positive-control sentinel FAILED, and the sentinel was never
   implemented in code.** The tool's docstring binds: *"if sR is ALSO at chance vs flips the
   instrument is untrusted and no verdict is admissible."* Measured: `|ρ_sR| ≤ 0.081` on **every**
   major pair while `|ρ_margin| ≥ 0.24`. The verdict block was emitted anyway, because
   `positive_controls: ["sR","margin"]` is a **JSON metadata string that no line of the verdict
   computation reads**. I discriminated misalignment vs. genuine-at-chance and **misalignment is
   refuted** (§2) ⇒ **`sR` — the field DEPLOYED as the replacement for the retired texture proxy —
   is 3–4× worse than the plain top1−top2 margin at predicting which pixels actually flip.**

3. **The asymmetry is real, but it is EXTENT, not DENSITY — and this inverts the design
   consequence the addendum drew from it.** MEASURED, **n600, frozen-scorer GT, vehicle-independent**:

   | class | area px | **shell % (depth ≤ 1)** | **mean depth px** |
   |---|---:|---:|---:|
   | **Lane** | 690,639 | **75.04** | **1.134** |
   | Movable | 1,460,325 | 9.73 | 7.267 |
   | Road | 27,407,046 | 4.57 | 9.062 |
   | MyCar | 29,993,509 | 1.03 | 11.312 |
   | Undrivable | 58,413,281 | 0.57 | 11.607 |

   Lane is a **7.7× outlier on shell fraction and 6.4× on mean depth, with nothing in between** —
   a threshold, not a gradient. `Lane→Road` is the **only** directed side in the whole table whose
   depth support truncates (population < 1 % of its depth-1 count by depth 4, < 0.1 % by depth 5);
   all ten other sides run past depth 12. Its **barrier integral is 5.10 vs 33.26–58.39** for every
   other side — **6.5×–11.4× cheaper to annihilate**.
   **And yet `Lane→Road` has the STEEPEST margin recovery of all 14 sides (+1.52/px) and an
   above-median `margin_d1` (0.593).** The Lane side is the **most expensive per pixel** and the
   **cheapest to destroy**. The asymmetry is in the **support**, not the cost density.

4. **The addendum's asymmetry does NOT generalize past Lane.** Measured directed flip-rate ratios:
   `Road↔Lane` **5.85×**, but `Road↔Movable` **1.03×**, `Road↔Undrivable` **1.05×**,
   `Undrivable↔Movable` 1.55×, `Road↔MyCar` 1.56×. **The separatrix is symmetric to within 5 % on
   its two largest interior–interior interfaces.** Asymmetry is localized to the one class with no
   interior — it is not a property of the separatrix, it is a property of Lane.

5. **Lead B: the holonomy is identically zero, and calling `ddm_br1`'s 1.0206× a holonomy is a
   category error.** The source memo **self-contradicts** (line 798: non-zero holonomy ⇒
   anti-additivity — correct; line 802: *"zero IFF super-additive"* — the converse). More
   decisively: drops compose by **union of dropped unit sets** (`br1` §S3, same 20 units), so the
   operator algebra is a **commutative idempotent join-semilattice**. Every commutator is
   identically zero and every loop closes trivially ⇒ **the designed
   `holonomy(loop_in_base)` primitive would return 0 for every input regardless of the 2 %
   non-additivity — a NO-FAKE class-1 instrument (canonical markers, no work).** Its registered
   `ΔS ∈ [−0.010, −0.003]` is unbacked. §5 gives the formalism that *does* apply.

6. **On the operator's unification question, the honest answer is NO — there is not one fact.**
   The negatives split into two independent families with a dual relationship (§6). Claiming one
   fact would require me to assert that the annihilation geometry predicts `br1`'s token-lattice
   entropy results, and **it does not**.

7. **I registered a falsifier against my own §6 framing, ran it, and it FIRED.** Erasure flip count
   follows **one interface-length law across all five classes** (`CV = 0.4598` for perimeter vs
   `1.6603` for area; Lane's z-score under the perimeter model is **+0.96**, and Movable is a larger
   outlier than Lane). There is **no annihilation regime visible in flip counts**. What survives is
   sharper than what I claimed:
   > **`d_seg` is a flip count, and flip count is governed by interface length uniformly — so
   > `d_seg` prices Lane erasure and Road boundary-nudge identically per pixel, while their barriers
   > differ 6.5×–11.4×. The metric is blind to the property that determines how hard a flip is to
   > fix.** This is the mechanism under `sx1`'s *"description cheap, realization expensive."*
   Consequence: `br1` showed the **byte** axis carries no ranking information; §6 now shows the
   **flip** axis carries little either. **The barrier integral (5.10–58.39, an 11.4× spread) is a
   ranking key that is already measured here, vehicle-independent and free, and is consumed by
   nothing.**

---

## 1. Lead A verified at source, and what the artifact actually says

**VERIFIED_VIA_SOURCE_INSPECTION.** The falsifier is pre-registered in
`.omx/research/t5_crucible/negatives_scale_validity_review_20260707.md` §7 with an explicit band —
*"Kill band: |ρ| < 0.1 on BOTH sides of every major pair ⇒ upgrade to SCALE-ROBUST dead; any side
with |ρ| ≥ 0.3 ⇒ a one-sided UNIWARD cost term … enters the never-fired queue with a real prior."*
The mechanism claim is `ORCHESTRATION_LEDGER.md` requirement **L** (line 1273, ASYMMETRY ADDENDUM):
*"an unsigned pooled estimator of a signed density has zero expectation when the two sides carry
opposite signs"* and *"the boundary is one-sided per class-pair (Road→Lane FP ≠ Lane→Road erasure)."*

**The correction to my charter:** the instrument `tools/signed_flip_asymmetry_correlator.py` was
built (commit `6bd8ddd86e`) **and run**, emitting
`experiments/results/t5_probe_waveB_20260708/q1_signed_asymmetry.json` on 2026-07-08. This is **not**
the unwired-but-built class (`m56`); it is the **unread-result** class — worse, because the apparatus
paid for the measurement and then discarded the signal. The landed table (96 witness frames,
`MuonBest`, annulus radius 2, β = 4.0):

| side | n px | flips | rate | ρ texprox | ρ tex | **ρ sR** | ρ margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0→1 Road→Lane | 289,822 | 12,997 | 0.0448 | −0.0714 | +0.0588 | **+0.0031** | −0.2448 |
| 1→0 Lane→Road | 108,455 | 28,470 | **0.2625** | −0.1116 | +0.1346 | **+0.0273** | −0.3787 |
| 0→2 Road→Undriv | 91,847 | 5,631 | 0.0613 | +0.0139 | −0.0156 | **+0.0081** | −0.2930 |
| 2→0 Undriv→Road | 92,935 | 5,423 | 0.0584 | +0.0275 | −0.0186 | **+0.0219** | −0.2847 |
| 0→3 Road→Movable | 29,966 | 3,945 | 0.1316 | −0.0272 | +0.0207 | **+0.0063** | −0.3092 |
| 3→0 Movable→Road | 27,869 | 3,567 | 0.1280 | **+0.1147** | −0.1027 | **+0.0510** | −0.3076 |
| 2→3 Undriv→Movable | 33,154 | 3,492 | 0.1053 | −0.0421 | +0.0355 | **−0.0025** | −0.2983 |
| 3→2 Movable→Undriv | 27,657 | 4,526 | 0.1636 | **−0.1242** | +0.0899 | **−0.0808** | −0.3293 |
| 4→1 MyCar→Lane | 2,295 | 75 | 0.0327 | **−0.2198** | +0.2236 | −0.0042 | −0.2326 |

Neither band fired on `texprox`: max major |ρ| = 0.1242 — above the 0.1 kill floor, below the 0.3
fire floor. **INDETERMINATE is the correct verdict on the registered question.** The value in the
artifact is in the columns nobody looked at.

---

## 2. The instrument's own sentinel failed — and I discriminated the two explanations

The `ρ_sR` column is at chance on every major pair (max |ρ| = 0.0808, mean |ρ| = 0.0250) while
`ρ_margin` is a clean −0.24 to −0.38 with the correct sign everywhere. Per the tool's own docstring
this is the **untrusted-instrument** condition and **no verdict was admissible**. Two explanations
with opposite consequences:

- **H1** — `sR` genuinely does not predict flips.
- **H2** — `sR` is **misindexed**. The correlator asserts frame alignment for `lstars`
  (`raise RuntimeError(f"frame alignment broken at witness frame {w}…")`, line ~198) but has **no
  equivalent assert for `sR`**, which is indexed by a *different* composed stride
  `g_idx = 3 · (2 · w)`. A wrong stride looks exactly like "at chance."

**Discriminator (MEASURED, `scratchpad/hg1_probe1_sr_alignment_and_pooling.py`).** `sR` is built from
the margin geometry (fragility-weighted margin-Jacobian reachability), so the correctly-aligned
frame must be structurally closer to its own frame's margin field than to another frame's. Sweep the
candidate maps against a random-offset control, 12 deterministic probe frames:

| index map | mean ρ(sR[g], own-frame margin) | sd |
|---|---:|---:|
| **g = 6w (DEPLOYED)** | **−0.3661** | 0.0396 |
| g = 3w | −0.3102 | 0.0367 |
| g = 2w | −0.2876 | 0.0529 |
| g = w+1 | −0.2829 | 0.0293 |
| g = w | −0.2749 | 0.0333 |
| RANDOM control | −0.3020 | 0.0653 |

The deployed map is the unique winner and beats the random control by **2.9 σ**; every alternative
stride sits **at or below** the random control. **H2 REFUTED, H1 STANDS.**

> **Consequence, and it is load-bearing.** `negatives_scale_validity_review` §7 recorded
> *"The DEPLOYED disposition is UNCHANGED and correct: S_R (exact, θ-indep, signed by construction
> at the flip level) replaces the proxy."* **That disposition is refuted by the instrument built to
> test it.** `sR` has |ρ| ≤ 0.081 on the flip populations where the plain margin has |ρ| ≥ 0.24. The
> proxy was retired for being at chance; its replacement is **also** at chance, and additionally
> costs a 450 MB precomputed sidecar. The margin — free, already cached — dominates both.
> *Scope: `[witness-vehicle, 96 frames]`, point-biserial linear response. FORMULATION-level, per
> requirement R. It does not kill reachability-weighting as a family.*

**The class, not the instance.** A positive control that lives in a metadata list instead of in the
verdict expression is the **silent-guard** pattern (operating manual §8.9, CLAUDE.md *Confound
self-protection* L3) — and it is exactly `m50` **VACUITY == PASS** at one remove: the sentinel did
not fail open, it **was never wired to a gate at all**, so its failure emitted the same symbol as its
success. Sister surfaces to sweep: every probe JSON in `experiments/results/` carrying a
`positive_control` / `sentinel` key should be checked for whether any code reads it.
*(One counter-example already exists and should be the template:
`.omx/research/ddm_pz1_dseg_n600_cx1_20260803.json` carries `positive_control.passes: true` with
`"if it does not, no seg number in this file is admissible"` **and** a computed `ratio` — a control
that was actually evaluated.)*

---

## 3. Pooling has THREE pathologies, not one

The addendum named **one** mechanism (opposite signs cancel). Recomputing the directed and pooled
accumulators exactly from the caches (same 96 frames, same fields) shows **three**, and only the
first is the named one:

| source class | pooled ρ | max directed \|ρ\| | dilution factor | sign split? |
|---|---:|---:|---:|:--|
| MyCar | −0.0181 | 0.2198 | **12.17×** | **no** |
| Movable | −0.0340 | 0.1242 | 3.65× | yes |
| Road | −0.0439 | 0.0714 | 1.62× | yes |
| Lane | −0.1100 | 0.1658 | 1.51× | yes |
| Undrivable | −0.0407 | 0.0421 | 1.03× | yes |
| **POOLED ALL** | **−0.1500** | — | — | — |

1. **Sign cancellation** (the named one) — Movable: `3→0` = **+0.1147** and `3→2` = **−0.1242** on the
   same source class and the same texture field, with near-equal mass (27,869 vs 27,657 px) ⇒ pooled
   **−0.034**. The mechanism caught in the act, 3.65×.
2. **Mass dilution — NEW, and it needs no sign flip.** MyCar: `4→0` (n = 102,630, ρ = −0.012) swamps
   `4→1` (n = 2,295, ρ = **−0.220**). Both negative; the pooled value is **12.17× smaller** than the
   real effect purely because a large near-null population outvotes a small strong one. **This is
   strictly more general than sign cancellation** and the addendum does not cover it.
3. **Simpson reversal — NEW.** Pooling across source classes gives **−0.150**, *larger in magnitude
   than every single within-class value* (max 0.110). Between-class differences in texture *and*
   flip rate **manufacture** an aggregate correlation that exists within no population. The original
   `msal_uni` measurement (pooled Pearson −0.033) sits between the within-class values and this
   spurious aggregate — **whatever it measured, it was not a within-population effect.**

This is `m88` (*a prefix of a skewed population is a different population*) with a second face: **an
aggregate of heterogeneous populations is a different population too**, and it fails in three
distinct ways. `br1` independently walked into the prefix face of this same trap today (its
ROUND-1 CATCH #2 sign flip) — two arms, one day, one estimator class.

---

## 4. The depth profile: the asymmetry is EXTENT, not DENSITY

`scratchpad/hg1_probe2_signed_depth_profile.py`, **n600, frozen-scorer GT only** (`lstars` +
`margins` from `gt_n600.npz`) — **no witness, no reconstruction, no θ**, so unlike §1–3 this
transfers to cx1/TR1 without the borrowed-vehicle caveat.

**Construction.** Per frame: EDT to every class; `rival(p) = argmin_{c ≠ L(p)} EDT_c(p)`;
`depth(p) = EDT_rival(p)`. The directed population `i→j` is `{p : L(p) = i, rival(p) = j}` — the
pixels of class *i* whose nearest rival is *j*, i.e. the ones that would flip that way. Profile =
mean margin binned at integer depth. This is the normal derivative of the frozen head's potential on
each side of the separatrix — literally "the depth on either side."

**Mean margin at depth (n600, sides with ≥ 100 k px):**

| side | n px | d1 | d2 | d3 | d4 | d5 | d6 | **pop < 1 %** | **barrier ∫** |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|
| 2→0 Undriv→Road | 33,473,082 | 0.519 | 1.628 | 2.864 | 4.068 | 4.838 | 5.164 | never | 51.81 |
| 4→0 MyCar→Road | 29,540,458 | 0.486 | 1.524 | 2.674 | 3.726 | 4.716 | 5.625 | never | 55.29 |
| 2→3 Undriv→Movable | 24,884,141 | 0.350 | 0.957 | 1.517 | 2.161 | 2.793 | 3.246 | never | 37.91 |
| 0→1 Road→Lane | 16,355,233 | 0.739 | 2.022 | 2.984 | 3.680 | 4.223 | 4.606 | never | 51.26 |
| 0→4 Road→MyCar | 8,733,650 | 0.484 | 1.554 | 2.806 | 4.242 | 5.509 | 6.002 | never | 58.39 |
| 0→2 Road→Undriv | 1,406,205 | 0.495 | 1.414 | 2.013 | 2.373 | 2.687 | 2.934 | never | 33.26 |
| 0→3 Road→Movable | 911,958 | 0.391 | 1.091 | 1.750 | 2.473 | 3.297 | 4.116 | never | 49.51 |
| 3→0 Movable→Road | 773,696 | 0.390 | 1.104 | 1.773 | 2.510 | 3.403 | 4.239 | never | 55.04 |
| **1→0 Lane→Road** | **683,786** | **0.593** | **1.955** | **2.554** | 2.770 | 3.165 | 1.441 | **depth 4** | **5.10** |
| 3→2 Movable→Undriv | 675,421 | 0.338 | 0.964 | 1.561 | 2.314 | 3.164 | 3.954 | never | 51.31 |
| 4→1 MyCar→Lane | 440,302 | 0.761 | 2.098 | 3.234 | 4.160 | 4.912 | 5.477 | never | 58.17 |

Read the two right-hand columns together with the `d1` column and the finding is unambiguous:

- **`Lane→Road` is the only side that runs out of material.** Its population drops below 1 % of its
  depth-1 count at depth 4 and below 0.1 % at depth 5. **Every** other side is still populated past
  depth 12.
- **Its barrier integral is 5.10 against 33.26–58.39** — the total margin standing between the
  separatrix and complete annihilation of the class is **6.5×–11.4× smaller** than for any other
  side.
- **And it is not cheap per pixel.** `d2 − d1 = +1.362` on the n600 table and **+1.520** on the n24
  table — the **steepest recovery of all 14 sides** — with `margin_d1 = 0.593`, above the median of
  0.49. Per pixel, moving the Lane→Road separatrix is among the **most** expensive moves available.

> **This inverts the addendum's design consequence.** Requirement L drew: *"losses/gates/amplifiers
> on the boundary may be SIGNED (one-sided hinge per class-pair direction), not symmetric bands"* —
> the B16 lever. A signed hinge weights by asymmetry in **cost density**. Measured, the density
> asymmetry runs the **wrong way**: a density-based one-sided hinge would **de-weight** the Lane side
> because it looks expensive, when the Lane side is precisely where a bounded perturbation destroys
> the class outright. **The correct primitive is an extent/existence term, not a signed cost hinge.**
> B16 as specified should not be built in that form.

### The registered prediction P1

Registered in the probe **before** the answer was computed: the directed flip rate should be
rank-predicted by the source side's profile; **REFUTED if |rs| < 0.5 for all four** candidates.
Spearman over the 8 major directed sides:

| predictor | story | rs |
|---|---|---:|
| **shell_fraction** | **EXTENT — how much of the class is already one flip out** | **+0.8571** |
| mean_depth | EXTENT — how much there is to defend | −0.5238 |
| margin_d1 | **DENSITY — "the shallower side is cheaper"** | **−0.4048** |
| barrier_integral | density × extent | −0.2619 |

**P1 not refuted, and the ordering is the finding: both EXTENT predictors clear the band; the
DENSITY predictor is the weakest of the four.**

> **I am obliged to discount my own strongest number.** The 8 sides come from only **4 distinct
> source classes** (Road ×3, Undrivable ×2, Movable ×2, Lane ×1), so they are not independent. At
> the class level the correlation is rs = **+0.80 with n = 4, p ≈ 0.33 — NOT significant.**
> **The load-bearing evidence here is not the rank correlation. It is the outlier coincidence:** the
> one class that is 7.7× out on shell fraction is the same class that is 6.4× out on mean depth, the
> only one whose support truncates, the only one with a 6.5–11.4× smaller barrier, and the only one
> whose interface is asymmetric (5.85× vs 1.03–1.56×). Five independent geometric measures agree on
> one class out of five. That is the claim; the Spearman is decoration on it.

### Where this lands on the score

`ddm_pc2` measured (`m91`) **Road↔Lane = 49.2 % of flips = 22.1 % of the entire gap**, and its
binding instruction was *"decompose per EDGE, never per class."* Decomposing that edge one level
further, **per DIRECTED edge**, splits it **68.66 % / 31.34 %** (28,470 vs 12,997 flips) —
so `Lane→Road` alone would be **≈ 15.2 % of the entire gap**, the largest single directed component
identified so far.

> **CAVEAT THAT TRAVELS WITH THAT NUMBER.** The 22.1 % is `ddm_pc2` on **our** vehicle; the
> 68.66/31.34 split is the **witness** vehicle at 96 frames. Multiplying them mixes vehicles, so
> **15.2 % is a HYPOTHESIS, not a measurement** (operating manual §8.5, borrowed number). The
> measurement that closes it: run the directed decomposition against **cx1's own predicted argmax at
> n600**. I confirmed no cx1 per-pixel argmax is cached anywhere (`ddm_pz1`'s n600 seg artifact is
> per-pair only), so this needs one scorer pass — **`ddm_pu2` holds that slot.**

---

## 5. Lead B: the holonomy is identically zero, and it is a category error

**The memo self-contradicts, four lines apart** (`set_theory_manifolds_geometry_deep_research_synthesis_20260518.md`):

- line 798 — *"zero holonomy = flat connection. Empirical **NON-ZERO** HOLONOMY corresponds to
  genuine curvature → Catalog #322 anti-additivity."* ← mathematically correct.
- line 802 — *"`holonomy(loop_in_base)` — compute holonomy of a closed loop (**zero IFF** Catalog
  #322 **super-additive**)."* ← the converse. **This is the line my charter quoted.**

So the surface reading — *"we measured 1.0206× superadditive, therefore non-zero holonomy"* — is
built on the contradicted line. But the deeper problem is not the direction. **It is that neither
line is applicable.**

**DERIVED.** `ddm_br1` §S3 measured superadditivity by comparing 20 individual unit-drop marginals
(Σ = 4,857 B) against **the group drop of those same 20 units** (4,957 B). So composition of drops is
**union of the dropped unit sets**:

> `drop_A ∘ drop_B = drop_{A∪B} = drop_B ∘ drop_A`, and `drop_A ∘ drop_A = drop_A`.

That is a **commutative, idempotent monoid — a join-semilattice**. Two consequences, both fatal to
the bundle formalization:

1. **No inverses.** Parallel transport is by construction an invertible group action (transport along
   γ then γ⁻¹ is the identity). A drop is a projection: there is no un-drop. **Holonomy is not
   defined on a semilattice.**
2. **Every commutator is identically zero.** Curvature in this formalism is the commutator of
   transition operators, `R(X,Y) = [∇_X, ∇_Y]`; here the operators commute *and* are idempotent, so
   **every closed loop closes trivially and the holonomy of every loop is exactly the identity —
   zero — no matter how non-additive the value function is.**

> **Therefore the routable `holonomy(loop_in_base)` primitive, if built, would return 0 for every
> input, forever, while the underlying 2 % non-additivity is real.** That is precisely NO-FAKE
> forbidden class 1 (*returns-canonical-markers-without-doing-work*) — the function would be
> structurally incapable of doing the work its name claims. Its registered
> `Predicted ΔS unlock: [-0.010, -0.003]` is unbacked. **Do not build it. This finding is worth the
> arm on its own as an anti-build.**
>
> *Convergent prior:* `ADVISORY_sdf_evaluator_quotient_geometry_…_20260710.md:964` already reached
> "holonomy is identically zero" for a **different** construction (every edge derived from
> potentials). Two independent routes to the same conclusion; the credit for the conclusion is
> shared, the route via drop-idempotency and the application to the §6.2 primitive are this arm's.

**The formalism that DOES apply.** The non-additivity is not in the operator algebra (which is
trivially flat); it is in the **value function on the semilattice**. `br1` measured two properties of
that value function:

- **superadditive at 1.0206×** on a matched 20-unit set;
- **not monotone** — §S2, *"two live units have NEGATIVE marginals … Brotli is not monotone in the
  drop."*

A set function that is neither modular nor monotone is a **general set function on a lattice**. It is
not submodular, so greedy waterfill carries **no** approximation guarantee — which is the real,
actionable content of `br1` §S2's *"a greedy one-at-a-time waterfill will wrongly reject profitable
units,"* now with a name. The right library is **polymatroid / set-function optimization**, and the
right diagnostic is the **Čech / sheaf-gluing obstruction the very same memo names at §2.6**
(*"the Catalog #322 anti-additive evidence IS empirical detection of non-sheafy behavior"*) — an
obstruction-to-gluing on a cover is well-defined without inverses, which is exactly why it is the
correct home for this fact and the bundle is not.

> **The transferable lesson, and it is the one that generalizes past this instance.** Both §6.2
> primitives were specified from a *structural analogy* (composition ↔ transport) without checking
> whether the analogy's **algebraic preconditions** (invertibility, a group action, a loop) hold on
> our object. **Register a geometric formalism only with its preconditions checked against the actual
> operator algebra.** The same memo's §2.6 got it right and §6.2 got it wrong — in the same document,
> about the same empirical fact.

---

## 6. The negatives as a shape — and the honest answer is that there is not one fact

The charter asks whether a single representational fact would predict all of: basis-is-not-the-lever;
coding-closed; description-cheap-realization-expensive; pre-compensation-orthogonal; raster-debt-in-
exact-float; seg/pose decoupling; the 6-pair pose axis; flat byte yield; the widening wedge;
superadditive drops. **I could write that sentence. It would be unfalsifiable, and it would be
false.** The measured split:

**FAMILY 1 — the code is saturated.** `br1`: 264 lossless re-expressions land within **0.83 %** of
`IDENT`; the coder's advantage is **1.109×** on live symbols (not 1.483×); spatial predictors **raise**
entropy — the token lattice has **no spatial correlation**; byte yield is flat at ~211 B/unit,
**uncorrelated with activity**, two units negative. Every one of these is a statement about the
**token stream as a compressible object**, and together they say: *the stream is at its own entropy;
there is no exploitable structure left in it.*

**FAMILY 2 — the target is not reachable by refining the code.** `sx1`: the description costs
**0.5349 B/flip against W = 1.2731** (42 % of budget, 5.91 bits/flip left over) — seg is **not**
description-bound. `pz1`: pre-compensation **lost at n600** (ΔS_seg **+0.000394**, 49.3 % of pairs
worse) — what it removes is **orthogonal noise** w.r.t. the GT error. `ra1`: **93.5 %** of camera-raster
debt exists in **exact float** — resampling, not quantizing. And now §4: the binding directed side is
the **annihilation of a class with no interior**, not the displacement of a boundary.

**These are two faces of one boundary, not one fact.** Family 1 says the encoder cannot get smaller;
Family 2 says the decoder cannot get closer. **We are at the expressive limit of this representation
on both of its faces simultaneously** — which is the operator's hypothesis, arrived at from
measurement rather than assumed.

**What I will NOT claim.** The annihilation geometry does **not** predict `br1`'s entropy results.
Family 2 is about the seg target's shape; Family 1 is about the token stream's statistics. I looked
for a bridge — "aliasing decorrelates, so a sub-resolution target explains the whitened lattice" —
and it does not survive: the token lattice is a *learned* code over *our* tokens, not a sampling of
the GT field, so its whiteness is evidence about our encoder, not about the scorer's sampling. **The
bridge is unsupported and I am recording it as refuted rather than shipping it as insight.**

### What the geometry does constrain, stated so it can be attacked

**STRATIFY THE CARRIER BY CODIMENSION, NOT BY CLASS.** The n600 table is not a smooth spread of five
classes — it is **two strata with a 7.7× gap and nothing between**:

- **interior stratum** (Road, Undrivable, Movable, MyCar): mean depth **7.3–11.6 px**, shell
  **≤ 9.7 %**, never truncates, barrier **33–58**, interfaces **symmetric to within 5 %**.
- **no-interior stratum** (Lane alone): mean depth **1.134 px**, shell **75.04 %**, truncates at
  depth 4, barrier **5.10**, interface asymmetric **5.85×** per pixel.

**The stratification is in the GEOMETRY, not in the flip counts — F2 below refutes the stronger
version of this claim** and I have restated it accordingly. The two strata do **not** obey different
*flip-count* scaling laws; they obey the same one. What differs is the **barrier**: the same flip
count costs 6.5×–11.4× more to undo on Lane. So the stratification is a statement about
**repair cost**, which the objective does not see, **not** about where the flips are.

**Credit where it is owed — this conclusion is already in the tree.** `src/tac/boundary_math/lane_sdf_component.py`
derived it independently and **built** it: *"the OPTIMAL FORM of the lane is NOT a per-pixel margin
loss over the whole lane region … but a SIGNED-DISTANCE FIELD"*, *"MANIFOLD: DOF = the ~7 coeffs/line
… NOT H·W pixel weights"*. `SPEC_v10` records *"thin-structure cells are all-border."* **I am not
claiming the curve-carrier idea.** What this arm adds is (a) the **five-class quantification** showing
it is a **threshold with a 7.7× gap**, not a gradient — so the stratification is *binary and
enumerable*, not a per-class tuning problem; (b) the measured fact that the interior stratum's
interfaces are **symmetric**, so the one-sided machinery is needed on **exactly one** interface; and
(c) the **extent-not-density** correction that changes what the one-sided term should be.

### F2 — I registered a falsifier against this framing, ran it, and IT FIRED

**F2 as registered:** *interior-stratum residual scales with interface length; Lane's scales with
area; REFUTED if Lane's flip count is better explained by interface length than by area.*
Closed at $0 (`scratchpad/hg1_probe4_f2_scaling_law.py`, `.omx/research/ddm_hg1_f2_scaling_law.json`):

| class | area | perimeter | **width 2A/P** | flips out | **flips/perim** | flips/area |
|---|---:|---:|---:|---:|---:|---:|
| **Lane** | 690,639 | 822,062 | **1.68** | 28,564 | **0.03475** | 4.14e−02 |
| Movable | 1,460,325 | 191,306 | 15.27 | 8,102 | **0.04235** | 5.55e−03 |
| Road | 27,407,046 | 1,512,234 | 36.25 | 27,227 | 0.01800 | 9.93e−04 |
| MyCar | 29,993,509 | 322,948 | 185.75 | 3,082 | 0.00954 | 1.03e−04 |
| Undrivable | 58,413,281 | 391,284 | 298.57 | 8,937 | 0.02284 | 1.53e−04 |

**`CV(flips/perimeter) = 0.4598` vs `CV(flips/area) = 1.6603` — the perimeter model wins by 3.6×,
and Lane's z-score under it is `+0.96`, comfortably inside 2 σ. Movable is a *larger* outlier than
Lane.** **F2 REFUTED. Erasure flip count follows one interface-length law across all five classes,
Lane included. There is no "annihilation regime" visible in flip counts.**

*(Independent cross-check, free: the `2A/P` column is computed from `sx1`'s crack-length counting,
a completely different estimator from my EDT. It reproduces the stratification — Lane 1.68 px,
next class 15.27 px, a 9.1× gap — confirming §4's `mean_depth` ordering by a second route.)*

**What survives the refutation, and it is sharper than what I claimed.** The *geometry* is
stratified (five measures, two independent estimators). The *flip count* is not. Reconciling those
two facts gives the finding:

> **`d_seg` is a flip count, and flip count is governed by interface length uniformly. So `d_seg`
> prices the erasure of Lane and the nudging of a Road boundary IDENTICALLY, per pixel — while
> their barriers differ by 6.5×–11.4×. The metric is blind to the property that determines how hard
> a flip is to FIX.**

That is the mechanism under `sx1`'s *"description is cheap (0.5349 B/flip vs W = 1.2731), the
what-RGB is everything."* Naming the flips is cheap because they lie on interfaces and interfaces
are uniform; *realizing* the fix is expensive and **unevenly** expensive, and no term in the
objective sees that unevenness.

**The actionable consequence — a ranking key that exists and is unused.** `br1` measured that the
byte side carries no ranking information (*"rank units by FLIP damage, not by bytes"* — yield flat
at ~211 B/unit). §4 now adds: **the flip side carries little ranking information either**
(flips/perimeter CV = 0.46 across a 178× span of class areas). **What discriminates is the barrier
integral — 5.10 to 58.39, an 11.4× spread — and nothing currently measures or consumes it.** It is
already computed here for all 11 major directed sides, n600, vehicle-independent, free. **This is a
candidate ranking key for the `#766` waterfill that neither the byte axis nor the flip axis
provides.**

### The falsifiers still owed

| # | prediction | refuted if |
|---|---|---|
| **F1** | On **cx1's own** n600 argmax, the `Road↔Lane` flip **mass** splits with `Lane→Road` ≥ 60 %. | within 55/45 — the witness-vehicle asymmetry does not transfer. |
| **F3** | An extent/existence term on Lane beats a density-weighted signed hinge (B16) at matched bytes. | B16 wins or ties — the density asymmetry mattered after all. |
| **F4** *(new, from F2's refutation)* | Ranking `#766` units by **barrier integral** beats ranking by flip count at matched bytes. | no separation — the barrier is not the missing signal either. |

### One more correction F2 forced: rate asymmetry is NOT flip-mass asymmetry

The `Road|Lane` annulus is **unbalanced 2.67×** (289,822 Road-side px vs 108,455 Lane-side), because
a 2-px annulus on the Lane side is truncated by Lane's own 1.68-px width — the same truncation as
§4, appearing in the population rather than the profile. Therefore:

- **per-pixel rate asymmetry = 5.85×** (0.2625 vs 0.0448)
- **flip-mass asymmetry = 2.19×** (28,470 vs 12,997)

Both are true and they are different quantities. **The gap-share arithmetic in §4 uses the
flip-mass split (68.66 %), which is the correct one for that purpose** — but anyone quoting "5.85×"
as a share of damage would be wrong by 2.7×.

---

## 7. Assumption ledger

| # | claim | status | basis |
|---|---|---|---|
| A1 | Lead A's falsifier is pre-registered with the stated bands | **VERIFIED_VIA_SOURCE_INSPECTION** | `negatives_scale_validity_review_20260707.md` §7 quoted |
| A2 | It ran 2026-07-08 and returned INDETERMINATE | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | `q1_signed_asymmetry.json`, `generated_utc` + verdict block |
| A3 | The `sR` positive control is declared but never read by the verdict code | **VERIFIED_VIA_SOURCE_INSPECTION** | `signed_flip_asymmetry_correlator.py` — `positive_controls` appears only in the result dict |
| A4 | `sR` is correctly aligned ⇒ genuinely at chance (H2 refuted) | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | probe 1, deployed map beats random control 2.9 σ; all alternatives ≤ random |
| A5 | Three pooling pathologies incl. mass dilution + Simpson | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | probe 1, exact re-accumulation from caches |
| A6 | n600 class geometry (shell %, mean depth) | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | probe 2, n600, GT only |
| A7 | `Lane→Road` is the only truncating side; barrier 5.10 vs 33–58 | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | probe 3 truncation ledger |
| A8 | Interior–interior interfaces are symmetric within 5 % | **VERIFIED_VIA_EMPIRICAL_ANCHOR** *(witness, 96 frames)* | probe 3 ratio table; **vehicle caveat travels** |
| A9 | Drops compose by union ⇒ idempotent commutative semilattice | **DERIVED** from `br1` §S3's "group drop of those same 20 units" | commutator ≡ 0 follows |
| A10 | Therefore `holonomy()` ≡ 0 and would be a fake instrument | **DERIVED** | A9 + the definition of holonomy |
| A11 | The memo self-contradicts at lines 798 vs 802 | **VERIFIED_VIA_SOURCE_INSPECTION** | both quoted |
| A12 | Lane's 1.134 px depth is below the SegNet stem's internal pitch | **INFERRED_FROM_DOMAIN_LITERATURE** — CLAUDE.md asserts a stride-2 stem; I did **not** re-derive it from `upstream/modules.py`. **PROVISIONAL. The §4 findings do not depend on it** — "a class with no interior can be annihilated by a bounded perturbation" is purely geometric. |
| A13 | `Lane→Road` ≈ 15.2 % of the gap | **HYPOTHESIS** — mixes `ddm_pc2` (cx1) with the witness split. Closed only by F1. |
| A14 | P1's rs = +0.857 | **MEASURED but NOT significant** — 4 independent clusters, class-level rs = 0.80, p ≈ 0.33 |
| A15 | Erasure flips follow one interface-length law across all 5 classes (**F2 refuted my own framing**) | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | probe 4: CV 0.4598 (perim) vs 1.6603 (area); Lane z = +0.96 |
| A16 | `2A/P` from `sx1` crack lengths reproduces the depth stratification | **VERIFIED_VIA_EMPIRICAL_ANCHOR** *(independent estimator)* | Lane 1.68 px vs next 15.27 px |
| A17 | Rate asymmetry (5.85×) ≠ flip-mass asymmetry (2.19×); annulus imbalance 2.67× | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | probe 4 edge table |

*Minor reconciliation:* probe 4's per-class flip totals (Lane 28,564, MyCar 3,082) differ by ≤ 0.25 %
from probe 1's (28,634 / 3,087) because the landed q1 artifact drops pairs below its
`--min-pixels 500` filter (`1→3` n = 252, `3→4` n = 39, `4→3` n = 36) while probe 1 re-accumulated
all pairs. Immaterial to every conclusion; recorded so the two tables can be reconciled.

---

## 8. Recursive adversarial review

**Round 1 — 4 findings (counter reset).**
1. *Charter premise wrong.* I was told the falsifier "was apparently never run." It ran. **Corrected
   in §0.1 and §1** — and the correction changes the class of the failure from unwired-but-built to
   unread-result.
2. *Alignment not assumed.* My first draft treated `ρ_sR ≈ 0` as evidence about `sR`. That is
   unwarranted while a stride is unasserted. **Added the H1/H2 discriminator (§2).** It confirmed the
   original reading, but the reading was not admissible before the test.
3. *Spearman oversold.* The 8 sides are 4 clusters. **Added the class-level n = 4, p ≈ 0.33 discount
   in §4 and A14**, and demoted the correlation below the outlier argument.
4. *Novelty overclaim.* My draft presented "Lane wants a curve carrier" as a finding. Corpus check
   found `lane_sdf_component.py` already derives **and builds** it. **§6 now credits it and states
   narrowly what is new** (`m38`: re-anchor ≠ discovery).

**Round 2 — 3 findings (counter reset).**
5. *Holonomy direction insufficient.* Fixing 798-vs-802 still leaves the formalism unchecked. The
   decisive objection is **algebraic, not directional** — drops have no inverses. **Rewrote §5**; the
   verdict changed from "the direction is wrong" to "the object does not exist here."
6. *Unification not earned.* My draft had a single-fact story in which sub-Nyquist geometry explained
   `br1`'s whitened lattice. It does not: the token lattice is our learned code, not a sampling of
   the GT field. **Recorded as refuted in §6** rather than shipped.
7. *`ρ_sR` magnitudes unstated.* "At chance" without numbers is a vibe. **Added max |ρ| = 0.0808,
   mean 0.0250 against margin's 0.24–0.38 (§2).**

**Round 3 — 2 findings (counter reset).**
8. *A12 was load-bearing and unverified.* I asserted sub-stem-pitch from CLAUDE.md testimony. **Marked
   PROVISIONAL and explicitly severed from the §4 conclusions**, which need no architectural claim.
9. *`br1` superadditivity scope.* `br1`'s own ROUND-1 CATCH #2 records the sign flipped between a
   stratified sample and the matched 20. My §5 argument must rest on the **matched** measurement only
   — it does, and A9 now says so explicitly. Note the 1.0206× itself is **irrelevant to the verdict**:
   the commutator is zero for *any* value, so the anti-build holds even if the sign flips again.

**Round 4 — CLEAN (1/3).** Re-derived every table cell against the JSON artifacts; checked that no
number is quoted without its vehicle and sample count; verified §6's negatives against their source
memos (`br1` 0.83 %/1.109×/211 B, `sx1` 0.5349 vs 1.2731, `pz1` +0.000394/49.3 %, `ra1` 93.5 %).

**Round 5 — CLEAN (2/3).** Attacked the load-bearing claim (§4 extent-not-density) three ways: (i)
could truncation be an artifact of the MAXD = 12 cap? No — Lane falls below 1 % at depth **4**, far
inside the cap. (ii) Could the rival-class assignment mislabel populations? The `i→j` population is
defined by *geometric* nearest rival, not logit runner-up; this is stated in §4 and is a genuine
limitation of the construction, but it is applied identically to all 14 sides so it cannot manufacture
a Lane-specific outlier. (iii) Could Lane's shell fraction be an artifact of its small area? No — area
does not enter the shell computation, and Movable (1.46 M px, 2.1× Lane's area) sits at 9.73 %, with
Undrivable (85× Lane's area) at 0.57 % — the ordering is **not** monotone in area, so area is not the
driver.

**Round 6 — clean at the time.** Every §0 headline traced to a JSON artifact; every negative carried
its verdict scope; A12 and A13 labelled and severed. **This was NOT a seal** — see round 7.

**Round 7 — 3 findings (counter RESET to 0). The arm's own falsifier fired.**
10. **F2 REFUTED my §6 framing.** I had listed F2 as owed-and-unbudgeted. That was polish-hoarding
    (operating manual §8.10): it was a 20-minute $0 test gating my own central claim. I ran it and
    **it refuted the interior/no-interior *scaling-law* dichotomy** — erasure flips follow one
    interface-length law across all five classes, Lane z = +0.96, and **Movable is a larger outlier
    than Lane**. §6 rewritten; §0 gains headline 7. *A registered falsifier that fires against the
    author's own framing is the point of registering it.*
11. **Rate asymmetry was being conflated with flip-mass asymmetry.** 5.85× is per-pixel; the
    flip-mass split is **2.19×** because the annulus is imbalanced 2.67×. My §4 gap-share arithmetic
    happened to use the correct one (68.66 % flip mass), but the memo did not distinguish them and a
    reader would have propagated 5.85× as a damage share — wrong by 2.7×. **Both now stated
    explicitly.**
12. **The refutation produced a better claim than the one it killed** — the barrier/flip-count
    blindness of `d_seg`, and the barrier integral as an unused ranking key (new F4). Added.

**Round 8 — 1 finding (counter RESET).**
13. *The new text is unreviewed new code.* Re-derived every number in the rewritten §6 from
    `ddm_hg1_f2_scaling_law.json` rather than from the console output I had just read. Found a
    reconciliation gap: probe 4's per-class flip totals differ from probe 1's by ≤ 0.25 % (the q1
    artifact's `--min-pixels 500` filter drops three tiny pairs). Immaterial, but unexplained
    discrepancies between two of my own tables are exactly what a reader should not have to chase.
    **Documented under the assumption ledger.**

**Round 9 — CLEAN (1/3).** Attacked the surviving §6 claim (`d_seg` is blind to repair cost) for
circularity: is "barrier" merely a restatement of "margin", which `d_seg` already reflects? **No** —
`d_seg` is a 0-1 count of argmax disagreements and contains no margin information at all; the
barrier is an integral of margin over *support*, and §4's P1 table shows support and density are
separately measurable and rank flips differently (`shell_fraction` +0.857 vs `margin_d1` −0.405).
Not circular.

**Round 10 — CLEAN (2/3).** Checked that F2's refutation does not undermine §0.3–0.5 or §5. It does
not: §0.3/0.4 are geometry and rate measurements, untouched by a flip-count scaling law; §5 is a
pure algebra argument about drop composition. Verified the `2A/P` cross-check is genuinely
independent of the EDT (different input file, different estimator, different definition) and that it
agrees on ordering and on the size of the gap.

**Round 11 — CLEAN (3/3). SEAL.** No new findings. Every claim carries its label; the one claim that
was refuted is reported as refuted in the headline rather than quietly removed; the two remaining
non-measurements (A12, A13) are severed from the conclusions.

---

## 9. What this forbids

1. **Do not build `holonomy(loop_in_base)` or `parallel_transport` as specified** (§6.2 of the
   geometry synthesis). The holonomy is identically zero on an idempotent semilattice. The
   `ΔS ∈ [−0.010, −0.003]` is unbacked.
2. **Do not build B16 as a density-weighted signed hinge.** The measured density asymmetry runs the
   wrong way (§4).
3. **Do not cite `sR` as a signed reachability authority** without re-establishing it against flips.
   On the only surface where it has been tested it is 3–4× worse than the free cached margin, and it
   costs 450 MB.
4. **Do not re-review other negatives "per-side" as a blanket action.** The addendum's binding
   consequence assumed general asymmetry; measured, interior–interior interfaces are symmetric within
   5 %. **Only Lane-touching negatives earn the re-review.**
5. **Do not quote the pooled `msal_uni` ρ = −0.033 as evidence of anything.** §3 shows the pooled
   estimator on this surface is subject to all three pathologies at once.

---

## NEXT-IF-RESUMED

Ranked by *distance to the next exact row*, not by interest. **F2 is CLOSED (refuted) — it is not
on this list.**

1. **[$0, ~1 h] Test F4 — the barrier integral as the `#766` ranking key.** This is now the highest
   item because it is the only one that can move bytes. `br1` proved the byte axis carries no
   ranking signal; F2 proved the flip axis carries little; the barrier integral spans **11.4×** and
   is **already measured, n600, vehicle-independent, free**
   (`.omx/research/ddm_hg1_signed_depth_profile_n600.json`). Re-rank the `#766` waterfill by
   per-directed-side barrier and compare against the flip-count ranking at matched bytes.
2. **[1 scorer pass, `ddm_pu2`'s slot] Close F1 / A13.** Directed `Road↔Lane` decomposition on
   **cx1's own** n600 argmax. Converts the ≈15.2 %-of-gap hypothesis into a measurement. No cx1
   per-pixel argmax is cached anywhere (`ddm_pz1`'s n600 seg artifact is per-pair only) — this
   genuinely needs the pass.
3. **[$0, ~30 min] Sweep the silent-sentinel class (§2).** Find every probe artifact under
   `experiments/results/` and `.omx/research/` with a `positive_control`/`sentinel` key and check
   whether any code reads it. `ddm_pz1`'s computed-`ratio` control is the template; the correlator is
   the counter-example. **Prediction: the correlator is not the only one.**
4. **[$0] Re-run the correlator with `margin` as the field under test.** It is already the strongest
   column in the artifact (|ρ| 0.24–0.38, correct sign everywhere) and it is **free and cached**,
   whereas `sR` costs 450 MB and is at chance. One flag; the accumulators already exist.
5. **[cheap] Re-scope, do not re-open, the texture family.** The registered bands still say
   INDETERMINATE. Before spending anything, note §3: the *directed* max was **0.1242**, which clears
   the 0.1 kill floor. The family cannot be declared robust-dead on the current evidence, and it
   should not be reopened without a reformulated cost field either.

**Files landed:** this memo · `.omx/research/ddm_hg1_probe1_sr_alignment_and_pooling.json` ·
`.omx/research/ddm_hg1_signed_depth_profile_n600.json` ·
`.omx/research/ddm_hg1_signed_depth_profile_n24.json` ·
`.omx/research/ddm_hg1_truncation_and_prediction.json` ·
`.omx/research/ddm_hg1_f2_scaling_law.json` · `scratchpad/hg1_probe{1,2,3,4}_*.py`

**Pointer `0.1910828242` UNMOVED. No score claim.** This arm produced no exact row. It produced
**two anti-builds** (`holonomy()`/`parallel_transport` as specified; B16 as a density-weighted
signed hinge), **one refuted deployed disposition** (`sR` as the signed reachability authority),
**one estimator-class correction** (three pooling pathologies, not one), **one measured
vehicle-independent geometric stratification**, **one self-refutation** (F2), and **one candidate
ranking key** (the barrier integral) that is already measured and consumed by nothing.
