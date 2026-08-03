---
schema: ddm_mg1_margin_geometry_cure.v1
date_utc: 2026-08-03
arm: ddm_mg1 (what IS useful for the argmax boundary, if Morse-Smale is not)
lane_id: "lane_ddm_mg1_20260803"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
verdict_scope: SEE-PER-ROW
axis: "[macOS-CPU scorer-free advisory]. NO SegNet or PoseNet forward or backward was
  fired; the evaluator slot (held by ddm_pu2) was neither requested nor touched. Every
  number is from already-cached n600 artifacts or from executed numerics on the real
  loss functions."
consumes:
  - .omx/research/ddm_mf1_margin_morse_licence_20260803.md            (the question)
  - .omx/research/ddm_rt2_realized_dseg_discarded_20260803.md         (C1/C2, and its own re-check 34011da8c1)
  - .omx/research/ddm_hg1_negatives_as_geometry_20260803.md           (F4)
  - .omx/research/ddm_hg1_signed_depth_profile_n600.json              (barrier profiles)
  - .omx/research/ddm_rs2_flip_damage_rerank_and_drop_seg_leg_20260803.md (the #766 currency)
  - experiments/ddm_wr1_reverse_waterfill.py                          (task #766)
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/      (live cx1 per-pixel argmax)
  - /Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_cell_sensitivity_atlas.npz
  - experiments/results/mlx_fleet_gt_cache/gt_n600.npz                (lstars, margins)
consumers: [MAIN, ddm_pu2, ddm_rs2, ddm_wf2]
tokens: [no-triality] [p0-ledger-ok]
---

# ddm_mg1 — the useful object is the head's scalar margin field, used as a WEIGHT; and the barrier is not a ranking key

**Operator's question:** *"If the framework was 89% right about where the object lives but still
useless for the question we're asking it, what would be useful?"*

**One-sentence answer, measured:** the frozen head's own margin `m = z_target − max_competing z`
selects the vehicle's realized failures with **98× enrichment** and **42.5% precision** in its
lowest band — **8.5× more precise than the Morse–Smale complex `mf1` measured (5.0%)** — and it is
one subtraction on an output we already compute; what is broken is not *where* the objective looks
but *how loudly it speaks there*, so the lever is the hinge **weight**, and the two cures I was sent
to build (raise the floor; rank by barrier) are both **refuted by execution**.

---

## 0. HEADLINE

| claim | verdict | evidence |
|---|---|---|
| C1 "raise `margin_floor` toward global p5% = 2.0582" | **REFUTED, 3 independent executed facts** | §1 |
| The head's margin field localizes THIS vehicle's failures | **VERIFIED — 98× enrichment, 42.5% precision** | §2 |
| `lane_guard`'s inherited claim *"100% of flips are in the bottom GT-margin decile"* (sg1) | **CORROBORATED at 99.95%**, n600, global field | §2.3 |
| The right lever is `margin_hinge_weight`, and its value can be **derived** | **DERIVED: w\* ∈ [0.5, 0.8] = 10–16× the shipped 0.05** | §3 |
| `hg1` F4 — barrier integral beats flip count as the `#766` key at matched bytes | **REFUTED at the operating point** (advantage **0.0000%** of damage out to 30% of bytes freed; shuffle control costs 5.6–65.4%) | §4 |
| `derive_margin_floor` "is NOT CALLED" | **CORRECTED — it IS called**, at `src/tac/optimization/lane_guard.py:810` | §5 |
| "all three constructor sites" | **CORRECTED — there are EIGHT** | §5 |
| A **third**, unnamed orphan: `margin_targets` is declared, assigned, and never read or passed | **NEW** | §5.2 |

**Pointer UNMOVED.** Live best `S = 0.7910689`, seg leg `0.4311790`, gap to the PR130 bar `0.6189279`.
Nothing here is a score claim.

**Instrument validated against an independent authority:** my recomputed flip mask gives seg leg
**0.4311795** against `cx1`'s evaluator row **0.4311790** — a difference of **4.70e-07**. Every §2/§4
number rides on a mask that reproduces the authority row.

---

## 1. C1 IS REFUTED — three facts, each executed, each independent of `rt2`'s re-check

`rt2`'s re-check (`34011da8c1`) withdrew C1 on the grounds that the shipped floor is
*derived-equivalent* (Lane-restricted p10 = 0.104748 vs shipped 0.1, and the fp32 drift guard ~0.096
agrees). That is a statement about **where the floor should sit**. Below is an independent refutation
from **what the floor does**, which holds regardless of the derived value.

### 1.1 Raising the floor cannot recruit one additional flipped site — exactly, not statistically

A site is a realized flip **iff** `argmax ≠ target` **iff** `m < 0`. The hinge `relu(f − m)` is
active whenever `m < f`. For any `f > 0`, every flipped site satisfies `m < 0 < f`. So **flip
coverage is already 100% at the shipped floor**, and every site a higher floor recruits has
`m ≥ f_old > 0` — a site the scorer **already gets right**.

Executed on the real hinge over a margin vector straddling the separatrix:

| floor | active sites | **flipped sites active / total** | distinct non-zero grad magnitudes |
|---:|---:|:--:|:--:|
| 0.1 | 7 | **3 / 3** | **1** |
| 2.0582 | 9 | **3 / 3** | **1** |

Raising the floor added 2 sites; **neither was a flip**.

### 1.2 The relu gradient is FLAT, so already-covered sites gain nothing

`∂/∂m mean(relu(f − m)) = −1/N` on active sites and `0` elsewhere — measured above as
**exactly one distinct non-zero gradient magnitude at both floors**. Raising the floor therefore does
**not** increase the push on any site that was already active. It adds *equal* push elsewhere. A
higher floor is not a sharper objective; it is a wider one.

### 1.3 The floor → ∞ limit is *exactly* the bulk objective #888 complains about

Executed at `f = 1e6`: `mean(relu(f − m))` vs `f − mean(m)` → **`abs_diff = 0.0`**.

So `lim_{f→∞} margin_floor_hinge = f − mean(margin)`: a **flat-weight, whole-plane mean-margin
objective**. Raising the floor moves the lever *monotonically toward* the bulk, label-driven
allocator that `er1` diagnosed in CE. This is the **mechanism** behind the charter's "would import
the exact poor-allocator pathology" — as an exact identity, not an analogy. On the measured field,
`f = 2.0582` already puts **91.3%** of the hinge's active support on sites that never flip (§2.1).

> **`verdict_scope: FORMULATION` — the `margin_floor_hinge_mlx` family.** Raising `margin_floor` is
> not a separatrix-aimer at any value. §1.1 and §1.3 are properties of the relu, not of 0.1 or of
> 2.0582.

---

## 2. WHAT *IS* USEFUL — the head's margin field, as a per-site weight

n600, **117,964,800 sites**, **508,640 realized flips**, controls in §8.

### 2.1 Enrichment: the margin field is a near-perfect selector

| GT margin < | share of all sites | share of all flips | **enrichment** |
|---:|---:|---:|---:|
| 0.01 | 0.0282% | 3.06% | **108.7×** |
| 0.05 | 0.1413% | 14.75% | **104.4×** |
| **0.1 (shipped floor)** | **0.2824%** | **27.69%** | **98.1×** |
| 0.2 | 0.5622% | 48.33% | 86.0× |
| 0.5 | 1.3824% | 81.87% | 59.2× |
| 1.0 | 2.6702% | 96.86% | 36.3× |
| 2.0582 (`rt2` C1 target) | 4.9520% | 99.60% | 20.1× |
| 4.0 | 10.3019% | 99.95% | 9.7× |

### 2.2 Precision — the direct answer to the operator's question

| band | sites | flips | **flip rate (precision)** |
|---|---:|---:|---:|
| margin ∈ [0, 0.096) | 319,903 (0.271%) | 136,003 | **42.51%** |
| margin ∈ [4, 8) | 103,728,616 (**87.93%** of the plane) | 267 | **0.000257%** |

A **165,165× ratio** in flip rate between the two ends of the same scalar field. Mean GT margin is
**0.3010** on flip sites vs **5.6346** on correct sites (**18.7×**).

**Against `mf1`:** Morse–Smale gave 89.3% recall at **5.0% precision** (a 17.8× superset). The
head's margin band `< 0.096` gives **42.5% precision** — **8.5× more precise** — and needs no cell
complex, no persistence pairing, and no extra forward pass: it is `top1 − top2` on logits the loss
already has in hand. *That* is what is useful. The object to keep from `mf1` is not the topology; it
is its own conclusion that the **frozen affine rank-4 head defines the boundary exactly** — and the
head hands us that boundary as a **scalar per site**, which is usable as a **weight**, whereas a cell
complex is only usable as a **set**.

### 2.3 A prior repo claim, checked rather than reinvented

`lane_guard.py:550-554` documents *"100% of flips are in the bottom GT-margin decile; sg1 §1.3."*
Measured here on the **global** field: the bottom decile is `margin < 4.0` (10.30% of sites) and holds
**508,373 / 508,640 = 99.95%** of flips. **Corroborated**, with the residual 267 flips named.

---

## 3. THE WEIGHT, DERIVED — no free parameter

The charter's instruction was *"derive what weight that implies rather than picking one."*

**Design target (stated, because it is a choice):** the margin term should be **as loud as CE exactly
at the separatrix** — the locus where the score is decided.

At the separatrix (`m = 0`), for the seg leg `CE + w · relu(f − m)`:

* hinge push on the target logit is exactly **`w`** (§1.2: the relu derivative is 1);
* CE push on the target logit is **`1 − p_target`**.

At `m = 0` the target logit ties the best competitor, so with `C = 5` classes the softmax denominator
lies between `2·e^{z_t}` (only the tied competitor matters) and `C·e^{z_t}` (all tied):

```
p_target ∈ [1/C, 1/2] = [0.20, 0.50]   ⇒   1 − p_target ∈ [0.50, 0.80]

w* ∈ [0.50, 0.80]        = 10× – 16× the shipped margin_hinge_weight = 0.05
```

Bounds only — no distributional assumption beyond `C = 5` and the softmax. The shipped 0.05 is
**an order of magnitude below** parity at the separatrix, which is the quantitative form of "the hinge
is inert": at `w = 0.05` the margin term contributes ~6–10% of the per-site push CE already supplies
at the boundary, and **0** outside the floor.

> **`verdict_scope: DERIVATION`** — an analytic bound, not a measurement. It says where to put the
> A/B's centre; it does **not** predict Δd_seg. The A/B is owed (§7).

---

## 4. F4 — the barrier integral is NOT the missing ranking key

`hg1` registered F4: *"Ranking `#766` units by barrier integral beats ranking by flip count at
matched bytes"*, kill = *"no separation."* `rs2` established the currency (`wr1:93`
`lexsort((-residual_mass, flip_mass))` — flip count ascending is already primary). `rs2` never
touched the barrier (**zero occurrences** in its memo). The test was open.

### 4.1 Barrier re-derived, not copied — and the naive formula is wrong

The barrier **truncates** at the depth where population falls below 1% of the depth-1 population.
Summing all depths gives `1→0 Lane→Road = 12.48`, not `hg1`'s **5.10** — which would have destroyed
the very 11.4× spread F4 exists to test. Correct formula, verified against **all 11** published rows:

```
barrier = Σ_d mean_margin_d   over d with  n_d ≥ 0.01 · n_depth1        worst |Δ| = 0.0037
```

### 4.2 Matched bytes, damage in the barrier currency, with a positive control

Units are `wr1`'s 768 cells; byte proxy is `wr1`'s own `residual_mass`; damage is barrier mass.

| bytes freed | damage, flip-count key | damage, **barrier key** | barrier advantage (% of total damage) | shuffled-key control |
|---:|---:|---:|---:|---:|
| 5% | 0 | 0 | **0.0000%** | +5.61% |
| 10% | 0 | 0 | **0.0000%** | +11.50% |
| 20% | 0 | 0 | **0.0000%** | +27.49% |
| 30% | 0 | 0 | **0.0000%** | +46.86% |
| 50% | 6,286 | 3,272 | 0.0175% | +59.84% |
| 70% | 9.369e5 | 9.039e5 | 0.1920% | +65.40% |
| 90% | 8.495e6 | 7.683e6 | 4.7282% | +42.52% |

**Drop order is bit-identical for the first 384 of 768 cells.** The shuffled-key control shows the
instrument resolves ranking differences at the 5–65% level, i.e. it is **~1000× more sensitive than
the effect it failed to find**. This is a measured null, not a blind one.

### 4.3 Grain law — and a vacuity I had to catch in my own statistic

The null could have been an aggregation artifact (a bounded per-flip weight cannot reorder cells
whose flip counts span 3,320×). Block-reducing the same per-pixel measurement to every grain:

| cell | cells | flip-bearing | mean flips/cell | ρ over ALL cells | **ρ, flip-bearing only** | drop overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 1 px | 196,608 | 43,798 | 11.6 | 0.9979 | **0.8999** | 88.08% |
| 4 px | 49,152 | 13,107 | 38.8 | 0.9979 | 0.9319 | 91.30% |
| 16 px | 12,288 | 3,816 | 133.3 | 0.9976 | 0.9469 | 93.03% |
| 64 px | 3,072 | 1,053 | 483.0 | 0.9974 | 0.9543 | 94.68% |
| **256 px (`wr1`)** | **768** | **282** | **1803.7** | 0.9966 | **0.9489** | 91.49% |

> **Self-caught vacuity.** My first grain table reported 100% drop overlap at every grain. That was
> **vacuous**: zero-flip cells are tied at 0 under *both* keys (barrier mass is a sum over flips), and
> at 1 px that tie block is **152,810 of 196,608 cells (78%)**. The statistic was measuring the zero
> set. Every ranking figure above is now restricted to **flip-bearing cells**, with the denominator
> printed. The tie-inflated column is retained beside it to show the size of the trap.

Guarded, the barrier **does** reorder — ρ = 0.90 at 1 px, 0.95 at `wr1`'s grain, ~8.5% of
flip-bearing cells. **But the reordering lands where barrier mass is negligible**, so matched-byte
damage is unmoved across the entire usable region. The control that grain-256 reproduces the
independently-computed 768-cell ρ to 9 decimals (`0.996611192`) confirms the reduction is sound.

**Mechanism:** barrier is a **per-side constant**, and flips are **spatially segregated by side**
(pixels on a Road/Lane boundary flip `0→1` or `1→0`, not `2→3`). So barrier mass ≈ (locally constant)
× flip count — a near-monotone rescaling, which barely reorders at any grain. **This kills the whole
family of "re-weight the flip count by a per-class-pair scalar" cures**, not just this one.

> **`verdict_scope: INSTANCE — the #766 waterfill on the live cx1 n600 at grains 1–256 px`:** F4
> **REFUTED**, per its own pre-registered kill. `hg1`'s underlying finding — that `d_seg` is blind to
> repair cost — is **untouched**; what is refuted is that the blindness shows up as a *reordering of
> this allocator's units*. Repair cost is a **per-site** quantity (`|m|`); the barrier is its
> **per-side average**, and averaging over a side is exactly what destroys it.

---

## 5. THE ORPHAN LEDGER, CORRECTED — and why nobody wired the derivation

### 5.1 Two corrections to the relayed picture

* **`derive_margin_floor` IS called** — `src/tac/optimization/lane_guard.py:810`, via
  `cfg.margin_floor_pct`. It is orphaned only **with respect to the joint-descent path**, which never
  imports `lane_guard`. (It also lives in `tac.optimization`, not `tac.witness_control`.)
* **There are EIGHT constructor callsites**, not three:
  `tools/launch_ddm_joint_descent.py:481, :1201, :1543, :2601` ·
  `tools/measure_ddm_fd2_posenull_gn_disambiguation.py:156` ·
  `tools/run_ddm_j12_receiver_coordinate_custody.py:455, :1772` ·
  `tools/smoke_ddm_fd1_gn_engine.py:59`. All hardcode the defaults.

### 5.2 The third orphan, and the reason the wiring was never done

`margin_targets` is **declared** (`direct_description_joint_descent.py:2279`), **assigned**
(`:2290`), and **never read anywhere and never passed by any of the eight callsites** (exhaustive:
`git grep margin_targets` returns exactly those two lines). It is a dead parameter — and it is
**the input `derive_margin_floor` would need**. That is why the derivation was never wired: there was
nothing to take a percentile of.

### 5.3 I did not wire it, and this is the reasoning

Wiring requires **two** changes, not one: plumb `margins` from the cache through eight callsites into
`margin_targets`, *then* call `derive_margin_floor` on the Lane-restricted subset. Adding the call
alone, behind an input nobody supplies, would ship a **never-fired lever** — the exact orphan class
the instruction is trying to close. And the measured payoff is nil: the derived floor is **0.104748**
vs shipped **0.1**, moving activatable sites **0.2824% → 0.2956%**.

**So: hygiene, not a lever.** It should be wired — with `margin_targets` plumbed — by whoever next
opens that path for the §3 weight A/B, in the same change, so it is exercised the moment it lands. I
am recording it as owed rather than half-wiring it in a slot that cannot test it.

---

## 6. WHAT THIS FORBIDS

1. **Do not raise `margin_floor`** to reach the separatrix. §1.1/§1.3 — it recruits only
   already-correct sites and limits to the bulk objective.
2. **Do not rank `#766` units by barrier integral.** §4 — 0.0000% matched-byte advantage against a
   control that resolves 5–65%.
3. **Do not reach for a squared hinge** as the "obvious" sharpening. `relu(·)²` gives gradient ∝
   deficit, i.e. it prioritizes the **deepest** (most expensive) flips. `d_seg` is a **count**, so
   under scarce capacity the score-optimal order is **cheapest-first**; a convex hinge is
   anti-aligned. (Derivation, untested — flagged in §7, not a licence.)
4. **Do not add the realized `d_seg` to the objective.** `rt2` executed it: gradient identically zero.
5. **Do not cite `mf1`'s Morse–Smale complex as a boundary selector** when the margin field is 8.5×
   more precise and free.

---

## 7. OWED (each needs the scorer slot `ddm_pu2` holds, or a run)

1. **The weight A/B — the one that can move the pointer.** `margin_hinge_weight` 0.05 → the §3
   derived band, matched seed and description budget, n600 through the real byte-close, reporting
   **d_seg *and* d_pose *and* bytes** against `S = 0.7910689` / seg leg `0.4311790`.
   **Pre-registered kills:** (a) Δd_seg ≤ single-seed noise; (b) proxy margin improves but realized
   `d_seg` does not; (c) **d_pose regresses** — a seg-only A/B is forbidden, `uv1` measured a 3,019×
   d_pose separation between bases under an identical solver; (d) **the L7 cross-hardware
   portability guard degrades** — raising the *weight* strengthens margins so it should not, and
   that prediction is itself the test.
2. **Plumb `margin_targets` + call `derive_margin_floor`** (§5.3), in the same change as (1).
3. **Repair cost as a per-SITE weight.** §4 refuted the per-side average; the per-site `|m|` is
   untested and is where `hg1`'s blindness actually lives. Needs candidate margins = a scorer pass.
4. **Concave vs flat vs convex hinge shape** (§6.3) — derivation only; no measurement.

---

## 8. CONTROLS (each would have exposed a dead instrument)

| control | result |
|---|---|
| `pu2` GT argmax vs canonical `lstars`, per site | **0 disagreements / 117,964,800** |
| `margin < 0` on the cached field | **exactly 0** — confirms GT-reference (self) margins |
| my flip mask vs `cx1`'s evaluator authority row | seg leg 0.4311795 vs 0.4311790, **Δ = 4.7e-07** |
| barrier re-derivation vs `hg1`'s 11 published rows | worst \|Δ\| **0.0037** |
| cell tiling covers the plane exactly once | 768 cells × 256 px, min = max = 256 |
| flips on sides with no published barrier | **0** (0.0% dropped to zero weight) |
| grain-256 vs independent 768-cell ρ | identical to 9 dp (`0.996611192`) |
| shuffled-key drop control | costs 5.6–65.4% of damage — instrument is live |
| **`pu2` cache completeness** | **caught mid-write**: first run saw 593/600 pairs written (tail all-zero memmap fill; `per_pair_directed.jsonl` = 593 lines) and the control **refused the cross-tab**; the re-run after `pu2` finished passed at 600/600. Detection is now in the probe. |

---

## 9. ROUND-1 ADVERSARIAL SELF-REVIEW

**The charter named the likely failure: "C1 works on the proxy while realized n600 d_seg does not
move, or it silently breaks the portability guard." I attacked both — and a third.**

1. **Did I refute C1 on a proxy?** The GT-reference margin field is **not** what the in-loop hinge
   sees (it sees *candidate* margins). If my refutation rested on the GT field it would be exactly the
   proxy failure named. It does not: §1.1 and §1.3 are **algebraic properties of the relu and of the
   flip definition**, true for any margin field. The GT field is used only in §2, for enrichment —
   where it is the right object, because the question there is *where the boundary is*, not *what the
   loop sees*. **Attack survived, by moving the load-bearing claim off the proxy.**
2. **The portability guard.** `rt2`'s pre-registered kill was "C1 degrades the L7 guard." My
   measurement says that kill **points the wrong way**: raising the floor pushes *more* sites further
   past the drift band, so it can only *strengthen* the guard — while spending capacity on sites that
   cannot flip. The real risk of my §3 recommendation (weight ↑) is also not guard degradation, since
   the support is unchanged and the push at the boundary increases. **I carried the guard kill into
   §7 anyway**, because "should not" is a prediction, and an untested prediction is not a guard.
3. **The vacuity in my own grain statistic** (§4.3) — caught, corrected, denominator now printed, and
   the trap retained in the table.
4. **The barrier formula** — my first implementation summed all depths and would have reported
   `Lane→Road = 12.48` instead of 5.10, deleting the 11.4× spread F4 exists to test. Caught by
   re-deriving against `hg1`'s published table instead of trusting my own reading. **This is the third
   relayed-number failure in this chain today; the pattern is that the *formula* travels worse than
   the *number*.**
5. **Did I check the repo before building?** Yes — and it changed three things: `derive_margin_floor`
   already exists (so I did not invent a floor derivation), `lane_guard`'s sg1 decile claim already
   exists (so I *checked* it instead of re-deriving it), and `rs2` already owned the `#766` currency
   question (so I tested only the untouched barrier half). The one thing I did **not** find is a
   weight derivation — hence §3 is derived rather than cited. Negative-existence scope: `git grep`
   over tracked `src/`, `tools/`, `experiments/`, `.omx/research/`.
6. **Where I am weakest.** §3's `w* ∈ [0.5, 0.8]` equalizes push **at `m = 0` only**; away from the
   separatrix CE and the hinge diverge, and I did not measure the aggregate ratio because that needs
   full logit vectors (a scorer pass). §6.3 (hinge convexity) is a derivation with **no** measurement.
   Both are labelled; neither is a licence to build.
7. **What would change my mind about the headline.** If the weight A/B (§7.1) moves realized `d_seg`
   by ≤ noise, then "the hinge is inert because it is too quiet" is wrong, and the remaining
   explanation is that the seg leg's **direction** — not its loudness — is the defect, which points at
   `rt2`'s C2 (`realized_margin_and_gradient`), not at any knob.

**Round 1: four findings against my own work (items 1, 3, 4, 6). Not a clean pass; all four are fixed
or labelled in the body above.**

---

## 10. PROBES (reproducible, scorer-free, controls inline)

1. `experiments/ddm_mg1_margin_floor_allocation_probe.py`
   → `.omx/research/ddm_mg1_margin_floor_allocation_n600.json` (§1, §2)
2. `experiments/ddm_mg1_barrier_rerank_probe.py`
   → `.omx/research/ddm_mg1_barrier_rerank_n600.json` (§4)

```
.venv/bin/python experiments/ddm_mg1_margin_floor_allocation_probe.py --pairs 600
.venv/bin/python experiments/ddm_mg1_barrier_rerank_probe.py       --pairs 600
```
