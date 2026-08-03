---
arm: ddm_p4x
task: 920
axis: "[macOS-CPU advisory]"
score_claim: false
promotable: false
research_only: true
scorer_forwards_run: 0
pointer: "0.1910828242 [contest-CPU] UNMOVED"
commits: [0b582aa070, e3146262f8, 2e8ce09749]
related: [ddm_gc16_from_here_20260803, ddm_cg1r_force_class_edge_ledger_20260803, ddm_gt2_gt_tongue_induction_20260803, ddm_pt2_lever_port_to_tr1_20260803]
---

# ddm_p4x — the lane EXISTENCE primitive + per-class BIRTH matrix

**Everything here is MEANS. The exact pointer did NOT move: `0.1910828242` [contest-CPU]
UNMOVED. Zero scorer forwards were run in this arm.** The scorer slot is held by `sq1`;
this arm emits a SEALED TICKET, not a race.

---

## §0 HEADLINE (answer first)

**BUILT TO ADMISSION** — the first verb-native force in the campaign is live-wired and
registry-clean, default-OFF and byte-identical when unarmed:

1. `src/tac/optimization/existence_hinge.py` — the COMPONENT-level existence hinge
   `s(c) = logsumexp_β(live margin over GT component c)`, `loss = mean_c w_c·relu(t − s(c))`.
   MLX == numpy reference to **4.4e-8**; gradient reaches the witness pixel (non-dead).
2. Wired into the LIVE TR1 vehicle as a **separate loss term**, not a `seg_pixel_w` addend —
   reusing the same `seg_logits` (no second forward). Anti-inertness proven structurally.
3. Six DSL `Lever` factories, registry-clean (`missing_flags` empty, `stub_marker` False,
   `trainer_declared` True, filed under the TR1 trainer).
4. 31 tests (21 primitive + 10 lever), all passing.

**TWO MEASURED FINDINGS this arm produced, both of which change how the row must be read:**

* **gt2's word grammar is 4-CONNECTED** — reproduced EXACTLY on 8/8 controls; the design
  prescribes 8-connected. Per-WORD rates are NOT comparable across grammars; S-arithmetic is.
* **A prefix subset over-states Lane annihilation by 26.3% at n=96**, outside the random-sample
  p95 of 7.3%. Bounded validation must use a seeded RANDOM pair sample, never a prefix.

**NOT BUILT (named honestly, with the exact gap):** the TR1 seed/amplify BIRTH path. The
island-protection machinery exists only on the RETIRED levelset trainer (VERIFIED: three of
its flags occur ZERO times in the TR1 trainer). See §5.

---

## §1 What the primitive is, and why the SHAPE is the whole design

`cg1r` (`ee848e88cd`) MEASURED that realized per-flip GT-margin depth is direction-SYMMETRIC on
all nine class edges (Road↔Lane **1.074×**) while the COUNT asymmetry runs to **15.88×**. The
lane-erasure discount is therefore **VOLUMETRIC/verb-level, not per-flip pricing**: a Lane word
dies at ~2.5 px of depth because Lane has no interior.

Every existing lane_guard mechanism (`lambda_lane`, born-mask, margin-floor) is an ADDITIVE
per-pixel addend folded into `seg_pixel_w`. That is precisely why the force ledger records
`protection=ABSENT` for the ANNIHILATE verb specifically: *a rim-peel guard up-weights
currently-WON support and does nothing whatever for a whole component being lost.*

So a per-pixel lever aimed at this channel is aimed at an already-symmetric quantity and should
be expected to measure NULL. The primitive is component-level:

```
s(c) = logsumexp_β( m_live(p) for p in c )        # β→∞ ⇒ the component's WITNESS pixel
L    = mean_c  w_c · relu(target − s(c))
```

`m_live(p) = logit[gt] − max_{k≠gt} logit[k]` is the SAME idiom the `margin` seg form already
uses (base trainer L1450-1451), so the term reads the vehicle's real decision surface.

**Why the max and not the mean** — a Lane component can have a strongly negative *mean* margin
while its witness pixel is comfortably alive. A word survives argmax iff **at least one** of its
pixels wins its class. Protecting the max is what makes the term area-blind and existence-
sensitive, which is the property every per-pixel surrogate lacks. (Pinned as a test.)

**Cost** — O(#components), not O(#pixels). The protected set is small: 1,151 Lane px/frame over
~24 components, 2,434 Movable px/frame over ~3.7. So the reduction is a dense `(K, n_comp)`
masked logsumexp over PROTECTED PIXELS ONLY (K≈3.6k, n_comp≈28) — no scatter kernel needed, and
`mx.logsumexp` supplies its own max-shift so it is stable without a separate scatter-max pass.
Index build MEASURED at 5.3 ms/frame, 3.2 s for the whole n600 corpus, cached — never on the
hot path.

---

## §2 The per-class BIRTH matrix — ONE mechanism, two geometries

| | Lane | Movable |
|---|---|---|
| GT words (8-conn) | 14,323 | 2,197 |
| annihilated | 7,789 (**54.38%**) | 356 (**16.20%**) |
| depth ≤1 share | 75.04% | — |
| GOUGE vs ERODE | 2,926 / 135,683 = 2.2% | 16,718 / 53,940 = **31.0%** |
| interior? | **no** | **yes** |
| `w_c` policy | `uniform` (area-blind) | `sqrt_area` |
| derived β | 7.4587 | 12.9896 |
| ANNIHILATE ceiling | 0.037276 S | 0.006900 S |

β is DERIVED, not a literal: `logsumexp_β` over `n` values overestimates their max by at most
`log(n)/β`, so holding that slack inside a declared tolerance gives
`β ≥ log(mean_component_area)/tolerance` (tolerance 0.5 margin units, DECLARED).

**Independent convergence, worth stating because it is evidence about the geometry rather than
about any one implementation.** Three surfaces reached the same partition separately:
this arm from the gt2 verb masses; **#323 `SeedIslandEased`** — *"movable via SDF forward-Euler
DILATION … + lane via openpilot VP-TANGENT along-tangent widening (manifold-preserving;
isotropic-of-a-curve is the NO-GO)"*; and **`birth_completion`** telemetry, which records
`classes: [1, 3]` — Lane and Movable.

Road (5.45%), Undrivable (6.00%) and MyCar (**0.00%**, zero annihilations in 600 frames) are
excluded. That exclusion is a measurement, not an oversight.

---

## §3 MEASURED FINDING 1 — gt2's grammar is 4-CONNECTED (the unit mismatch)

Reproduced from the same cached corpus, $0, zero scorer forwards
(`tools/ddm_p4x_connectivity_control.py`):

| class | conn | comps | annih | rate | ann_px | S |
|---|---|---|---|---|---|---|
| Lane | **4** | **16,581** | **9,655** | **0.5823** | **47,226** | 0.040034 |
| Movable | **4** | **2,207** | **361** | **0.1636** | **8,180** | 0.006934 |
| Lane | 8 | 14,323 | 7,789 | 0.5438 | 43,972 | 0.037276 |
| Movable | 8 | 2,197 | 356 | 0.1620 | 8,139 | 0.006900 |

**All 8 exact-match controls PASS at 4-connectivity** (and GT pixel totals match 1.0000, proving
the same corpus). gt2 is 4-connected; the design prescribes 8-connected (Rosenfeld: a seed that
does not respect 8-connectivity is deleted by the receiver's measured consolidation). For Lane
the grammars differ by **13.6%** — 2,258 diagonal joins.

**Why this matters and how it is resolved.** The S-arithmetic is grammar-INVARIANT per pixel, so
ceilings are safe once named. Per-WORD rates are NOT — "58.23% of Lane words annihilated" is a
4-connected statement, and a capture fraction quoted in words is meaningless across grammars.
The default stays 8-connected because the receiver constraint is physical; both denominators are
carried in the module; and the grammar is an explicit RACED lever, never implicit. The
0.002793 S difference does not vanish — it MOVES to the ERODE/GOUGE channel, which is the rim
guard's instrument, not this one.

---

## §4 MEASURED FINDING 2 — a prefix is the wrong subset (m88, reproduced on a new quantity)

Population (n600, 8-conn): word-annihilation rate 0.5438, mean **73.29** annihilated px/frame.

| subset | ann_px/frame | ratio to population |
|---|---|---|
| prefix n=32 | 72.94 | 0.995× |
| prefix n=64 | 76.09 | 1.038× |
| **prefix n=96** | **92.59** | **1.263×** |
| prefix n=181 | 87.14 | 1.189× |
| random n=64 (200 draws) | — | mean 0.993, p05 0.926, **p95 1.080** |
| random n=96 (200 draws) | — | mean 1.003, p05 0.943, **p95 1.073** |

A prefix at n=96 over-states the target quantity by **26.3%**, far outside the random-sample p95
of 7.3%. Video order is temporally correlated, so a prefix is a contiguous SCENE BLOCK rather
than a sample — an independent reproduction of the m88 law on Lane annihilation mass (m88's own
anchor was d_pose). The n=32/64 prefixes look benign by luck, not by construction.

**Operational rule for this row:** bounded validation uses a **seeded RANDOM pair sample**, never
a prefix, and any capture claim at n=96 must exceed the **±7.3%** random-sampling band.

---

## §5 What is NOT built — named, with the exact gap

**The TR1 seed/amplify BIRTH path.** The island-protection family attacks the same debt and
MEASURED a real n600 effect on the ancestor vehicle
(`experiments/results/island_survival_n600.log`: Lane survival 0.5646 → 0.9304,
`seed_birth_gain` **0.3658**; Movable 0.9102 → 0.9773, gain 0.0671). It is **not importable**,
for two independent reasons:

1. **The flags do not exist on TR1.** VERIFIED by grep: `--seed-islands`,
   `--witness-alone-island-loss`, `--amplify-weight` each occur **ZERO** times in the TR1 trainer
   and once in the RETIRED levelset trainer. There is nothing to rename — a TR1 seed/amplify
   path must be BUILT.
2. **Its numbers are ancestor-vehicle numbers** (L18). The 0.3658 gain was measured against a
   SIMULATED erasure (`erase_factor 4`), not this decode's realized annihilation, on a different
   vehicle. Cited as a DIRECTIONAL prior that the debt is attackable — never as an effect size.

What DOES transfer is the geometric split (§2), and `tac.boundary_math.island_protection.eased_island_masks`
is a MODULE (not a trainer flag), so it is reusable from TR1 without a port. That is the concrete
integration point for the next increment.

**Pose collateral (#889) is NOT measured.** The charter requires every existence/birth force to be
evaluated with pose on a MATCHED BASE, ≥32 pairs (m85). This arm ran zero scorer forwards, so the
pose leg is entirely OWED and is a gating condition of the ticket below, not an afterthought.

---

## §6 A registry-scope finding (reported, not silently absorbed)

`lever_registry.completeness()` reports my six flags as **UNMAPPED** against TR1 — but so are
`--lane-guard-born-weight` (lg1) and `--distill-weight` (dw1), both landed levers. Cause:
`dsl_referenced_flags()` reads `curriculum_dsl.py` only, while its docstring says
"ANYWHERE (module-wide)". The authoritative `build_completeness()` surface (which globs the
package) DOES hold all six cleanly. So this is a **pre-existing registry-scope gap affecting at
least three landed TR1 lever families**, not an orphan I introduced. Not fixed here: changing the
scan would move `unmapped` counts globally and could trip sister gates — that is its own row, and
doing it at the tail of a budget-limited arm is exactly the built-instead-of-paid trap.

---

## §7 THE SEALED MEASUREMENT TICKET (does NOT fire in this arm)

```
ticket:      ddm_p4x_existence_hinge_first_race
DSL_HASH:    66d7399f05426a8860069414fda33524   (sha256/32 over the arm override dicts)
git HEAD:    2e8ce09749
arms:        control        {}                                          (trainer default = OFF)
             A_lane_only    --existence-hinge-weight 0.1 --existence-hinge-classes lane
             B_birth_matrix --existence-hinge-weight 0.1 --existence-hinge-classes lane,movable
subset rule: seeded RANDOM pair sample (NEVER a prefix) -- §4; n=96 noise band +/-7.3%
pose:        MANDATORY matched-base control, >=32 pairs (m85/#889); seg-only result INADMISSIBLE
```

**Gap denominator: 0.6189279. 1% of gap = 0.0061893 S.** Ceiling at 8-conn = 0.044175 S =
**7.137% of gap** (100% capture, which will not happen — it bounds the row, it does not predict it).

**Pre-registered outcomes, in RELATIVE units so they cannot drift with the baseline:**

| # | condition | typed outcome |
|---|---|---|
| F1 | realized n600 d_seg improves ≥ 1% of gap on A or B | `PROCEED` — sweep weight, then grammar + w_c races |
| F2 | improves, but < 1% of gap and inside the seed noise floor | `DEBT_NAMED(stage=scale, cure=weight sweep ×10 up)` |
| F3 | no d_seg change AND per-class `at_risk` counts unchanged | `DEBT_NAMED(stage=wiring/scale, cure=telemetry-first)` — an IMPLEMENTATION verdict, NOT a mechanism verdict |
| F4 | `at_risk` demonstrably falls but d_seg does not | closes the MECHANISM at `INSTANCE` — words are saved without paying d_seg, i.e. the ANNIHILATE channel is not where the S lives on this vehicle |
| F5 | `area` w_c policy ≥ `uniform` | falsifies the VOLUMETRIC premise as applied to the training force (never the cg1r depth-symmetry measurement itself) |
| F6 | d_seg improves but pose regresses on the matched base | `DEBT_NAMED(stage=pose collateral, cure=joint pose term)` — #889 confirmed, row not adoptable seg-only |

Per the realizability doctrine: **never `ROW_DEAD`, always `DEBT_NAMED(stage, cure)`.** Note F3
vs F4 is the load-bearing distinction — without the `at_risk` telemetry (emitted unconditionally
whenever the term is built) a null is uninterpretable between "didn't reach the words" and
"reached them and it didn't pay."

**FIRE CONDITION:** the scorer A/B fires when the n600 scorer slot is released by `sq1` AND a
pose-carrying base is available for the matched control. It does NOT fire on a seg-only base.

---

## §8 NEXT-IF-RESUMED (in order)

1. **Fire the §7 ticket** when the scorer slot frees — A/B `control` vs `A_lane_only` first
   (single class, cleanest attribution), then `B_birth_matrix`.
2. **Weight ladder.** 0.1 is a race START derived from the term's UNITS (hinge O(1-10) margin
   units vs `w_seg·seg_l` O(1)), not a measured optimum. Sweep before any adopt/kill verdict.
3. **Build the TR1 birth path** (§5) — reuse `island_protection.eased_island_masks` from TR1;
   do NOT port the retired trainer's seed/amplify flags.
4. **Grammar + w_c races** (`lever_existence_grammar`, `lever_existence_weight_policy`) — the
   `area` arm is adversarial by design and should be run, not assumed away.
5. **Registry-scope row** (§6) — fix `dsl_referenced_flags()` scope as its own change, with the
   sister-gate blast radius checked first.

Triality: DAG FEED + canonical-equation legs are OWED on the first MEASURED row (this arm has
no measured scorer row, so no equation is claimed). Storage:
`/Volumes/VertigoDataTier/pact/ddm_p4x_20260803/p4x_connectivity_control.json`.
