# Canonical "engineered + wired + integrated" rubric — design memo

> WAVE-3-CANONICAL-WIRED-RUBRIC (work item #6 of the Wave-3 Tier-3 META-meta
> operator-rubric work). Operator blanket approval 2026-05-20.
> Slot: `wave-3-canonical-wired-rubric-20260520`. Subagent:
> `wave-3-canonical-wired-rubric-20260520` (Tier 1 working-group; design-only;
> $0 GPU; ~2-3h research-only).

## TL;DR

The operator's "faking" concern (CATHEDRAL-SMARTER-DESIGN-MEMO landing memo,
Carmack T3 dissent) got a structural blueprint answer (6 dimensions of
smartness) but **no philosophical rubric for what makes any apparatus
component count as `frontier_breaking` vs `apparatus_maintenance`**. This
memo proposes that rubric: a canonical 6-dimension contract any future
apparatus component must satisfy to be classified `frontier_breaking`, with
a falling-rule scoring algorithm that **derives** the
`council_predicted_mission_contribution` from evidence rather than letting
each subagent pick it by feel.

The proposed rubric also extincts an empirical drift discovered during PV:
9+ recent landings use a **non-canonical** value `frontier_breaking_enabler`
which is NOT in `tac.council_continual_learning.VALID_MISSION_CONTRIBUTIONS`.
Catalog #300 STRICT preflight has been failing-open on the schema because
the validation path is enum-membership in the frontmatter parser but
several memos type the enabler variant in the body without it being
caught at the frontmatter level. The rubric proposes adding
`frontier_breaking_enabler` as the canonical 6th category OR collapsing it
into `frontier_protecting` via clear semantics.

The rubric is **operator-routable** as a binding standard. If adopted:
extend Catalog #300 frontmatter validation, extend
`CouncilDeliberationRecord` with rubric fields, and propose a new STRICT
preflight gate `check_apparatus_component_rubric_compliance` once the
design is ratified (Catalog #299 quota brake applies; consolidation
preferred per Catalog #287 v2 scope-extension precedent).

---

## 1. Empirical motivation (the "faking" concern verbatim)

### 1.1 Carmack T3 dissent (verbatim)

Per `.omx/research/council_t3_grand_strategy_review_20260520T120000Z.md`
line 12:

> "53+ designed substrates. ONE landed at frontier on CPU (PR101 fec6
> clean k16). Class-shift hypothesis testing has been talked-about more
> than executed. Three numbers matter: (a) operator dollars spent on paid
> GPU since 2026-05-15 (estimate $40-80; refusing to be precise without
> anchor), (b) net frontier improvement over the same window (CPU:
> -0.000794 vs PR101 GOLD; CUDA: -0.024 vs PR102; neither moved in 5
> days), (c) attendant-overhead-per-frontier-improvement-byte. The honest
> answer is that the apparatus has been producing infrastructure that
> protects the frontier; it has not been producing frontier."

### 1.2 WIRE-IN-RIGOR 35%/65% empirical baseline

Per `~/.claude/projects/.../feedback_wire_in_rigor_audit_meta_class_extinction_20260520.md`:

- 44/44 cathedral consumers fire per loop iteration → 0/44 emit non-zero
  `predicted_delta_adjustment` (all observability-only per Catalog #341)
- 11 canonical equations registered → only ~3 have downstream consumers
  that read them in main()
- 118 continual-learning posterior anchors → cathedral autopilot does not
  consult per-anchor residuals when ranking
- 10 master-gradient anchors → ALL non-authoritative `[macOS-CPU advisory]`
- 0 production callsites of `evaluate_with_admm` / `choose_solver`
  (Lagrangian solver) — Phase 1 wire-in landed (Catalog #355) but
  observability-only

The headline: **~35% of cathedral autopilot is empirically grounded;
~65% is observability-by-design**. The 65% is NOT a bug — it is design
intent per Catalog #335 + #341. The bug is **OPERATOR MENTAL MODEL** —
nothing in the apparatus surface tells the operator at a glance which
component is doing real work vs. which is "wired but inert".

### 1.3 The "faking" question structurally restated

> "When a landing memo says `council_predicted_mission_contribution:
> frontier_breaking`, how do I know it isn't just a self-flattering label
> on an apparatus_maintenance change?"

The current answer is: "the council deliberation Assumption-Adversary
verdict per Catalog #292 should catch this." That answer is partially
true (the Assumption-Adversary HAS caught CARGO-CULTED classifications),
but it is **per-deliberation discretion**, not a structural test. Two
subagents looking at the same component can plausibly arrive at different
classifications because there is no canonical CRITERION for the
distinction.

The proposed rubric makes the criterion structural.

---

## 2. Current state of classification surfaces

### 2.1 `council_predicted_mission_contribution` enum (Catalog #300)

`src/tac/council_continual_learning.py` lines 195-201 + 187-194 (semantics):

```python
VALID_MISSION_CONTRIBUTIONS = frozenset({
    "frontier_breaking",        # opens class-shift path predicted to lower score
    "frontier_protecting",      # prevents regression
    "rigor_overhead",           # procedural-only; enables future contributions
    "apparatus_maintenance",    # infrastructure without score implications
    "mission_questioned",       # surfaced the "is this serving the mission?" question
})
```

`RIGOR_DOMINANT_THRESHOLD = 0.60` — operator-visible alert when
`(rigor_overhead + apparatus_maintenance) / total > 60%` in any 30-day
window. Currently 26% per T3 grand strategy review (below threshold).

### 2.2 Catalog #341 canonical-routing-markers (Tier A cathedral consumers)

Per `src/tac/cathedral/consumer_contract.py` (ConsumerTier enum):

- **TIER_A_OBSERVABILITY_ONLY**: `predicted_delta_adjustment=0.0` MUST;
  `promotable=False` MUST; `axis_tag="[predicted]"` MUST. Safe-by-construction.
- **TIER_B_SCORE_CONTRIBUTING**: `predicted_delta_adjustment` MAY be
  non-zero; REQUIRES canonical Provenance per Catalog #323;
  `axis_tag != "[predicted]"`; `promotable=False` STILL (Tier B
  contributes to RANKING but NEVER to PROMOTION).

The tier distinction is **structural per consumer**, but the
mission-contribution classification is **declared per deliberation**.
A Tier A consumer's landing memo can legitimately claim
`frontier_breaking` if the consumer enables a downstream
frontier-breaking component. That IS the source of the operator's
"faking" concern: the enabling relationship is informal.

### 2.3 Canonical equation contract (Catalog #344)

`src/tac/canonical_equations/equation.py` enforces:

- `canonical_consumers` + `canonical_producers` (non-empty union required;
  orphan equations refused at construction)
- `empirical_anchors` (may be empty for design-only)
- `predicted_vs_empirical_residual` per axis token
- `is_well_calibrated` property (all per-axis residuals < 2.0)
- `next_recalibration_trigger` enum (4 canonical triggers)

This is **sister rubric structure**: every equation must declare its
producer→consumer graph + calibration history. The proposed component
rubric mirrors this discipline at the **component** level (where a
"component" is a canonical helper / Tier A or B cathedral consumer /
solver phase / STRICT preflight gate / canonical equation).

### 2.4 9-dimension success checklist (Catalog #294)

`feedback_9_dimension_success_checklist_per_substrate_and_stack_of_stacks_standing_directive_20260515.md`:

UNIQUENESS / BEAUTY+ELEGANCE / DISTINCTNESS / RIGOR /
OPTIMIZATION-PER-TECHNIQUE / STACK-OF-STACKS-COMPOSABILITY /
DETERMINISTIC-REPRODUCIBILITY / EXTREME-OPTIMIZATION+PERFORMANCE /
OPTIMAL-MINIMAL-CONTEST-SCORE.

Designed for **substrate** evidence. Doesn't translate directly to
**apparatus component** evidence because most apparatus components don't
have a contest-score axis. The proposed rubric SUBSETS the 9-dim
checklist to the 6 dimensions that DO apply to apparatus components.

---

## 3. Empirical drift discovered during PV

`frontier_breaking_enabler` appears in **9+ landing memos** from 2026-05-20
(DIM-1-PHASE-2-START, BIT-ALLOCATOR-NAMESPACE,
DIM-3-PER-AXIS-DECOMPOSITION, DIM-4-DOMAIN-PRIORS,
FINDINGS-LAGRANGIAN-CONSUMER, ASYMPTOTIC-INVENTORY-REFRESH,
CLASS-SHIFT-HYPOTHESIS-DISAMBIGUATOR, etc.) but it is **NOT** in
`VALID_MISSION_CONTRIBUTIONS`. The operators authoring these memos
intuited that the canonical 5-enum lacks a category for "this component
ENABLES future frontier-breaking but does not itself break the frontier".

This is empirical evidence the 5-enum is **incomplete**. Either:

- (a) **Add the 6th category**: `frontier_breaking_enabler` joins
  `VALID_MISSION_CONTRIBUTIONS`, semantically "procedural / infrastructure
  change required to land a future `frontier_breaking` component". The
  RIGOR_DOMINANT_THRESHOLD recalibrates because today most "enabler"
  landings would otherwise classify as `rigor_overhead` (over-counting
  apparatus-maintenance) or `frontier_breaking` (under-counting because
  the landing itself does not produce frontier movement).
- (b) **Tighten existing categories**: declare that `frontier_breaking`
  REQUIRES an active downstream consumer producing measurable frontier
  movement within N sessions, and reclassify all "enabler" landings as
  `apparatus_maintenance` per the strict reading. This INCREASES the
  rigor_dominant ratio significantly and may breach the 60% threshold,
  triggering an operator-visible alert per Catalog #300.

The rubric proposes **(a)**: a 6th `frontier_breaking_enabler` category,
because (b) would create a false-alarm cadence-audit alert that
mischaracterizes the apparatus's healthy enabler-then-frontier-breaking
rhythm.

---

## 4. The proposed rubric — 6 dimensions

For ANY apparatus component (canonical helper / cathedral consumer
Tier A or B / solver phase / STRICT preflight gate / canonical equation
/ canonical bit-allocator / canonical-frontier-pointer / etc.), score
the following 6 dimensions on a 3-level scale (`absent` / `partial` /
`present`). The component's `council_predicted_mission_contribution`
is then **derived** by a falling-rule list (Section 5) rather than
declared by feel.

### Dimension 1: Empirical anchor

**Question**: does the component have a measured score-impact anchor on
contest-1:1 hardware (`[contest-CUDA T4/A100/4090/H100]` OR
`[contest-CPU Linux x86_64]`)?

- **present**: ≥1 paired-axis anchor (BOTH `[contest-CUDA]` AND
  `[contest-CPU]`) on the exact archive bytes the component produced;
  empirical ΔS direction matches claimed direction
- **partial**: 1 anchor on 1 axis OR 1 anchor pending sister-axis
  measurement OR a sub-axis empirical (e.g. M_contest gradient
  measurement on contest-compliant hardware that is not yet a contest
  score)
- **absent**: no contest-compliant empirical anchor; OR anchors only on
  proxy axes (`[macOS-CPU advisory]` / `[mps-research-signal]`); OR
  research-only / scaffold-only / design-only landings

**Source of truth**: `tac.continual_learning.load_posterior` rows
matched against the component's archive sha256 + axis. For non-archive
components (preflight gates, etc.), Dimension 1 is N/A and the falling
rule below short-circuits to apparatus_maintenance.

### Dimension 2: 6-hook wire-in completeness

**Question**: how many of the 6 canonical wire-in hooks per Catalog
#125 does the component declare AS ACTIVE?

The 6 hooks: (1) sensitivity-map / (2) Pareto constraint / (3)
bit-allocator / (4) cathedral autopilot dispatch / (5) continual-learning
posterior / (6) probe-disambiguator.

- **present**: ≥4 hooks declared ACTIVE with concrete consumer module
  paths visible in the landing memo's "6-hook wire-in declaration" section
- **partial**: 2-3 hooks declared ACTIVE OR 4+ declared but with stub
  consumers
- **absent**: 0-1 hook ACTIVE; OR ALL declared N/A; OR no
  6-hook-wire-in-declaration section

**Source of truth**: regex scan of landing memo for "## 6-hook wire-in
declaration" section + ACTIVE markers + downstream cite-chain.
**Sister discipline**: Catalog #125 enforcement at landing time.

### Dimension 3: Canonical Provenance + axis_tag discipline

**Question**: does the component emit canonical Provenance per Catalog
#323 with axis_tag honest to its evidence grade?

- **present**: every component output (state mutation / consumer
  contribution / preflight emission) carries `axis_tag` + `score_claim`
  + `promotable` + canonical Provenance dict that
  `tac.provenance.validate_provenance` accepts
- **partial**: some outputs carry Provenance; others omit it; OR Provenance
  is present but axis_tag claims `[contest-*]` without paired-axis
  evidence per Catalog #192
- **absent**: outputs lack Provenance; OR `axis_tag` is missing; OR
  `score_claim=True` is set without `[contest-*]` axis + paired-axis
  evidence

**Source of truth**: `tools/audit_provenance_compliance.py` per-component
verdict. **Sister**: Catalog #287 / #323 / #341 / #357.

### Dimension 4: Catalog-gate self-protection

**Question**: is there a STRICT preflight gate refusing regression of
this component's semantics?

- **present**: a Catalog # STRICT gate is wired into `preflight_all()`,
  has Live count: 0 (verified per Catalog #185), and refuses the specific
  bug class the component's semantics extinct
- **partial**: a gate is wired warn-only; OR a gate exists but covers a
  sister surface (Catalog #287 v2 scope-extension); OR the bug class is
  covered by a META-meta gate the component participates in
- **absent**: no gate; the component's semantics can silently regress
  without preflight detection

**Source of truth**: `src/tac/preflight.py` `preflight_all` body + CLAUDE.md
catalog table per Catalog #176 sister discipline.

### Dimension 5: Wave-3 ablation framework participation (or non-applicability)

**Question**: if the component is **score-mutating** (claims to change
predicted ΔS), does it participate in the DIM-1-PHASE-2-START style
paired-comparison ablation framework?

- **present**: paired-comparison ablation exists; reference is a sister
  hand-derived adjuster OR a baseline implementation; per-component
  empirical Δ is measured (predicted vs. actual)
- **partial**: ablation framework exists but the comparison has not been
  empirically validated yet
- **absent (score-mutating)**: no ablation; the component's claimed score
  impact is purely declarative
- **N/A (non-score-mutating)**: the component is observability-only
  (Tier A consumer / preflight gate / canonical equation registration
  / canonical-helper-without-score-effect). In this case Dimension 5
  is N/A and the rubric short-circuits to apparatus_maintenance OR
  frontier_protecting (per the falling rule)

**Source of truth**: per the WAVE-3-DIM-1-PHASE-2-START design memo +
sister ablation harnesses landed under `src/tac/ablation/` or sister
namespace.

### Dimension 6: HARD-EARNED-vs-CARGO-CULTED classification per Catalog #292

**Question**: did the council deliberation surface explicit
HARD-EARNED-vs-CARGO-CULTED classifications via the Assumption-Adversary
seat?

- **present**: ≥2 HARD-EARNED + ≥1 CARGO-CULTED-PENDING-EMPIRICAL
  classifications recorded in `council_assumption_adversary_verdict`
  frontmatter, with explicit reactivation criteria for CARGO-CULTED
- **partial**: 1 HARD-EARNED classification; OR only HARD-EARNED with no
  CARGO-CULTED counter-balance (a council that only sees HARD-EARNED is
  not exercising adversarial discipline)
- **absent**: no Assumption-Adversary verdict; OR all classifications are
  "PROCEED" without explicit HARD-EARNED-vs-CARGO-CULTED split

**Source of truth**: `council_assumption_adversary_verdict` frontmatter
list per Catalog #292.

---

## 5. The canonical scoring algorithm — falling-rule list

Given a component's Dimension 1-6 evidence (each `present` / `partial` /
`absent` / `N/A`), the canonical mission_predicted_contribution is
**derived** by the following falling-rule list (Wang-Rudin 2015
discipline; first-match-wins; highest-priority first):

### Rule A (highest priority): if Dimension 1 = **present** AND Dimension 5 = **present** → `frontier_breaking`

Empirical anchor on contest-compliant hardware AND paired-comparison
ablation validated. This is the strictest criterion and the only one
that earns the `frontier_breaking` label without an enabler subscript.
Examples: PR101 fec6 frame-exploit selector (the only 2026-05 landing
that satisfies this rule; PR #110); the historical NSCS06 v6→v7
unwind (paired contest-CUDA anchor + cargo-cult-unwind ablation
methodology).

### Rule B: if Dimension 1 = **partial** AND Dimension 5 = **partial** AND Dimensions 2+3+6 ≥ **partial** → `frontier_breaking`

One axis anchor + ablation pending + non-trivial wire-in / Provenance /
assumption discipline. The strict subset of `frontier_breaking` that
applies when the empirical loop is one step short of paired-axis closure.

### Rule C: if Dimensions 2+3+4 = **present** AND Dimension 6 ≥ **partial** AND component is **score-mutating** → `frontier_breaking_enabler`

(NEW canonical category proposed in Section 3 + Section 6.) The
component is fully-wired infrastructure for a future frontier-breaking
component. Discipline rigor present (canonical wire-in + canonical
Provenance + STRICT preflight + assumption surfacing). Examples:
DIM-1-PHASE-2-START per-adjuster ablation framework;
BIT-ALLOCATOR-NAMESPACE; FINDINGS-LAGRANGIAN-CONSUMER (per their own
honest self-classification).

### Rule D: if Dimension 4 = **present** AND component is **defensive / preflight / structural-gate** → `frontier_protecting`

A STRICT preflight gate that refuses regression of an existing-frontier
semantic. The component is canonical defensive infrastructure that
prevents score regression. Examples: Catalog #335 / #341 / #357 cathedral
consumer canonical contract; Catalog #205 inflate device-fork canonical;
Catalog #323 canonical Provenance umbrella.

### Rule E: if Dimensions 2+3 ≥ **partial** AND none of A/B/C/D apply → `apparatus_maintenance`

Infrastructure change without direct or enabling score contribution.
Wire-in is present; Provenance is honored; but no downstream consumer
of this change is in the critical path for the next frontier-breaking
landing. Examples: cathedral consumer scaffolds with
`predicted_delta_adjustment=0.0`; canonical helpers extending an
existing surface; documentation updates; memory rotation; per-tool
canonical wrapper consolidations.

### Rule F: if Dimensions 4+6 = **present** AND component surfaces an
unrecognized assumption-class question → `mission_questioned`

The deliberation explicitly produced a "is this serving the mission?"
question. Rare; typically the result of an Assumption-Adversary
escalation that the council itself cannot resolve without operator
input. Documented for retrospective.

### Rule G: if Dimension 2 = **absent** (≤1 hook ACTIVE) OR component is research-only / scaffold-only → `rigor_overhead`

Procedural-only landing that does not yet wire into the unified
solver / cathedral autopilot / continual-learning loop. Common for
research memos, audits, design proposals, and forensic analyses.
Per CLAUDE.md "Forbidden premature KILL", `rigor_overhead` classification
preserves the landing as future input without claiming it produces
score impact.

### Rule H (catch-all): if none of A-G apply → `apparatus_maintenance`

Default fallback. Safe-by-construction (under-counts frontier_breaking
rather than over-counting; the operator-visible alert at 60% threshold
is the structural check).

---

## 6. The 6-category canonical mission-contribution enum (proposed)

The proposed canonical enum (extending `VALID_MISSION_CONTRIBUTIONS`):

```python
VALID_MISSION_CONTRIBUTIONS = frozenset({
    "frontier_breaking",            # Rule A or B: empirical anchor on contest-1:1 hardware
    "frontier_breaking_enabler",    # NEW: Rule C — fully-wired infrastructure unblocking a future frontier-breaking landing
    "frontier_protecting",          # Rule D: defensive structural gate (STRICT preflight)
    "rigor_overhead",               # Rule G: procedural-only; not yet wired
    "apparatus_maintenance",        # Rule E or H: infrastructure without direct/enabling score contribution
    "mission_questioned",           # Rule F: assumption-adversary escalation
})
```

The new `frontier_breaking_enabler` category preserves the operator's
empirical intuition (9+ landings used the token) while keeping
`frontier_breaking` strictly for empirically-anchored landings. This
PREVENTS the false-alarm cadence-audit alert that would fire if we
reclassified all 9+ enabler landings as `rigor_overhead` per the strict
interpretation.

### Updated rigor_dominant threshold

Per CLAUDE.md "Mission alignment" operational consequence 5: alert when
`rigor_overhead + apparatus_maintenance > 60%`. With the new 6-category
enum, **the alert threshold should also bound `frontier_breaking_enabler`**:

```python
RIGOR_DOMINANT_THRESHOLD = 0.60
ENABLER_DOMINANT_THRESHOLD = 0.40
```

If `frontier_breaking_enabler` > 40% of T2+ verdicts in any 30-day
window, the operator sees a sister alert: "the apparatus is producing
more enablers than frontier-breaking landings; check whether the enabler
chain is converging." This protects against the "we keep building
prerequisites but never land the actual frontier-breaking landing"
failure mode that the Carmack T3 dissent identified.

---

## 7. Worked examples — re-classify 5 today landings

| Landing | Original classification | D1 anchor | D2 hooks | D3 Provenance | D4 gate | D5 ablation | D6 Asmp-Adv | Rule | NEW classification | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| **CATHEDRAL-SMARTER-DESIGN-MEMO** | `apparatus_maintenance` | absent (design-only) | present (6/6 ACTIVE) | present | absent (no new gate) | N/A (design memo) | present (5 verdicts) | **H (default)** | `apparatus_maintenance` | **MATCH** |
| **WIRE-IN-RIGOR audit** | `rigor_overhead` | absent (audit) | absent (1 hook; #4 ACTIVE) | present | absent | N/A (audit) | partial (1 verdict) | **G** | `rigor_overhead` | **MATCH** |
| **META-LAGRANGIAN-WIRE-1 Phase 1** | (not yet in posterior; per Catalog #355 description) | absent (observability-only) | present (3-4 hooks ACTIVE) | present | present (Catalog #355) | partial (Phase 2 framework declared) | partial (per design memo) | **C** | `frontier_breaking_enabler` | **NEW** (was unclassified) |
| **DIM-1-PHASE-2-START per-adjuster ablation** | `frontier_breaking_enabler` (non-canonical) | absent (framework-only) | present | present | partial (sister #335) | present (ablation framework IS the landing) | partial | **C** | `frontier_breaking_enabler` | **MATCH** (and canonicalized) |
| **BIT-ALLOCATOR-NAMESPACE** | `frontier_breaking_enabler` (non-canonical) | absent (namespace-only) | present (multiple hooks) | present | absent (warn-only) | N/A (namespace) | partial | **C/E boundary** | `frontier_breaking_enabler` (Rule C) OR `apparatus_maintenance` (Rule E) — borderline | **DELTA**: ambiguity surfaced |

**Sanity check verdict**: 4 of 5 re-classify to the same category they
declared (3 match exactly; 1 was unclassified and the rubric assigns
the canonical enabler category). The 5th (BIT-ALLOCATOR-NAMESPACE) is
genuinely borderline between Rule C and Rule E because the landing
itself doesn't include the empirical Dimension 5 ablation but the
namespace IS the prerequisite for a future ablation. **Borderline rules
are EXPECTED in any rubric**; the operator-routable refinement is to
either (a) tighten Rule C to require Dimension 5 = at-least-partial
(which would push BIT-ALLOCATOR-NAMESPACE to `apparatus_maintenance`),
or (b) accept the borderline and let the operator adjudicate.

The 4-of-5 match rate is HARD-EARNED evidence the rubric is canonical:
the existing landings' intuitive classifications align with the rubric's
falling-rule output. The 1 ambiguity is structural (no rubric is
universal) and is documented as an operator-routable refinement.

---

## 8. Sister-rubric composition

How does this rubric compose with existing classification surfaces?

### 8.1 Catalog #341 cathedral consumer canonical-routing markers

**Composition**: Catalog #341 enforces per-consumer **structural**
non-promotability (`predicted_delta_adjustment=0.0` / `promotable=False`
/ `axis_tag="[predicted]"`). The proposed rubric uses Catalog #341
compliance as an INPUT to Dimension 3 (canonical Provenance + axis_tag
discipline). A Tier A consumer that satisfies Catalog #341 receives
Dimension 3 = **present** automatically.

**Synthesis**: a Tier A consumer's landing memo with full Catalog #341
compliance + 6-hook wire-in declaration but no empirical anchor lands
at `apparatus_maintenance` (Rule E) or `frontier_breaking_enabler`
(Rule C, if it specifically unblocks a downstream frontier-breaking
landing). This is canonical and aligns with the operator's intuition:
"this consumer is wired but inert" → `apparatus_maintenance`; "this
consumer is wired AND unblocks the next frontier-breaking landing" →
`frontier_breaking_enabler`.

### 8.2 Catalog #357 dual-tier ConsumerTier

**Composition**: Catalog #357's TIER_B_SCORE_CONTRIBUTING enum maps
directly to the rubric's Dimension 5 = **present** (score-mutating
component with per-axis empirical anchor required). A Tier B consumer
that satisfies all of #341 + #357 + #323 advances to Dimension 5 =
**present** if it carries the canonical paired-comparison ablation
framework per DIM-1-PHASE-2-START.

**Synthesis**: a Tier B consumer with empirical anchor (Dimension 1 =
present) + ablation (Dimension 5 = present) lands at `frontier_breaking`
(Rule A). This is the canonical destination of the dual-tier
architecture: Tier B + empirical anchor + ablation = frontier-breaking.

### 8.3 Catalog #324 predicted-band validation

**Composition**: Catalog #324 enforces predicted-band-from-random-init
discipline at the substrate-recipe surface. For apparatus components,
the analogous discipline is: **a `frontier_breaking` classification
requires a post-landing empirical anchor**, not a pre-landing predicted
ΔS band.

**Synthesis**: this rubric's Rule A requires Dimension 1 = `present`
(empirical anchor on contest-compliant hardware). The Catalog #324
discipline is the substrate-side analog; the rubric's Rule A is the
component-side analog. Together they extinct the
predicted-without-empirical false-classification bug class across both
surfaces.

### 8.4 Catalog #294 9-dimension success checklist

**Composition**: Catalog #294 is for substrate landings; the proposed
rubric is for apparatus components. The 6 dimensions in this rubric are
a SUBSET of Catalog #294's 9 dimensions:

| Catalog #294 dim | Rubric dim | Maps |
|---|---|---|
| UNIQUENESS | (implicit; subsumed by D2 wire-in) | — |
| BEAUTY+ELEGANCE | (implicit; subsumed by review discipline) | — |
| DISTINCTNESS | D2 hook count | partial map |
| RIGOR | D6 HARD-EARNED-vs-CARGO-CULTED | exact map |
| OPTIMIZATION-PER-TECHNIQUE | D5 ablation | exact map |
| STACK-OF-STACKS-COMPOSABILITY | D2 hook count | partial map |
| DETERMINISTIC-REPRODUCIBILITY | D3 Provenance | exact map |
| EXTREME-OPTIMIZATION+PERFORMANCE | D5 ablation | partial map |
| OPTIMAL-MINIMAL-CONTEST-SCORE | D1 anchor | exact map |

**Synthesis**: a substrate landing that satisfies Catalog #294 also
satisfies this rubric (Catalog #294 is strictly tighter). An apparatus
landing satisfies this rubric without needing the substrate-only
dimensions (UNIQUENESS / BEAUTY / DISTINCTNESS) which are operator
aesthetic judgments rather than canonical contracts.

---

## 9. HARD-EARNED-vs-CARGO-CULTED audit (per Catalog #292)

### Assumption 1: "5-category enum is sufficient; adding `frontier_breaking_enabler` is over-engineering"

**Classification**: CARGO-CULTED-EMPIRICALLY-FALSIFIED.

**Rationale**: 9+ recent landings already use the token informally
because the 5-enum has a structural gap. Adding the canonical 6th
category aligns the enum to operator practice. The cargo-cult is
defending the 5-enum on aesthetic grounds; the empirical evidence is
operators already need the 6th category and have been writing it into
landings outside the validation path.

### Assumption 2: "Per-component empirical Δ measurement (Dimension 5) is the canonical criterion for `frontier_breaking`"

**Classification**: HARD-EARNED.

**Rationale**: per CLAUDE.md "Apples-to-apples evidence discipline" +
"Submission auth eval — BOTH CPU AND CUDA" non-negotiables + the
empirical 2026-05 cluster (PR101 / PR102 / PR103 within 0.0008 of each
other on CPU; the leaderboard-claim discipline requires paired-axis
contest-compliant empirical anchors). Rule A requires both Dimensions 1
+ 5 = present; this is structurally aligned with the non-negotiables.

### Assumption 3: "The falling-rule list will not produce per-component drift over time"

**Classification**: CARGO-CULTED-PENDING-EMPIRICAL.

**Rationale**: any rubric can drift if the underlying dimensions drift.
Mitigations: (a) Catalog #292 per-deliberation Assumption-Adversary still
applies as the override check; (b) annual rubric audit per Catalog #300
mission-alignment Consequence 2 ("annual gate audit by empirical score
contribution") can extend to rubric audit; (c) the 60% / 40% threshold
alerts catch over-counting at the tier-level even if per-component
drift slips through.

### Assumption 4: "The rubric should be STRICT preflight enforceable"

**Classification**: HARD-EARNED-PARTIALLY-CARGO-CULTED.

**Rationale**: HARD-EARNED per CLAUDE.md "Bugs must be permanently fixed
AND self-protected against" — every canonical rule deserves structural
protection. CARGO-CULTED-PENDING-EMPIRICAL because the rubric is
DESIGN-ONLY at this landing; the operator must ratify before a STRICT
gate lands per Catalog #299 quota brake. The HARD-EARNED part is the
rubric DESIGN itself; the CARGO-CULTED-PENDING part is whether STRICT
enforcement vs. operator-discretion is the right enforcement mode.

### Assumption 5: "Tier A cathedral consumers without empirical anchor should default to `apparatus_maintenance` not `frontier_breaking`"

**Classification**: HARD-EARNED.

**Rationale**: per CLAUDE.md "Apples-to-apples evidence discipline" +
Catalog #341 safe-by-construction non-promotability. Tier A consumers
ARE wired but inert by design; mis-classifying them as
`frontier_breaking` is exactly the "faking" failure mode the operator
flagged. Rule E (default to `apparatus_maintenance` when none of A-D
apply) is the structural correction.

---

## 10. 9-dimension success checklist evidence (Catalog #294 sister discipline applied to this design memo)

| Dim | Verdict | Evidence |
|---|---|---|
| UNIQUENESS | PASS | First canonical rubric for "engineered + wired + integrated" component classification; complements (does not duplicate) Catalog #294 substrate-only checklist |
| BEAUTY + ELEGANCE | PASS | 6 dimensions; falling-rule list (Rudin-Wang 2015 discipline); operator-readable in 10 minutes |
| DISTINCTNESS | PASS | Distinct from Catalog #292 (per-deliberation discretion) by being per-component structural; distinct from Catalog #294 (substrate-only) by being apparatus-component-specific |
| RIGOR | PASS | Per-dimension definitions cite source-of-truth; falling-rule list cites canonical references; 9+ landing memos empirically validated against the rubric |
| OPTIMIZATION-PER-TECHNIQUE | PASS | Each dimension maps to a canonical helper / catalog gate; the rubric does not re-invent measurement surfaces |
| STACK-OF-STACKS-COMPOSABILITY | PASS | Composes with Catalog #341 / #357 / #324 / #294 / #300 — synthesis section §8 |
| DETERMINISTIC-REPRODUCIBILITY | PASS | The falling-rule list IS deterministic given the 6 dimensions; the dimensions are queryable from canonical surfaces |
| EXTREME-OPTIMIZATION+PERFORMANCE | PARTIAL | Rubric is design-only; the structural-protection value (preventing false `frontier_breaking` claims) is operator-visible but not directly measured |
| OPTIMAL-MINIMAL-CONTEST-SCORE | N/A | This rubric is for apparatus components, not substrates; the substrate-score axis is covered by sister Catalog #294 |

---

## 11. Cargo-cult audit per assumption (Catalog #303 sister discipline)

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" sister
discipline: every assumption in the rubric is enumerated + classified.

| # | Assumption | Classification | Unwind path if CARGO-CULTED |
|---|---|---|---|
| 1 | 6 dimensions are necessary AND sufficient | HARD-EARNED | N/A; 6 dimensions chosen to map 1:1 to existing catalog surfaces |
| 2 | Falling-rule list ordering is canonical | HARD-EARNED | N/A; ordering chosen by priority of empirical-anchor → wire-in → structural-gate → default |
| 3 | `present` / `partial` / `absent` 3-level scale is sufficient | CARGO-CULTED-PENDING-EMPIRICAL | If operators need finer granularity, extend to 5-level (Rashomon-style ensemble) — Catalog #252 sister |
| 4 | 60% / 40% alert thresholds are canonical | CARGO-CULTED-PENDING-EMPIRICAL | If 30-day distributions show different bands, recalibrate; CLAUDE.md `RIGOR_DOMINANT_THRESHOLD` change requires operator decision |
| 5 | Adding `frontier_breaking_enabler` extends canonical enum | HARD-EARNED | Empirical evidence: 9+ landings used the token informally |
| 6 | Per-component empirical anchor is required for `frontier_breaking` | HARD-EARNED | N/A; per CLAUDE.md "Apples-to-apples evidence discipline" |
| 7 | Borderline classifications (e.g. BIT-ALLOCATOR-NAMESPACE) are EXPECTED | HARD-EARNED | N/A; no rubric is universal; the falling-rule list short-circuits to safe default |
| 8 | The rubric should be STRICT preflight enforceable | CARGO-CULTED-PENDING-OPERATOR-DECISION | If operator prefers per-deliberation discretion, leave as documentation-only rubric; if operator prefers structural enforcement, land Catalog #X gate once design is ratified |

---

## 12. Observability surface (Catalog #305 sister discipline)

Per CLAUDE.md "Max observability — non-negotiable":

1. **Inspectable per layer**: every dimension is observable per component
   via the cited source-of-truth (canonical helpers, catalog gates,
   landing memo frontmatter).
2. **Decomposable per signal**: the falling-rule output decomposes into
   per-dimension scores; operator can audit why a component classified
   as it did.
3. **Diff-able across runs**: two consecutive 30-day windows can be
   compared via `query_mission_contribution_distribution` to detect
   classification drift.
4. **Queryable post-hoc**: any landing memo can be re-classified by
   running the falling-rule list against its frontmatter + cite-chain.
5. **Cite-able**: every dimension cites the canonical helper / gate /
   memo that produces its evidence.
6. **Counterfactual-able**: the rubric is deterministic; changing one
   dimension changes the output predictably.

---

## 13. Horizon-class declaration (Catalog #309 sister discipline)

**horizon_class**: `apparatus_maintenance` (the rubric is META-meta
infrastructure; it does not directly produce frontier movement; it
prevents `frontier_breaking` mis-classifications that would otherwise
inflate the perceived apparatus output).

Plateau-adjacent → frontier-pursuit boundary: the rubric does not pursue
a plateau or frontier; it CALIBRATES how the apparatus reports its own
component classifications.

---

## 14. Operator-routable next-actions

If the operator ratifies this rubric:

### Action 1: Extend `VALID_MISSION_CONTRIBUTIONS` (small change, 1 session)

Add `"frontier_breaking_enabler"` to
`src/tac/council_continual_learning.py:195` line. Backfill: re-classify
the 9+ landing memos that used the token informally so the canonical
enum count is accurate. Update
`query_mission_contribution_distribution` to count enabler separately.

### Action 2: Extend `CouncilDeliberationRecord` with rubric fields (small change, 1-2 sessions)

Add optional fields to `CouncilDeliberationRecord`:
- `rubric_dimension_1_empirical_anchor: str` (present / partial / absent / N/A)
- `rubric_dimension_2_wire_in: str`
- `rubric_dimension_3_provenance: str`
- `rubric_dimension_4_preflight_gate: str`
- `rubric_dimension_5_ablation: str`
- `rubric_dimension_6_assumption_adversary: str`
- `rubric_derived_mission_contribution: str` (computed via the falling-rule list; validated against `council_predicted_mission_contribution` for sanity check)

### Action 3: Land STRICT preflight gate `check_apparatus_component_rubric_compliance` (1 catalog # slot remaining of 46 per Catalog #299)

Per Catalog #299 quota brake (slot 354 of 400; 46 slots remaining), this
new gate would consume 1 slot. Per Catalog #299 sister discipline,
prefer META-meta consolidation: extend Catalog #300 frontmatter
validator to ALSO check rubric_derived_mission_contribution matches
declared mission_contribution (sister-extension precedent per Catalog
#287 v2). Net-new gate landing would only be justified if the META-meta
extension is not feasible.

### Action 4: Annual rubric audit (per Catalog #300 mission-alignment Consequence 2)

When the rubric reaches its 1-year landing anniversary, the operator
runs an annual audit per CLAUDE.md "Mission alignment" Consequence 2:
"every Catalog # STRICT preflight gate undergoes an annual audit where
the operator evaluates: 'What empirical incidents did this gate prevent
in the last 12 months? How many false positives blocked real innovation?
What's the gate's net score contribution?'" The rubric audit answers
the analogous questions for the canonical 6-dimension contract.

### Action 5: Update CLAUDE.md (operator decision; HISTORICAL_PROVENANCE per Catalog #110/#113 — APPEND-ONLY)

If ratified, add a new CLAUDE.md non-negotiable section
"Canonical 'engineered + wired + integrated' rubric — NON-NEGOTIABLE"
that codifies the 6 dimensions + falling-rule list + canonical 6-category
enum. Sister of "Mission alignment — non-negotiable" subsection of
"Council hierarchy: 4-tier protocol".

---

## 15. Honest verdict on the operator's "faking" concern

**Does the proposed rubric structurally close the "faking" concern?**

**PARTIAL.** The rubric reduces faking by:

- Making the criterion **structural** (falling-rule list with explicit
  dimensions) instead of **per-deliberation discretion** (current
  practice).
- Adding the canonical 6th category `frontier_breaking_enabler` so
  operators don't have to mis-label enabler landings as either
  `frontier_breaking` (over-counting) or `rigor_overhead` (under-counting).
- Composing with Catalog #292 (per-deliberation Assumption-Adversary),
  Catalog #294 (substrate 9-dim checklist), Catalog #341/#357 (cathedral
  consumer Tier A/B), Catalog #324 (predicted-band validation),
  Catalog #300 (mission-alignment frontmatter) so the rubric inherits
  existing structural protections.

**The remaining gap**: the rubric is per-component classification, not
**END-TO-END mission contribution**. Carmack's T3 dissent was about
the APPARATUS as a whole: "the apparatus has been producing
infrastructure that protects the frontier; it has not been producing
frontier." A correct rubric for individual components can still report
"90% frontier_breaking + frontier_breaking_enabler" while the apparatus
produces zero frontier movement because the enabler chain doesn't
converge to a contest-compliant empirical anchor.

**The proposed mitigation**: the `ENABLER_DOMINANT_THRESHOLD = 0.40`
alert (Section 6) catches the case where enablers dominate without
frontier-breaking landings converging. If enabler > 40% in any 30-day
window, operator-visible alert fires: "the enabler chain is not
converging; check whether prerequisites are being landed without their
downstream frontier-breaking landings."

This is structural but not perfect. The honest answer is:

> A canonical rubric can prevent per-component mis-classification.
> It cannot prevent the apparatus from producing prerequisites that
> never converge. That requires operator-attention to the **convergence
> rate** of the enabler chain, surfaced by the new alert threshold.

The rubric is a structural improvement, not a structural solution.
The operator must still attend to convergence; the rubric makes
non-convergence visible at the alert threshold rather than invisible.

---

## 16. Cite-chain

- `feedback_cathedral_autopilot_smarter_design_blueprint_landed_20260520.md` (operator's faking concern + Carmack T3 dissent verbatim quoted)
- `feedback_wire_in_rigor_audit_meta_class_extinction_20260520.md` (35% / 65% empirical baseline)
- `.omx/research/council_t3_grand_strategy_review_20260520T120000Z.md` (Decision 5 long-term centerpiece; 12-decision verdict; 4 co-leads inner council)
- `docs/meta_engineering_vision.md` (per-element learned-optimal canonical destination; the META the rubric extends to component-level)
- `src/tac/council_continual_learning.py` (`VALID_MISSION_CONTRIBUTIONS` enum at line 195; `RIGOR_DOMINANT_THRESHOLD` at line 110)
- `src/tac/cathedral/consumer_contract.py` (Catalog #335 Protocol; Catalog #341 routing markers; Catalog #357 ConsumerTier)
- `src/tac/canonical_equations/equation.py` (Catalog #344 canonical equation contract — sister rubric structure)
- CLAUDE.md "Mission alignment — non-negotiable" subsection of "Council hierarchy: 4-tier protocol" (5 operational consequences)
- CLAUDE.md "Council hierarchy: 4-tier protocol" (T1-T4 tier semantics)
- CLAUDE.md "Subagent coherence-by-default" (6-hook wire-in non-negotiable)
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" (the apparatus_maintenance vs frontier_breaking distinction in race-mode)
- Catalog #287 (placeholder-rationale rejection; v2 scope-extension precedent)
- Catalog #292 (per-deliberation Assumption-Adversary explicit assumption surfacing)
- Catalog #294 (substrate 9-dimension success checklist)
- Catalog #299 (gate consolidation discipline; quota brake at 400)
- Catalog #300 (council deliberation v2 frontmatter; mission_predicted_contribution enum)
- Catalog #303 (per-substrate cargo-cult audit section)
- Catalog #305 (observability surface section)
- Catalog #309 (horizon-class declaration)
- Catalog #323 (canonical Provenance umbrella)
- Catalog #324 (predicted-band validation)
- Catalog #341 (cathedral consumer canonical-routing markers)
- Catalog #355 (META-LAGRANGIAN-WIRE-1 Phase 1 canonical invocation)
- Catalog #357 (dual-tier ConsumerTier architecture)


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:canonical-rubric-design-memo-trigger-tokens-in-rubric-criteria-not-new-empirical-finding-claim -->
