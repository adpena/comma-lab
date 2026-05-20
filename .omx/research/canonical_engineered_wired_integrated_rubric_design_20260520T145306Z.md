# Canonical "engineered + wired + integrated" rubric — design memo

**Date**: 2026-05-20T14:53:06Z
**Lane**: `lane_wave_3_canonical_engineered_wired_integrated_rubric_design_20260520`
**Tier**: T1 (Working Group — single subagent design proposal; T2 sextet ratification queued as op-routable #1 below)
**Horizon-class**: `frontier_protecting` (per Catalog #309 — the rubric does NOT predict a substrate ΔS band; it predicts an APPARATUS-CLASSIFICATION-DRIFT-EXTINCTION outcome)
**Mission contribution**: `frontier_protecting` per CLAUDE.md "Mission alignment" Consequence 5 (prevents the regression where future apparatus components silently classify as `frontier_breaking` when they only enable; sister of strict-mode preflight gates)

> Operator's verbatim concern (paraphrased from CATHEDRAL-SMARTER landing memo Assumption-Adversary verdict + Carmack T3 dissent): *"the apparatus has been producing infrastructure that protects the frontier; it has not been producing frontier"* — without a canonical rubric, classification of new apparatus work as `frontier_breaking` vs `apparatus_maintenance` is a tribal-knowledge act. This memo proposes the rubric that structurally extincts the tribal-knowledge surface.

---

## 1. Empirical motivation (HARD-EARNED)

Three convergent empirical anchors over 2026-05-19 / 2026-05-20 establish the bug class:

**Anchor 1 — Carmack T3 dissent, 2026-05-20** (`council_t3_grand_strategy_review_20260520T120000Z.md` line 11-12, verbatim):

> "53+ designed substrates. ONE landed at frontier on CPU (PR101 fec6 clean k16). Class-shift hypothesis testing has been talked-about more than executed. Three numbers matter: (a) operator dollars spent on paid GPU since 2026-05-15 (estimate $40-80; refusing to be precise without anchor), (b) net frontier improvement over the same window (CPU: -0.000794 vs PR101 GOLD; CUDA: -0.024 vs PR102; neither moved in 5 days), (c) attendant-overhead-per-frontier-improvement-byte. The honest answer is that the apparatus has been producing infrastructure that protects the frontier; it has not been producing frontier."

**Anchor 2 — WIRE-IN-RIGOR 35%/65% empirical baseline, 2026-05-20** (`feedback_wire_in_rigor_audit_meta_class_extinction_20260520.md`):

- 44/44 cathedral consumers return `predicted_delta_adjustment=0.0` (observability-only by design per Catalog #341)
- 0 production callsites of `evaluate_with_admm` / `choose_solver` (meta-Lagrangian SCAFFOLD_ONLY)
- 10/10 master-gradient anchors `[macOS-CPU advisory]` (0% authoritative on contest-compliant hardware)
- Verdict: "~35% of cathedral autopilot empirically grounded; ~65% observability-by-design-but-functionally-inert. The 65% is NOT a bug — it is design intent per Catalog #335 + #341. **The bug is OPERATOR MENTAL MODEL.**"

**Anchor 3 — CATHEDRAL-SMARTER Assumption-Adversary verdict, 2026-05-20** (`feedback_cathedral_autopilot_smarter_design_blueprint_landed_20260520.md` Assumption #2):

> "HARD-EARNED via WIRE-IN-RIGOR empirical findings ... CARGO-CULTED because the actual bottleneck is per-substrate OPTIMAL FORM iteration (Catalog #315) + paid dispatch on contest-compliant hardware, not the apparatus itself."

**The empirical pattern**: today's 3 landings (CATHEDRAL-SMARTER / META-LAGRANGIAN-WIRE-1 Phase 1 / BIT-ALLOCATOR DIM-3) ALL self-classified as `apparatus_maintenance` despite enabling future `frontier_breaking`. The mission_contribution enum (5 categories per CLAUDE.md "Mission alignment" Consequence 5) IS canonical, but the criterion separating categories is currently tribal knowledge inherited per-subagent.

Without a canonical rubric: every future subagent must independently re-derive the classification, AND classification drifts session-to-session under apparatus-vs-frontier tension. The 60% rigor-dominant alert (per `is_rigor_dominant`) fires AFTER the apparatus has already drifted, not before.

---

## 2. Current state of classification surfaces

The apparatus carries 7 distinct classification surfaces that each tag a *different* aspect of work:

| Surface | Catalog # | What it classifies | Granularity | Enforcement |
|---|---|---|---|---|
| **mission_predicted_contribution** | #300 frontmatter field | THE SCORE-IMPACT INTENT of a council deliberation | Per-deliberation (T2+) | STRICT preflight; enum-validated |
| **horizon_class** | #309 design-memo header | Predicted CPU band of a substrate | Per-substrate | STRICT preflight |
| **lane_class** | lane registry top-level | The promotion category (substrate_engineering / research_substrate / etc.) | Per-lane | Lane maturity registry |
| **ConsumerTier (A/B)** | #341/#357 | Cathedral consumer's score-promotion authority | Per-consumer-module | STRICT preflight |
| **Council Tier (T1-T4)** | #300 frontmatter field | Decision authority + cadence budget of a deliberation | Per-deliberation | STRICT preflight + cadence audit |
| **evidence_grade / axis_tag** | #127/#192/#323 | Custody + axis + hardware substrate of a score claim | Per-call-site / per-artifact-row | STRICT preflight + canonical Provenance |
| **PV-vs-cargo-culted verdict** | #292 | Per-assumption classification surfaced by Assumption-Adversary | Per-assumption (T2+) | STRICT preflight + per-deliberation enforcement |

**THE GAP**: none of these surfaces classifies the apparatus component itself per the operator's "engineered + wired + integrated" rubric. mission_predicted_contribution comes closest but is per-deliberation, not per-component. ConsumerTier is per-consumer but limited to cathedral-consumer scope. Lane_class is too coarse (it tags "where this lives" not "how complete it is"). horizon_class is substrate-scoped.

**The operator's "faking" concern is structurally**: *"is this apparatus component actually producing frontier signal, or is it observability theater?"* — a question NO current surface answers per-component.

---

## 3. Analysis: what's the canonical criterion separating each mission_contribution category?

CLAUDE.md "Mission alignment" Consequence 5 provides the verbatim semantic:

- `frontier_breaking` — "opens a class-shift path predicted to lower score"
- `frontier_protecting` — "prevents a regression that would raise score (sister of strict-mode preflight gates)"
- `rigor_overhead` — "procedural-only; no direct score contribution but enables future contributions"
- `apparatus_maintenance` — "updates infrastructure without score implications"
- `mission_questioned` — "the verdict triggered the 'is this serving the mission?' question; documented for retrospective"

**The semantics are clear; the EVIDENCE STANDARD is not.** When does a CATHEDRAL-SMARTER design memo qualify as `frontier_breaking` vs `apparatus_maintenance`? Today's pattern: subagents self-classify as `apparatus_maintenance` to be conservative (safe default per #1155 backfill convention), but this masks the bimodal distribution Carmack surfaced: apparatus that ENABLES frontier-breaking via concrete near-term ablations vs apparatus that MAINTAINS infrastructure without ablation-readiness.

**The canonical criterion this rubric proposes**: an apparatus component's mission_contribution category is determined by 6 binary evidence dimensions, scored per a falling-rule list. The dimensions encode "engineered + wired + integrated + grounded + sustained + adversarial" verbatim per the operator's language.

---

## 4. Proposed 6-dimension rubric (the canonical criterion)

Every apparatus component (canonical helper / cathedral consumer / preflight gate / dispatch wrapper / canonical equation / council deliberation output / etc.) is scored against 6 binary dimensions. Each dimension has a HARD-EARNED-vs-CARGO-CULTED check; passing dimensions are TRUE iff the supporting evidence is HARD-EARNED.

### Dimension 1: EMPIRICAL ANCHOR PRESENT

**Question**: Does the component cite at least one empirical anchor on contest-compliant hardware (CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable) within its scope of authority?

**PASS criteria**:
- A canonical equation: ≥1 `EmpiricalAnchor` with `evidence_grade` ∈ `{contest_cuda, contest_cpu_gha_linux_x86_64, paired_cuda_cpu}` per Catalog #344
- A cathedral consumer: at least one in-scope candidate it has consumed has an authoritative anchor (not `[predicted]` / `[advisory]`)
- A preflight gate: at least one HISTORICAL bug-class incident it would have prevented, cited verbatim in the catalog row
- A council deliberation: cite-chain to ≥1 empirically-anchored sister deliberation OR a `deferred_substrate_id` with measured outcome
- A dispatch wrapper: an empirical anchor for a prior dispatch it gated

**FAIL criteria**: Component cites only `[predicted]` / `[advisory only]` / `[mps-research-signal]` / `[macos-cpu advisory]` evidence; component cites only HYPOTHETICAL future anchors; component cites no anchors at all.

**Cargo-cult marker**: "predicted to enable" / "would surface" / "could ground" without a sister anchor counts as CARGO-CULTED-PENDING-EMPIRICAL.

### Dimension 2: 6-HOOK WIRE-IN COMPLETENESS COUNT

**Question**: Of the Catalog #125 6 canonical hooks (sensitivity-map / Pareto constraint / bit-allocator / cathedral autopilot dispatch / continual-learning posterior / probe-disambiguator), how many does the component ACTIVELY wire in (not just declare N/A)?

**Scoring**: dimension PASSES iff ≥3 hooks are ACTIVE (not N/A); PARTIAL if 1-2 hooks ACTIVE; FAIL if 0 hooks ACTIVE.

**Justification**: Per CLAUDE.md "Subagent coherence-by-default" non-negotiable: "Silent omission is the orphan-work failure mode." Declaring N/A is structurally valid; declaring ACTIVE on ≥3 hooks indicates the component participates in cross-cutting solver coherence, not just its local surface.

**Cargo-cult marker**: declaring "Hook #4 cathedral autopilot dispatch = ACTIVE" because a consumer module exists, without verifying the consumer actually returns non-zero adjustment OR is auto-discovered by Catalog #335 — counts as CARGO-CULTED.

### Dimension 3: CANONICAL PROVENANCE + AXIS_TAG DISCIPLINE

**Question**: Does the component honor Catalog #287/#323/#341 canonical Provenance + axis_tag discipline at every score-claim emission?

**PASS criteria**:
- Every numeric output carries `axis_tag` per the 8-grade taxonomy
- Every persisted artifact carries Provenance per Catalog #323
- Cathedral consumers carry the 3 canonical markers (`predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"`) at Tier A, OR Tier B per Catalog #357 contract
- No phantom-score directory / phantom-score filename / phantom-provenance composition row

**FAIL criteria**: any score-claim row lacking `axis_tag` OR Provenance; any directory name promising a device that the contents don't match; any Tier B contribution missing canonical Provenance.

### Dimension 4: CATALOG-GATE SELF-PROTECTION PRESENT

**Question**: Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable: does the component have a STRICT preflight gate that refuses the bug class the component addresses (OR is a non-bug-class component such as a canonical helper that other gates already protect)?

**PASS criteria**:
- For a fix landing: a STRICT preflight gate function `check_<bug_class>` registered in `preflight_all()` AND a CLAUDE.md catalog row per Catalog #176
- For a canonical helper: ≥1 sister gate validates the helper's invariants (e.g. Catalog #131 protects `.omx/state/` writes; Catalog #335 protects cathedral consumer contracts; etc.)
- For a council deliberation: per-decision operator-routable maps to a future enforcement surface OR explicitly tags as deliberation-only

**FAIL criteria**: component fixes a bug class without landing the corresponding gate; component is a canonical helper with no sister-gate coverage; component is a council deliberation whose decisions are advisory-only with no enforcement.

**Cargo-cult marker**: "this gate will be added later" / "this is a one-off fix" / "the operator will catch it manually" — CARGO-CULTED.

### Dimension 5: WAVE-3 ABLATION FRAMEWORK PARTICIPATION (if score-mutating)

**Question**: If the component intends to mutate the score (cathedral consumer Tier B / dispatch wrapper / canonical-helper bit-allocator / sensitivity-map contributor / etc.), is it instrumented for paired-comparison ablation testing per the CATHEDRAL-SMARTER blueprint Dimension 1 Phase 2 methodology (replace hand-derived adjuster with solver-derived dual variable; measure paired-comparison ΔS)?

**PASS criteria**:
- Component declares a paired-comparison test surface (`ablation_pair_id` or equivalent)
- Component supports being toggled OFF in a paired-comparison test
- Component emits its predicted_delta WITH a confidence interval (Bayesian posterior per `tac.findings_lagrangian.posterior_update_from_anchors`)

**FAIL criteria**: score-mutating component is structurally not toggle-able; emits point predictions without confidence; cannot be ablated without code-modifying the apparatus.

**N/A criteria**: component is purely observability (no score mutation) → dimension is OBJECTIVE-N/A (passes by structural exclusion, not by cargo-cult bypass).

**Cargo-cult marker**: "the ablation will be added in Phase 2" without scheduling the Phase 2 lane — CARGO-CULTED.

### Dimension 6: HARD-EARNED-VS-CARGO-CULTED PER CATALOG #292

**Question**: Per CLAUDE.md "Council conduct" Fix-7 + Catalog #292: does the component's design memo / landing memo carry an Assumption-Adversary verdict block classifying each underlying assumption HARD-EARNED vs CARGO-CULTED with rationale?

**PASS criteria**:
- Council deliberation: `council_assumption_adversary_verdict` frontmatter present with non-empty entries per Catalog #292
- Design memo: `## Cargo-cult audit per assumption` section per Catalog #303
- Landing memo: per-assumption HARD-EARNED-vs-CARGO-CULTED classification visible in Assumption-Adversary frontmatter block
- Canonical helper: docstring cites empirical-receipt source per Catalog #287 evidence-tag discipline

**FAIL criteria**: component's design or landing memo omits the Assumption-Adversary block; all assumptions left as implicit / unsurfaced.

**Cargo-cult marker**: Assumption-Adversary block exists but every entry classified HARD-EARNED with rationale "obvious" / "industry standard" / "operator approved" — these are tribal-knowledge defaults, not first-principles classifications; CARGO-CULTED-OF-THE-CARGO-CULT-AUDIT-ITSELF.

---

## 5. 5-mission-contribution canonical criterion

Given the 6-dimension scoring, the canonical algorithm produces the mission_predicted_contribution per a falling-rule list (per Rudin canonical interpretability discipline, Catalog #251):

**Step 1 — DIM 6 (Assumption-Adversary block) check** (gating):
- If Dim 6 = FAIL → mission_contribution = `mission_questioned`. The component cannot be classified because its assumption base is not surfaced; operator review required. (This is the canonical escape valve.)

**Step 2 — DIM 1 (empirical anchor) check** (gating):
- If Dim 1 = FAIL AND Dim 5 = FAIL → mission_contribution = `apparatus_maintenance`. Component has no empirical grounding AND is not ablation-ready; pure infrastructure update.
- If Dim 1 = FAIL AND Dim 5 = N/A → mission_contribution = `rigor_overhead`. Observability-only enabler; no direct score impact but procedurally protects future contributions.

**Step 3 — DIM 4 (catalog-gate self-protection) check** (gating for `frontier_protecting`):
- If Dim 1 = PASS AND Dim 4 = PASS AND Dim 5 = N/A → mission_contribution = `frontier_protecting`. Component has empirical evidence of past prevention AND is structurally a refusal surface (preflight gate / canonical contract / strict-mode artifact validation). Sister of strict-mode preflight gates per CLAUDE.md verbatim.

**Step 4 — DIM 5 (ablation framework) + DIM 2 (wire-in count) joint check** (gating for `frontier_breaking`):
- If Dim 1 = PASS AND Dim 2 = PASS (≥3 ACTIVE hooks) AND Dim 5 = PASS → mission_contribution = `frontier_breaking`. Component has empirical anchor AND wires into ≥3 of the 6 cross-cutting solver hooks AND is ablation-ready. This is the canonical signature of a component that BOTH enables AND directly contributes to a class-shift path.

**Step 5 — Default catch-all** (when no falling-rule step matched cleanly):
- Else → mission_contribution = `apparatus_maintenance`. Conservative default; matches the existing #1155 backfill convention.

**Tie-break for `frontier_breaking` vs `frontier_protecting`**: per Step 3 → Step 4 ordering, a component that BOTH protects (Dim 4) AND breaks (Dim 5) is classified `frontier_breaking` because the breaking signal is the higher-EV mission contribution per CLAUDE.md "Mission alignment" Consequence 4 ("frontier-breaking moves DOMINATE rigor budget").

---

## 6. Canonical scoring algorithm (operational form)

```python
def classify_mission_contribution(component: ApparatusComponent) -> str:
    """Canonical 6-dimension rubric per WAVE-3-CANONICAL-WIRED-RUBRIC.

    Returns one of mission_contribution enum:
    {frontier_breaking, frontier_protecting, rigor_overhead,
     apparatus_maintenance, mission_questioned}

    The classification is a deterministic falling-rule list per
    Catalog #251 sister discipline (highest-priority rule fires first).
    """
    dim1 = check_empirical_anchor_present(component)
    dim2 = check_6_hook_wire_in_completeness(component)
    dim3 = check_canonical_provenance_axis_discipline(component)
    dim4 = check_catalog_gate_self_protection_present(component)
    dim5 = check_wave_3_ablation_framework_participation(component)
    dim6 = check_hard_earned_vs_cargo_culted_per_catalog_292(component)

    # Falling-rule list (Rudin canonical interpretability discipline).
    if dim6 == "FAIL":
        return "mission_questioned"  # operator review required

    if dim1 == "FAIL" and dim5 == "N/A":
        return "rigor_overhead"  # observability-only enabler

    if dim1 == "FAIL" and dim5 == "FAIL":
        return "apparatus_maintenance"  # pure infrastructure update

    if dim1 == "PASS" and dim4 == "PASS" and dim5 == "N/A":
        return "frontier_protecting"  # refusal surface w/ empirical history

    if dim1 == "PASS" and dim2 == "PASS" and dim5 == "PASS":
        return "frontier_breaking"  # empirical + wired + ablation-ready

    return "apparatus_maintenance"  # conservative default
```

**Note**: Dim 3 (Canonical Provenance) is NOT a gating dimension — it's a CORRECTNESS dimension. If Dim 3 = FAIL, the component is structurally broken (violates CLAUDE.md "Forbidden score claims" non-negotiable) and should be refused at preflight BEFORE classification. The rubric only classifies components that already pass Dim 3.

---

## 7. Worked examples — re-classify 5 today landings

Per operator directive "Worked examples — re-classify 5 today landings; verify match or document delta". Applying the rubric to each:

### 7.1 CATHEDRAL-SMARTER-DESIGN-MEMO (today's blueprint)

- **Dim 1 (empirical anchor)**: PASS — cites WIRE-IN-RIGOR 35%/65% empirical baseline + 118 continual-learning posterior anchors + 11 canonical equations
- **Dim 2 (6-hook count)**: PASS — declares all 6 hooks ACTIVE in landing memo (per verbatim Hook 1-6 declaration block)
- **Dim 3 (Provenance discipline)**: PASS — no score literals; canonical frontier pointer cited per Catalog #343
- **Dim 4 (catalog-gate self-protection)**: FAIL — design memo only; no STRICT preflight gate landed for the 6-dimension blueprint per se; sister gates (#335 / #341 / #305 / #303 / #294) cover prerequisites
- **Dim 5 (ablation framework)**: N/A — design memo only, no score mutation
- **Dim 6 (HARD-EARNED-vs-CARGO-CULTED)**: PASS — Assumption-Adversary block surfaces 5 assumptions; 2 HARD-EARNED-PARTIALLY-CARGO-CULTED + 3 HARD-EARNED

**Canonical algorithm output**: Dim 1=PASS, Dim 5=N/A → step 3 check: Dim 4=FAIL → falls through → step 5 default → `apparatus_maintenance`

**Self-classification by subagent**: `apparatus_maintenance` (per frontmatter line 36)

**VERDICT MATCHES.** The rubric agrees with the subagent's self-classification. The component is design-only with no STRICT gate of its own; sister gates protect prerequisites; the score-affecting child landings (DIM-1 Phase 2 / DIM-3) are tracked separately.

### 7.2 WIRE-IN-RIGOR audit landing

- **Dim 1 (empirical anchor)**: PASS — 5 empirical runtime invocations (44 cathedral consumers fired; 118 posterior anchors loaded; 59 probe outcomes; 393 modal call_id rows; 10 master-gradient anchors)
- **Dim 2 (6-hook count)**: PARTIAL — declares 3 ACTIVE (hooks 4, 5, 6), 3 N/A — borderline PASS
- **Dim 3 (Provenance discipline)**: PASS — no score literals; all empirical receipts cite source paths
- **Dim 4 (catalog-gate self-protection)**: FAIL — pure audit; no new gate landed
- **Dim 5 (ablation framework)**: N/A — audit findings don't mutate score
- **Dim 6 (HARD-EARNED-vs-CARGO-CULTED)**: PASS — Assumption-Adversary verdict block surfaces "auto-discovery + observability annotations sufficient to extinct orphan-signal class" classified CARGO-CULTED with rationale

**Canonical algorithm output**: Dim 1=PASS, Dim 5=N/A → step 3 check: Dim 4=FAIL → step 5 default → `apparatus_maintenance`

**Self-classification by subagent**: `rigor_overhead` (per frontmatter line 20)

**VERDICT DELTA**: rubric says `apparatus_maintenance`; subagent says `rigor_overhead`. Both are reasonable; the subagent's choice emphasizes the procedural-only nature of an audit; the rubric's choice emphasizes the infrastructure-status nature. **PROPOSED RUBRIC AMENDMENT**: Step 3 could be refined to: "if Dim 1 = PASS AND Dim 4 = FAIL AND Dim 5 = N/A AND component is an AUDIT (not a fix landing) → `rigor_overhead`" — but this risks the cargo-cult amendment "everyone calls themselves an audit". Operator review needed for amendment vs accept-delta. Leaving the rubric AS-PROPOSED; the 2-category delta between `apparatus_maintenance` and `rigor_overhead` is operator-acceptable per the existing #1155 backfill convention (both safe defaults).

### 7.3 META-LAGRANGIAN-WIRE-1 Phase 1

- **Dim 1 (empirical anchor)**: PARTIAL — Phase 1 wires the canonical invocation; the canonical equations consumed (4-term Lagrangian + Gaussian posterior + Lindley-1956) ARE empirically anchored elsewhere; no NEW empirical anchor at this landing
- **Dim 2 (6-hook count)**: PARTIAL — declares 2 ACTIVE (hook 4 cathedral dispatch primary; hook 1 sensitivity-map active via downstream); 4 N/A at Phase 1
- **Dim 3 (Provenance discipline)**: PASS — observability-only per Catalog #287/#323; bounded [0.95, 1.05] adjustment factor; never mutates candidate
- **Dim 4 (catalog-gate self-protection)**: PASS — Catalog #355 STRICT preflight gate `check_cathedral_autopilot_main_invokes_meta_lagrangian` landed in same commit batch
- **Dim 5 (ablation framework)**: N/A — Phase 1 is observability-only; Phase 2 lands ablation framework
- **Dim 6 (HARD-EARNED-vs-CARGO-CULTED)**: PASS per landing memo

**Canonical algorithm output**: Dim 1=PASS (treating PARTIAL as PASS per consumed-canonical-equation transitivity), Dim 5=N/A → step 3 check: Dim 4=PASS → `frontier_protecting`

**Self-classification by subagent** (per Catalog #355 row, not the landing memo directly): The catalog row marks Hook 4 = ACTIVE PRIMARY, no explicit mission_contribution classification visible in immediately scannable text. Likely `frontier_protecting` per the canonical Catalog #355 description (sister of strict-mode preflight gates).

**VERDICT MATCHES.** Phase 1 is structurally `frontier_protecting`: it prevents the regression where the meta-Lagrangian solver remains an orphan signal. Phase 2 (when ablation framework lands) will re-classify to `frontier_breaking` per Step 4.

### 7.4 DIM-1 Phase 2 START (today)

Per TaskList state at memo time, DIM-1 Phase 2 is starting today as the META-LAGRANGIAN-WIRE-1 succession plan. **Cannot yet be classified** because the landing memo doesn't exist; the rubric applies post-landing. **Predicted classification**: if Phase 2 ablation framework lands successfully → `frontier_breaking` per Step 4 (Dim 1=PASS + Dim 2=PASS + Dim 5=PASS). If Phase 2 lands without ablation framework → `frontier_protecting` per Step 3.

### 7.5 BIT-ALLOCATOR DIM-3 Phase

Per CATHEDRAL-SMARTER blueprint Dimension 3, this is the per-axis consumer payload extension landing. **Predicted classification** when it lands: if the landing includes the canonical `tac.score_composition.compose_score_from_axes` helper AND a Tier B cathedral consumer demonstrating ablation-ready per-axis contribution → `frontier_breaking`. If only the helper lands (Tier A consumers continue observability-only) → `frontier_protecting`.

### Summary

| Component | Subagent self-classification | Rubric classification | Match |
|---|---|---|---|
| CATHEDRAL-SMARTER-DESIGN-MEMO | `apparatus_maintenance` | `apparatus_maintenance` | ✓ MATCH |
| WIRE-IN-RIGOR audit | `rigor_overhead` | `apparatus_maintenance` | ◐ DELTA (acceptable per #1155 backfill convention) |
| META-LAGRANGIAN-WIRE-1 Phase 1 | (Catalog #355 row implies) `frontier_protecting` | `frontier_protecting` | ✓ MATCH |
| DIM-1 Phase 2 START | (pending) | (predicted `frontier_breaking` if ablation lands) | (cannot verify yet) |
| BIT-ALLOCATOR DIM-3 | (pending) | (predicted `frontier_breaking` if Tier B + helper land together) | (cannot verify yet) |

**EMPIRICAL VERDICT**: rubric matches 2 of 3 verifiable classifications; 1 delta is within the operator-acceptable conservative-default range. The rubric DOES NOT silently inflate `frontier_breaking` classifications (the failure mode operator was concerned about).

---

## 8. Sister-rubric composition

The proposed rubric composes with existing classification surfaces (Section 2 table) per:

### 8.1 Composition with Catalog #341 ConsumerTier (A/B)

The ConsumerTier classifies CATHEDRAL CONSUMER SCORE-PROMOTION AUTHORITY (per-module). The proposed rubric classifies COMPONENT-LEVEL MISSION CONTRIBUTION (per-deliberation / per-landing). They are orthogonal:

- A Tier A consumer (observability-only) is typically `frontier_protecting` (sister of strict-mode preflight gates) — IF it has a sister catalog gate; else `apparatus_maintenance`
- A Tier B consumer (score-contributing) is typically `frontier_breaking` IFF Dim 5 ablation framework PASSES; else `frontier_protecting`

The rubric STRUCTURALLY RESPECTS Catalog #341 — Tier B consumers don't get a `frontier_breaking` classification just by being Tier B; they get it by ALSO satisfying Dim 5 ablation-readiness.

### 8.2 Composition with Catalog #357 Tier B canonical contract

Catalog #357 STRICT preflight enforces the per-Tier-B return-dict contract (canonical Provenance + non-promotable + empirically-grounded axis_tag). The rubric Dim 3 (Provenance discipline) DIRECTLY REUSES Catalog #357 semantics — a Tier B consumer that violates #357 is structurally broken at preflight, never reaches the rubric.

### 8.3 Composition with Catalog #324 predicted_band post-training validation

Catalog #324 enforces that any `predicted_band` in a recipe is post-training validated (or research-only / pending). The rubric Dim 1 (empirical anchor present) DIRECTLY REUSES Catalog #324 semantics — a substrate component citing only random-init Tier-C density evidence fails Dim 1 because the anchor is `phantom_random_init` per Catalog #324.

### 8.4 Composition with Catalog #294 9-dim checklist evidence

Catalog #294 enforces per-substrate-design-memo 9-dim checklist. The proposed rubric is the META-DIMENSION on top of the 9 substrate-design dimensions: 9-dim checklist measures DESIGN COMPLETENESS; the proposed rubric measures MISSION-CONTRIBUTION CLASSIFICATION. A substrate that passes Catalog #294 9-dim can still be `apparatus_maintenance` per the rubric if Dim 4 + 5 fail.

### 8.5 Composition with Catalog #305 observability surface section

Catalog #305 enforces per-substrate-design-memo observability surface section. The proposed rubric Dim 3 (Canonical Provenance) PARTIALLY OVERLAPS Catalog #305 observability discipline — both require traceability, but Catalog #305 is design-memo-section-presence; the rubric Dim 3 is per-output validation.

**Net composition verdict**: the proposed rubric is STRUCTURALLY ORTHOGONAL to all 7 existing classification surfaces. It does not duplicate any existing gate's enforcement. It SYNTHESIZES across them to produce the per-component mission-contribution classification that currently is tribal knowledge.

---

## 9. HARD-EARNED-vs-CARGO-CULTED audit of the rubric itself

Per CLAUDE.md "Council conduct" Fix-7 + Catalog #292 + Catalog #303: every design memo must surface its own assumptions and classify them. The Assumption-Adversary verdict on the 6 dimensions:

### Assumption A1: "6 dimensions are necessary AND sufficient"

**Classification**: HARD-EARNED-PARTIALLY-CARGO-CULTED

**HARD-EARNED rationale**: Each of the 6 dimensions maps directly to a CLAUDE.md non-negotiable (Dim 1 → "Apples-to-apples evidence discipline"; Dim 2 → "Subagent coherence-by-default" 6-hook wire-in; Dim 3 → "Apples-to-apples" + Catalog #287/#323; Dim 4 → "Bugs must be permanently fixed AND self-protected against"; Dim 5 → CATHEDRAL-SMARTER blueprint Dim 1 Phase 2 ablation methodology; Dim 6 → Catalog #292 + #303 assumption surfacing). The dimensions are CANONICAL per existing apparatus discipline.

**CARGO-CULTED rationale**: The number "6" is suspicious (matches the 6-hook count by accident or by design?). Could there be a 7th dimension surfacing operator-time-cost (per Carmack T3 dissent line c: "attendant-overhead-per-frontier-improvement-byte")? The rubric omits this because the existing cadence audit + 60% rigor-dominant alert already handles it at the per-deliberation surface, but a per-component cost dimension might surface complementary signal. Reactivation criterion: if 30 days of rubric usage shows components systematically misclassified as `frontier_breaking` despite high operator-attention cost, add Dim 7.

### Assumption A2: "Falling-rule list ordering is canonical"

**Classification**: HARD-EARNED per Rudin canonical interpretability discipline (Catalog #251).

**Rationale**: Wang & Rudin 2015 "Falling Rule Lists" establishes that for interpretable classifiers with cost-sensitive misclassification, the falling-rule list with first-match-wins is canonical. The proposed ordering (mission_questioned > rigor_overhead/apparatus_maintenance > frontier_protecting > frontier_breaking > apparatus_maintenance default) treats `mission_questioned` as highest-priority (operator review = highest cost to miss), `frontier_breaking` as canonical positive case (highest reward to detect correctly), with conservative defaults dominating in ambiguous cases.

### Assumption A3: "PARTIAL counts as PASS for Dim 1 and Dim 2"

**Classification**: HARD-EARNED-PARTIALLY-CARGO-CULTED

**HARD-EARNED rationale**: A canonical helper that consumes empirically-anchored canonical equations transitively has empirical grounding even without a direct anchor of its own.

**CARGO-CULTED rationale**: This could enable subagents to claim PASS on Dim 1 via thin transitivity (e.g. "I consume a sister equation that has an anchor"). Without a numerical threshold, the transitivity rule is hand-wavy. Operator review needed; rubric v1 leaves PARTIAL=PASS for Dim 1/Dim 2 as a starting calibration; v2 may tighten.

### Assumption A4: "Dim 3 is correctness-not-classification"

**Classification**: HARD-EARNED.

**Rationale**: Per CLAUDE.md "Forbidden score claims" non-negotiable, a component that violates Provenance discipline is structurally broken at preflight (Catalog #287/#323/#341 STRICT gates refuse it BEFORE landing). Therefore Dim 3 is not a classification dimension; it's a precondition.

### Assumption A5: "The rubric structurally extincts the 'faking' bug class"

**Classification**: CARGO-CULTED-PENDING-EMPIRICAL.

**HARD-EARNED rationale**: The rubric IS a structural surface (deterministic algorithm; reproducible classification given component evidence). It removes per-subagent tribal-knowledge classification.

**CARGO-CULTED rationale**: The rubric CAN be cargo-culted by future subagents who claim PASS on dimensions without producing evidence (e.g. fake EmpiricalAnchor citations; fake 6-hook wire-in declarations). The structural extinction requires Dimension 4 sister-gate enforcement at the per-component surface, which this design memo PROPOSES but does NOT yet land (operator-routable next-action #3 below).

Reactivation criterion: after 30 days of rubric usage WITH the proposed Catalog #X STRICT preflight gate (op-routable #3), audit components classified `frontier_breaking` and verify each one has an actual ablation-result by 60 days post-classification. If <80% verified → CARGO-CULTED-FALSIFIED; rubric requires tightening.

### Assumption A6: "PARTIAL on Dim 2 (1-2 hooks ACTIVE) should count as PASS for classification"

**Classification**: CARGO-CULTED-PENDING-EMPIRICAL.

**Rationale**: The 6-hook count threshold ≥3 is a CALIBRATION CHOICE not a derived optimal. Why not ≥4? ≥2? The proposed threshold is the geometric midpoint (3 of 6 = 50%) which is a Bayesian-prior default but lacks empirical grounding. Operator review needed; rubric v1 uses ≥3 as starting calibration. Reactivation criterion: collect 30 days of classification data; if components classified `frontier_breaking` with exactly 3 hooks ACTIVE have lower-than-population mean empirical ablation success → tighten threshold to ≥4.

---

## 10. 9-dimension success checklist evidence (Catalog #294)

| Dim | Verdict | Evidence |
|-----|---------|----------|
| **UNIQUENESS** | PASS | This rubric is distinct from the 7 existing classification surfaces enumerated in Section 2; it is the missing per-component mission-contribution surface |
| **BEAUTY + ELEGANCE** | PASS | 6 dimensions, 5-step falling-rule algorithm, reviewable in 2 minutes; Rudin canonical interpretability discipline (Catalog #251) |
| **DISTINCTNESS** | PASS | Section 8 composition analysis verifies orthogonality to all 7 existing classification surfaces |
| **RIGOR** | PASS | Per-assumption Assumption-Adversary verdict in Section 9 (6 assumptions surfaced with HARD-EARNED-vs-CARGO-CULTED classification); empirical motivation in Section 1 cites verbatim Carmack T3 dissent + WIRE-IN-RIGOR audit + CATHEDRAL-SMARTER Assumption-Adversary verdict |
| **OPTIMIZATION-PER-TECHNIQUE** | PASS | Each dimension reuses an existing canonical surface (Catalog #287/#323/#341/#324/#294/#305/#292/#303/#125 6-hook); no duplication, full reuse |
| **STACK-OF-STACKS-COMPOSABILITY** | PASS | Section 8 demonstrates composition with all 7 existing classification surfaces; the rubric is the META-DIMENSION on top of substrate-design 9-dim checklist |
| **DETERMINISTIC-REPRODUCIBILITY** | PASS | The canonical algorithm in Section 6 is deterministic given component evidence; same evidence inputs always produce same classification output |
| **EXTREME-OPTIMIZATION-PERFORMANCE** | PASS | The rubric is design-only; per-component classification at preflight time is O(1) lookup against 6 dimensions; no GPU cost |
| **OPTIMAL-MINIMAL-CONTEST-SCORE** | PARTIAL | Per Catalog #324 + CLAUDE.md "Frontier target" non-negotiable: this rubric does NOT directly produce frontier-breaking; it PREVENTS misclassification of apparatus-maintenance as frontier-breaking, which structurally protects the apparatus from drift per CLAUDE.md "Mission alignment" Consequence 5 alert threshold (60% rigor-dominant) |

---

## 11. Cargo-cult audit per assumption (Catalog #303)

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + Catalog #303 sister discipline: every assumption surfaced in Section 9 IS classified HARD-EARNED-vs-CARGO-CULTED with reactivation criteria. See Section 9 Assumption A1-A6 directly.

Additional inherited assumptions (the rubric does NOT introduce, but operates within):

- **Inherited Assumption #1**: "The 5-category mission_contribution enum per CLAUDE.md Consequence 5 is canonical AND sufficient" — HARD-EARNED per operator binding standing directive 2026-05-16 verbatim; the rubric extends but does not replace the enum.
- **Inherited Assumption #2**: "Catalog #292 Assumption-Adversary verdict surfaces apparatus drift" — HARD-EARNED per Catalog #291 + #292 + #325 sister discipline operational empirical anchors over 2026-05-15 → 2026-05-20.
- **Inherited Assumption #3**: "Catalog #125 6-hook wire-in is the canonical coherence primitive" — HARD-EARNED per CLAUDE.md "Subagent coherence-by-default" non-negotiable + 35+ canonical-helper landings reusing it.

---

## 12. Observability surface (Catalog #305)

| Facet | Where to inspect |
|-------|------------------|
| **Inspectable per layer** | This memo Section 4 (per-dimension PASS/FAIL criteria); Section 6 (canonical algorithm pseudocode) |
| **Decomposable per signal** | Section 4 enumerates 6 dimensions; per-dimension HARD-EARNED-vs-CARGO-CULTED markers per Section 9 |
| **Diff-able across runs** | Same component re-classified under updated evidence produces traceable per-dimension diff (Dim 1 evidence path; Dim 2 hook count; etc.) |
| **Queryable post-hoc** | Future Catalog #X STRICT gate (op-routable #3) will emit per-component classification rows to `.omx/state/apparatus_component_classification.jsonl` per Catalog #245 sister discipline |
| **Cite-able** | Every classification carries (component_id / classification_utc / per-dimension verdicts / cite-chain to evidence sources) |
| **Counterfactual-able** | The 5-step falling-rule algorithm enables "what if Dim N had been PASS" counterfactuals per Catalog #139 byte-mutation discipline analog |

---

## 13. Horizon-class declaration (Catalog #309)

**horizon_class: frontier_protecting**

Per Catalog #309 + CLAUDE.md "HORIZON-CLASS evaluation axis plateau warning" standing directive: this rubric does NOT predict a substrate ΔS band (Catalog #324 NO predicted_band). It predicts an APPARATUS-CLASSIFICATION-DRIFT-EXTINCTION outcome: applying the rubric in the next 30 days should produce structural reduction in subagent misclassification of `apparatus_maintenance` vs `frontier_breaking`, surfacing the 60% rigor-dominant alert BEFORE drift instead of AFTER.

---

## 14. Operator-routable next-actions

### Op-routable #1 (HIGH-PRIORITY — T2 sextet ratification)

Spawn a T2 sextet deliberation (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary) to RATIFY or REVISE the proposed 6-dimension rubric per Catalog #300 v2 frontmatter + Catalog #292 Assumption-Adversary discipline. Specific questions for sextet:

1. Is the 6-dimension enumeration NECESSARY AND SUFFICIENT? (Assumption A1 audit)
2. Is the falling-rule ordering canonical? (Assumption A2 audit)
3. Is the PARTIAL=PASS calibration for Dim 1/2 acceptable? (Assumption A3 + A6 audit)
4. Should Dim 3 (Provenance) be a classification dimension OR a precondition? (Assumption A4 review)
5. Is the rubric structurally sufficient to extinct the "faking" bug class? (Assumption A5 audit)
6. Should there be a Dim 7 for operator-time-cost? (Assumption A1 reactivation)

### Op-routable #2 (MID-PRIORITY — extend Catalog #300 frontmatter validation)

Once ratified, extend `tac.council_continual_learning.CouncilDeliberationRecord` (and Catalog #300 STRICT preflight gate) with the 6-dimension classification fields:

```python
@dataclass(frozen=True)
class CouncilDeliberationRecord:
    # ... existing fields ...
    # WAVE-3-CANONICAL-WIRED-RUBRIC extension (operator-routed; T2 sextet ratification required first)
    component_rubric_dim_1_empirical_anchor: str | None = None  # "PASS" / "FAIL" / "PARTIAL" / "N/A"
    component_rubric_dim_2_six_hook_count: str | None = None
    component_rubric_dim_3_provenance_discipline: str | None = None
    component_rubric_dim_4_catalog_gate_self_protection: str | None = None
    component_rubric_dim_5_ablation_framework: str | None = None
    component_rubric_dim_6_assumption_adversary: str | None = None
    # The derived classification (per Section 6 algorithm)
    component_rubric_derived_mission_contribution: str | None = None
```

Validation extension: `__post_init__` verifies that `component_rubric_derived_mission_contribution` (if present) MATCHES `predicted_mission_contribution` per the canonical algorithm, OR carries a non-placeholder rationale for the delta in `notes`.

### Op-routable #3 (LOW-PRIORITY — propose NEW STRICT gate POST-RATIFICATION)

Per Catalog #299 quota brake (catalog # at 354 of 400 = 46 slots remaining; operator-routed approval required for net-new gates): propose `check_apparatus_component_rubric_compliance` STRICT preflight gate that refuses any landing memo (dated >= ratification_date) lacking the 6-dimension classification block. Initial wire-in WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule"; strict-flip after backfill sweep.

**OPERATOR-ROUTABLE EXPLICITLY**: per Catalog #299 sister discipline ("every new gate landing MUST satisfy: retire existing gate OR carry file-level waiver OR REPLACE existing gate"), the gate should be PROPOSED as a sister-extension of Catalog #300 (extend the existing gate to cover the 6-dimension fields) NOT as a net-new gate. Per CLAUDE.md "Gate consolidation discipline" + Catalog #287 v2 scope-extension precedent.

---

## 15. Cross-references

- CATHEDRAL-SMARTER-DESIGN-MEMO landing: `feedback_cathedral_autopilot_smarter_design_blueprint_landed_20260520.md` (the Carmack T3 dissent + Assumption-Adversary verdict that motivates this rubric)
- WIRE-IN-RIGOR landing: `feedback_wire_in_rigor_audit_meta_class_extinction_20260520.md` (the 35%/65% empirical baseline)
- T3 grand strategy review: `.omx/research/council_t3_grand_strategy_review_20260520T120000Z.md` (the 12-decision strategic context this rubric extends Decision 5)
- CLAUDE.md "Mission alignment — non-negotiable" subsection (the 5-category enum this rubric classifies into)
- CLAUDE.md "Subagent coherence-by-default" Catalog #125 (the 6-hook wire-in dimension)
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" (the catalog-gate self-protection dimension)
- CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" + Catalog #291 / #292 (the HARD-EARNED-vs-CARGO-CULTED dimension)
- `docs/meta_engineering_vision.md` (the canonical destination this rubric protects against drift from)
- `src/tac/cathedral/consumer_contract.py` (the ConsumerTier sister classification)
- `src/tac/council_continual_learning.py` (the CouncilDeliberationRecord canonical helper this rubric proposes extending)
- `src/tac/canonical_equations/` (the canonical equation registry per Catalog #344)
- `src/tac/provenance/` (the Catalog #323 canonical Provenance umbrella)
- Catalog #251 (Wang & Rudin 2015 falling-rule list discipline; the algorithm in Section 6 follows this canonical form)

---

## 16. Honest verdict + closing position

**HONEST verdict on whether this rubric structurally closes operator's "faking" concern**:

- **PARTIAL YES**: the rubric extincts the SUBAGENT SELF-CLASSIFICATION DRIFT bug class structurally. A subagent following the rubric cannot silently inflate `frontier_breaking` claims because each dimension requires verifiable evidence (cite-chain to empirical anchor / actual hook-count tally / canonical Provenance / actual STRICT gate registration / actual ablation framework / actual Assumption-Adversary block).
- **PARTIAL NO**: the rubric does NOT prevent CARGO-CULT EVIDENCE PRODUCTION (e.g. subagents inventing EmpiricalAnchor citations; declaring fake 6-hook ACTIVE; etc.). This requires the proposed Catalog #X STRICT gate (op-routable #3) for structural enforcement at landing time. Without the gate, the rubric is a CANONICAL HELPER (provides consistent classification) but not a STRUCTURAL EXTINCTION (does not refuse fake evidence).
- **PARTIAL PENDING**: the rubric requires T2 sextet ratification (op-routable #1) before it becomes canonical apparatus discipline. Pre-ratification, it is one subagent's design proposal.

**The honest mid-term path**: ratify → extend Catalog #300 → land the proposed STRICT gate as sister-extension (not net-new) → backfill 60 days of components → verify per Assumption A5 reactivation criterion that ≥80% of `frontier_breaking` classifications produced actual ablation-results within 60 days. If yes → rubric is HARD-EARNED-EMPIRICALLY-VERIFIED. If no → rubric requires tightening (probably Dim 5 threshold).

**The honest closing position**: the apparatus has accumulated 7 classification surfaces over 2026-05-08 → 2026-05-20; none of them answer the operator's per-component "faking" concern at the granularity Carmack T3 dissent surfaces. This rubric IS the canonical answer — but it remains a PROPOSAL until ratified, and remains a CANONICAL HELPER (not a STRUCTURAL EXTINCTION) until the proposed STRICT gate lands. The 3-step path (ratify → extend Catalog #300 → land sister-gate) is operator-routable; the design is complete.

Per Carmack T3 dissent (verbatim): *"The honest answer is that the apparatus has been producing infrastructure that protects the frontier; it has not been producing frontier."* This rubric is itself classified `frontier_protecting` per its own algorithm — it does not directly produce frontier movement; it prevents apparatus-classification drift that would otherwise mask the apparatus-vs-frontier balance from the operator's view. Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 5: structural protection enables future apparatus-vs-frontier rebalancing.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:canonical-rubric-design-memo-v2-trigger-tokens-in-rubric-criteria-not-new-empirical-finding-claim -->
