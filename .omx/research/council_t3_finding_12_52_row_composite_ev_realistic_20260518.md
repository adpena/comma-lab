---
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Boyd, Hassabis]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "52-row composite EV [-0.139, -0.026] is OPTIMISTIC because it assumes additive composition. Wave 1 NSCS06 v8 anchor shows -78% non-monotonicity when composing cargo-cult unwinds. Realistic compounded EV is -0.02 to -0.05 per Wave 1's empirical anti-pattern."
council_assumption_adversary_verdict:
  - assumption: "52-row arbitrariness extinction audit composite EV [-0.139, -0.026] aggregates additively"
    classification: CARGO-CULTED
    rationale: "Wave 1 NSCS06 v8 anchor empirically refutes additivity at composition: when 4-of-7 cargo-cults were unwound simultaneously, the score CHANGED by -78% from the additive prediction (i.e., the composition introduced REGRESSIONS where the previously-fixed cargo-cult assumptions still mattered structurally). Cargo-cult-unwind methodology does NOT compose monotonically across architectural changes."
  - assumption: "Realistic compounded EV is -0.02 to -0.05"
    classification: HARD-EARNED-FROM-WAVE-1-EMPIRICAL-ANCHOR
    rationale: "Wave 1 v8 anchor + sister findings provide empirical envelope. Still frontier-breaking but tempered by composition reality."
council_decisions_recorded:
  - "op-routable #1: dispatch-priority ranking should adopt REALISTIC EV bound -0.02 to -0.05, NOT optimistic -0.139 to -0.026, per Wave 1 anti-pattern"
  - "op-routable #2: per-substrate symposium per Catalog #325 BEFORE composing multiple extinctions: each composition is its own deliberation (cargo-cult-composition K-coverage gap)"
  - "op-routable #3: new sister-Catalog gate `check_substrate_design_memo_has_cargo_cult_composition_K_coverage_section` extending Catalog #303 (queued per Wave 1 anti-pattern memo)"
  - "op-routable #4: prioritize extinctions whose EV is LARGEST INDEPENDENTLY (not via composition); avoid multi-extinction stacking on a single substrate without per-substrate composition symposium"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: frontier_breaking
finding_action_class: pursue
finding_followup_dispatch_envelope_usd: 0.00
finding_canonical_path: experimental
---

# Finding 12: 52-row composite EV [-0.139, -0.026] optimistic; realistic -0.02 to -0.05

## What happened

52-row arbitrariness extinction audit composite EV is **[-0.139, -0.026]** assuming additive composition of independent extinctions. Wave 1 NSCS06 v8 anchor empirically demonstrates non-additive composition (-78% non-monotonicity when 4-of-7 cargo-cults unwound simultaneously vs additive prediction). Realistic compounded EV is **-0.02 to -0.05** per the Wave 1 anti-pattern.

## Council deliberation (T3 grand council)

### Shannon LEAD (operating-within: Pareto-frontier compounding rules)
Pareto-frontier compounding is sub-additive (when constraints are not orthogonal); composition of N extinctions does NOT yield N × per-extinction-EV. Wave 1 anti-pattern empirically confirms; realistic envelope is tighter.

### Hassabis (operating-within: risk-adjusted EV for paid-dispatch ranking)
The dispatch-priority ranker should use realistic envelope -0.02 to -0.05 for ranking decisions, NOT optimistic -0.139. Otherwise we over-prioritize compositions and under-prioritize independent largest-EV extinctions.

### Boyd (operating-within: composition is convex-feasibility intersection)
Composing N convex feasibilities yields intersection that is convex; the intersection may be FAR SMALLER than sum of per-constraint regions. Each composition needs its own Dykstra-feasibility check per Catalog #296. **Per-substrate composition symposium per Catalog #325 is canonical.**

### Contrarian (operating-within: Wave 1 anti-pattern is binding)
Wave 1 NSCS06 v8 was paradigm-falsified at -78% non-monotonicity. Adopt realistic envelope; rank extinctions by LARGEST INDEPENDENT EV, not stacked.

### Yousfi + Fridrich + Dykstra: AGREE
Composition discipline per Catalog #325; realistic envelope for ranking.

### Assumption-Adversary
"52-row additive composition" CARGO-CULTED (Wave 1 anti-pattern); "realistic envelope -0.02 to -0.05" HARD-EARNED.

## Verdict + rationale

**PROCEED_WITH_REVISIONS**: adopt realistic envelope for ranking; require per-substrate composition symposium per Catalog #325 before composing multiple extinctions on a single substrate.

## Action class + next-step dispatch

**pursue** (editor + ranking-update) — apply realistic EV envelope to cathedral autopilot dispatch ranking. Queue new sister-Catalog gate `check_substrate_design_memo_has_cargo_cult_composition_K_coverage_section` extending Catalog #303 (per Wave 1 anti-pattern memo standing recommendation).

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t3_finding_12_52_row_composite_ev_realistic_20260518", deliberation_id="finding_12_52_row_composite_ev_realistic", council_tier="T3", council_verdict="PROCEED_WITH_REVISIONS", predicted_impact_lower=-0.05, predicted_impact_upper=-0.02, cost_envelope_usd=0.00)`
- Posterior anchor via `append_council_anchor(...)`
- Probe outcome: not applicable (this is meta-finding about composition discipline, not single-extinction)
- Cross-references: standing-directive finding #12; Wave 1 NSCS06 v8 anchor; Catalog #303 cargo-cult audit; Catalog #325 per-substrate symposium

## Reactivation criteria

- New empirical evidence of additive composition: revise envelope upward (toward optimistic bound)
- Additional non-monotonicity anchors: revise envelope downward
