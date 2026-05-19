---
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Tao, Boyd]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Per-instance correction of phantom-API mentions structurally extincts the bug class"
    classification: CARGO-CULTED
    rationale: "16+ phantom-API instances despite per-instance correction is empirical proof that per-instance correction does NOT extinct the bug class. Structural mitigation is required."
  - assumption: "Scope-extending Catalog #287 (docstring-overstatement-without-evidence-tag) to `.omx/research/*.md` cited-module-names structurally prevents recurrence"
    classification: HARD-EARNED
    rationale: "Catalog #287 already validates evidence-tags adjacent to claims; extending to MODULE-NAME mentions in research memos applies the same structural mechanism to a sister surface. The pattern is canonical per CLAUDE.md 'Bugs must be permanently fixed AND self-protected against'."
council_decisions_recorded:
  - "op-routable #1: write new STRICT preflight gate `check_research_memo_module_name_mentions_resolve_to_importable_modules` (extends Catalog #287 to `.omx/research/*.md`)"
  - "op-routable #2: scope: scan `.omx/research/*.md` for `tac.X.Y` patterns; verify via `importlib.import_module()` at preflight time"
  - "op-routable #3: WARN-ONLY initially per Strict-flip atomicity rule; backfill 16+ known instances; STRICT-flip when live count = 0"
  - "op-routable #4: claim next-available catalog # via canonical serializer per Catalog #186"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: frontier_protecting
finding_action_class: delegate
finding_followup_dispatch_envelope_usd: 0.00
finding_canonical_path: experimental
---

# Finding 5: 16+ META-audit phantom-API recurrence

## What happened

Wave 2A PV-4 caught yet another phantom-API mention (`tac.preflight_rudin_daubechies.rashomon_ensemble_ranker` cited without verification). This is the 16th (or 17th) instance of phantom-API mention across `.omx/research/*.md` despite per-instance correction.

## Council deliberation

### Shannon LEAD (operating-within: structural extinction beats per-instance whack-a-mole)
16+ instances despite correction = per-instance correction does NOT extinct bug class. Catalog #287 evidence-tag discipline structurally prevents phantom-claim recurrence in docstrings; sister extension to research memos applies the same mechanism to a sister surface.

### Tao (operating-within: mathematical-induction over surfaces)
Bug class persists across surfaces because the structural protection covers only one surface. Extend coverage; per-surface extension of structural protection IS the canonical mechanism.

### Boyd (operating-within: convex-feasibility of every surface)
Phantom-API mention is a feasibility violation (claimed import doesn't exist). Structural feasibility-check at preflight time IS the canonical fix.

### Contrarian (operating-within: are phantom mentions actually causing damage?)
Phantom mentions in research memos are usually caught by next sister via PV. Damage: wasted PV time + occasional propagation. Structural extinction is appropriate but priority is medium-low.

### Yousfi + Fridrich + Dykstra: AGREE
Structural mitigation is canonical practice.

### Assumption-Adversary
- "Per-instance correction works" CARGO-CULTED (empirically refuted at scale)
- "Structural extension prevents recurrence" HARD-EARNED-VS-CANONICAL-PATTERN

## Verdict + rationale

**PROCEED**: write new STRICT preflight gate scope-extending Catalog #287 to research memos. Editor-only ($0); estimated 1-2h. Backfill known instances WARN-ONLY; STRICT-flip when live count = 0.

## Action class + next-step dispatch

**delegate** to follow-on subagent (a single-PR write of new gate + backfill sweep). No paid GPU. New catalog # via canonical serializer.

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t3_finding_5_meta_audit_phantom_api_recurrence_20260518", deliberation_id="finding_5_meta_audit_phantom_api_recurrence", council_tier="T3", council_verdict="PROCEED", cost_envelope_usd=0.00)`
- Posterior anchor via `append_council_anchor(...)`
- Probe outcome: not applicable (this is a META-meta-gate finding, not empirically dispatched)
- MEMORY.md index entry: paired with deliberation wave landing
- Cross-references: standing-directive memory file finding #5; Catalog #287; Catalog #185 (sister META-meta-meta drift gate); CLAUDE.md "Bugs must be permanently fixed AND self-protected against"

## Reactivation criteria

- New phantom-API instances after gate landing: gate is fail-open, fix gate
- Live count > 0 after backfill: investigate why backfill missed instances
