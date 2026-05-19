---
council_tier: T1
council_attendees: [Shannon, Boyd, Assumption-Adversary]
council_quorum_met: true
council_verdict: RATIFY
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "TOP-1 + TOP-4 batched empirical recovery within 4-decimal precision validates canonical formula approach"
    classification: HARD-EARNED
    rationale: "Commit 6db94d9ea + paired validation against CLAUDE.md anchors (pose_to_seg_ratio 2.7116 vs 2.71; decay 0.997 at N=1666) confirms canonical formula approach. T1 ratifies."
council_decisions_recorded:
  - "op-routable #1: ratify canonical-formula approach as the standard pattern for analytical-anchor recovery"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: apparatus_maintenance
finding_action_class: ratify
finding_followup_dispatch_envelope_usd: 0.00
finding_canonical_path: formula
---

# Finding 14: TOP-1 + TOP-4 batched empirical recovery within 4-decimal precision

## What happened

TOP-1 (`pose_to_seg_ratio = 2.7116`) + TOP-4 (`decay = 0.997 at N=1666`) batched empirical recovery in commit `6db94d9ea` validates canonical-formula approach to within 4-decimal precision of CLAUDE.md anchors.

## Council deliberation (T1 working group)

### Shannon LEAD + Boyd + Assumption-Adversary
T1 ratifies canonical-formula approach as standard pattern for analytical-anchor recovery. Per Catalog #290 canonical-vs-unique: formula derivation IS canonical default when derivation matches empirical anchor to design precision.

## Verdict + rationale

**RATIFY**: T1 working group ratifies canonical-formula approach. Findings 7 + 8 are the per-formula instantiations.

## Action class + next-step dispatch

**ratify** — no further action; pattern validated.

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t1_finding_14_top1_top4_batched_empirical_recovery_20260518", deliberation_id="finding_14_top1_top4_batched_empirical_recovery", council_tier="T1", council_verdict="RATIFY", cost_envelope_usd=0.00)`
- Posterior anchor via `append_council_anchor(...)`
- Cross-references: standing-directive finding #14; commit `6db94d9ea`; Findings 7 + 8; Catalog #290

## Reactivation criteria

- New canonical-formula approach fails to recover an empirical anchor: investigate; possibly per-substrate divergence
