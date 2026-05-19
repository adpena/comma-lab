---
council_tier: T1
council_attendees: [Shannon, Boyd, Assumption-Adversary]
council_quorum_met: true
council_verdict: RATIFY
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "5 contest_fixed rows correctly EXCLUDED from extinction"
    classification: HARD-EARNED
    rationale: "Per operator clarification: contest-fixed parameters (PoseNet weights / SegNet weights / evaluate.py SHA / video bytes / scoring formula) are NOT arbitrary; they are contest oracles. Audit correctly excluded these from arbitrariness extinction."
  - assumption: "Contest_fixed-as-oracles DEFINE the path-to-solution"
    classification: HARD-EARNED
    rationale: "The contest scorer + evaluate.py define the score gradient ORACLE. Analytical helpers derived from these oracles (tac.score_lagrangian.compute_lambda_multipliers_for_operating_point) IS the canonical path. The contest_oracle package (commit d17a9826c) operationalizes."
council_decisions_recorded:
  - "op-routable #1: ratify exclusion + contest_oracle package as canonical infrastructure"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: apparatus_maintenance
finding_action_class: ratify
finding_followup_dispatch_envelope_usd: 0.00
finding_canonical_path: contest_fixed
---

# Finding 13: 5 contest_fixed rows correctly EXCLUDED but DEFINE the path-to-solution

## What happened

5 audit rows reference contest-fixed parameters (PoseNet weights / SegNet weights / evaluate.py SHA / video bytes / scoring formula). Audit correctly excluded from arbitrariness extinction (these are contest oracles, not arbitrary choices). Operator clarification per `.omx/research/contest_fixed_as_oracles_15_implications_design_memo_20260518.md` (commit `07b24f303`): contest_fixed parameters DEFINE the path-to-solution via canonical analytical helpers (e.g. `tac.score_lagrangian`).

## Council deliberation (T1 working group; bounded scope)

### Shannon LEAD + Boyd + Assumption-Adversary
T1 working group ratifies. Exclusion is correct; contest_oracle package (commit `d17a9826c`) operationalizes contest-fixed-as-oracle pattern; canonical infrastructure validated.

## Verdict + rationale

**RATIFY**: T1 working group ratifies exclusion + contest_oracle infrastructure.

## Action class + next-step dispatch

**ratify** — no further action; canonical infrastructure landed.

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t1_finding_13_contest_fixed_excluded_20260518", deliberation_id="finding_13_contest_fixed_excluded_define_path", council_tier="T1", council_verdict="RATIFY", cost_envelope_usd=0.00)`
- Posterior anchor via `append_council_anchor(...)`
- Cross-references: standing-directive finding #13; `.omx/research/contest_fixed_as_oracles_15_implications_design_memo_20260518.md` (commit `07b24f303`); `tac.contest_oracle` package (commit `d17a9826c`)

## Reactivation criteria

- New contest-fixed parameter surfaces: classify per same pattern; extend contest_oracle package
