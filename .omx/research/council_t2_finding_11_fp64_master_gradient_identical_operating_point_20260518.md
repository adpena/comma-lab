---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Boyd]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "ALL 3 PR101-family archives at IDENTICAL operating point (d_seg=0.0009/d_pose=0.00175/rate=0.00475/score=0.339) is suspicious. Either the archives are byte-similar at this resolution OR the master-gradient is INSENSITIVE to per-archive layout. Confirm via 600-pair-independence test before concluding 'variance purely byte-level Jacobian per archive layout'."
council_assumption_adversary_verdict:
  - assumption: "fp64 master-gradient at PR101-family resolves all 3 archives to IDENTICAL operating point"
    classification: HARD-EARNED
    rationale: "Commit 99373f010 empirically confirmed at fp64 precision. Variance from canonical: 0 across the 3 archives."
  - assumption: "Variance is purely byte-level Jacobian per archive layout"
    classification: CARGO-CULTED
    rationale: "Three archives sharing IDENTICAL operating point may indicate (a) byte-similar archives (then Jacobian-variance hypothesis is correct), OR (b) master-gradient INSENSITIVE to layout (then archives differ in non-Jacobian-detectable ways). Contrarian's 600-pair-independence test is canonical disambiguator."
council_decisions_recorded:
  - "op-routable #1: dispatch 600-pair-independence test ($0; editor + local-CPU) per Impl 4: do the 3 archives at this operating point produce DIFFERENT per-pair gradients? If yes → Jacobian-variance hypothesis HARD-EARNED. If no → master-gradient INSENSITIVE; pivot needed"
  - "op-routable #2: per Impl 13 per-pair Thompson sampling: if 600-pair-independent, use Thompson sampling per pair for next-rank-decision"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: frontier_protecting
finding_action_class: research
finding_followup_dispatch_envelope_usd: 0.00
finding_canonical_path: experimental
---

# Finding 11: fp64 master-gradient — ALL 3 PR101-family archives at IDENTICAL operating point

## What happened

Commit `99373f010`: fp64 master-gradient computation on 3 PR101-family archives produces IDENTICAL operating point:
- d_seg=0.0009 / d_pose=0.00175 / rate=0.00475 / score=0.339

Hypothesis: variance is purely byte-level Jacobian per archive layout. **But this is unverified.**

## Council deliberation (T2 sextet + Boyd)

### Shannon LEAD (operating-within: information-theoretic limit of gradient estimator)
Three archives at IDENTICAL operating point could mean: (a) archives are at the same actual point in score-space (byte-similar); (b) master-gradient estimator is insensitive at this precision. fp64 should be precise enough to detect <1e-4 differences; identical to 4-decimal precision is striking.

### Dykstra CO-LEAD (operating-within: convex-feasibility per-pair gradient independence)
Per-pair gradients should differ across pairs (600 pairs × 3 archives = 1800 gradients). If gradients are identical per archive at the operating-point but differ per pair, the layout determines per-pair routing. If gradients are identical per archive AND per pair, master-gradient is degenerate.

### Boyd (operating-within: convex-optimization sensitivity at operating point)
The master-gradient is the canonical d(score)/d(byte) row tensor; at the PR101 frontier, all 3 archives should have DIFFERENT byte distributions ⟹ different gradients UNLESS the score-function is locally flat at this operating point. Local flatness is testable via 2nd-order partial-derivative.

### Contrarian (operating-within: confirm before pivot)
Suspicious result. 600-pair-independence test IS canonical disambiguator before concluding either hypothesis.

### Yousfi + Fridrich + Assumption-Adversary: AGREE
600-pair-independence empirical test resolves.

## Verdict + rationale

**PROCEED_WITH_REVISIONS**: dispatch 600-pair-independence test ($0; local-CPU editor) per Impl 4. If passes: ratify Jacobian-variance hypothesis. If fails: pivot to investigate master-gradient sensitivity.

## Action class + next-step dispatch

**research** — $0 editor + local-CPU 600-pair-independence test.

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t2_finding_11_fp64_master_gradient_identical_operating_point_20260518", ...)`
- Posterior anchor via `append_council_anchor(...)`
- Probe outcome: `register_probe_outcome(probe_id="fp64_master_gradient_600_pair_independence_pending_20260518", substrate="pr101-family-multi-archive", verdict="DEFER", metric_name="600_pair_gradient_variance", metric_value=0.0, evidence_path="commit 99373f010", next_action="dispatch 600-pair-independence test")`
- Cross-references: standing-directive finding #11; commit `99373f010`; Impl 4 + Impl 13

## Reactivation criteria

- 600-pair-independence test passes: ratify Jacobian-variance hypothesis; emit per-pair Thompson sampling helper
- Test fails: investigate master-gradient sensitivity; possibly need higher-precision (fp128) or different gradient estimator
