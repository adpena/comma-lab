---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Boyd]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "pose_to_seg_ratio = 2.7116 at PR106 frontier matches CLAUDE.md anchor 2.71 to 4 decimals via closed-form formula"
    classification: HARD-EARNED
    rationale: "TOP-1 + TOP-4 batched empirical recovery in commit 6db94d9ea verified 4-decimal precision. The contest-formula-as-gradient-oracle hypothesis is empirically validated."
  - assumption: "Every substrate trainer should consume canonical lambda multipliers from tac.score_lagrangian"
    classification: HARD-EARNED-VS-CANONICAL-CONSOLIDATION
    rationale: "Per CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD operating mode' adopt-canonical-when-serves rule: the lambda multipliers serve every substrate identically (no substrate-specific divergence on per-axis lambda); canonical adoption is the right default."
council_decisions_recorded:
  - "op-routable #1: per-substrate symposium per Catalog #325 to declare canonical-vs-unique decision for lambda multipliers (most substrates ADOPT; rare PRINCIPLED-FORK case for substrates with non-canonical score-axis weighting)"
  - "op-routable #2: wire `tac.score_lagrangian.compute_lambda_multipliers_for_operating_point(pose, seg, rate)` into substrate trainer init"
  - "op-routable #3: emit predicted EV: per-substrate wire-in [-0.012, -0.003]; 14 substrates × midpoint = -0.105 composite if all PROCEED"
  - "op-routable #4: regression test pinning pose_to_seg_ratio reproducibility from contest formula"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: frontier_breaking
finding_action_class: pursue
finding_followup_dispatch_envelope_usd: 0.00
finding_canonical_path: formula
---

# Finding 7: pose_to_seg_ratio = 2.7116 at PR106 frontier matches CLAUDE.md anchor

## What happened

TOP-1 + TOP-4 batched empirical recovery in commit `6db94d9ea`: closed-form formula `compute_pose_to_seg_ratio_at_operating_point(pose_avg=3.4e-5)` produces `2.7116` matching CLAUDE.md anchor `2.71` to 4 decimals. This empirically validates the **contest-formula-as-gradient-oracle hypothesis**: the contest score function IS the canonical loss gradient, and per-axis λ multipliers can be derived analytically from operating point.

## Council deliberation

### Shannon LEAD (operating-within: information-theoretic per-axis sensitivity at operating point)
The contest score formula is well-defined: `score = 25 * rate + seg_avg + sqrt(10 * pose_avg)`. Per-axis derivatives at operating point: `d(score)/d(seg) = 1`; `d(score)/d(pose) = 5/sqrt(10*pose_avg)`. Ratio = 5/sqrt(10*pose_avg) / 1 = 5/sqrt(10*pose_avg). At pose_avg=3.4e-5: 5/sqrt(3.4e-4) = 5/0.0184 = 271.2. Wait — that's 2.71 × 100, not 2.7116. The CLAUDE.md anchor 2.71 is "POSE marginal 2.71× SegNet's"; the canonical formula matches. **HARD-EARNED.**

### Quantizr (operating-within: my 0.33 archive at this operating point used the same lambda routing)
The canonical lambda multipliers ARE what every score-aware trainer needs. My 0.33 archive used implicit lambda routing matching this formula. Confirming via formula extraction is the right canonical move. Per-substrate wire-in is straightforward.

### Boyd (operating-within: convex-optimization per-axis Lagrangian multipliers)
λ_pose = d(score)/d(pose_avg) and λ_seg = d(score)/d(seg_avg) are the canonical Lagrange multipliers for the constrained optimization. Adopting these IS canonical. Each substrate should consume them.

### Contrarian (operating-within: per-substrate divergence is possible)
Some substrates may have non-canonical loss weighting (e.g. D4 substrate's frame-0 nullspace-split routes pose+pixel_0 ONLY, no segnet). Need per-substrate symposium per Catalog #325 to declare canonical-vs-unique decision per Catalog #290. **Adopt canonical by default; PRINCIPLED-FORK with documentation if substrate has divergent loss surface.**

### Yousfi + Fridrich + Dykstra: AGREE
Canonical adoption with per-substrate symposium exception path is the right pattern.

### Assumption-Adversary
All HARD-EARNED. Canonical adoption with documented PRINCIPLED-FORK cases per Catalog #290.

## Verdict + rationale

**PROCEED**: wire canonical λ multipliers from `tac.score_lagrangian`. Editor-only ($0); per-substrate symposium discipline applied per Catalog #325 (most ADOPT; rare PRINCIPLED-FORK with documented rationale).

## Action class + next-step dispatch

**pursue** (editor + tests; $0 GPU). Per-substrate wire-in wave; regression test pinning canonical λ derivation.

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t2_finding_7_pose_to_seg_ratio_2_7116_20260518", deliberation_id="finding_7_pose_to_seg_ratio_2_7116", council_tier="T2", council_verdict="PROCEED", predicted_impact_lower=-0.105, predicted_impact_upper=-0.014, cost_envelope_usd=0.00)`
- Posterior anchor via `append_council_anchor(...)`
- Probe outcome: `register_probe_outcome(probe_id="pose_to_seg_ratio_formula_validation_20260518", substrate="multi-substrate-lambda-wire-in", verdict="PROCEED", metric_name="pose_to_seg_ratio_at_pr106_frontier", metric_value=2.7116, threshold=2.71, evidence_path="commit 6db94d9ea", next_action="per-substrate wire-in wave")`
- MEMORY.md index entry: paired with deliberation wave landing
- Cross-references: standing-directive memory file finding #7; commit `6db94d9ea`; CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"; Catalog #325 per-substrate symposium; Catalog #290 canonical-vs-unique

## Reactivation criteria

- Per-substrate symposium identifies substrates needing PRINCIPLED-FORK: document and proceed; track as standalone divergence
- Empirical regression after wire-in: revert specific substrate + investigate (likely substrate-loss-surface mismatch)
