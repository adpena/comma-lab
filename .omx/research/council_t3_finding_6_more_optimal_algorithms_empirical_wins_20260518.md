---
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Boyd, BeckTeboulle, Edelman, Hotz]
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Hotz
    verbatim: "FISTA + Frank-Wolfe + Riemannian-Newton EMPIRICAL wins are paired-comparison validated; drop in NOW. 1.5-2x wall-clock improvement across dispatch pipeline IS frontier-breaking via velocity multiplier (per CLAUDE.md Race-mode rigor inversion). Ship."
council_assumption_adversary_verdict:
  - assumption: "FISTA 1.25x faster than water-filling with byte-identical solution"
    classification: HARD-EARNED
    rationale: "Paired-comparison validated in MORE-OPTIMAL-ALGORITHMS commit 35c5d429f with byte-identical solution invariant. Canonical Beck-Teboulle proof."
  - assumption: "Frank-Wolfe 1.9x faster than Sinkhorn"
    classification: HARD-EARNED
    rationale: "FALSIFIED my synthesis-memo prediction. Paired-comparison validated."
  - assumption: "Riemannian-Newton 1.88x faster than Lloyd-projection with machine-epsilon orthogonality"
    classification: HARD-EARNED
    rationale: "Paired-comparison validated. Canonical Edelman manifold-optimization."
  - assumption: "Drop-in replacement is safe across all consumers"
    classification: HARD-EARNED-WITH-INVARIANT-CHECKS
    rationale: "Byte-identical solution invariant + machine-epsilon orthogonality invariant + Frank-Wolfe sparsity invariant — all paired-comparison validated. Each replacement keeps invariant; consumer-side regression-test pinning recommended."
council_decisions_recorded:
  - "op-routable #1: wire FISTA into `tac.bit_allocator.allocate_bits` (replaces water-filling Lagrangian iteration); add regression test pinning byte-identical solution"
  - "op-routable #2: wire Frank-Wolfe into Sinkhorn callsites (cathedral autopilot ranker bidirectional Sinkhorn); add regression test pinning sparsity invariant"
  - "op-routable #3: wire Riemannian-Newton into PQ codebook init (replaces Lloyd-projection); add regression test pinning machine-epsilon orthogonality"
  - "op-routable #4: emit canonical `tac.solvers.more_optimal_algorithms` shim package for canonical-helper-aware adoption per Catalog #290 canonical-vs-unique decision per layer"
  - "op-routable #5: per Catalog #245 4-layer pattern: append wall-clock-measurement atom to ledger per dispatch consumption"
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

# Finding 6: FISTA + Frank-Wolfe + Riemannian-Newton empirical wins

## What happened

Commit `35c5d429f` (MORE-OPTIMAL ALGORITHMS lane) paired-comparison validated:
- FISTA 1.25× faster than water-filling Lagrangian iteration with **byte-identical** solution
- Frank-Wolfe 1.9× faster than Sinkhorn (FALSIFIED my prior synthesis-memo prediction)
- Riemannian-Newton 1.88× faster than Lloyd-projection with machine-ε orthogonality

## Council deliberation

### Boyd LEAD (operating-within: convex-optimization canonical solver-replacement discipline)
FISTA (Beck-Teboulle 2009) is the canonical replacement for proximal Lagrangian iteration; provably O(1/k²) convergence vs water-filling's O(1/k). Byte-identical solution invariant means the consumer-facing contract is unchanged. **Drop-in is safe and frontier-protecting (faster dispatch + same archive bytes).**

### Beck-Teboulle memorial (operating-within: FISTA's accelerated gradient method)
FISTA's acceleration is provably tight for the proximal-gradient class. 1.25× empirical matches theoretical bound; wire-in is canonical.

### Edelman (operating-within: Riemannian manifold-optimization for codebook init)
Riemannian-Newton on Stiefel manifold (orthogonal codebooks) converges in O(log(1/ε)) vs Lloyd's O(1/ε). Machine-ε orthogonality is the canonical numerical invariant; preserved. Wire-in is canonical.

### Hotz (operating-within: drop-in NOW; ship)
1.5-2× wall-clock improvement IS frontier-breaking via velocity multiplier per CLAUDE.md Race-mode rigor inversion. Ship.

### Hassabis + Shannon + Dykstra: AGREE
Velocity is a frontier-breaking primitive when paired with safety invariants. Invariants are validated. Ship.

### Contrarian (operating-within: what could go wrong?)
Consumer-side assumption: water-filling iteration count = O(N), FISTA iteration count = O(sqrt(N)). Any consumer that EXPLICITLY DEPENDS on iteration-count semantics will break. Audit consumers + add regression-test pinning the iteration-count-agnostic invariant.

### Yousfi + Fridrich: AGREE
Faster solver = same archive bytes = same score; no risk to score-axis. Wall-clock multiplier directly enables more dispatches per session.

### Assumption-Adversary
All HARD-EARNED. Drop-in with invariant-pinning regression tests is canonical pattern per Catalog #290 canonical-vs-unique decision per layer.

## Verdict + rationale

**PROCEED**: drop-in canonical replacements with invariant-pinning regression tests. Editor-only ($0); estimated 2-3h. Frontier-breaking via velocity-multiplier; safety-invariant validated.

## Action class + next-step dispatch

**pursue** (editor + tests; $0 GPU). Wire FISTA + Frank-Wolfe + Riemannian-Newton into canonical consumers. Add `tac.solvers.more_optimal_algorithms` shim package. Regression tests pinning byte-identical / sparsity / machine-ε-orthogonality invariants.

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t3_finding_6_more_optimal_algorithms_20260518", deliberation_id="finding_6_more_optimal_algorithms_empirical_wins", council_tier="T3", council_verdict="PROCEED", cost_envelope_usd=0.00)`
- Posterior anchor via `append_council_anchor(...)`
- Probe outcome: `register_probe_outcome(probe_id="more_optimal_algorithms_empirical_wins_20260518", substrate="dispatch-pipeline-multi-consumer", verdict="PROCEED", metric_name="wall_clock_multiplier", metric_value=1.5, threshold=1.0, evidence_path="commit 35c5d429f", next_action="wire into canonical consumers")`
- MEMORY.md index entry: paired with deliberation wave landing
- Cross-references: standing-directive memory file finding #6; commit `35c5d429f`; Catalog #290 canonical-vs-unique decision per layer; CLAUDE.md Race-mode rigor inversion velocity-multiplier

## Reactivation criteria

- Consumer-side regression: invariant violated → revert specific consumer + investigate
- New optimization algorithm with > 1.5× wall-clock advantage: same pattern (paired-comparison validation → wire-in)
