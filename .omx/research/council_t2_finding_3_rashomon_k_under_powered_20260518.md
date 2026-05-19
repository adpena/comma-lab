---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Rudin, Daubechies, Fisher, MacKay]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "K=599 raw analytical formula assumes default tight CI on Rashomon disagreement; the question is what CI we actually NEED for ranker quality. K=8 may produce SUFFICIENT disagreement for next-rank-decision QUALITY even if statistically under-powered for tight CI claims. The cap at 64 may be the sweet spot."
council_assumption_adversary_verdict:
  - assumption: "Rashomon K=599 raw is the canonical ensemble size for default tight CI"
    classification: HARD-EARNED
    rationale: "Rashomon set bootstrap-refit confidence interval analysis (Rudin et al. canonical) gives K=599 for default CI width on dispatch-ranking decisions; cited by Wave 2A `8b987215a` row #9."
  - assumption: "K=8 (current default) is grossly under-powered for ranker quality"
    classification: CARGO-CULTED
    rationale: "Under-powered for STATISTICAL CI claims, NOT necessarily for ranker DECISION quality. The cathedral autopilot ranker may produce sufficient-quality rankings with K=8 because the variance in next-rank-decision is dominated by candidate-level signal, not ensemble disagreement."
council_decisions_recorded:
  - "op-routable #1: empirical K-sweep on cathedral autopilot ranker: measure ranker DECISION variance (kendall-tau across K-bootstrap-resamples) at K in {8, 16, 32, 64, 128, 256, 512}"
  - "op-routable #2: cost-band budget for K: at K=8 ranker dispatch is ~instant; at K=512 it's 64× more expensive (in CPU-time); operator-decision required on tradeoff"
  - "op-routable #3: per-decision adaptive K: cheap decisions (low-stakes) use K=8, expensive decisions (paid dispatch >$5) use K=64+; route via tac.rashomon_ensemble.adaptive_k_for_decision"
  - "op-routable #4: pivot to UCB / Thompson sampling instead of Rashomon ensemble for some decisions per Wave 2A MORE-OPTIMAL-ALGORITHMS pattern"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: rigor_overhead
finding_action_class: research
finding_followup_dispatch_envelope_usd: 0.50
finding_canonical_path: analytical_solve
---

# Finding 3: Rashomon K=8 grossly under-powered

## What happened

Wave 2A `8b987215a` row #9: analytical formula for Rashomon ensemble size at default tight CI on ranker disagreement = **K=599** raw. Current default in `tac.preflight_rudin_daubechies.rashomon_ensemble_ranker` is K=8 (ceiling-capped from analytical 599 via budget constraint). 75× discrepancy.

## Council deliberation

### Rudin LEAD (operating-within: Rashomon ensemble = ALL models within ε of optimal loss)
The canonical Rashomon ensemble is a SET (sometimes infinite) of all models within ε-tolerance of optimal. Bootstrap-refit with K=8 vs K=599 produces different ESTIMATES of the set's disagreement; quality of ranker DECISIONS depends on whether the disagreement signal saturates well before K=599. Empirically — for falling-rule lists with sparse coefficient sets — K=32 often suffices; K=599 is a tight-CI claim, not a decision-quality claim.

### Daubechies CO-LEAD (operating-within: compressive sensing recovers signal from few measurements)
By analogy: Rashomon disagreement signal is SPARSE in decision-relevant coordinates; K=8 may achieve sufficient signal recovery if the disagreement is concentrated in the top-rank candidates (which is what we care about). Compressive-sensing analogy: under sparsity, K=O(log N) measurements suffice (here N = candidate count); K=8-32 may be plenty.

### Fisher (operating-within: information-matrix bound on ensemble variance)
K=8 bootstrap-refit gives variance ~1/8 = 12.5% of single-model variance; K=64 gives ~1.5%; K=599 gives ~0.17%. For ranker DECISIONS (top-3 selection), 12.5% variance MAY produce decision-flipping noise; 1.5% is firmly safe. K=64 is a defensible sweet spot.

### MacKay (memorial; operating-within: MDL → cost of ensemble vs cost of decision-flipping)
MDL framework: cost of K-fold bootstrap is K × refit_cost; cost of bad decision is large (paid dispatch on wrong candidate). For dispatches < $1, K=8 is fine; for dispatches > $10, K=64+ is justified. Per-decision adaptive K is the canonical MDL answer.

### Shannon LEAD (operating-within: H(decision | K) decreases with K)
Information content per bootstrap sample saturates with K; H(decision | K=599) ≈ H(decision | K=64) up to numerical precision. The "true" K is when adding more samples doesn't change the top-3 ranking. Empirical sweep is canonical answer.

### Dykstra CO-LEAD (operating-within: convex-feasibility on the decision-relevant coordinate)
The Rashomon disagreement signal lives in a low-dim subspace of model space; K-bootstrap recovers the subspace as K → K_intrinsic. Canonical answer: empirical kendall-tau saturation curve.

### Contrarian (operating-within: K=8 may be the right answer for cheap decisions)
K=599 raw formula assumes tight CI on the ENTIRE disagreement distribution; we may need only top-rank quality. K=8 disagreement on top-3 candidates may be sufficient. Cap=64 is canonical decision sweet spot. **Don't pursue K=599 wire-in; ratify K=64 cap and add adaptive-K-per-decision.**

### Yousfi + Fridrich (operating-within: dispatch-ranker quality IS the score-lowering oracle)
The ranker's decisions feed paid dispatch; bad rankings waste paid GPU. K-quality trade-off IS the canonical tradeoff. Per-decision adaptive K (cheap for low-cost decisions, high-K for high-cost decisions) is the canonical answer.

### Assumption-Adversary (operating-within: HARD-EARNED vs CARGO-CULTED)
- K=599 raw formula is HARD-EARNED (Rashomon set CI canonical analysis)
- "K=8 is grossly under-powered" is CARGO-CULTED (statistical under-powerment ≠ decision-quality under-powerment)
- Cap=64 sweet spot is HARD-EARNED-VS-CANONICAL-PRACTICE (Rudin et al. typical empirical range)

## Verdict + rationale

**PROCEED_WITH_REVISIONS**: dispatch cheap K-sweep ($0.50 local-CPU or Modal-T4 since refit is small); empirical kendall-tau saturation curve determines per-decision adaptive K. Adopt sweet-spot K=64 default if curve plateaus before K=64; otherwise raise default.

## Action class + next-step dispatch

**research** — local-CPU OR $0.50 Modal smoke. Outcome enables adaptive-K helper in `tac.rashomon_ensemble.adaptive_k_for_decision` (per-decision-cost-aware K selection).

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t2_finding_3_rashomon_k_under_powered_20260518", deliberation_id="finding_3_rashomon_k_under_powered", council_tier="T2", council_verdict="PROCEED_WITH_REVISIONS", cost_envelope_usd=0.50)`
- Posterior anchor via `append_council_anchor(...)`
- Probe outcome: `register_probe_outcome(probe_id="rashomon_k_kendall_tau_saturation_pending_20260518", substrate="cathedral-autopilot-ranker", verdict="DEFER", metric_name="kendall_tau_saturation_k", metric_value=8.0, threshold=599.0, evidence_path=".omx/research/arbitrariness_extinction_audit_20260518.jsonl", next_action="dispatch K-sweep with kendall-tau diagnostic")`
- MEMORY.md index entry: paired with deliberation wave landing
- Cross-references: standing-directive memory file finding #3; Wave 2A `8b987215a` row #9; Catalog #313

## Reactivation criteria

- **Saturation at K<8**: K=8 ratified; close Wave 2A row as OVER-PESSIMISTIC
- **Saturation at K∈[8,64]**: raise default to saturation point; emit adaptive-K helper
- **Saturation at K>64**: raise cap with operator-decision; emit adaptive-K helper that uses K=cap for high-cost decisions only
