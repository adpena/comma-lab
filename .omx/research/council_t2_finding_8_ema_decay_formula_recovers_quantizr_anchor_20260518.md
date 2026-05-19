---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Polyak, TarvainenValpola, Quantizr]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "decay = 1 - 1/(0.2 * total_steps) recovers Quantizr 0.997 anchor at N=1666"
    classification: HARD-EARNED
    rationale: "TOP-1 + TOP-4 batched empirical recovery in commit 6db94d9ea verified the formula at N=1666 produces 0.997. Quantizr's 0.33 archive used decay=0.997 implicitly; formula extraction reverse-engineers + canonicalizes."
  - assumption: "Every substrate trainer should consume tac.training.EMA.decay_from_total_steps"
    classification: HARD-EARNED-VS-CANONICAL-CONSOLIDATION
    rationale: "EMA decay 0.997 IS a canonical CLAUDE.md non-negotiable. Per Catalog #88 ema-correctness check, every training path must instantiate EMA correctly. Canonical formula-based decay is the right canonical."
council_decisions_recorded:
  - "op-routable #1: wire `tac.training.EMA.decay_from_total_steps(total_steps)` into substrate trainer init"
  - "op-routable #2: per Catalog #88 ema-correctness check: regression test that decay value matches formula at total_steps=1666"
  - "op-routable #3: per-substrate compounding effect (small but present): EV [-0.003, -0.001] per substrate × 14 = [-0.042, -0.014] composite"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: frontier_protecting
finding_action_class: pursue
finding_followup_dispatch_envelope_usd: 0.00
finding_canonical_path: formula
---

# Finding 8: decay = 1 - 1/(0.2*total_steps) recovers Quantizr 0.997 at N=1666

## What happened

TOP-1 + TOP-4 batched empirical recovery in commit `6db94d9ea`: closed-form formula `decay = 1 - 1/(0.2 * total_steps)` at total_steps=1666 produces 0.997, matching Quantizr's 0.33 archive's empirical EMA decay anchor. The canonical EMA decay formula is empirically validated.

## Council deliberation

### Polyak memorial (operating-within: stochastic-averaging convergence to optimum)
EMA decay determines mixing time of stochastic average. Polyak averaging with decay = 1 - 1/(c*N) for c~0.2 is canonical practice; produces averaging over ~0.2*N most recent steps. Matches Tarvainen-Valpola Mean Teacher empirical sweet spot.

### Tarvainen-Valpola (operating-within: Mean Teacher EMA)
Mean Teacher EMA at decay=0.997 corresponds to effective window ~333 steps. Per-substrate training at 1666 steps → window/N ≈ 0.2 → empirical sweet spot for student-teacher consistency. **Canonical Mean Teacher anchor matches.**

### Quantizr (operating-within: my 0.33 archive at N=1666 used decay=0.997)
Confirmed; my training used exactly this regime. Canonical formula extraction reverse-engineers + canonicalizes; future substrates should consume.

### Shannon LEAD (operating-within: information-rate of stochastic averaging)
EMA decay sets the effective bandwidth of averaging filter. 1/(0.2*N) is the canonical low-pass cutoff for substrate training. Decay-from-total-steps formula is the right canonical helper.

### Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary: AGREE
Canonical formula adoption per Catalog #88 + Catalog #290.

## Verdict + rationale

**PROCEED**: wire canonical decay formula. Editor-only ($0); per-substrate compounding. EV small per substrate but compounds across 14 substrates × Catalog #88 already-mandatory EMA wiring.

## Action class + next-step dispatch

**pursue** (editor + tests; $0 GPU). Per-substrate wire-in (mostly already done via Catalog #88; this gate canonicalizes the formula-derivation).

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t2_finding_8_ema_decay_formula_recovers_quantizr_20260518", deliberation_id="finding_8_ema_decay_formula_recovers_quantizr", council_tier="T2", council_verdict="PROCEED", predicted_impact_lower=-0.042, predicted_impact_upper=-0.014, cost_envelope_usd=0.00)`
- Posterior anchor via `append_council_anchor(...)`
- Probe outcome: `register_probe_outcome(probe_id="ema_decay_formula_recovers_quantizr_20260518", substrate="multi-substrate-ema-canonical-wire-in", verdict="PROCEED", metric_name="decay_at_n_1666", metric_value=0.997, threshold=0.997, evidence_path="commit 6db94d9ea", next_action="per-substrate canonical-formula wire-in")`
- MEMORY.md index entry: paired with deliberation wave landing
- Cross-references: standing-directive memory file finding #8; commit `6db94d9ea`; CLAUDE.md "EMA — non-negotiable"; Catalog #88; Catalog #290

## Reactivation criteria

- Substrate-specific divergence: PRINCIPLED-FORK with documented rationale
- Empirical regression after wire-in: revert + investigate
