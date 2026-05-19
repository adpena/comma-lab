---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Boyd, Hassabis]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Hassabis
    verbatim: "SGLD t_final 17.4× current default is a budget statement, not a quality statement. If t_final=1.0 already reaches the loss plateau on the actual substrate posterior, running 17× longer wastes $$$ without quality improvement. The analytical formula's raw value is correct given posterior temperature; the ceiling-cap is correct given budget. The right answer is empirical convergence-diagnostic, not formula vs cap."
council_assumption_adversary_verdict:
  - assumption: "SGLD analytical formula t_final = 17.4 reflects the actual stack_of_stacks posterior convergence requirement"
    classification: CARGO-CULTED
    rationale: "The analytical formula uses canonical posterior parameters; the actual stack_of_stacks substrate posterior may have HEAVIER tails (slower convergence → t_final too LOW) OR LIGHTER tails (faster convergence → t_final correctly capped at 1.0). Without empirical convergence diagnostic, the 17.4× claim is unverified."
  - assumption: "Ceiling-cap at 1.0 reflects the actual cost-band budget for stack_of_stacks dispatch"
    classification: HARD-EARNED
    rationale: "Cost-band budgets are operator-set and reflect actual paid GPU constraints. The cap IS hard-earned."
council_decisions_recorded:
  - "op-routable #1: dispatch convergence-diagnostic smoke ($0.50-1.50 Modal T4): SGLD on stack_of_stacks substrate, log loss every 0.1*t_final to t_final=2.0; identify plateau"
  - "op-routable #2: if plateau at t_final<1.0: cap is CORRECT and analytical formula is over-estimating; mark Wave 2A row as REGIME_TRANSFER_INVALID"
  - "op-routable #3: if plateau at t_final>1.0 (toward 17.4): raise cost-band cap OR add SGLD-budget-aware operator-decision before dispatch"
  - "op-routable #4: emit canonical convergence-diagnostic helper to tac.solvers per MORE-OPTIMAL-ALGORITHMS pattern; future SGLD callers consume diagnostic to auto-cap"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: rigor_overhead
finding_action_class: research
finding_followup_dispatch_envelope_usd: 1.50
finding_canonical_path: analytical_solve
---

# Finding 2: SGLD t_final raw 17.4 → ceiling-cap 1.0

## What happened

Wave 2A `8b987215a` row #8 (stack_of_stacks substrate): SGLD posterior sampling analytical t_final = 17.4 (canonical formula given posterior temperature + chain length to MCMC convergence within tight CI). Ceiling-cap at 1.0 was applied per cost-band budget constraints. **17.4× discrepancy** between analytical formula and applied value.

## Council deliberation

### Shannon LEAD (operating-within: posterior entropy-rate per Langevin diffusion)
SGLD convergence to stationary distribution requires t_final >> mixing time. Canonical formula gives mixing time per posterior temperature; 17.4 = mixing time / chosen step size for tight CI. If the actual posterior has faster mixing than canonical, 1.0 may be sufficient. The formula is correct under canonical assumptions; the cap is correct under budget constraints — these answer DIFFERENT questions.

### Dykstra CO-LEAD (operating-within: convergence-diagnostic IS the alternating-projection feasibility)
SGLD convergence is mathematically a fixed-point iteration. Diagnose convergence empirically (loss plateau / gradient norm decay / autocorrelation time). Convergence diagnostic IS the canonical feasibility check; running longer than necessary wastes resources, running shorter than necessary produces non-converged samples. The cap-vs-formula tension is resolved by **empirical diagnostic, not formula adoption**.

### Contrarian (operating-within: budget cap may be MASKING under-converged samples)
If the actual posterior needs t_final=17.4 and we cap at 1.0, samples are NOT from stationary distribution. Wave-3 dispatches downstream of SGLD samples then produce DOWNSTREAM artifacts based on biased samples — invisible bug class. **Convergence diagnostic is MANDATORY before adoption of any SGLD result as ground truth.**

### Assumption-Adversary (operating-within: HARD-EARNED vs CARGO-CULTED)
- Cap=1.0 is HARD-EARNED (cost-band budget)
- Formula=17.4 is HARD-EARNED-VS-CANONICAL-ASSUMPTIONS (correct given parameters)
- "17.4× MISMATCH IMPLIES THE CAP IS WRONG" is CARGO-CULTED (the cap may be correct given actual posterior mixing time)
- The Wave 2A row reports BOTH values but doesn't connect them empirically

### Boyd (operating-within: empirical convergence diagnostic from convex-optimization lens)
For SGLD on convex (or log-concave) posteriors, mixing time has closed-form bounds; for non-convex (substrate score-aware posterior), mixing time is empirical. The 17.4 formula likely assumes log-concave; actual posterior is non-convex. Both formula and cap are upper bounds (formula on time, cap on cost); the EMPIRICAL convergence point is the canonical answer.

### Hassabis (operating-within: budget-aware research vs perfect-Bayes pursuit)
SGLD t_final 17.4× is a budget statement. If t_final=1.0 reaches the loss plateau on the actual substrate posterior, running 17× longer wastes $$$. The right answer is empirical convergence-diagnostic, not formula vs cap. **Dispatch the diagnostic; let the data arbitrate.**

### Yousfi (operating-within: deviating from canonical = inverse-steganalysis risk)
Under-converged SGLD samples introduce systematic bias in archive bytes; sophisticated steganalysis (PoseNet/SegNet scorers) may detect the bias. Need to ensure SGLD samples are converged enough to be statistically indistinguishable from true posterior samples; the diagnostic IS the test.

### Fridrich (operating-within: cap may be CORRECT if substrate posterior is shallow)
Stack_of_stacks substrate is a composition of pre-trained components; the posterior may be SHALLOW (small KL divergence from prior). Shallow posteriors mix fast; SGLD at t_final=1.0 may suffice. Or may NOT. Diagnostic is the only answer.

## Verdict + rationale

**PROCEED_WITH_REVISIONS**: dispatch convergence-diagnostic smoke (~$0.50-1.50) BEFORE adopting either value. Council is unanimous that convergence diagnostic is canonical answer; formula and cap answer different questions.

## Action class + next-step dispatch

**research** — $0.50-1.50 Modal T4 convergence-diagnostic smoke. Outcome:
- Plateau at t_final < 1.0: cap is correct; Wave 2A formula assumption invalid for substrate-posterior
- Plateau at t_final > 1.0: cap is binding; raise budget OR document under-convergence as systematic bias
- Plateau matches formula ~17.4: cap is wrong; raise budget OR pivot to faster-mixing sampler

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t2_finding_2_sgld_t_final_ceiling_cap_20260518", deliberation_id="finding_2_sgld_t_final_ceiling_cap", council_tier="T2", council_verdict="PROCEED_WITH_REVISIONS", predicted_impact_lower=-0.005, predicted_impact_upper=-0.001, cost_envelope_usd=1.50)`
- Posterior anchor via `append_council_anchor(...)`
- Probe outcome: `register_probe_outcome(probe_id="sgld_t_final_convergence_diagnostic_pending_20260518", substrate="stack_of_stacks", verdict="DEFER", metric_name="t_final_convergence", metric_value=1.0, threshold=17.4, evidence_path=".omx/research/arbitrariness_extinction_audit_20260518.jsonl", next_action="dispatch convergence-diagnostic smoke")`
- MEMORY.md index entry: paired with deliberation wave landing
- Cross-references: standing-directive memory file finding #2; Wave 2A `8b987215a` row #8; Catalog #233 promotion gate; Catalog #313 probe outcomes ledger

## Reactivation criteria

- **Convergence at t_final<1.0**: ratify cap, mark Wave 2A formula assumption INVALID for substrate-posterior
- **Convergence at t_final∈[1.0, 17.4)**: raise cap to empirical plateau; mark formula as upper-bound (not tight)
- **Convergence at t_final≥17.4**: raise cap to 20.0 OR pivot to faster-mixing sampler (HMC/NUTS); operator-decision required (cost-band reclass)
