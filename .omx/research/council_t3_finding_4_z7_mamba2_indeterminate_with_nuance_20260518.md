---
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Schmidhuber, Hassabis, Hafner, Tao, Hotz]
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_dissent:
  - member: Hotz
    verbatim: "NaN-explode at canonical 64-pair scale at TWO learning rates (5e-4 + 2e-4) is a STRONG signal that the Mamba-2 selective-SSM training surface is fragile. The tiny-config win (-0.0004 + -27,283 bytes) is in the noise floor — could easily be a fluke. Don't chase this; pivot to S4 or DreamerV3-RSSM which have known-stable training dynamics on dashcam-scale inputs."
  - member: Hafner
    verbatim: "DreamerV3-RSSM uses categorical latent z + GRU dynamics core — known-stable on dashcam-scale inputs. The KL-balancing + free-bits prevent posterior collapse + NaN-explode by design. Mamba-2 has no such structural stability primitives. Pivot to RSSM is the canonical answer per CC-9 reformulation."
council_assumption_adversary_verdict:
  - assumption: "Mamba-2 selective-SSM training stability is a tuning problem (grad-clip + LR-warmup will fix it)"
    classification: CARGO-CULTED
    rationale: "NaN-explode at TWO different LR (5e-4 + 2e-4) at the canonical 64-pair scale is a STRUCTURAL signal that Mamba-2's training surface has narrow stable region. Tuning grad-clip + LR-warmup MAY help but may also be band-aid; structural alternatives (S4, RWKV, DreamerV3-RSSM, FiLM-LSTM) exist with KNOWN-STABLE training surfaces."
  - assumption: "Tiny-config win (-0.0004 + -27,283 bytes) generalizes to canonical-scale"
    classification: CARGO-CULTED
    rationale: "Tiny-config NaN-explodes at canonical scale, so the win does NOT directly generalize. The win is REAL at tiny-config but may be a smoke-only artifact at the canonical scale we care about. Per CLAUDE.md 'Apples-to-apples evidence discipline': tiny-config != canonical-scale, do not extrapolate."
council_decisions_recorded:
  - "op-routable #1: dispatch grad-clip + LR-warmup paired-comparison smoke ($2-5 Modal T4) on Z7 substrate at canonical 64-pair scale: try grad_clip in {0.5, 1.0, 5.0} × LR_warmup_steps in {500, 2000, 5000}"
  - "op-routable #2: IF grad-clip + LR-warmup achieves stable training: ratify CC-9 reformulation Mamba-2; proceed to TOP-5 #2 [-0.025, -0.008] pursuit"
  - "op-routable #3: IF grad-clip + LR-warmup fails: PIVOT to per-Hafner DreamerV3-RSSM (KL-balancing + free-bits) OR per-Schmidhuber S4 (linear stable dynamics) OR FiLM-LSTM (proven-stable on dashcam-scale)"
  - "op-routable #4: design memo CC-9 reformulation: re-classify Mamba-2 from CARGO-CULTED-EMPIRICALLY-CONFIRMED-AT-DASHCAM-SCALE to CARGO-CULTED-PENDING-STABILITY-FIX-OR-PIVOT until canonical-scale stable training achieved"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: z7_mamba2
deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
predicted_mission_contribution: frontier_breaking
finding_action_class: research
finding_followup_dispatch_envelope_usd: 5.00
finding_canonical_path: experimental
---

# Finding 4: Z7-Mamba-2 INDETERMINATE_WITH_NUANCE

## What happened

Commit `c88ac969a`: Z7-Mamba-2 paired-comparison smoke reports:
- Tiny-config (8-pair, smoke): Mamba-2 wins by -0.0004 score + -27,283 bytes
- Canonical 64-pair @ LR=5e-4: NaN-explode mid-training (loss → inf at epoch ~10)
- Canonical 64-pair @ LR=2e-4: ALSO NaN-explode (loss → inf at epoch ~25)

CC-9 (cargo-cult assumption "Mamba-2 with default hyperparameters is stable on dashcam-scale inputs") was provisionally classified CARGO-CULTED-EMPIRICALLY-CONFIRMED-AT-DASHCAM-SCALE based on tiny-config win, but the canonical-scale NaN-explode INVALIDATES that classification.

## Council deliberation

### Shannon LEAD (operating-within: information-theoretic capacity of selective-SSM)
Mamba-2's selective-SSM should have capacity strictly ≥ Mamba-1 ≥ S4 (per architectural progression). Tiny-config win confirms capacity advantage. NaN-explode at canonical-scale is a STABILITY issue, not a capacity issue. Stability fixes (grad-clip + LR-warmup + maybe weight-decay) typically work for selective-SSM training; canonical scale should be achievable.

### Schmidhuber (operating-within: selective-SSM as ANN with adaptive memory)
S4/Mamba/Mamba-2 lineage uses HIPPO-ON parameterization which is provably stable IF initial conditions and step-size respect the canonical bounds. NaN-explode at canonical scale suggests initial-condition or step-size violation. Standard fixes: HIPPO re-init at the canonical scale + step-size scheduler. Likely tuning-solvable; not a structural defect.

### Hafner (operating-within: DreamerV3-RSSM proven on dashcam inputs)
DreamerV3-RSSM uses categorical latent z + GRU dynamics core — known-stable on dashcam-scale inputs. The KL-balancing + free-bits prevent posterior collapse + NaN-explode by design. Mamba-2 has no such structural stability primitives. **Pivot to RSSM is the canonical answer per CC-9 reformulation.**

### Hassabis (operating-within: risk-adjusted EV for paid-dispatch pursuit)
TOP-5 #2 predicted EV [-0.025, -0.008] is frontier-breaking; even at 50% probability of stability-fix-success, expected EV is [-0.013, -0.004] which still dominates most other candidates. Dispatch grad-clip+LR-warmup smoke at $2-5 to determine stability-fix-feasibility. If stable: pursue. If NaN-still-explodes: pivot to RSSM or S4.

### Hotz (operating-within: don't chase fragile baselines)
NaN-explode at TWO different LR (5e-4 + 2e-4) at canonical 64-pair scale is STRONG signal that Mamba-2 training surface is fragile. Tiny-config win is in noise floor; could easily be a fluke. **Pivot to S4 or DreamerV3-RSSM with known-stable training.**

### Tao (operating-within: mathematical-stability conditions for selective-SSM)
Mamba-2 selective-SSM stability requires the selective-gate to respect non-negativity + bounded-spectral-norm conditions. Default initialization may violate these at canonical scale (gradient magnitude scales with sequence length). Grad-clip is a NECESSARY but possibly insufficient fix; LR-warmup helps spectral-norm stay bounded. Empirical: try both.

### Contrarian (operating-within: tiny-config win may be over-fit to tiny-config)
8-pair smoke training has tiny sample size; -0.0004 win could be lucky initialization. NaN-explode at canonical-scale is the REAL signal: Mamba-2 training surface is too fragile for production use at this paradigm. **Tiny-config win does not generalize; refute CC-9 promotion.**

### Yousfi (operating-within: PoseNet/SegNet response to selective-SSM substrate)
Selective-SSM should produce smooth temporal continuity that matches dashcam structure; PoseNet should be well-conditioned. If stable training achieved, expect Pareto improvement. Stability IS the dominant question.

### Fridrich (operating-within: inverse-steganalysis of selective-SSM signals)
Mamba-2 latent state may concentrate signal in steganalysis-blind spectral bands; this is the architectural advantage. Realizing it requires stable training. Fall-back: S4 (linear stable) likely 80% of Mamba-2's capacity with much-improved training dynamics.

### Dykstra CO-LEAD (operating-within: convex-feasibility of training procedure)
Training procedure = composition of optimizer steps; convergence requires each step to land in stable region. NaN-explode = trajectory leaves stable region. Grad-clip enforces per-step bounded update; LR-warmup ensures initial steps are small enough to stay in stable region. Both are convex-feasibility-preserving primitives; standard practice.

### Assumption-Adversary (operating-within: HARD-EARNED vs CARGO-CULTED)
- Tiny-config win is HARD-EARNED-AT-TINY-CONFIG (empirically confirmed at the actual scale tested)
- "Win generalizes to canonical scale" is CARGO-CULTED (refuted by NaN-explode)
- "Grad-clip + LR-warmup will fix it" is HARD-EARNED-VS-CANONICAL-PRACTICE (standard fix for selective-SSM instability)
- "Pivot to S4/RWKV/DreamerV3-RSSM" is HARD-EARNED-VS-ALTERNATIVES (known-stable architectures)

## Verdict + rationale

**DEFER_PENDING_EVIDENCE**: deferred per Catalog #298 / #315 / "Forbidden premature KILL". Substrate not killed; the paradigm (selective-SSM on dashcam) is intact. The implementation (Mamba-2 default hyperparameters) is questionable. Three reactivation paths:

1. Grad-clip + LR-warmup smoke succeeds → reactivate as Mamba-2-canonical
2. Grad-clip + LR-warmup smoke fails → pivot to DreamerV3-RSSM (Hafner-recommended)
3. Pivot fallback to S4 (linear stable, Schmidhuber-canonical)

## Action class + next-step dispatch

**research** ($2-5 Modal T4 stability-fix smoke). Council DEFERs pursuit pending stability-fix outcome. If fix succeeds, escalate to T3 PROCEED for canonical-scale paid dispatch.

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t3_finding_4_z7_mamba2_indeterminate_20260518", deliberation_id="finding_4_z7_mamba2_indeterminate", council_tier="T3", council_verdict="DEFER_PENDING_EVIDENCE", predicted_impact_lower=-0.025, predicted_impact_upper=-0.008, cost_envelope_usd=5.00)`
- Posterior anchor: `append_council_anchor(CouncilDeliberationRecord(..., deferred_substrate_id="z7_mamba2", deferred_substrate_retrospective_due_utc="2026-06-17T00:00:00Z"))` per Catalog #300 mission-alignment retrospective wire-in
- Probe outcome: `register_probe_outcome(probe_id="z7_mamba2_canonical_scale_stability_20260518", substrate="z7_mamba2", verdict="DEFER", metric_name="canonical_scale_loss_convergence", metric_value=float('inf'), evidence_path="commit c88ac969a", next_action="dispatch grad-clip + LR-warmup smoke", staleness_window_days=30)`
- MEMORY.md index entry: paired with deliberation wave landing
- Cross-references: standing-directive memory file finding #4; commit `c88ac969a`; design memo CC-9 reformulation pending; Catalog #298 / #315 forbidden-premature-KILL; HNeRV parity L13 (KILL is LAST RESORT)

## Reactivation criteria (per Catalog #298)

- **Grad-clip+LR-warmup at canonical scale converges**: Mamba-2 paradigm reactivated as CC-9-REFORMULATED-CONDITIONALLY; per-substrate symposium per Catalog #325 with stability-fix as design-memo §3 caveat
- **Pivot to DreamerV3-RSSM**: new substrate scaffold per CC-9-PIVOT-1; Hafner's KL-balancing + free-bits architecture; expected EV closer to upper bound
- **Pivot to S4 (linear stable)**: new substrate scaffold per CC-9-PIVOT-2; Schmidhuber's HIPPO-canonical; expected EV ~80% of upper bound
- **All fail**: substrate KILL with paradigm-vs-implementation classification per Catalog #307 (paradigm = selective-SSM-on-dashcam survives; only implementations falsified) + N>=3 alternative probe methodologies per Catalog #308
