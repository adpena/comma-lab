---
council_tier: T3
council_attendees: [Shannon, Dykstra, Schmidhuber, Hassabis, Hafner, Tao, Hotz, Contrarian, Assumption-Adversary]
council_quorum_met: false
council_verdict: DRAFT_PENDING_OPERATOR_CONVOCATION
council_dissent:
  - member: Contrarian
    verbatim: "DRAFT only — full T3 convocation requires operator-attention budget per Catalog #300. Per-substrate symposium per Catalog #325 14-day window starts at convocation, not at DRAFT landing."
  - member: Hotz
    verbatim: "DRAFT MUST not silently bypass Wave N+1 council ratification. The DRAFT is the agenda, not the verdict."
council_assumption_adversary_verdict:
  - assumption: "Mamba-2 stability fix at canonical 64-pair scale will land within Wave N+2 9-config sweep"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Per parent T3 finding 4 Assumption-Adversary: NaN at TWO LRs is structural; tuning MAY band-aid. Wave N+2 empirical anchor required."
  - assumption: "Wave N+2 9-config sweep cost $5 is correct"
    classification: HARD-EARNED-PARTIAL
    rationale: "Modal T4 at ~$0.55/hr × 9 configs × 100ep ≈ 18 min/config = $1.50; sweep total $13.50 NOT $5. Sweep cost-estimate in path-forward memo UNDER-ESTIMATES by ~$8. RECOMMEND adjust to $15 envelope; sweep stays affordable but operator should approve correct envelope."
  - assumption: "predictive-coding-recurrent paradigm class-shift survives Wave N+3 path B"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "IF Mamba-2 + FiLM-LSTM both fail, predictive-coding-recurrent paradigm DEFER per Catalog #298. Per Catalog #308 N>=3 alternative-probe-methodologies: pivot to NeRV-family predictive-coding-without-recurrent-state OR foveation IDEAS without recurrent state. Operator should know this pivot exists BEFORE committing budget."
council_decisions_recorded:
  - "DRAFT enumerates 6-step Catalog #325 contract"
  - "Recommended Wave N+2 envelope: $15 (corrected per Assumption-Adversary)"
  - "Operator-routable: ratify DRAFT via full T3 convocation; current week's T3 cadence per tools/audit_council_tier_cadence.py"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: z7_mamba2
deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
horizon_class: asymptotic_pursuit
---

# DRAFT: T3 grand council symposium — Z7-Mamba-2 stability path forward

**Status**: DRAFT — operator-convocation pending. NOT a binding council verdict.
**Lane**: `lane_z7_mamba_2_stability_multi_week_path_forward_20260518` L1
**Per Catalog #325**: this DRAFT satisfies the 6-step contract structurally; full convocation activates symposium evidence per Catalog #325 14-day window.

## Symposium attendees (proposed)

**Sextet pact** (REQUIRED per CLAUDE.md "Council conduct" amendment):
- **Shannon LEAD** — information-theoretic capacity of stability-fix vs pivot
- **Dykstra CO-LEAD** — convex-feasibility of training procedure (per parent T3 verbatim)
- **Yousfi** — PoseNet/SegNet response to selective-SSM substrate (parent T3 verbatim retained)
- **Fridrich** — inverse-steganalysis of selective-SSM vs LSTM signals (parent T3 verbatim retained)
- **Contrarian** — VETO power on lazy consensus
- **Assumption-Adversary** — per-round shared-assumption-violation hypothesis (Catalog #291 + #292)

**Grand council attendees added per topic** (per Catalog #300 v2 + CLAUDE.md "Grand Council (advisory)"):
- **Schmidhuber** — HIPPO-ON stability for SSM family + S4 lineage
- **Hassabis** — risk-adjusted EV for paid-dispatch pursuit
- **Hafner** — DreamerV3-RSSM canonical stability primitive (KL-balancing + free-bits)
- **Tao** — selective-SSM stability bounds + spectral-norm conditions
- **Hotz** — engineering shortcuts + don't-chase-fragile-baselines
- **Atick + Redlich** — cooperative-receiver framing applicability to stability-fix vs pivot
- **Rao + Ballard** — predictive-coding architectural primacy across all 5 candidates
- **Tishby memorial + Zaslavsky** — IB-Lagrangian β-parameter inheritance from C6 IBPS Phase 2
- **Wyner** — Wyner-Ziv side-info preservation across pivot candidates

## Step 1 — Cargo-cult audit per Catalog #303

Per `## Cargo-cult audit per assumption` discipline. Cross-references sibling design-space memo §2 + parent design memo §2 + parent T3 finding 4 Assumption-Adversary verdict.

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| CC-stab-1 | "grad-clip + LR-warmup will fix Mamba-2 NaN-explode at canonical 64-pair scale" | CARGO-CULTED-PENDING-EMPIRICAL | Wave N+2 9-config sweep (grad_clip ∈ {0.5, 1.0, 5.0} × LR_warmup ∈ {500, 2000, 5000}) IS the empirical disambiguator. |
| CC-stab-2 | "Predicted ΔS [-0.025, -0.008] transfers from language SSM benchmarks to dashcam 600-pair sequence" | CARGO-CULTED-PENDING-EMPIRICAL | Wave N+3 empirical anchor on stability-fixed Mamba-2 OR pivot candidate IS the disambiguator. |
| CC-stab-3 | "Wave N+2 sweep cost $5" | HARD-EARNED-PARTIAL-CORRECTED-TO-$15 | Per Assumption-Adversary above: Modal T4 9 × 18 min ≈ $13.50; corrected envelope $15. |
| CC-stab-4 | "If Mamba-2 fails, FiLM-LSTM is best pivot per highest P(success)" | CARGO-CULTED-PENDING-EMPIRICAL | FiLM-LSTM P(success) 90-95% is sister-class proven (Quantizr 0.33 / PR95 0.193 lineage); ΔS band [-0.012, -0.003] is sister-territory of Z6-v1 75K-FiLM. Empirical anchor required for novel substrate. |
| CC-stab-5 | "DreamerV3-RSSM > Mamba-2 expected EV per Hafner verbatim" | CARGO-CULTED-PENDING-EMPIRICAL | Hafner verbatim is informed by world-model domain; dashcam contest extrapolation is research prior, not empirical. |
| CC-stab-6 | "predictive-coding-recurrent paradigm is sound architecture for dashcam contest temporal coherence" | CARGO-CULTED-PENDING-EMPIRICAL (META) | Sister to parent design memo §14 Assumption-Adversary item #8 hypothesis. PR106 format0d_latent_score_table (lane `pr106_format0d_latent_score_table`; archive sha 9cb989cef519) at 0.20533 [contest-CUDA] IS canonical CUDA frontier + STATELESS decoder. Per Catalog #316 frontier preservation: recurrent-state-as-winning-pattern is operator's working theory + needs validation. |

## Step 2 — 9-dimension success checklist evidence per Catalog #294

| # | Dimension | Per-symposium-DRAFT evidence |
|---|---|---|
| 1 | UNIQUENESS | ✓ Stability-fix preserves Mamba-2 architectural uniqueness (selective state-space at d_state=16) within asymptotic_pursuit class. Pivot candidates each enter their own substrate-class-shift territory. |
| 2 | BEAUTY + ELEGANCE | ✓ Stability fix = 2 torch one-liners + 2 CLI flags + LR scheduler. Wave N+2 commit ~50 LOC. Pivot to FiLM-LSTM ~3 days editor; per HNeRV parity L4 substrate-engineering waiver applies. |
| 3 | DISTINCTNESS | ✓ Each of 5 candidates is architecturally orthogonal per sibling design-space memo §candidate breakdown. |
| 4 | RIGOR | ✓ THIS DRAFT + sibling design-space + sibling path-forward = 3 memos × cargo-cult audit + observability surface + Dykstra-feasibility + horizon-class + canonical-vs-unique decision per layer + Catalog #229 PV + Catalog #313 probe-ledger consultation + Catalog #302 sister coordination. |
| 5 | OPTIMIZATION PER TECHNIQUE | ✓ Wave N+2 grad-clip + LR-warmup IS the Mamba-2-optimal stability fix per Tao + Dykstra + Schmidhuber convergent verbatim. Pivot candidates each get their own per-substrate symposium per Catalog #325. |
| 6 | STACK-OF-STACKS-COMPOSABILITY | ✓ Z7-stability-winner + NSCS06v8 chroma + DP1 pretraining + D1 SegNet overlay (4 orthogonal axes per design memo §3.6 Dimension 6 inheritance). |
| 7 | DETERMINISTIC REPRODUCIBILITY | ✓ Z7MCM2 archive byte-stable per Catalog #5 + Mamba-2 deterministic unroll + grad-clip + LR-warmup are deterministic primitives (no stochastic component). Pivot candidates similarly deterministic except DreamerV3-RSSM stochastic categorical z (Wave N+4 promotion concern). |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | ✓ Wave N+2 9-config sweep is parallelizable on Modal T4 (9-instance fanout); wall-clock ~20 min/instance + ~5 min harvest. Empirical optimization budget $15. |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | INDETERMINATE — per Catalog #316 + parent design memo Dimension 10: predicted band [0.167, 0.184] [contest-CPU prediction] sits BELOW current canonical frontier 0.19205 [contest-CPU] IF realized. Wave N+3/N+4 empirical anchor required for promotion claim. |

## Step 3 — Observability surface declaration per Catalog #305

**Per-stability-fix-candidate observability** (Mamba-2 stability-fix):
1. **Inspectable per layer**: per-epoch loss decomposition (segnet + posenet + rate) + per-pair Mamba-2 hidden state norm + selectivity matrix range + grad-clip-norm-actual (sister monitor: how often does clip fire?) + LR-warmup-progress
2. **Decomposable per signal**: per-config (grad_clip × LR_warmup) convergence vs NaN binary verdict + per-pair predictor residual magnitude
3. **Diff-able across runs**: 9 configs × identity-control = 18 paired runs; diff via canonical Z7MCM2 archive byte-stable + Mamba-2 deterministic unroll
4. **Queryable post-hoc**: per-config Modal call_id ledger row per Catalog #245 + per-config probe-outcome ledger row per Catalog #313
5. **Cite-able**: 9 paired comparisons cite parent T3 finding 4 + sibling design-space memo + this DRAFT
6. **Counterfactual-able**: "what if grad_clip=0.5 vs 1.0 vs 5.0 at fixed LR_warmup?" + "what if LR_warmup=500 vs 2000 vs 5000 at fixed grad_clip?" — 9-config sweep IS the counterfactual matrix

**Per-pivot-candidate observability**: each pivot candidate (S4 / RWKV-7 / DreamerV3-RSSM / FiLM-LSTM) gets its own per-substrate symposium per Catalog #325; observability surface declaration deferred to that symposium.

## Step 4 — Sextet pact deliberation (DRAFT positions)

### Shannon LEAD position (DRAFT)

*"Operating-within assumption: Mamba-2's selective-SSM capacity is strictly ≥ Mamba-1 ≥ S4 (architectural progression). The information-theoretic question is whether stability-fix changes the achievable rate-distortion frontier — answer: NO. Grad-clip bounds per-step update magnitude; LR-warmup bounds initial-step magnitude. Neither changes the substrate's capacity. Wave N+2 IS the capacity-realization disambiguator: if Mamba-2 trains stably with stability-fix → capacity realized → ΔS prediction tractable; if NOT → pivot is canonical."*

### Dykstra CO-LEAD position (DRAFT)

*"Operating-within assumption: training procedure = composition of optimizer steps; convergence requires each step to land in stable region. Grad-clip + LR-warmup are convex-feasibility-preserving primitives that ENFORCE each step's stable-region residency. Sister Atick-Redlich cooperative-receiver framing: stability-fix preserves the encoder-decoder shared-prior channel (Mamba-2 hidden state); pivot to non-recurrent architectures DROPS this channel. The architectural-class question is whether the cooperative-receiver channel is necessary for dashcam contest score-lowering; Wave N+2 + N+3 sweep determines this."*

### Hafner position (DRAFT)

*"Operating-within assumption: DreamerV3-RSSM is proven on dashcam-scale inputs (per arxiv 2301.04104). KL-balancing + free-bits prevent posterior collapse + NaN-explode BY DESIGN. Mamba-2 has no such structural stability primitives. STRONG RECOMMENDATION: skip Wave N+2 grad-clip sweep; pivot directly to DreamerV3-RSSM. The $15 sweep IS rigor_overhead; the $30 DreamerV3-RSSM dispatch IS frontier_breaking."* COUNTER-PROPOSAL.

### Schmidhuber position (DRAFT)

*"Operating-within assumption: S4 HIPPO-ON parameterization is provably stable per Gu 2021 arxiv 2111.00396. If Mamba-2 fragility is selective-gate-specific, S4 fallback gives ~80% capacity at proven stability. Three-stage sequencing: (1) Wave N+2 grad-clip sweep (cheap; preserves Mamba-2 if works), (2) Wave N+3 S4 pivot if (1) fails, (3) Wave N+4 DreamerV3-RSSM if (2) ALSO fails. Pivot-class-order should be S4 → DreamerV3-RSSM, not the reverse."* COUNTER-PROPOSAL on pivot ordering.

### Hassabis position (DRAFT)

*"Operating-within assumption: risk-adjusted EV. Wave N+2 sweep at $15 with P(success)=50-60% per Tao + Dykstra; expected EV [-0.013, -0.005] still dominates most alternative candidates. Approve Wave N+2. IF Wave N+2 succeeds → Wave N+3 path A full dispatch $20-30 (high-EV continuation). IF Wave N+2 fails → Wave N+3 path B FiLM-LSTM pivot $5-15 (low-cost / high-P(success) / low-EV; produces empirical anchor for predictive-coding-recurrent paradigm). Reject Hafner's skip-Wave-N+2 counter-proposal; the $15 sweep IS worth the disambiguation."*

### Hotz position (DRAFT)

*"Operating-within assumption: don't chase fragile baselines. NaN at TWO LRs is STRUCTURAL signal that Mamba-2's training surface is too fragile for production use at this paradigm. Tiny-config win is in noise floor. STRONG RECOMMENDATION: skip Mamba-2 entirely; FiLM-LSTM pivot Wave N+3 path B is the canonical answer (cheapest + highest P(success) + produces empirical anchor for paradigm). The $15 Wave N+2 sweep IS sunk cost on architectural-fragility hypothesis."* COUNTER-PROPOSAL.

### Tao position (DRAFT)

*"Operating-within assumption: Mamba-2 selective-SSM stability requires selective-gate non-negativity + bounded-spectral-norm. Default initialization may violate these at canonical scale (gradient magnitude scales with sequence length 600). Grad-clip NECESSARY but possibly insufficient; LR-warmup helps spectral-norm stay bounded. Empirically testable via 9-config sweep. APPROVE Wave N+2."*

### Contrarian position (DRAFT)

*"Operating-within assumption: tiny-config 8-pair win may be over-fit to tiny-config. NaN at canonical-scale is the REAL signal: Mamba-2 training surface is too fragile. Wave N+2 sweep at $15 IS worth disambiguation BUT operator MUST be prepared for pivot per Catalog #298. The Wave N+3 path B FiLM-LSTM pivot is the canonical fallback per CLAUDE.md 'Forbidden premature KILL'. VETO any Wave N+1 council verdict that doesn't enumerate the pivot path explicitly."*

### Assumption-Adversary position (DRAFT) [Catalog #291 + #292]

*"Operating-within assumption (META): the entire predictive-coding-recurrent paradigm assumes recurrent-state is the winning bit-allocation strategy on dashcam temporal coherence. PR106 format0d_latent_score_table (STATELESS decoder; canonical CUDA frontier 0.20533) is HARD-EARNED counter-evidence. IF the recurrent-state hypothesis is wrong, the entire Z6/Z7/Z8 asymptotic_pursuit budget allocation per HORIZON-CLASS Consequence 2 ($30-50/month minimum) is mis-allocated. RECOMMENDATION: Wave N+2 + N+3 must include explicit comparison to PR106 format0d frontier; if Mamba-2 stability-fixed + FiLM-LSTM both fail to beat PR106, the predictive-coding-recurrent paradigm DEFER per Catalog #298 (NOT KILL) → reactivation criterion = NeRV-family predictive-coding-without-recurrent-state OR foveation IDEAS without recurrent state. The Catalog #313 probe-outcomes ledger MUST record this paradigm-level claim AS the disambiguator question."* — VETO if not engaged.

## Step 5 — Per-substrate reactivation criteria pinned per Catalog #298 + #308

Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

| Wave N+ | If verdict | Reactivation path |
|---|---|---|
| N+2 | ALL 9 configs NaN | Pivot to Wave N+3 path B (FiLM-LSTM); preserve Mamba-2 paradigm as DEFERRED-pending-research per Catalog #298 |
| N+3 path A | Z7-Mamba-2 full dispatch score WORSE than PR101 frontier 0.19205 | Wave N+4 composition (Z7-Mamba-2 + NSCS06v8 + DP1 + D1); IF still WORSE → DEFER substrate per Catalog #298 |
| N+3 path B | FiLM-LSTM ALSO fails | Pivot to (b) S4 OR (c) RWKV-7 OR (d) DreamerV3-RSSM per Catalog #308 N>=3; predictive-coding-recurrent paradigm STILL alive |
| N+4 | All 5 candidates fail to beat PR101 | Predictive-coding-recurrent paradigm DEFER per Catalog #298; reactivation criterion = "NeRV-family predictive-coding-without-recurrent-state OR foveation IDEAS without recurrent state lands empirical anchor beating PR101" |

## Step 6 — Catalog #324 post-training Tier-C validation discipline

Recipe declares `predicted_band_validation_status: pending_post_training`. Reactivation criterion: post-training Tier-C density measurement on Z7-Mamba-2 archive after Wave N+3 path A full dispatch completes. Per Catalog #324: predicted_band [0.167, 0.184] is research prior; promotion-eligible only after `validated_post_training` status.

## Operator-routable decisions (for full convocation)

1. **Ratify DRAFT into full T3 convocation** — operator-attention budget ≤3 T3s/week per Catalog #300; current week's T3 cadence per `tools/audit_council_tier_cadence.py`
2. **Approve Wave N+2 $15 envelope** (CORRECTED from $5 per Assumption-Adversary)
3. **Decide pivot ordering**: Hafner's skip-Wave-N+2-go-DreamerV3-RSSM ($30) vs Schmidhuber's Mamba-2-then-S4-then-DreamerV3-RSSM ($15 + $10 + $30 = $55 sequential) vs Hassabis's Mamba-2-then-FiLM-LSTM ($15 + $15 = $30 sequential) vs Hotz's skip-Mamba-2-go-FiLM-LSTM ($15 pivot)
4. **Acknowledge Assumption-Adversary META veto**: if NOT engaged with PR106 format0d frontier comparison, deliberation INVALID per Catalog #292
5. **Cross-substrate sequencing**: Z7-GRU Wave 2 council convocation in parallel (independent T3 slot); C6 IBPS Phase 2 β-anchor convocation in parallel

## Per CLAUDE.md "Forbidden premature KILL"

NO KILL VERDICT. DRAFT enumerates 5 candidates × 4 waves × multiple reactivation paths. Per Catalog #308 N>=3 alternative-probe-methodologies for paradigm-level claim. Per Catalog #298 30-day staleness window per probe outcome.

## 6-hook wire-in declaration per Catalog #125

Inherited from sibling memos (sensitivity-map N/A; Pareto constraint ACTIVE; bit-allocator N/A; cathedral autopilot dispatch ACTIVE; continual-learning posterior ACTIVE via `append_council_anchor` on full convocation; probe-disambiguator ACTIVE via Wave N+2 sweep).

## Catalog #229 PV

Inherited from sibling design-space memo PV-0 through PV-4.

## Atom emission per Catalog #245/#323

DEFERRED to full convocation (NOT a binding council verdict at DRAFT). On convocation: `build_council_deliberation_atom(atom_id="council_t3_z7_mamba_2_stability_path_forward_<convocation_utc>", deliberation_id="council_t3_z7_mamba_2_stability_path_forward", council_tier="T3", council_verdict="<ratified_verdict>", predicted_impact_lower=-0.030, predicted_impact_upper=-0.003, cost_envelope_usd=<approved_envelope>, memory_path=".omx/research/council_t3_z7_mamba_2_stability_path_forward_symposium_<convocation_utc>.md")`

## Cross-references

- Parent T3 council: `.omx/research/council_t3_finding_4_z7_mamba2_indeterminate_with_nuance_20260518.md`
- Parent design memo: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- Sibling design-space memo: `.omx/research/z7_mamba_2_stability_design_space_20260518.md`
- Sibling path-forward memo: `.omx/research/z7_mamba_2_multi_week_path_forward_20260518.md`
- Predecessor audit: `.omx/research/3_smoke_paid_gpu_reactivation_wave_blocker_audit_20260519T042301Z.md`
- Z7 parent symposium 2026-05-17: `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md`
- Mamba-2 reference: arxiv 2405.21060 (Dao-Gu 2024)
- DreamerV3 reference: arxiv 2301.04104 (Hafner 2023)
- S4 reference: arxiv 2111.00396 (Gu 2021)
- RWKV-7 reference: arxiv 2503.14456 (Peng 2025)
- FiLM reference: arxiv 1709.07871 (Perez 2017)
- LSTM reference: Hochreiter-Schmidhuber 1997
- Canonical frontier per Catalog #316: PR101 0.19205 [contest-CPU] + PR106 format0d 0.20533 [contest-CUDA]
- CLAUDE.md non-negotiables: "Council hierarchy: 4-tier protocol", "Council conduct" sextet pact + Fix-7 amendment, "META-ASSUMPTION ADVERSARIAL REVIEW", "Forbidden premature KILL", "Apples-to-apples evidence discipline"
