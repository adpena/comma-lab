---
name: z7-mamba-2-stability-design-space-20260518
council_tier: T1
council_attendees: [Hafner, Schmidhuber, Tao, Hotz, Hassabis, Contrarian, Assumption-Adversary]
council_quorum_met: false
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Hotz
    verbatim: "NaN-explode at TWO LRs is structural fragility, not a tuning problem. Rank pivots by stability-track-record, not by Mamba-2 hype."
  - member: Contrarian
    verbatim: "All 5 candidates have research-prior bands but ZERO empirical anchors on this contest at the canonical scale. Bands are CARGO-CULTED-PENDING per Catalog #303."
council_assumption_adversary_verdict:
  - assumption: "grad-clip + LR-warmup will fix Mamba-2 instability"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Sister T3 finding 4 Assumption-Adversary verbatim: NaN at TWO LRs is STRUCTURAL. Tuning MAY work but pivot candidates exist with KNOWN-STABLE training surfaces."
  - assumption: "predicted ΔS bands for S4/RWKV/DreamerV3-RSSM/FiLM-LSTM transfer from language/world-model benchmarks to dashcam contest 600-pair"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Per CLAUDE.md FORBIDDEN_PATTERNS docstring-overstatement-trap: cross-domain ΔS transfer is research prior, not empirical evidence. Each candidate needs paired smoke before promotion."
council_decisions_recorded:
  - "5 stability candidates enumerated with comparative cost/risk/EV/reactivation"
  - "Recommended sequencing: (a) MPS-proxy probe-disambiguator at $0 → (b) lowest-cost candidate empirical → escalate"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: z7_mamba2
deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
horizon_class: asymptotic_pursuit
---

# Z7-Mamba-2 stability design space — 5 candidates ranked

**Lane**: `lane_z7_mamba_2_stability_multi_week_path_forward_20260518` L1
**Parent**: `.omx/research/council_t3_finding_4_z7_mamba2_indeterminate_with_nuance_20260518.md`
**Anchor**: commit `c88ac969a` Z7-Mamba-2 NaN-explode at canonical 64-pair scale (LR=5e-4 AND LR=2e-4 both NaN); tiny-config (-0.0004 + -27,283 bytes) wins but doesn't generalize.
**Predicted ΔS basis**: deep-research wave `.omx/research/comprehensive_research_wave_20260518.md` TOP-5 #2 [-0.025, -0.008] over PR101 frontier 0.19205 [contest-CPU].

## TL;DR (60 sec)

5 candidates, 3 stability classes:
- **STABILITY-FIX-CLASS** (Mamba-2 paradigm INTACT): (a) grad-clip + LR-warmup
- **ARCHITECTURAL-SIBLING-CLASS** (sub-quadratic SSM family preserved): (b) S4, (c) RWKV-7
- **PIVOT-CLASS** (predictive-coding paradigm preserved, recurrence primitive swapped): (d) DreamerV3-RSSM, (e) FiLM-LSTM

Recommended sequencing per Race-mode Rule 3 (cheap signal gates expensive): try (a) FIRST at $5 (9 configs × Modal T4 smoke); pivot to (d) or (e) only if (a) ALL diverge.

## 5 candidates

### (a) grad-clip + LR-warmup (stability-fix; Mamba-2 paradigm intact)

| Field | Value |
|---|---|
| Architecture | Mamba-2 selective state-space (UNCHANGED) |
| Stability mechanism | `torch.nn.utils.clip_grad_norm_(parameters, max_norm)` + linear LR-warmup over first N steps |
| Sweep config | grad_clip ∈ {0.5, 1.0, 5.0} × LR_warmup_steps ∈ {500, 2000, 5000} = 9 configs |
| Predicted ΔS band | [-0.025, -0.008] (inherits TOP-5 #2; UNCHANGED if stability fixed) |
| Implementation cost | ~30 min editor (CLI flags + 2 torch one-liners in `_full_main`) |
| GPU cost | $5 (9 × Modal T4 ~$0.50 each at 100ep smoke) |
| P(success) | 50-60% (per T3 council Tao + Dykstra + Hassabis: standard fix for selective-SSM instability; Hotz dissent 30%) |
| Reactivation | If ALL 9 configs NaN → pivot per Catalog #298; document in probe-outcomes |
| Composability | Z7-Mamba-2 + NSCS06v8 + DP1 + D1 (per design memo §3.6) UNCHANGED |
| Catalog #319 | DeliverabilityProof inheritance UNCHANGED (Z7MCM2 archive grammar) |

### (b) S4 (architectural sibling; linear-stable predecessor of Mamba-2)

| Field | Value |
|---|---|
| Architecture | S4 (HIPPO-ON parameterization; Gu et al. 2021 arxiv 2111.00396) |
| Stability mechanism | STRUCTURAL — HIPPO-ON is provably stable IF step-size respects canonical bounds |
| Implementation cost | ~1 day editor (new `tac.optimization.s4_predictor` + trainer config branch) |
| GPU cost | $5-10 (smoke + 1-2 hyperparameter configs) |
| P(success) | 80-85% (Schmidhuber: HIPPO-ON proof-of-stability; sister to Mamba lineage) |
| Predicted ΔS band | [-0.020, -0.005] (per Fridrich: ~80% of Mamba-2 capacity with much-improved training dynamics) |
| Reactivation | If S4 also unstable → pivot per Catalog #308 to (c)/(d)/(e); paradigm reformulated to PRE-Mamba-2-era SSM |
| Composability | UNCHANGED at SSM-family layer (S4 → Mamba → Mamba-2 architectural progression) |
| Catalog #319 | NEW DeliverabilityProof needed (S4 hidden-state differs from Mamba-2; archive grammar Z7SSM1 sister) |

### (c) RWKV-7 (architectural sibling; receptance-weighted key-value)

| Field | Value |
|---|---|
| Architecture | RWKV-7 "Goose" (Peng et al. 2025 arxiv 2503.14456) |
| Stability mechanism | STRUCTURAL — RWKV's receptance gating IS bounded by construction (sigmoid-clamped) |
| Implementation cost | ~1.5 days editor (new `tac.optimization.rwkv7_predictor` + trainer config branch) |
| GPU cost | $8-15 (smoke + 2-3 configs) |
| P(success) | 70-75% (per design memo §10 pivot path (a)) |
| Predicted ΔS band | [-0.020, -0.005] (sister-class to Mamba-2; different bias-variance) |
| Reactivation | If RWKV-7 also unstable → DEFER predictive-coding-recurrent paradigm per Catalog #298 |
| Composability | UNCHANGED at recurrent-state-bottleneck level |
| Catalog #319 | NEW DeliverabilityProof needed (RWKV state shape differs; archive grammar Z7RWKV1 sister) |

### (d) DreamerV3-RSSM (pivot; categorical latent + GRU dynamics; KL-balanced)

| Field | Value |
|---|---|
| Architecture | DreamerV3 Recurrent State-Space Model (Hafner 2023 arxiv 2301.04104; categorical z + GRU dynamics core + KL-balancing + free-bits) |
| Stability mechanism | STRUCTURAL — KL-balancing + free-bits prevent posterior collapse + NaN-explode BY DESIGN (T3 Hafner verbatim) |
| Implementation cost | ~2 weeks editor (new `tac.substrates.time_traveler_l5_z7_dreamerv3_rssm/` substrate scaffold + full Catalog #325 6-step contract per-substrate symposium) |
| GPU cost | $15-30 (full substrate validation; sister to Z6-v2 build effort) |
| P(success) | 85-90% (Hafner: dashcam-scale proven; PROCEED-unconditional per CC-9-PIVOT-1 in design memo §10) |
| Predicted ΔS band | [-0.030, -0.010] (per Hafner + design memo §10; expected EV CLOSER TO UPPER BOUND than Mamba-2) |
| Reactivation | If DreamerV3-RSSM also fails (extreme): substrate-class-shift to NeRV-family predictive-coding-without-recurrent-state |
| Composability | NEW — DreamerV3 RSSM composability unanchored; sister to Z6 Multi-layer FiLM at hidden-state level |
| Catalog #319 | NEW DeliverabilityProof needed (DreamerV3 categorical z + GRU state; archive grammar Z7DV3RSSM1 sister) |
| Per Catalog #325 | REQUIRES per-substrate symposium for `z7_dreamerv3_rssm` substrate BEFORE dispatch |
| Per Catalog #310 | PRIMARY substrate (architectural core, not bolt-on) |
| Per Catalog #311 | predictive ego-motion conditioning APPLICABLE |

### (e) FiLM-LSTM (pivot; well-understood reliable fallback)

| Field | Value |
|---|---|
| Architecture | FiLM-conditioned LSTM (FiLM Perez et al. 2017 + LSTM Hochreiter-Schmidhuber 1997) |
| Stability mechanism | STRUCTURAL — LSTM gating + forget gate empirically proven stable on dashcam-scale (canonical Quantizr 0.33 → PR95 0.193 lineage) |
| Implementation cost | ~3 days editor (sister to Z6-v2 Multi-layer FiLM patterns; minimal new code) |
| GPU cost | $5-15 (smoke + 1-2 configs) |
| P(success) | 90-95% (HIGHEST stability confidence; lowest expected ΔS) |
| Predicted ΔS band | [-0.012, -0.003] (LOWER expected EV; sister to Z6-v1 75K-FiLM territory) |
| Reactivation | If FiLM-LSTM fails: predictive-coding-recurrent paradigm DEFINITIVELY DEFER per Catalog #298 + #308 (3-of-3 alternatives empirically fail) |
| Composability | UNCHANGED at Z6 sister layer |
| Catalog #319 | INHERITS Z6 DeliverabilityProof if architecturally compatible |

## Recommended sequencing

Per Race-mode Rule 3 (cheap signal gates expensive) + Catalog #315 OPTIMAL-FORM iteration + Hassabis risk-adjusted-EV verbatim:

**Wave N+1 (immediate, $0 editor)**:
- (a) MPS proxy probe — verify Mamba-2 reference_torch backend FORWARD pass doesn't NaN at canonical 64-pair scale on M5 Max. This is the CHEAPEST disambiguator per Catalog #313 + design memo §13 LOCAL-MPS pattern. If forward NaNs on MPS, the issue is architectural not just CUDA-precision; if forward is clean, the issue is CUDA-Modal-specific.

**Wave N+2 (gated on N+1 + symposium ratification, $5)**:
- (a) grad-clip + LR-warmup sweep on Modal T4 (9 configs). If ANY converges → reactivate Mamba-2 per Catalog #315 OPTIMAL-FORM iteration.

**Wave N+3 (gated on N+2 outcome, $5-30)**:
- IF (a) converges → escalate to Z7-Mamba-2 full dispatch per design memo §9 sequencing
- IF (a) ALL diverge → pivot to (e) FiLM-LSTM (lowest cost / highest P(success) / lowest EV; produces empirical anchor for predictive-coding-recurrent paradigm)
- HOLD (b) S4, (c) RWKV-7, (d) DreamerV3-RSSM as alternative-probe-methodologies per Catalog #308 N>=3 IF (e) also fails

**Wave N+4 (gated on N+3 success, $20-50)**:
- IF (e) FiLM-LSTM produces empirical anchor → per-substrate symposium for whichever stability-fix-class won; full Modal A100 dispatch per Catalog #325 6-step contract

## Cross-substrate dependencies

- **Z6 Wave 2 Candidate 4c**: ego-source choice for Z7-Mamba-2 + sister candidates per design memo §11 Revision #4 inheritance
- **C6 IBPS Phase 2**: β-IB-Lagrangian parameter initialization per design memo §11 Revision #5 inheritance (applies to Mamba-2, DreamerV3-RSSM, ANY IB-Lagrangian-using candidate)
- **Z7-GRU Wave 2**: paired identity-disambiguator for ALL Z7 candidates per CC-2 unwind

## Per CLAUDE.md "Forbidden premature KILL"

NO KILL VERDICT. Z7-Mamba-2 paradigm INTACT (selective-SSM-on-dashcam architecturally sound); ONLY default-hyperparameter implementation is questionable per T3 finding 4 Assumption-Adversary classification. 5 reactivation paths enumerated; each per Catalog #298 + #308 N>=3 + #313 staleness-window discipline.

## 6-hook wire-in declaration per Catalog #125

1. Sensitivity-map: N/A (design-space memo, no signal emission)
2. Pareto constraint: ACTIVE (5 candidates each emit predicted ΔS band that feeds autopilot ranker per `tac.optimization.substrate_composition_matrix`)
3. Bit-allocator: N/A (no per-tensor importance change)
4. Cathedral autopilot dispatch: ACTIVE (Wave N+1/N+2/N+3 sequencing IS the dispatch routing)
5. Continual-learning posterior: ACTIVE — `tac.council_continual_learning.append_council_anchor` for THIS T1 working-group memo
6. Probe-disambiguator: ACTIVE — MPS-proxy + paired stability sweep IS the disambiguator per Catalog #313

## Sister coordination per Catalog #230

- This subagent OWNS: NEW `.omx/research/z7_mamba_2_*` memos (this file + path forward + symposium DRAFT) + memory entry
- 4 sister subagents in flight (DISJOINT per `.omx/state/subagent_progress.jsonl` audit):
  - Slot 1 MPS infrastructure (NEW `src/tac/mps_gap_experiment/`)
  - Slot 2 E.7+E.8 PREP (EXISTING VQ-VAE trainer + SGLD recipe edits + 2 NEW variant recipes + 2 symposium DRAFTs)
  - Slot 3 Phantom-API backfill Wave 1 (EXISTING `.omx/research/*.md` edits ~25 memos)
  - Slot 5 META-bug retroactive audit (NEW audit memo + READ-ONLY across state)
- NO file overlap; checkpoint discipline per Catalog #206

## Catalog #229 PV (5 premises verified pre-edit)

- PV-0: canonical helpers (`tac.scorer.load_default_scorers`, `tac.deploy.modal.call_id_ledger.register_dispatched_call_id`, `tac.probe_outcomes_ledger.register_probe_outcome`, `tac.council_continual_learning.append_council_anchor`, `tac.optimization.mamba2_predictor.Mamba2Predictor`) ALL importable
- PV-1: predecessor memo `.omx/research/3_smoke_paid_gpu_reactivation_wave_blocker_audit_20260519T042301Z.md` exists + cited 8 dispatch_blockers per Z7-Mamba-2 recipe
- PV-2: Z7-Mamba-2 trainer `_full_main` IS implemented at `experiments/train_substrate_time_traveler_l5_z7_mamba2.py:856` (predecessor audit + design memo §17 list `NotImplementedError` but code IS the full implementation — design memo is stale per `_full_main` docstring at line 856-895; this is a META cargo-cult of stale-trainer-state claims; documented for sister Slot 5 META-bug audit)
- PV-3: probe outcome `z7_mamba2_canonical_scale_stability_20260518` blocking-status confirmed via `tac.probe_outcomes_ledger.latest_blocking_outcome_by_substrate('z7_mamba2')` → verdict=DEFER + next_action="dispatch grad-clip + LR-warmup smoke; if fails pivot to RSSM/S4"
- PV-4: sister subagents 4-active per `.omx/state/subagent_progress.jsonl`; DISJOINT scope per Catalog #302

## Atom emission per Catalog #245/#323

Atom: `build_council_deliberation_atom(atom_id="z7_mamba_2_stability_design_space_20260518", deliberation_id="z7_mamba_2_stability_design_space", council_tier="T1", council_verdict="PROCEED_WITH_REVISIONS", predicted_impact_lower=-0.030, predicted_impact_upper=-0.003, cost_envelope_usd=0.00, memory_path=".omx/research/z7_mamba_2_stability_design_space_20260518.md")` — design-space enumeration only; NO score claim per CLAUDE.md "Apples-to-apples evidence discipline"

## Cross-references

- Parent T3 council: `.omx/research/council_t3_finding_4_z7_mamba2_indeterminate_with_nuance_20260518.md`
- Parent design memo: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- Predecessor audit: `.omx/research/3_smoke_paid_gpu_reactivation_wave_blocker_audit_20260519T042301Z.md`
- Z7-GRU sister: `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md`
- Sister recipe (research_only): `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml`
- Multi-week path forward: `.omx/research/z7_mamba_2_multi_week_path_forward_20260518.md`
- Symposium DRAFT: `.omx/research/council_t3_z7_mamba_2_stability_path_forward_symposium_DRAFT_20260518.md`
- Research wave TOP-5 #2: `.omx/research/comprehensive_research_wave_20260518.md`
