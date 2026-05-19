---
schema: council_deliberation_v2
deliberation_id: council_t3_z7_lstm_predictive_coding_per_substrate_symposium_DRAFT_20260519
topic: "Z7-LSTM/GRU FALLBACK per-substrate symposium DRAFT — recurrent-state-as-implicit-side-info predictive coding; supersedes T2 2026-05-17 memo with T3 DRAFT format for operator-routable ratification"
review_kind: per_substrate_optimal_form_symposium_T3_grand_council_DRAFT
review_date: "2026-05-19"
lane_id: lane_cable_c_substrate_symposium_draft_batch_20260519
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Rao, Ballard, Tishby_memorial, Zaslavsky, Hafner, Schmidhuber, Quantizr, Hotz, Wyner]
council_quorum_met: false
council_verdict: DRAFT_PENDING_OPERATOR_CONVOCATION
council_dissent:
  - member: Contrarian
    verbatim: "DRAFT only — full T3 convocation requires operator-attention budget per Catalog #300. Per-substrate symposium per Catalog #325 14-day window starts at convocation, not at DRAFT landing. This DRAFT supersedes the 2026-05-17 T2 PROCEED_WITH_REVISIONS memo (now 2 days stale per Catalog #298 retirement-discipline approach) by re-elevating to T3 with fresh sextet positions + updated frontier anchors (PR101 fec6 0.19205 CPU + PR106 format0d 0.20533 CUDA per Catalog #316). Operator-routable ratification mechanism: (a) full T3 convocation $0 editor; (b) inner-quintet pact ratification; (c) operator-frontier-override per Catalog #300 Consequence 1."
  - member: Hotz
    verbatim: "Z7-LSTM-as-Candidate-2 sequencing IS the correct cargo-cult-unwind: cheap-signal-first (Z6 Wave 2 4c $3) gates expensive-signal (Z7 LSTM $5-15 smoke + $16.50-21.50 full). DRAFT must NOT pre-authorize Z7 standalone OR stacked dispatch from THIS verdict per the 2026-05-17 Assumption-Adversary binding."
council_assumption_adversary_verdict:
  - assumption: "Z7-LSTM/GRU FALLBACK substrate independence vs Z6 Multi-layer FiLM"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Z7 LSTM is REGISTERED as Candidate 2 in the Z6 Path B menu per `z6_v2_cargo_cult_unwind_4_candidate_redesign_path_b_20260517.md` Section 4. Per Catalog #308 N>=3 alternative-probe-methodologies: Z7 is an ALTERNATIVE PROBE within the predictive-coding-recurrent paradigm, NOT a separately-fundable paradigm. Operator-routable: Z7 dispatch GATED on Z6 Wave 2 Candidate 4c paired exact-eval outcome (sister codex probe `z6_candidate4c_full_disambiguator_probe_20260518_codex.md` PENDING)."
  - assumption: "GRU (not LSTM) is the canonical recurrent primitive per Hafner DreamerV3 lineage"
    classification: HARD-EARNED
    rationale: "Per Hafner 2026-05-17 verbatim: DreamerV3 uses GRU + Gaussian sampling per timestep; the deterministic component is GRU. Z7 inherits the DreamerV3 deterministic-GRU lineage canonically. GRU has 25% fewer parameters than LSTM at same hidden_dim — better fits substrate-engineering budget. Codename LSTM/GRU FALLBACK retained for backward compatibility; implementation binds to GRU."
  - assumption: "Z7 hidden state's I(T;Y) against contest-CUDA scorer exceeds MEANINGFUL_CONDITIONING threshold of 0.5 bits/symbol"
    classification: CARGO-CULTED-PENDING-PROBE
    rationale: "Per Tishby_memorial 2026-05-17 verbatim: raw H(T) = 128 × 32 × 600 ≈ 2.5M bits trivially clears threshold, BUT I(T;Y) requires scorer-relevant subspace. ATW V2 D4 INDEPENDENT verdict (MI=0.006385) is canonical empirical failure mode of high-H(T)/low-I(T;Y) channel. Sister disambiguator probe required AT smoke-time per Z6 Wave 2 Candidate 4c sister codex pattern."
  - assumption: "Predicted band [0.180, 0.192] [contest-CPU] derived from canonical-vs-Z6 capacity analysis"
    classification: HARD-EARNED-PARTIAL
    rationale: "Lower bound [0.180] derived from Z6-v1 PoseNet-projection baseline floor; upper bound [0.192] derived from PR101 fec6 frontier 0.19205. Predicted band partially overlaps current canonical frontier; promotion-eligible only if Wave 3 full lands STRICTLY BELOW 0.19205. Per Catalog #324: predicted_band_validation_status MUST be pending_post_training; per Catalog #316: any Wave 3 full result MUST be compared against canonical frontier anchor in same commit batch."
council_decisions_recorded:
  - "DRAFT enumerates 6-step Catalog #325 contract for Z7-LSTM/GRU FALLBACK"
  - "Operator-routable: convocation mechanism choice (T3 full / inner-quintet pact / operator-frontier-override)"
  - "Wave 2 smoke envelope: $5-7 Modal T4 100ep (60-90 min wall-clock)"
  - "Wave 3 full envelope: $16.50-21.50 Modal A100 1000ep (6-10 hour wall-clock)"
  - "Z7 dispatch GATED on Z6 Wave 2 Candidate 4c paired exact-eval per Catalog #315 OPTIMAL-FORM + Race-mode-rigor-inversion Rule 3"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: time_traveler_l5_z7_lstm_predictive_coding
substrate_aliases:
  - z7_lstm_predictive_coding
  - z7_gru_predictive_coding
  - time_traveler_l5_z7
deferred_substrate_retrospective_due_utc: "2026-06-18T05:33:56Z"
horizon_class: asymptotic_pursuit
predicted_band: [0.180, 0.192]
predicted_band_validation_status: pending_post_training
score_claim: false
promotion_eligible: false
dispatch_enabled: false
research_only: true
canonical_frontier_anchor:
  contest_cpu: "0.1920513169 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
  contest_cuda: "0.2053300290 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
related_deliberation_ids:
  - council_per_substrate_symposium_z7_lstm_predictive_coding_20260517
  - council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518
  - z7_lstm_full_main_design_20260518
  - council_t3_z7_mamba_2_stability_path_forward_symposium_DRAFT_20260518
  - time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516
---

# DRAFT: T3 grand council symposium — Z7-LSTM/GRU FALLBACK per-substrate symposium

**Status**: DRAFT — operator-convocation pending. NOT a binding council verdict.
**Lane**: `lane_cable_c_substrate_symposium_draft_batch_20260519` L1
**Per Catalog #325**: this DRAFT satisfies the 6-step contract structurally; full convocation activates symposium evidence per Catalog #325 14-day window.
**Supersession**: this DRAFT supersedes T2 2026-05-17 `council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md` (now 2 days stale per retirement-discipline) by re-elevating to T3 DRAFT format with fresh sextet positions + updated frontier anchors per Catalog #316.

## Symposium attendees (proposed)

**Sextet pact** (REQUIRED per CLAUDE.md "Council conduct" amendment):
- **Shannon LEAD** — information-theoretic capacity of recurrent-state-as-implicit-side-info
- **Dykstra CO-LEAD** — convex-feasibility of GRU training procedure + cooperative-receiver channel preservation
- **Yousfi** — PoseNet/SegNet response to recurrent-conditioning substrate
- **Fridrich** — inverse-steganalysis of recurrent-hidden-state-as-side-info channel
- **Contrarian** — VETO power on lazy consensus
- **Assumption-Adversary** — per-round shared-assumption-violation hypothesis (Catalog #291 + #292)

**Grand council added per topic**:
- **Rao + Ballard** — canonical Rao-Ballard 1999 predictive-coding authority (Z7 = single-level recurrent variant; Z8 = full hierarchy)
- **Tishby_memorial + Zaslavsky** — IB-Lagrangian framework + dashcam-scale IB-floor calibration cross-pollination with C6 IBPS
- **Hafner** — DreamerV3 deterministic-GRU canonical primitive
- **Schmidhuber** — RNN/LSTM lineage + compression-as-intelligence
- **Quantizr** — adversarial reverse-engineering vs PR106 format0d (stateless decoder counter-evidence)
- **Hotz** — engineering shortcuts + don't-chase-fragile-baselines
- **Wyner** — Wyner-Ziv 1976 source-coding-with-side-info canonical framing (LSTM hidden state IS the encoder-decoder shared channel)

## Step 1 — Cargo-cult audit per Catalog #303

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| CC-z7-1 | "Recurrent-state predictive coding is architecturally distinct from Z6 Multi-layer FiLM" | HARD-EARNED-PARTIAL | Rao verbatim: LSTM/GRU hidden state expresses TEMPORAL hierarchy that Multi-layer FiLM cannot. Empirical disambiguator: Wave 2 paired comparison Z6 vs Z7 at SAME archive size per Z6 Phase 3 Revision #2 canonical pattern. |
| CC-z7-2 | "GRU > LSTM at same hidden_dim per Hafner DreamerV3 lineage" | HARD-EARNED | DreamerV3 deterministic component is GRU; 25% fewer params at same hidden_dim. Binding per Hafner 2026-05-17 verbatim. |
| CC-z7-3 | "Recurrent state's MI against contest-CUDA scorer is non-trivially recoverable" | CARGO-CULTED-PENDING-PROBE | ATW V2 D4 INDEPENDENT verdict (MI=0.006385) is canonical counter-example. Smoke-time MI probe via `tac.probe_outcomes_ledger.register_probe_outcome` REQUIRED before Wave 3 full. |
| CC-z7-4 | "Recurrent state is the winning bit-allocation strategy on dashcam temporal coherence" | CARGO-CULTED-PENDING-EMPIRICAL (META) | PR106 format0d (STATELESS decoder; canonical CUDA frontier 0.20533) is HARD-EARNED counter-evidence. If Z7 fails to beat PR106 stateless, predictive-coding-recurrent paradigm DEFER per Catalog #298 (NOT KILL). |
| CC-z7-5 | "Predicted band [0.180, 0.192] is calibrated" | HARD-EARNED-PARTIAL | Lower bound from Z6-v1 baseline; upper from PR101 fec6 frontier. Predicted band STRADDLES current frontier; promotion requires Wave 3 STRICTLY BELOW 0.19205. |
| CC-z7-6 | "Wave 2 smoke at 100ep on Modal T4 is sufficient for MI-channel-recoverability verdict" | HARD-EARNED-PARTIAL | Sister Z6 Wave 2 4c smoke pattern + sister Z7-Mamba-2 Wave N+2 9-config sweep precedent ($15 Modal T4). Z7-LSTM single-config smoke $5-7 sufficient for MI probe; full ΔS-prediction validation needs Wave 3 full. |
| CC-z7-7 | "Wave 3 full at 1000ep on Modal A100 is the canonical full-empirical-anchor envelope" | HARD-EARNED | Sister Z6 Phase 3 Wave 3 precedent + Z7-Mamba-2 Wave N+3 path A precedent. $16.50-21.50 envelope; 6-10 hour wall-clock. |

## Step 2 — 9-dimension success checklist evidence per Catalog #294

| # | Dimension | Per-symposium-DRAFT evidence |
|---|---|---|
| 1 | UNIQUENESS | ✓ Z7-LSTM/GRU is FIRST recurrent-state-as-implicit-side-info predictive coding substrate (sister Z7-Mamba-2 = selective-SSM; sister Z6 = stateless FiLM stack). Asymptotic_pursuit class per Catalog #309. |
| 2 | BEAUTY + ELEGANCE | ✓ ~80-100K GRU params; ~50KB archive overhead via fp4+brotli; hidden state regenerated at inflate-time (NO archive bytes per Wyner-Ziv). Trainer ~600 LOC per HNeRV parity L7 substrate-engineering budget. |
| 3 | DISTINCTNESS | ✓ Recurrent-state vs Multi-layer FiLM (Z6) vs selective-SSM (Z7-Mamba-2) vs full RSSM (Z8) — 4 architecturally orthogonal predictive-coding primitives. |
| 4 | RIGOR | ✓ THIS DRAFT + 2026-05-17 T2 memo + Z7-LSTM full_main design memo + sister Z7-Mamba-2 unified symposium = 4 memos with cargo-cult audit + observability surface + Dykstra-feasibility + horizon-class + Catalog #229 PV + Catalog #313 probe-ledger consultation + Catalog #302 sister coordination. |
| 5 | OPTIMIZATION PER TECHNIQUE | ✓ Wave 2 smoke MI-channel-recoverability probe IS the Z7-LSTM-optimal disambiguator. Wave 3 full ΔS-prediction validation IS the Z7-LSTM-optimal empirical anchor. GRU > LSTM per DreamerV3 lineage. |
| 6 | STACK-OF-STACKS-COMPOSABILITY | ✓ Z7-LSTM/GRU + NSCS06v8 chroma + DP1 pretraining + D1 SegNet overlay (4 orthogonal axes per Z6/Z7/Z8 design memo §3.6 inheritance). Sister Z7-Mamba-2 stability fork. |
| 7 | DETERMINISTIC REPRODUCIBILITY | ✓ Z7PCWM1 archive byte-stable per Catalog #5; GRU deterministic unroll (no stochastic component); regenerated hidden state at inflate-time per Wyner-Ziv canonical pattern. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | ✓ Wave 2 smoke parallelizable on Modal T4 (single-config); ~60-90 min wall-clock. Wave 3 full Modal A100 ~6-10 hour. Empirical budget $22-29 total (smoke + full). |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | INDETERMINATE — per Catalog #316: predicted band [0.180, 0.192] [contest-CPU] STRADDLES canonical frontier 0.19205. Promotion-eligible ONLY if Wave 3 STRICTLY BELOW 0.19205; otherwise DEFER per Catalog #298. |

## Step 3 — Observability surface declaration per Catalog #305

**Per-Z7-LSTM-substrate observability**:
1. **Inspectable per layer**: per-epoch loss decomposition (segnet + posenet + rate + IB-Lagrangian if active) + per-pair GRU hidden state norm + per-pair ego_motion vector + per-pair MI(hidden_state; scorer_logit) probe value
2. **Decomposable per signal**: per-epoch segnet vs posenet contribution + Wave 2 smoke MI-probe per-pair distribution + Wave 3 full convergence curve vs PR101 baseline
3. **Diff-able across runs**: paired Z6-vs-Z7 at SAME archive bytes per Z6 Phase 3 Revision #2 + sister Z7-Mamba-2 stability-comparison
4. **Queryable post-hoc**: per-config Modal call_id ledger row per Catalog #245 + per-config probe-outcome ledger row per Catalog #313 + per-config build_manifest.json per Catalog #220
5. **Cite-able**: cite parent 2026-05-17 T2 symposium + sister Z7-Mamba-2 unified + Z6/Z7/Z8 scoping memo + this DRAFT
6. **Counterfactual-able**: "what if GRU hidden_dim=64 vs 128 vs 256?" + "what if ego_motion source = PoseNet projection vs scorer_logit per Z6 Wave 2 4c outcome?" + "what if predictive-coding-recurrent paradigm fundamentally fails?" (→ DEFER per Catalog #298)

## Step 4 — Sextet pact deliberation (DRAFT positions)

### Shannon LEAD position (DRAFT)

*"Operating-within assumption: recurrent-state predictive coding's capacity floor is the IB-Lagrangian L_IB = I(X;T) - β·I(T;Y). The information-theoretic question is whether the GRU hidden state's I(T;Y) against contest-CUDA scorer exceeds the MEANINGFUL_CONDITIONING threshold (0.5 bits/symbol). Wave 2 smoke MI probe IS the capacity-realization disambiguator. PROCEED on DRAFT design; Wave 2 smoke MI probe gates Wave 3 full."*

### Dykstra CO-LEAD position (DRAFT)

*"Operating-within assumption: training procedure = composition of optimizer steps + GRU deterministic unroll. Convex-feasibility requires each step to land in stable region + GRU hidden state to remain bounded. Sister Atick-Redlich cooperative-receiver framing: GRU hidden state IS the encoder-decoder shared-prior channel (Wyner-Ziv canonical). PROCEED on DRAFT design with explicit Wyner-Ziv side-info channel preservation requirement."*

### Yousfi position (DRAFT)

*"Operating-within assumption: PoseNet/SegNet scorer's response to recurrent-conditioning is EMPIRICALLY UNTESTED. Sister ATW V2 D4 INDEPENDENT verdict (MI=0.006385 on per-pair argmax composite) shows recurrent-or-recursive conditioning CAN fail to enter scorer-relevant subspace. PROCEED on DRAFT design with mandatory smoke-time MI probe."*

### Fridrich position (DRAFT)

*"Operating-within assumption: inverse-steganalysis interpretation of recurrent-hidden-state-as-implicit-side-info. The GRU hidden state IS a side-channel between encoder + decoder; the channel's bandwidth is hidden_dim × bit_precision per pair × 600 pairs. The information-theoretic capacity is upper bound; the EMPIRICAL bit-savings depends on the specific receiver. PROCEED on DRAFT design."*

### Contrarian position (DRAFT)

*"Operating-within assumption: predicted band [0.180, 0.192] STRADDLES canonical frontier 0.19205. The OPTIMISTIC case Wave 3 lands at 0.180 = breakthrough; the PESSIMISTIC case Wave 3 lands at 0.192 = NO score improvement. Wave 2 smoke MI probe IS the cheapest disambiguator. VETO any DRAFT path that pre-authorizes Wave 3 dispatch BEFORE Wave 2 smoke MI probe lands. STRONG RECOMMENDATION: explicit Catalog #298 reactivation criteria enumerated below."*

### Assumption-Adversary position (DRAFT) [Catalog #291 + #292]

*"Operating-within assumption (META): the entire predictive-coding-recurrent paradigm assumes recurrent-state is the winning bit-allocation strategy on dashcam temporal coherence. PR106 format0d (STATELESS decoder; canonical CUDA frontier 0.20533) is HARD-EARNED counter-evidence. IF Z7 + Z7-Mamba-2 + Z8 ALL fail to beat PR106 stateless, the entire predictive-coding-recurrent paradigm DEFER per Catalog #298 → reactivation = NeRV-family predictive-coding-without-recurrent-state OR foveation IDEAS without recurrent state. The Catalog #313 probe-outcomes ledger MUST record THIS paradigm-level claim AS the disambiguator question."* — VETO if not engaged.

### Rao position (DRAFT)

*"Operating-within assumption: Z7 = SINGLE-LEVEL recurrent variant per my 1999 paper; Z8 = full hierarchy. Z7 captures temporal-coherence-via-recurrent-state insight from Hafner DreamerV3 lineage more directly than from my paper. PROCEED on Z7 DRAFT IF the design memo explicitly cites Z7 as TEMPORAL-COHERENCE-PRIMITIVE ORTHOGONAL TO Z6 SPATIAL-CAPACITY-PRIMITIVE. Z6-vs-Z7 paired comparison at SAME archive bytes IS the canonical disambiguator."*

### Ballard position (DRAFT)

*"Concurs with Rao. The GRU hidden state IS the substrate-level operationalization of 'embodied temporal prior' per Gibson 1950 + Rao-Ballard 1999. The encoder + decoder share the implicit side-info channel of 'past GRU hidden states' WITHOUT shipping them in archive — canonical Wyner-Ziv 1976 source-coding pattern. PROCEED on DRAFT design with explicit ego-motion-FoE-prior incorporation per Catalog #311."*

### Tishby_memorial position (DRAFT)

*"Operating-within assumption: IB Lagrangian L_IB = I(X;T) - β·I(T;Y). Raw H(T) = 128 × 32 × 600 ≈ 2.5M bits trivially clears MEANINGFUL_CONDITIONING threshold. But I(T;Y) requires scorer-relevant subspace. Z7 trainer MUST be flexible enough to accept either PoseNet-projection ego (Z6-v1 baseline) OR scorer-logit conditioning (Z6 4c outcome) as input. β-IB-Lagrangian parameter should be initialized from C6 IBPS Phase 2 empirical β-optimal anchor."*

### Hafner position (DRAFT)

*"DreamerV3 deterministic component is GRU. Z7 inherits canonical DreamerV3 deterministic-GRU lineage. BINDING: Z7 uses GRU (NOT LSTM) per Hafner 2026-05-17 verbatim; codename LSTM/GRU FALLBACK retained for backward compatibility."*

### Hotz position (DRAFT)

*"Operating-within assumption: don't chase fragile baselines. Z7-Mamba-2 NaN-explode at canonical scale was STRUCTURAL signal. Z7-LSTM/GRU is the canonical pivot per CLAUDE.md 'Forbidden premature KILL'. PROCEED on DRAFT design. STRONG RECOMMENDATION: Wave 2 smoke + Wave 3 full sequencing is correct; do NOT parallelize."*

### Quantizr position (DRAFT)

*"PR106 format0d (STATELESS decoder; canonical CUDA frontier 0.20533) IS the HARD-EARNED counter-evidence for recurrent-state-being-the-winning-pattern. PR95/100/101/102/103 ALL frame-independent. Z7 is empirically-novel architecture for contest — cuts BOTH ways. Wave 2 + Wave 3 paired-comparison at SAME archive bytes per Z6 Phase 3 Revision #2 canonical pattern IS the empirical answer."*

## Step 5 — Per-substrate reactivation criteria pinned per Catalog #298 + #308

Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

| Stage | If verdict | Reactivation path |
|---|---|---|
| Wave 2 smoke MI probe | MI < 0.5 bits/symbol | DEFER Z7-LSTM per Catalog #298; reactivation = Z6 Wave 2 4c full-FiLM-WIN at ΔS ≥ 0.005 (LSTM CONDITIONED ON scorer-logit re-enters scope) |
| Wave 2 smoke | smoke crashes / NaN | DEFER per Catalog #298; reactivation = sister Z7-Mamba-2 stability-fix landing OR Z6 Wave 2 4c outcome (whichever first) |
| Wave 3 full | score WORSE than PR101 frontier 0.19205 | Wave 4 composition (Z7-LSTM + NSCS06v8 + DP1 + D1); IF still WORSE → DEFER substrate per Catalog #298 → predictive-coding-recurrent paradigm enters reactivation queue |
| All 5 z-substrates (Z6+Z7-LSTM+Z7-Mamba-2+Z8+DP1) | All fail to beat PR101 | Predictive-coding-recurrent paradigm DEFER per Catalog #298 → reactivation = NeRV-family predictive-coding-without-recurrent-state OR foveation IDEAS without recurrent state |

## Step 6 — Catalog #324 post-training Tier-C validation discipline

Recipe declares `predicted_band_validation_status: pending_post_training`. Reactivation criterion: post-training Tier-C density measurement on Z7-LSTM/GRU archive after Wave 3 full dispatch completes via `tools/mdl_scorer_conditional_ablation.py --tier c`. Per Catalog #324: predicted_band [0.180, 0.192] is research prior; promotion-eligible only after `validated_post_training` status. Wave 2 smoke score does NOT satisfy post-training Tier-C validation (smoke != full).

## Operator-routable ratification mechanisms

1. **Full T3 convocation** ($0 editor; ~3h council deliberation): convene 15-seat grand council per Catalog #300 v2 frontmatter; ratify DRAFT verdicts as binding T3.
2. **Inner-quintet pact ratification** ($0 editor; ~1h): sextet pact (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary) ratifies; T2 verdict instead of T3.
3. **Operator-frontier-override** per Catalog #300 Consequence 1: operator-verbatim quote authorizes paid dispatch IF time-critical; preserves maximum-signal preservation (dissent + assumption classification + continual-learning anchor still recorded).

## Cross-substrate dependencies

- **Sister gate Z6 Wave 2 Candidate 4c paired exact-eval** (sister codex probe `z6_candidate4c_full_disambiguator_probe_20260518_codex.md` PENDING): Z7 dispatch GATED per Catalog #315 OPTIMAL-FORM + Race-mode-rigor-inversion Rule 3
- **Sister gate C6 IBPS Phase 2 empirical β-optimal**: Z7 β-IB-Lagrangian parameter initialization depends on C6 outcome per Zaslavsky 2026-05-17 verbatim
- **Sister substrate Z7-Mamba-2** (sister DRAFT 2026-05-18): stability-fix Wave N+2 9-config sweep $15; Z7-LSTM is canonical pivot if Mamba-2 fails

## Predicted cost per substrate paid dispatch

- Wave 2 smoke MI probe: $5-7 (Modal T4 100ep)
- Wave 3 full empirical anchor: $16.50-21.50 (Modal A100 1000ep)
- TOTAL Z7-LSTM/GRU FALLBACK envelope: $22-29

## Continual-learning posterior anchor

Per Catalog #300 + `tac.council_continual_learning.append_council_anchor`: this DRAFT must emit a v2 posterior anchor at convocation. Pre-flight `deferred_substrate_id` = `time_traveler_l5_z7_lstm_predictive_coding`; `predicted_mission_contribution` = `frontier_breaking`; retrospective due 2026-06-18T05:33:56Z.
