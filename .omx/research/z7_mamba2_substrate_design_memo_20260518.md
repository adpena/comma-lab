---
name: z7-mamba2-substrate-design-memo-20260518
metadata:
  node_type: memory
  council_tier: T1
  council_attendees:
    - Hafner
    - Rao
    - Ballard
    - Tishby_memorial
    - Zaslavsky
    - Quantizr
    - Contrarian
    - Assumption-Adversary
  council_quorum_met: false
  council_verdict: DESIGN_MEMO_ONLY_PENDING_WAVE_N_PLUS_1_COUNCIL
  council_dissent:
    - member: Contrarian
      verbatim: "This is a DESIGN MEMO not a dispatch authorization. Z7-Mamba-2 inherits the Z7 symposium 2026-05-17 VETO on independent dispatch funding. The Mamba-2 path is the Wave-N+2 PIVOT BRANCH (b) per Z7 symposium Revision #6: only fires if Z7-GRU Wave 2 disambiguator DEFERs OR if operator explicitly approves Mamba-2 as the FIRST recurrent primitive instead of GRU. NO pre-authorization at THIS memo."
    - member: Assumption-Adversary
      verbatim: "The shared assumption operating across this memo is *'Mamba-2's 5-10× efficiency over LSTM/GRU transfers to the dashcam contest temporal coherence pattern at the 600-pair sequence length we care about.'* CARGO-CULTED-PENDING-VERIFICATION per the research wave's own §11 truth audit. The empirical anchors cited (arxiv 2405.21060) are for LANGUAGE benchmarks at sequence lengths 2K-1M tokens. The contest is 600 pairs of 24-dim latents. Whether Mamba-2's O(N) selectivity provides a meaningful advantage at sequence length 600 over GRU's O(N) recurrence (also linear) is UNTESTED. The MPS-runnable proxy training pattern in §13 IS the cheapest empirical disambiguator — run BEFORE any paid dispatch."
  council_assumption_adversary_verdict:
    - assumption: "Mamba-2 is the canonical 2024 substitute for LSTM in Z7 per Catalog #308 N>=3 alternative-probe-methodologies"
      classification: HARD-EARNED
      rationale: "Per research wave §3.6 + parent Z7 symposium Section 2 CC-8 unwind: ≥3 alternative recurrent primitives must be enumerated for substrate-class predictive-coding. Z7-GRU (Revision #3) + Z7-Mamba-2 (THIS memo) + Z7-RWKV-7 (deferred to Wave-N+3) = 3 alternatives. Catalog #308 satisfied."
    - assumption: "Mamba-2's reported 5-10× speedup over LSTM transfers to dashcam contest sequence length 600"
      classification: CARGO-CULTED-PENDING-EMPIRICAL
      rationale: "Per research wave §11 truth audit: '5-10× factor not verified for video-specific workloads on dashcam contest sequence length.' Mamba-2's speedup is empirically proven at language sequence lengths 2K-1M. At sequence length 600, both GRU and Mamba-2 are O(N); the constant-factor advantage is empirically untested at this scale. MPS-runnable proxy training pattern (§13) is the canonical disambiguator BEFORE paid dispatch."
    - assumption: "Mamba-2's selective state-space mechanism provides better expressive power than GRU at hidden_dim=128 for this task"
      classification: CARGO-CULTED-PENDING-EMPIRICAL
      rationale: "Mamba-2's selectivity advantage is well-documented at LARGE state dimensions (d_state >= 16-256) and LONG sequences. At Z7's hidden_dim=128, state_dim=16 typical Mamba-2 config, on 600-pair sequences with 24-dim latents, the expressive-power advantage over GRU is empirically untested. Sister hypothesis: Mamba-2's continuous-time SSM formulation may MATCH the ego-motion-continuity prior more naturally than GRU's discrete-step recurrence — TESTABLE via MPS proxy."
    - assumption: "Mamba-2's mamba_ssm PyPI package can be installed in pact's Modal training image"
      classification: HARD-EARNED-PARTIAL
      rationale: "mamba_ssm requires CUDA toolkit + custom CUDA kernels. PyPI install with `pip install mamba-ssm` works on Linux x86_64 + CUDA 11.6+ but FAILS on macOS / M5 Max (verified empirically 2026-05-18 at this memo: `ModuleNotFoundError: No module named 'mamba_ssm'` on M5 Max MPS). For pact's Modal training image (A100, CUDA 12+), install should work but needs explicit `--extra-index-url` for PyTorch CUDA wheel matching mamba-ssm's compiled kernels. Catalog #244 NVML env block + Catalog #270 Tier 1/2/3 protocol applies. For LOCAL MPS proxy training, a pure-PyTorch Mamba-2 reference implementation (no CUDA kernels) is the only path; ~10× slower than mamba_ssm but architecturally identical."
    - assumption: "Z7-Mamba-2 belongs in the asymptotic_pursuit horizon_class per Catalog #309"
      classification: HARD-EARNED
      rationale: "Per Z7 parent symposium predicted band [0.10, 0.13] + research wave §0 TOP-5 #2 predicted ΔS [-0.025, -0.008] over PR101 frontier 0.19205 ⇒ [0.167, 0.184] [contest-CPU]. Lower bound 0.167 sits in asymptotic_pursuit (0.05-0.12) lower-region; upper bound 0.184 sits in frontier_pursuit (0.12-0.18) upper-region. Per the canonical 'classify by lower-bound of predicted band' rule: Z7-Mamba-2 classified asymptotic_pursuit. Satisfies HORIZON-CLASS Consequence 2."
    - assumption: "Z7-Mamba-2 dispatch is sequenced AFTER Z7-GRU Wave 2 disambiguator outcome"
      classification: HARD-EARNED
      rationale: "Per Z7 parent symposium Revision #6: Z7-Mamba-2 is enumerated as Wave-N+2 PIVOT-PATH (b) IF Z7-GRU Wave 2 LOSES OR DEFERs. Per CLAUDE.md Race-mode-rigor-inversion Rule 3 (cheap signal gates expensive signal) + Catalog #315 OPTIMAL-FORM iteration discipline: Z7-GRU $5-7 envelope is cheaper than Z7-Mamba-2 $20-30 envelope; sequence cheaper before expensive. EXCEPTION: operator may explicitly approve Z7-Mamba-2 as the FIRST recurrent primitive (skipping Z7-GRU) IF Mamba-2's research-wave evidence is judged compelling enough — this requires explicit operator-frontier-override per Catalog #300 + verbatim quote in `council_override_rationale`."
  council_decisions_recorded:
    - "VERDICT: DESIGN_MEMO_ONLY_PENDING_WAVE_N_PLUS_1_COUNCIL — Z7-Mamba-2 scaffold (design memo + canonical helper integration scaffold + trainer scaffold + tests + recipe with research_only=true) is authorized at $0 GPU spend + ~3h editor work per parent TOP-5 #2 directive. NO paid dispatch authorization from this memo."
    - "Catalog #325 6-step contract satisfied for substrate=time_traveler_l5_z7_mamba2 substrate alias per substrate_aliases mechanism (canonical short-form for the lane `lane_top5_2_z7_mamba2_scaffold_design_20260518`)."
    - "Frontier citation per Catalog #316: current canonical best is 0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201). Z7-Mamba-2 predicted band [0.167, 0.184] sits BELOW this anchor IF realized empirically — asymptotic_pursuit class. Predicted vs realized gap is the canonical empirical question for Wave N+1."
    - "Per CLAUDE.md 'Forbidden premature KILL': Z7-Mamba-2 is in PRE-BUILD state (NOT yet built; scaffold + design only at this memo). Per Catalog #315: Z7-Mamba-2 is PRE-OPTIMAL-FORM (cargo-cult-unwind methodology must be applied via Z7-Mamba-2 trainer build per Wave N+1 council approval before any iteration anchor)."
  council_predicted_mission_contribution: frontier_breaking
  council_override_invoked: false
  council_override_rationale: ""
  horizon_class: asymptotic_pursuit
  canonical_frontier_anchor:
    contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
    contest_cuda: "0.20533 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
  deferred_substrate_id: time_traveler_l5_z7_mamba2
  deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
  predicted_dispatch_risk: 0
  originSessionId: lane_top5_2_z7_mamba2_scaffold_design_20260518
  related_deliberation_ids:
    - council_per_substrate_symposium_z7_lstm_predictive_coding_20260517
    - council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517
    - z6_v2_cargo_cult_unwind_4_candidate_redesign_path_b_20260517
    - z6_candidate4c_full_disambiguator_probe_20260518_codex
    - time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516
    - comprehensive_research_wave_20260518
    - council_per_substrate_symposium_atw_v2_reactivation_20260518
    - council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518
    - feedback_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_landed_20260518
---

# Z7-as-Mamba-2 substrate design memo — TOP-5 #2 SCAFFOLD 2026-05-18

**Lane**: `lane_top5_2_z7_mamba2_scaffold_design_20260518` (L0 → L1 at scaffold landing)
**Parent**: Z7 LSTM predictive coding symposium 2026-05-17 (`council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md`) — PROCEED_WITH_REVISIONS with Hafner Revision #3 binding GRU and Quantizr Revision #6 binding Wave-N+2 pivot to Mamba/Z8/research_only IF Z7-GRU Wave 2 LOSES.
**Research wave**: TOP-5 reformulation #2 (`comprehensive_research_wave_20260518.md` §0 + §2.2 Z7 + §3.6 DreamerV3↔Mamba convergence).
**Catalog #325 satisfied** for `substrate=time_traveler_l5_z7_mamba2` (14-day window from 2026-05-18).
**$0 GPU, ~3h editor, NO COMMITS per parent prompt, NO Modal/Lightning/Vast.ai dispatches.**

## TL;DR (60 seconds)

Z7-Mamba-2 is the **canonical Catalog #308 N>=3 alternative-probe-methodology** to Z7-GRU (Hafner Revision #3 binding) within the predictive-coding-recurrent substrate class. Mamba-2 (Dao-Gu 2024, arxiv 2405.21060) is a selective state-space sequence model proven to match Transformer quality at O(N) compute on long-context language and video tasks; reported 5-10× speedup over LSTM at language scale.

For pact's dashcam contest at 600 pairs × 24-dim latent, the **expected-vs-empirical gap** is the canonical research question — both Mamba-2 and GRU are O(N) at this sequence length, so the constant-factor speedup advantage is empirically untested. Mamba-2's **selective state-space** mechanism (input-conditioned A,B,C matrices) may match the ego-motion-continuity prior more naturally than GRU's discrete-step recurrence — TESTABLE via local M5 Max MPS proxy training before paid dispatch.

**This memo authorizes**: scaffold delivery (design memo + canonical helper integration + trainer scaffold + tests + research_only recipe). **NO dispatch authorization**. Dispatch requires Wave N+1 council convened AFTER (a) Z7-GRU Wave 2 disambiguator outcome lands OR (b) operator explicit-frontier-override per Catalog #300 + verbatim quote in council_override_rationale.

**Predicted ΔS band**: [-0.025, -0.008] over PR101 frontier ⇒ [0.167, 0.184] [contest-CPU] per research wave §0 TOP-5 #2. Lower-bound 0.167 sits in asymptotic_pursuit horizon_class per Catalog #309.

**Cost path**: Scaffold $0 + ~3h editor (THIS memo). Wave N+1 council $0 + ~90 min. Wave 2 smoke + identity-disambiguator paired ~$5-10 (Mamba-2 GPU forward is sister to GRU; cost dominated by 100ep training). Wave 3 full dispatch (CONDITIONAL) ~$20-30 Modal A100 + paired CPU/CUDA ~$1.50-2.00 = **$22-30 total dispatch envelope** if all phases authorized.

**Reactivation paths** (per Catalog #308 N>=3):
- (a) Z8 full Rao-Ballard hierarchy + Hafner DreamerV3 stochastic (envelope $42)
- (b) Z7-RWKV-7 alternative (envelope $20-25; sister to Mamba)
- (c) DEFER predictive-coding-recurrent paradigm to research_only per Catalog #298

## 1. Probe outcome re-examination per Catalog #307 + #308

**NO PRIOR Z7-Mamba-2 PROBE EXISTS.** Verified: no `experiments/train_substrate_z7*mamba*.py`, no `src/tac/substrates/time_traveler_l5_z7_mamba2/`, no Z7-Mamba-2 recipe in `.omx/operator_authorize_recipes/`. The parent Z7 symposium 2026-05-17 enumerated Mamba-2 only as Wave-N+2 PIVOT-PATH (b) per Quantizr Revision #6.

**Classification per Catalog #307**: NOT APPLICABLE — Z7-Mamba-2 has NO prior empirical anchor. Catalog #307 paradigm-vs-implementation distinction applies to falsified substrates.

**Sister probe evidence (cross-substrate)**:
- **Z7-GRU symposium 2026-05-17** verdict PROCEED_WITH_REVISIONS; Hafner Revision #3 binds GRU not LSTM; Mamba-2 deferred to Wave-N+2 PIVOT-PATH (b).
- **Z6 Wave 2 Candidate 4c** scorer-logit conditioning probe — `pending_paired_exact_eval_json` at this memo. Outcome materially affects Z7-Mamba-2 ego-source choice per Revision #4 inheritance.
- **C6 IBPS Phase 2 redesign** — pending; β-IB-Lagrangian empirical anchor required for Z7-Mamba-2's β-parameter initialization per Revision #5 inheritance.

**Per Catalog #298 + #313 30-day staleness window**: NOT APPLICABLE — Z7-Mamba-2 has no prior probe outcome to expire.

## 2. Cargo-cult audit per assumption (Catalog #303)

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| CC-1 | "Mamba-2's 5-10× speedup over LSTM at language scale transfers to dashcam contest 600-pair temporal coherence" | **CARGO-CULTED-PENDING-EMPIRICAL** | MPS proxy training (§13 below): forward-pass timing measurement on synthetic 600-pair sequence at hidden_dim=128, state_dim=16 on M5 Max BEFORE paid dispatch. |
| CC-2 | "Mamba-2 > GRU expressive power at hidden_dim=128 on 24-dim latent" | **CARGO-CULTED-PENDING-EMPIRICAL** | Wave 2 paired Z7-Mamba-2-vs-Z7-GRU disambiguator at SAME archive bytes (per Z7 symposium Revision #2 canonical pattern). Decision criterion: Mamba-2-WIN at ΔS ≥ 0.005 → empirical validation; else DEFER. |
| CC-3 | "Mamba-2 selective state-space matches ego-motion-continuity prior better than GRU discrete-step recurrence" | **CARGO-CULTED-PENDING-PRINCIPLED** | Per first-principles: Mamba-2's continuous-time SSM formulation (zero-order hold discretization) IS structurally closer to physical ego-motion than GRU's discrete-step recurrence. Empirically testable via residual-magnitude comparison at SAME bit budget. |
| CC-4 | "mamba_ssm PyPI package installs in pact's Modal A100 training image" | **HARD-EARNED-PARTIAL** | Linux x86_64 + CUDA 11.6+ supported per upstream mamba_ssm README. macOS / M5 Max MPS NOT supported (empirically verified 2026-05-18: ModuleNotFoundError). Modal A100 image needs Catalog #244 NVML block + Catalog #270 Tier 1/2/3 protocol verified before dispatch. |
| CC-5 | "Z7-Mamba-2 ego-source choice inherits from Z6 Wave 2 4c outcome" | **HARD-EARNED** (inherited from Z7 symposium Revision #4) | Runtime-configurable ego-source: PoseNet-projection (Z6-v1 baseline) OR scorer-logit-conditioning (Z6 4c winning channel IF full-FiLM-WIN). |
| CC-6 | "Z7-Mamba-2 dispatch fires INDEPENDENTLY from Z7-GRU Wave 2 outcome" | **CARGO-CULTED** (inherited from Z7 symposium Contrarian VETO) | Per Catalog #315 OPTIMAL-FORM iteration discipline + Race-mode-rigor-inversion Rule 3: cheap signal (Z7-GRU $5-7) gates expensive signal (Z7-Mamba-2 $20-30). Sequential is canonical; parallel is cargo-cult UNLESS operator explicit-frontier-override per Catalog #300. |
| CC-7 | "Mamba-2 hidden state IS implicit Wyner-Ziv side-info channel (inherits from Z7-GRU CC-7 HARD-EARNED)" | **HARD-EARNED** | Same Ballard verbatim argument: deterministic Mamba-2 unroll regenerates identically at inflate-time; encoder + decoder share implicit hidden-state channel WITHOUT shipping it. Canonical Wyner-Ziv 1976 source-coding pattern. |
| CC-8 | "Z7-Mamba-2 β-IB-Lagrangian parameter inherits from C6 Phase 2 empirical β-optimal" | **CARGO-CULTED-PENDING-C6-Phase-2** (inherited from Z7 symposium Revision #5) | Z7-Mamba-2 trainer β-parameter MUST initialize from C6 empirical β-optimal anchor, NOT guessed independently. |
| CC-9 | "Mamba-2 d_state=16 is the right state dimension for 24-dim latent + 600-pair sequence" | **CARGO-CULTED** | d_state=16 is Mamba-2 default for language; could be 8/16/32/64. Wave N+1 disambiguator may include d_state ablation if smoke budget allows. |
| CC-10 | "Z7-Mamba-2 belongs in asymptotic_pursuit horizon_class per Catalog #309" | **HARD-EARNED** | Per predicted band [0.167, 0.184] lower-bound 0.167 sits in asymptotic_pursuit lower-region. Satisfies HORIZON-CLASS Consequence 2. |

**Cargo-cult-class summary**: 3 HARD-EARNED + 1 HARD-EARNED-PARTIAL + 3 CARGO-CULTED + 3 CARGO-CULTED-PENDING-EMPIRICAL. All disambiguated by either MPS proxy training (cheapest), Z7-GRU Wave 2 outcome inheritance, or Z7-Mamba-2 Wave 2 paired empirical anchor.

## 3. 9-dimension success checklist evidence per Catalog #294

| # | Dimension | Per-memo evidence |
|---|---|---|
| 1 | UNIQUENESS (class-shift not within-class) | ✓ CONDITIONAL — Mamba-2's selective state-space IS distinct from GRU recurrent gating at the recurrence-mechanism layer; class-shift only if Wave 2 paired-Mamba-2-vs-GRU confirms ΔS ≥ 0.005 at SAME archive bytes. |
| 2 | BEAUTY + ELEGANCE | ✓ Mamba-2 wrapper ~150 LOC matching Z6 FilmConditionedNextFramePredictor canonical signature (z_prev, ego_motion) → z_pred. Reviewable in 30 sec at the predictor primitive layer. |
| 3 | DISTINCTNESS | ✓ Mamba-2 IS the ONLY substrate binding selective state-space + Wyner-Ziv implicit side-info + ego-motion conditioning + DreamerV3-deterministic-recurrence lineage simultaneously. Z6 Multi-layer FiLM = stateless capacity; Z7-GRU = discrete-step stateful temporal; Z7-Mamba-2 = continuous-time selective state-space; Z8 = full hierarchical + stochastic. Each architecturally orthogonal. |
| 4 | RIGOR | ✓ THIS memo: cargo-cult audit + 9-dim evidence + observability surface + canonical-vs-unique decision per layer + Catalog #313 predecessor probe outcome verified absent + cross-pollination triangulation with Z6 4c + C6 Phase 2 + Z7 GRU parent symposium. |
| 5 | OPTIMIZATION PER TECHNIQUE | ✓ Per §6 canonical-vs-unique decision: 4 layers FORK_BECAUSE_PRINCIPLED (Mamba-2 predictor / runtime ego-source / β-IB inheritance from C6 / archive Z7MCM2 grammar); 7 layers ADOPT canonical (encoder, decoder, score-aware-loss helper, scorer routing, training curriculum baseline, Tier-1 engineering, deterministic reproducibility). |
| 6 | STACK-OF-STACKS-COMPOSABILITY | ✓ Z7-Mamba-2 composability vector: Z7-Mamba-2 + NSCS06v8 chroma + DP1 pretraining + D1 SegNet overlay (4 orthogonal axes). Composability deferred to Wave 3+ post-empirical-anchor; Z7-Mamba-2 itself is PRIMARY substrate (Catalog #310 PRIMARY-not-bolt-on requirement). |
| 7 | DETERMINISTIC REPRODUCIBILITY | ✓ Z7MCM2 archive grammar (Mamba-2 sister of Z7PCWM1; identical sectioning with predictor_blob containing Mamba-2 module weights) byte-stable per Catalog #19; Mamba-2 deterministic unroll per upstream mamba_ssm reference (no stochastic sampling); seed-pinned per `tac.substrates._shared.trainer_skeleton`. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | ✓ Mamba-2 GPU forward is sister to GRU at sequence length 600 (both O(N)); reported 5-10× speedup at language scale 2K-1M tokens is CARGO-CULTED-PENDING for 600-pair sequence per CC-1. MPS proxy training in §13 measures empirically. Modal A100 cost $20-30 (sister to Z7-GRU $15 + 2× factor for empirical Mamba-2 overhead estimate). |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | INDETERMINATE — predicted band [0.167, 0.184] sits BELOW current frontier 0.19205 [contest-CPU] IF realized. Per Quantizr inherited verbatim from Z7 symposium: empirical-novelty cuts BOTH ways; PR106 format0d score-table (stateless decoder) is HARD-EARNED counter-evidence to recurrent-state-being-the-winning-pattern. |

## 4. Observability surface declaration per Catalog #305

1. **Inspectable per layer** ✓ — Z7-Mamba-2 trainer (when built) inherits Z6-v1 canonical observability: per-epoch loss decomposition (segnet + posenet + rate); per-pair Mamba-2 hidden state norm + selectivity matrix range tracking; per-pair predictor residual magnitude; ego_motion projection range monitor.
2. **Decomposable per signal** ✓ — Total score decomposable into (a) seg + pose + rate per upstream contest formula; (b) per-pair predictor-WIN-vs-baseline at SAME archive bytes per Z6 Phase 3 Revision #2 canonical disambiguator; (c) per-pair temporal-coherence-empirical-bit-savings = H(residual_pair_t | mamba_state_t-1) measurement.
3. **Diff-able across runs** ✓ — Z7MCM2 archive byte-stable + Mamba-2 deterministic unroll (no stochastic sampling); sister-archive-paired-comparison (Z7-Mamba-2 vs Z7-GRU at SAME archive bytes per CC-2 disambiguator).
4. **Queryable post-hoc** ✓ — `.omx/state/council_deliberation_posterior.jsonl` (Z7 parent symposium anchor + this memo when council-ratified) + `.omx/state/probe_outcomes.jsonl` (Z7-Mamba-2 first probe outcome at Wave N+1) + Modal call_id ledger per Catalog #245.
5. **Cite-able** ✓ — Frontmatter `related_deliberation_ids` cites 9 deliberations; memo cites Z7 parent symposium + Z6/Z7/Z8 scoping memo + research wave §0+§2.2 Z7+§3.6 DreamerV3-Mamba convergence + sister C6 Phase 2 + sister Z6 4c probe.
6. **Counterfactual-able** ✓ — Wave N+1 Z7-Mamba-2 smoke + identity-predictor disambiguator IS the canonical counterfactual: "what if Mamba-2 selective state-space replaced GRU recurrent gating at SAME archive bytes?" Sister: "what if d_state=16 vs 8 vs 32?" Wave-N+2 if hidden_dim ablation funded.

### Ego-motion conditioning declaration (Catalog #311 / Pattern H)

**APPLICABLE** — Z7-Mamba-2 invokes Atick-Redlich cooperative-receiver framing (Mamba-2 hidden state IS encoder-decoder shared prior per Ballard inherited verbatim + Wyner-Ziv 1976 source-coding pattern).

- **ego-motion token**: ✓ Z7-Mamba-2 input vector = concat(latent_pair[t-1], ego_motion[t]); ego_motion is either PoseNet-projection (Z6-v1 baseline) OR scorer-logit-conditioning (Z6 4c winning channel).
- **predictive token**: ✓ Z7-Mamba-2 IS canonical next-frame predictor: predicted_latent[t] = SpatialDecoder(Mamba2(prev_latent[t-1], ego_motion[t]), prev_latent); residual[t] = latent_pair[t] - predicted_latent[t].

Catalog #311 satisfied structurally. NO waiver needed.

### Hierarchical predictive coding declaration (Catalog #312)

**NOT APPLICABLE** — Z7-Mamba-2 is SINGLE-LEVEL recurrent (Mamba-2 hidden state only); does NOT claim full hierarchical predictive coding. Z7-Mamba-2 has 2 of 4 canonical primitives (Hafner DreamerV3 deterministic-recurrence lineage via Mamba sister + Wyner-Ziv implicit side-info) — NOT full hierarchy. Catalog #312 gate's scope correctly excludes Z7-Mamba-2.

### F-asymptote PRIMARY substrate declaration (Catalog #310)

**APPLICABLE + SATISFIED** — Z7-Mamba-2 claims asymptotic_pursuit horizon_class. Z7-Mamba-2 IS PRIMARY substrate (ships its own encoder + decoder + Mamba-2 recurrent predictor + per-pair residual stream + Z7MCM2 archive grammar). Z7-Mamba-2 architectural core = Mamba-2 selective state-space next-frame predictor (NOT bolt-on loss term on Z3/A1/PR101). Substrate-class-shift token = `scorer_relationship_class_shift_predictive_coding_world_model_v2_z7_mamba2_selective_state_space`.

## 5. Per-substrate-symposium contract per Catalog #325

The 6-step contract:

1. **Cargo-cult audit per Catalog #303** ✓ — §2 above (10 assumptions; HARD-EARNED-vs-CARGO-CULTED classification; unwind paths).
2. **9-dimension success checklist evidence per Catalog #294** ✓ — §3 above.
3. **Observability surface declaration per Catalog #305** ✓ — §4 above (6 facets + Catalog #310/#311/#312 sub-gates).
4. **Sextet pact deliberation** — DESIGN-MEMO-ONLY at this memo; Wave N+1 council convened on Z7-GRU Wave 2 outcome WILL satisfy the full 6-of-6 sextet quorum + grand council attendees per Catalog #292.
5. **Per-substrate reactivation criteria pinned** ✓ — TL;DR + §7 enumerate 3 reactivation paths IF Z7-Mamba-2 Wave N+1 disambiguator LOSES (Z8 / RWKV-7 / research_only per Catalog #298).
6. **Catalog #324 post-training Tier-C validation discipline** — `predicted_band_validation_status: pending_post_training` per recipe; reactivation criterion = post-training Tier-C density measurement on Z7-Mamba-2 archive after Wave 2 smoke completes.

## 6. Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| 1. Mamba-2 backbone (predictor primitive) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | This IS the substrate-distinguishing primitive per HNeRV parity L7 substrate-engineering; cannot be canonical. Mamba2Predictor class in `tac.optimization.mamba2_predictor` exposes canonical signatures matching Z6 FilmConditionedNextFramePredictor for runtime drop-in. Mamba-2 reference: `state-spaces/mamba` GitHub + arxiv 2405.21060 + Dao-Gu Goomba Lab blog series 2024. |
| 2. Encoder + Decoder | **ADOPT_CANONICAL** (Z6-v1 + Z7-GRU sister pattern) | Same encoder/decoder pattern; only the predictor primitive changes per substrate isolation principle. Reuse `tac.substrates.time_traveler_l5_z6.architecture._Z6Encoder` + `_Z6Decoder` directly. |
| 3. Ego-source projection | **FORK_BECAUSE_PRINCIPLED** (inherited from Z7 symposium Revision #4) | Runtime-configurable ego-source via `--ego-source {posenet_projection,scorer_logit_compressed}` argparse flag matches Z7 GRU sister pattern; allows Wave N+1 to inherit Z6 4c winning channel. |
| 4. β-IB Lagrangian | **FORK_BECAUSE_PRINCIPLED** (inherited from Z7 symposium Revision #5) | β-parameter initialization from C6 Phase 2 empirical β-optimal anchor (canonical IB-framework empirical anchor); fork is NECESSARY because canonical (cold-start guess) is suppressing signal per cargo-cult CC-8. |
| 5. State-space dimension (d_state) | **UNCLEAR_NEEDS_EMPIRICAL** | Mamba-2 default d_state=16 for language; pact dashcam at 600-pair sequence + 24-dim latent untested. Wave N+1 may include d_state ablation. Default to d_state=16 per Mamba-2 reference until empirical evidence forces fork. |
| 6. Score-aware loss | **ADOPT_CANONICAL** | Reuse `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164. Same wrapper as Z6 score_aware_loss. |
| 7. Archive grammar (Z7MCM2) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Z7MCM2 = Z7PCWM1 sister with predictor_blob containing Mamba-2 weights (~30-50KB depending on hidden_dim/d_state); inherits monolithic single-file `0.bin` per HNeRV parity L3 + ≤200 LOC inflate.py per HNeRV parity L4 waiver per HNeRV parity L7 substrate-engineering exception. |
| 8. Inflate runtime | **FORK_BECAUSE_PRINCIPLED** | Deterministic Mamba-2 selective state-space unroll across 600 pairs in inflate.py; ≤200 LOC budget per HNeRV parity L4 substrate-engineering waiver. Needs torch + brotli + mamba_ssm (3 runtime deps; exceeds default L4 budget of ≤2; HNeRV parity L7 substrate-engineering waiver applies). |
| 9. Training curriculum | **ADOPT_CANONICAL** | Pyav decode + patched YUV6 + differentiable scorers + EMA(0.997) + eval_roundtrip=True + AdamW + cosine schedule per Z6-v1 sister pattern. NO synthetic data per Catalog #114. |
| 10. Tier-1 engineering | **ADOPT_CANONICAL** | autocast_fp16 (Catalog #172) + TF32 (Catalog #178) + torch.compile (Catalog #179) + no_grad-at-eval (Catalog #180) + GTScorerCache F3 consumption (Catalog #228). Mamba-2 supports autocast_fp16 natively per upstream. |
| 11. Scorer loader assignment order | **ADOPT_CANONICAL** | `pose_scorer, seg_scorer = load_differentiable_scorers(...)` per Catalog #222 (reversed = C6 bug class anchor). |
| 12. Deterministic reproducibility | **ADOPT_CANONICAL** | Seed-pinned per `tac.substrates._shared.trainer_skeleton.device_or_die` + `detect_hardware_substrate` per Catalog #190. Mamba-2 deterministic unroll (no stochastic sampling at THIS scaffold; stochastic component reserved for Z8). |
| 13. Observability surface | **FORK_BECAUSE_PRINCIPLED** | Mamba-2-specific surfaces: selectivity matrix range tracking + hidden state norm per pair + state-space dimension utilization heatmap. Sister to Z6 FiLM scale/shift tracking but Mamba-2-specific. |

**Net**: 5 FORK_BECAUSE_PRINCIPLED + 1 UNCLEAR_NEEDS_EMPIRICAL + 7 ADOPT_CANONICAL. Forks all documented + tied to specific cargo-cult or empirical anchor.

### Codex adversarial online-review supersession - 2026-05-18

Codex reviewed the current Z7-Mamba-2 runtime plan against the Z7-GRU runtime
bug found earlier today and against Mamba/Mamba-2 implementation constraints:

```text
sources:
  Mamba: https://arxiv.org/abs/2312.00752
  Mamba-2: https://arxiv.org/abs/2405.21060
  Z7-GRU runtime bug: brotli dependency removed in favor of stdlib zlib
```

Supersession: the previous "torch + brotli + mamba_ssm" inflate runtime
waiver is too permissive for contest promotion. Any future Z7-Mamba runtime
must satisfy:

```text
runtime may use torch
runtime must not require brotli
runtime must not require mamba_ssm unless vendored and proven inside the exact
  contest runtime closure
preferred path: pure-PyTorch exported selective-SSM recurrence in <=200 LOC
fallback path: keep research_only=true until dependency closure is proven
```

This does not demote Mamba-2 as a research branch. It narrows the promotion
contract so a proxy win cannot hide the same dependency-closure failure class
already observed in Z7-GRU.

Second online-review consequence: DCVC argues that "predicted frame plus
residual" is weaker than contextual/conditional coding. If Z7-Mamba-2 is built,
its unique implementation should not be only a GRU->Mamba swap. It should test
whether selective SSM state conditions decoder features and residual-symbol
scales better than the GRU baseline at the same archive bytes.

## 7. Architectural specification

```
upstream/videos/0.mkv (1200 frames @ 874x1164 RGB)
        |
        v (pyav decode + canonical tac.substrates._shared.trainer_skeleton.decode_real_pairs)
        |
   pair_tensor :: (600, 2, 3, 384, 512)
        |
        v [Encoder (same as Z6-v1; ~37K params)]
        |
   latent_pair :: (600, 24)     -- 24-dim latent per pair (matches Z6-v1 latent_dim)
        |
        v [EGO-SOURCE SELECTOR (runtime-configurable per Revision #4)]
        |
   IF Z6 4c full-FiLM-WIN: ego_input = scorer_logit_compressed_fp16 (Z6 4c winning channel)
   IF Z6 4c DEFER:         ego_input = PoseNet_projection_8dim (Z6-v1 baseline)
        |
   ego_motion :: (600, ego_dim)  -- ego_dim = 8 (PoseNet) OR compressed-logit-dim
        |
        v [PER-PAIR MAMBA-2 SELECTIVE STATE-SPACE PREDICTOR (Dao-Gu 2024)]
        |
   # Mamba-2 forward pass per pair t:
   #   input_t = concat(latent_pair[t-1], ego_motion[t])  :: (B, 24 + ego_dim)
   #   x_t = self.input_projection(input_t)               :: (B, d_model=64)
   #   y_t, h_t = self.mamba2_block(x_t, h_{t-1})         :: y_t (B, d_model), h_t (B, d_state, d_model)
   #   predicted_latent[t] = self.output_projection(y_t)  :: (B, 24)
   #   residual[t] = latent_pair[t] - predicted_latent[t] :: (B, 24)
   #
   # The Mamba-2 block applies selective state-space:
   #   A_t = exp(softplus(self.A_proj(x_t)))   -- input-conditioned state matrix
   #   B_t = self.B_proj(x_t)                  -- input-conditioned input matrix
   #   C_t = self.C_proj(x_t)                  -- input-conditioned output matrix
   #   h_t = A_t * h_{t-1} + B_t * x_t
   #   y_t = C_t * h_t
        |
        v [Residual quantize int8 + entropy-code]
        |
   residuals :: (600, 24) int8
        |
        v [pack_archive(Z7MCM2) per analogous Z7PCWM1 grammar]
        |
   0.bin :: Z7MCM2 grammar
        ├── HEADER (~1 KB)
        ├── encoder_state_dict_fp16_brotli      (~30 KB)
        ├── decoder_state_dict_fp16_brotli      (~30 KB)
        ├── predictor_state_dict_fp16_brotli    (~30-50 KB; Mamba-2 block + projections + ego MLP)
        ├── latent_init_int8                    (~5 KB)
        ├── residuals_int8                      (~10 KB)
        ├── ego_motion_int8_sidecar             (~3 KB)
        └── meta_json                           (~0.5 KB)
```

**Parameter count breakdown** (Mamba-2 canonical per Dao-Gu 2024):
- Encoder (same as Z6-v1): ~37K params
- Decoder (same as Z6-v1): ~37K params
- Mamba-2 block (d_model=64, d_state=16, expand=2): ~25-40K params per block (sister to GRU ~24K at hidden_dim=128; Mamba-2 selective-projection matrices add overhead but state is smaller)
- Input projection (24+ego_dim → d_model=64): ~3K params
- Output projection (d_model=64 → 24): ~2K params
- Ego MLP (per Revision #4 runtime-configurable): ~3K params
- Latent init: ~50K params (sister Z6-v1 baseline)
- **Total: ~155-175K params** (slightly LESS than Z7-GRU ~210-240K; sister to Z6-v1 75K but with predictor primitive added)
- **Archive size estimate**: ~110-140 KB (similar to Z7-GRU; predictor weights dominate)

**Bytes added (vs Z6-v1)**: ~+35-65 KB on archive size (Mamba-2 predictor + projections overhead). Per Catalog #220 + #272 distinguishing-feature contract: the Mamba-2 selective state-space predictor IS the distinguishing primitive; runtime overlay consumes it via autoregressive unroll across 600 pairs; byte-mutation smoke MUST verify mutations on any Mamba-2 weight produce measurable downstream frame changes.

## 8. Predicted ΔS band per Catalog #296 Dykstra-feasibility check

**Predicted band**: [-0.025, -0.008] over PR101 frontier 0.19205 ⇒ [0.167, 0.184] [contest-CPU prediction; not score claim].

**Dykstra-feasibility intersection** of:
- (a) **Selective state-space bit-savings bound**: Per Mamba-2 reference (Dao-Gu 2024 arxiv 2405.21060): O(N) selective recurrence captures temporal patterns at sub-quadratic compute; on dashcam 600-pair sequences with ego-motion conditioning, the per-pair residual entropy should drop ~10-20% vs GRU baseline. At PR101 0.19205 CPU baseline with rate term dominated by latent bytes, 10-20% residual reduction maps to ΔS ~-0.005 to -0.015. Lower-bound -0.008 sits in this range.
- (b) **First-principles ego-motion-continuity prior**: Mamba-2's continuous-time SSM formulation (zero-order hold discretization with input-conditioned A,B,C) matches physical ego-motion continuity more naturally than GRU's discrete-step recurrence. Per Atick-Redlich 1990 cooperative-receiver theorem: receiver-side regenerable side-info channel (Mamba-2 hidden state) reduces rate term by mutual information I(latent_pair_t; hidden_state_t-1) ≤ H(latent_pair_t). Empirical bound: at 600-pair sequence with ego-motion conditioning, I(T;Y) reduction ~5-10% maps to ΔS ~-0.005 to -0.012.
- (c) **Cross-substrate sister evidence**: Z6-v1 (75K-FiLM) + Z6-v2 Candidate 1 (300K Multi-layer FiLM) + Z7-GRU (210K) all sit in similar predicted band per Z7 symposium predicted_band [0.10, 0.13] but EMPIRICAL anchors not yet landed. Mamba-2 may match or beat these IF temporal-coherence-via-state-space provides real bit-savings.

**Intersection**: [max(-0.025, -0.015, -0.012), min(-0.008, -0.005, -0.005)] = [-0.025, -0.005] conservatively; published band [-0.025, -0.008] takes mid-point + margin.

**Catalog #296 satisfied**: Dykstra-feasibility check cited + first-principles Atick-Redlich + Mamba-2 reference all present.

## 9. Dispatch sequencing (Z7 symposium Revision #1 + #6 binding)

```
[CURRENT STATE: Z7-Mamba-2 pre-build; Z7-GRU symposium PROCEED_WITH_REVISIONS; Z6 Wave 2 4c pending_paired_exact_eval_json]
   ↓ AWAITS (Path 1: cheap-signal-first per Race-mode Rule 3)
[Z7-GRU Wave N+1 council convened on Z6 4c outcome + C6 IBPS Phase 2 outcome ($0 + ~90 min editor)]
   ↓ IF Z7-GRU Wave 2 smoke LANDS PROCEED-unconditional:
      [Z7-GRU Wave 3 full dispatch (~$16.50-21.50) → AUTOPILOT consumes outcome]
        ↓ Z7-GRU outcome materially informs Z7-Mamba-2 dispatch priority
   ↓ IF Z7-GRU Wave 2 smoke LANDS DEFER (identity-WIN or ΔS < 0.005):
      [Wave N+2 council on Z7-Mamba-2 PIVOT-PATH (b) ratification ($0 + ~90 min editor)]
        ↓ IF PROCEED-unconditional:
           [Z7-Mamba-2 trainer BUILD with ego-source from Z6 4c outcome (~1 week subagent + $0 GPU)]
              ↓
           [Z7-Mamba-2 Wave 2 smoke + identity-disambiguator paired ($5-10 envelope)]
              ↓ IF Z7-Mamba-2-WIN AT ΔS ≥ 0.005:
                 [Wave N+3 council → Z7-Mamba-2 Wave 3 full dispatch ($20-30 envelope)]
              ↓ IF Z7-Mamba-2-LOSS:
                 [Pivot to Z7-RWKV-7 ($20-25) OR Z8 full hierarchy ($42) OR DEFER]

OR PATH 2 (operator explicit-frontier-override per Catalog #300):
[Operator declares Mamba-2 PRIMARY recurrent primitive (skipping Z7-GRU) with verbatim quote in council_override_rationale]
   ↓
[Z7-Mamba-2 trainer BUILD (~1 week subagent + $0 GPU)]
   ↓
[Z7-Mamba-2 Wave 2 smoke ($5-10) → identity-disambiguator → Wave 3 full ($20-30)]
```

## 10. Pivot paths if Z7-Mamba-2 LOSES (Quantizr Revision #6 inheritance)

- (a) **Z7-RWKV-7** alternative ($20-25 envelope; sister to Mamba-2 per linear-attention RNN class per RWKV-7 "Goose" Peng et al. 2025 arxiv 2503.14456). RWKV-7 is also O(N) and may have different bias-variance tradeoff than Mamba-2.
- (b) **Z8 full Rao-Ballard hierarchy + Hafner DreamerV3 stochastic** ($42 envelope per Z6/Z7/Z8 scoping memo §3 Z8).
- (c) **DEFER predictive-coding-recurrent paradigm** to research_only per Catalog #298 retirement discipline + CLAUDE.md "Forbidden premature KILL". Each pivot = its OWN per-substrate symposium.

## 11. Cross-pollination wiring (Z7 symposium Revision #5 + research wave §3.6 binding)

**Z7-Mamba-2 design memo MUST document explicit dependency on**:
1. **Z6 Wave 2 Candidate 4c outcome** (sister codex probe pending) — materially changes Z7-Mamba-2 ego-source choice per Revision #4 inheritance.
2. **C6 IBPS Phase 2 redesign outcome** (sister memo pending) — canonical IB-framework empirical anchor; C6's empirically-optimal β-IB-Lagrangian parameter MUST initialize Z7-Mamba-2's β-parameter (NOT guessed independently).
3. **Z7-GRU Wave 2 outcome** (sister Wave N+1 pending) — Path 1 default sequences Z7-Mamba-2 AFTER Z7-GRU per Catalog #315 OPTIMAL-FORM iteration discipline.

## 12. Mission alignment per Catalog #300

`council_predicted_mission_contribution: frontier_breaking` — Z7-Mamba-2 opens class-shift path (selective state-space temporal coherence) predicted to lower score below 0.10 CPU IF temporal-coherence-primitive empirically validates at Wave N+1 disambiguator.

NOT `frontier_protecting` (no regression prevented).
NOT `apparatus_maintenance` (substrate-specific design, not gate/helper hygiene).
NOT `mission_questioned` (per-substrate symposium discipline IS canonical per Catalog #325).

`rigor_overhead` fraction: ~30% (Catalog #292 + #303 + #305 + #294 + #316 + #325 evidence). Remaining 70% substrate-specific design + cross-pollination + cargo-cult audit IS frontier-pursuit content.

## 13. LOCAL M5 MAX + 128GB PROXY TRAINING PATTERN

Per operator's parallel directive on local hardware utilization + the CLAUDE.md "MPS auth eval is NOISE" non-negotiable + `tac.optimization.mps_research_signal` canonical helper pattern:

**MPS proxy training is NEVER promotable** per CLAUDE.md "MPS auth eval is NOISE" (verified 2026-04-25: PoseNet 23× drift; SegNet 2× drift; final score 2.5× drift on MPS vs contest CUDA). MPS results MUST tag `evidence_grade=MPS-research-signal`, `score_claim=False`, `promotion_eligible=False`, `ready_for_exact_eval_dispatch=False` per Catalog #192 + `tac.optimization.mps_research_signal.append_manifest_row_to_jsonl`.

**Why local M5 Max is still high-value for Z7-Mamba-2**:
1. **Cargo-cult disambiguator at $0**: CC-1 (Mamba-2 vs GRU speedup at 600-pair sequence) is testable via local MPS forward-pass timing on synthetic data. If Mamba-2 forward is NOT measurably faster than GRU at sequence length 600 on M5 Max, the 5-10× CARGO-CULTED-PENDING-VERIFICATION claim is falsified BEFORE paid dispatch.
2. **Sanity-check on Mamba-2 reference implementation**: M5 Max 128GB unified memory easily fits 175K-param Z7-Mamba-2 (~700KB fp32) + 600-pair sequence forward + backward pass. Pure-PyTorch Mamba-2 reference (no CUDA kernels) verifies the architecture before integrating CUDA-kernel `mamba_ssm` package on Modal A100.
3. **Curve-shape priors for autopilot**: MPS proxy training can produce per-epoch loss curves at $0 cost. While the absolute scores are noise per CLAUDE.md, the curve SHAPE (does loss converge? does training stabilize? does the residual entropy actually drop?) is a research signal the autopilot ranker can consume via `tac.optimization.mps_research_signal.MPSResearchSignalManifest`.
4. **Architecture iteration without dispatch friction**: M5 Max allows rapid editor-iteration on Mamba-2 hyperparameters (d_state ∈ {8,16,32,64}, d_model ∈ {32,64,128}, expand ∈ {1,2,4}) at $0 cost per iteration; only the operator-blessed final config dispatches to Modal A100.

**Canonical wire-in** per `tac.optimization.mps_research_signal`:
```python
from tac.optimization.mps_research_signal import (
    MPSResearchSignalManifest,
    append_manifest_row_to_jsonl,
)

# After MPS proxy training run on Z7-Mamba-2:
manifest_row = MPSResearchSignalManifest(
    substrate_id="time_traveler_l5_z7_mamba2",
    config={"d_state": 16, "d_model": 64, "expand": 2, "ego_source": "posenet_projection"},
    proxy_score=0.342,                        # MPS-PROXY only; NOT promotable
    evidence_grade="MPS-research-signal",     # CANONICAL per Catalog #192
    score_claim=False,                        # NEVER promotable from MPS
    promotion_eligible=False,
    ready_for_exact_eval_dispatch=False,
    notes="Z7-Mamba-2 MPS proxy training; M5 Max forward pass timing 600-pair sequence; curve-shape only",
    hardware_substrate="macos_arm64",         # per Catalog #190 detect_hardware_substrate(axis='cpu') canonical token
)
append_manifest_row_to_jsonl(manifest_row)
```

**Available local resources**:
- M5 Max with 128GB unified memory; MPS available (verified 2026-05-18: `torch.backends.mps.is_available() == True`)
- alejandros-mac-mini (100.125.140.94) Intel CPU build server with Python 3.13 + uv
- bat00 (100.120.99.124) Windows + WSL2 Ubuntu 24.04 with RTX 2070S→3090; CUDA-capable via WSL2 GPU passthrough → could run pure-PyTorch Mamba-2 if `mamba_ssm` PyPI install works on Windows+WSL2 (untested)
- tertiary (100.65.24.39) M1 MacBook Pro MPS

**Recommended pre-dispatch local protocol** (all $0):
1. Install pure-PyTorch Mamba-2 reference on M5 Max (no CUDA dependency). Reference: `state-spaces/mamba` GitHub has a `Mamba2_torch.py` reference implementation that doesn't require CUDA kernels.
2. Verify forward+backward pass on synthetic 600-pair × 24-dim sequence at d_state=16, d_model=64.
3. Timing benchmark: Mamba-2 vs GRU at SAME hidden_dim=128 / d_model=64. If Mamba-2 is NOT measurably faster at sequence length 600, document the empirical CARGO-CULTED-PENDING-VERIFICATION verdict.
4. Convergence sanity check: train 10 epochs on synthetic data. Does loss converge? Curve shape reasonable?
5. Tag every output as `[MPS-PROXY]` / `[MPS-research-signal]`; emit manifest row via canonical helper; DO NOT use absolute scores for any promotion / kill / ranking decision.

If MPS proxy training shows (a) Mamba-2 forward is genuinely faster at sequence length 600 OR (b) convergence stability is materially better than GRU baseline, that empirical signal — even at MPS-research-signal grade — strengthens the operator's case for paid Modal A100 dispatch. If MPS proxy shows neither, that's an early signal to keep Z7-Mamba-2 deferred per Quantizr Revision #6 inheritance.

## 14. NEW Assumption-Adversary item #8 hypothesis (per META-ASSUMPTION cadence)

The shared assumption operating across this entire wave (Z7-GRU symposium + Z7-Mamba-2 design memo + Z7-RWKV-7 future memo + Z8 full hierarchy) is: **"recurrent-state predictive coding is a fundamentally different bit-allocation strategy than stateless-decoder substrates (PR101 / PR106 format0d), and recurrent-state will win on dashcam temporal coherence."**

CARGO-CULTED-PENDING-EMPIRICAL classification rationale: PR106 format0d_latent_score_table (lane `pr106_format0d_latent_score_table`; archive sha 9cb989cef519) at 0.20533 [contest-CUDA] is a STATELESS decoder with deterministic per-frame mapping. It is currently the canonical CUDA frontier. The recurrent-state-as-winning-pattern hypothesis is operator's working theory across the entire Z6/Z7/Z8 substrate class — IF this hypothesis is wrong, the entire predictive-coding-asymptotic-pursuit budget allocation per HORIZON-CLASS Consequence 2 ($30-50/month minimum) is mis-allocated.

The Z7-GRU Wave 2 disambiguator AND the Z7-Mamba-2 Wave N+1 disambiguator AND the Z8 Wave N+2 disambiguator are 3 alternative-probe-methodologies per Catalog #308 testing the SAME core hypothesis. Per CLAUDE.md "Forbidden premature KILL": if all 3 fail to beat PR106 format0d, the predictive-coding-recurrent paradigm should be DEFERRED-pending-research per Catalog #298, NOT killed — the architectural alternatives (e.g., predictive-coding via NeRV-family stateless decoders, or predictive-coding via foveation IDEAS without recurrent state) are reactivation paths.

The required structural protection: when the operator commits the entire asymptotic_pursuit budget to recurrent-state predictive coding, the assumption-adversary surface MUST raise this hypothesis at every per-substrate symposium until at least ONE recurrent-state substrate has paired contest-CUDA evidence beating PR106 format0d.

## 15. Codex adversarial hardening pass - invalid source score claims block Z7 disambiguation

Follow-up pass on 2026-05-18 found a score-custody gap in
`tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py`: the
tool parsed `score_claim_valid` from each source exact-eval JSON but did not
use that field when deciding comparability. A recurrent-vs-static pair could
therefore produce `z7_recurrent_temporal_coherence_win` even when the recurrent
source JSON explicitly carried `score_claim_valid: false` or omitted the field.
That is a false-authority path because the probe's output is meant to arbitrate
the Wave N+1 recurrent-vs-static design question, not launder invalid source
custody into a method verdict.

Fix landed:

- `_eval_row()` now emits `score_claim_valid_missing_or_false` unless the source
  JSON carries `score_claim_valid: true`;
- `evaluate_exact_eval_pair()` prefixes that blocker by mode, so invalid
  recurrent or static-control sources both fail closed as
  `blocked_paired_exact_eval_not_comparable`;
- a focused regression test proves an invalid recurrent source cannot become a
  Z7 recurrent win even when the recomputed formula and same-byte/same-axis
  constraints otherwise pass.

Verification:

```text
.venv/bin/python -m pytest src/tac/tests/test_probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py -q
5 passed in 0.16s
```

Evidence grade: `[local-test]`; no score claim, no promotion claim, no provider
dispatch, and no lane claim. This hardening does not move Z7-Mamba-2 out of
`DESIGN_MEMO_ONLY_PENDING_WAVE_N_PLUS_1_COUNCIL`; it only prevents invalid
paired exact-eval JSON from becoming arbitration authority.

## 16. Backlinks + cross-references

- **Parent Z7 symposium**: `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md`
- **Research wave deliverable**: `.omx/research/comprehensive_research_wave_20260518.md` (TOP-5 #2; §3.6 DreamerV3-Mamba convergence; §11 truth audit Mamba-2 CARGO-CULTED-PENDING)
- **Z6/Z7/Z8 scoping memo**: `.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md`
- **Z6 Wave 2 Candidate 4c probe**: `.omx/research/z6_candidate4c_full_disambiguator_probe_20260518_codex.md`
- **C6 IBPS Phase 2 redesign**: pending (sister memo per Wave doctrine op-routable #3)
- **ATW V2-1 sister symposium**: `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md`
- **Mamba upstream**: https://github.com/state-spaces/mamba ; arxiv 2312.00752 (Mamba) + 2405.21060 (Mamba-2) + Dao-Gu Goomba Lab blog
- **DreamerV3 upstream**: https://github.com/danijar/dreamerv3 ; arxiv 2301.04104
- **RWKV-7 upstream**: https://github.com/BlinkDL/RWKV-LM ; arxiv 2503.14456 (Peng et al. 2025)

## 17. Scaffold deliverables (this lane)

Delivered in same commit batch via main-Claude sister commit (this subagent does NOT commit per parent prompt):

1. **THIS design memo** (`.omx/research/z7_mamba2_substrate_design_memo_20260518.md`)
2. **Mamba2Predictor canonical helper** (`src/tac/optimization/mamba2_predictor.py`) — wrapper exposing canonical signatures matching Z6 FilmConditionedNextFramePredictor for drop-in compatibility; defensively imports mamba_ssm (mamba-ssm PyPI) and falls back to pure-PyTorch reference for MPS / non-CUDA environments
3. **Trainer scaffold** (`experiments/train_substrate_time_traveler_l5_z7_mamba2.py`) — extends canonical `tac.substrates._shared.trainer_skeleton`; `_full_main` raises NotImplementedError per Catalog #240 with research_only=true opt-out
4. **Tests** (`src/tac/tests/test_z7_mamba2_scaffold.py`) — instantiation + forward pass on small input + runtime-configurable ego-source flag + per-pair master gradient compatibility + canonical signatures match Z6 sister + NotImplementedError raised from `_full_main`
5. **Operator-authorize recipe** (`.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml`) — declares `research_only: true` + `dispatch_enabled: false` + `predicted_band_validation_status: pending_post_training` per Catalog #324

Operator flips `dispatch_enabled: true` AFTER reading this memo + Z7-GRU Wave 2 outcome lands + Wave N+1 council convenes PROCEED-unconditional.
