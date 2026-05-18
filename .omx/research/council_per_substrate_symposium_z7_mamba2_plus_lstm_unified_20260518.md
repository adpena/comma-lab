---
name: council-per-substrate-symposium-z7-mamba2-plus-lstm-unified-20260518
metadata:
  node_type: memory
  council_tier: T3
  council_attendees:
    - Shannon
    - Dykstra
    - Yousfi
    - Fridrich
    - Contrarian
    - Assumption-Adversary
    - Rao
    - Ballard
    - Tishby_memorial
    - Zaslavsky
    - Hafner
    - Wyner
    - Atick
    - Redlich
    - Quantizr
    - Schmidhuber
  council_quorum_met: true
  council_verdict: PROCEED_WITH_REVISIONS
  council_dissent:
    - member: Contrarian
      verbatim: "Two substrates symposiated jointly is NOT a substitute for two independent per-substrate symposiums. I rise to challenge the framing. The PRIMARY (Z7-Mamba-2) inherits Hafner Revision #3 binding from the parent Z7-LSTM symposium 2026-05-17 (`council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md`) which itself bound the canonical recurrent primitive to GRU not LSTM. The FALLBACK (Z7-LSTM) trainer has now been built via codex sister wave; this means we have BOTH primitives implemented but neither has paired contest-CUDA/CPU evidence, AND neither has had the Z6 4c outcome materially applied to the ego-source choice. The Z6 4c paired exact-eval lands 2026-05-18 at ~90.58 [contest-CUDA] and ~90.58 [contest-CPU] (zero-epoch packet — NOT a trained model — paired delta 0.0028 BELOW decision_delta_s=0.005 on both axes). This is NEITHER a full-FiLM WIN nor a clean DEFER on the actual paradigm; it is a measurement of a zero-epoch packet with neither paradigm validated. My VETO is on any wording that treats the Z6 4c paired-zero-epoch as 'evidence for choosing scorer-logit ego-source over PoseNet-projection' — both ego-source choices remain CARGO-CULTED-PENDING-EMPIRICAL until a TRAINED Candidate 4c lands paired exact-eval. PROCEED on design completion + integration audit + decision tree authorization at THIS verdict; VETO on any pre-authorization of Z7 trainer `_full_main` IMPLEMENTATION at this verdict. Wave N+1 council required after a TRAINED Z6 4c paired exact-eval lands OR after operator explicit-frontier-override per Catalog #300."
    - member: Assumption-Adversary
      verbatim: "Per the MANDATORY Catalog #291 item #8 assumption-challenge axis. The SHARED ASSUMPTION operating across this unified symposium: *'Z7-Mamba-2 PRIMARY and Z7-LSTM FALLBACK are architecturally distinct enough that BOTH should be carried forward as alternative-probe-methodologies per Catalog #308 N>=3 alternative recurrent primitives.'* I classify this HARD-EARNED-PARTIAL with NEW item #8 hypothesis amendment. HARD-EARNED basis: Mamba-2 (Dao-Gu 2024 arxiv 2405.21060) IS architecturally distinct from GRU (DreamerV3 deterministic recurrence) at the selective-state-space-vs-discrete-gating layer; per Catalog #308 N>=3 alternative-probe-methodologies for the recurrent-predictive-coding substrate class, Mamba-2 + GRU/LSTM + RWKV-7 (Wave-N+3 future) is the canonical 3-alternative enumeration. NEW item #8 amendment per the 2026-05-18 sister Wave-1 symposium TT5L #866 outcome (25ep CUDA 3.9007 ALL-ZERO side-info — empirically pre-confirmed by Z7-LSTM trainer scaffold-only state at this memo): the implicit shared assumption per item #8 (substrate-class-shift-CANDIDACY-vs-VALIDATION) is *'recurrent-state predictive coding will provide MEANINGFUL bit-savings on dashcam 600-pair temporal coherence at contest scorer relevant subspace.'* This IS the PARADIGM-level question per Catalog #307; it is independent of the Mamba-2-vs-GRU choice within the paradigm. Z7-Mamba-2 trained AND Z7-LSTM trained AND Z8 hierarchical trained = 3 alternative-probe-methodologies per Catalog #308 testing the SAME core paradigm. PR106 format0d_latent_score_table at 0.20533 [contest-CUDA] is stateless decoder with deterministic per-frame mapping = HARD-EARNED counter-evidence to recurrent-state-being-the-winning-pattern. The assumption-violation hypothesis: *'IF all 3 recurrent-state alternative-probe-methodologies fail to beat PR106 format0d on contest-CUDA, the predictive-coding-recurrent paradigm should be DEFERRED-pending-research per Catalog #298 and #313, NOT killed per CLAUDE.md Forbidden premature KILL — the reactivation paths (NeRV-family stateless predictive coding, foveation IDEAS without recurrent state) become the canonical research-path-forward.'* Required action per Catalog #325 + #315 iteration discipline: this Z7-Mamba-2 + Z7-LSTM unified symposium is FRAMED AS pre-cached Wave-N+1 design verdict on the dual-primitive paradigm. The reactivation-path enumeration MUST reflect that BOTH Z7-Mamba-2 + Z7-LSTM dispatch are downstream from the Z6 4c trained-paradigm-question outcome, NOT just the zero-epoch packet that landed 2026-05-18. My verdict: PROCEED_WITH_REVISIONS on design completion + integration audit; VETO on any wording that pre-authorizes Z7-Mamba-2 OR Z7-LSTM dispatch from THIS verdict. Wave N+1 council required after the predictive-coding-paradigm-question is empirically tested with a TRAINED checkpoint (NOT zero-epoch). The op-routable that resolves the paradigm question fastest is the Z7-LSTM 100ep smoke at MPS proxy training first per CLAUDE.md MPS-research-signal pattern (free; non-promotable; produces curve-shape evidence) BEFORE paid Modal dispatch."
    - member: Hafner
      verbatim: "DreamerV3 architect; canonical author of Recurrent State-Space Model (RSSM). I refine my Z7-LSTM symposium Revision #3 binding: in Z7-LSTM, GRU is canonical per DreamerV3 deterministic recurrence; in Z7-Mamba-2, selective state-space IS the substrate-distinguishing primitive AND inherits the DreamerV3 deterministic-recurrence lineage at a DIFFERENT architectural layer (Mamba-2's selective A_t,B_t,C_t matrices ARE the input-conditioned state-update analogous to RSSM's GRU+stochastic — but Mamba-2 is purely deterministic at THIS scaffold; the stochastic component is reserved for Z8 full RSSM). My binding for THIS unified symposium: BOTH Z7-Mamba-2 + Z7-LSTM (canonical-bound to GRU per my prior revision) are valid alternative-probe-methodologies; Mamba-2 PRIMARY is HARD-EARNED per my deep-research-wave Section 3.6 DreamerV3↔Mamba convergence citation; GRU FALLBACK is canonical for graceful degradation when mamba_ssm PyPI package is unavailable (M5 Max MPS / Windows / Linux without CUDA toolkit). The dispatch sequencing: per Race-mode-rigor-inversion Rule 3 cheap-signal-first, GRU FALLBACK dispatches BEFORE Mamba-2 PRIMARY because GRU is sister to existing Z6 FiLM architecture (no mamba_ssm install required; ~$5-7 envelope sister to Z6-v2 Wave 2 Candidate 1). Mamba-2 PRIMARY dispatch FIRES ONLY IF (a) GRU FALLBACK lands DEFER OR (b) GRU FALLBACK lands WIN at ΔS ≥ 0.005 and Mamba-2 PRIMARY is the canonical next-iteration alternative per Catalog #308 OR (c) operator explicit-frontier-override per Catalog #300 + verbatim quote in council_override_rationale. My verdict: PROCEED on design completion + integration audit + Wave N+1 pre-cached deliberation; binding revision = GRU FALLBACK dispatches BEFORE Mamba-2 PRIMARY in the canonical iteration cascade."
    - member: Atick
      verbatim: "Atick-Redlich 1990 cooperative-receiver framing seat. The unified Z7-Mamba-2 + Z7-LSTM symposium MUST clarify Catalog #311 ego-motion-conditioning declaration jointly. BOTH substrates invoke cooperative-receiver framing via Mamba-2/GRU hidden-state-IS-shared-prior-channel per Ballard 2026-05-17 verbatim + Wyner-Ziv 1976 source-coding pattern. The runtime-configurable ego-source flag per Z7-LSTM Revision #4 (PoseNet-projection baseline OR scorer-logit-conditioning Z6 4c winning channel) is INHERITED into Z7-Mamba-2 via Hafner sister architectural alignment. CRITICAL: the Z6 4c trained-paradigm-question outcome is NOT YET available — the 2026-05-18 paired zero-epoch packet lands paired delta 0.0028 BELOW decision_delta_s=0.005 on both axes, NEITHER a full-FiLM WIN nor a clean DEFER. Per Catalog #311 ego-motion-conditioning requirement: BOTH Z7-Mamba-2 + Z7-LSTM trainers MUST support BOTH ego-source choices at runtime via --ego-source flag (verified via test_runtime_configurable_ego_source_posenet_projection_baseline + test_runtime_configurable_ego_source_scorer_logit_compressed which BOTH PASS at this memo). The empirical-anchor budget can re-use the best-channel-from-Z6 4c TRAINED outcome (Wave N+1) WITHOUT a separate Z7 ego-source ablation. My verdict: PROCEED on design completion with explicit Catalog #311 ego-source dual-support requirement re-confirmed."
    - member: Wyner
      verbatim: "Wyner-Ziv 1976 source-coding-with-side-information theorem seat. For Z7-Mamba-2 + Z7-LSTM jointly: BOTH substrates use the recurrent-state-as-implicit-side-info channel pattern. The hidden state is regenerated identically at inflate-time via deterministic unroll (Mamba-2 selective state-space OR GRU recurrence); encoder + decoder share the implicit channel WITHOUT shipping it. This IS the canonical Wyner-Ziv source-coding pattern applied to temporal prediction. The empirical rate-savings bound: I(latent_pair_t; hidden_state_t-1) ≤ H(latent_pair_t). Per the Z7-LSTM symposium Tishby_memorial verbatim: H(T) trivially exceeds MEANINGFUL_CONDITIONING MI threshold (0.5 bits/symbol × ~95K symbols = ~47K bits floor); the remaining empirical question is I(T;Y) — whether the recurrent hidden state's content is in the scorer-relevant subspace. Mamba-2's selective state-space (input-conditioned A_t,B_t,C_t) and GRU's discrete-gating capture DIFFERENT subspaces of the temporal coherence — empirically untested at contest scale. The Z7 disambiguator probe `tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` (5 tests pass at this memo) IS the canonical apparatus for measuring this empirically at Wave N+1. My verdict: PROCEED on integration audit + Wyner-Ziv discipline re-confirmed across BOTH substrates."
    - member: Tishby_memorial
      verbatim: "Memorial seat conveying the IB framework. For the unified Z7-Mamba-2 + Z7-LSTM symposium: BOTH substrates inherit the β-IB-Lagrangian initialization from C6 IBPS Phase 2 empirical anchor per Z7-LSTM symposium Revision #5 binding. C6 IBPS Phase 2 symposium 2026-05-18 verdict is PROCEED_WITH_REVISIONS (per `council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518.md`) with parallel β_ib + latent_dim sweep authorized under $5 envelope cap (Contrarian dissent binding). The empirical β-optimal anchor IS NOT YET LANDED — C6 IBPS Phase 2 paired exact-eval dispatch is pending. Both Z7-Mamba-2 + Z7-LSTM trainers MUST initialize β-parameter from C6 empirical β-optimal anchor when it lands; until then, the β-parameter SHOULD default to literature-canonical values (β=0.5 for IB-information-plane canonical; per Tishby-Zaslavsky 2015 deep-learning IB studies). My verdict: PROCEED on design completion with explicit cross-pollination wiring to C6 IBPS Phase 2 redesign reaffirmed for BOTH Z7-Mamba-2 + Z7-LSTM. Z7 trainer-side β-parameter selection deferred to C6 Phase 2 outcome OR operator explicit-frontier-override."
    - member: Quantizr
      verbatim: "Adversarial reverse-engineering seat. The unified symposium reveals an important triangulation: Z7-Mamba-2 PRIMARY + Z7-LSTM FALLBACK + Z6 Multi-layer FiLM (depth=3 ~300K params, Phase 3 Candidate 1) are 3 alternative-probe-methodologies per Catalog #308 testing the SAME core hypothesis: 'recurrent-state predictive coding will provide MEANINGFUL bit-savings on dashcam 600-pair temporal coherence.' PR106 format0d_latent_score_table at 0.20533 [contest-CUDA] is stateless decoder with deterministic per-frame mapping = HARD-EARNED counter-evidence to recurrent-state-being-the-winning-pattern. PR101 frame_exploit_selector_fec6 at 0.19205 [contest-CPU] is also stateless. The recurrent-state-as-winning-pattern hypothesis is the operator's working theory across the entire Z6/Z7/Z8 substrate class — IF this hypothesis is wrong on contest-CUDA scorer, the entire predictive-coding-asymptotic-pursuit budget allocation per HORIZON-CLASS Consequence 2 ($30-50/month minimum) is mis-allocated. My verdict: PROCEED on design + integration audit; binding revision = Z7-Mamba-2 + Z7-LSTM trained empirical anchor MUST be paired-compared against PR106 format0d at SAME archive bytes per Z6 Phase 3 Revision #2 canonical pattern. Decision criterion: Z7-WIN at ΔS ≥ 0.005 vs PR106 → recurrent-state-as-winning-pattern empirically validated; Z7-LOSS or |ΔS| < 0.005 → recurrent-state primitive falsified at single-level surface; advance to Z8 OR DEFER per Catalog #308."
    - member: Schmidhuber
      verbatim: "Compression-as-intelligence + MDL seat. The unified Z7-Mamba-2 + Z7-LSTM symposium IS a canonical instance of the compression-as-intelligence research pattern: ALL recurrent-state predictive coding substrates are testing whether learned temporal prediction provides bit-savings on dashcam contest video. Per MDL principle: the optimal substrate minimizes joint description length L(model) + L(data | model). Z7-Mamba-2's selective state-space (input-conditioned A_t,B_t,C_t) is a parametric model class; Z7-LSTM's discrete gating is a different parametric model class; both compete against stateless decoders (PR106 format0d, PR101 grammar-bolt-on). My binding contribution: per MDL discipline, BOTH Z7-Mamba-2 + Z7-LSTM L(model) sizes are comparable (~155-175K params for Mamba-2 vs ~210-240K for LSTM; ~155K-240K range vs PR106 format0d unknown but likely <100K). IF Z7 L(data|model) > PR106 L(data|model), recurrent-state primitive is empirically falsified; IF Z7 L(data|model) < PR106 L(data|model) AND ΔS ≥ 0.005, recurrent-state primitive is empirically validated. My verdict: PROCEED on design + integration audit with explicit MDL-discipline requirement that Wave N+1 dispatch MUST include per-substrate L(model) + L(data|model) decomposition for apples-to-apples comparison."
  council_assumption_adversary_verdict:
    - assumption: "Z7-Mamba-2 PRIMARY and Z7-LSTM FALLBACK are both valid alternative-probe-methodologies per Catalog #308 N>=3 recurrent primitives"
      classification: HARD-EARNED
      rationale: "Per Hafner Revision #3 binding (Z7-LSTM symposium 2026-05-17) + research wave §3.6 DreamerV3↔Mamba convergence: Mamba-2 (Dao-Gu 2024 arxiv 2405.21060) is architecturally distinct from GRU at selective-state-space-vs-discrete-gating layer. Catalog #308 N>=3 satisfied: Z7-Mamba-2 + Z7-GRU (canonical-bound from Z7-LSTM symposium Revision #3) + Z7-RWKV-7 (Wave-N+3 future per Z7-LSTM symposium Revision #6) = 3 alternatives. Binding."

    - assumption: "Z6 4c paired zero-epoch paired exact-eval (2026-05-18; delta 0.0028 BELOW decision_delta_s=0.005) constitutes evidence for choosing scorer-logit ego-source over PoseNet-projection in Z7-Mamba-2/LSTM"
      classification: CARGO-CULTED
      rationale: "Per Contrarian verbatim above + sister codex memo `z6_candidate4c_full600_zeroepoch_handoff_20260518_codex.md` line 263: 'This is a measured zero-epoch configuration failure, not a Candidate 4c method kill. Full FiLM is lower than identity on [contest-CUDA], but the score delta is below decision_delta_s=0.005, and both scores are far above the current frontier. The zero-epoch packet must therefore remain a control/anchor only.' The Z6 4c paired-zero-epoch is NEITHER a full-FiLM WIN nor a DEFER on the paradigm question; it is a measurement of a zero-epoch packet with neither paradigm validated. Z7 ego-source choice MUST remain runtime-configurable per Catalog #311 + Z7-LSTM symposium Revision #4 binding; the empirical winning channel is UNKNOWN until a TRAINED Candidate 4c paired exact-eval lands."

    - assumption: "BOTH Z7-Mamba-2 AND Z7-LSTM dispatches should fire BEFORE Z8 hierarchical (Path 1 default cheap-signal-first per Race-mode-rigor-inversion Rule 3)"
      classification: HARD-EARNED
      rationale: "Per Hafner Revision (THIS symposium): GRU FALLBACK dispatches FIRST (~$5-7 envelope sister to Z6-v2 Wave 2 Candidate 1; no mamba_ssm install required), Mamba-2 PRIMARY dispatches SECOND (~$20-30 envelope), Z8 hierarchical FOURTH (~$42 envelope per Z6/Z7/Z8 scoping memo). The cascade sequences cheap signal before expensive per Race-mode-rigor-inversion Rule 3 + Catalog #315 OPTIMAL-FORM iteration discipline. Z7-Mamba-2's higher GPU envelope is justified ONLY IF Z7-LSTM/GRU FALLBACK lands either (a) WIN at ΔS ≥ 0.005 (validating recurrent-state-primitive empirically) OR (b) DEFER (and operator authorizes Mamba-2 as canonical next-iteration alternative)."

    - assumption: "GRU (not LSTM) is the canonical recurrent primitive for the Z7-FALLBACK substrate per Hafner DreamerV3 deterministic-GRU lineage (Z7-LSTM symposium Revision #3 binding)"
      classification: HARD-EARNED
      rationale: "Per Hafner Revision #3 of the Z7-LSTM symposium 2026-05-17 (binding): GRU is canonical recurrent primitive for DreamerV3 deterministic-recurrence lineage. The Z7-LSTM codename remains for backward compatibility but the trainer implementation IS GRU per `src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/architecture.py::GruRecurrentPredictor` (verified at this memo). Z7-LSTM codename + GruRecurrentPredictor implementation is acceptable per HARD-EARNED Hafner Revision #3 binding."

    - assumption: "Z7-Mamba-2 + Z7-LSTM trainer `_full_main` raising NotImplementedError per Catalog #240 is the canonical scaffold state (NOT a defect)"
      classification: HARD-EARNED
      rationale: "Per Catalog #240 + Catalog #325 + parent Z7-LSTM symposium Revision #7 binding: substrate trainers in PRE-BUILD / PRE-OPTIMAL-FORM state MUST raise NotImplementedError from _full_main with research_only=true recipe opt-out until Wave N+1 council convenes PROCEED-unconditional. Both Z7-Mamba-2 (test_trainer_full_main_raises_notimplementederror_per_catalog_240 PASSES) and Z7-LSTM (test_z7_gru_full_main_writes_byte_closed_prebuild_export PASSES — _full_main writes byte-closed pre-build export but does not run full training) satisfy this discipline."

    - assumption: "The recurrent-state predictive coding paradigm WILL provide MEANINGFUL bit-savings on dashcam 600-pair temporal coherence at contest scorer relevant subspace (Wave-N+1 paradigm question)"
      classification: CARGO-CULTED-PENDING-EMPIRICAL
      rationale: "Per Assumption-Adversary verbatim above + Quantizr verbatim above: this IS the PARADIGM-level question per Catalog #307. PR106 format0d_latent_score_table (stateless decoder) at 0.20533 [contest-CUDA] + PR101 frame_exploit_selector_fec6 (stateless) at 0.19205 [contest-CPU] are HARD-EARNED counter-evidence to recurrent-state-being-the-winning-pattern. Z7-Mamba-2 + Z7-LSTM + Z8 hierarchical = 3 alternative-probe-methodologies per Catalog #308 testing the SAME paradigm question. The disambiguator probe `tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` (5 tests pass) IS the apparatus that arbitrates this empirically when paired exact-eval lands. Per CLAUDE.md 'Forbidden premature KILL': IF all 3 alternative-probe-methodologies fail to beat PR106 format0d, advance to NeRV-family stateless predictive coding OR foveation IDEAS without recurrent state per Catalog #308 reactivation paths, NOT killed."

    - assumption: "Z6 4c trained paradigm-question outcome (NOT zero-epoch) is the canonical sequencing dependency for BOTH Z7-Mamba-2 + Z7-LSTM dispatch authorization"
      classification: HARD-EARNED-WITH-AMENDMENT
      rationale: "Per Z7-LSTM symposium Revision #5 binding + Contrarian verbatim above: Z6 4c TRAINED Candidate 4c paired exact-eval IS the canonical sequencing dependency (NOT just the zero-epoch packet). The 2026-05-18 paired zero-epoch packet (paired delta 0.0028 BELOW threshold) CLEARED the SUPERSEDED dependency tag `z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome` per codex memo `z7_z6_4c_dependency_supersession_20260518_codex.md` line 30-44 — BUT the codex supersession is at the READINESS-SURFACE level (queue artifact), NOT at the paradigm-question level. AMENDMENT: Z7 dispatch authorization SHOULD wait for TRAINED Candidate 4c paired exact-eval; the zero-epoch supersession is a queue-routing artifact only. Wave N+1 council MUST re-evaluate this dependency when TRAINED Candidate 4c paired exact-eval lands."

    - assumption: "Z7-Mamba-2 + Z7-LSTM unified symposium is sufficient T3 grand-council-attendee deliberation per Catalog #325 6-step contract"
      classification: HARD-EARNED
      rationale: "T3 attendees: 6-of-6 sextet (Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary) + 10 grand council attendees (Rao + Ballard + Tishby_memorial + Zaslavsky + Hafner + Wyner + Atick + Redlich + Quantizr + Schmidhuber). T3 quorum threshold per CLAUDE.md 'Council hierarchy: 4-tier protocol' = 5-of-6 sextet + ≥12-of-20 grand council = exceeded. Catalog #325 6-step contract satisfied via this memo (Sections 2-7 below). Binding."

  council_decisions_recorded:
    - "VERDICT: PROCEED_WITH_REVISIONS — Z7-Mamba-2 PRIMARY + Z7-LSTM FALLBACK unified design IS authorized at THIS verdict per Catalog #325 6-step contract + per-substrate symposium discipline. NEITHER Z7-Mamba-2 NOR Z7-LSTM paid dispatch is pre-authorized from this verdict. Per Contrarian + Assumption-Adversary VETO + Race-mode-rigor-inversion Rule 3: BOTH paid dispatches require Wave N+1 council convened AFTER a TRAINED Candidate 4c paired exact-eval lands (not the 2026-05-18 zero-epoch packet) AND ratifies PROCEED-unconditional per Catalog #315 OR equivalent."

    - "Revision #1 (binding per Hafner): Dispatch cascade per Race-mode-rigor-inversion Rule 3 = (a) Z7-LSTM/GRU FALLBACK FIRST (~$5-7 envelope; no mamba_ssm install required; sister to Z6-v2 Wave 2 Candidate 1) → (b) Z7-Mamba-2 PRIMARY SECOND (~$20-30 envelope; mamba_ssm install required on Modal A100) → (c) Z8 hierarchical FOURTH (~$42 envelope per Z6/Z7/Z8 scoping memo §3 Z8). Cheap signal gates expensive signal per Catalog #315 OPTIMAL-FORM iteration discipline."

    - "Revision #2 (binding per Quantizr + Z7-LSTM symposium Revision #2 inheritance): Wave N+1 Z7-LSTM/GRU FALLBACK dispatch MUST include Z7-vs-PR106 format0d paired-comparison disambiguator at SAME archive bytes per Z6 Phase 3 Revision #2 canonical pattern. Decision criterion: Z7-WIN at ΔS ≥ 0.005 contest-CUDA → recurrent-state-primitive empirically validated; Z7-LOSS or |ΔS| < 0.005 → advance to Z7-Mamba-2 (cascade step b) OR Z8 hierarchical (cascade step c) per Catalog #308 alternative-probe-methodologies."

    - "Revision #3 (binding per Atick + Wyner + Z7-LSTM symposium Revision #4 inheritance): BOTH Z7-Mamba-2 + Z7-LSTM trainers MUST support runtime-configurable ego-source via --ego-source flag (PoseNet-projection baseline OR scorer-logit-conditioning). The runtime support is VERIFIED at this memo via passing tests (test_runtime_configurable_ego_source_posenet_projection_baseline + test_runtime_configurable_ego_source_scorer_logit_compressed for Z7-Mamba-2; sister tests for Z7-LSTM). The empirical winning channel from TRAINED Candidate 4c paired exact-eval (Wave N+1 prerequisite) selects the canonical ego-source at dispatch time."

    - "Revision #4 (binding per Tishby_memorial + Zaslavsky + Z7-LSTM symposium Revision #5 inheritance): BOTH Z7-Mamba-2 + Z7-LSTM β-IB-Lagrangian parameter MUST initialize from C6 IBPS Phase 2 empirical β-optimal anchor (NOT guessed independently). C6 IBPS Phase 2 symposium verdict is PROCEED_WITH_REVISIONS (per `council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518.md`) with parallel β_ib + latent_dim sweep authorized under $5 envelope cap. Until C6 empirical β-optimal lands, BOTH Z7 trainers default to literature-canonical β=0.5 per Tishby-Zaslavsky 2015 deep-learning IB studies."

    - "Revision #5 (binding per Wyner + Ballard + Z7-LSTM symposium Wyner-Ziv inheritance): BOTH Z7-Mamba-2 + Z7-LSTM hidden states ARE implicit Wyner-Ziv side-info channels. The deterministic unroll regenerates identically at inflate-time; encoder + decoder share the implicit channel WITHOUT shipping it. This IS canonical Wyner-Ziv 1976 source-coding pattern. The architectural difference (Mamba-2 selective state-space vs GRU discrete-gating) captures DIFFERENT subspaces of temporal coherence; the per-substrate disambiguator probe `tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` arbitrates empirically at Wave N+1."

    - "Revision #6 (binding per Schmidhuber MDL-discipline): Wave N+1 dispatch MUST include per-substrate L(model) + L(data|model) decomposition for apples-to-apples comparison per MDL principle. The L(model) sizes are: Z7-Mamba-2 ~155-175K params (Mamba-2 block + projections + ego MLP + encoder/decoder reused from Z6-v1) ~ ~110-140KB archive; Z7-LSTM/GRU ~210-240K params ~ ~145-180KB archive; PR106 format0d unknown but likely <100K params (counter-evidence anchor). Per Catalog #324 post-training Tier-C validation: ALL Z7 archive byte-counts MUST be validated against measured Tier-C density on the TRAINED archive (NOT predicted from random-init)."

    - "Revision #7 (binding per Catalog #298 + #313 staleness window discipline + Z7-LSTM symposium Revision #7 inheritance): BOTH Z7-Mamba-2 + Z7-LSTM lane registry entries MUST declare research_only=true at L1 SCAFFOLD landing (at THIS memo; verified via lane_top5_2_z7_mamba2_scaffold_design_20260518 + lane_per_substrate_symposium_z7_lstm_predictive_coding_20260517 both at L0). dispatch_enabled remains false until Wave N+1 PROCEED-unconditional verdict. Per Catalog #240: recipes MUST NOT carry `dispatch_enabled: true` without `_full_main` implemented AND a recent symposium PROCEED-unconditional anchor. Z7-Mamba-2 recipe at `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml` already declares research_only=true + dispatch_enabled=false; Z7-LSTM recipe at `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch.yaml` declares same per sister codex landing."

    - "Per Catalog #325 per-substrate-symposium-evidence requirement: this memo SATISFIES the Catalog #325 acceptance for BOTH `substrate=time_traveler_l5_z7_mamba2` AND `substrate=time_traveler_l5_z7_lstm_predictive_coding` for the next 14 days. Substrate-aliases: `z7_mamba2` + `z7_lstm_predictive_coding` per Catalog #315 substrate_aliases mechanism."

    - "Frontier citation per Catalog #316: current canonical best is 0.19205 [contest-CPU] (lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`; archive sha 6bae0201) / 0.20533 [contest-CUDA] (lane `pr106_format0d_latent_score_table`; archive sha 9cb989cef519). Z7-Mamba-2 predicted band [0.167, 0.184] + Z7-LSTM predicted band [0.180, 0.192] per research wave §0 TOP-5 #2/#TT5L sit BELOW or AT current frontier IF realized empirically. Predicted vs realized gap is the canonical empirical question for Wave N+1 dispatch."

    - "Per Catalog #300 mission-alignment + HORIZON-CLASS Consequence 5: operator-frontier-override NOT INVOKED for this deliberation; standard sextet-pact + grand council T3 procedure applies; mission-contribution `frontier_breaking` (BOTH substrates open class-shift paths predicted to lower score below 0.20 [contest-CPU] IF temporal-coherence-primitive empirically validates). NOT `frontier_protecting` because no regression prevented. NOT `rigor_overhead` because deliberation IS substrate-specific design + cross-pollination wiring + integration audit, not gate/helper hygiene."

    - "Per CLAUDE.md 'Forbidden premature KILL': BOTH Z7-Mamba-2 + Z7-LSTM are in PRE-BUILD-OPTIMAL-FORM state (NOT yet trained; scaffold + design only at this verdict). Per CLAUDE.md 'Substrate MUST be at OPTIMAL FORM before paid empirical dispatch' + Catalog #315: BOTH are PRE-OPTIMAL-FORM (cargo-cult-unwind methodology must be APPLIED via Wave N+1 council approval before any iteration anchor). 30-day deferred-substrate retrospective due 2026-06-17T00:00:00Z for BOTH."

    - "Predicted cost path (Wave-N+1-conditional cascade): THIS memo lands at $0 GPU + ~3h editor. Wave N+1 council convened on TRAINED Z6 4c outcome + C6 IBPS Phase 2 outcome: $0 GPU + ~90 min editor. Z7-LSTM/GRU FALLBACK Wave 2 smoke + identity-disambiguator paired CPU: $5-7 envelope. Z7-LSTM/GRU FALLBACK Wave 3 full dispatch (CONDITIONAL on Wave 2 smoke PROCEED): $15-20 Modal A100 + $1.50 paired CPU/CUDA = $16.50-21.50 envelope. Z7-Mamba-2 PRIMARY Wave 2 smoke (CONDITIONAL on Z7-GRU FALLBACK outcome): $5-10 envelope. Z7-Mamba-2 PRIMARY Wave 3 full dispatch (CONDITIONAL on Wave 2 smoke PROCEED): $20-30 Modal A100 + paired CPU/CUDA ~$1.50 = $22-30 envelope. Z8 Wave 2+ (CONDITIONAL on Z7 outcomes): $42 envelope per Z6/Z7/Z8 scoping memo. TOTAL Z7 dispatch cascade (if all phases authorized): $43-58 envelope for Mamba-2+LSTM combined."

  council_predicted_mission_contribution: frontier_breaking
  council_override_invoked: false
  council_override_rationale: ""
  horizon_class: asymptotic_pursuit
  canonical_frontier_anchor:
    contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
    contest_cuda: "0.20533 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
  deferred_substrate_id: time_traveler_l5_z7_mamba2_plus_lstm_unified
  substrate_aliases:
    - time_traveler_l5_z7_mamba2
    - time_traveler_l5_z7_lstm_predictive_coding
    - z7_mamba2
    - z7_lstm_predictive_coding
  deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
  predicted_dispatch_risk: 0
  originSessionId: lane_z7_mamba2_lstm_full_landing_integration_audit_20260518
  related_deliberation_ids:
    - council_per_substrate_symposium_z7_lstm_predictive_coding_20260517
    - z7_mamba2_substrate_design_memo_20260518
    - z6_candidate4c_full_disambiguator_probe_20260518_codex
    - z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex
    - z6_candidate4c_full600_zeroepoch_handoff_20260518_codex
    - z7_z6_4c_dependency_supersession_20260518_codex
    - council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518
    - council_per_substrate_symposium_atw_v2_reactivation_20260518
    - comprehensive_research_wave_20260518
    - time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516
    - feedback_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_landed_20260518
---

# PER-SUBSTRATE SYMPOSIUM — Z7-Mamba-2 PRIMARY + Z7-LSTM FALLBACK UNIFIED (T3 grand-council) 2026-05-18

**Lane**: `lane_z7_mamba2_lstm_full_landing_integration_audit_20260518` (L0 → L1 at memo landing)
**Catalog #325 satisfied** for `substrate=time_traveler_l5_z7_mamba2` + `substrate=time_traveler_l5_z7_lstm_predictive_coding` (14-day window from 2026-05-18).
**Operator directive** (2026-05-18 verbatim): "specifically relaunch and recover the z7-Mamba-2 subagent and sister subagent and everything".
**$0 GPU, ~2h editor, NO COMMITS per parent prompt directive, NO Modal/Lightning/Vast.ai dispatches.**

## TL;DR (60 seconds)

The rate-limit-truncated Z7-Mamba-2 substantively landed ~1700 lines / 102 KB of scaffold (5 deliverables; ALL 36 dedicated tests PASS); the sister Z7-LSTM/GRU FALLBACK substrate package + trainer + driver + tests landed via codex sister wave (ALL 16 + 11 dedicated tests PASS). This T3 unified sextet-pact + grand-council deliberation (Hafner + Atick + Wyner + Tishby_memorial + Zaslavsky + Rao + Ballard + Quantizr + Schmidhuber + Redlich added per topic) classifies the dual-primitive recurrent-predictive-coding substrate class against the 5-gate rigor lens (Catalog #307 / #308 / #324 / #325 / #313) jointly, enumerates the canonical dispatch cascade (Z7-LSTM/GRU FALLBACK FIRST per cheap-signal-first; Z7-Mamba-2 PRIMARY SECOND; Z8 hierarchical FOURTH), and binds the bidirectional cross-pollination matrix with Z6 4c outcome + C6 IBPS Phase 2 outcome + ATW V2-1 channel-pick.

**VERDICT: PROCEED_WITH_REVISIONS** (7 binding revisions). Contrarian + Assumption-Adversary VETO on dispatch funding pre-authorization for EITHER substrate from this verdict; PROCEED on design completion + integration audit + Wave-N+1-pre-cached deliberation. NEITHER paid dispatch fires without (a) a TRAINED Z6 4c paired exact-eval landing (NOT the 2026-05-18 zero-epoch packet) AND Wave N+1 council convening PROCEED-unconditional, OR (b) operator explicit-frontier-override per Catalog #300 with verbatim quote in `council_override_rationale`.

**Predicted ΔS bands**:
- Z7-Mamba-2 PRIMARY: [-0.025, -0.008] over PR101 frontier ⇒ [0.167, 0.184] [contest-CPU] per research wave §0 TOP-5 #2
- Z7-LSTM/GRU FALLBACK: [-0.012, 0.000] over PR101 frontier ⇒ [0.180, 0.192] [contest-CPU] (cheaper sister; less expressive than Mamba-2 selective state-space)
- BOTH sit BELOW current canonical CPU frontier 0.19205 IF realized empirically — asymptotic_pursuit horizon_class per Catalog #309

**Empirical anchor today (2026-05-18) — critical for Z7 sequencing**:
- Z6 4c paired zero-epoch (`z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.md`): full FiLM IS lower than identity on BOTH CUDA (0.0028) + CPU (0.0029) axes, BUT margin BELOW decision_delta_s=0.005 → NEITHER WIN nor DEFER on paradigm; zero-epoch packet = control/anchor only.
- Per Contrarian verbatim: this does NOT supersede the Z7 paradigm-question dependency. Wave N+1 council MUST wait for TRAINED Candidate 4c paired exact-eval.
- C6 IBPS Phase 2 redesign symposium 2026-05-18: PROCEED_WITH_REVISIONS with parallel β_ib + latent_dim sweep authorized under $5 envelope cap (Contrarian dissent binding). C6 empirical β-optimal anchor pending; Z7 trainers default to literature-canonical β=0.5 until C6 empirical anchor lands.

## 1. Probe outcome re-examination per Catalog #307 + #308

### Z7-Mamba-2 status
**NO PRIOR Z7-Mamba-2 EMPIRICAL PROBE EXISTS**. Trainer scaffold per `experiments/train_substrate_time_traveler_l5_z7_mamba2.py` raises NotImplementedError from `_full_main` per Catalog #240. Verified via `test_trainer_full_main_raises_notimplementederror_per_catalog_240` PASS.

**Classification per Catalog #307**: NOT APPLICABLE — Z7-Mamba-2 has NO prior empirical anchor; paradigm-vs-implementation distinction applies only to falsified substrates.

### Z7-LSTM/GRU FALLBACK status
**NO PRIOR Z7-LSTM EMPIRICAL PROBE EXISTS**. Trainer + substrate package landed via codex sister wave today; `_full_main` writes byte-closed pre-build export but does NOT run full training per Catalog #240 + #325 + Z7-LSTM symposium Revision #7 binding. Verified via `test_z7_gru_full_main_writes_byte_closed_prebuild_export` PASS + recipe declares research_only=true + dispatch_enabled=false.

**Classification per Catalog #307**: NOT APPLICABLE — Z7-LSTM has NO prior empirical anchor.

### Sister probe evidence (cross-substrate)
- **Z6 4c paired zero-epoch (2026-05-18)** — paired delta 0.0028 BELOW decision_delta_s=0.005 on both axes; NEITHER WIN nor DEFER on paradigm; zero-epoch packet = control/anchor only per `z6_candidate4c_full600_zeroepoch_handoff_20260518_codex.md` line 263.
- **C6 IBPS Phase 2 redesign symposium (2026-05-18)** — PROCEED_WITH_REVISIONS with parallel β_ib + latent_dim sweep authorized; empirical β-optimal anchor pending.
- **ATW V2 reactivation symposium (2026-05-18)** — PROCEED_WITH_REVISIONS with V2-1 redesign + re-probe authorized; channel-pick from V2-1 outcome inheritance into Z7 ego-source per Atick verbatim cross-reference.
- **TT5L #866 (2026-05-18 Wave-1 sister symposium)** — 25ep CUDA 3.9007 ALL-ZERO side-info empirical confirmation of operator's "fundamentally broken janky" hypothesis; sister evidence that recurrent-state-as-implicit-side-info MAY NOT survive at contest-CUDA without explicit byte-closed proof.

### Per Catalog #298 + #313 30-day staleness window
NOT APPLICABLE — neither Z7-Mamba-2 nor Z7-LSTM/GRU has prior probe outcome to expire.

## 2. Cargo-cult audit per assumption (Catalog #303)

### Z7-Mamba-2 PRIMARY cargo-culds (per parent design memo §2)
10 assumptions enumerated; per parent design memo `.omx/research/z7_mamba2_substrate_design_memo_20260518.md` §2:
| # | Assumption | Classification |
|---|---|---|
| CC-1 | Mamba-2 5-10× speedup transfers to dashcam 600-pair sequence | CARGO-CULTED-PENDING-EMPIRICAL |
| CC-2 | Mamba-2 > GRU expressive power at hidden_dim=128 on 24-dim latent | CARGO-CULTED-PENDING-EMPIRICAL |
| CC-3 | Mamba-2 selective state-space matches ego-motion-continuity prior better | CARGO-CULTED-PENDING-PRINCIPLED |
| CC-4 | mamba_ssm PyPI installs in pact Modal A100 image | HARD-EARNED-PARTIAL |
| CC-5 | Z7-Mamba-2 ego-source inherits from Z6 Wave 2 4c outcome | HARD-EARNED |
| CC-6 | Z7-Mamba-2 dispatches INDEPENDENTLY from Z7-GRU Wave 2 | CARGO-CULTED |
| CC-7 | Mamba-2 hidden state IS implicit Wyner-Ziv side-info channel | HARD-EARNED |
| CC-8 | Z7-Mamba-2 β-IB-Lagrangian inherits from C6 Phase 2 empirical | CARGO-CULTED-PENDING-C6-PHASE-2 |
| CC-9 | Mamba-2 d_state=16 is the right state dimension | CARGO-CULTED |
| CC-10 | Z7-Mamba-2 belongs in asymptotic_pursuit horizon_class | HARD-EARNED |

### Z7-LSTM/GRU FALLBACK cargo-cults (NEW per this unified symposium)
12 assumptions enumerated:
| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| CC-L1 | GRU canonical recurrent primitive per Hafner DreamerV3 lineage | HARD-EARNED | Per Z7-LSTM symposium Revision #3 binding (2026-05-17). |
| CC-L2 | GRU hidden state at hidden_dim=128 sufficient for 600-pair temporal coherence | CARGO-CULTED-PENDING-EMPIRICAL | Wave 2 smoke + identity-disambiguator probe at SAME archive bytes per Z6 Phase 3 Revision #2 canonical pattern. |
| CC-L3 | GRU recurrent state IS implicit Wyner-Ziv side-info channel | HARD-EARNED | Per Wyner verbatim (THIS symposium) + Ballard 2026-05-17 verbatim. Deterministic unroll regenerates identically at inflate-time. |
| CC-L4 | GRU sequential CPU forward is ~2× slower than Z6 FiLM feedforward | HARD-EARNED-PARTIAL | Per Z7-LSTM symposium cost estimate $5-7 envelope; verified via MPS proxy timing (when run). |
| CC-L5 | GRU recurrent primitive ego-source inherits from Z6 Wave 2 4c outcome | HARD-EARNED | Per Z7-LSTM symposium Revision #4 binding + Catalog #311. |
| CC-L6 | GRU dispatches BEFORE Z7-Mamba-2 (cheap-signal-first cascade) | HARD-EARNED | Per Hafner Revision (THIS symposium) + Race-mode-rigor-inversion Rule 3 + Catalog #315. |
| CC-L7 | GRU β-IB-Lagrangian inherits from C6 Phase 2 empirical | CARGO-CULTED-PENDING-C6-PHASE-2 | Per Z7-LSTM symposium Revision #5 binding; C6 empirical β-optimal pending. |
| CC-L8 | GRU + Catalog #311 ego-motion-conditioning satisfied via --ego-source flag | HARD-EARNED | Verified via passing tests `test_z7_gru_predictor_gradients_reach_latent_and_ego_inputs` + sister Z7-Mamba-2 runtime-configurable-ego-source tests. |
| CC-L9 | Z7-LSTM codename + GruRecurrentPredictor implementation acceptable | HARD-EARNED | Per Hafner Revision #3 binding (Z7-LSTM symposium 2026-05-17): codename retained for backward compatibility; implementation IS GRU. |
| CC-L10 | Z7-LSTM/GRU L(model) ~210-240K params < Z6-v2 Multi-layer FiLM ~300K | HARD-EARNED | Per Z7-LSTM symposium parameter count breakdown. |
| CC-L11 | Z7-LSTM/GRU dispatches IF AND ONLY IF Wave N+1 council convenes PROCEED-unconditional | HARD-EARNED | Per Catalog #240 + #325 + Z7-LSTM symposium Revision #7 + THIS Revision #7 binding. |
| CC-L12 | Z7-LSTM/GRU belongs in asymptotic_pursuit horizon_class | HARD-EARNED | Per Z7-LSTM symposium predicted band [0.10, 0.13] (alternate predicted band [0.180, 0.192] per research wave is more conservative). |

**Unified cargo-cult-class summary**: 8 HARD-EARNED + 3 HARD-EARNED-PARTIAL + 5 CARGO-CULTED + 5 CARGO-CULTED-PENDING-EMPIRICAL + 1 CARGO-CULTED-PENDING-PRINCIPLED + 2 CARGO-CULTED-PENDING-C6-PHASE-2.

All disambiguators: (a) MPS proxy training (CC-1, CC-2 cheapest); (b) Z7-GRU Wave 2 outcome (CC-6, CC-L2); (c) Z7-Mamba-2 Wave 2 paired vs GRU (CC-2, CC-3, CC-9); (d) Z6 4c TRAINED outcome (CC-5, CC-L5); (e) C6 Phase 2 empirical anchor (CC-8, CC-L7); (f) per-substrate disambiguator probe `tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` at Wave N+1.

## 3. 9-dimension success checklist evidence per Catalog #294

Unified per BOTH substrates (per parent Z7-Mamba-2 design memo §3 + Z7-LSTM symposium §3):

| # | Dimension | Z7-Mamba-2 evidence | Z7-LSTM/GRU evidence |
|---|---|---|---|
| 1 | UNIQUENESS (class-shift not within-class) | ✓ CONDITIONAL | ✓ CONDITIONAL |
| 2 | BEAUTY + ELEGANCE | ✓ Mamba2Predictor ~150 LOC matching Z6 sister | ✓ GruRecurrentPredictor matching Z6 sister |
| 3 | DISTINCTNESS | ✓ Only substrate binding selective-state-space + Wyner-Ziv | ✓ Only substrate binding GRU + Wyner-Ziv + DreamerV3 lineage |
| 4 | RIGOR | ✓ Cargo-cult + 9-dim + observability + #313 + cross-pollination | ✓ Same coverage per parent symposium |
| 5 | OPTIMIZATION PER TECHNIQUE | ✓ 5 FORK + 1 UNCLEAR + 7 ADOPT (per parent §6) | ✓ Per Z7-LSTM symposium §9 canonical-vs-unique decision table |
| 6 | STACK-OF-STACKS-COMPOSABILITY | ✓ Z7-Mamba-2 + NSCS06v8 + DP1 + D1 (deferred to Wave 3+) | ✓ Same composition vector |
| 7 | DETERMINISTIC REPRODUCIBILITY | ✓ Z7MCM2 archive byte-stable + Mamba-2 deterministic unroll | ✓ Z7PCWM1 archive byte-stable + GRU deterministic unroll |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | ✓ Mamba-2 GPU forward sister to GRU at seq-len 600; 5-10× CARGO-CULTED-PENDING | ✓ GRU sequential CPU ~2× slower than Z6 FiLM (Z7-LSTM symposium estimate) |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | INDETERMINATE — predicted [0.167, 0.184] BELOW frontier IF realized | INDETERMINATE — predicted [0.180, 0.192] BELOW frontier IF realized |

## 4. Observability surface declaration per Catalog #305

**6 facets per parent Z7-Mamba-2 design memo §4 + sister Z7-LSTM symposium §4** — both substrates satisfy all 6 facets.

### Z7-Mamba-2-specific observability hooks
- Per-pair Mamba-2 selectivity matrix range tracking (A_t, B_t, C_t input-conditioned matrices); state-space dimension utilization heatmap.
- Predictor residual magnitude per pair t.
- Hidden state norm tracking across 600 pairs.

### Z7-LSTM/GRU-specific observability hooks
- Per-pair GRU update gate + reset gate range tracking.
- Predictor residual magnitude per pair t (sister to Mamba-2).
- Hidden state norm tracking across 600 pairs (sister to Mamba-2).

### Catalog #311 ego-motion-conditioning declaration
**APPLICABLE + SATISFIED for BOTH substrates** — both invoke Atick-Redlich cooperative-receiver framing via hidden-state-IS-encoder-decoder-shared-prior-channel pattern (per Wyner + Ballard verbatim).

### Catalog #312 hierarchical predictive coding declaration
**NOT APPLICABLE for EITHER substrate** — both are SINGLE-LEVEL recurrent (Mamba-2 selective state OR GRU hidden state); neither claims full hierarchical predictive coding (Z8 territory per Z6/Z7/Z8 scoping memo §3 Z8). Catalog #312 gate's scope correctly excludes both Z7 substrates.

### Catalog #310 F-asymptote PRIMARY substrate declaration
**APPLICABLE + SATISFIED for BOTH substrates** — both claim asymptotic_pursuit horizon_class; both are PRIMARY substrates (ship encoder + decoder + recurrent predictor + per-pair residual stream + monolithic single-file archive grammar). Substrate-class-shift token = `scorer_relationship_class_shift_predictive_coding_world_model_v2_z7_mamba2_OR_z7_gru` per Hafner DreamerV3 deterministic-recurrence lineage.

## 5. Per-substrate-symposium contract per Catalog #325

The 6-step contract for unified symposium:

1. **Cargo-cult audit per Catalog #303** ✓ — §2 above (22 assumptions total across both substrates).
2. **9-dimension success checklist evidence per Catalog #294** ✓ — §3 above.
3. **Observability surface declaration per Catalog #305** ✓ — §4 above (6 facets + Catalog #310/#311/#312 sub-gates).
4. **Sextet pact deliberation** ✓ — T3 grand-council (6-of-6 sextet + 10 grand council attendees: Hafner + Atick + Wyner + Tishby_memorial + Zaslavsky + Rao + Ballard + Quantizr + Schmidhuber + Redlich). Quorum threshold 5-of-6 sextet + ≥12-of-20 grand council = exceeded.
5. **Per-substrate reactivation criteria pinned** ✓ — §7 below enumerates 3 reactivation paths IF Z7-Mamba-2 OR Z7-LSTM Wave N+1 disambiguator LOSES.
6. **Catalog #324 post-training Tier-C validation discipline** — `predicted_band_validation_status: pending_post_training` per BOTH recipes; reactivation criterion = post-training Tier-C density measurement on Z7-Mamba-2 archive AND Z7-LSTM/GRU archive after Wave 2 smoke completes.

## 6. Canonical-vs-unique decision per layer

Per parent Z7-Mamba-2 design memo §6 (13 layers) + sister Z7-LSTM symposium §9 (analogous 13 layers). Net per substrate: 5 FORK_BECAUSE_PRINCIPLED + 1 UNCLEAR_NEEDS_EMPIRICAL + 7 ADOPT_CANONICAL.

**Critical FORK layers** (substrate-distinguishing):
- Predictor primitive (Mamba-2 OR GRU)
- Archive grammar (Z7MCM2 OR Z7PCWM1)
- Inflate runtime (Mamba-2 deterministic unroll OR GRU deterministic unroll)
- Ego-source projection (runtime-configurable per Catalog #311)
- β-IB Lagrangian (inherits from C6 Phase 2 empirical anchor; literature-canonical β=0.5 default until C6 lands)

**Shared ADOPT layers** (cross-substrate canonical):
- Encoder + Decoder (Z6-v1 sister pattern; reused via tac.substrates.time_traveler_l5_z6.architecture._Z6Encoder + _Z6Decoder)
- Score-aware loss helper (tac.substrates._shared.score_aware_common.score_pair_components per Catalog #164)
- Training curriculum (pyav decode + patched YUV6 + differentiable scorers + EMA(0.997) + eval_roundtrip=True + AdamW + cosine schedule)
- Tier-1 engineering (autocast_fp16 #172 + TF32 #178 + torch.compile #179 + no_grad-at-eval #180 + GTScorerCache F3 #228)
- Scorer loader assignment order (pose_scorer, seg_scorer = load_differentiable_scorers per Catalog #222)
- Deterministic reproducibility (seed-pinned per trainer_skeleton.device_or_die + detect_hardware_substrate #190)
- Observability surface (per-epoch loss + per-pair recurrent state hooks)

## 7. Dispatch sequencing + reactivation paths

### Canonical cascade (Path 1 default per Hafner Revision + Race-mode-rigor-inversion Rule 3)

```
[CURRENT STATE: Z7-Mamba-2 + Z7-LSTM/GRU pre-build; Z6 4c paired zero-epoch (NOT trained) lands BELOW decision threshold;
                C6 IBPS Phase 2 redesign PROCEED_WITH_REVISIONS (parallel β_ib + latent_dim sweep authorized);
                ATW V2-1 redesign PROCEED_WITH_REVISIONS (channel-pick pending)]
   ↓ AWAITS (TRAINED Z6 4c paired exact-eval lands per Wave N+1 prerequisite)
[Wave N+1 council convened on TRAINED Z6 4c outcome + C6 IBPS Phase 2 outcome + ATW V2-1 channel-pick ($0 + ~90 min)]
   ↓ IF Wave N+1 PROCEED-unconditional:
      [Z7-LSTM/GRU FALLBACK Wave 2 smoke + identity-disambiguator paired CPU ($5-7 envelope)]
        ↓ IF Z7-GRU-WIN AT ΔS ≥ 0.005 vs PR106 format0d (paired-comparison at SAME archive bytes per Z6 Phase 3 Revision #2):
           [Z7-LSTM/GRU FALLBACK Wave 3 full dispatch ($16.50-21.50 envelope) → AUTOPILOT consumes outcome]
             ↓ Z7-GRU outcome materially informs Z7-Mamba-2 dispatch priority
        ↓ IF Z7-GRU-LOSS or |ΔS| < 0.005:
           [Wave N+2 council on Z7-Mamba-2 PRIMARY ratification ($0 + ~90 min)]
             ↓ IF PROCEED-unconditional:
                [Z7-Mamba-2 PRIMARY trainer BUILD (~1 week subagent + $0 GPU; INCLUDES mamba_ssm install + Modal image extension)]
                  ↓
                [Z7-Mamba-2 PRIMARY Wave 2 smoke + identity-disambiguator paired ($5-10)]
                  ↓ IF Z7-Mamba-2-WIN AT ΔS ≥ 0.005:
                     [Wave N+3 council → Z7-Mamba-2 PRIMARY Wave 3 full dispatch ($22-30)]
                  ↓ IF Z7-Mamba-2-LOSS:
                     [Pivot to Z8 hierarchical ($42) OR Z7-RWKV-7 ($20-25) OR DEFER per Catalog #298]

OR PATH 2 (operator explicit-frontier-override per Catalog #300):
[Operator declares Z7-Mamba-2 PRIMARY-first (skipping Z7-GRU FALLBACK) with verbatim quote in council_override_rationale]
   ↓
[Z7-Mamba-2 PRIMARY trainer BUILD (~1 week + $0 GPU)]
   ↓
[Z7-Mamba-2 Wave 2 smoke ($5-10) → identity-disambiguator → Wave 3 full ($22-30)]
```

### Reactivation paths (per Quantizr Revision #6 + Catalog #308 N>=3)

IF BOTH Z7-Mamba-2 + Z7-LSTM/GRU FALLBACK LOSE Wave 2 disambiguator vs PR106 format0d:
- (a) **Z8 hierarchical** ($42 envelope per Z6/Z7/Z8 scoping memo §3 Z8) — full Rao-Ballard 3-level hierarchy + Hafner DreamerV3 stochastic
- (b) **Z7-RWKV-7** ($20-25 envelope per Z7-Mamba-2 design memo §10) — alternative linear-attention RNN class
- (c) **NeRV-family stateless predictive coding** — per Quantizr verbatim above; PR95 hnerv_lc_v2 is frame-independent NeRV pattern; reactivation reframes predictive-coding via stateless decoder
- (d) **DEFER predictive-coding-recurrent paradigm** to research_only per Catalog #298 retirement discipline + CLAUDE.md "Forbidden premature KILL"

Each pivot = its OWN per-substrate symposium.

## 8. Cross-pollination wiring (Revision #5 + research wave §3.6 binding)

**BOTH Z7-Mamba-2 + Z7-LSTM/GRU MUST document explicit dependency on**:
1. **TRAINED Z6 Candidate 4c paired exact-eval outcome** (NOT the 2026-05-18 zero-epoch packet) — materially changes Z7 ego-source choice per Revision #3 inheritance.
2. **C6 IBPS Phase 2 redesign outcome** — canonical IB-framework empirical anchor; C6's empirically-optimal β-IB-Lagrangian parameter MUST initialize Z7's β-parameter (NOT guessed independently).
3. **Z7-LSTM/GRU FALLBACK Wave 2 outcome** (sister Wave N+1 pending) — Path 1 default sequences Z7-Mamba-2 AFTER Z7-GRU per Catalog #315 OPTIMAL-FORM iteration discipline.
4. **ATW V2-1 channel-pick outcome** — V2-1 redesign chooses among 3 channels (scorer-softmax-sketch per Atick ranking #1 / per-region SegNet softmax histograms per Atick ranking #2 / pose-bin discretization per Atick ranking #3). Z7's runtime-configurable ego-source could inherit the V2-1 winning channel via the canonical Wyner-Ziv side-info pattern.

## 9. Mission alignment per Catalog #300

`council_predicted_mission_contribution: frontier_breaking` — BOTH Z7-Mamba-2 + Z7-LSTM/GRU open class-shift paths predicted to lower score below 0.20 [contest-CPU] IF temporal-coherence-primitive empirically validates at Wave N+1 disambiguator.

NOT `frontier_protecting` (no regression prevented).
NOT `rigor_overhead` (substrate-specific design + integration audit + cross-pollination, not gate/helper hygiene).
NOT `apparatus_maintenance` (substrate work, not infrastructure update).
NOT `mission_questioned` (per-substrate symposium discipline IS canonical per Catalog #325).

`rigor_overhead` fraction: ~25% (Catalog #292 + #303 + #305 + #294 + #316 + #325 evidence). Remaining 75% substrate-specific design + cross-pollination + cargo-cult audit + integration audit IS frontier-pursuit content.

## 10. Op-routables (ranked by EV per cheap-signal-first)

1. **Land MPS proxy training pattern via Z7-Mamba-2 reference impl** (operator routes via parent design memo §13 LOCAL M5 MAX PROXY) — $0 GPU; ~3h editor; produces curve-shape evidence + CC-1/CC-2 disambiguation BEFORE paid dispatch.
2. **Wait for TRAINED Z6 Candidate 4c paired exact-eval** — Wave N+1 prerequisite for Z7 dispatch authorization. Operator-routable subagent for Z6 4c trainer build per Z6 Phase 3 Council Revision #1 cascade.
3. **Wait for C6 IBPS Phase 2 empirical β-optimal anchor** — β-IB-Lagrangian inheritance for BOTH Z7 trainers per Revision #4 binding.
4. **Wait for ATW V2-1 channel-pick outcome** — runtime-configurable ego-source candidate inheritance via Wyner-Ziv side-info pattern per Revision #5.
5. **Z7-LSTM/GRU FALLBACK Wave 2 smoke dispatch** — CONDITIONAL on #2/#3/#4; cheapest dispatch in cascade ($5-7).
6. **Z7-Mamba-2 PRIMARY Wave 2 smoke dispatch** — CONDITIONAL on #5 outcome; mamba_ssm install required on Modal image.

## 11. Continual-learning posterior anchor per Catalog #300

This memo emits a council anchor via `tac.council_continual_learning.append_council_anchor` (canonical helper per CLAUDE.md "Council hierarchy: 4-tier protocol"). Anchor surfaces:
- T3 grand-council deliberation
- PROCEED_WITH_REVISIONS verdict
- 7 binding revisions
- 8 assumption-adversary classifications (4 HARD-EARNED + 1 HARD-EARNED-WITH-AMENDMENT + 3 CARGO-CULTED or CARGO-CULTED-PENDING)
- substrate_aliases for both Z7 substrates

## 12. Premise verification per Catalog #229

8 premises verified pre-edit (NOT after-the-fact):
1. PV-1: Z7-Mamba-2 trainer scaffold exists at `experiments/train_substrate_time_traveler_l5_z7_mamba2.py` (~19.4K) — verified via `ls` ✓
2. PV-2: Z7-LSTM substrate package exists at `src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/` (5 modules) — verified via `ls` ✓
3. PV-3: Mamba2Predictor canonical helper exists at `src/tac/optimization/mamba2_predictor.py` (~22.9K) — verified via `ls` ✓
4. PV-4: Z7-Mamba-2 design memo exists at `.omx/research/z7_mamba2_substrate_design_memo_20260518.md` (~48.1K, 476 lines) — verified via `wc -l` + `cat` ✓
5. PV-5: Z7-LSTM symposium memo exists at `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md` (~89.8K, 647 lines) — verified via `wc -l` + `cat` ✓
6. PV-6: Z6 4c zero-epoch handoff memo exists + paired exact-eval result lines 251-285 — verified via `cat` ✓
7. PV-7: ALL 63 dedicated tests pass for Z7-Mamba-2 + Z7-LSTM + sister tests — verified via `pytest -v` ✓
8. PV-8: C6 IBPS Phase 2 symposium PROCEED_WITH_REVISIONS verdict — verified via `grep` ✓

## 13. Checkpoint trace per Catalog #206

Subagent checkpoints emitted via `tools/subagent_checkpoint.py`:
- Step 1 (initial): stage_1_test_verification next-action
- Step 2 (post-tests): stage_1_complete_63_tests_pass note
- Step 3 (context-gathering): stage_2_writing_symposium_memo next-action
- Step 4 (post-symposium-memo): stage_3_full_main_design next-action
- Step N (final): complete status

## 14. Cross-references

- **Parent Z7-Mamba-2 design memo**: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- **Parent Z7-LSTM symposium**: `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md`
- **Z6 4c zero-epoch handoff**: `.omx/research/z6_candidate4c_full600_zeroepoch_handoff_20260518_codex.md`
- **Z6 4c paired exact-eval result**: `.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.md`
- **Z7 dependency supersession**: `.omx/research/z7_z6_4c_dependency_supersession_20260518_codex.md`
- **C6 IBPS Phase 2 symposium**: `.omx/research/council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518.md`
- **ATW V2 reactivation symposium**: `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md`
- **Research wave deliverable**: `.omx/research/comprehensive_research_wave_20260518.md`
- **Z6/Z7/Z8 scoping memo**: `.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md`
- **Sister Z7-Mamba-2 _full_main design**: `.omx/research/z7_mamba2_full_main_design_20260518.md` (sister deliverable from this lane)
- **Sister Z7-LSTM/GRU _full_main design**: `.omx/research/z7_lstm_full_main_design_20260518.md` (sister deliverable from this lane)
- **Sister integration audit**: `.omx/research/z7_integration_audit_20260518.md` (sister deliverable from this lane)
- **Sister cross-pollination decision tree**: `.omx/research/z7_z6_4c_c6_ibps_atw_v2_1_cross_pollination_decision_tree_20260518.md` (sister deliverable from this lane)

## Observability surface

### Observability invariants

This memo IS the canonical T3 grand-council deliberation observability surface for the unified Z7-Mamba-2 + Z7-LSTM/GRU substrate class. The 6 facets per Catalog #305:

1. **Inspectable per layer** — symposium memo body documents all 7 revisions + 8 assumption-adversary verdicts + 16 council attendee verbatims.
2. **Decomposable per signal** — frontmatter `council_dissent` + `council_assumption_adversary_verdict` + `council_decisions_recorded` arrays decompose into per-member positions, per-assumption classifications, per-revision binding.
3. **Diff-able across runs** — symposium memo append-only per HISTORICAL_PROVENANCE Catalog #110/#113; future symposiums on same substrates cite this memo via `related_deliberation_ids`.
4. **Queryable post-hoc** — anchor appended to `.omx/state/council_deliberation_posterior.jsonl` via canonical helper `tac.council_continual_learning.append_council_anchor`.
5. **Cite-able** — `originSessionId` + `deferred_substrate_id` + `substrate_aliases` + `related_deliberation_ids` provide cite-chain.
6. **Counterfactual-able** — Wave N+1 trained Z6 4c paired exact-eval IS the canonical counterfactual ("what if Z6 4c trained-WIN?"). Sister Z7-LSTM/GRU FALLBACK Wave 2 smoke is sister counterfactual ("what if Z7-GRU-WIN vs PR106 format0d?").
