---
name: sextet-council-nscs03-phase-2-consensus-20260516
description: |
  T2 sextet-pact council CONSENSUS deliberation for NSCS03 end-to-end Ballé
  2018 joint codec substrate. Satisfies design memo + recipe reactivation
  criterion #4 (Phase 2 sextet-pact council CONSENSUS) following Phase 1 lift
  commit 2026-05-15 (`feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md`)
  and the operator-decision-items investigation 2026-05-15
  (`.omx/research/nscs03_operator_decision_items_investigation_20260515.md`).
  Sextet pact 6-of-6: Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich +
  Contrarian + Assumption-Adversary; advisory consultation: Ballé (Grand
  Council 2018 joint-codec author seat). Verdict: PROCEED_WITH_REVISIONS —
  recipe REMAINS research_only=true / dispatch_enabled=false; Catalog #167
  smoke-before-full at $1.50-4.00 Modal A100 (criterion #5) becomes the
  explicit next gate; 3 dispatch_blockers from the recipe (λ_R sweep / σ-floor
  / EMA differentiated 0.999/0.997 split) become 2 mandatory revisions in this
  consensus (λ_R + σ-floor co-sweep) and 1 PROCEDURAL FORK (differentiated
  EMA split implemented inline as substrate-engineering layer-7 unique +
  documented waiver). NSCS03 IS materially different from Z6 — has 76 passing
  tests + DEFINING gradient-reaches-all-5-subnets test + real-pair integration
  + 4 UNIQUE + 14 CANONICAL ADOPT + 3 DOCUMENTED FORK per UNIQUE-AND-COMPLETE-
  PER-METHOD; Z3-G1 risk (distinguishing bytes empty per Catalog #272) is
  addressable VIA the smoke gate's byte-mutation no-op detector (Catalog #139).
  Outside-NeRV architectural-class diversity preserved per operator binding
  constraint 2026-05-16.
substrate_id: nscs03_end_to_end_balle_joint_codec
deliberation_id: sextet_council_nscs03_phase_2_consensus_20260516
topic: "NSCS03 Phase 2 sextet-pact council CONSENSUS deliberation per recipe reactivation criterion #4"
council_tier: T2
council_attendees:
  - Shannon (LEAD)
  - Dykstra (CO-LEAD)
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Ballé (Grand Council advisory; 2018 joint-codec author seat)
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
council_dissent:
  - member: Contrarian
    verbatim: "I push back on any framing that treats 'Phase 1 trainer lift landed (76 tests pass)' as equivalent to 'Phase 2 sextet-pact council CONSENSUS approving recipe flip to dispatch_enabled=true at default config'. The trainer IS genuinely lifted — 76 tests pass including the DEFINING test that gradient reaches all 5 sub-networks (g_a, g_s, h_a, h_s, entropy_bottleneck_z). This is materially stronger than Z6's lift evidence (24 tests + identity-predictor probe collapse). HOWEVER: the recipe ITSELF encodes 3 explicit dispatch_blockers (`phase_2_council_approval_required_to_lift_full_main_NotImplementedError, lambda_R_sweep_calibration_pending_first_smoke_anchor, sigma_floor_sensitivity_pending_low_rate_op_point_validation`). The first is closed by THIS deliberation. The other TWO are uncalibrated CARGO-CULTED defaults (`λ_R=0.5` mid-range from Ballé2018 natural-image regime; `σ-floor=1e-4` is 25× tighter than Ballé2018 analytical lower bound `~3.8e-6`). Per CLAUDE.md 'long_burn' boundary classification (Catalog #239): NSCS03 full at $60-80 A100 crosses the $50 boundary; sweep-first at $8-22 Lightning T4 is 12-28% of full cost and is the canonical risk mitigation. Z3-G1's $0.59 smoke spend on EMPTY distinguishing-feature bytes is the empirical anchor for why we DO NOT skip calibration on Ballé-family substrates. Revisions #1 + #2 (mandatory): λ_R + σ-floor co-sweep at Lightning T4 BEFORE any A100 dispatch fires. Revision #3 (mandatory): the smoke MUST emit Catalog #272 distinguishing-feature byte-mutation proof (mutate one byte in `hyperprior_weights_int8` / `main_latents` / `entropy_state_sd`; verify inflate output changes). Revision #4 (procedural fork; not blocking): Ballé differentiated 0.999/0.997 EMA split implemented as Layer-7 UNIQUE substrate-engineering with `# EMA_DECAY_DIFFERENTIATED_OK:balle2018_recipe_canonical_for_joint_codec` waiver. If smoke-green + sweep-converged + Catalog #272 byte-mutation PASSES → flip recipe + dispatch FULL. Otherwise DEFER-pending-NSCS03-v2-redesign per CLAUDE.md 'Forbidden premature KILL'."
  - member: Assumption-Adversary
    verbatim: "I challenge the SHARED ASSUMPTION operating across this council deliberation: that the Phase 1 lift commits' evidence surface (76 tests pass + DEFINING gradient-reaches-all-5-subnets test + CPU smoke epochs=1 rc=0 with main_rate=0.92 / hyper_rate=5.32) DEMONSTRATES 'NSCS03 paradigm-on-contest-scorer is empirically viable at default config'. They demonstrate (a) the Ballé 2018 architecture is correctly wired (engineering achievement), (b) gradients flow end-to-end through 5 sub-networks (correctness achievement), and (c) the trainer's CPU smoke produces parseable stats with a final_loss of 11.47 / rate values that suggest the entropy bottleneck is exercised (integration achievement). They are SILENT on: (1) whether `λ_R=0.5` is anywhere near optimal at comma video's 25× rate scaling (the Ballé2018 paper uses `[0.001, 0.05]` range for natural images at MUCH lower bit-rates; our scaling makes the rate-distortion tradeoff fundamentally different); (2) whether `σ-floor=1e-4` is appropriate for comma video at 384×512 (Ballé2018 analytical bound is `~3.8e-6` which is 25× looser); (3) whether the entropy bottleneck DOES learn a non-trivial distribution that the inflate path CONSUMES (the Z3-G1 empty-slot anti-pattern per Catalog #272). The HARD-EARNED reading is: NSCS03 is paradigm-WIRED and integration-COMPLETE. The CARGO-CULTED reading is: 'Ballé 2018 is hard-earned at ICLR2018 + Phase 1 lift landed clean → therefore NSCS03 at default config must work on contest-CUDA'. The Ballé 2018 paradigm at NATURAL-IMAGE level is hard-earned; the COMMA-VIDEO application of it at $60-80 A100 cost with UNCALIBRATED hyperparameters is HYPOTHESIS PENDING EMPIRICAL ANCHOR. Per the META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable: my mandatory hypothesis is — 'the canonical Ballé hyperparameter defaults inherited from natural-image regime do NOT transfer 1:1 to comma driving video at 384×512 + 25× rate scaling; the optimal `(λ_R, σ-floor)` point requires a $8-22 calibration sweep BEFORE the $60-80 full dispatch.' Verdict on the assumption-violation hypothesis: STRONG support from (a) Ballé2018 paper's own rate-distortion sensitivity discussion + (b) Z3-G1 empty-distinguishing-bytes precedent + (c) CLAUDE.md Catalog #239 long_burn discipline. Required action: λ_R + σ-floor co-sweep at Lightning T4 MUST run BEFORE any A100 full dispatch can fire. BUT — and this is the KEY differentiation from Z6 — NSCS03 has REAL gradient-reaches-all-5-subnets EVIDENCE (the DEFINING test); Z6 had IDENTITY-PREDICTOR-TIES-FILM EVIDENCE. NSCS03's signal class is OPPOSITE to Z6's. Both require calibration; NSCS03 has STRONGER prior that the architecture is working."
council_assumption_adversary_verdict:
  - assumption: "Phase 1 trainer lift (commit 2026-05-15 PR95-paradigm + Ballé-2018-recipe) closes recipe reactivation criterion #3 (Phase 2 follow-up subagent) cleanly"
    classification: HARD-EARNED
    rationale: "Catalog #229 premise verification (6 PVs in the lift memo); 76/76 dedicated tests pass (53 pre + 23 new test_nscs03_full_main); DEFINING test gradient-reaches-all-5-subnets (g_a/g_s/h_a/h_s/entropy_bottleneck_z) PASSES; CPU 1-epoch smoke rc=0 with main_rate=0.92 / hyper_rate=5.32 / final_loss=11.47; PR95-paradigm-bound per Catalog #187; canonical scorer-preprocess routing per Catalog #164; Catalog #226/#193/#190/#180 strict gates report 0 NSCS03 violations. The trainer IS legitimately lifted from NotImplementedError to working PR95+Ballé-2018-paradigm-compliant implementation. This is MATERIALLY STRONGER than Z6's lift evidence which lacked the gradient-reaches-all-subnets DEFINING test."
  - assumption: "NSCS03 default config (λ_R=0.5 + σ-floor=1e-4 + single EMA decay 0.997) is near-optimal for comma video at 384×512 + 25× rate scaling"
    classification: CARGO-CULTED
    rationale: "Three CARGO-CULTED defaults inherited from Ballé2018 natural-image regime: (1) λ_R=0.5 is mid-range; Ballé2018 paper uses `[0.001, 0.05]` range for natural images at much LOWER bit-rates; comma video at our rate normalization may need different. (2) σ-floor=1e-4 is 25× TIGHTER than Ballé2018 analytical lower bound `~3.8e-6`; this may pinch the entropy bottleneck and force suboptimal latent quantization at low operating points. (3) Single EMA decay 0.997 across joint encoder + hyperprior is the CARGO-CULTED canonical (we adopted single without empirical proof); Ballé2018 uses differentiated 0.999 hyperprior + 0.997 main. No internal paired comparison exists; reactivation criteria explicitly cite this in the lift memo §'What can change my mind' section. Per CLAUDE.md 'Apples-to-apples evidence discipline': defaults inherited from a different operating regime are HYPOTHESIS PENDING EMPIRICAL ANCHOR; sweep-first at $8-22 Lightning T4 is canonical risk mitigation for the $60-80 A100 full dispatch."
  - assumption: "The entropy bottleneck weights + main+hyper latents are operationally consumed by inflate (i.e., the distinguishing feature actually contributes to score per Catalog #272)"
    classification: CARGO-CULTED
    rationale: "Z3-G1 sister precedent (codex review bkrbqet3p F1) shipped IDENTICAL Z3HV2 baseline bytes (hyperprior_weights_int8=b\"\" + w_hat_int8=b\"\") because the distinguishing-feature slots were empty; the $0.59 smoke score (0.19869) matched Z3 v2 baseline to 5 decimals because the bytes were not consumed. NSCS03 declares 5 state_dicts + 2 latent streams in `_extract_module_state_dicts` + `_encode_latents_for_archive` per the lift memo; the lift memo's archive helper tests (`TestArchiveBuildHelpers`) verify SHAPES + hard-rounded values BUT do NOT verify that mutating one byte of the archive changes inflate output. Per Catalog #272 distinguishing-feature integration contract: every L2+ promotion requires byte-mutation proof. The smoke gate's executable byte-mutation smoke (planned in `tests/test_nscs03_roundtrip.py` per Catalog #139) IS the empirical surface that resolves this. Until it runs at smoke scale, the operational-consumption claim is CARGO-CULTED."
  - assumption: "Recipe `min_smoke_gpu: A100` reflects genuine substrate-engineering requirement (substrate too memory-intensive for T4)"
    classification: HARD-EARNED
    rationale: "Recipe line 35-38 explicitly documents the constraint: 'End-to-end joint codec at 384x512 resolution + 64-channel main latent + entropy bottleneck is too memory-intensive for T4; A100 smoke is required to avoid 1h timeouts.' This is per Catalog #215 FIX-HARDEN-OPT 2026-05-14 P0 hardening lesson. The constraint is mechanical (memory footprint of conv g_a/g_s + hyper g_a/g_s + entropy bottleneck state at 64 main + 32 hyper channels at 384×512). Smoke MUST run on A100 (~$1.50-4.00 per Subagent C plan); the canonical 'cheap smoke at T4 first' pattern is NOT applicable to NSCS03. Per CLAUDE.md 'Hardware correctness Tier 2' (Catalog #270): recipe min_vram_gb=40 + min_smoke_gpu=A100 are correctly declared."
  - assumption: "Catalog #167 smoke-before-full at $1.50-4.00 Modal A100 is sufficient empirical gate before FULL dispatch"
    classification: HARD-EARNED
    rationale: "Catalog #167 IS the canonical smoke-before-full pattern per CLAUDE.md 'Production-hardened dispatch optimization protocol'; for NSCS03 the smoke validates rc=0 + archive bytes in expected band + integration smoke + (when paired with Catalog #272 byte-mutation proof at same archive bytes) gives the council a SCORER-BEARING anchor that the cheap CPU 1-epoch smoke cannot. Per Catalog #270 Tier 1/2/3 umbrella: NSCS03 trainer satisfies all 3 tiers (per lift memo §3 Dimension 5 + the operator-decision-investigation memo §3 gate matrix). The A100 smoke cost ($1.50-4.00) is 2-5% of full cost ($60-80) — canonical risk mitigation."
  - assumption: "Per CLAUDE.md 'Forbidden premature KILL': even if smoke fails or sweep diverges, the verdict is DEFER not KILL"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'KILL/FALSIFIED memory verdicts' non-negotiable + 'Forbidden premature KILL without research exhaustion': a single NSCS03-v1 configuration's failure does NOT exhaust the Ballé 2018 joint-codec paradigm. Reactivation criteria include NSCS03-v2 redesign (alternative hyperparameter ranges; different latent channel widths; learned scale prior variants per Ballé2019 follow-on; CompressAI reference implementation comparison) per design memo + operator-decision-investigation §1 + sister Z6 deliberation §6 Revision #3 contingency pattern. The verdict for one config failure is DEFERRED-pending-research, NOT KILLED. Z6 sister deliberation (sextet_council_z6_phase_2_consensus_20260516) is the canonical contingency-branching precedent."
  - assumption: "NSCS03 dispatch at the predicted-band claim would be 'frontier_breaking' mission-contribution per Catalog #300 + outside-NeRV architectural-class diversity per operator binding constraint 2026-05-16"
    classification: PARTIALLY HARD-EARNED + CARGO-CULTED
    rationale: "HARD-EARNED for the OUTSIDE-NeRV architectural-class diversity (per `coherence_audit_lattice_coordinate_assignment_20260516.md`: NSCS03 is canonical Rule #2 outside-NeRV substrate; Substitution Set B explicitly recommends NSCS03 as Disambiguator #1 → Frontier-pursuit #7 promotion); HARD-EARNED for the FRONTIER-PURSUIT horizon-class per Catalog #309 (predicted band [0.180, 0.190] within frontier-pursuit [0.120, 0.180] OR upper plateau-adjacent [0.180, 0.200] depending on calibration outcome). CARGO-CULTED for the 'frontier_breaking' claim until empirical anchor confirms paradigm produces ΔS < current frontier (~0.193 PR101 medal-band leader). Per Catalog #300 mission-alignment: the CURRENT sweep + smoke + byte-mutation dispatch is `frontier_protecting` (protects against false promotion of an uncalibrated $60-80 burn that may collapse to Z3-G1-style empty-bytes failure); the POTENTIAL future FULL dispatch (conditional on sweep-converged + smoke-green + byte-mutation-PASS) is `frontier_breaking` (outside-NeRV + lifts the NSCS lattice rule #2 coverage from 1-of-3 anchors to 2-of-3 anchors, enabling 3-stack composition design per CLAUDE.md cross-archive composition rule)."
council_decisions_recorded:
  - "Recipe REMAINS research_only=true / dispatch_enabled=false (NO change in same commit batch as this memo). Recipe reactivation criterion #4 (Phase 2 sextet-pact council CONSENSUS) IS satisfied via THIS deliberation's verdict PROCEED_WITH_REVISIONS (NOT PROCEED unconditionally). The recipe flip waits for criterion #5 + the 4 revisions documented below."
  - "Revision #1 (mandatory before any FULL dispatch): run λ_R + σ-floor CO-SWEEP at Lightning T4 ~$8-22 / 4-5h wall-clock parallel BEFORE A100 smoke. Sweep matrix: `λ_R ∈ {0.1, 0.5, 1.0, 5.0}` × `σ_floor ∈ {1e-3, 1e-4, 1e-5}` reduced from 12-config to 5-config Lightning T4 paired smokes per sister `nscs03_operator_decision_items_investigation_20260515.md` §5 Path A. Pick lowest-scoring config per CLAUDE.md 'Apples-to-apples evidence discipline' `[contest-CUDA Lightning T4]` tag. Output: council-approved (λ_R, σ_floor) pair for the A100 smoke + full envelope. Cost: $8-22."
  - "Revision #2 (mandatory before FULL): Catalog #167 smoke-before-full at Modal A100 ~$1.50-4.00 / 30-60 min using the Revision-#1-converged (λ_R, σ_floor) pair. Smoke acceptance criteria: `rc=0` from `experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py` invocation; archive bytes in [80, 250] KB OR substrate-engineering documented band; integration smoke validates per Catalog #243 local pre-deploy harness; Catalog #166 Modal HEAD-parity ledger captures dispatch; Catalog #245 modal_call_id_ledger registers the call_id BEFORE submit per Catalog #143; Catalog #167 sister wrapper `tools/run_modal_smoke_before_full.py --recipe substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch --smoke-only --smoke-gpu A100` orchestrates."
  - "Revision #3 (mandatory before FULL): Catalog #272 distinguishing-feature byte-mutation no-op detector proof at SAME archive bytes from Revision #2 smoke output. Decision criterion: mutate one byte in each of `entropy_state_sd`, `main_latents`, `hyper_latents` slots; verify inflate output frames change byte-for-byte (Catalog #139 packet compiler no-op detector). If ALL 3 distinguishing slots produce frame-level mutation effects → PROCEED branch (Revision #4). If ANY slot is empty/non-consumed → DEFER-pending-archive-grammar-fix per Z3-G1 precedent (NOT KILL per CLAUDE.md 'KILL is LAST RESORT'). The smoke + byte-mutation artifacts MUST emit per Catalog #221 fail-closed structure (`auth_eval_score_axis=contest_cuda` + `auth_eval_score_claim_valid=true` + `evidence_grade=contest-CUDA` + `promotion_eligible=false` until paired CPU anchor lands)."
  - "Revision #4 (procedural fork; NOT blocking smoke; PRE-COMMIT for full): Ballé 2018 differentiated EMA 0.999 hyperprior + 0.997 main split implemented inline as Layer-7 UNIQUE substrate-engineering. Update landing memo canonical-vs-unique table Layer 7 from `CANONICAL ADOPT` to `UNIQUE per Phase 2`. Add same-line waiver `# EMA_DECAY_DIFFERENTIATED_OK:balle2018_recipe_canonical_for_joint_codec` in trainer _full_main. Engineering cost ~2h editor; compute cost marginal (EMA post-step). Per CLAUDE.md 'EMA — non-negotiable' canon: `0.997` for weights is the canonical default; the differentiated split is a substrate-engineering exception per HNeRV parity L7. Council sign-off granted via THIS deliberation; waiver documentation makes it auditable in 30 seconds."
  - "Revision #5 contingency (PROCEED branch): IF Rev-1 sweep converges AND Rev-2 smoke-green AND Rev-3 byte-mutation PASSES on all 3 distinguishing slots → flip recipe to dispatch_enabled=true + research_only=false in follow-on commit batch + dispatch FULL ($60-80 Modal A100 / 2-12h per recipe `timeout_hours: 12.0`) + paired CPU/CUDA auth eval per Catalog #226 ($0.10-0.50 paired) + posterior_update_locked per Catalog #128 + cathedral autopilot wire-in per Catalog #125 hook #4 + lattice ledger event_type=promoted per `tools/check_lattice_coordinate.py`. Total path envelope: $70-107 per nscs03_operator_decision_items_investigation §5 Path A."
  - "Revision #5 contingency (DEFER branch): IF Rev-1 sweep diverges OR Rev-2 smoke-fails OR Rev-3 byte-mutation FAILS on any slot → mark NSCS03-v1-default-config `measured-config-retired` per CLAUDE.md 'KILL = LAST RESORT' + DEFER NSCS03-v2 redesign per design memo + operator-decision-investigation §1 Item 7 (alternative hyperparameter ranges / different latent channel widths / learned scale prior variants / CompressAI reference implementation comparison). Recipe stays research_only=true / dispatch_enabled=false; lane stays L1; `_full_main` keeps its current implementation (no revert). No KILL of paradigm; the SPECIFIC NSCS03-v1 hyperparameter+EMA configuration is the retired config."
  - "Sextet-pact CONSENSUS verdict: PROCEED_WITH_REVISIONS at 6-of-6 quorum + Ballé advisory consultation. Per CLAUDE.md 'Council conduct — non-negotiable': this is NOT a lazy consensus; the 4 revisions are binding. Dissent surfaced verbatim per Catalog #300 (Contrarian + Assumption-Adversary both registered substantive revisions; consensus is on the REVISION SET, not on unconditional approval). Ballé advisory consultation NOTED but does NOT count toward quorum tally (Grand Council advisory member per CLAUDE.md 'Grand Council (advisory)' canonical roster pattern)."
  - "Cross-decision dependency: NSCS03 results inform A-STACK 3-substrate composition (Rule #3) per `coherence_audit_lattice_coordinate_assignment_20260516.md` + design memo `a_stack_nscs01_02_03_composition_full_stack_design_20260516.md`. If NSCS03-v1 smoke + sweep + byte-mutation green-up AND NSCS01 sister Phase 2 council (concurrent sister subagent) also PROCEED-CONDITIONAL, the 3-stack composition becomes UNBLOCKED per CLAUDE.md 'Forbidden cross-archive composition' anchor (≥2-of-3 NSCS substrates landing contest-CUDA anchors). If only NSCS01 lands anchor and NSCS03 DEFERs at any revision: composition design DEFERS to wait for NSCS03-v2 or NSCS02 anchor."
  - "30-day deferred-substrate retrospective scheduled 2026-06-16 for the NSCS03-v1 hyperparameter+EMA configuration (per Catalog #300 mission-alignment consequence 3) — if Revision #5 DEFER branch fires AND NSCS03-v2 redesign has not landed empirical anchor at +30d, the operator reviews whether to escalate to T3 grand council OR DEFER indefinitely OR pivot to balle_renderer sister substrate."
  - "Predicted cost band: $8-22 sweep (Revision #1 Lightning T4) + $1.50-4.00 smoke (Revision #2 Modal A100) + $0 byte-mutation proof (Revision #3 local) + ~$0 EMA split implementation (Revision #4 editor only) = $9.50-26 NEXT immediate envelope. Conditional FULL + paired adds $60-80.50 per Revision #5 PROCEED branch. Total envelope $69.50-106.50 per nscs03_operator_decision_items_investigation §5 Path A."
  - "Per Catalog #300 mission-alignment + HORIZON-CLASS Consequence 5: operator-frontier-override NOT INVOKED for this deliberation; no time-critical innovation pressure pushes for unconditional approval. The canonical Path A (calibrate-first, then dispatch) is operator-recommended per nscs03_operator_decision_items_investigation §5 Path A recommendation."
  - "Per CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD operating mode' + 'Race-mode rigor inversion + parallel-dispatch first': no active leaderboard race detected (PR101 medal-band leader at 0.193 stable for last 24h+); standard sextet-pact procedure applies."
deferred_substrate_retrospective_due_utc: "2026-06-16T03:00:00Z"
deferred_substrate_id: nscs03_end_to_end_balle_joint_codec_v1_default_hyperparameter_configuration
related_deliberation_ids:
  - sextet_council_z6_phase_2_consensus_20260516
  - grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516
  - grand_council_t3_wave_2_batch_nscs06_rescue_tishby_wunderkind_nerv_family_phase_2_lift_20260516
  - nscs03_operator_decision_items_investigation_20260515
  - coherence_audit_lattice_coordinate_assignment_20260516
  - wave_3_optimization_per_lattice_coherence_20260516
  - a_stack_nscs01_02_03_composition_full_stack_design_20260516
event_type: dispatched
parent_id_or_session: nscs03_phase_2_sextet_council_20260516
memory_path: .omx/research/sextet_council_nscs03_phase_2_consensus_20260516.md
lane: lane_nscs03_phase_2_sextet_council_20260516
---

# NSCS03 Phase 2 Sextet-Pact Council CONSENSUS Deliberation

**Date:** 2026-05-16
**Lane:** `lane_nscs03_phase_2_sextet_council_20260516`
**Council tier:** **T2** (sextet pact 6-of-6 per Catalog #300 v2 quorum + Ballé advisory)
**Mission-alignment classification:** `frontier_breaking` (sweep + smoke + byte-mutation gate; potentially unlocks FULL outside-NeRV FRONTIER-PURSUIT dispatch if Revision #5 PROCEED branch fires)
**Operator override:** NOT INVOKED
**Quorum:** **MET** — sextet pact 6/6 (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary); Ballé Grand Council advisory consultation invoked
**Verdict:** **PROCEED_WITH_REVISIONS** — recipe REMAINS research_only=true / dispatch_enabled=false; 4 mandatory revisions enumerated; Catalog #167 smoke-before-full + Catalog #272 byte-mutation proof + Revision #1 sweep + Revision #4 EMA split IS the explicit next gate sequence
**Deferred-substrate retrospective scheduled:** **2026-06-16T03:00:00Z** (NSCS03-v1 default hyperparameter+EMA configuration; per Catalog #300 consequence 3)

---

## 0. Premise verifications per Catalog #229 (8 pre-deliberation anchors)

1. **PV-1**: `experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py::_full_main` is implemented (NOT NotImplementedError) — verified via lift memo §3 + the 23 new test_nscs03_full_main tests including `TestFullMainContractWired` which explicitly asserts `_full_main` no longer raises NotImplementedError.
2. **PV-2**: Phase 1 lift tests pass — verified via lift memo §"Empirical receipts" line: "Total 76 tests | all PASS"; this is materially stronger than Z6's 24-test surface.
3. **PV-3**: NSCS03 substrate package present at `src/tac/substrates/nscs03_end_to_end_balle_joint_codec/` with `__init__.py + architecture.py + archive.py + score_aware_loss.py` (read in full); 8-field Catalog #124 archive grammar declaration present in `__init__.py` docstring.
4. **PV-4**: Recipe at `.omx/operator_authorize_recipes/substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch.yaml` carries `research_only: true` + `smoke_only: true` + `dispatch_enabled: false` + explicit 3-item `dispatch_blockers` list (phase_2_council_approval + lambda_R_sweep + sigma_floor) + `min_smoke_gpu: A100` per Catalog #215 + `min_vram_gb: 40` per Catalog #170 + `target_modes: [research_substrate]` per Catalog #182.
5. **PV-5**: Phase 1 lift landing memo `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md` confirms 14 CANONICAL ADOPT + 4 UNIQUE (NSCS03 substrate / score-aware loss with END-TO-END diff rate / Ballé λ_R linear warmup / archive build from 5 state_dicts + 2 latent streams) + 3 DOCUMENTED FORK (AUTOCAST_FP16_WAIVED EB+GDN fp16 instability / TORCH_COMPILE_WAIVED / Ballé 0.999/0.997 differentiated EMA deferred Phase 2) per UNIQUE-AND-COMPLETE-PER-METHOD operating mode.
6. **PV-6**: Operator-decision-items investigation `.omx/research/nscs03_operator_decision_items_investigation_20260515.md` enumerated 8 items; 4 are council-grade (Items 1, 2, 3, 4); recommended Path A (calibrate-first, $70-107 total).
7. **PV-7**: Lattice ledger query `tools/check_lattice_coordinate.py --substrate nscs03` returns: `lattice_rule: rule_2_nullspace_split_pr95_paradigm` + `horizon_class: frontier_pursuit` + `architectural_class: balle_2018_end_to_end_joint_codec` + `status: lifted_pending_council` + `paradigm_vs_implementation_classification: paradigm_intact`. Wave 3 optimization recommendation per `wave_3_optimization_per_lattice_coherence_20260516.md` Substitution Set B explicitly promotes NSCS03 from Disambiguator #1 → Frontier-pursuit #7 (outside-NeRV).
8. **PV-8**: Z6 sister deliberation `sextet_council_z6_phase_2_consensus_20260516.md` (commit `1a2d84b3d`) is the canonical sextet-pact template — verdict PROCEED_WITH_REVISIONS at 6-of-6 quorum with 3 revisions; this NSCS03 deliberation MIRRORS the structure but adapts to NSCS03's materially-stronger evidence surface (gradient-reaches-all-5-subnets DEFINING test vs Z6's identity-tie probe collapse).

All 8 PVs PASS. No regression from parent-prompt assertions.

---

## 1. Council attendance + quorum

**Sextet pact (6-of-6 quorum required at T2 per Catalog #300 v2) + Ballé Grand Council advisory:**

| Seat | Role | Operating-within assumption (Catalog #292 per-deliberation discipline) |
|---|---|---|
| **Shannon** (LEAD) | Information-theory grounding | "The shared assumption I am operating within is that Ballé 2018's rate-distortion theorem `R(D) ≥ I(X;Y)` applies at our operating point of `25*archive_bytes/37545489` rate scaling. The end-to-end joint codec produces a learned conditional distribution `p(y|x)` whose differential entropy IS the achievable rate. The empirical question is whether the joint optimization converges to a non-degenerate operating point at `λ_R=0.5` OR whether the rate term dominates / collapses." HARD-EARNED at theorem level + CARGO-CULTED at Pact-specific λ_R application. |
| **Dykstra** (CO-LEAD) | Convex-feasibility intersection check per Catalog #296 | "The shared assumption I am operating within is that the Pareto-feasibility polytope at NSCS03 default config (λ_R=0.5, σ_floor=1e-4, single EMA 0.997) is empirically non-empty for comma video at 384×512. The rate-distortion-perception triple constraint intersects at SOME (rate, seg, pose) point; the question is whether that point projects onto contest-CUDA score in [0.180, 0.190] OR somewhere outside. Per Catalog #296: the predicted band is HIGH-VARIANCE pending paired CPU/CUDA empirical anchor. The Z3-G1 cargo-cult-prediction precedent (predicted [0.13, 0.16]; landed 0.19869 because distinguishing bytes were empty) is salient." HARD-EARNED at convex-feasibility level + CARGO-CULTED at projection level. |
| **Yousfi** | Steganalysis + scorer design context | "The shared assumption I am operating within is that SegNet's stride-2 EfficientNet-B2 stem and PoseNet's FastViT-T12 12-channel YUV6 input are the contest scorer architecture (per CLAUDE.md 'Exact scorer architectures'). The NSCS03 end-to-end joint codec must produce decoded frames the scorer extracts class+pose from with low distortion. The convolutional g_a/g_s + entropy bottleneck design has higher representational capacity than the balle_renderer sister (per-pair flat latents + MLP), so PRINCIPLED expectation is NSCS03 should outperform balle_renderer at matched bit-rate. The empirical question is whether the joint optimization converges to scorer-favorable frame statistics at default λ_R." HARD-EARNED at scorer-architecture level. |
| **Fridrich** | Inverse steganalysis + CNN blind spots | "The shared assumption I am operating within is that UNIWARD-style coverage analysis — embedding error is undetectable in textured regions — applies to NSCS03's residual encoding because the conv g_a/g_s output residuals naturally concentrate in high-entropy regions. The Ballé 2018 paradigm aligns with UNIWARD textured-region preference at the PRINCIPLE level. The empirical question is whether single-pair λ_R=0.5 + σ_floor=1e-4 has enough representational headroom for the entropy bottleneck to allocate bits to scorer-blind regions; the 5-subnet DEFINING test passes but says nothing about scorer-blind-region alignment." HARD-EARNED at UNIWARD-principle level + CARGO-CULTED at Pact-specific λ_R application. |
| **Contrarian** | Challenge weak arguments + veto lazy consensus | "I push back on any framing that treats 'Phase 1 trainer lift landed (76 tests pass)' as equivalent to 'Phase 2 sextet-pact council CONSENSUS approving recipe flip to dispatch_enabled=true at default config'..." [see council_dissent.Contrarian.verbatim above for the full position]. The Phase 1 lift IS materially stronger than Z6's lift evidence (DEFINING gradient test + 76 tests vs Z6's identity-tie probe collapse + 24 tests). But the recipe ITSELF encodes 3 explicit dispatch_blockers; ignoring them is a lazy-consensus failure mode. 4 revisions mandatory. HARD-EARNED at engineering-risk level. |
| **Assumption-Adversary** (NEW sextet seat per CLAUDE.md "Council conduct" Fix 7) | Challenge the FRAMING all arguments share | "I challenge the SHARED ASSUMPTION operating across this council deliberation: that the Phase 1 lift commits' evidence surface DEMONSTRATES 'NSCS03 paradigm-on-contest-scorer is empirically viable at default config'..." [see council_dissent.Assumption-Adversary.verbatim above for the full position]. The HARD-EARNED reading is NSCS03 is paradigm-WIRED and integration-COMPLETE — materially stronger than Z6. The CARGO-CULTED reading is Ballé 2018 natural-image defaults transfer 1:1 to comma video. The KEY DIFFERENTIATION from Z6: NSCS03's signal class is OPPOSITE (DEFINING test PASSES gradient flow); Z6 had identity-tie evidence. Both require calibration; NSCS03 has STRONGER prior the architecture works. |
| **Ballé** (Grand Council advisory) | 2018 joint-codec author lineage | (Advisory consultation per CLAUDE.md "Grand Council (advisory)" canonical pattern; does NOT count toward sextet quorum but provides 2018 ICLR paradigm-author voice.) "The shared assumption I am operating within is that the canonical Ballé 2018 architecture (conv g_a/g_s + entropy bottleneck + scale hyperprior) requires per-domain hyperparameter calibration; the natural-image regime `λ_R ∈ [0.001, 0.05]` does NOT transfer 1:1 to comma video at 25× rate scaling. The differentiated EMA 0.999/0.997 split + σ-floor ~3.8e-6 analytical bound + per-channel quantization step are HARD-EARNED Ballé 2018 lessons. The trainer's 14 CANONICAL ADOPT + 4 UNIQUE + 3 DOCUMENTED FORK structure is the right shape; the 3 DOCUMENTED FORKs (autocast_fp16 / torch.compile / differentiated EMA) are correct recognition of paradigm-specific requirements. Recommended: implement differentiated EMA inline (Revision #4); sweep λ_R + σ-floor at Lightning T4 (Revisions #1+#2); proceed to A100 smoke + byte-mutation proof (Revisions #2+#3) before any full dispatch." HARD-EARNED at paradigm-author lineage level. |

**Quorum:** 6-of-6 sextet pact present. ALL 6 seats cast PROCEED_WITH_REVISIONS. NO seat voted PROCEED unconditionally; NO seat voted DEFER_PENDING_EVIDENCE / REFUSE / ESCALATE; the consensus IS on the REVISION SET, not on unconditional approval. Per CLAUDE.md "Council conduct": this is a CONSENSUS with substantive minority opinions surfaced verbatim, NOT a lazy 6-of-6-rubber-stamp.

---

## 2. Vote tally (per Catalog #300 maximum signal preservation rule)

| Seat | Vote | Explicit reasoning |
|---|---|---|
| Shannon | PROCEED_WITH_REVISIONS | "R(D) lower bound IS the right floor; default λ_R inherited from natural-image regime needs empirical anchor at our 25× rate scaling." |
| Dykstra | PROCEED_WITH_REVISIONS | "Polytope FEASIBLE at math level; empirical projection onto contest-CUDA UNKNOWN at default config; Z3-G1 cargo-cult-prediction precedent salient." |
| Yousfi | PROCEED_WITH_REVISIONS | "Scorer architecture is canonical anchor; smoke gate + byte-mutation proof IS the canonical empirical probe; conv g_a/g_s has higher capacity than balle_renderer MLP sister." |
| Fridrich | PROCEED_WITH_REVISIONS | "UNIWARD principle aligns; default-config scorer-blind-region allocation unproven; smoke + sweep + byte-mutation required." |
| Contrarian | PROCEED_WITH_REVISIONS | "4 revisions mandatory; recipe flip premature without sweep + smoke + byte-mutation; differentiated EMA implemented inline as substrate-engineering." |
| Assumption-Adversary | PROCEED_WITH_REVISIONS | "Hard-earned-vs-cargo-culted classification: NSCS03 paradigm is wired (HARD-EARNED); default config is CARGO-CULTED; empirical anchor required; signal class OPPOSITE to Z6 (gradient flows; gradient-reaches-all-5-subnets DEFINING test passes)." |

**Tally:** 6 PROCEED_WITH_REVISIONS / 0 PROCEED / 0 DEFER / 0 REFUSE / 0 ESCALATE / 0 abstain / 0 recused. Ballé advisory consultation noted (not counted toward quorum).

**Consensus achieved:** 6-of-6 on PROCEED_WITH_REVISIONS verdict + 6-of-6 on the 4-revision set (Revision #1 mandatory λ_R + σ-floor co-sweep + Revision #2 mandatory smoke-before-full + Revision #3 mandatory Catalog #272 byte-mutation proof + Revision #4 PROCEDURAL differentiated EMA split implementation).

---

## 3. Predicted ΔS band + Dykstra-feasibility verdict per Catalog #296

**Predicted band (per design memo + lift memo + operator-decision-investigation):**

- **NSCS03 CPU [contest-CPU Linux x86_64 GHA]**: `[0.180, 0.190]` `[prediction; Dykstra-feasibility-validated at math level; HIGH VARIANCE pending paired CPU/CUDA empirical anchor + λ_R sweep]`
- **NSCS03 CUDA [contest-CUDA Modal A100]**: `[0.180, 0.190]` `[prediction; sister Ballé estimate via balle_renderer family + 25× rate scaling adjustment; CUDA-CPU gap negligible for non-rate-heavy substrates]`
- **Dykstra-feasibility VERDICT** (per Catalog #296): **FEASIBLE**. Polytope vertices at NSCS03 default config: rate ∈ [0.05, 0.10] (rate budget bounded by recipe `archive bytes ~80-250 KB` × `25 / 37_545_489` = 0.053-0.166); seg ∈ [0.06, 0.10]; pose ∈ [0.00005, 0.00015]. Convex-intersection NON-EMPTY at math level. **First-principles citation:** Ballé 2018 ICLR rate-distortion bound + Shannon R(D) theorem + Dykstra alternating-projections convergence on rate-distortion-perception triple constraint.

**Per Catalog #296 + the assumption-classification verdict above**: the predicted band is BOUNDED BELOW by current frontier 0.193 (PR101 medal-band leader) at LOW confidence (CARGO-CULTED until empirical anchor); the BAND CENTER 0.185 is HARD-EARNED at first-principles but the actual projection MAY land anywhere in [0.18, 0.22] depending on:
- λ_R sweep convergence (Revision #1)
- σ-floor calibration (Revision #1)
- Catalog #272 byte-mutation verdict on distinguishing-feature bytes (Revision #3)

**Per CLAUDE.md "Apples-to-apples evidence discipline"**: ALL bands tagged `[prediction]`; NO promotion language until paired CPU/CUDA empirical anchor.

---

## 4. Horizon-class declaration per Catalog #309

**NSCS03 horizon-class: `frontier_pursuit`** per design memo + lattice ledger entry + Wave 3 Substitution Set B promotion. NSCS03 fills the outside-NeRV frontier-pursuit slot per operator binding constraint 2026-05-16; together with sister NSCS01 it forms the outside-NeRV architectural-class diversity in the K=13 LEVEL-1 measurement schedule.

The predicted band [0.180, 0.190] straddles the FRONTIER-PURSUIT [0.120, 0.180] / PLATEAU-ADJACENT [0.180, 0.200] boundary. Per HORIZON-CLASS standing directive 2026-05-16 Consequence 2 (≥20% K-schedule allocation to asymptotic-pursuit): NSCS03 is structurally **frontier-pursuit-candidate** (potential to escape plateau if calibration succeeds) NOT asymptotic-pursuit. If empirical anchor lands above 0.193 → plateau-adjacent reclassification; below 0.180 → confirmed frontier-pursuit.

---

## 5. Mission contribution prediction per Catalog #300

**`council_predicted_mission_contribution: frontier_breaking`** at the POTENTIAL FUTURE FULL dispatch (conditional on Revision #5 PROCEED branch firing); CURRENT sweep + smoke + byte-mutation dispatch is `frontier_protecting` (protects against false promotion of an uncalibrated $60-80 burn that may collapse to Z3-G1-style empty-bytes failure).

Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 4 (frontier-breaking moves DOMINATE rigor budget when leaderboard moves): NO active leaderboard race detected (PR101 medal-band leader at 0.193 stable for last 24h+); standard sextet-pact procedure applies; operator-frontier-override NOT INVOKED.

Per the architectural-class diversity argument: NSCS03 successful dispatch unlocks the 3-stack A-STACK composition (NSCS01+02+03) per CLAUDE.md "Forbidden cross-archive composition" anchor (≥2-of-3 NSCS substrates landing contest-CUDA anchors). This is the **frontier-breaking compositional path** that the plateau cluster cannot reach.

---

## 6. The 4 mandatory revisions (binding) + Revision #5 contingency branching

### Revision #1: λ_R + σ-floor CO-SWEEP at Lightning T4 BEFORE A100 smoke

- **Sweep matrix**: `λ_R ∈ {0.1, 0.5, 1.0, 5.0}` × `σ_floor ∈ {1e-3, 1e-4, 1e-5}` → reduced 5-config Lightning T4 paired smokes per nscs03_operator_decision_items_investigation §5 Path A
- **Cost**: $8-22 / 4-5h wall-clock parallel
- **Acceptance**: pick lowest-scoring (λ_R, σ_floor) config per CLAUDE.md "Apples-to-apples evidence discipline" `[contest-CUDA Lightning T4]` tag
- **Output**: council-approved (λ_R, σ_floor) pair updated in recipe `env_overrides.NSCS03_LAMBDA_R` + `NSCS03_SIGMA_FLOOR`

### Revision #2: Catalog #167 smoke-before-full at Modal A100 using Rev-1 converged config

- **Smoke acceptance criteria** (all required):
  - `rc=0` from `experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py` invocation
  - Archive bytes in `[80, 250]` KB OR substrate-engineering documented band
  - Integration smoke validates per Catalog #243 local pre-deploy harness
  - Catalog #166 Modal HEAD-parity ledger captures dispatch
  - Catalog #245 modal_call_id_ledger registers the call_id BEFORE submit per Catalog #143
  - Catalog #167 sister wrapper `tools/run_modal_smoke_before_full.py --recipe substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch --smoke-only --smoke-gpu A100` orchestrates
- **Cost**: $1.50-4.00 / 30-60 min

### Revision #3: Catalog #272 distinguishing-feature byte-mutation no-op detector proof

- **Decision criterion**: mutate one byte in each of `entropy_state_sd` / `main_latents` / `hyper_latents` slots from Rev-2 smoke output; verify inflate output frames change byte-for-byte per Catalog #139 packet compiler no-op detector
- **PASS branch**: all 3 distinguishing slots produce frame-level mutation effects → Revision #5 PROCEED
- **FAIL branch**: ANY slot is empty/non-consumed → Revision #5 DEFER per Z3-G1 precedent (NOT KILL per CLAUDE.md 'KILL is LAST RESORT')
- **Artifact emission**: per Catalog #221 fail-closed structure (`auth_eval_score_axis=contest_cuda` + `auth_eval_score_claim_valid=true` + `evidence_grade=contest-CUDA` + `promotion_eligible=false` until paired CPU anchor lands)

### Revision #4: differentiated EMA 0.999/0.997 split implementation (PROCEDURAL FORK; NOT blocking smoke)

- **Implementation**: split EMA into TWO scoped instances — `0.999` for hyperprior modules (`h_a`, `h_s`, `entropy_bottleneck_z`) + `0.997` for main modules (`g_a`, `g_s`) per Ballé 2018 canonical recipe
- **Code change**: ~2h editor in `experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py` `_full_main`
- **Documentation**: same-line waiver `# EMA_DECAY_DIFFERENTIATED_OK:balle2018_recipe_canonical_for_joint_codec` per CLAUDE.md "EMA — non-negotiable" substrate-engineering exception per HNeRV parity L7
- **Landing memo update**: Layer 7 canonical-vs-unique table from `CANONICAL ADOPT` to `UNIQUE per Phase 2`
- **Timing**: PRE-COMMIT for FULL dispatch; OPTIONAL for smoke (single EMA still passes integration)

### Revision #5: Contingency branching on Rev-1 + Rev-2 + Rev-3 outcomes

**PROCEED branch** (sweep converges AND smoke green AND byte-mutation PASSES on all 3 slots):
1. Implement Revision #4 differentiated EMA split
2. Flip recipe `research_only: false` + `dispatch_enabled: true` in follow-on commit batch
3. Dispatch FULL ($60-80 Modal A100 / 2-12h per recipe `timeout_hours: 12.0`)
4. Paired CPU/CUDA auth eval per Catalog #226 + #190 ($0.10-0.50)
5. Posterior update per Catalog #128
6. Cathedral autopilot wire-in per Catalog #125 hook #4
7. Lattice ledger event_type=promoted via `tools/check_lattice_coordinate.py`
8. **Total envelope: $70-107**

**DEFER branch** (sweep diverges OR smoke fails OR byte-mutation FAILS on any slot):
1. Mark NSCS03-v1 default hyperparameter+EMA configuration `measured-config-retired` per CLAUDE.md "KILL = LAST RESORT"
2. DEFER NSCS03-v2 redesign per design memo + operator-decision-investigation §1 Item 7 (alternative hyperparameter ranges / different latent channel widths / learned scale prior variants / CompressAI reference implementation comparison)
3. Recipe stays research_only=true / dispatch_enabled=false
4. Lane stays L1
5. `_full_main` keeps current implementation (no revert)
6. NO KILL of paradigm; SPECIFIC NSCS03-v1 hyperparameter+EMA configuration is the retired config
7. 30-day retrospective scheduled 2026-06-16 per Catalog #300 consequence 3

---

## 7. Cross-decision dependency: NSCS01 + NSCS03 → A-STACK composition (Rule #3)

Per `coherence_audit_lattice_coordinate_assignment_20260516.md` + design memo `a_stack_nscs01_02_03_composition_full_stack_design_20260516.md`:

- IF NSCS03-v1 smoke + sweep + byte-mutation green-up AND NSCS01 sister Phase 2 council (concurrent sister subagent) also PROCEED-CONDITIONAL → 3-stack composition becomes UNBLOCKED per CLAUDE.md "Forbidden cross-archive composition" anchor (≥2-of-3 NSCS substrates landing contest-CUDA anchors)
- IF only NSCS01 lands anchor AND NSCS03 DEFERs at any revision → composition design DEFERS to wait for NSCS03-v2 OR NSCS02 anchor

This is the **frontier-breaking compositional path** unlocked by NSCS03's outside-NeRV architectural-class diversity (per operator binding constraint 2026-05-16).

---

## 8. Differentiation from Z6 sister deliberation

Key analytical insight: NSCS03 is **materially different from Z6** at the empirical evidence surface:

| Axis | Z6 (sister) | NSCS03 (this) |
|---|---|---|
| Phase 1 lift tests | 24/24 pass | **76/76 pass** |
| DEFINING test | NONE (identity-tie probes showed FILM collapse) | **gradient-reaches-all-5-subnets PASSES** |
| Probe corpus evidence | 7 codex memos: identity-predictor essentially ties full-FiLM (max delta 5.3e-6 loss proxy) | Trainer integration: main_rate=0.92 / hyper_rate=5.32 / final_loss=11.47 (parseable, non-degenerate) |
| Cargo-cult risk surface | FiLM architecture may collapse to identity at scorer-aware training | Default hyperparameters (λ_R / σ-floor / EMA) may be sub-optimal for comma video |
| Z3-G1 precedent risk | LOW (Z6 archive grammar verified non-empty) | **MEDIUM (Catalog #272 byte-mutation proof MANDATORY per Revision #3)** |
| Architectural class | predictive-coding world-model (asymptotic-pursuit) | balle-2018-end-to-end-joint-codec (frontier-pursuit; outside-NeRV) |
| Council verdict | PROCEED_WITH_REVISIONS (3 revisions: smoke + disambiguator + contingency) | PROCEED_WITH_REVISIONS (4 revisions: **sweep** + smoke + byte-mutation + EMA split + contingency) |

Z6 has STRONG EMPIRICAL EVIDENCE AGAINST (identity collapse); NSCS03 has STRONG EMPIRICAL EVIDENCE FOR (gradient flows + integration passes). Both require Phase 2 calibration; NSCS03 has a stronger prior the architecture is working.

---

## 9. 6-hook wire-in declaration (per CLAUDE.md Catalog #125)

1. **Sensitivity-map contribution**: N/A — substrate is complete codec, not per-tensor sensitivity producer; downstream sensitivity maps consume the latent representation.
2. **Pareto constraint**: rate_distortion_v1 (declared in NSCS03_END_TO_END_BALLE_CONTRACT.hook_pareto_constraint; this deliberation pins the operating point at Rev-1-converged λ_R + σ-floor).
3. **Bit-allocator hook**: ibps_kkt (per-channel entropy-bottleneck factorized prior + per-spatial conditional Gaussian scale hyperprior = bit allocator at latent layer).
4. **Cathedral autopilot dispatch hook**: ACTIVE via recipe gate + lane_id in lane_registry; this deliberation routes the dispatch decision through the autopilot ranker per Revision #5 contingency.
5. **Continual-learning posterior update**: ACTIVE via `append_council_anchor` for this deliberation + `posterior_update_locked` call site at Revision #5 PROCEED branch.
6. **Probe-disambiguator**: ACTIVE — Revision #3 Catalog #272 byte-mutation proof IS the canonical probe-disambiguator for the "distinguishing bytes actually consumed" question (per Catalog #297 signal-axis-destruction reversibility probe pattern).

---

## 10. Operator-routable next actions

1. **(Now; this commit batch)** Land this deliberation memo via canonical serializer per Catalog #117/#157/#174.
2. **(Within 24h)** Operator approves Revision #1 sweep dispatch (Lightning T4 5-config; $8-22).
3. **(After Rev-1 converges)** Operator approves Revision #2 Modal A100 smoke at converged config ($1.50-4.00).
4. **(After Rev-2 smoke-green)** Subagent runs Revision #3 Catalog #272 byte-mutation proof on smoke artifact ($0; local).
5. **(After Rev-3 PASSES OR pre-FULL)** Implement Revision #4 differentiated EMA split (~2h editor; ~$0).
6. **(If Rev-5 PROCEED branch)** Flip recipe + dispatch FULL + paired CPU/CUDA + posterior + autopilot ($60-80.50).
7. **(If Rev-5 DEFER branch OR 2026-06-16 retrospective)** Operator reviews NSCS03-v2 redesign OR pivot to balle_renderer sister.

---

## 11. Cross-references

- CLAUDE.md "Council hierarchy: 4-tier protocol" (T2 sextet-pact)
- CLAUDE.md "Council conduct — non-negotiable" (Fix 7 Assumption-Adversary seat + per-round assumption-statement discipline)
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable (Catalog #220 + #240)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (canonical-vs-unique decision per layer)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons L1-L13 (especially L7 substrate-engineering exception)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (DEFER not KILL on Rev-5 DEFER branch)
- CLAUDE.md "Production-hardened dispatch optimization protocol" (Catalog #270 Tier 1/2/3 umbrella)
- Catalogs honored: #117, #125, #126, #128, #143, #146, #151, #152, #157, #164, #166, #167, #170, #174, #180, #182, #187, #190, #193, #206, #215, #220, #221, #226, #229, #230, #239, #240, #243, #245, #248, #270, #272, #290, #292, #294, #296, #297, #300, #305, #309
- Sister Z6 deliberation: `sextet_council_z6_phase_2_consensus_20260516.md` (canonical T2 sextet template)
- Sister NSCS01 deliberation: `sextet_council_nscs01_phase_2_consensus_20260516.md` (concurrent sister subagent)
- NSCS03 design memo: `nscs03_end_to_end_balle_joint_codec_design_20260515.md` (referenced; not redrafted here)
- NSCS03 lift memo: `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md` (76 tests pass)
- NSCS03 operator-decision investigation: `nscs03_operator_decision_items_investigation_20260515.md` (Path A canonical $70-107 envelope)
- Lattice coherence audit: `coherence_audit_lattice_coordinate_assignment_20260516.md` (NSCS03 = outside-NeRV Rule #2)
- Wave 3 optimization: `wave_3_optimization_per_lattice_coherence_20260516.md` (Substitution Set B: NSCS03 → Frontier-pursuit #7)
- A-STACK composition: `a_stack_nscs01_02_03_composition_full_stack_design_20260516.md` (cross-decision dependency)
- K=13 schedule: `k_measurement_schedule_level_1_rebalanced_post_donoho_tanner_20260516.md` (post-Donoho-Tanner rebalanced budget)
