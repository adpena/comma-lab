---
name: sextet-council-nscs01-phase-2-consensus-20260516
description: |
  T2 sextet-pact council CONSENSUS deliberation for NSCS01 nullspace-split-renderer
  FRONTIER-PURSUIT substrate. Satisfies the substrate's L2-promotion preconditions
  per Catalog #240 (recipe-vs-trainer-state) + Catalog #220 (substrate L1+ scaffold
  operational mechanism) + design memo §10 reactivation criteria, following the
  Phase 1b _full_main lift commit landing 2026-05-16 per memo
  `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md` and
  the COHERENCE-AUDIT Substitution Set B authorization 2026-05-16 per
  `.omx/research/coherence_audit_lattice_coordinate_assignment_20260516.md`.

  Sextet pact 6-of-6: Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich +
  Contrarian + Assumption-Adversary. Verdict: PROCEED_WITH_REVISIONS — recipe
  REMAINS research_only=true / dispatch_enabled=false in same commit batch as
  this memo; Catalog #167 smoke-before-full at $1 Modal T4 with the
  head0-architecture disambiguator at SAME archive bytes becomes the explicit
  next gate; FULL dispatch + recipe flip waits on smoke-green AND
  disambiguator-decision per Revision #3 contingency.

  The NSCS01 paradigm is the SegNet last-frame-only nullspace exploit per
  `upstream/modules.py:108` `x[:, -1, ...]` slice (PV-2 verified). frame[0] is
  in SegNet's structural nullspace; PoseNet uses both frames. NSCS01 RENDERS
  both frames with split-head architecture (frame_0_head small/4-bit, only
  PoseNet gradient; frame_1_head large/8-bit, SegNet+PoseNet gradient).
  Sister D4 substrate exploits the same nullspace by DERIVING frame[0] from
  frame[1] via Wyner-Ziv — NSCS01 is structurally distinct (RENDER vs DERIVE).

  Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" the Phase 1b
  lift bound 14 layers (11 ADOPT canonical + 2 FORK unique + 1 REJECTED-divergence)
  into ONE ~600 LOC coherent packet per the PR95 paradigm. The 45 dedicated
  tests pass including the defining `TestNullspaceGradientProperty` end-to-end
  verification that `seg_term.backward()` leaves `frame_0_head.grad == 0`.

substrate_id: nscs01_nullspace_split_renderer
deliberation_id: sextet_council_nscs01_phase_2_consensus_20260516
topic: "NSCS01 Phase 2 sextet-pact council CONSENSUS deliberation per design memo §10 + COHERENCE-AUDIT Substitution Set B authorization"
council_tier: T2
council_attendees:
  - Shannon (LEAD)
  - Dykstra (CO-LEAD)
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
council_dissent:
  - member: Contrarian
    verbatim: "I push back on any framing that treats the Phase 1b _full_main lift as equivalent to empirical evidence that the nullspace-split paradigm produces ΔS lift at contest-CUDA archive byte savings. The trainer is genuinely lifted (45/45 tests pass; PR95 paradigm bound; the defining `TestNullspaceGradientProperty` end-to-end test proves `seg_term.backward()` leaves `frame_0_head.grad == 0`). But that test proves the MATHEMATICAL invariant of the exploit (the nullspace gradient routing IS structurally correct); it does NOT prove the exploit YIELDS ΔS lift when scorer-aware training drives `frame_0_head` to convergence on real contest video frames. The probe-disambiguator at `tools/probe_nscs01_head0_arch_disambiguator.py` is currently a STUB (~50 LOC; emits a probe plan; does NOT execute the 4 measurements named in its docstring: posenet_gradient_norm_frame0_vs_frame1 + segnet_frame0_perturbation_invariance + head0_cnn_vs_mlp_ablation + paired_cpu_cuda_exact_eval_for_promoted_candidate). The recipe `predicted_band: null` + `smoke_score_band: null` is HONEST — we have NO empirical anchor on this substrate yet. Per CLAUDE.md `predicted_delta: \"unranked until frame-0/frame-1 PoseNet gradient norms and head0 CNN-vs-MLP probe land\"` — that condition is NOT met. Flipping `dispatch_enabled=true` now spends $10-15 FULL Modal T4 on a substrate where the probe-disambiguator EXPLICITLY says do-not-rank-until-probe-emits-measured-component-deltas. Revision #1 (mandatory): execute the head0-arch probe at $1 Modal T4 with the 4 required measurements per `tools/probe_nscs01_head0_arch_disambiguator.py::build_probe_plan().required_measurements`. Revision #2 (mandatory): $1 smoke-before-full per Catalog #167 produces archive bytes + integration smoke; this is PAIRED with the head0 probe at same archive bytes. Revision #3 (mandatory contingency): if smoke-green AND probe shows frame_0 PoseNet gradient norm is non-trivial (i.e., frame_0_head receives real training signal and does NOT collapse to constant) AND the CNN-vs-MLP ablation indicates the chosen head0 architecture is the right capacity → only THEN flip recipe to dispatch_enabled=true. Otherwise DEFER-pending-Z6-style-redesign per design memo §10 reactivation criteria (NOT KILL per CLAUDE.md `Forbidden premature KILL`)."
  - member: Assumption-Adversary
    verbatim: "I challenge the SHARED ASSUMPTION operating across THIS council deliberation: that the nullspace structural property (frame[0] is in SegNet's nullspace per `upstream/modules.py:108` `x[:, -1, ...]` slice) DEMONSTRATES `the per-head bit-width compression strategy will yield ΔS lift at contest-CUDA archive byte savings`. The structural property IS HARD-EARNED (PV-2 verified; `TestNullspaceGradientProperty` proves the gradient routing invariant end-to-end). But that is a CONSTRAINT (SegNet contribution from frame[0] is mathematically EXACTLY ZERO); it is NOT a GUARANTEE the substrate will outperform A1 baseline. Three CARGO-CULTED implicit assumptions: (1) `frame_0_head at HEAD0_BITS=4` is sufficient capacity for PoseNet-only gradient training — if 4-bit collapses, the rate-savings axis vanishes and we get a 2-head renderer (HEAD0+HEAD1) where HEAD0 produces constant output (the auxiliary `lambda_pixel_0 * MSE(frame_0_pred, gt_frame_0)` keeps an anchor but is NOT the same as PoseNet semantic conditioning). (2) The split-head architecture does NOT compound pose error via the 2-frame curriculum (PoseNet sees the pair; if frame_0 is degraded, frame_1's pose-error compounds). (3) The predicted band in the design memo §9 is `[0.190, 0.205]` CUDA / `[0.185, 0.198]` CPU — this is at-or-near-A1-cluster (A1 ≈ 0.193 contest-CUDA medal-band leader). The HARD-EARNED reading is: NSCS01 is an ARCHITECTURAL REFACTOR within the existing rate-distortion polytope; it is NOT a substrate-class shift per Z1 ablation framework. The CARGO-CULTED reading is: 'the nullspace structural property IS the score-axis improvement mechanism'. Per Z1 within-class refinements yield bounded improvements; expected best case ΔS ≤ 0.005 vs A1 baseline (and could regress). The probe-disambiguator MUST run BEFORE FULL spend. The predicted_band-vibes risk per Catalog #296 is non-trivial; the recipe explicitly carries `predicted_band: null` + `smoke_score_band: null` which is the canonical operator-honest stance. Per the META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable: my mandatory hypothesis is — 'the nullspace structural property is NECESSARY but NOT SUFFICIENT for ΔS lift; the smoke + probe is the canonical empirical anchor that resolves the sufficiency question.' Verdict on the assumption-violation hypothesis: STRONG support from the recipe's own null-band declaration + probe-disambiguator's do-not-rank-until-probe-emits language. Required action: probe + smoke MUST run BEFORE any council consensus can flip recipe to dispatch_enabled=true."
council_assumption_adversary_verdict:
  - assumption: "Phase 1b _full_main lift (PR95-paradigm-bound; 45/45 tests pass; defining nullspace gradient property test passes) closes the engineering-completeness preconditions for L2 promotion"
    classification: HARD-EARNED
    rationale: "Catalog #229 premise verification (8 PVs all PASS); 45/45 dedicated tests pass; PR95-paradigm-bound per Catalog #187; canonical scorer-preprocess routing per Catalog #164 (via `tac.losses.scorer_loss_terms_btchw` low-level helper which still calls both scorers' `preprocess_input`); Catalog #205 inflate device select; Catalog #218 mini-batch reconstruct; Catalog #226 canonical auth-eval helper; Catalog #190 hardware substrate detect. The lift memo `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md` documents 14 layers of canonical-vs-unique decisions per Catalog #290. Defining `TestNullspaceGradientProperty::test_frame_0_head_receives_zero_segnet_gradient` end-to-end verifies the structural invariant the substrate's name claims."
  - assumption: "The nullspace structural property (frame[0] is in SegNet's mathematical nullspace via `x[:, -1, ...]` slice) yields ΔS lift at contest-CUDA archive byte savings"
    classification: CARGO-CULTED
    rationale: "The nullspace property IS hard-earned (PV-2 verified at `upstream/modules.py:108`); it is a NECESSARY but NOT SUFFICIENT condition for ΔS lift. The structural fact `∂(seg_term)/∂frame_0_head_params == 0` is correct (the `TestNullspaceGradientProperty` test verifies it); but converting this into archive byte savings requires (a) frame_0_head at HEAD0_BITS=4 has sufficient capacity for PoseNet-only training (not collapse-to-constant), (b) the auxiliary `lambda_pixel_0 * MSE(frame_0_pred, gt_frame_0)` does NOT dominate the PoseNet gradient (Yousfi-anchored concern), (c) the 2-frame pose curriculum does NOT compound error via degraded frame_0. The recipe HONESTLY declares `predicted_delta: \"unranked until ... probe land\"` + `predicted_band: null` + `smoke_score_band: null`. Per Z1 ablation framework + Catalog #227 Tier C density: NSCS01 is an ARCHITECTURAL REFACTOR (within-class), NOT a substrate-class shift (across-class). Expected band per design memo §9 [0.190, 0.205] CUDA = AT-OR-NEAR A1 cluster (0.193 medal-band leader). The empirical anchor (smoke + probe) IS the canonical sufficiency probe."
  - assumption: "The predicted ΔS band [0.190, 0.205] CUDA / [0.185, 0.198] CPU per design memo §9 holds under paired CPU/CUDA empirical anchor"
    classification: CARGO-CULTED
    rationale: "Per Catalog #296 (substrate predicted band has Dykstra-feasibility check): the design memo §9 derivation cites first-principles per CLAUDE.md `SegNet vs PoseNet importance — operating-point dependent` empirical receipts on PR106 + canonical R-D contest formula `25 * archive_bytes / 37_545_489` + first-principles upper bound on SegNet contribution from frame_1 + pose-rate trade. Dykstra-feasibility check is IMPLICIT (the rate-budget + segnet-budget + pose-budget convex intersection at A1 cluster operating point is well-known; design memo §13 cites NSCS01 as within-class refactor). The PROJECTION onto contest-CUDA at runtime is CARGO-CULTED because the per-head bit-width strategy has NEVER been empirically tested on contest video. Z3-G1 cargo-cult-prediction precedent applies (predicted [0.13, 0.16]; landed 0.19869 because distinguishing bytes were empty); the analogous risk for NSCS01 is `frame_0_head collapses to constant` (the auxiliary pixel-MSE keeps an anchor) which would make the per-head bit-width split structurally meaningless. The recipe's own `predicted_band: null` declaration IS the operator-honest CARGO-CULTED tag."
  - assumption: "Catalog #167 smoke-before-full at $1 Modal T4 + head0-architecture probe at SAME archive bytes IS sufficient empirical gate before recipe flip"
    classification: HARD-EARNED
    rationale: "Catalog #167 IS the canonical smoke-before-full pattern per CLAUDE.md `Production-hardened dispatch optimization protocol`; the $1 smoke validates rc=0 + archive bytes in expected band + integration smoke; paired with the head0 probe (executing the 4 measurements named at `tools/probe_nscs01_head0_arch_disambiguator.py::build_probe_plan().required_measurements`: posenet_gradient_norm_frame0_vs_frame1 + segnet_frame0_perturbation_invariance + head0_cnn_vs_mlp_ablation + paired_cpu_cuda_exact_eval_for_promoted_candidate) the council gets a SCORER-BEARING anchor that resolves the sufficiency question of the nullspace exploit. Per Catalog #270 Tier 1/2/3 umbrella: NSCS01 trainer satisfies all 3 tiers (per lift memo §`Canonical-vs-unique decision per layer` table — Tier-1 engineering all ADOPT canonical; Tier-2 hardware per recipe min_vram_gb=16 + min_smoke_gpu=T4 + video_input_strategy + pyav_decode_strategy; Tier-3 substrate per canonical scorer-loss helper routing + canonical inflate device + canonical auth-eval helper + recipe-vs-trainer-state consistency)."
  - assumption: "Per CLAUDE.md `Forbidden premature KILL`: even if smoke+probe shows nullspace exploit yields no ΔS lift, the verdict is DEFER not KILL"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md `KILL/FALSIFIED memory verdicts` non-negotiable + `Forbidden premature KILL without research exhaustion`: a single NSCS01-v1 configuration's failure (smoke-green-but-probe-shows-no-frame_0-gradient OR smoke-regression-from-A1-baseline) does NOT exhaust the nullspace-exploit paradigm. Reactivation criteria per design memo §10 include: (1) investigate whether `frame_0_head` was actually trained with sufficient PoseNet gradient (it might collapse to constant — try `lambda_pixel_0` ablation); (2) alternative `frame_0_head` architectures (CNN vs MLP per probe-disambiguator); (3) try lower `lambda_pixel_0` to let PoseNet drive frame_0 more; (4) investigate whether HEAD0_BITS=4 is too aggressive (try HEAD0_BITS=6). The verdict for one config failure is DEFERRED-pending-research, NOT KILLED. Sister Z6 v1 ego-conditioning surface DEFER precedent applies (Z6 sextet council 2026-05-16 verdict `measured-config-retired` + reactivation criteria preserved + Z6-v2/Z7-LSTM/Comma2k19-pretraining sister paths preserved)."
  - assumption: "NSCS01 dispatch at the predicted-band claim would be `frontier_breaking` mission-contribution per Catalog #300"
    classification: PARTIALLY HARD-EARNED + CARGO-CULTED
    rationale: "HARD-EARNED for the FRONTIER-PURSUIT horizon-class per Catalog #309 (NSCS01 lattice registered as `rule_2_nullspace_split_pr95_paradigm` + `horizon_class=frontier_pursuit` per `.omx/state/lattice_state.jsonl`). CARGO-CULTED for the `frontier_breaking` claim: predicted band [0.190, 0.205] CUDA / [0.185, 0.198] CPU is AT-OR-NEAR A1 cluster (A1 ≈ 0.193 medal-band leader); NSCS01 is an architectural refactor (within-class per Z1) not a substrate-class shift. Per Catalog #300 mission-alignment: the CURRENT smoke + probe dispatch is `frontier_protecting` (protects against false promotion of an architecture whose nullspace exploit may not materialize at contest-CUDA scorer); the POTENTIAL future FULL dispatch (conditional on Revision #3 PROCEED branch) MIGHT be `frontier_breaking` IF the per-head bit-width compression yields measurable ΔS ≤ A1 (most plausibly via the A-STACK composition with sister NSCS02+NSCS03 per `.omx/research/a_stack_nscs01_02_03_composition_full_stack_design_20260516.md` predicted [0.155, 0.175] CPU Dykstra-validated)."
council_decisions_recorded:
  - "Recipe REMAINS research_only=true / dispatch_enabled=false / smoke_only=true (NO change in same commit batch as this memo). Design memo §10 reactivation criteria + Catalog #240 recipe-vs-trainer-state precondition SATISFIED at L1-SCAFFOLD level (the trainer IS lifted; the recipe HONESTLY declares the substrate as research-only pending probe + smoke empirical anchor). The recipe flip to dispatch_enabled=true waits for Revision #2 + Revision #3 PROCEED branch firing."
  - "Revision #1 (mandatory before any FULL dispatch): execute the head0-architecture disambiguator probe at $1 Modal T4 first per `tools/probe_nscs01_head0_arch_disambiguator.py::build_probe_plan().required_measurements`. The probe MUST emit measured component deltas for: (a) frame-0 vs frame-1 PoseNet gradient norms on the same candidate batch; (b) SegNet invariance under frame-0 perturbation; (c) head0 CNN-vs-MLP no-train or short-smoke ablation. The probe artifact MUST emit per Catalog #221 fail-closed structure (`evidence_grade=head0_probe_diagnostic_not_contest_cpu_or_cuda` + `score_claim=false` + `promotion_eligible=false`)."
  - "Revision #2 (mandatory before FULL): Catalog #167 smoke-before-full at $1 Modal T4 PAIRED with the head0 probe at SAME archive bytes (so the council can see whether scorer-aware training rescues frame_0_head from collapse-to-constant that a no-train probe alone might suggest). Smoke acceptance criteria: rc=0 + archive bytes in [70, 120] KB per design memo §5 default budget + integration smoke validates per Catalog #243 + Catalog #166 Modal HEAD-parity ledger captures dispatch + Catalog #245 modal_call_id_ledger registers call_id BEFORE submit per Catalog #143."
  - "Revision #3 (mandatory contingency): IF smoke-green AND probe shows frame_0 PoseNet gradient norm is non-trivial (>=0.1 of frame_1 gradient norm baseline) AND CNN-vs-MLP ablation indicates the chosen head0 architecture has sufficient capacity → only THEN flip recipe `research_only: false` + `dispatch_enabled: true` in follow-on commit batch + dispatch FULL ($10 Modal T4 100ep) + paired CPU/CUDA auth eval per Catalog #226 + posterior_update_locked per Catalog #128 + cathedral autopilot wire-in per Catalog #125 hook #4. Total path cost: $2.50 NEXT + $11.50 conditional = $14.00 GRAND TOTAL."
  - "Revision #3 contingency (mandatory): IF smoke-green AND probe shows frame_0 PoseNet gradient norm is trivial (frame_0_head collapse-to-constant) OR CNN-vs-MLP ablation indicates insufficient capacity → mark NSCS01-v1 configuration `measured-config-retired` per CLAUDE.md `KILL = LAST RESORT` + DEFER NSCS01-v2 redesign per design memo §10 reactivation criteria (HEAD0_BITS=6 instead of 4 / alternative head0 CNN-vs-MLP architecture / lower lambda_pixel_0 / pretraining stage). Recipe stays research_only=true / dispatch_enabled=false; lane stays L1; `_full_main` keeps its current implementation (no revert). No KILL of paradigm; the SPECIFIC NSCS01-v1 head0 capacity surface is the retired config."
  - "Sextet-pact CONSENSUS verdict: PROCEED_WITH_REVISIONS at 6-of-6 quorum (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary all PROCEED_WITH_REVISIONS). Dissent surfaced verbatim per Catalog #300 (Contrarian + Assumption-Adversary both registered substantive revisions; consensus is on the REVISION SET, not on unconditional approval). Per CLAUDE.md `Council conduct — non-negotiable`: this is NOT a lazy consensus; the revisions are binding."
  - "Cross-decision dependency: NSCS01 results inform A-STACK composition (Rule #3 Dykstra-feasibility stack composition) per `.omx/research/a_stack_nscs01_02_03_composition_full_stack_design_20260516.md`. The A-STACK predicted band [0.155, 0.175] CPU Dykstra-validated assumes NSCS01 contributes architectural-refactor axis (orthogonal to NSCS02 minimalism-bytecount + NSCS03 entropy-coding). If NSCS01-v1 smoke + probe shows null-result, the A-STACK substitutes with NSCS01-v2 redesign OR a third Rule #2 sister substrate. NSCS03 sister council (concurrent) is owned by sister subagent per Catalog #230 ownership map."
  - "30-day deferred-substrate retrospective scheduled 2026-06-16 for the NSCS01-v1 head0 capacity surface (per Catalog #300 mission-alignment consequence 3) — if Revision #3 DEFER contingency fires AND NSCS01-v2 redesign has not landed empirical anchor at +30d, the operator reviews whether to escalate to T3 grand council OR DEFER indefinitely per design memo §10 op-routables (HEAD0_BITS=6 / alternative head0 CNN-vs-MLP / Comma2k19-pretrained head0 sister paths preserved)."
  - "Predicted cost band: $1 smoke (Catalog #167) + $1 head0 probe (Modal T4 ~20 min paired with smoke) + $0.50 GHA CPU paired for probe = $2.50 NEXT immediate envelope per pre-pre-pre-anchor cost-band posterior; conditional FULL + paired adds $11.50 if Revision #3 PROCEED branch fires. Total envelope $14.00 matching the canonical Z6 exemplar envelope."
  - "Per Catalog #300 mission-alignment + HORIZON-CLASS Consequence 5: operator-frontier-override NOT INVOKED for this deliberation; the council is operating within the standard sextet-pact procedure with revisions; NO time-critical innovation pressure pushes for unconditional approval (NO active leaderboard race; PR101 medal-band leader at 0.193 stable; Substitution Set B context per `.omx/research/wave_3_optimization_per_lattice_coherence_20260516.md` authorizes the council deliberation, not a race-mode override)."
deferred_substrate_retrospective_due_utc: "2026-06-16T11:43:00Z"
deferred_substrate_id: nscs01_nullspace_split_renderer_v1_head0_capacity_surface
related_deliberation_ids:
  - sextet_council_z6_phase_2_consensus_20260516
  - grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516
related_design_memos:
  - .omx/research/nscs01_nullspace_split_renderer_design_20260515.md
  - .omx/research/a_stack_nscs01_02_03_composition_full_stack_design_20260516.md
  - .omx/research/coherence_audit_lattice_coordinate_assignment_20260516.md
  - .omx/research/wave_3_optimization_per_lattice_coherence_20260516.md
event_type: dispatched
parent_id_or_session: nscs01_phase_2_sextet_council_20260516
memory_path: .omx/research/sextet_council_nscs01_phase_2_consensus_20260516.md
lane: lane_nscs01_phase_2_sextet_council_20260516
---

# NSCS01 Phase 2 Sextet-Pact Council CONSENSUS Deliberation

**Date:** 2026-05-16
**Lane:** `lane_nscs01_phase_2_sextet_council_20260516`
**Council tier:** **T2** (sextet pact 6-of-6 per Catalog #300 v2 quorum)
**Mission-alignment classification:** `frontier_protecting` (smoke + probe gate is `frontier_protecting`; potentially unlocks FULL FRONTIER-PURSUIT dispatch if Revision #3 PROCEED branch fires)
**Operator override:** NOT INVOKED
**Quorum:** **MET** — sextet pact 6/6 (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary)
**Verdict:** **PROCEED_WITH_REVISIONS** — recipe REMAINS research_only=true / dispatch_enabled=false; 3 revisions enumerated; Catalog #167 smoke-before-full + head0-architecture disambiguator at SAME archive bytes IS the explicit next gate
**Deferred-substrate retrospective scheduled:** **2026-06-16T11:43:00Z** (NSCS01-v1 head0 capacity surface; per Catalog #300 consequence 3)

---

## 0. Premise verifications per Catalog #229 (8 pre-deliberation anchors)

1. **PV-1**: `experiments/train_substrate_nscs01_nullspace_split_renderer.py::_full_main` is implemented (NOT NotImplementedError) — verified via `grep -n "NotImplementedError" experiments/train_substrate_nscs01_nullspace_split_renderer.py` returns ONLY a docstring mention at line 11 (no actual raise in `_full_main` body).
2. **PV-2**: `upstream/modules.py:108` contains the SegNet `x[:, -1, ...]` last-frame slice — verified via grep returns `108:0:, -1, ...] # Use only last frame`. The nullspace structural property the substrate exploits IS empirically present in the upstream contest scorer code.
3. **PV-3**: 45 dedicated tests collected at `src/tac/substrates/nscs01_nullspace_split_renderer/tests/test_nscs01_substrate.py` (was 42 before Phase 1b lift; +3 including the defining `TestNullspaceGradientProperty::test_frame_0_head_receives_zero_segnet_gradient` end-to-end test).
4. **PV-4**: Recipe at `.omx/operator_authorize_recipes/substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch.yaml` carries `research_only: true` + `dispatch_enabled: false` + `smoke_only: true` + explicit `dispatch_blockers: [phase_2_council_approval_required_to_lift_full_main_NotImplementedError, paired_cpu_cuda_tier_c_anchor_required_for_l2_promotion]`.
5. **PV-5**: Phase 1b lift memo at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md` documents 14-layer canonical-vs-unique decision table (11 ADOPT canonical + 2 FORK unique + 1 REJECTED-divergence) + 6-hook wire-in declaration + 9-dim checklist evidence + Catalog #220 violations = 0.
6. **PV-6**: NSCS01 registered in lattice state at `.omx/state/lattice_state.jsonl` as `lattice_node_id=nscs01 / lattice_rule=rule_2_nullspace_split_pr95_paradigm / horizon_class=frontier_pursuit / status=lifted_pending_council / paradigm_vs_implementation_classification=paradigm_intact` per the COHERENCE-AUDIT lattice coordinate assignment 2026-05-16.
7. **PV-7**: Sister A-STACK lattice node (`a_stack`) registered as `lattice_rule=rule_3_dykstra_stack_composition / horizon_class=frontier_pursuit / status=lifted_pending_council / notes="3-substrate composition NSCS01 x NSCS02 x NSCS03 per T4 SYMPOSIUM V2/V6 + Dykstra-feasibility check. Predicted band [0.155, 0.175] [contest-CPU; Dykstra-validated convex-hull lower envelope]. OUTSIDE-NeRV."`.
8. **PV-8**: NSCS01 probe-disambiguator at `tools/probe_nscs01_head0_arch_disambiguator.py` EXISTS as canonical stub (~50 LOC; emits `build_probe_plan()` with `score_claim=false / promotion_eligible=false` + 4 required measurements: posenet_gradient_norm_frame0_vs_frame1 + segnet_frame0_perturbation_invariance + head0_cnn_vs_mlp_ablation + paired_cpu_cuda_exact_eval_for_promoted_candidate). Dispatch rule: "Do not rank NSCS01 predicted_delta until this probe emits measured component deltas and a paired-axis follow-up plan."

All 8 PVs PASS. No regression from parent-prompt assertions.

---

## 1. Council attendance + quorum

**Sextet pact (6-of-6 quorum required at T2 per Catalog #300 v2):**

| Seat | Role | Operating-within assumption (Catalog #292 per-deliberation discipline) |
|---|---|---|
| **Shannon** (LEAD) | Information-theory grounding | "The R(D) lower bound for NSCS01 archive bytes ~70-120 KB at contest-CPU operating point is dominated by rate term `25 * 95_000 / 37_545_489 = 0.063`; the distortion term `100 × seg + sqrt(10×pose)` at A1 cluster operating point is ~0.13 (seg≈0.067 + pose≈0.018 + rate≈0.103 = 0.193 ≈ A1 frontier per CLAUDE.md SegNet vs PoseNet empirical receipts). The nullspace exploit removes SegNet contribution from frame[0] but does NOT change frame[1] SegNet contribution; the rate-savings axis depends ENTIRELY on whether per-head bit-width compression (HEAD0_BITS=4 vs HEAD1_BITS=8) yields a measurable archive-byte reduction at scorer-aware training. The information-theoretic question: does `H(frame_0_pred | pose_constraints_only) < H(frame_1_pred | seg+pose_constraints)` enough to justify the per-head bit-width split? Empirical question pending probe + smoke." HARD-EARNED at theorem level + CARGO-CULTED at Pact application level. |
| **Dykstra** (CO-LEAD) | Convex-feasibility intersection check | "Design memo §9 NSCS01 polytope: rate∈[0.060,0.070] (rate budget per ~95KB archive) ∩ seg∈[0.060,0.075] (frame_1 SegNet contribution unchanged from A1) ∩ pose∈[0.000034,0.000050] (PoseNet on rendered pair). Convex-intersection NON-EMPTY at the A1 cluster operating point. The Dykstra-feasibility VERDICT is FEASIBLE at the mathematical construction level (the design memo §9 derivation is sound). The empirical question is whether the polytope PROJECTION onto contest-CUDA scoring lands in [0.190, 0.205] CUDA or somewhere else. Z3-G1 cargo-cult-prediction precedent (predicted [0.13, 0.16]; landed 0.19869 because distinguishing bytes were empty) is the canonical analogous risk for NSCS01 (`frame_0_head` collapse-to-constant would make the per-head split structurally meaningless). Per Catalog #296 the cargo-cult-prediction risk is non-trivial; the recipe's `predicted_band: null` IS the operator-honest CARGO-CULTED tag." HARD-EARNED at convex-feasibility level + CARGO-CULTED at runtime-projection level. |
| **Yousfi** | Steganalysis + scorer design context | "SegNet's stride-2 EfficientNet-B2 stem at preprocess_input `x[:, -1, ...]` slice (PV-2 verified) IS the structural anchor for the nullspace exploit. The mathematical fact `∂(seg_term)/∂frame_0 == 0` is correct. The empirical question: does frame_0_head trained ONLY against PoseNet (no SegNet supervision) produce frames where PoseNet's FastViT-T12 12-channel YUV6 input can still extract meaningful pose from the pair? PoseNet uses BOTH frames; if frame_0_head produces low-fidelity output, PoseNet's pose extraction degrades (the `lambda_pixel_0 * MSE(frame_0_pred, gt_frame_0)` auxiliary anchor exists but its weight choice is a hyperparameter that needs probe-disambiguator empirical anchor). The 7-15% byte-savings hypothesis from design memo §2 is unranked until the head0 probe emits measured PoseNet gradient norms on frame_0 vs frame_1." HARD-EARNED at scorer-architecture level + CARGO-CULTED at frame-0-PoseNet-conditioning capacity. |
| **Fridrich** | Inverse steganalysis + CNN blind spots | "UNIWARD-style coverage analysis: errors in textured regions are undetectable to CNN steganalysis (CLAUDE.md `Fridrich inverse steganalysis`). NSCS01's frame_0_head producing coarser detail (4-bit packed) is structurally similar to UNIWARD's textured-region-error preference — if PoseNet's FastViT blind spots include the regions where frame_0 lossiness concentrates, the per-head bit-width split COULD yield ΔS lift. The empirical question is whether PoseNet's FastViT-T12 stride-2 stem has textured-region blind spots that align with `frame_0_head at HEAD0_BITS=4` lossiness. Sister D4 substrate (`d4_wyner_ziv_frame_0`) is the structurally-distinct alternative (DERIVE vs RENDER); NSCS01 + D4 composition is preserved as a future option per design memo §8. The probe must measure frame_0-perturbation-invariance of SegNet (PV-8 measurement #2) to validate the nullspace property holds under realistic per-pair latent perturbations." HARD-EARNED at UNIWARD-principle level + CARGO-CULTED at NSCS01-v1-capacity application. |
| **Contrarian** | Challenge weak arguments + veto lazy consensus | "I push back on any framing that treats Phase 1b _full_main lift as equivalent to empirical evidence the nullspace-split paradigm produces ΔS lift at contest-CUDA archive byte savings..." [see council_dissent.Contrarian.verbatim above for the full position]. The Phase 1b lift is engineering-discipline achievement (45/45 tests pass; PR95 paradigm bound; nullspace gradient property test passes end-to-end); it is NOT empirical-evidence-of-paradigm-validity. The probe-disambiguator stub explicitly says do-not-rank-until-probe-emits. The 3 revisions are mandatory. HARD-EARNED at engineering-risk level. |
| **Assumption-Adversary** (NEW sextet seat per CLAUDE.md "Council conduct" Fix 7) | Challenge the FRAMING all arguments share | "I challenge the SHARED ASSUMPTION operating across THIS council deliberation: that the nullspace structural property DEMONSTRATES the per-head bit-width compression strategy will yield ΔS lift at contest-CUDA archive byte savings..." [see council_dissent.Assumption-Adversary.verbatim above for the full position]. The structural property IS hard-earned; the per-head bit-width application is CARGO-CULTED until empirical anchor. The predicted band [0.190, 0.205] is AT-OR-NEAR A1 cluster (within-class refactor per Z1). HARD-EARNED at nullspace-mathematics level; CARGO-CULTED at per-head-bit-width capacity application. |

**Quorum:** 6-of-6 sextet pact present. ALL 6 seats cast PROCEED_WITH_REVISIONS. NO seat voted PROCEED unconditionally; NO seat voted DEFER_PENDING_EVIDENCE / REFUSE / ESCALATE; the consensus IS on the REVISION SET, not on unconditional approval. Per CLAUDE.md "Council conduct": this is a CONSENSUS with substantive minority opinions surfaced verbatim, NOT a lazy 6-of-6-rubber-stamp.

---

## 2. Vote tally (per Catalog #300 maximum signal preservation rule)

| Seat | Vote | Explicit reasoning |
|---|---|---|
| Shannon | PROCEED_WITH_REVISIONS | "R(D) lower bound IS the right floor; per-head bit-width information-theoretic question requires empirical anchor; smoke + probe is canonical." |
| Dykstra | PROCEED_WITH_REVISIONS | "Polytope FEASIBLE at math level; runtime projection onto contest-CUDA UNKNOWN; cargo-cult-prediction risk per Catalog #296 non-trivial." |
| Yousfi | PROCEED_WITH_REVISIONS | "Nullspace property structural anchor HARD-EARNED; frame_0-PoseNet-conditioning capacity application UNRANKED; head0 probe IS the canonical empirical probe." |
| Fridrich | PROCEED_WITH_REVISIONS | "UNIWARD principle aligns; NSCS01-v1 capacity application unproven; head0 probe + smoke required." |
| Contrarian | PROCEED_WITH_REVISIONS | "3 revisions mandatory; recipe flip premature without probe + smoke; probe-disambiguator stub do-not-rank language binding." |
| Assumption-Adversary | PROCEED_WITH_REVISIONS | "Hard-earned-vs-cargo-culted classification: shared assumption is CARGO-CULTED at per-head bit-width application level; empirical anchor required." |

**Tally:** 6 PROCEED_WITH_REVISIONS / 0 PROCEED / 0 DEFER / 0 REFUSE / 0 ESCALATE / 0 abstain / 0 recused.

**Consensus achieved:** 6-of-6 on PROCEED_WITH_REVISIONS verdict + 6-of-6 on the 3-revision set (Revision #1 mandatory head0-architecture disambiguator + Revision #2 mandatory smoke-before-full paired at same archive bytes + Revision #3 mandatory contingency branching on probe outcome).

---

## 3. Predicted ΔS band + Dykstra-feasibility verdict per Catalog #296

**Per design memo §9 (referenced; NOT redrafted here):**

- **NSCS01 CPU [contest-CPU Linux x86_64 GHA]**: `[0.185, 0.198]` `[prediction; Dykstra-feasibility-validated; HIGH VARIANCE pending paired CPU/CUDA empirical anchor + head0-architecture probe]`
- **NSCS01 CUDA [contest-CUDA T4]**: `[0.190, 0.205]` `[prediction; sister precedent A1 cluster operating point; CUDA-CPU gap ~-0.005 to -0.010 per PR102 + Z3 v2 empirical anchor]`
- **Dykstra-feasibility VERDICT**: FEASIBLE at construction level. Polytope vertices: rate ∈ [0.060, 0.070] (rate budget per ~95KB archive); seg ∈ [0.060, 0.075] (frame_1 SegNet contribution unchanged from A1 baseline ≈0.067); pose ∈ [0.000034, 0.000050] (PoseNet on rendered pair). Convex-intersection non-empty.

**Per Catalog #296 + the design-memo derivation + the recipe `predicted_band: null` honest tag**: the predicted band is BOUNDED AT-OR-NEAR A1 cluster (0.193 PR101 medal-band leader). NSCS01 is an ARCHITECTURAL REFACTOR within the A1 rate-distortion polytope; it is NOT a substrate-class shift per Z1 ablation framework. Expected best case ΔS ≤ 0.005 vs A1 baseline (and could regress if frame_0_head collapses to constant). The probe + smoke gate IS designed to resolve this band uncertainty at $2 cost.

**Per CLAUDE.md "Apples-to-apples evidence discipline"**: ALL bands tagged `[prediction]`; NO promotion language until paired CPU/CUDA empirical anchor.

---

## 4. Horizon-class declaration per Catalog #309

**NSCS01 horizon-class: `frontier_pursuit`** per `.omx/state/lattice_state.jsonl` registered classification (`lattice_rule=rule_2_nullspace_split_pr95_paradigm / horizon_class=frontier_pursuit`). NSCS01 sits in the FRONTIER-PURSUIT band [0.120, 0.180] per HORIZON-CLASS standing directive 2026-05-16 Consequence 2 — but the empirical band may land closer to PLATEAU-ADJACENT [0.180, 0.200] (within-class refactor) given the Z1 ablation framework risk. The smoke + probe gate IS designed to resolve this horizon-class uncertainty.

A-STACK composition (Rule #3) with NSCS01+NSCS02+NSCS03 is the FRONTIER-PURSUIT band [0.155, 0.175] per `.omx/research/a_stack_nscs01_02_03_composition_full_stack_design_20260516.md` Dykstra-validated convex-hull lower envelope. NSCS01's contribution to A-STACK depends on its individual ΔS resolution.

---

## 5. Mission contribution prediction per Catalog #300

**`council_predicted_mission_contribution: frontier_protecting`** at the CURRENT smoke + probe dispatch (protects against false promotion of an architecture whose nullspace exploit may collapse to within-class refactor); the POTENTIAL FUTURE FULL dispatch (conditional on Revision #3 PROCEED branch firing) MIGHT be `frontier_breaking` IF the per-head bit-width compression yields measurable ΔS < A1 baseline AND if A-STACK composition realizes the predicted [0.155, 0.175] CPU band.

Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 4 (frontier-breaking moves DOMINATE rigor budget when leaderboard moves): NO active leaderboard race detected (PR101 medal-band leader at 0.193 stable for last 24h+); standard sextet-pact procedure applies; operator-frontier-override NOT INVOKED.

---

## 6. The 3 mandatory revisions (binding)

**Revision #1: head0-architecture disambiguator probe at $1 Modal T4 BEFORE any FULL dispatch.**

Probe acceptance criteria per `tools/probe_nscs01_head0_arch_disambiguator.py::build_probe_plan().required_measurements`:
- `posenet_gradient_norm_frame0_vs_frame1` measured on the same candidate batch
- `segnet_frame0_perturbation_invariance` validates the nullspace property holds at realistic per-pair latent perturbations
- `head0_cnn_vs_mlp_ablation` short-smoke comparison of frame_0_head architecture variants
- `paired_cpu_cuda_exact_eval_for_promoted_candidate` paired-axis follow-up plan declared
- Probe artifact MUST emit per Catalog #221 fail-closed structure (`evidence_grade=head0_probe_diagnostic_not_contest_cpu_or_cuda` + `score_claim=false` + `promotion_eligible=false`)

Note: the current stub at `tools/probe_nscs01_head0_arch_disambiguator.py` (~50 LOC) emits the probe plan but does NOT execute the measurements. Revision #1 REQUIRES executing the measurements (likely via a sister-subagent that extends the stub OR a separate probe-execution wrapper).

**Revision #2: Catalog #167 smoke-before-full at $1 Modal T4 PAIRED with the head0 probe at SAME archive bytes.**

Smoke acceptance criteria:
- `rc=0` from `experiments/train_substrate_nscs01_nullspace_split_renderer.py --smoke` (or `--profile smoke` equivalent) invocation
- Archive bytes in `[70, 120]` KB per design memo §5 default budget
- Integration smoke validates per Catalog #243 local pre-deploy harness
- Catalog #166 Modal HEAD-parity ledger captures dispatch
- Catalog #245 modal_call_id_ledger registers the call_id BEFORE submit per Catalog #143
- Catalog #167 sister wrapper `tools/run_modal_smoke_before_full.py` orchestrates
- Smoke + probe MUST run at SAME archive bytes so the council can see whether scorer-aware training rescues frame_0_head from collapse-to-constant that a no-train probe alone might suggest

**Revision #3: contingency branching on probe + smoke outcome.**

**PROCEED branch** (smoke-green AND probe shows frame_0 PoseNet gradient norm >= 0.1 of frame_1 baseline AND CNN-vs-MLP ablation indicates sufficient head0 capacity):
1. Flip recipe `research_only: false` + `dispatch_enabled: true` + `smoke_only: false` in follow-on commit batch
2. Dispatch FULL $10 Modal T4 (100 epoch)
3. Paired CPU/CUDA auth eval per Catalog #226 + #190 ($1.50)
4. Total envelope $14.00 total
5. Cathedral autopilot wire-in per Catalog #125 hook #4
6. Posterior update per Catalog #128 + #131 + #245
7. Lane marks: `impl_complete=true / strict_preflight=true / real_archive_empirical=true / contest_cuda=true / contest_cpu=true (post-paired) / three_clean_review=true (post-sextet+probe-WIN) / memory_entry=true / deploy_runbook=true` → L3 candidate

**DEFER branch** (smoke-green AND probe shows frame_0 collapse-to-constant OR CNN-vs-MLP ablation indicates insufficient capacity):
1. Mark NSCS01-v1 head0 capacity surface `measured-config-retired` per CLAUDE.md "KILL = LAST RESORT"
2. Recipe STAYS `research_only: true` + `dispatch_enabled: false` (NO change from current)
3. `_full_main` keeps its current implementation (NO revert — the lift is permanent engineering value)
4. NO KILL of NSCS01 paradigm
5. Reactivation criteria per design memo §10:
   - HEAD0_BITS=6 instead of 4 (less aggressive quantization)
   - Alternative head0 architecture (CNN vs MLP per probe-disambiguator measurement #3)
   - Lower `lambda_pixel_0` to let PoseNet drive frame_0 more
   - Sister D4 composition (D4 derives frame[0] from frame[1] via Wyner-Ziv; NSCS01-v2 could use D4-derived frame[0] + NSCS01 frame_1_head)
   - Comma2k19-pretrained head0 sister
6. 30-day retrospective scheduled 2026-06-16 per Catalog #300 mission-alignment consequence 3
7. A-STACK composition substitutes NSCS01-v2 redesign OR a third Rule #2 sister substrate

---

## 7. Cross-decision dependency: NSCS01 results inform A-STACK composition (Rule #3)

Per `.omx/research/a_stack_nscs01_02_03_composition_full_stack_design_20260516.md`:

- **IF NSCS01-v1 smoke + probe shows null-result** (Revision #3 DEFER branch fires): A-STACK composition substitutes with NSCS01-v2 redesign per the 5 reactivation criteria OR a third Rule #2 sister substrate. The A-STACK predicted band [0.155, 0.175] CPU shifts based on the substitute's individual contribution.
- **IF NSCS01-v1 smoke + probe shows PROCEED branch fires AND FULL dispatch lands ΔS at predicted band**: A-STACK composition NSCS01 + NSCS02 + NSCS03 becomes the primary Level-2 frontier candidate at the FRONTIER-PURSUIT slot. Sister NSCS03 council (concurrent) is owned by sister subagent per Catalog #230; the joint A-STACK dispatch waits on BOTH NSCS01 + NSCS03 sextet-pact CONSENSUS verdicts.

---

## 8. Predicted cost band (per Catalog #175/#177 posterior + Catalog #270 Tier 2 hardware correctness)

| Stage | Cost | Notes |
|---|---:|---|
| THIS deliberation (memo + checkpoint) | $0 | Editor + 4 checkpoints per Catalog #206 |
| Revision #1: head0-architecture disambiguator probe | $1 | Modal T4 ~20 min |
| Revision #2: $1 smoke-before-full (Catalog #167) paired at same archive bytes | $1 | Modal T4 ~20 min (paired with probe) |
| Revision #1+#2 GHA CPU paired for probe | $0.50 | GHA CPU eval; verifies frame_0_head behavior on contest-CPU axis |
| **TOTAL Revision #1+#2 envelope** | **$2.50** | NEXT immediate action set; gated by operator-authorize per Catalog #199 |
| Revision #3 PROCEED FULL dispatch | $10 | Conditional on probe-WIN + smoke-green |
| Revision #3 PROCEED paired CPU/CUDA auth eval | $1.50 | Per Catalog #226 + CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" |
| **TOTAL conditional Revision #3 envelope** | **$11.50** | Conditional on Revision #2 disambiguator outcome |
| **GRAND TOTAL envelope** | **$14.00** | $2.50 NEXT + $11.50 conditional (matches Z6 exemplar envelope) |

Per Catalog #270 dispatch optimization protocol Tier 1/2/3: NSCS01 trainer + recipe + remote driver all satisfy the umbrella per Phase 1b lift memo §`Canonical-vs-unique decision per layer` table (Tier-1 engineering all ADOPT canonical; Tier-2 hardware per recipe min_vram_gb=16 + min_smoke_gpu=T4 + video_input_strategy + pyav_decode_strategy; Tier-3 substrate per canonical scorer-loss helper + canonical inflate device + canonical auth-eval helper + recipe-vs-trainer-state consistency).

---

## 9. 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: post-smoke (Revision #1+#2 lands), register `tac.sensitivity_map.nscs01_nullspace_split_renderer_v1` per per-head parameter count + per-head gradient norm. Currently `not_applicable_with_rationale` (L1 SCAFFOLD has no operational sensitivity surface until smoke).
2. **Pareto constraint**: post-smoke, register `tac.pareto.nscs01_v1` with `per_head_bit_width_v1 ∩ rate_distortion ∩ scorer_polytope`. Currently planned.
3. **Bit-allocator hook**: per-head bit-width split (HEAD0_BITS=4 vs HEAD1_BITS=8) IS the bit-allocator contribution — frame_0 gets aggressive quantization because PoseNet tolerates it (HARD-EARNED at neuroscience level; CARGO-CULTED at empirical anchor pending). ACTIVE in design memo.
4. **Cathedral autopilot dispatch hook**: recipe at `.omx/operator_authorize_recipes/substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch.yaml`; gated by Catalog #167 smoke-before-full per THIS revision set. ACTIVE.
5. **Continual-learning posterior**: post-smoke + post-probe, fires `posterior_update_locked_from_auth_eval_json` per Catalog #128. Currently planned.
6. **Probe-disambiguator**: `tools/probe_nscs01_head0_arch_disambiguator.py` (EXISTS as canonical stub per PV-8; Revision #1 mandates executing the 4 measurements at SAME archive bytes as the $1 smoke). ACTIVE per PV-8.

---

## 10. Cargo-cult audit per assumption (Catalog #303 addendum + Catalog #292 per-deliberation discipline)

Per design memo §16 (sister-class) PLUS 6 NEW deliberation-specific assumptions:

| # | NEW assumption | Classification | Rationale |
|---|---|---|---|
| 1 | Phase 1b _full_main lift CONSTITUTES empirical proof of NSCS01 paradigm-on-contest-scorer | **CARGO-CULTED** | Phase 1b lift IS engineering-discipline achievement; NOT empirical-evidence-of-paradigm-validity. The `TestNullspaceGradientProperty` test verifies the MATHEMATICAL invariant but does NOT prove ΔS lift at contest-CUDA scorer. |
| 2 | 6-of-6 sextet consensus on Phase 1b lift quality → 6-of-6 consensus on recipe flip to dispatch_enabled=true | **CARGO-CULTED** | Different gates; conflating engineering quality with empirical validity is the conflation Contrarian veto guards against. Sister Z6 council 2026-05-16 precedent applies. |
| 3 | NSCS01-v1 frame_0_head at HEAD0_BITS=4 has sufficient capacity for PoseNet-only training | **CARGO-CULTED** | 4-bit packed weights for ~30K params is design-memo-spec; recipe `predicted_band: null` IS the operator-honest CARGO-CULTED tag. Probe must measure. |
| 4 | The `lambda_pixel_0 * MSE(frame_0_pred, gt_frame_0)` auxiliary anchor prevents frame_0_head collapse-to-constant | **PARTIALLY HARD-EARNED + CARGO-CULTED** | HARD-EARNED at the MSE-as-anchor level (the auxiliary IS structurally an anchor); CARGO-CULTED at the choice of `lambda_pixel_0` weight (hyperparameter unranked; design memo §10 reactivation criterion #3 explicitly says "try lower lambda_pixel_0 to let pose loss drive frame_0 more"). |
| 5 | Catalog #167 smoke-before-full at $1 Modal T4 + head0 probe at SAME archive bytes IS sufficient empirical gate | **HARD-EARNED** | Sister Z6 council precedent (Z6 sextet PROCEED_WITH_REVISIONS 2026-05-16 with identical $1 smoke + identity-predictor disambiguator pattern) shows scorer-bearing $2-3 CPU probe IS the canonical empirical disambiguator at this cost tier. |
| 6 | The DEFER branch (Revision #3) preserves paradigm AND lifts engineering value | **HARD-EARNED** | Per CLAUDE.md "Forbidden premature KILL": `measured-config-retired` ≠ KILL; reactivation criteria documented in design memo §10 (HEAD0_BITS=6 / alternative head0 CNN-vs-MLP / lower lambda_pixel_0 / D4 composition / Comma2k19-pretrained sister); lane stays L1 SCAFFOLD with implemented `_full_main`; sister substrate paths preserved. |

**Summary classification across 6 deliberation-specific assumptions:** 1 HARD-EARNED for the smoke gate + 1 HARD-EARNED for the DEFER discipline + 1 PARTIALLY HARD-EARNED + CARGO-CULTED for the auxiliary anchor + 3 CARGO-CULTED for the per-head capacity application + the lift-vs-paradigm-validity conflation + the sextet-quality-vs-recipe-flip conflation.

The CARGO-CULTED items concentrate around `NSCS01-v1 specific per-head bit-width capacity will transfer from neuroscience-level nullspace property to contest-CUDA archive byte savings` — the empirical anchor sequence (Revision #1+#2+#3) is the canonical unwind path.

---

## 11. 9-dimension success checklist evidence per Catalog #294

See `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md` §`9-dimension success checklist evidence` for the canonical 9-dim table covering NSCS01 substrate-level evidence (Phase 1b lift). THIS deliberation extends the table with the per-revision empirical anchor commitments:

- **(4) Rigor**: 8 PVs pre-deliberation + 1 probe-disambiguator stub artifact consumed + sextet-pact 6-of-6 consensus + per-member assumption-statements per Catalog #292 + 6 deliberation-specific assumptions classified per Catalog #303
- **(9) Optimal minimal contest score**: $14.00 total envelope to resolve `[0.190, 0.205]` CUDA / `[0.185, 0.198]` CPU predicted band uncertainty per design memo §9 Dykstra-feasibility; sister Z6 PROCEED_WITH_REVISIONS precedent shows scorer-bearing probe + smoke IS canonical empirical anchor at this cost tier

---

## 12. Observability surface per Catalog #305

The 6-facet observability surface for THIS council deliberation:

1. **Inspectable per layer**: 6 council positions surfaced verbatim with operating-within assumptions per Catalog #292 (per-member readable).
2. **Decomposable per signal**: vote tally per seat + dissent verbatim + assumption-adversary verdicts per assumption (6 verdicts; HARD-EARNED vs CARGO-CULTED classification).
3. **Diff-able across runs**: this memo's canonical-helper-anchor at `tac.council_continual_learning.append_council_anchor(record)` (planned post-commit) appends to fcntl-locked JSONL store; future deliberations can `query_anchors_by_topic("NSCS01")` to see position evolution.
4. **Queryable post-hoc**: structured frontmatter per Catalog #300 v2 (council_tier, council_attendees, council_verdict, council_dissent, council_assumption_adversary_verdict, council_decisions_recorded, mission-alignment fields, deferred-substrate retrospective).
5. **Cite-able**: cite-chain to related_deliberation_ids (sister Z6 sextet council exemplar; T3 batched Z6 lift council) + related_design_memos (NSCS01 design memo; A-STACK composition memo; COHERENCE-AUDIT lattice memo; WAVE-3 optimization memo); Phase 1b landing memo; probe-disambiguator stub.
6. **Counterfactual-able**: Revision #3 contingency branching IS the counterfactual surface — the empirical anchor from probe + smoke MUST fire BOTH branches' commitments depending on outcome, so the decision is structurally testable.

## Observability surface

Per the MAX-OBSERVABILITY-INTO-BEHAVIOR standing directive 2026-05-16 + Catalog #305: 6-facet observability surface documented in §12 above. Council deliberation memos are structurally compatible with the design-memo scope (filename suffix differs but documentation discipline carries; sister Z6 council exemplar carries identical structure).

---

## 13. Cross-references

**Empirical anchors consumed:**
- Phase 1b lift landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md` (sister-owned by `NSCS01-FULL-MAIN-IMPLEMENTATION-PR95-PARADIGM-20260515` subagent)
- NSCS01 design memo: `.omx/research/nscs01_nullspace_split_renderer_design_20260515.md`
- A-STACK composition memo: `.omx/research/a_stack_nscs01_02_03_composition_full_stack_design_20260516.md`
- COHERENCE-AUDIT lattice coordinate memo: `.omx/research/coherence_audit_lattice_coordinate_assignment_20260516.md`
- WAVE-3 optimization-per-lattice memo: `.omx/research/wave_3_optimization_per_lattice_coherence_20260516.md`
- Sister Z6 sextet council exemplar: `.omx/research/sextet_council_z6_phase_2_consensus_20260516.md` (canonical T2 sextet-pact CONSENSUS format)
- Lattice state: `.omx/state/lattice_state.jsonl` (NSCS01 + A-STACK + NSCS02 + NSCS03 all `lifted_pending_council` per Substitution Set B)
- Probe-disambiguator stub: `tools/probe_nscs01_head0_arch_disambiguator.py` (~50 LOC; emits 4 required measurements; do-not-rank dispatch rule)
- 45 dedicated tests: `src/tac/substrates/nscs01_nullspace_split_renderer/tests/test_nscs01_substrate.py`
- Lane registry: `.omx/state/lane_registry.json` (`lane_nscs01_nullspace_split_renderer_20260515` L1 4/7 gates)

**Standing directives anchoring this deliberation:**
- CLAUDE.md "Council hierarchy: 4-tier protocol" + "Council conduct — non-negotiable" (sextet pact + Assumption-Adversary seat + per-round assumption-statement discipline Fix 7)
- CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW — NON-NEGOTIABLE" + Catalog #292 per-deliberation discipline
- CLAUDE.md "Mission alignment — non-negotiable" + Catalog #300 mission-contribution frontmatter
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (DEFER branch preserves paradigm)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" (Revision #3 PROCEED branch requires paired)
- CLAUDE.md "Apples-to-apples evidence discipline" (all bands tagged `[prediction]` until paired CPU/CUDA anchor)
- CLAUDE.md "Production-hardened dispatch optimization protocol" + Catalog #270 (Tier 1/2/3 umbrella; Phase 1b satisfies)
- CLAUDE.md "Race-mode rigor inversion" (NO active leaderboard race; standard sextet-pact procedure)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (Phase 1b lift bound 14 layers; canonical-vs-unique decisions documented)
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" (recipe REMAINS research_only=true per Revision #3 contingency)
- HORIZON-CLASS standing directive 2026-05-16 Consequence 2 (≥20% K-schedule frontier-pursuit allocation; NSCS01 occupies frontier-pursuit slot)
- 9-dim checklist standing directive 2026-05-15

**Catalog gates relevant:**
- #117/#157/#174 commit serializer + --expected-content-sha256
- #125 6-hook wire-in
- #126 lane pre-registration
- #128 + #131 fcntl-locked JSONL posterior
- #143 Lightning paid-job-register-before-submit
- #146 Phase 1 trainer 3-positional-arg contract
- #164 canonical scorer-preprocess routing
- #166 Modal HEAD-parity ledger
- #167 smoke-before-full pattern
- #190 hardware substrate detection
- #199 + #202 operator-authorize bypass + clean-bypass discipline
- #205 select_inflate_device
- #206 subagent crash-resume discipline
- #220 substrate L1+ operational mechanism
- #226 canonical auth-eval helper
- #229 premise verification
- #230 sister-subagent ownership map
- #240 recipe-vs-trainer-state consistency
- #243 local pre-deploy harness
- #245 modal_call_id_ledger
- #248 conflict markers refused
- #249 phantom-score directory
- #270 dispatch optimization protocol umbrella
- #272 distinguishing-feature integration contract
- #290 substrate canonical-vs-unique decision per layer
- #291 META-ASSUMPTION cadence
- #292 per-deliberation assumption surfacing
- #294 9-dim checklist
- #296 Dykstra-feasibility band
- #300 mission-alignment frontmatter
- #303 hard-earned-vs-cargo-culted addendum
- #305 observability surface
- #309 horizon-class declaration

---

## 14. Sister-subagent ownership map honored (Catalog #230)

- **NSCS03 Phase 2 sextet council (sister, concurrent)**: owns NSCS03 deliberation memo + NSCS03 recipe + NSCS03 lane gates. **Disjoint scope from this lane.**
- **THIS lane** owns ONLY: `.omx/research/sextet_council_nscs01_phase_2_consensus_20260516.md` (THIS memo) + `lane_nscs01_phase_2_sextet_council_20260516` checkpoint record + lane marks for `lane_nscs01_phase_2_sextet_council_20260516`. Recipe `.omx/operator_authorize_recipes/substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch.yaml` REMAINS unchanged (research_only=true / dispatch_enabled=false / smoke_only=true preserved); follow-on commit batch (post-Revision-#1+#2 empirical anchor) is the canonical place to flip the recipe.
- The lift memo `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md` is sister-owned (`NSCS01-FULL-MAIN-IMPLEMENTATION-PR95-PARADIGM-20260515` subagent already complete); NOT touched by this deliberation.
- The probe-disambiguator stub at `tools/probe_nscs01_head0_arch_disambiguator.py` is sister-owned (predecessor subagent); NOT extended by this deliberation. Revision #1 may require a future sister-subagent to extend the stub to execute the 4 measurements.

---

## 15. Continual-learning anchor (post-commit; per CLAUDE.md "Continual learning wire-in rule")

After THIS memo lands via canonical commit serializer with `--expected-content-sha256`, append continual-learning anchor via:

```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="sextet_council_nscs01_phase_2_consensus_20260516",
    topic="NSCS01 Phase 2 sextet-pact council CONSENSUS per design memo §10 + COHERENCE-AUDIT Substitution Set B",
    council_tier=CouncilTier.T2,
    council_attendees=("Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"),
    council_quorum_met=True,
    council_verdict="PROCEED_WITH_REVISIONS",
    council_dissent=(
        {"member": "Contrarian", "verbatim": "[see council_dissent.Contrarian.verbatim in frontmatter]"},
        {"member": "Assumption-Adversary", "verbatim": "[see council_dissent.Assumption-Adversary.verbatim in frontmatter]"},
    ),
    council_assumption_adversary_verdict=(
        {"assumption": "Phase 1b _full_main lift closes engineering-completeness preconditions for L2 promotion", "classification": "HARD-EARNED", "rationale": "45/45 tests pass + Catalog #187 + canonical scorer routing"},
        {"assumption": "Nullspace structural property yields ΔS lift at contest-CUDA archive byte savings", "classification": "CARGO-CULTED", "rationale": "Mathematical fact is necessary not sufficient; recipe predicted_band=null is operator-honest tag"},
        {"assumption": "Predicted ΔS band [0.190, 0.205] CUDA / [0.185, 0.198] CPU holds under paired empirical anchor", "classification": "CARGO-CULTED", "rationale": "Z3-G1 cargo-cult-prediction precedent applies"},
        {"assumption": "Catalog #167 smoke + head0 probe at SAME archive bytes IS sufficient empirical gate", "classification": "HARD-EARNED", "rationale": "Sister Z6 sextet council 2026-05-16 PROCEED_WITH_REVISIONS precedent"},
        {"assumption": "DEFER branch (Revision #3) preserves paradigm AND lifts engineering value", "classification": "HARD-EARNED", "rationale": "CLAUDE.md Forbidden premature KILL; reactivation criteria documented in design memo §10"},
        {"assumption": "NSCS01 dispatch claim 'frontier_breaking' mission-contribution", "classification": "PARTIALLY HARD-EARNED + CARGO-CULTED", "rationale": "frontier-pursuit horizon-class HARD-EARNED; frontier_breaking conditional on probe-WIN + A-STACK realization"},
    ),
    council_decisions_recorded=(
        "Recipe REMAINS research_only=true / dispatch_enabled=false / smoke_only=true (NO change in same commit batch as this memo)",
        "Revision #1 mandatory: head0-architecture disambiguator probe at $1 Modal T4 executes the 4 required measurements",
        "Revision #2 mandatory: Catalog #167 smoke-before-full at $1 Modal T4 paired with head0 probe at SAME archive bytes",
        "Revision #3 mandatory contingency: PROCEED branch (flip recipe + dispatch FULL + paired CPU/CUDA) IF probe-WIN + smoke-green; DEFER branch (measured-config-retired; no KILL; NSCS01-v2 reactivation criteria including HEAD0_BITS=6 / alternative head0 / lower lambda_pixel_0 / D4 composition / Comma2k19-pretrained) IF probe-LOSS",
        "Cross-decision dependency: NSCS01 results inform A-STACK composition Rule #3 (NSCS01+NSCS02+NSCS03)",
        "30-day deferred-substrate retrospective scheduled 2026-06-16T11:43:00Z for NSCS01-v1 head0 capacity surface",
    ),
    predicted_mission_contribution="frontier_protecting",
    override_invoked=False,
    override_rationale=None,
    deferred_substrate_retrospective_due_utc="2026-06-16T11:43:00Z",
    deferred_substrate_id="nscs01_nullspace_split_renderer_v1_head0_capacity_surface",
)
append_council_anchor(record)
```

This canonical helper invocation IS the explicit Catalog #128 + #131 fcntl-locked JSONL persistence; future deliberations + cathedral autopilot ranker + Rashomon ensemble + Assumption-Adversary classification-history queries all consume this anchor.

Also register lattice-state update via:

```python
from tac.lattice_state_ledger import update_lattice_node

update_lattice_node(
    "nscs01",
    status="phase_2_sextet_council_PROCEED_WITH_REVISIONS",
    evidence_artifact_path=".omx/research/sextet_council_nscs01_phase_2_consensus_20260516.md",
    notes="Phase 2 sextet-pact CONSENSUS at 6-of-6. Recipe REMAINS research_only=true pending Revision #1 (head0 probe) + Revision #2 (smoke-before-full) + Revision #3 contingency. $2.50 NEXT envelope.",
)
```

---

## 16. Compliance + sister regression checklist

- **Catalog #229 premise-verification-before-edit**: 8 PVs pre-deliberation (§0)
- **Catalog #230 sister-subagent ownership map**: NSCS03 sister concurrent noted; THIS lane owns ONLY this memo + checkpoint records + lane marks (§14)
- **Catalog #117/#157/#174 canonical commit serializer with `--expected-content-sha256`**: PENDING commit (POST-EDIT working-tree sha)
- **Catalog #126 lane pre-registration**: `lane_nscs01_phase_2_sextet_council_20260516` registered at L0 BEFORE this memo write (verified via `tools/lane_maturity.py add-lane`)
- **Catalog #185 LIVE_COUNT-zero discipline**: NO catalog claims requiring strict-mode verification; THIS memo cites existing catalogs only
- **Catalog #206 subagent crash-resume discipline**: 4 checkpoints planned (init / pre-write / post-write / completion)
- **Catalog #248 conflict markers**: NONE in THIS memo body
- **Catalog #292 per-deliberation assumption surfacing**: 6 per-member operating-within assumptions (§1) + 6 Assumption-Adversary verdicts (council_assumption_adversary_verdict frontmatter)
- **Catalog #300 v2 frontmatter**: ALL required T2+ fields present (council_tier / council_attendees / council_quorum_met / council_verdict / council_dissent / council_decisions_recorded / council_assumption_adversary_verdict / council_predicted_mission_contribution / council_override_invoked / council_override_rationale / deferred_substrate_retrospective_due_utc / deferred_substrate_id)

---

## 17. Status

**SEXTET-PACT COUNCIL CONSENSUS DELIBERATION LANDED 2026-05-16.**

**NEXT OPERATOR ACTION:** approve $2.50 envelope for Revision #1 + #2 (head0-architecture disambiguator probe at $1 Modal T4 + Catalog #167 smoke-before-full at $1 Modal T4 paired at SAME archive bytes + GHA CPU paired ~$0.50) via `tools/operator_authorize.py` per Catalog #199/#202 paired-env discipline. The probe + smoke IS the canonical empirical anchor that resolves the per-head bit-width capacity uncertainty + the cargo-culted NSCS01-v1 frame_0_head capacity assumption + branches Revision #3 PROCEED-vs-DEFER cleanly.

**Recipe state**: `research_only: true / dispatch_enabled: false / smoke_only: true` REMAINS — NO flip in same commit batch as this memo per Revision #1+#2+#3 sequencing.

**Lane gates marked**:
- `impl_complete=true` (Phase 1b inherited)
- `strict_preflight=true` (Phase 1b inherited)
- `real_archive_empirical=false` (pending Revision #1+#2 smoke + probe empirical anchor)
- `contest_cuda=false` (pending Revision #3 PROCEED)
- `contest_cpu=false` (pending Revision #3 PROCEED + GHA Linux x86_64 paired)
- `three_clean_review=true` (THIS sextet-pact 6-of-6 consensus satisfies Catalog #233 gate 3 evidence basis for `auth_eval_100ep` once Revision #3 PROCEED fires)
- `memory_entry=true` (THIS file is the entry; Phase 1b lift memo also counts)
- `deploy_runbook=true` (Phase 1b inherited)

**Status post-deliberation**: design memo §10 reactivation criteria + Catalog #240 recipe-vs-trainer-state precondition SATISFIED at L1-SCAFFOLD level (the trainer IS lifted; the recipe HONESTLY declares the substrate as research-only pending probe + smoke empirical anchor); Phase 2 sextet-pact council CONSENSUS verdict PROCEED_WITH_REVISIONS achieved (NOT PROCEED unconditionally); Revision #1+#2 (probe + smoke) becomes the explicit next blocker; recipe flip happens in follow-on commit batch contingent on Revision #2 probe outcome.

**END OF NSCS01 PHASE 2 SEXTET-PACT COUNCIL CONSENSUS DELIBERATION MEMO**
