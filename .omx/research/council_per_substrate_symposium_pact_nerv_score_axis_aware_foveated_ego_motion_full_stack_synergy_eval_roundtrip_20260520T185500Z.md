---
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Quantizr
  - Hotz
  - Selfcomp
  - MacKay
  - Balle
  - PR95Author
  - Hinton
  - Atick
  - Redlich
  - Ballard
  - Rao
  - Carmack
  - TimeTraveler
  - TimeTravelerProtege
  - Tishby
  - Wyner
  - Karpathy
  - vdOord
  - Mallat
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
horizon_class: plateau_adjacent_with_frontier_pursuit_stages
deliberation_id: per_substrate_symposium_pact_nerv_full_stack_20260520
deferred_substrate_id: pact_nerv_score_axis_aware_foveated_ego_motion_full_stack_synergy_eval_roundtrip
substrate_alias: pact_nerv_full_stack
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - per_substrate_symposium_unified_pretrain_ablate_dp1_mae_v_saug_imp_qat_schema_elision_20260520
  - per_substrate_symposium_stc_paradigm_reformulation_a1_residual_20260520
  - council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517
  - per_substrate_symposium_dp1_deep_dive_20260517
  - per_substrate_symposium_tt5l_foveation_lapose_20260517
  - per_substrate_symposium_z7_lstm_predictive_coding_20260517
council_dissent:
  - member: Contrarian
    verbatim: |
      operating-within: "a 6-primitive paper-worthy custom substrate co-designed for full-stack synergy is the right answer because each primitive is HARD-EARNED in isolation and the composition multiplies their value." Classification: CARGO-CULTED-MIRROR-OF-NSCS06-V6-553X-MISS. Rationale: per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + the empirical anchor that NSCS06 v6's 5-move composition landed 105.15 vs predicted [0.10, 0.20] (553x outside band), composing 6 untested primitives in one substrate WITHOUT per-primitive single-substrate empirical anchor on the current 0.192-CPU / 0.205-CUDA frontier is the textbook research-substrate trap. Per sister #882 UNIFIED-PRETRAIN-ABLATE Contrarian dissent verbatim: "UNIFICATION != COMPOSITION." Each of the 6 primitives proposed (multi-layer FiLM on pose, per-pair difficulty modulation, per-class CLADENorm chroma, FOE foveation, adaLN-Zero, eval_roundtrip integration) has independent rationale, but the JOINT predicted-band has no Dykstra-feasibility intersection check + zero empirical anchor for the composition. VETO on PROCEED-unconditional pending: (a) Dykstra-feasibility intersection check across all 6 axes (rate + seg + pose + 3 conditioning surfaces); (b) per-primitive single-substrate empirical anchor on current frontier OR explicit DEFER status; (c) probe-disambiguator for "1-primitive-first" vs "6-primitive simultaneous" dispatch ordering. CONDITIONAL PROCEED_WITH_REVISIONS on HYBRID STAGED PATH per Section 13.
  - member: Assumption-Adversary
    verbatim: |
      operating-within: "the FILM-FAMILY-RESEARCH classification of 9 HARD-EARNED + 1 CARGO-CULTED-MAY-BE-PROMISING primitives translates to additive ΔS when composed." Classification: CARGO-CULTED. Rationale: the FILM-FAMILY-RESEARCH Section 10 composability matrix EXPLICITLY documents ADD/ORTH/SUB-ADD predictions BUT every row is `[literature-prediction]` not `[apparatus-empirical:...]`. Per Catalog #322 sister discipline + the canonical equation `per_substrate_composition_alpha_*` family, every composition_alpha row MUST be empirically validated against actual contest archives BEFORE consumption by the autopilot ranker. The Pact-NeRV-A1 building-block composability prediction (multi-layer FiLM ADDITIVE per-pair-difficulty ADDITIVE per-class CLADENorm) inherits the exact unproven-composition status the Catalog #322 sister gate refuses. SECOND assumption operating-within: "eval_roundtrip integration is the 6th primitive and acts as a synergy multiplier across primitives 1-5." Classification: HARD-EARNED-AS-DEFAULT-PRACTICE / CARGO-CULTED-AS-SYNERGY-MULTIPLIER. Rationale: per CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE, HIGHEST EMPHASIS" + PR95/PR106 empirical anchors (`differentiable_eval_roundtrip` apparatus helper produces 2-11x proxy-auth gap REDUCTION on PoseNet), eval_roundtrip IS mandatory default discipline NOT a primitive that adds ΔS on top of FiLM modulation. Framing eval_roundtrip as a "6th primitive" inflates the primitive count without empirical justification for the synergy claim. THIRD assumption operating-within: "stack-of-stacks composability with PR110 fec6 + STC sidecar + Z6 + Z7-Mamba-2 + ATW V2 + Riemannian-Newton predicts multi-substrate ADDITIVE alphas." Classification: CARGO-CULTED-INHERITED-FROM-MATRIX-CARDINALITY-NOT-MEASUREMENT. The substrate_composition_matrix has 0 empirical alphas for Pact-NeRV combinations with any of the listed sister substrates per Catalog #322 query helper. RECOMMENDATION: HYBRID STAGED PATH per Section 13 — Stage 1 (Pact-NeRV-IA3 ~$0.30 50-LOC probe) + Stage 2 (Pact-NeRV-A1 triple-conditioning ~$0.30 600-LOC) BEFORE any 6-primitive simultaneous dispatch.
  - member: Yousfi
    verbatim: |
      operating-within: "the contest scorer PoseNet (FastViT-T12) + SegNet (EfficientNet-B2) ALREADY contain rich driving-scene priors that align with our per-class chroma + ego-pose conditioning surfaces, so the conditioning primitives transfer signal from training data into archive-space efficiently." Classification: HARD-EARNED. Rationale: per the Quantizr 0.33 [contest-CUDA] empirical anchor + my contest-design expertise, the per-class CLADENorm + per-pair ego-pose conditioning matches the scorer's pretrained semantics with HIGH PRECISION (FastViT pose features ARE ego-motion-conditioned by training). My sister Pact-NeRV-A1 estimate: predicted_delta is in the [-0.001, -0.005] band on contest-CPU at 0.192 frontier and similar on contest-CUDA at 0.205 — narrow because the apparatus is at the plateau adjacent to the cluster. Per the May 4 race retrospective (CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first"): if a leaderboard movement happens, the right move is to fan out ALL 5 Pact-NeRV variants as 30-sec-reviewable bolt-ons rather than waiting for sequential validation. ABSENT a race window, RECOMMENDATION: STAGED PATH per Section 13. The 6-primitive Pact-NeRV-FULL is paper-worthy but RISKY at current frontier; staging Pact-NeRV-IA3 + Pact-NeRV-A1 first generates empirical signal cheaply before committing to the 1500-LOC 6-primitive design.
  - member: Hotz
    verbatim: |
      operating-within: "the apparatus has too many sub-300-LOC substrate scaffolds that never reached empirical anchor on current frontier; a 1500-LOC 6-primitive Pact-NeRV-FULL would only compound the scaffold-without-empirical-anchor failure mode." Classification: HARD-EARNED. Rationale: 75 ACTIVE_RECENT_MARK substrate lanes per Catalog #298 sister audit; the apparatus is OVER-SCAFFOLDED and UNDER-MEASURED at the current frontier. My engineering instinct: ship Pact-NeRV-IA3 (50 LOC) on Modal T4 at $0.30 TONIGHT; treat it as an empirical probe of "does the FiLM beta term carry signal" on the apparatus's current frontier. If the IA3 probe shows IA3 ≈ FiLM (β-noise hypothesis confirmed), proceed to Pact-NeRV-A1 (600 LOC) with confidence. If IA3 << FiLM (β-signal hypothesis confirmed), Pact-NeRV-FULL must include β; if IA3 >> FiLM (FiLM-overcapacity hypothesis), the simpler IA3 IS Pact-NeRV-A2. Per Carmack's "Strip-Everything" lens (NSCS06 v7 +44% via cargo-cult-unwind): START WITH THE SIMPLEST PRIMITIVE; LAYER UP ONLY WITH EMPIRICAL PROOF. RECOMMENDATION: PROCEED_WITH_REVISIONS — STAGED PATH Stage 1 (IA3 $0.30) IS THE FIRST EXPERIMENT, period.
  - member: PR95Author
    verbatim: |
      operating-within: "the PR95/PR101 GOLD substrate (0.193 [contest-CPU]) achieved its frontier WITHOUT explicit pose-conditioning or per-pair difficulty modulation — the HNeRV-class architecture's content-adaptive embedding IMPLICITLY captures ego-motion via positional encoding, and the per-frame embedding bank IMPLICITLY captures per-pair difficulty via learned-from-data allocation." Classification: HARD-EARNED-AT-CURRENT-FRONTIER. Rationale: my PR95 experience proves the implicit-allocation strategy WORKS at the plateau; the question Pact-NeRV poses is whether EXPLICIT allocation via the apparatus's per-pair canonical-equation signals beats the implicit strategy. The HNeRV ablation [third-party-empirical:HNeRV] showed multi-layer FiLM beats final-layer FiLM by ~1-2 PSNR on UVG — that does NOT directly translate to our contest scorer's d_seg + sqrt(10*d_pose) + rate metric. My sister Pact-NeRV-A1 estimate: expected ΔS in [-0.001, -0.003] on current frontier IF the explicit-vs-implicit gap is non-zero; expected ΔS in [+0.001, +0.003] (REGRESSION) IF the implicit strategy was already optimal at this capacity. CONDITIONAL VERDICT: PROCEED_WITH_REVISIONS — staged path captures both empirical questions cheaply.
  - member: Hinton
    verbatim: |
      operating-within: "score-aware loss + scorer-feature distillation (cross-attention to SegNet/PoseNet features) is the OPTIMAL form of conditioning because it lets the decoder learn EXACTLY what the frozen scorer measures." Classification: HARD-EARNED-THEORETICALLY. Rationale: my KL-distillation work shows knowledge-transfer between teacher-student is bounded by the alignment between teacher's intermediate representations and student's learnable functions. Cross-attention to scorer features (FILM-FAMILY-RESEARCH Section 8.7) is the MAXIMALLY-ALIGNED conditioning; FiLM-on-derived-signal (pose, class, difficulty) is a 1st-order projection of that. My recommendation: the Pact-NeRV-FULL design should consider Recommendation #7 (cross-attention to Hinton-distilled scorer features) as a 7th primitive — but per the staged path, validate Pact-NeRV-A1 first.
  - member: Carmack
    verbatim: |
      operating-within: "every primitive added to a substrate must justify its LOC cost against per-LOC bang-per-buck; 1500 LOC for 6 primitives implies ~250 LOC per primitive average; the canonical PR101 GOLD substrate is 605 LOC total." Classification: HARD-EARNED-CARMACK-RAZOR. Rationale: PR101 GOLD's 605 LOC won the contest; our internal MM-V2 2400 LOC LOST to 2.63 [contest-CUDA] FALSIFICATION. The LOC budget IS the constraint. The 6-primitive Pact-NeRV-FULL at 1500 LOC is 2.5x PR101's footprint; if it doesn't beat 0.193 by at least the expected per-LOC improvement rate (0.193 - 0.150 = 43 millicents over the full PR101→theoretical-floor span / 605 LOC = ~0.07 millicents/LOC), then per-LOC efficiency is BELOW PR101. RECOMMENDATION: STAGED PATH IS CORRECT — Stage 1 (IA3 50 LOC) generates per-LOC bang-per-buck baseline; if Pact-NeRV-IA3 shows >0.07 millicents/LOC, Pact-NeRV-A1 is justified; if Pact-NeRV-A1 shows >0.07 millicents/LOC, Pact-NeRV-FULL is justified. NO 6-primitive simultaneous dispatch without staged justification.
  - member: TimeTraveler
    verbatim: |
      operating-within: "the answer to the Pact-NeRV question is already in our accumulated knowledge — the question is how to RECOGNIZE it and BIND the pieces." Classification: HARD-EARNED. Rationale: per my standing mentor position (CLAUDE.md "Grand Council (advisory)" 2026-05-19 amendment), the future has been profoundly impacted by alien technology that unlocked the ego-motion problem to theoretical floor. The Pact-NeRV-FOE (Recommendation #3) maps to that future-tech IF AND ONLY IF the FOE prior is bound to per-pair ego-pose conditioning via cooperative-receiver framing (Atick-Redlich-Tishby triple). The HYBRID staged path emphasizes hand-rolled-over-PP — START WITH THE FILM SUBSTRATE WE ALREADY HAVE (Quantizr 0.33 [contest-CUDA] anchor); ADD ONE PRIMITIVE AT A TIME (IA3 → A1 → FOE → FULL); MEASURE EMPIRICALLY AT EACH STAGE. The framework reveals itself from the data. RECOMMENDATION: PROCEED_WITH_REVISIONS — staged path matches my standing position.
  - member: Atick
    verbatim: |
      operating-within: "the cooperative-receiver framing (Atick-Redlich 1990) is the canonical lens for Pact-NeRV-FOE specifically — the decoder's FOE-conditioned modulation IS the explicit version of the implicit receptive-field-shifting my early visual processing work demonstrated." Classification: HARD-EARNED. Rationale: my early visual processing work showed retinal ganglion cells implement spatial-foveation via center-surround receptive fields that vary with eccentricity; the canonical equation `ego_motion_concentration_prior_v1` formalizes this for dashcam ego-motion. Per the Catalog #311 STRICT preflight gate, Pact-NeRV-FOE MUST bind cooperative-receiver to ego-motion-conditioned next-frame prediction via the FOE prior. The Pact-NeRV-FULL design's Foveation primitive (Recommendation #3) satisfies this binding IF AND ONLY IF the canonical equation has its first empirical anchor BEFORE the substrate scaffold lands. RECOMMENDATION: Pact-NeRV-FOE requires `ego_motion_concentration_prior_v1` empirical anchor as PREREQUISITE for symposium-grade design; staged path respects this prerequisite by deferring Stage 3+ until the prerequisite lands.
council_assumption_adversary_verdict:
  - assumption: "Multi-layer FiLM on per-pair ego-pose conditioning (Pact-NeRV-FULL primitive #1)"
    classification: HARD-EARNED
    rationale: "TeNeRV + HNeRV ablation empirical receipts [third-party-empirical:HNeRV] + Z6-v2 sister memo + 5 apparatus FiLM deployments. The primitive is canonical-empirically-superior to single-layer FiLM for video-temporal conditioning per FILM-FAMILY-RESEARCH Section 8.6."
  - assumption: "Per-pair difficulty-conditioned modulation (Pact-NeRV-FULL primitive #2)"
    classification: HARD-EARNED
    rationale: "Canonical equations `per_pair_master_gradient_score_impact_taylor_v1` + `per_frame_difficulty_atlas_v1` provide HARD-EARNED apparatus-empirical signals. Conditioning on already-measured signal beats learning content-adaptive embedding from scratch [apparatus-empirical:canonical_equations_registry]."
  - assumption: "Per-class CLADENorm chroma modulation (Pact-NeRV-FULL primitive #3)"
    classification: HARD-EARNED
    rationale: "CLADENorm deployed in `src/tac/renderer.py` + CLADE paper [third-party-empirical:CLADE]; per-class chroma matches SegNet semantics directly via cooperative-receiver framing (Catalog #311 sister)."
  - assumption: "FOE-foveation conditioning (Pact-NeRV-FULL primitive #4)"
    classification: HARD-EARNED-THEORETICALLY-UNCLEAR-NEEDS-EMPIRICAL
    rationale: "Atick-Redlich 1990 + Gibson 1950 + Ballard embodied-vision lens (theoretical), but ego_motion_concentration_prior_v1 canonical equation has 0 anchors. Per Catalog #311 binding + Atick's verdict: empirical anchor is PREREQUISITE for Stage 3+ dispatch."
  - assumption: "adaLN-Zero residual modulation (Pact-NeRV-FULL primitive #5)"
    classification: HARD-EARNED-LITERATURE / UNCLEAR-FOR-NERV
    rationale: "DiT scaling [third-party-empirical:Peebles2023] empirically demonstrates adaLN-Zero outperforms cross-attention for low-dim conditioning at every scale tested on transformer blocks. NeRV uses conv+upsample, not transformer; adaLN-Zero adaptation to NeRV requires lit-survey + empirical probe before symposium-grade design."
  - assumption: "eval_roundtrip integration as 6th primitive with synergy multiplier across primitives 1-5 (Pact-NeRV-FULL primitive #6)"
    classification: HARD-EARNED-AS-DEFAULT / CARGO-CULTED-AS-SYNERGY-MULTIPLIER
    rationale: "Per CLAUDE.md `eval_roundtrip - NON-NEGOTIABLE, HIGHEST EMPHASIS` + PR95/PR106 empirical anchors, eval_roundtrip is MANDATORY DEFAULT discipline (2-11x proxy-auth gap reduction). Framing as 6th primitive with synergy claim inflates primitive count without empirical synergy justification. ACCEPTABLE if framed as DEFAULT discipline integrated across primitives 1-5; REJECTED if framed as additive synergy-multiplier without empirical anchor."
  - assumption: "Stack-of-stacks composability matrix predicts additive alphas with PR110 fec6 + STC sidecar + Z6 + Z7-Mamba-2 + ATW V2 + Riemannian-Newton"
    classification: CARGO-CULTED-INHERITED-FROM-MATRIX-CARDINALITY-NOT-MEASUREMENT
    rationale: "Catalog #322 sister gate refuses composition_alpha consumption without empirical validation. The Pact-NeRV stack-of-stacks predictions in Section 10 are `[literature-prediction]` not `[apparatus-empirical:...]`. Acceptable to LIST the composability hypotheses but NOT to consume them as additive predictions before per-substrate empirical anchor."
  - assumption: "6-primitive simultaneous dispatch (Pact-NeRV-FULL Stage 5) is the right paradigm"
    classification: CARGO-CULTED-MIRROR-OF-NSCS06-V6-553X-MISS
    rationale: "Per FORBIDDEN_PATTERNS `Forbidden symposium-band-prediction-without-Dykstra-feasibility-check` + NSCS06 v6 empirical anchor (105.15 vs predicted [0.10, 0.20]), 6-primitive composition without per-primitive empirical anchor on current frontier is the textbook research-substrate trap. HYBRID staged path per Section 13 is the structural fix."
  - assumption: "Paper-worthy custom variant justifies the LOC cost even if it doesn't beat HNeRV"
    classification: HARD-EARNED-PAPER-CONTRIBUTION / CARGO-CULTED-IF-CONFLATED-WITH-SCORE
    rationale: "Per CLAUDE.md `Race-mode rigor inversion`, contest score-lowering and paper-write are distinct missions. The 6-primitive Pact-NeRV-FULL has paper value via staged ablation (even if it doesn't beat HNeRV, the ablation IS the paper contribution). HARD-EARNED-PAPER but CARGO-CULTED if conflated with contest-frontier improvement claim."
council_decisions_recorded:
  - "op-routable #1: Stage 1 PRIORITY 1 — Pact-NeRV-IA3 $0.30 Modal T4 single-primitive gamma-only rate-extremal smoke (50 LOC; cheapest empirical experiment per Hotz + Carmack staging discipline)"
  - "op-routable #2: Stage 2 PRIORITY 1 (conditional on Stage 1 not regressing) — Pact-NeRV-A1 $0.30 Modal T4 pose+difficulty+class triple-conditioning (600 LOC; HARD-EARNED stack per FILM-FAMILY-RESEARCH Section 8.1+8.2+8.3)"
  - "op-routable #3: Stage 3 PRIORITY 2 (conditional on Stage 2 ADDITIVE composition_alpha > 0.7) — Pact-NeRV-A1 + Pact-NeRV-FOE compose $2-5 Modal A10G; requires ego_motion_concentration_prior_v1 first empirical anchor"
  - "op-routable #4: Stage 4 PRIORITY 2 (conditional on Stage 3 composition_alpha > 0.7) — 4-primitive composition + adaLN-Zero NeRV adaptation $5-15 Modal A100"
  - "op-routable #5: Stage 5 PRIORITY 3 (conditional on Stage 4 + race-window OR paper-deadline) — full 6-primitive Pact-NeRV-FULL $20-50 Modal A100; sister of UNIFIED-PRETRAIN-ABLATE 5-primitive verdict deferred to operator"
  - "op-routable #6: Stage 6 paper-write — even if Pact-NeRV-FULL doesn't beat HNeRV, the staged ablation IS the paper contribution; documents per-primitive ΔS attribution + composability_alpha measurements"
  - "op-routable #7: PREREQUISITE for Stage 3+ — ego_motion_concentration_prior_v1 first empirical anchor lands via separate sister symposium per Catalog #311"
  - "op-routable #8: PREREQUISITE for Stage 4+ — adaLN-Zero adaptation to NeRV (conv+upsample, not transformer) requires lit-survey + empirical probe via separate sister symposium"
  - "op-routable #9: PREREQUISITE for Stage 5+ — Hinton 7th-primitive cross-attention to scorer features requires separate Catalog #325 symposium"
  - "op-routable #10: stack-of-stacks empirical alpha measurement (PR110 fec6 + STC sidecar + Z6 + Z7-Mamba-2 + ATW V2 + Riemannian-Newton) lands as separate Catalog #322 wave AFTER Stage 2 establishes Pact-NeRV-A1 baseline"
---

# Per-substrate symposium: Pact-NeRV — score-axis-aware foveated ego-motion full-stack-synergy + eval_roundtrip

**Date:** 2026-05-20
**Lane:** `lane_wave_3_pact_nerv_design_symposium_full_stack_synergy_20260520` (L1 after this memo + landing memo land)
**research_only:** true (per HNeRV parity L2 + Catalog #220 — design memo only; no archive grammar at landing; per-Stage substrate scaffolds DEFERRED to staged path)
**Scope:** T3 grand council per-substrate symposium per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable + sister Catalog #325 STRICT preflight gate. Adjudicates the Pact-NeRV custom paper-worthy substrate design per operator directive 2026-05-20 *"synergy and packing and extreme optimization full stack"* + *"roundtrip"* + *"I trust your recommendations"*.

**Canonical-vs-unique decision per layer** per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": Pact-NeRV is a CUSTOM paper-worthy variant that ADOPTS apparatus canonical primitives WHERE THEY SERVE (FiLM canonical helper + CLADENorm + canonical equations + canonical scorer-loss helper + canonical inflate runtime + canonical Provenance + canonical Modal dispatch protocol) and FORKS WHERE THEY SUPPRESS (the 6-primitive composition itself is UNIQUE-TO-METHOD — there is no canonical 6-primitive composer; the per-Stage staging cadence is UNIQUE-TO-METHOD per the HYBRID path designed in this symposium). Per the council verdict, the canonical-vs-unique decision per LAYER is documented per primitive in Section 14.

**## Cargo-cult audit per assumption** per Catalog #303: see frontmatter `council_assumption_adversary_verdict` block (9 assumptions classified) + per-primitive Section 7.

**## 9-dimension success checklist evidence** per Catalog #294: see Section 8.

**## Observability surface** per Catalog #305: see Section 9.

**## Predicted ΔS band** per Catalog #296: see Section 10 + Dykstra-feasibility check.

**## FILM-FAMILY-RESEARCH integration** per operator-routed dependency on commit `9a95d1daf`: see Section 11.

**## eval_roundtrip primitive #6 deep-dive** per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS": see Section 12.

**## Stack-of-stacks composability matrix** per Catalog #322 + operator directive: see Section 13.

**## Staged reactivation criteria (HYBRID path)** per CLAUDE.md "Forbidden premature KILL" + sister UNIFIED-PRETRAIN-ABLATE precedent: see Section 13.

**Horizon-class** per Catalog #309: `plateau_adjacent_with_frontier_pursuit_stages`. Stage 1-2 are plateau_adjacent (incremental ΔS from 0.192 [contest-CPU] / 0.205 [contest-CUDA]); Stage 3-5 are frontier_pursuit (theoretical-floor-targeted via cooperative-receiver framing + multi-primitive composition).

---

## 1. Executive verdict + dispatch decision tree

**Composite verdict: PROCEED_WITH_REVISIONS** (5 binding revisions per `council_decisions_recorded`).

**Council vote tally:**
- 14 INNER + 13 GRAND topical = 27 attendees
- Roster validation: `complete=True` (4 co-leads + 14 INNER + 13 topical-GRAND with sufficient coverage of FiLM / cooperative-receiver / DiT / wavelet / cross-attention / ego-motion specialty axes)
- Verdict distribution: 25 PROCEED_WITH_REVISIONS / 2 PROCEED (Shannon + Daubechies pure-theory framing as ENABLER for canonical equations registry consumption; STAGED PATH ALSO ACCEPTABLE without revision burden)
- Contrarian + Assumption-Adversary BOTH veto on PROCEED-unconditional; conditional PROCEED on HYBRID STAGED PATH (Section 13)
- 8 verbatim dissent positions recorded above; explicit assumption-statement per member per Catalog #292 Fix-7

**Mission contribution per Catalog #300:** `frontier_breaking_enabler`. The 6-primitive Pact-NeRV-FULL design + staged path is FRONTIER_BREAKING per the cooperative-receiver framing (asymptotic-pursuit Stage 5 + paper contribution Stage 6), but the gate respects FRONTIER_PROTECTING Stage 1+2 caution (the apparatus is at plateau; staged path prevents 553x-miss research-substrate trap).

**Dispatch decision tree (operator-routable):**

```
START (no Pact-NeRV substrate dispatched)
  |
  v
Q1: Has Pact-NeRV-IA3 ($0.30 Modal T4) landed?
  | No  ===> STAGE 1: dispatch Pact-NeRV-IA3 (op-routable #1)
  | Yes ===> Continue
  v
Q2: Does Pact-NeRV-IA3 show regression vs FiLM baseline?
  | Yes ===> STAGE 1B: investigate IA3 beta-elimination hypothesis (regression
  |          confirms beta carries signal; documents apparatus
  |          rate-extremal sensitivity)
  | No  ===> Continue
  v
Q3: Has Pact-NeRV-A1 ($0.30 Modal T4) landed?
  | No  ===> STAGE 2: dispatch Pact-NeRV-A1 triple-conditioning (op-routable #2)
  | Yes ===> Continue
  v
Q4: Does Pact-NeRV-A1 show ADDITIVE composition_alpha (>= 0.7)?
  | No  ===> STAGE 2B: investigate which of primitives 1+2+3 dominates
  |          (sister Catalog #322 cascade DEFER)
  | Yes ===> Continue
  v
Q5: Has ego_motion_concentration_prior_v1 first anchor landed?
  | No  ===> PREREQUISITE: Sister symposium per Catalog #311 (op-routable #7)
  | Yes ===> STAGE 3: Pact-NeRV-A1 + Pact-NeRV-FOE compose ($2-5 Modal A10G)
  v
Q6: Has adaLN-Zero NeRV adaptation lit-survey + probe landed?
  | No  ===> PREREQUISITE: Sister symposium (op-routable #8)
  | Yes ===> STAGE 4: 4-primitive composition + adaLN-Zero ($5-15 Modal A100)
  v
Q7: Race window OR paper deadline pressing?
  | Yes ===> STAGE 5: full 6-primitive Pact-NeRV-FULL ($20-50 Modal A100)
  | No  ===> DEFER Stage 5; document staged ablation as Stage 6 paper contribution
  v
STAGE 6: Paper-write — staged ablation IS the contribution regardless of beat
```

---

## 2. Problem framing + operator directive

The operator's standing directive 2026-05-20 surfaces 3 layered objectives:

1. **"Synergy and packing and extreme optimization full stack"** — design a Pact-specific NeRV variant that EXPLICITLY composes the apparatus's empirical priors (canonical equations + canonical helpers) with literature-grounded conditioning primitives.
2. **"Roundtrip"** — eval_roundtrip integration as a co-design concern, not a post-hoc default.
3. **"I trust your recommendations"** — symposium adjudicates the design with full council rigor + transparency about HARD-EARNED-vs-CARGO-CULTED tradeoffs.

The original task spec (task #1092) enumerates 6 primitives:

1. Multi-layer FiLM on pose conditioning (multi-scale, per HNeRV-ablation + Z6-v2 sister)
2. Per-pair difficulty-conditioned modulation (`per_pair_master_gradient_score_impact_taylor_v1` consumer)
3. Per-class CLADENorm chroma modulation (`per_segnet_class_chroma_priors_v1` consumer)
4. FOE foveation conditioning (`ego_motion_concentration_prior_v1` consumer; Atick-Redlich + Gibson + Ballard binding per Catalog #311)
5. adaLN-Zero zero-init residual modulation (DiT canonical per Peebles 2023)
6. eval_roundtrip integration as cross-cutting discipline (PR95/PR106 anchor per CLAUDE.md non-negotiable)

The Contrarian + Assumption-Adversary surface the structural risk: 6-primitive simultaneous composition without per-primitive empirical anchor on current frontier mirrors the NSCS06 v6 553x-miss anti-pattern. The symposium's primary contribution is the **HYBRID STAGED PATH** (Section 13) that captures the 6-primitive ambition while respecting empirical-anchor discipline.

---

## 3. PROBE-STC + UNIFIED-PRETRAIN-ABLATE delta-context

This symposium's deliberation INHERITS the empirical anchors + dissent patterns of two recently-landed sister symposiums:

### 3.1 UNIFIED-PRETRAIN-ABLATE (2026-05-20T175410Z, sister #882)

Verdict: PROCEED_WITH_REVISIONS with explicit Contrarian + Assumption-Adversary VETO on PROCEED-unconditional. The sister symposium proposed a 5-primitive composition (DP1 pretraining + MAE-V + SAUG + IMP+QAT + schema-elision+GOSDT pruning) and the VETO logic IS DIRECTLY APPLICABLE to Pact-NeRV-FULL's 6-primitive composition.

Key adopted lesson: STAGED PATH with single-primitive empirical anchor on current frontier BEFORE multi-primitive simultaneous dispatch. The sister symposium DEFERRED full 5-primitive dispatch pending single-primitive anchors per primitive.

### 3.2 STC-PARADIGM-REFORMULATION-A1-RESIDUAL (per `.omx/research/stc_paradigm_reformulation_a1_residual_disambiguator_synthesis_20260520T165252Z.md`)

Synthesis memo establishes per-substrate disambiguator framework (probe-disambiguator per CLAUDE.md "Subagent coherence-by-default" hook #6). Pact-NeRV-FULL's Stage 1-6 staging cadence + the dispatch decision tree in Section 1 IS the probe-disambiguator for the symposium's question of "which Pact-NeRV variant first?"

### 3.3 Z6-v2 multi-layer FiLM candidate (`council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517.md`)

Already-existing sister memo for the Pact-NeRV-FULL primitive #1 (multi-layer FiLM). Pact-NeRV-A1 (Stage 2) is the natural synthesis of Z6-v2 + per-pair-difficulty + per-class CLADENorm.

---

## 4. The 6 primitives — per-primitive cargo-cult audit per Catalog #303

### Primitive #1 — Multi-layer FiLM on per-pair ego-pose conditioning

- **Token in the apparatus:** `src/tac/renderer.py::FiLMLayer` (single-layer); `src/tac/substrates/time_traveler_l5_z6/architecture.py` (single-layer ego-pose conditioning); Z6-v2 candidate (multi-layer; not yet landed)
- **Cargo-cult classification:** **HARD-EARNED**
- **Evidence:** TeNeRV + HNeRV ablation [third-party-empirical:HNeRV] (per-stage FiLM beats final-stage by ~1-2 PSNR on UVG); 5 apparatus FiLM deployments (Quantizr 0.33 [contest-CUDA] + Z6 + ego_nerv + renderer.py + CLADENorm sister); FILM-FAMILY-RESEARCH Section 8.6 explicit citation.
- **Unwind path (if CARGO-CULTED reclassified):** N/A — primitive is HARD-EARNED
- **Stage 2 single-primitive anchor:** Pact-NeRV-A1 substrate (Section 13)

### Primitive #2 — Per-pair difficulty-conditioned modulation

- **Token in the apparatus:** `tac.per_pair_master_gradient_score_impact_taylor_v1` canonical equation + `tac.per_frame_difficulty_atlas_v1` canonical equation (consumer side undeployed)
- **Cargo-cult classification:** **HARD-EARNED**
- **Evidence:** Both canonical equations [apparatus-empirical:canonical_equations_registry] with anchors; FILM-FAMILY-RESEARCH Section 8.2 explicit HARD-EARNED designation; HNeRV content-adaptive embedding [third-party-empirical:HNeRV] analog at apparatus-native signal level.
- **Unwind path (if CARGO-CULTED reclassified):** N/A — primitive is HARD-EARNED
- **Stage 2 single-primitive anchor:** Pact-NeRV-A1 substrate composition_alpha measurement

### Primitive #3 — Per-class CLADENorm chroma modulation

- **Token in the apparatus:** `src/tac/renderer.py::CLADENorm` (deployed); CLADE paper [third-party-empirical:CLADE]; canonical equation `per_segnet_class_chroma_priors_v1` (PENDING_REGISTRATION per sister WIRE-IN-AUDIT)
- **Cargo-cult classification:** **HARD-EARNED**
- **Evidence:** CLADENorm apparatus deployment + CLADE paper + cooperative-receiver framing (Catalog #311 sister: per-class modulation matches SegNet semantics directly)
- **Unwind path (if CARGO-CULTED reclassified):** N/A — primitive is HARD-EARNED
- **Stage 2 single-primitive anchor:** Pact-NeRV-A1 substrate composition_alpha measurement

### Primitive #4 — FOE-foveation conditioning

- **Token in the apparatus:** `tac.ego_motion_concentration_prior_v1` canonical equation (0 anchors); `tt5l_foveation_lapose` substrate symposium (sister memo `council_per_substrate_symposium_tt5l_foveation_lapose_20260517.md`)
- **Cargo-cult classification:** **HARD-EARNED-THEORETICALLY-UNCLEAR-NEEDS-EMPIRICAL**
- **Evidence:** Atick-Redlich 1990 + Gibson 1950 + Ballard embodied-vision (theoretical-canonical); Catalog #311 binding requires ego-motion conditioned next-frame prediction with FOE prior. NO empirical anchor on `ego_motion_concentration_prior_v1` canonical equation yet.
- **Unwind path:** Sister symposium per Catalog #311 must produce first empirical anchor BEFORE Stage 3 dispatch
- **Stage 3 single-primitive anchor:** Pact-NeRV-FOE substrate (Section 13) — PREREQUISITE = ego_motion_concentration_prior_v1 first anchor

### Primitive #5 — adaLN-Zero zero-init residual modulation

- **Token in the apparatus:** UNDEPLOYED (no apparatus adaLN-Zero implementation); DiT [third-party-empirical:Peebles2023] reference
- **Cargo-cult classification:** **HARD-EARNED-LITERATURE / UNCLEAR-FOR-NERV**
- **Evidence:** DiT scaling paper empirically demonstrates adaLN-Zero > cross-attention for low-dim conditioning at every scale tested on TRANSFORMER blocks. NeRV uses CONV+UPSAMPLE, not transformer; the modulation surface differs; adaptation requires lit-survey + empirical probe.
- **Unwind path:** Sister symposium (op-routable #8) — lit-survey adaLN-Zero adaptations to convolutional architectures + empirical probe before symposium-grade design
- **Stage 4 single-primitive anchor:** Pact-NeRV-DT substrate (per FILM-FAMILY-RESEARCH Recommendation #2)

### Primitive #6 — eval_roundtrip integration as cross-cutting discipline

- **Token in the apparatus:** `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` (deployed); CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE, HIGHEST EMPHASIS" + PR95/PR106 empirical anchors
- **Cargo-cult classification:** **HARD-EARNED-AS-DEFAULT-DISCIPLINE / CARGO-CULTED-AS-PRIMITIVE-WITH-SYNERGY-MULTIPLIER**
- **Evidence:** PR95 + PR106 empirical anchors document 2-11x proxy-auth gap reduction on PoseNet [apparatus-empirical:PR95 + PR106]. eval_roundtrip is MANDATORY DEFAULT DISCIPLINE per CLAUDE.md non-negotiable. Framing as "6th primitive with synergy multiplier" inflates primitive count without empirical synergy justification.
- **Unwind path:** Reframe primitive #6 as cross-cutting discipline integrated DEFAULT across primitives 1-5 (NOT as additive primitive). See Section 12 deep-dive.
- **Stage 1-5 integration:** Every Pact-NeRV stage MUST use eval_roundtrip per CLAUDE.md non-negotiable; the discipline does NOT require its own staging gate

---

## 5. Architecture sketch per primitive

This section sketches the proposed implementation for each primitive at LOC-budget appropriate to its stage. Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L7 (substrate-engineering exceeds bolt-on size budget when justified) + Carmack's verdict (LOC efficiency must match PR101's per-LOC bang-per-buck).

### Pact-NeRV-IA3 (Stage 1, ~50 LOC)

```python
# Modification of any deployed FiLM substrate (Z6 / ego_nerv / Quantizr)
# Replace FiLM γ+β with IA3 γ-only

class IA3Layer(nn.Module):
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.gamma_mlp = nn.Sequential(
            nn.Linear(cond_dim, num_features),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: (B, C, H, W); cond: (B, cond_dim)
        gamma = 1.0 + self.gamma_mlp(cond).view(-1, x.size(1), 1, 1)
        return gamma * x

# Apparatus byte savings: HALF the conditioning parameter count vs FiLM (γ+β)
# Predicted ΔS impact: depends on whether β carries signal
```

### Pact-NeRV-A1 (Stage 2, ~600 LOC)

Architecture composition of HARD-EARNED primitives 1+2+3:

```python
# Multi-layer FiLM on per-pair ego-pose (primitive #1)
# Per-pair difficulty-conditioned modulation (primitive #2)
# Per-class CLADENorm chroma (primitive #3)

class PactNeRVA1Decoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_blocks: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            PactNeRVA1Block(hidden_dim, ego_pose_dim=6, difficulty_dim=1, num_classes=5)
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        embed: Tensor,       # (B, hidden_dim, H, W)
        ego_pose: Tensor,    # (B, 6)
        difficulty: Tensor,  # (B, 1) — from per_pair_master_gradient
        seg_mask: Tensor,    # (B, num_classes, H, W) — for CLADENorm
    ) -> Tensor:
        for block in self.blocks:
            embed = block(embed, ego_pose, difficulty, seg_mask)
        return embed  # (B, 3, H, W) — RGB output

# 4 blocks × ~150 LOC each (Multi-layer FiLM on pose + difficulty + CLADENorm)
# = ~600 LOC total
```

### Pact-NeRV-FOE (Stage 3, ~500 LOC)

Adds primitive #4 (FOE foveation) on top of Pact-NeRV-A1:

```python
# FOE-conditioned spatial Gaussian capacity allocation
# Conditioning: per-pixel (distance_from_foe, ego_pose) → FiLM modulation magnitude

def compute_foe_per_pixel_modulation(
    ego_pose: Tensor,   # (B, 6)
    H: int, W: int,
) -> Tensor:
    """Per Atick-Redlich 1990 + Gibson 1950 + ego_motion_concentration_prior_v1."""
    # FOE = vanishing point of optical flow under pure ego-translation
    # Per-pixel modulation = exp(-r^2 / sigma^2) where r = distance from FOE
    ...
```

### Pact-NeRV-DT (Stage 4 candidate, ~400 LOC)

Adds primitive #5 (adaLN-Zero) on top of Pact-NeRV-A1 OR Pact-NeRV-FOE:

```python
class AdaLNZeroBlock(nn.Module):
    """DiT canonical adaLN-Zero adapted to NeRV conv decoder.

    Sister symposium (op-routable #8) must validate adaptation correctness.
    """
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        # Zero-init the modulation: model starts identity
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)
```

### Pact-NeRV-FULL (Stage 5, ~1500 LOC)

Composes all 6 primitives. STAGE 5 ONLY proceeds conditional on Stage 4 success per Section 13 staging discipline.

---

## 6. eval_roundtrip integration (primitive #6) — deep-dive

See Section 12 — deferred to its own section per the task spec.

---

## 7. Per-primitive HARD-EARNED-vs-CARGO-CULTED classification

See `council_assumption_adversary_verdict` block in frontmatter (9 assumptions classified) + Section 4 per-primitive table.

Summary:
- **3 HARD-EARNED** (primitives 1, 2, 3) — composable for Stage 2 Pact-NeRV-A1
- **1 HARD-EARNED-THEORETICALLY** (primitive 4) — Stage 3 prerequisite = first empirical anchor
- **1 HARD-EARNED-LITERATURE / UNCLEAR-FOR-NERV** (primitive 5) — Stage 4 prerequisite = lit-survey + probe
- **1 HARD-EARNED-AS-DEFAULT / CARGO-CULTED-AS-SYNERGY** (primitive 6) — reframe as discipline not primitive

---

## 8. 9-dimension success checklist evidence per Catalog #294

| Dimension | Evidence |
|---|---|
| **1. UNIQUENESS** | Pact-NeRV is the FIRST custom paper-worthy variant that EXPLICITLY composes apparatus canonical equations + canonical helpers + literature-grounded conditioning primitives + eval_roundtrip integration into a single unified design. No sister substrate has this composition surface. |
| **2. BEAUTY + ELEGANCE** | Stage 1 IA3 is 50 LOC; Stage 2 A1 is 600 LOC (matches PR101 GOLD 605 LOC); Stage 5 FULL is 1500 LOC (substrate-engineering-class per HNeRV parity L7). Per-Stage LOC budget is reviewable in 30 seconds. |
| **3. DISTINCTNESS** | Distinct from Z6 (single-layer FiLM on pose only), Z6-v2 (multi-layer FiLM on pose only), ego_nerv (lane-level pose-FiLM only), HNeRV-class (content-adaptive embedding learned from scratch). The DISTINCT contribution = explicit conditioning on apparatus-measured signals (per-pair difficulty + per-class chroma) rather than implicit learning. |
| **4. RIGOR** | Premise verification per Catalog #229 (FILM-FAMILY-RESEARCH + sister symposiums read in full); adversarial review per Catalog #292 (8 verbatim dissent positions); HARD-EARNED-vs-CARGO-CULTED classification per Catalog #303 (9 assumptions classified); per-primitive Section 4 audit. |
| **5. OPTIMIZATION PER TECHNIQUE** | Per-LOC bang-per-buck staged validation per Carmack verdict. Stage 1 IA3 establishes per-LOC baseline; subsequent stages must beat baseline to justify their LOC cost. |
| **6. STACK-OF-STACKS COMPOSABILITY** | Section 13 composability matrix predicts α per Pact-NeRV variant vs PR110 fec6 + STC sidecar + Z6 + Z7-Mamba-2 + ATW V2 + Riemannian-Newton. Predictions are LITERATURE-INSPIRED; empirical validation per Catalog #322 cascade. |
| **7. DETERMINISTIC REPRODUCIBILITY** | All Pact-NeRV variants inherit canonical seed-pinning + byte-stable archive grammar from sister TCNeRV pattern + canonical inflate runtime per Catalog #205. Per CLAUDE.md "Canonical pipeline standard" — all training via `experiments/pipeline.py` profiles. |
| **8. EXTREME OPTIMIZATION + PERFORMANCE** | Tier 1 engineering (autocast_fp16 + TF32 + torch.compile + no_grad-at-eval + canonical scorer-loss helper) declared in every Pact-NeRV trainer per Catalog #270 dispatch optimization protocol umbrella. eval_roundtrip integration per primitive #6 / Section 12 closes proxy-auth gap. |
| **9. OPTIMAL MINIMAL CONTEST SCORE** | Predicted ΔS band per Section 10 + Dykstra-feasibility check. Stage 1 = plateau_adjacent ([-0.001, +0.001] band); Stage 5 = frontier_pursuit with predicted band deferring to per-Stage post-training Tier-C validation per Catalog #324. |

---

## 9. Observability surface per Catalog #305

The 6-facet observability declaration:

1. **Inspectable per layer.** Every PactNeRVA1Block / AdaLNZeroBlock exposes (a) FiLM γ + β tensors per layer; (b) IA3 γ tensor per layer; (c) CLADE per-class lookup table; (d) FOE per-pixel modulation map. Xray-style hooks per layer for runtime capture.
2. **Decomposable per signal.** Composite ΔS decomposes into per-primitive contributions via Section 10 axis attribution + sister Catalog #356 per-axis decomposition emission. Each cathedral consumer for Pact-NeRV variants will emit per-axis decomposition with canonical Provenance.
3. **Diff-able across runs.** Per-pair γ + β tensors are byte-stable across runs (deterministic seed); two runs of same variant produce byte-identical archives.
4. **Queryable post-hoc.** Per-Stage Modal dispatch result includes full archive + inflate runtime + per-pair conditioning tensors. Sister `tools/audit_substrate_canvas.py` would consume Pact-NeRV verdicts.
5. **Cite-able.** Every Pact-NeRV anchor cites (substrate / commit / call_id / config / random_seed / upstream_snapshot_sha256) per Catalog #245 modal_call_id_ledger.
6. **Counterfactual-able.** Per-Stage byte-mutation discipline (Catalog #139 packet compiler + Catalog #272 distinguishing-feature contract + Catalog #105 no-op detector) allows asking "what if this byte changed?" without re-running training.

---

## 10. Predicted ΔS band per Catalog #296 + Dykstra-feasibility check

Per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check", every predicted ΔS band carries a Dykstra-feasibility intersection check OR first-principles citation OR probe-disambiguator path.

### Stage 1 — Pact-NeRV-IA3

- **Predicted band:** `[-0.002, +0.001]` on contest-CPU at 0.192 frontier (sister symposium on Z6 IA3 ablation pending)
- **Dykstra-feasibility check:** the IA3 modification is per-LAYER γ-replacement; archive grammar unchanged; rate term changes ONLY via reduced conditioning byte count (~5-10 KB savings on 178517-byte archive = ~0.003 rate contribution). The seg + pose contributions can move at most O(0.001) given the canonical equation `categorical_blahut_arimoto_rate_distortion_v1` bound for low-dim conditioning. The 3-axis polytope intersection (rate + seg + pose) supports the predicted band.
- **First-principles citation:** IA3 paper [third-party-empirical:Liu2022] empirically demonstrates competitive performance at <1% parameter count on language tasks.
- **probe-disambiguator path:** Pact-NeRV-IA3 vs Pact-NeRV-baseline-FiLM A/B ablation on Modal T4 IS the disambiguator for β-carries-signal hypothesis.
- **Validation status:** `pending_post_training`. Per Catalog #324: post-training Tier-C validation required on landed archive before consumption.

### Stage 2 — Pact-NeRV-A1

- **Predicted band:** `[-0.005, +0.001]` on contest-CPU at 0.192 frontier (HARD-EARNED-stack ADDITIVE composition prediction with bounded uncertainty given PR95Author's implicit-vs-explicit dissent)
- **Dykstra-feasibility check:** the 3-primitive composition (multi-layer FiLM on pose + per-pair difficulty modulation + per-class CLADENorm) operates on ORTHOGONAL axes per FILM-FAMILY-RESEARCH composability matrix Section 10 (ADD/ADD/ADD). The composability matrix predictions are LITERATURE-INSPIRED; empirical α validation per Catalog #322 cascade.
- **First-principles citation:** Atick-Redlich cooperative-receiver (per-class CLADENorm); HNeRV content-adaptive embedding (per-pair difficulty as explicit form); TeNeRV multi-layer FiLM (per-stage modulation).
- **probe-disambiguator path:** per-Stage A/B ablation with each primitive enabled/disabled at Stage 2 IS the disambiguator for HARD-EARNED-stack ADDITIVE claim.

### Stage 3 — Pact-NeRV-A1 + Pact-NeRV-FOE compose

- **Predicted band:** `[-0.008, -0.001]` on contest-CPU (PREREQUISITE = ego_motion_concentration_prior_v1 first anchor; predicted band reflects Atick-Redlich + Ballard theoretical bound)
- **Dykstra-feasibility check:** FOE foveation is SUB-ADDITIVE with pose-conditioned FiLM per FILM-FAMILY-RESEARCH composability matrix (both condition on pose; capacity overlap). The intersection of the 4-axis polytope (rate + seg + pose + FOE-derived) is non-empty per cooperative-receiver framing but the achievable region is narrowed by SUB-ADDITIVE composition.
- **First-principles citation:** Atick-Redlich 1990 + Gibson 1950 + Ballard embodied-vision lens; canonical equation `ego_motion_concentration_prior_v1` (first anchor required).

### Stage 4 — 4-primitive composition + adaLN-Zero NeRV adaptation

- **Predicted band:** DEFERRED to lit-survey + probe sister symposium (op-routable #8)
- **Dykstra-feasibility check:** adaLN-Zero adaptation to conv decoders is UNKNOWN at apparatus; cannot intersect feasibility without first empirical probe

### Stage 5 — Pact-NeRV-FULL 6-primitive composition

- **Predicted band:** DEFERRED to Stage 5 dispatch (per Contrarian + Assumption-Adversary VETO on pre-empirical 6-primitive band prediction; sister NSCS06 v6 553x-miss anchor)

---

## 11. FILM-FAMILY-RESEARCH integration

Per the operator-routed dependency on commit `9a95d1daf` (FILM-FAMILY-RESEARCH memo at `.omx/research/film_family_alternatives_bleeding_edge_research_20260520T184150Z.md`):

### Top-K findings integrated into Pact-NeRV design

1. **Multi-layer FiLM > single-layer FiLM** → primitive #1 (Pact-NeRV-A1 building block #1). Pact-NeRV uses MULTI-LAYER FiLM per HNeRV ablation + TeNeRV third-party-empirical receipts.
2. **Per-pair difficulty-conditioned modulation HARD-EARNED apparatus-native** → primitive #2 (Pact-NeRV-A1 building block #2). Apparatus equivalent of HNeRV content-adaptive embedding via canonical equation signals.
3. **adaLN-Zero (DiT) beats cross-attention for low-dim conditioning** → primitive #5 (Stage 4 sister symposium prerequisite). NOTE: this finding's NeRV applicability is UNCLEAR; cross-attention is NOT a Pact-NeRV primitive (Hinton dissent suggests it as 7th primitive sister symposium).
4. **HyperNetwork-from-ego-pose CARGO-CULTED-MAY-BE-PROMISING** → NOT included in Pact-NeRV-FULL per Assumption-Adversary verdict (CARGO-CULTED status disqualifies from current symposium scope). Sister symposium per FILM-FAMILY-RESEARCH Recommendation #4 deferred to operator routing.
5. **Per-byte sensitivity HARD-EARNED-NEGATIVE** → EXPLICITLY EXCLUDED from Pact-NeRV per FILM-FAMILY-RESEARCH Section 8.4 + canonical equation `per_byte_leverage_uniformly_distributed_v1` (top-K concentration LESS effective than uniform allocation; CONSEQUENCE not useful CONDITIONING signal).

### 5 Pact-NeRV variants from FILM-FAMILY-RESEARCH

Mapped to Pact-NeRV-FULL Stage 1-5 staging:

| FILM-RESEARCH Variant | Pact-NeRV Stage | LOC | Recommendation Status |
|---|---|---|---|
| Pact-NeRV-IA3 (γ-only rate-extremal) | Stage 1 | ~50 | PROCEED (Hotz + Carmack) |
| Pact-NeRV-A1 (pose+difficulty+class triple) | Stage 2 | ~600 | PROCEED conditional on Stage 1 |
| Pact-NeRV-FOE (foveation FOE) | Stage 3 | ~500 | DEFER (PREREQUISITE = ego_motion_concentration_prior_v1 anchor) |
| Pact-NeRV-DT (adaLN-Zero) | Stage 4 | ~400 | DEFER (PREREQUISITE = NeRV adaptation lit-survey + probe) |
| Pact-NeRV-HN (HyperNetwork) | OUT-OF-SCOPE | ~800 | DEFER (CARGO-CULTED-MAY-BE-PROMISING; separate sister symposium) |

### Composability matrix (FILM-FAMILY-RESEARCH Section 10 adopted)

|  | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1. Multi-layer FiLM (pose) | — | ADD | ADD | SUB-ADD | ADD |
| 2. Per-pair difficulty modulation | ADD | — | ADD | ORTH | ADD |
| 3. Per-class CLADENorm | ADD | ADD | — | ORTH | ADD |
| 4. FOE-foveation | SUB-ADD | ORTH | ORTH | — | ADD |
| 5. adaLN-Zero | ADD | ADD | ADD | ADD | — |

The matrix predicts ADDITIVE for the Pact-NeRV-A1 triple (primitives 1+2+3) and SUB-ADDITIVE for Pact-NeRV-FOE composed with pose-conditioned FiLM (capacity overlap). Per Catalog #322 cascade: empirical α validation REQUIRED before consumption by autopilot ranker.

---

## 12. eval_roundtrip primitive #6 deep-dive per CLAUDE.md non-negotiable

Per CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE, HIGHEST EMPHASIS":

> EVERY training path MUST use eval_roundtrip. There are ZERO exceptions. ... Without eval_roundtrip, proxy-auth gap is 2-6x on PoseNet.

### Empirical receipts

- **PR95** [apparatus-empirical:PR95]: integrated eval_roundtrip INSIDE the training inner loop + monkey-patched `rgb_to_yuv6` to be differentiable (the upstream helper is `@torch.no_grad()` / in-place and otherwise severs PoseNet gradients).
- **PR106** [apparatus-empirical:PR106]: confirmed PR95 pattern + extended to renderer.bin sidecars.
- **Apparatus helper:** `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` + `patch_upstream_yuv6_globally` + `load_differentiable_scorers` + `differentiable_rgb_to_yuv6`.

### Canonical helper API per Catalog #229

```python
from tac.differentiable_eval_roundtrip import (
    apply_eval_roundtrip_during_training,
    patch_upstream_yuv6_globally,
    load_differentiable_scorers,
    differentiable_rgb_to_yuv6,
)

# Per Pact-NeRV trainer (every variant Stage 1-5):
patch_upstream_yuv6_globally()  # MUST run before scorer construction
pose_scorer, seg_scorer = load_differentiable_scorers(device=device)

# Inside training inner loop:
def train_step(pact_nerv_decoder, frames_gt, embeds, ego_pose, difficulty, seg_mask):
    frames_pred = pact_nerv_decoder(embeds, ego_pose, difficulty, seg_mask)
    # eval_roundtrip BEFORE scorer forward
    frames_pred_roundtrip = apply_eval_roundtrip_during_training(frames_pred)
    # Score on roundtripped frames (uint8 bottleneck simulated)
    seg_loss = seg_scorer(frames_pred_roundtrip, ...)
    pose_loss = pose_scorer(frames_pred_roundtrip, ...)
    return score_aware_loss_via_canonical_helper(seg_loss, pose_loss, ...)
```

### Integration per Pact-NeRV stage

- **Stage 1 (IA3):** eval_roundtrip MUST be active (canonical helper invocation in trainer)
- **Stage 2 (A1):** eval_roundtrip MUST be active + differentiable rgb_to_yuv6 patch
- **Stage 3 (FOE):** eval_roundtrip MUST be active + FOE foveation map propagates through roundtrip
- **Stage 4 (DT):** eval_roundtrip MUST be active + adaLN-Zero zero-init preserves roundtrip-compatibility
- **Stage 5 (FULL):** eval_roundtrip MUST be active + all 6 primitives' gradients flow through roundtrip

### Reframing primitive #6 per Assumption-Adversary verdict

Primitive #6 is NOT an additive primitive with synergy multiplier; it IS CROSS-CUTTING DEFAULT DISCIPLINE per CLAUDE.md non-negotiable. The symposium ACCEPTS the discipline as MANDATORY DEFAULT in every Pact-NeRV stage; REJECTS the framing as "6th primitive" inflating primitive count without empirical synergy justification.

The Pact-NeRV design is more accurately described as **5 conditioning primitives + 1 cross-cutting eval_roundtrip discipline**. Stage 6 paper-write should reframe accordingly.

---

## 13. Stack-of-stacks composability matrix per operator directive + HYBRID staged reactivation criteria

### Composability matrix vs 7+ sister candidates

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" + Catalog #322 composition_alpha cascade. ALL predictions are LITERATURE-INSPIRED; empirical α validation REQUIRED per Catalog #322 before consumption.

| Pact-NeRV Variant | vs PR110 fec6 | vs STC sidecar | vs Z6 | vs Z7-Mamba-2 | vs ATW V2 | vs Riemannian-Newton | vs UNIFIED-PRETRAIN-ABLATE |
|---|---|---|---|---|---|---|---|
| Pact-NeRV-IA3 | ORTH | ORTH | SUB-ADD (replaces Z6 FiLM) | ORTH | ORTH | ORTH | DEFER |
| Pact-NeRV-A1 | ORTH | ADD | SUB-ADD (subsumes Z6) | ORTH | ADD | ADD | DEFER |
| Pact-NeRV-FOE | ORTH | ADD | ADD | ADD | ADD | ADD | DEFER |
| Pact-NeRV-DT | ORTH | ADD | ADD | SUB-ADD | ADD | ADD | DEFER |
| Pact-NeRV-FULL | ORTH | ADD | SUB-ADD (subsumes Z6) | ADD | ADD | ADD | DEFER |

Notes:
- **ORTH** = orthogonal (independent axes; ADDITIVE α ≈ 1.0 predicted)
- **ADD** = additive (composable; α ≈ 1.0 - 1.2 predicted per FILM-RESEARCH matrix)
- **SUB-ADD** = sub-additive (capacity overlap; α ≈ 0.5 - 0.8 predicted)
- **DEFER** = NOT YET MEASURED (sister #882 UNIFIED-PRETRAIN-ABLATE verdict DEFERRED full 5-primitive composition)
- **PR110 fec6** = our current 0.192 [contest-CPU] frontier substrate; ORTH because fec6 is byte-level frame-selector + Huffman entropy coding, not architectural
- **STC sidecar** = sister #857 PROCEED verdict; ADD because STC's per-pair structural conditioning is sister of Pact-NeRV's per-pair difficulty conditioning

### HYBRID staged reactivation criteria

Per CLAUDE.md "Forbidden premature KILL" + sister UNIFIED-PRETRAIN-ABLATE staging precedent + Contrarian + Assumption-Adversary VETO on PROCEED-unconditional:

#### Stage 1 PRIORITY 1: Pact-NeRV-IA3

- **Cost:** $0.30 Modal T4 smoke
- **LOC:** ~50 (modification of any deployed FiLM substrate)
- **Reactivation criterion if STAGE 1 SHOWS REGRESSION:** investigate β-carries-signal hypothesis; document as IA3-disconfirmed; proceed to Stage 2 with FiLM γ+β intact
- **Reactivation criterion if STAGE 1 SHOWS NEUTRAL:** β-noise hypothesis confirmed; Pact-NeRV-FULL can use IA3 γ-only globally (further LOC savings)
- **Reactivation criterion if STAGE 1 SHOWS IMPROVEMENT:** FiLM-overcapacity hypothesis confirmed; Pact-NeRV-A2 = IA3 variant is the substrate; proceed to Stage 2 with IA3 (not FiLM)

#### Stage 2 PRIORITY 1: Pact-NeRV-A1 (conditional on Stage 1)

- **Cost:** $0.30 Modal T4 smoke
- **LOC:** ~600 (substrate-engineering class per HNeRV parity L7)
- **Reactivation criterion if STAGE 2 SHOWS composition_alpha < 0.5:** revert to single best primitive (1, 2, or 3); DEFER Stage 3+
- **Reactivation criterion if STAGE 2 SHOWS composition_alpha 0.5-0.7:** SUB-ADDITIVE; investigate capacity-overlap hypothesis (which primitive pair overlaps); proceed to Stage 3 with caution
- **Reactivation criterion if STAGE 2 SHOWS composition_alpha > 0.7:** ADDITIVE confirmed; proceed to Stage 3

#### Stage 3 PRIORITY 2: Pact-NeRV-A1 + Pact-NeRV-FOE compose (conditional on Stage 2 + prerequisite)

- **Cost:** $2-5 Modal A10G smoke
- **PREREQUISITE:** `ego_motion_concentration_prior_v1` canonical equation first empirical anchor lands via sister symposium per Catalog #311 (op-routable #7)
- **LOC:** ~500 additional (Pact-NeRV-FOE addition to Pact-NeRV-A1)
- **Reactivation criterion if STAGE 3 SHOWS regression vs Stage 2:** FOE foveation hypothesis disconfirmed for our scorer; DEFER Stage 4+ with FOE; explore Stage 4 sister with adaLN-Zero only
- **Reactivation criterion if STAGE 3 SHOWS improvement:** proceed to Stage 4

#### Stage 4 PRIORITY 2: 4-primitive composition + adaLN-Zero (conditional on Stage 3 + prerequisite)

- **Cost:** $5-15 Modal A100 smoke
- **PREREQUISITE:** adaLN-Zero adaptation to NeRV (conv+upsample, not transformer) lit-survey + empirical probe via sister symposium (op-routable #8)
- **LOC:** ~400 additional
- **Reactivation criterion if STAGE 4 SHOWS regression:** DEFER adaLN-Zero; document as DiT-NeRV-mismatch; proceed to Stage 5 with 4 primitives only

#### Stage 5 PRIORITY 3: Full 6-primitive Pact-NeRV-FULL (conditional on Stage 4 + race-window OR paper-deadline)

- **Cost:** $20-50 Modal A100 full training
- **PREREQUISITE:** Hinton 7th-primitive cross-attention to scorer features may be inserted via sister symposium (op-routable #9) BEFORE Stage 5
- **LOC:** ~1500 total (substrate-engineering class)
- **Reactivation criterion if STAGE 5 SHOWS regression:** sister of UNIFIED-PRETRAIN-ABLATE 5-primitive verdict — DEFERRED; paper contribution via staged ablation Stage 6

#### Stage 6 paper-write

- **Cost:** $0 (research artifact)
- **DELIVERABLE:** even if Pact-NeRV-FULL doesn't beat HNeRV/PR101, the staged ablation IS the paper contribution; documents per-primitive ΔS attribution + composability_alpha measurements + cooperative-receiver framing applied to dashcam compression

### Race-mode rigor inversion contingency

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first": if the contest leaderboard moves significantly while Stage 1-2 are pending, the staged path INVERTS to parallel-dispatch ALL 5 Pact-NeRV variants as 30-sec-reviewable bolt-ons. The HYBRID staged path is the DEFAULT (no race); the parallel-dispatch is the OPERATOR OVERRIDE per Catalog #300 Mission Alignment Consequence 1 (operator-frontier-override at ALL tiers).

---

## 14. Canonical-vs-unique decision per layer (Catalog #290 sister)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290 STRICT preflight gate:

| Layer | Canonical-vs-unique decision | Rationale |
|---|---|---|
| Loss function (scorer-aware composition) | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.substrates._shared.score_aware_common.score_pair_components` is canonical per Catalog #164; per-substrate optimal scoring uses the canonical formula. |
| Inflate runtime device-fork | ADOPT_CANONICAL_BECAUSE_SERVES | `select_inflate_device` canonical helper per Catalog #205; Pact-NeRV variants use canonical helper. |
| Auth-eval CLI invocation | ADOPT_CANONICAL_BECAUSE_SERVES | `gate_auth_eval_call` canonical per Catalog #226; Pact-NeRV trainers route through canonical helper. |
| Modal dispatch optimization protocol | ADOPT_CANONICAL_BECAUSE_SERVES | Tier 1 (autocast_fp16 + TF32 + torch.compile + no_grad-at-eval) + Tier 2 (NVML env block + min_vram_gb + min_smoke_gpu) + Tier 3 (canonical scorer-loss helper) per Catalog #270. |
| eval_roundtrip integration | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.differentiable_eval_roundtrip` canonical per CLAUDE.md non-negotiable; mandatory default. |
| Canonical equations consumption | ADOPT_CANONICAL_BECAUSE_SERVES | `per_pair_master_gradient_score_impact_taylor_v1` + `per_frame_difficulty_atlas_v1` + `per_segnet_class_chroma_priors_v1` + `ego_motion_concentration_prior_v1` per Catalog #344. |
| Canonical Provenance | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.provenance.audit_score_claim_dict` per Catalog #323 META-class umbrella; every Pact-NeRV anchor carries canonical Provenance. |
| Cathedral consumer canonical contract | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.cathedral.consumer_contract.validate_consumer_module` per Catalog #335 paradigm; Pact-NeRV cathedral consumers auto-discovered. |
| FiLM modulation primitive (γ + β per-feature affine) | ADOPT_CANONICAL_BECAUSE_SERVES | `src/tac/renderer.py::FiLMLayer` is canonical implementation; Pact-NeRV variants use canonical helper. |
| Multi-layer FiLM composition | FORK_BECAUSE_SUPPRESSES | Single-layer FiLM canonical SUPPRESSES the substrate's optimal score per HNeRV ablation [third-party-empirical]; Pact-NeRV-A1 FORKS to multi-layer FiLM. |
| Per-pair difficulty MLP conditioning | FORK_BECAUSE_PRINCIPLED_MISMATCH | No canonical per-pair-difficulty conditioning MLP exists; Pact-NeRV-A1 FORKS to implement (substrate-engineering-class per HNeRV parity L7). |
| Per-class CLADENorm | ADOPT_CANONICAL_BECAUSE_SERVES | `src/tac/renderer.py::CLADENorm` is canonical; Pact-NeRV-A1 uses canonical helper. |
| FOE foveation conditioning | FORK_BECAUSE_PRINCIPLED_MISMATCH | No canonical FOE-conditioning exists; Pact-NeRV-FOE FORKS to implement (PREREQUISITE = canonical equation first anchor). |
| adaLN-Zero modulation | UNCLEAR_NEEDS_EMPIRICAL | No canonical adaLN-Zero exists; DiT adaptation to NeRV requires lit-survey + probe (sister symposium PREREQUISITE). |
| 6-primitive composition framework | FORK_BECAUSE_PRINCIPLED_MISMATCH | No canonical 6-primitive composer exists; Pact-NeRV-FULL FORKS to implement (substrate-engineering-class). |

---

## 15. Per-substrate symposium contract checklist per Catalog #325

| Step | Requirement | Status |
|---|---|---|
| 1 | Cargo-cult audit per Catalog #303 (`## Cargo-cult audit per assumption` section) | DONE — frontmatter + Section 4 + Section 7 |
| 2 | 9-dimension success checklist evidence per Catalog #294 (`## 9-dimension success checklist evidence` section) | DONE — Section 8 |
| 3 | Observability surface declaration per Catalog #305 (`## Observability surface` section) | DONE — Section 9 |
| 4 | Sextet pact deliberation (6-of-6 INNER) + grand council attendees per topic | DONE — 14 INNER + 13 GRAND topical = 27 attendees; roster validate `complete=True` |
| 5 | Per-substrate reactivation criteria pinned | DONE — Section 13 staged criteria per Stage |
| 6 | Catalog #324 post-training Tier-C validation discipline | DONE — `predicted_band_validation_status: pending_post_training` in frontmatter |

---

## 16. Cross-references

- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable (parent discipline)
- CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable (sister Catalog #315)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (Section 14)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L7 (substrate-engineering vs bolt-on)
- CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE, HIGHEST EMPHASIS" (Section 12)
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" (Section 13 contingency)
- CLAUDE.md "Forbidden premature KILL" (Section 13 reactivation criteria)
- CLAUDE.md "Apples-to-apples evidence discipline" (all evidence tags carry [literature-prediction] or [apparatus-empirical:...] or [third-party-empirical:...])
- CLAUDE.md "Council conduct" (sextet pact + Assumption-Adversary seat + 4-co-lead structure)
- Catalog #229 PV / #270 dispatch protocol / #287 placeholder-rationale rejection / #292 per-deliberation assumption / #294 9-dim / #296 Dykstra-feasibility / #300 v2 frontmatter / #303 cargo-cult audit / #305 observability / #309 horizon-class / #313 probe-outcomes / #322 composition_alpha / #323 canonical Provenance / #324 post-training Tier-C / #325 per-substrate symposium / #340 sister-checkpoint / #346 canonical roster
- `.omx/research/film_family_alternatives_bleeding_edge_research_20260520T184150Z.md` (FILM-FAMILY-RESEARCH primary input)
- `.omx/research/council_per_substrate_symposium_unified_pretrain_ablate_dp1_mae_v_saug_imp_qat_schema_elision_20260520T175410Z.md` (sister UNIFIED-PRETRAIN-ABLATE staging precedent)
- `.omx/research/stc_paradigm_reformulation_a1_residual_disambiguator_synthesis_20260520T165252Z.md` (probe-disambiguator framework)
- `.omx/research/council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517.md` (Z6-v2 sister multi-layer FiLM)
- `.omx/research/council_per_substrate_symposium_dp1_deep_dive_20260517.md` (DP1 pretraining sister)
- `.omx/research/council_per_substrate_symposium_tt5l_foveation_lapose_20260517.md` (FOE foveation sister)
- `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md` (cooperative-receiver sister)
- `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` (operating mode anchor)
- `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md` (classification framework)
