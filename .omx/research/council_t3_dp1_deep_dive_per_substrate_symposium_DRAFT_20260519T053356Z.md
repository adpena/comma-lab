---
schema: council_deliberation_v2
deliberation_id: council_t3_dp1_deep_dive_per_substrate_symposium_DRAFT_20260519
topic: "DP1 (Pretrained Driving Prior) deep-dive per-substrate symposium DRAFT — Comma2k19 codebook distillation + dual-stacking composition (composition + training-time prior); supersedes prior DP1 symposiums with T3 DRAFT for operator-routable ratification"
review_kind: per_substrate_optimal_form_symposium_T3_grand_council_DRAFT
review_date: "2026-05-19"
lane_id: lane_cable_c_substrate_symposium_draft_batch_20260519
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Hinton, Hafner, Atick, Wyner, Tishby_memorial, Quantizr, Hotz, Schmidhuber, MacKay_memorial]
council_quorum_met: false
council_verdict: DRAFT_PENDING_OPERATOR_CONVOCATION
council_dissent:
  - member: Contrarian
    verbatim: "DRAFT only — supersedes prior DP1 symposium memos. CRITICAL anchor: PATH 1 composition rate arithmetic correction 2026-05-18 (per `dp1_pr101_composition_noop_probe_20260518_codex.json`): L1 packet overhead is +25,827 bytes / +0.017197 score, NOT the older +0.0000172. PATH 1 with prior_strength=0.0 produces CONTROL at 0.209247 (NOT 0.19207). Any L2 prior-effect path MUST buy back > 0.017197 score before promotion-eligible. The reformulation question is whether DP1 prior IS empirically validated as net-positive after correcting rate arithmetic."
  - member: Hotz
    verbatim: "DP1 has TWO canonical paths: PATH 1 composition (DP1 + fec6 archive composition via tac.substrates.pretrained_driving_prior.composition.compose_with) and PATH 2 training-time prior (DP1 codebook seeds renderer init). PATH 1 corrected rate +0.017197 means PATH 1 IS NO-OP unless prior-effect dominates. PATH 2 is theoretically less rate-overhead BUT untested. Operator-routable: which PATH deserves Wave 1 priority?"
council_assumption_adversary_verdict:
  - assumption: "DP1 codebook distillation from Comma2k19 OOD dashcam data IS canonical (per CLAUDE.md HNeRV parity L1)"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md HNeRV parity L1 + Catalog #209 + Catalog #213: codebook MUST be distilled from OOD dashcam data, NEVER from contest video. Comma2k19 IS canonical OOD source. Sister Catalog #213 strict gate enforces canonical Comma2k19LocalCache routing structurally."
  - assumption: "DP1 + fec6 composition rate +0.017197 IS empirically baked in"
    classification: HARD-EARNED-EMPIRICAL
    rationale: "Per 2026-05-18 `dp1_pr101_composition_noop_probe_20260518_codex.json` byte-closed structural proof: composed archive 204344 bytes; base fec6 178517; DP1 prefix 25814; DPCOMP header 13; total L1 overhead 25827; rate delta 25 × 25827 / 37545489 = +0.017197. Empirical proof. PATH 1 control with prior_strength=0.0 IS 0.209247 (+0.017197 above 0.19205 fec6 baseline). PATH 1 is structurally rate-positive."
  - assumption: "PATH 2 (training-time prior) is theoretically less rate-overhead than PATH 1 (composition)"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "PATH 2 seeds renderer init from DP1 codebook; codebook bytes are NOT shipped in archive (regenerated at inflate via codebook-seeded init). Theoretically 0 archive overhead — BUT empirically untested. Wave 1 PATH 2 smoke MUST verify renderer-converges-from-codebook-init AND archive size matches PATH 0 baseline."
  - assumption: "Predicted ΔS [-0.012, -0.004] per Deep-Research wave Top-4 is calibrated"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Per Deep-Research wave 2026-05-18 Top-4: DP1+PR101 predicted ΔS [-0.012, -0.004]. Lower bound = -0.012 from theoretical codebook informativeness; upper bound = -0.004 from sister Quantizr 0.33 lineage. Per Catalog #324: predicted_band_validation_status MUST be pending_post_training; per Catalog #316: comparison vs canonical frontier 0.19205 [contest-CPU] / 0.20533 [contest-CUDA] AT smoke-time."
  - assumption: "PATH 1 prior_strength sweep ($10-15 envelope) is sufficient disambiguator"
    classification: HARD-EARNED-PARTIAL
    rationale: "Sister 2026-05-17 dp1_dual_stacking_design memo specifies PATH 1 with prior_strength ∈ {0.1, 0.3, 0.5, 0.7, 1.0} sweep. PATH 1 IS rate-positive +0.017197; prior-effect MUST buy back > 0.017197 score. Wave 1 PATH 1 sweep at $10-15 Modal T4 5-config empirical anchor. Operator-routable: if NO prior_strength config beats +0.017197, PATH 1 retire to research_only=true."
council_decisions_recorded:
  - "DRAFT enumerates 6-step Catalog #325 contract for DP1 deep-dive"
  - "PATH 1 composition rate arithmetic corrected: +0.017197 (not +0.0000172)"
  - "PATH 2 training-time prior is canonical alternative with theoretically 0 archive overhead"
  - "Operator-routable: PATH 1 vs PATH 2 priority + ratification mechanism choice"
  - "Sister Catalog #213 enforces Comma2k19LocalCache canonical routing structurally"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: pretrained_driving_prior
substrate_aliases:
  - dp1_pretrained_driving_prior
  - dp1_dual_stacking
  - pretrained_driving_prior_substrate
deferred_substrate_retrospective_due_utc: "2026-06-18T05:33:56Z"
horizon_class: frontier_pursuit
predicted_band: [0.180, 0.188]
predicted_band_validation_status: pending_post_training
score_claim: false
promotion_eligible: false
dispatch_enabled: false
research_only: true
canonical_frontier_anchor:
  contest_cpu: "0.1920513169 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
  contest_cuda: "0.2053300290 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
predecessor_probe_outcomes:
  - probe_id: dp1_pr101_composition_noop_probe_20260518
    verdict: STRUCTURAL_PROOF_RATE_CORRECTED
    notes: "Composed archive SHA-256: 507d2a000ecf5a... archive bytes 204344; base fec6 178517; DP1 prefix 25814; DPCOMP header 13; total L1 overhead 25827; rate delta +0.017197139182"
related_deliberation_ids:
  - council_per_substrate_symposium_dp1_deep_dive_20260517
  - pretrained_driving_prior_lane_scaffold_landed_20260513
  - dp1_dual_stacking_design_20260517
  - dp1_pr101_composition_noop_probe_20260518_codex
  - dp1_engineering_status_gate_20260515_codex
  - codex_routing_directive_dp1_pr101_op1_op2_zero_cost_probes_20260518
  - lane_dp1_comma2k19_autoload_log_incremental_20260514_directive_layered_chunking_efficiency_oss_20260514
  - feedback_deep_research_wave_landed_20260518
---

# DRAFT: T3 grand council symposium — DP1 (Pretrained Driving Prior) deep-dive

**Status**: DRAFT — operator-convocation pending. NOT a binding council verdict.
**Lane**: `lane_cable_c_substrate_symposium_draft_batch_20260519` L1
**Per Catalog #325**: this DRAFT satisfies the 6-step contract structurally; full convocation activates symposium evidence per Catalog #325 14-day window.
**Supersession**: this DRAFT supersedes 2026-05-17 dp1_dual_stacking_design memo + 2026-05-17 per-substrate symposium + 2026-05-18 PATH 1 rate-arithmetic correction memo by re-elevating to T3 DRAFT format with PATH 1 corrected arithmetic + PATH 2 sister option + Deep-Research wave Top-4 predicted band integration.

## Symposium attendees (proposed)

**Sextet pact**:
- **Shannon LEAD** — information-theoretic capacity of DP1 codebook as decoder side-info
- **Dykstra CO-LEAD** — convex-feasibility of PATH 1 (composition) + PATH 2 (training-time prior)
- **Yousfi** — PoseNet/SegNet response to DP1-pretrained-init substrate
- **Fridrich** — inverse-steganalysis of pretrained-prior bit-allocation
- **Contrarian** — VETO on lazy PATH-1-rate-arithmetic-blindness
- **Assumption-Adversary** — challenges DP1 vs canonical PR101/PR106 frontier

**Grand council added per topic**:
- **Hinton** — canonical knowledge distillation authority (DP1 = codebook distillation from OOD Comma2k19)
- **Hafner** — DreamerV3 latent-dynamics-as-pretrained-prior framing
- **Atick** — cooperative-receiver canonical (codebook IS encoder-decoder shared prior)
- **Wyner** — Wyner-Ziv source-coding-with-side-info canonical
- **Tishby_memorial** — IB framework for codebook capacity
- **Quantizr** — Quantizr 0.33 lineage (sister codebook-distillation insights)
- **Hotz** — engineering complexity-vs-payoff
- **Schmidhuber** — compression-as-intelligence + transfer-learning
- **MacKay_memorial** — MDL framework for prior selection

## Step 1 — Cargo-cult audit per Catalog #303

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| CC-dp1-1 | "PATH 1 composition rate arithmetic is +0.0000172" | HARD-EARNED-EMPIRICALLY-CORRECTED | 2026-05-18 byte-closed structural proof CORRECTS to +0.017197 (1000x larger). PATH 1 IS rate-positive +0.017197; prior-effect MUST buy back > 0.017197 for promotion. |
| CC-dp1-2 | "DP1 codebook distillation from Comma2k19 IS canonical OOD source" | HARD-EARNED | Per CLAUDE.md HNeRV parity L1 + Catalog #209 + #213: codebook MUST be distilled from OOD dashcam data, NEVER from contest video. Catalog #213 strict gate enforces canonical Comma2k19LocalCache routing. |
| CC-dp1-3 | "PATH 2 training-time prior is theoretically 0 archive overhead" | CARGO-CULTED-PENDING-EMPIRICAL | Theoretically PATH 2 ships ONLY the renderer (init from DP1 codebook); archive bytes match PATH 0 baseline. Empirically untested. Wave 1 PATH 2 smoke MUST verify archive size match. |
| CC-dp1-4 | "DP1 codebook IS dense in dashcam scene-distribution coverage" | HARD-EARNED-PARTIAL | Quantizr 0.33 lineage shows that compact codebooks (88-94K params; sigma=15; qint_max=7) CAN cover dashcam scene-distribution. DP1 codebook size + structure derived from Comma2k19; sufficiency empirically untested at contest scale. |
| CC-dp1-5 | "Predicted ΔS [-0.012, -0.004] per Deep-Research wave Top-4 is calibrated" | CARGO-CULTED-PENDING-EMPIRICAL | Theoretical extrapolation from Quantizr lineage; sister codebook-based predicted bands have NOT yet landed at contest level. Wave 1 smoke MUST verify predicted band lower bound. |
| CC-dp1-6 | "DP1 + PR101 stacking IS additive ΔS" | CARGO-CULTED-PENDING-EMPIRICAL | NSCS06 v8 all-at-once → -78% IS canonical counter-example. PATH 1 / PATH 2 paired comparison at SAME archive bytes (or sister normalized comparison) IS Dykstra-feasibility disambiguator. |
| CC-dp1-7 | "Comma2k19LocalCache canonical routing IS the only acceptable codebook source" | HARD-EARNED | Per Catalog #213: bare URL downloads forbidden; Comma2k19LocalCache mandatory. Catalog enforces structurally. Binding. |

## Step 2 — 9-dimension success checklist evidence per Catalog #294

| # | Dimension | Per-symposium-DRAFT evidence |
|---|---|---|
| 1 | UNIQUENESS | ✓ DP1 = FIRST OOD-pretrained-prior substrate; sister to canonical frontier substrates (PR101 fec6 / PR106 format0d) which use NO pretrained prior. Frontier_pursuit class per Catalog #309. |
| 2 | BEAUTY + ELEGANCE | ✓ DP1 codebook distillation IS clean knowledge-distillation pattern per Hinton authority. PATH 1 composition: byte-closed compose_with; PATH 2 training-time prior: init renderer from codebook. Both implementations ≤ 600 LOC per HNeRV parity L7. |
| 3 | DISTINCTNESS | ✓ PATH 1 (composition) vs PATH 2 (training-time prior) — 2 architecturally orthogonal stacking patterns. Sister Z6/Z7/Z8 ego-conditioning + TT5L V2 foveation — DP1 IS ORTHOGONAL pretrained-prior axis. |
| 4 | RIGOR | ✓ THIS DRAFT + 2026-05-17 dual-stacking design + 2026-05-17 per-substrate symposium + 2026-05-18 PATH 1 rate-arithmetic correction probe + Deep-Research wave Top-4 + sister DP1 forensic-audit + sister Catalog #209/#210/#213 strict-gate. |
| 5 | OPTIMIZATION PER TECHNIQUE | ✓ Knowledge distillation from OOD dashcam IS the canonical pretrained-prior approach per Hinton lineage. Comma2k19LocalCache canonical helper per Catalog #213. |
| 6 | STACK-OF-STACKS-COMPOSABILITY | ✓ DP1 + PR101 fec6 (PATH 1 composition) + Z6/Z7/Z8 ego-conditioning + TT5L V2 + NSCS06 v8 Variant C + D1 SegNet (6 orthogonal axes possible). |
| 7 | DETERMINISTIC REPRODUCIBILITY | ✓ DP1 codebook byte-stable per Catalog #5; PATH 1 compose_with deterministic; PATH 2 codebook-seeded init deterministic with fixed seed. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | ✓ Wave 1 PATH 1 prior_strength sweep $10-15 (Modal T4 5-config); Wave 1 PATH 2 single-config $5; Wave 2 paired comparison $15-25. TOTAL DP1 envelope $30-45. |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | ✓ Predicted band [0.180, 0.188] STRICTLY BELOW canonical frontier 0.19205 — IF realized, DP1 IS frontier-breaking. PATH 1 MUST buy back +0.017197 rate; PATH 2 theoretically 0 rate overhead. |

## Step 3 — Observability surface declaration per Catalog #305

**Per-DP1-substrate observability**:
1. **Inspectable per layer**: PATH 1 = composed archive structure (fec6 + DP1 prefix + DPCOMP header) + prior_strength conditioning weights + per-pair codebook-conditioning influence on renderer; PATH 2 = renderer state init signature + codebook-seeded epoch-0 weights + per-epoch divergence-from-init trajectory
2. **Decomposable per signal**: per-config (PATH 1 prior_strength sweep) score-vs-rate trade + per-PATH paired comparison vs PATH 0 baseline
3. **Diff-able across runs**: PATH 1 prior_strength ∈ {0, 0.1, 0.3, 0.5, 0.7, 1.0} 6-config sweep at SAME archive bytes (corrected for +0.017197 rate); PATH 2 vs PATH 0 at SAME archive bytes
4. **Queryable post-hoc**: per-config Modal call_id ledger row per Catalog #245 + per-config probe-outcome ledger row per Catalog #313 + per-PATH build_manifest.json per Catalog #220
5. **Cite-able**: cite Hinton knowledge distillation + Comma2k19 Atomic Toronto dataset + sister Catalog #209/#210/#213 + Quantizr 0.33 lineage + 2026-05-18 PATH 1 rate-arithmetic correction probe
6. **Counterfactual-able**: "what if codebook size = small vs large?" + "what if prior_strength = 0.1 vs 0.5 vs 1.0?" + "what if PATH 2 init signal preserved or destroyed by epoch N?" — empirical-anchor matrix

## Step 4 — Sextet pact deliberation (DRAFT positions)

### Shannon LEAD position (DRAFT)

*"Operating-within assumption: DP1 codebook IS encoder-decoder shared prior (Wyner-Ziv canonical). The information-theoretic question is whether codebook conditioning provides bit-savings beyond uniform-prior baseline. PATH 1 composition: archive bytes +25,827 = rate +0.017197; prior-effect MUST buy back. PATH 2 training-time prior: 0 archive overhead theoretically; empirical disambiguator MUST verify. PROCEED on DRAFT design with mandatory paired comparison + corrected rate arithmetic."*

### Dykstra CO-LEAD position (DRAFT)

*"Operating-within assumption: PATH 1 = additive constraint composition (DP1 + fec6 archive constraints); PATH 2 = init-distribution constraint (codebook-seeded epoch-0 weights). PATH 1 + PATH 2 paired comparison at SAME archive bytes IS Dykstra-feasibility disambiguator. APPROVE DRAFT design with mandatory PATH 1 sweep + PATH 2 baseline."*

### Yousfi position (DRAFT)

*"PoseNet/SegNet response to DP1-pretrained-init substrate: codebook-seeded init may provide better initialization for scorer-aware optimization (sister Quantizr 0.33 lineage). PROCEED on PATH 2 DRAFT design with mandatory smoke-time MI probe."*

### Fridrich position (DRAFT)

*"Inverse-steganalysis: codebook-conditioned bit-allocation may put bits in scorer-canonical regions per Quantizr 0.33 lineage. PROCEED on DRAFT design."*

### Contrarian position (DRAFT)

*"Operating-within assumption: PATH 1 rate arithmetic corrected to +0.017197 means PATH 1 IS NO-OP unless prior-effect dominates. STRONG RECOMMENDATION: PATH 1 prior_strength sweep BUT cap envelope at $10; if NO config beats +0.017197 prior-effect, PATH 1 retire to research_only=true. PATH 2 theoretically less rate-overhead is canonical alternative; Wave 1 PATH 2 priority. VETO any DRAFT path that pre-authorizes PATH 1 dispatch WITHOUT PATH 2 paired comparison."*

### Assumption-Adversary position (DRAFT) [Catalog #291 + #292]

*"Operating-within assumption (META): DP1 OOD-pretrained-prior provides bit-savings beyond canonical PR101/PR106 frontier substrates which use NO pretrained prior. The SHARED ASSUMPTION is OOD-pretrained-prior IS a winning paradigm. PR101 fec6 (canonical CPU frontier 0.19205; NO pretrained prior) + PR106 format0d (canonical CUDA frontier 0.20533; NO pretrained prior) ARE HARD-EARNED counter-evidence. IF DP1 PATH 1 + PATH 2 BOTH fail to beat PR101/PR106 frontier, the OOD-pretrained-prior paradigm DEFER per Catalog #298 → reactivation = NEW codebook design OR sister Quantizr-class lineage transfer. THIS DRAFT MUST record the paradigm-level disambiguator question."* — VETO if not engaged.

### Hinton position (DRAFT)

*"Canonical knowledge distillation authority: DP1 codebook distillation from OOD Comma2k19 IS canonical knowledge-distillation pattern. Sister 2014 Hinton-Vinyals-Dean distillation paper canonical KL-T=2.0 may apply at codebook-distillation time. PROCEED on DRAFT design with explicit Hinton-distillation citation for codebook construction methodology."*

### Hafner position (DRAFT)

*"DreamerV3 latent-dynamics-as-pretrained-prior framing: codebook IS sister to DreamerV3 categorical-z latent prior. PATH 2 init = sister DreamerV3-init-from-prior pattern. PROCEED on DRAFT design."*

### Atick position (DRAFT)

*"Cooperative-receiver canonical: DP1 codebook IS encoder-decoder shared prior. PATH 1 composition: codebook regenerated at inflate from compose_with archive bytes; PATH 2 training-time prior: codebook regenerated at inflate from codebook-seeded renderer init. BOTH PATHs canonical cooperative-receiver. PROCEED on DRAFT design."*

### Wyner position (DRAFT)

*"Wyner-Ziv 1976 source-coding-with-side-info: codebook IS the canonical side-info channel. PATH 1 ships codebook bytes; PATH 2 ships codebook-seeded weights. PATH 2 is closer to Wyner-Ziv ideal (side-info regenerated NOT shipped). PROCEED on DRAFT design with PATH 2 priority."*

### Tishby_memorial position (DRAFT)

*"IB framework for codebook capacity: I(X; T_codebook) = mutual information between input and codebook representation; I(T_codebook; Y) = mutual information between codebook and scorer output. Empirical sister Quantizr 0.33 codebook achieves high I(T; Y) at compact codebook size. PROCEED on DRAFT design."*

### Quantizr position (DRAFT)

*"Adversarial reverse-engineering: Quantizr 0.33 codebook (88-94K params; sigma=15; qint_max=7) IS canonical sister-substrate-engineering. DP1 codebook size + structure SHOULD inherit Quantizr lineage. PATH 1 composition rate +0.017197 IS structural; prior-effect MUST buy back. PATH 2 init theoretically 0 rate; sister Quantizr's own architecture is init-from-distilled-codebook. PROCEED on DRAFT design with explicit Quantizr lineage inheritance."*

### Hotz position (DRAFT)

*"Engineering complexity-vs-payoff: DP1 dual-stacking $30-45 envelope vs sister z-substrate $5-22. DP1 IS canonical pretrained-prior axis (architecturally orthogonal to ego-conditioning + foveation). STRONG RECOMMENDATION: PATH 2 PRIORITY (cheapest + theoretically 0 rate overhead); PATH 1 LATER (after PATH 2 lands empirical anchor). VETO any all-at-once dispatch of both PATHs."*

### Schmidhuber position (DRAFT)

*"Compression-as-intelligence + transfer-learning canonical: DP1 IS canonical transfer-learning pattern (OOD source → in-distribution target). PROCEED on DRAFT design."*

### MacKay_memorial position (DRAFT)

*"MDL framework for prior selection: DP1 codebook IS canonical Bayesian prior. Per-class codebook conditioning at PATH 1 IS canonical Bayesian framework. PROCEED on DRAFT design."*

## Step 5 — Per-substrate reactivation criteria pinned per Catalog #298 + #308

Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

| Stage | If verdict | Reactivation path |
|---|---|---|
| Wave 1 PATH 2 smoke | PATH 2 fails to converge from codebook-seeded init | DEFER PATH 2 per Catalog #298; reactivation = codebook redesign (size / structure / distillation methodology) |
| Wave 1 PATH 1 sweep | NO prior_strength config beats PATH 1 rate +0.017197 | DEFER PATH 1 per Catalog #298 → research_only=true; PATH 2 continues |
| Wave 2 paired | PATH 1 + PATH 2 BOTH fail to beat PR101 frontier 0.19205 | DEFER DP1 substrate per Catalog #298 → reactivation = NEW codebook design OR sister Quantizr-class lineage transfer |
| All composition stacks (DP1 + PR101 + Z6/Z7 + TT5L V2 + NSCS06 v8 Variant C + D1) | All fail to beat PR101 | OOD-pretrained-prior paradigm DEFER per Catalog #298 → reactivation = NEW pretrained-prior paradigm OR stateless-frontier extensions |

## Step 6 — Catalog #324 post-training Tier-C validation discipline

Recipe declares `predicted_band_validation_status: pending_post_training`. Reactivation criterion: post-training Tier-C density measurement on DP1 archive (PATH 1 best config OR PATH 2 baseline) after Wave 2 full dispatch via `tools/mdl_scorer_conditional_ablation.py --tier c`. Predicted band [0.180, 0.188] is research prior; promotion-eligible only after `validated_post_training` status. PATH 1 rate arithmetic +0.017197 is HARD-EARNED-EMPIRICAL.

## Operator-routable decisions

**Decision 1**: PATH priority
- (A) PATH 2 PRIORITY (cheapest $5; theoretically 0 rate overhead; canonical Wyner-Ziv ideal)
- (B) PATH 1 PRIORITY ($10-15 prior_strength sweep; HARD-EARNED rate +0.017197; prior-effect disambiguator)
- (C) Both paths PARALLEL ($15-20 total)

**Decision 2**: convocation mechanism (full T3 / inner-quintet / operator-override)

**Decision 3**: dispatch order priority vs sister substrates (Z6 Wave 2 4c $3 + Z7 LSTM Wave 2 $5-7 + TT5L V2 $5 + DP1 PATH 2 $5)

## Cross-substrate dependencies

- **Sister Catalog #209 + #210 + #213 strict gates**: DP1 codebook provenance + Comma2k19 canonical routing + canonical helper invocation — Catalog enforces structurally
- **Sister 2026-05-18 PATH 1 rate-arithmetic correction probe** (`dp1_pr101_composition_noop_probe_20260518_codex.json`): rate +0.017197 HARD-EARNED-EMPIRICAL
- **Sister Quantizr 0.33 lineage**: DP1 codebook size + structure SHOULD inherit Quantizr canonical
- **Sister TT5L V2 + Z6/Z7/Z8**: DP1 stacking composition (DP1 IS orthogonal pretrained-prior axis)
- **Sister Hinton knowledge distillation lineage**: KL-T=2.0 may apply at codebook-distillation time

## Predicted cost per PATH per Wave

- Wave 1 PATH 2 smoke: $5 (Modal T4 ~30 min single-config)
- Wave 1 PATH 1 prior_strength sweep: $10-15 (Modal T4 5-config)
- Wave 2 PATH 1 + PATH 2 paired full: $15-25
- TOTAL DP1 deep-dive envelope: $30-45

## Continual-learning posterior anchor

Per Catalog #300 + `tac.council_continual_learning.append_council_anchor`: this DRAFT must emit v2 posterior anchor at convocation. `deferred_substrate_id` = `pretrained_driving_prior`; `predicted_mission_contribution` = `frontier_breaking`; retrospective due 2026-06-18T05:33:56Z.
