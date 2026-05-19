<!-- # PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: review memo cataloging codex findings + cross-referencing not-yet-implemented canonical helpers per Catalog #287 sub-scope B; this is an HTML comment so markdown renderers ignore it; waiver inherited from sister Wave 1 backfill -->
---
schema: codex_findings_review_v1
review_id: codex_findings_review_top_findings_and_operator_routable_recommendations_20260519
authority: subagent (CODEX-FINDINGS-REVIEW slot — sibling to slots 19/20/21)
review_date: "2026-05-19"
review_actor: claude-main (Opus 4.7, 1M context)
operator_directive_verbatim: "we should maybe queue a slot too to review the latest and highest signal and most releevant and itnersting and hgih value codex fidnings in .omx"
lane_id: lane_codex_findings_review_top_findings_and_operator_routable_recommendations_20260519
sweep_window: "2026-05-02 → 2026-05-19 (last 17 days; bulk concentrated last 7 days)"
sweep_corpus_size:
  total_codex_research_memos: 1183  # _codex.md suffix
  total_codex_prefix_memos: 181     # codex_* prefix
  total_memory_feedback_codex: 26
  routing_directives_last_7d: 41
  session_summaries_last_7d: 30
  unique_findings_classified: 60+
score_claim: false
promotion_eligible: false
provider_spend: false
6_hook_wire_in_declaration:
  hook_1_sensitivity_map: N/A (review-only memo; no signal contribution)
  hook_2_pareto_constraint: N/A
  hook_3_bit_allocator: N/A
  hook_4_cathedral_autopilot_dispatch: ACTIVE — operator-routable queue feeds autopilot ranker via EV/$ sort
  hook_5_continual_learning_posterior: ACTIVE — recommendations propose canonical_equations registry registrations + probe_outcomes_ledger anchors per Catalog #313
  hook_6_probe_disambiguator: ACTIVE — orphan-signal identification IS the disambiguator between consumed-vs-orphan codex findings
---

# CODEX FINDINGS REVIEW — TOP-15 RANKED + ORPHAN-SIGNAL IDENTIFICATION + OPERATOR-ROUTABLE QUEUE (2026-05-19)

## Executive Summary

**Operator-directive 2026-05-19** spawned this CODEX-FINDINGS-REVIEW slot to surface the latest highest-signal codex findings, classify them, and identify orphan-signal-class candidates. Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE": *"if a research artifact can affect score but is not visible to the selector, it is orphaned work"*. This review closes the orphan-signal loop for codex's autonomous research output.

**Sweep scale**: 1183 `_codex.md` memos + 181 `codex_*` prefixed memos under `.omx/research/` + 26 `feedback_codex_*` in Claude memory + 41 routing directives in last 7 days + 30 session summaries in last 7 days. **60+ unique findings classified.**

**TOP-3 findings by composite score** (signal × relevance × value × interest):
1. **TOP-1 Arbitrariness Extinction: λ_seg/λ_pose/λ_rate analytical_solve** — HARD-EARNED-FIRST-PRINCIPLES + CRITICAL + HIGH-EV (predicted ΔS [-0.012, -0.003]) + CROSS-DISCIPLINARY (Boyd-Vandenberghe Lagrange ↔ CLAUDE.md operating-point-dependent SegNet/PoseNet rule). **$0 cost. Status: PARTIAL (canonical helper `tac.score_lagrangian` proposed but not landed). ORPHAN.**
2. **PR101 OP-7 raw-byte-delta paired CPU/CUDA exact-eval REGRESSED** — HARD-EARNED-EMPIRICALLY-VERIFIED + CRITICAL + INFRASTRUCTURE (retires only `pr101-op7-rank1-raw-byte-delta-same-length`; routes follow-ups to per-pair/per-region/SegNet-boundary-preserving variants). **Status: OPERATIONALIZED (probe_outcomes_ledger + cathedral_autopilot_evidence.jsonl).**
3. **Codex META-FIX rounds 1-8 (custody/concurrency/fail-open class)** — HARD-EARNED-EMPIRICALLY-VERIFIED + CRITICAL + INFRASTRUCTURE + NOVEL. **21+ instances fixed, 17+ STRICT preflight META gates landed (#117/#119/#126-#148 + cumulative). Status: FULLY OPERATIONALIZED.**

**Orphan-signal-class count: 8 findings with zero or partial cathedral_consumers integration.** All 8 documented in §5 with recommended wire-ins.

**Recommended dispatch queue (top-3 by EV/$)**:
1. **$0 — Land `tac.score_lagrangian` canonical helper** per TOP-1 arbitrariness directive (predicted ΔS [-0.012, -0.003], rank score per dollar = 12.0 HIGHEST)
2. **$0 — Wire 9 remaining arbitrariness-extinction directives (TOP-2 through TOP-10)** — all closed-form / formula / learned paths with $0 cost; cumulative predicted ΔS up to -0.05
3. **$10.20-15 Vast.ai 4090 — C6.1 lane_17_imp LTH reactivation** per `feedback_pre_rigor_kill_defer_falsified_inventory_landed_20260517.md` op-routable #856 (HIGH-EV: predicted ΔS [-0.05, -0.005]; KILL was Catalog #91+#94 stub-loop artifact NOT paradigm)

---

## 1. Sweep methodology + dedup

### Inventory commands

```bash
# .omx/research codex sweep (full)
find .omx/research/ -name "codex_*" -type f 2>/dev/null | wc -l  # 181
find .omx/research/ -name "*_codex.md" -type f 2>/dev/null | wc -l  # 1183

# Last 7 days narrowing
find .omx/research/ -name "codex_*" -type f -mtime -7 2>/dev/null | wc -l  # 161
find .omx/research/ -name "*_codex.md" -type f -mtime -7 2>/dev/null | wc -l  # 646

# Memory feedback codex sweep
find ~/.claude/projects/-Users-adpena-Projects-pact/memory/ -name "feedback_codex_*" 2>/dev/null | wc -l  # 26

# Routing directives + session summaries
find .omx/research/ -name "codex_routing_directive_*" -mtime -7 2>/dev/null | wc -l  # 41
ls /Users/adpena/Projects/pact/.omx/research/codex_session_summary_* 2>/dev/null | wc -l  # 30+
```

### Dedup strategy

Per CLAUDE.md "Apples-to-apples evidence discipline", findings are deduplicated by **canonical topic-slug + chronological-latest-wins**. Earlier memos on the same topic are superseded by their successor unless the successor explicitly preserves both (Catalog #110/#113 HISTORICAL_PROVENANCE).

Codex's own taxonomy uses 3 anchor-format families:
- **`codex_routing_directive_*`** — Claude→codex tasking with bounded operator-approved budgets
- **`codex_session_summary_*`** — codex→Claude landings with verification evidence
- **`codex_findings_*`** — codex's adversarial-review + audit output

This review consumes all 3 families chronologically.

---

## 2. TOP-15 ranked findings table

Ranking formula: `signal_weight × relevance_weight × value_weight × interest_weight` (per Phase 3 methodology).

| Rank | Title | Signal | Relevance | Value | Interest | Status | Next-Step |
|------|-------|--------|-----------|-------|----------|--------|-----------|
| **1** | TOP-1 Arbitrariness Extinction: λ_seg/λ_pose/λ_rate analytical_solve ($0 closed-form) | HEFP | CRITICAL | HIGH-EV (-0.012, -0.003) | CROSS-DISCIPLINARY | **ORPHAN** | LAND `tac.score_lagrangian` canonical helper |
| **2** | PR101 OP-7 paired CPU/CUDA exact-eval REGRESSED both axes (raw-byte-delta) | HEEV | CRITICAL | INFRASTRUCTURE (retires 1 candidate; opens 4 routing paths) | CONFIRMATORY | **OPERATIONALIZED** | Route to per-pair / per-region / SegNet-boundary-preserving / procedural variants |
| **3** | Codex META-FIX rounds 1-8 (custody/concurrency/fail-open class) | HEEV | CRITICAL | INFRASTRUCTURE | NOVEL | **OPERATIONALIZED** (17+ META gates landed) | Continue META-FIX rounds; pattern is structural |
| **4** | TOP-3 Arbitrariness: per-pair focal weighting (uncertainty-weighted multi-task loss) | HEFP | HIGH | HIGH-EV (-0.006, -0.001) | CROSS-DISCIPLINARY (Kendall 2018) | **PARTIAL** (sister TOP-6 exists; uncertainty-weighted helper proposed) | LAND `UncertaintyWeightedScoreLoss` |
| **5** | TOP-2 Arbitrariness: epochs-per-substrate early-stopping ($0 NET-NEGATIVE — saves money) | HEFP | HIGH | HIGH-EV (-0.006, -0.001) | CONFIRMATORY | **ORPHAN** | LAND `SlopeWatcher` canonical helper |
| **6** | Cathedral per-byte sensitivity consumer verification (CONFIRMED auto-discovery + runtime invocation) | HEEV | HIGH | INFRASTRUCTURE | CONFIRMATORY | **OPERATIONALIZED TODAY** (commit `a3777ac05`) | Top-K per-byte indices → packet-valid mutation/allocator path |
| **7** | TAC Compliance Authority Guard (FEC6 CPU writeup downgrade + procedural-gen gate + deterministic compiler refs) | HEEV | HIGH | INFRASTRUCTURE | NOVEL | **OPERATIONALIZED TODAY** (commit `7fef71884`) | Preserve upstream PR/comment evidence locally |
| **8** | Z7 Mamba2 adversarial hardening (research_only DEFER; 6 evidence gates remain fail-closed) | HEEV | HIGH | INFRASTRUCTURE | NOVEL (state-space-model substrate class) | **OPERATIONALIZED TODAY** (commit `558230385`) | Wait for slot 1 silent-no-spawn fix + 6-gate evidence land |
| **9** | Catalog #329 ProvenanceKind contract extension (PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED + WEIGHT_DERIVED_CODEBOOK + FORBIDDEN_OUT_OF_ARCHIVE) | HEEV | HIGH | INFRASTRUCTURE | NOVEL | **OPERATIONALIZED** (sister to Catalog #323 canonical Provenance umbrella) | Add archive-contained derivation proof gate (sister codex op-routable) |
| **10** | Rate-Attack Vector G1 CPU-axis optimization (Hotz binding $0 IMMEDIATE) | HEFP | HIGH | INFRASTRUCTURE | CROSS-DISCIPLINARY | **PARTIAL** (canonical helper proposed; CPU-axis selector pending) | LAND `tools/cpu_axis_optimal_archive_selector.py` |
| **11** | Catalog #331 Codex-to-Claude inbox channel (bidirectional design-question resolution) | HEEV | HIGH | INFRASTRUCTURE | NOVEL | **OPERATIONALIZED** (canonical `.omx/state/codex_to_claude_inbox.jsonl` + CLI + STRICT gate) | Wire into Claude main-loop pre-flight |
| **12** | TOP-4 Arbitrariness: EMA decay per-substrate formula (replaces 0.997 universal default) | HEFP | MEDIUM | HIGH-EV (-0.005, -0.001) | CROSS-DISCIPLINARY (training-stage-aware EMA) | **ORPHAN** | LAND `tac.ema_decay_formula` canonical helper |
| **13** | TOP-5 Arbitrariness: inflate device pin per-archive (PACT_INFLATE_DEVICE persistent per-archive) | HEEV | MEDIUM | HIGH-EV (-0.005, -0.001) | CONFIRMATORY | **PARTIAL** (Catalog #205 lands canonical helper; per-archive pin sidecar pending) | LAND `.omx/state/inflate_device_pins.jsonl` sidecar |
| **14** | Rate-Attack Vector F1 PoseNet Hydra dims 7-12 (probe-required PROCEED) | HEFP | HIGH | HIGH-EV (predicted [-0.005, -0.002]) | NOVEL (Hydra adversarial-blind-spot hypothesis) | **PARTIAL** (probe scoping landed; archive grammar pending) | RUN `tools/probe_hydra_dim_7_12_score_invariance.py` 600-pair CPU+CUDA |
| **15** | Cross-stack synthesis 9 design landings unified framework (3-set Venn / Fisher / Riemannian-Newton / Tropical / Z8 / TT5L / pose-axis / theoretical floor) | HEEV | HIGH | INFRASTRUCTURE (unified 9×9 cross-pollination matrix) | NOVEL + CROSS-DISCIPLINARY | **PARTIAL** (T2 PROCEED_WITH_REVISIONS; some canonical helpers landed, several still routing-only) | Operationalize remaining canonical helper packages per §11.2 TOP-5 |

**Legend**: HEEV = HARD-EARNED-EMPIRICALLY-VERIFIED. HEFP = HARD-EARNED-FIRST-PRINCIPLES. CCID = CARGO-CULTED-INHERITED-DEFAULT. CCPL = CARGO-CULTED-PATH-OF-LEAST-RESISTANCE.

---

## 3. Per-finding executive summaries

### #1 — TOP-1 Arbitrariness Extinction: λ_seg/λ_pose/λ_rate analytical_solve

**Memo**: `.omx/research/codex_routing_directive_arbitrariness_extinction_top1_lambda_seg_pose_rate_multipliers_unprincipled_20260518.md`

The CONTEST FORMULA `total_score = sqrt(10·pose_avg) + 100·seg_avg + 25·archive_bytes/37545489` fixes the optimal Lagrange multiplier ratios at any operating point IN CLOSED FORM:

```
∂score/∂pose_avg = 5 / sqrt(10 · pose_avg)
∂score/∂seg_avg  = 100  (constant)
∂score/∂rate     = 25 / 37545489
```

At PR106 frontier `pose_avg = 3.4e-5`, the marginal ratio `λ_pose/λ_seg = 2.71` — INVERTED from the old 1.x operating point's 77× SegNet > PoseNet rule. ~30 substrate trainers currently use HAND-TUNED `λ` values almost certainly NOT reflecting this operating-point-dependent flip. **Canonical helper `tac.score_lagrangian.compute_marginal_multipliers` PROPOSED but not landed.** **ORPHAN — closing this loop is the HIGHEST EV/$ action in this entire review** (rank score per dollar = 12.0, predicted ΔS [-0.012, -0.003], $0 cost).

**Operator-routable next-step**: Land the canonical helper (~150 LOC + 20 tests) + wire into `tac.substrates._shared.score_aware_common.score_pair_components` as the DEFAULT λ source.

### #2 — PR101 OP-7 paired CPU/CUDA exact-eval REGRESSED

**Memo**: `.omx/research/codex_session_summary_pr101_op7_exact_eval_score_response_20260519T110500Z_codex.md`

Codex completed paired Modal exact eval for PR101 OP-7 raw-byte-delta candidate `30826b37093ee3af9512a1b46bd0b569fecbc4ccf75b8ff2dd746de113a5144a`:
- `[contest-CPU]`: `0.1928480 → 0.1945418` (Δ = +0.0017)
- `[contest-CUDA]`: `0.2263495 → 0.2277121` (Δ = +0.0014)

**Retires ONLY** `pr101-op7-rank1-raw-byte-delta-same-length`. Does NOT retire master-gradient / per-pair gradient / null-space / procedural packet / deterministic-byte-derivation families. Routes follow-ups to: (1) smaller trust-region byte perturbations; (2) SegNet-boundary-preserving projection; (3) per-pair/per-region perturbation; (4) procedural or deterministic packet compiler variants with new archive SHA + paired exact eval. **OPERATIONALIZED** via probe_outcomes_ledger + reports/cathedral_autopilot_evidence.jsonl + backfilled call IDs in `.omx/state/modal_call_id_ledger.jsonl`.

### #3 — Codex META-FIX rounds 1-8 (custody/concurrency/fail-open class)

**Memos**: 8 round-N landing feedbacks in memory (`feedback_codex_round{2,3,4,5,6,7+8}_findings_fix_with_self_protection_landed_20260509.md` + sister rounds 1 and intermediate)

Across 8 sequential codex adversarial-review rounds, ~21 distinct instances of the META class "custody validation + concurrency + fail-open" bug were extincted. Cumulative STRICT preflight gates landed: 17+ (#117, #119, #126-#148 across multiple sub-clusters). The structural pattern: every codex round catches MORE META instances in the SAME class — leading indicator of how deeply embedded the class was. Examples:
- Round 5 HIGH 1: Lightning ambiguous-submit-failure orphan window → Catalog #143/#147
- Round 6 MEDIUM 2: setup_first_seen single-transaction lost-update race → Catalog #135
- Round 8 HIGH 2: Phase B authorization scoping `consult_session_state=True` → Catalog #150

**FULLY OPERATIONALIZED.** No outstanding action; continue the META-FIX pattern as it surfaces more instances.

### #4 — TOP-3 Arbitrariness: per-pair focal weighting (uncertainty-weighted multi-task loss)

**Memo**: `.omx/research/codex_routing_directive_arbitrariness_extinction_top3_per_pair_focal_weighting_20260518.md`

Every substrate trainer weights the 1199 (or 600) per-pair losses UNIFORMLY. But per-pair `pose_avg + seg_avg` vary by 100× across the 1200 pairs (highly skewed difficulty distribution). Uniform weighting under-penalizes hard pairs → optimizer spends gradient on easy pairs already near zero. Sister TOP-6 (uncertainty-weighted multi-task loss per Kendall et al 2018) is complementary: derive analytic baseline (TOP-1), learn perturbations around it (TOP-6). **PARTIAL — canonical `UncertaintyWeightedScoreLoss(nn.Module)` proposed but not landed.** **ORPHAN — predicted ΔS [-0.006, -0.001] at $0 cost.**

**Operator-routable next-step**: Land `tac.uncertainty_weighted_loss` canonical helper + wire into `score_pair_components`.

### #5 — TOP-2 Arbitrariness: epochs-per-substrate early-stopping

**Memo**: `.omx/research/codex_routing_directive_arbitrariness_extinction_top2_epochs_per_substrate_early_stopping_20260518.md`

Per-substrate `--epochs` defaults are wildly arbitrary (1, 100, 200, 1000, 2000 across substrates). Combined with TOP-9 (early-stopping patience undeclared), most trainers run to full `args.epochs` regardless of convergence. **$0 NET-NEGATIVE — saves money** because early-stop on plateau prevents paid GPU spend continuing past the convergence knee. Canonical `SlopeWatcher` helper proposed: tracks val_score slope across a window; halts when slope falls below noise floor. **ORPHAN.**

### #6 — Cathedral per-byte sensitivity consumer verification (CONFIRMED TODAY)

**Memo**: `.omx/research/codex_session_summary_cathedral_per_byte_reweight_verification_20260519T123603Z_codex.md` (commit `a3777ac05`)

Codex verified slot 6 `per_byte_sensitivity_consumer` closes Wire-in #2 through Catalog #335 auto-discovery AND the runtime cathedral consumer invocation path. Live FEC6 invocation emits `[macOS-CPU advisory]` + `predicted_delta_adjustment=0.0` + `promotable=false` per CLAUDE.md "MPS auth eval is NOISE". 50 tests pass. **Confirms the CATHEDRAL-AUTO-INGEST-PARADIGM-SHIFT (Catalog #335) is structurally extincting orphan-signal-at-cathedral-autopilot.** This was a STRUCTURAL VERIFICATION milestone.

### #7 — TAC Compliance Authority Guard (TODAY)

**Memo**: `.omx/research/codex_session_summary_tac_compliance_authority_guard_20260519T124205Z_codex.md` (commit `7fef71884`)

Closed xhigh audit P1/P2 findings: (a) FEC6 CPU writeup downgraded from `[contest-CPU GHA Linux x86_64]` to `[Modal Linux x86_64 CPU; GHA pending]` until real GHA artifact exists per CLAUDE.md "Apples-to-apples evidence discipline"; (b) procedural-generation promotion gate now requires `archive_seeded` or `weight_derived`; `runtime_constant` requires explicit ruling + non-payload proof; (c) stale deterministic packet compiler refs updated to `tac.packet_compiler.deterministic_compiler.compile_packet(...)`. `tools/check_tac_terminology.py` extended with guards for all 3 bug classes. **OPERATIONALIZED** — closes 3 phantom-score class instances at the TAC-naming surface.

### #8 — Z7 Mamba2 adversarial hardening

**Memo**: `.omx/research/codex_session_summary_z7_mamba2_blocker_burndown_20260519T133715Z_codex.md` (commit `558230385`)

Hardened Z7 Mamba2 lane against false authority + LSTM identity leakage + scaffold gaps. 83 tests pass; readiness assessment remains DEFER with evidence-only blockers. State: `research_only=true`, `dispatch_enabled=false`. **NOT a score claim**; burns down local implementation blockers but does not authorize paid dispatch. **OPERATIONALIZED at the readiness-gate surface.** This is the canonical pattern for state-space-model substrate-class scaffolding: build → harden → defer-pending-evidence; never falsely promote.

### #9 — Catalog #329 ProvenanceKind contract extension

**Memo**: `.omx/research/codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518.md` Items 1-2

Extended `tac.provenance.ProvenanceKind` with 3 new kinds: `PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED`, `WEIGHT_DERIVED_CODEBOOK`, `FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD`. Made `audit_score_claim_dict` fail closed when a score-claiming payload carries the forbidden out-of-archive sentinel. Sister WZ DeliverabilityProof hardening made `contest_compliance_rationale` mandatory non-empty + added `contest_compliance_citation_chain` requiring ≥1 compliant route anchor. **OPERATIONALIZED** as Catalog #329 sister to canonical Provenance umbrella Catalog #323. Hardens the phantom-score-from-research-sidecar bug class at the Provenance contract surface.

### #10 — Rate-Attack Vector G1 CPU-axis optimization (Hotz binding $0)

**Memo**: `.omx/research/codex_routing_directive_rate_attack_vector_2_g1_cpu_axis_optimization_20260518.md`

Hotz binding verdict: **PROCEED_IMMEDIATELY_HOTZ_BINDING** at zero GPU cost. Pure re-ranking on existing dual-eval data: the canonical frontier per axis may differ. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": the leaderboard ranks by CPU. CPU-axis-optimal archive selection IS the operator-facing action. Canonical helper `tools/cpu_axis_optimal_archive_selector.py` proposed + extension to `tools/scan_best_anchor_per_axis.py` (existing per Catalog #316). **PARTIAL — helper not yet landed.** **ORPHAN at zero cost.**

### #11 — Catalog #331 Codex-to-Claude inbox channel

**Memo**: `.omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md`

4-layer canonical landing: canonical helper `src/tac/codex_to_claude_inbox.py` + CLI `tools/codex_to_claude_inbox.py` + Catalog #331 STRICT gate + `tools/operator_briefing.py` summary wire-in. Codex now persistent-loop polls inbox for Claude's responses; on ambiguity blocking progress, codex `ask`s Claude instead of guessing; on novel info worth surfacing, codex `relay`s without expecting response. The inbox JSONL at `.omx/state/codex_to_claude_inbox.jsonl` has 7+ events demonstrating bidirectional flow. **OPERATIONALIZED — closes the bidirectional design-question loop.**

### #12-#14 — Arbitrariness Extinction TOP-4, TOP-5, F1

(see individual rows in §2 table)

### #15 — Cross-stack synthesis 9 design landings unified framework

**Memo**: `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` (149.9 KB, T2 PROCEED_WITH_REVISIONS)

Unifies 9 design landings (3-set Venn / Fisher-precondition / Riemannian-Newton / Tropical d_seg / Z8 / TT5L V2 / cargo-cult resurrection TOP-3 / pose-axis T3 / theoretical floor) into ONE coherent framework with 9×9 cross-pollination matrix + composition order + canonical task queue. This is the architectural-synthesis anchor for the entire late-May 2026 design wave. PARTIAL operationalization — some canonical helper packages landed (Provenance #329, deliverability #319), several still routing-only. **HIGH-VALUE NOVEL.**

---

## 4. Cross-reference to slot 19 (canonical_equations) + slot 21 (findings Lagrangian)

### Slot 19 — `tac.canonical_equations` registry inventory

Currently registered 6 builtin equations (per `src/tac/canonical_equations/builtins.py`):
1. `brotli_cascade_bounded_per_stream_v1`
2. `mps_drift_architecture_class_dependent_v1`
3. `per_byte_leverage_uniformly_distributed_v1`
4. `per_pair_master_gradient_score_impact_taylor_v1`
5. `master_gradient_locality_violation_by_codec_v1`
6. `canonical_frontier_pointer_v1`

### Codex findings → canonical_equations registry candidates

The following codex findings SHOULD be registered as either new `CanonicalEquation` rows OR as `EmpiricalAnchor` rows on existing equations:

| Codex Finding | Registry Action | Equation ID | Rationale |
|---------------|-----------------|-------------|-----------|
| #1 TOP-1 λ_seg/λ_pose/λ_rate analytical solve | **NEW equation** | `score_marginal_lagrange_multipliers_v1` | Closed-form derivation from contest formula; predicted score impact per operating point |
| #2 PR101 OP-7 paired CPU/CUDA REGRESSED | **EmpiricalAnchor** on `per_pair_master_gradient_score_impact_taylor_v1` | (existing) | Negative anchor: raw-byte-delta operator violates Taylor first-order in brotli-cascaded archive |
| #4 TOP-3 per-pair focal weighting | **NEW equation** | `per_pair_loss_weighting_optimal_v1` | Uncertainty-weighted multi-task loss + per-pair difficulty distribution |
| #5 TOP-2 epochs-per-substrate early-stopping | **NEW equation** | `convergence_slope_early_stop_v1` | Slope-watcher canonical helper output predicts substrate-specific optimal epochs |
| #10 G1 CPU-axis optimization | **EmpiricalAnchor** on `canonical_frontier_pointer_v1` | (existing) | CPU-vs-CUDA frontier divergence per family; canonical selector consults this |
| #12 TOP-4 EMA decay formula | **NEW equation** | `ema_decay_substrate_stage_aware_v1` | Per-substrate training-stage-dependent EMA decay |
| #14 F1 PoseNet Hydra dims 7-12 | **EmpiricalAnchor** (pending probe) on (new) `posenet_hydra_dim_invariance_v1` | TBD | Hypothesis: dims 7-12 are adversarial-blind-spot; probe pending |

**Recommendation**: Slot 21 (findings Lagrangian) should consume the above EmpiricalAnchor rows as data-fit residuals. Slot 19's `EmpiricalAnchor` schema is the canonical landing surface for codex empirical findings — closing this loop CONVERTS codex orphan-signal into autopilot-consumable canonical data.

### Slot 21 — findings Lagrangian implications

Per the symposium memo `.omx/research/grand_council_t3_findings_lagrangian_and_pp_integration_design_symposium_20260519.md` Q1-Q9 PROCEED verdicts, the 4-term Lagrangian:
- **Data-fit term** (closed-form Gaussian posterior σ): codex findings #2 (PR101 OP-7 regression) feeds this as a HIGH-PRECISION empirical anchor
- **Occam (Catalog #299 quota) term**: codex META-FIX rounds 1-8 (finding #3) IS the empirical Occam pressure (17+ META gates close 21+ instances ≈ 80% compression ratio)
- **Partition term**: slot 17's 4-class cascade taxonomy is the initial partition; codex findings #1/#3/#4/#5/#7/#9/#10 partition naturally into "analytical_solve | empirical_verification | architectural_hardening | provenance_extension"
- **μ_explore term**: codex's arbitrariness-extinction TOP-10 corpus IS the canonical exploration queue (10 closed-form/formula/learned paths with explicit predicted ΔS bands)

---

## 5. Orphan-signal-class section

**Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable**: *"if a research artifact can affect score but is not visible to the selector, it is orphaned work"*. The following codex findings have ZERO or PARTIAL cathedral_consumers integration and are CANDIDATE ORPHANS:

### Confirmed orphans (zero consumers)

1. **TOP-1 λ_seg/λ_pose/λ_rate analytical_solve** — predicted ΔS [-0.012, -0.003]. Canonical helper `tac.score_lagrangian` not landed. Sister cathedral consumer not registered. **HIGHEST-EV ORPHAN.**
2. **TOP-2 epochs-per-substrate early-stopping** — predicted ΔS [-0.006, -0.001] + NET-NEGATIVE cost. `SlopeWatcher` not landed. No cathedral consumer.
3. **TOP-3 per-pair focal weighting** — predicted ΔS [-0.006, -0.001]. `UncertaintyWeightedScoreLoss` not landed.
4. **TOP-4 EMA decay per-substrate formula** — predicted ΔS [-0.005, -0.001]. Replacing universal 0.997 default not yet wired.
5. **TOP-10 VQ-VAE codebook init K-means++** — predicted ΔS [-0.003, -0.0005]. Codebook init not migrated from `torch.randn * 0.1`.

### Partial orphans (some integration, downstream consumers missing)

6. **Rate-Attack G1 CPU-axis selector** — `tools/cpu_axis_optimal_archive_selector.py` not landed; only `tools/scan_best_anchor_per_axis.py` (Catalog #316) exists upstream. Hotz binding verdict says **PROCEED IMMEDIATELY**.
7. **Rate-Attack F1 PoseNet Hydra dims 7-12** — probe scoping landed in `tools/probe_hydra_dim_7_12_score_invariance.py` (proposed); never run 600-pair CPU+CUDA empirical anchor.
8. **Cross-stack synthesis 9 design landings** — Riemannian-Newton + Tropical d_seg + Dynamic per-candidate composition canonical helper packages are routing-only (codex directives written) but the actual packages not landed.

### Recommended wire-ins (orphan-closure)

For each orphan, the wire-in pattern per Catalog #335 (CATHEDRAL-AUTO-INGEST-PARADIGM-SHIFT) is:

```python
# src/tac/cathedral_consumers/<consumer_name>/__init__.py
CONSUMER_NAME = "<orphan_name>_consumer"
CONSUMER_VERSION = "1.0"
CONSUMER_HOOK_NUMBERS = (1, 3, 4)  # sensitivity-map + bit-allocator + cathedral autopilot

def update_from_anchor(anchor): ...
def consume_candidate(candidate): ...
```

The auto-discovery loop at `tools/cathedral_autopilot_autonomous_loop.discover_and_register_consumers` ingests new consumers without manual cathedral_autopilot edits. **This is THE canonical orphan-closure pattern.**

---

## 6. META-pattern section (cross-finding patterns surfaced)

### META-pattern 1: Codex META-FIX rounds 1-8 — bug class density

Across 8 rounds, ~21 instances of the SAME META class (custody validation + concurrency + fail-open) were extincted with 17+ STRICT preflight gates. **Leading indicator**: the bug class was embedded across the entire codebase. The pattern was: round N catches NEW instances at NEW surfaces; sister gates extend coverage incrementally; the META class becomes structurally extinct only after 17+ gates × multiple surfaces.

**Operator-routable**: This pattern suggests EVERY new STRICT preflight gate landing should evaluate "is this a single-instance fix OR a META class symptom?" per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable.

### META-pattern 2: Arbitrariness extinction = $0 closed-form/formula/learned wins

10 of 10 TOP-N arbitrariness extinction directives have **$0 cost** + ALL closed-form / formula / learned paths. Cumulative predicted ΔS up to -0.05 across all 10. **Empirical receipt**: TOP-1 alone has rank score per dollar = 12.0 (HIGHEST in the corpus). The operator's standing directive *"identifying arbitrariness or less than optimal being applied across the board and using all techniques and exploits and contest rules and allowed and everything to either experimentally determine the proper solution or solve it and use the optimal solution or use a formula instead or learn and train against the values or use neural or self or some other alien tech or combination of teks"* is producing structural wins.

**Operator-routable**: Create a `lane_arbitrariness_extinction_wire_in_wave_<utc>` to land TOP-1 through TOP-10 canonical helpers in a single commit batch.

### META-pattern 3: Codex findings → canonical equations registry IS the orphan-closure mechanism

Slot 19's `EmpiricalAnchor` schema converts codex empirical findings into autopilot-consumable canonical data. Currently only 6 equations registered; the corpus of 60+ codex findings could feed 10-15+ new equations + 30+ EmpiricalAnchor rows. **Closing this loop converts codex from "research producer" to "first-class autopilot signal source".**

### META-pattern 4: Phantom-score class extinction across 6 surfaces

Catalog #323 canonical Provenance umbrella + #287 docstring overstatement + #249 misleading directory name + #319 autopilot Venn reweight + #321 research sidecar + #823 byte-identity = 6 surfaces extincting the same META class. Codex finding #9 (Catalog #329) extended ProvenanceKind contract; codex finding #7 (TAC Compliance Authority Guard) hardened TAC naming surface. **The phantom-score class is now structurally extinct across 8+ orthogonal surfaces** — the canonical example of META-class extinction.

### META-pattern 5: Operational bug classes recurring → driver/recipe consistency gates

Z6-v2 Wave 2 silent-no-spawn + STC v2 3rd consecutive silent-no-spawn failure + Modal harvester ledger gap → Catalog #326 (driver-recipe mode consistency) + Catalog #330 (Modal harvester call-id ledger) + sister fixes. **Operational bug classes are surfacing at the dispatch-layer surface** and codex has been consistently producing canonical gate landings as remediation.

### META-pattern 6: PR-replay regression as canonical evidence

PR101 OP-7 (#2) is the canonical example of "byte-perturbation operator with predicted score-improvement REGRESSES under paired exact-eval custody". Routes to per-pair / per-region / SegNet-boundary-preserving / procedural variants. **The pattern is: every raw-byte-delta operator on brotli-compressed archives is suspect; the locality basis must be POST-BROTLI-DECOMPRESS** (per codex's OP-7 iteration items 3+4 landing).

---

## 7. Operator-routable queue (sorted by EV/$)

### TIER A — $0 cost (immediate)

1. **LAND `tac.score_lagrangian` canonical helper + wire into `score_pair_components`** (TOP-1 arbitrariness extinction; predicted ΔS [-0.012, -0.003]; rank score per dollar = 12.0)
2. **LAND `tools/cpu_axis_optimal_archive_selector.py`** (G1 Hotz binding PROCEED IMMEDIATELY; pure re-rank of existing data)
3. **LAND `tac.uncertainty_weighted_loss` + `SlopeWatcher` + `tac.ema_decay_formula`** (TOP-3 + TOP-2 + TOP-4 arbitrariness extinctions; cumulative predicted ΔS [-0.017, -0.003])
4. **REGISTER codex findings as `EmpiricalAnchor` rows in slot 19 `tac.canonical_equations`** (closes orphan-signal-class for 6+ findings)
5. **LAND cathedral consumer packages for `score_lagrangian`, `cpu_axis_selector`, `uncertainty_weighted_loss`** (per Catalog #335 auto-discovery pattern)

### TIER B — $0.30-$3 cost (cheap probes)

6. **RUN `tools/probe_hydra_dim_7_12_score_invariance.py` 600-pair CPU+CUDA** (Rate-Attack F1 PoseNet Hydra dims 7-12; PROCEED conditional on probe passing; $0.30 Modal smoke)
7. **RUN `tools/probe_b1_pyav_av1_cpu_cuda_bit_identity.py` + `tools/probe_b1_patch_distribution_density.py`** (Rate-Attack B1 contest-video-codebook; PROCEED_WITH_REVISIONS conditional)

### TIER C — $10-$35 (paid dispatch per operator-approved 60.30 USD envelope)

8. **DISPATCH C6.1 lane_17_imp LTH reactivation** ($10.20-15 Vast.ai 4090; predicted ΔS [-0.05, -0.005]; per `feedback_pre_rigor_kill_defer_falsified_inventory_landed_20260517.md` op-routable #856)
9. **DISPATCH C6.3 PR106 #05+#06 REFORMULATED paired smoke** ($10 Modal A10G; predicted ΔS via paradigm-INTACT/design-CARGO-CULTED unwind)
10. **DISPATCH C6.5 mae_v + saug operational-fix** ($10-35 Vast.ai 4090; operational-completion fix)

### TIER D — DEFERRED-PENDING-SISTER-LAND

11. **Z6 Wave 2 4c re-fire** ($3 Modal A10G) — BLOCKED on slot 1 silent-no-spawn fix landing
12. **STC v2 RATIFY-or-DEFER** ($0.20 Modal T4) — BLOCKED on slot 1 silent-no-spawn fix landing

---

## 8. Strict-discipline declarations

- **Catalog #229 PV**: 4 premises verified pre-write (a) sweep size 1183 `_codex.md`, (b) 181 `codex_*` prefix, (c) slot 19 6 builtins via `grep equation_id`, (d) slot 21 symposium memo Q1-Q9 PROCEED verdicts via `head -80`
- **Catalog #117/#157/#174 commit serializer + POST-EDIT --expected-content-sha256** for all 3 files committed (review memo + landing memo + MEMORY.md prepend)
- **Catalog #206 checkpoint discipline**: 3 checkpoints written (`tools/subagent_checkpoint.py`)
- **Catalog #230 sister-subagent ownership map**: ALL NEW files; ZERO mutations to slot 19/20/21 owned files
- **Catalog #287/#323 canonical Provenance**: this memo is observability-only (`score_claim=false`, `promotion_eligible=false`, `provider_spend=false`); evidence_grade=[predicted] for all classifications; HTML comment file-level waiver `# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE` for not-yet-implemented tac.X module-name citations

## 9. Sister-coordination notes

Slots 19/20-supplemental/20-second-supplemental/20-maintenance/21-parallel-dual-track are running in parallel. THIS slot's work is DISJOINT:
- **Files written by THIS slot**: review memo + landing memo + MEMORY.md prepend + lane registry mutation (4 files; none touched by sister slots)
- **Read-only consultation**: slot 19 `tac.canonical_equations` source (no edits); slot 21 findings Lagrangian symposium memo (no edits)
- **No editor collision risk**: per Catalog #340 sister-checkpoint guard verification (subagent_progress.jsonl checked at start)

---

**END OF REVIEW MEMO. 60+ codex findings classified. 8 orphans identified. Top-3 dispatch queue sorted by EV/$. Closing the orphan-signal loop is the canonical next-step per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable.**
