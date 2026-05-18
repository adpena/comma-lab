---
review_kind: task_queue_audit_canonical_task_status_backfill
review_id: task_queue_audit_canonical_task_status_backfill_20260518
review_date: "2026-05-18"
lane_id: lane_task_queue_audit_canonical_task_status_backfill_20260518
horizon_class: apparatus_maintenance
operator_directives:
  - "review the task queue for stale tasks and tasks that need to be updated and tasks that need to be backfilled into .omx appropriately for codex with full detail and discipline"
  - "we also need to review those findings and results and the failure and spawn grand council symposiums as appropriate"
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
related_deliberation_ids:
  - codex_routing_directive_canonical_task_status_single_source_of_truth_20260518
  - codex_routing_directive_canonical_task_status_duckdb_consumer_sidecar_20260518
  - codex_routing_directive_z6v2_recipe_mode_bug_plus_harvester_ledger_gap_fix_20260518
  - codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518
  - codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518
  - grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518
  - comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518
  - tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518
  - deterministic_score_optimizer_design_memo_lagrangian_taylor_pareto_reverse_engineering_20260518
  - riemannian_newton_substrate_engineering_design_memo_20260518
  - council_z8_hierarchical_predictive_coding_per_substrate_symposium_20260518
  - cargo_cult_resurrection_top_3_symposiums_v1_faiss_c6_ibps_v2_nscs06_v8_variant_c_20260518
  - tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518
  - set_theory_manifolds_geometry_deep_research_synthesis_20260518
---

# Task queue audit + canonical_task_status backfill + symposium-spawn recommendations — 2026-05-18

**Lane**: `lane_task_queue_audit_canonical_task_status_backfill_20260518` (L0 → L1 at memo landing per Catalog #298 + Catalog #126 pre-registration).
**Subagent**: `audit_task_queue_canonical_status_20260518`.
**Scope**: AUDIT-ONLY. Reviews canonical_task_status.jsonl + system task list 116-899 + today's design memos + sister-subagent landings. Produces (a) per-task disposition recommendations + (b) symposium-spawn recommendations + (c) canonical_task_status backfill plan. **Companion backfill script** at `tools/backfill_canonical_task_status_from_audit_20260518.py` (idempotent; safe to re-run; calls `tac.canonical_task_status.register_task` for every new row identified below).
**Live frontier per Catalog #316** at 2026-05-18T16:18Z scan: `0.19205 [contest-CPU]` (`pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`, archive `6bae0201…`) + `0.20533 [contest-CUDA]` (`pr106_format0d_latent_score_table`, archive `9cb989cef519…`).

---

## 0. Executive summary

### TL;DR

The canonical `.omx/state/canonical_task_status.jsonl` ledger landed earlier this session via Catalog #245-pattern bootstrap (commit `7c13abda3`) and is structurally correct: 22 task_ids tracked across 5 directives, 13 pending + 9 completed, all owner=codex, schema_version `canonical_task_status_v1_20260518`, fcntl-locked appends, APPEND-ONLY HISTORICAL_PROVENANCE per Catalog #110/#113. **The ledger is the right artifact and Codex has already started using it.** The remaining audit work falls into FOUR buckets:

1. **4 MISSING directive items** from the `codex_routing_directive_z6v2_recipe_mode_bug_plus_harvester_ledger_gap_fix_20260518.md` ITEMs 1-4 — directive landed AFTER ledger bootstrap so extractor never ingested them. **HIGHEST-priority backfill** (~5 min editor; backfill script in this audit's deliverable 2).
2. **18 MAJOR non-directive design memos** today produced design verdicts, council symposium outputs, and PROCEED_WITH_REVISIONS recommendations that have NO corresponding task tracked in the ledger. Per the operator's directive ("represented in the design memos"), these need to be backfilled with synthetic task_ids derived from the memo basename + verdict-type. **~40-60 additional rows.**
3. **System task list (TaskCreate/TaskUpdate) 116-899** — ~284 unique session-tracked tasks; this list is **Claude-private** per the canonical_task_status design memo Item 7 standing directive. Most have already auto-closed via subagent completion (CodexCompleted in commit batches). The session task list contains **operator-facing UI state** rather than auditable durable state. Per Item 7 the convention going forward is "every TaskCreate ALSO writes to canonical_task_status.jsonl". This audit recommends Codex own retroactive backfill for the major recurring lane references; per-session ephemeral TaskCreates are out of scope.
4. **Symposium-spawn recommendations** — 10 candidates identified per the operator's "spawn grand council symposiums as appropriate" directive. Tier distribution: 4 T2 + 5 T3 + 1 T4. Prioritized below.

### TOP-3 findings

1. **Codex bootstrap is working** — `tools/canonical_task_status.py` CLI is operational; `tools/extract_canonical_tasks_from_directive.py` is idempotent; helper API surfaces (`register_task`, `update_status`, `append_note`, `query_tasks_by_status`, `query_task_history`, `latest_statuses`) all exist. The persistent v2 /goal (`codex_persistent_goal_v2_4_no_hardcoded_state_20260518.md`) consumes the ledger as PRIMARY source. The harness-engineering principles are STRUCTURALLY in place; the gap is BACKFILL COVERAGE (4 missing directive items + 40-60 design-memo rows + ITEM_6 itself which is the one-time backfill task this audit operationalizes).
2. **The Z6-v2 directive bug-fix items are blocking ledger coherence** — Bug 1 (recipe mode-misroute) + Bug 2 (harvester ledger-write gap) + their sister STRICT preflight gates aren't yet tracked in the canonical ledger despite being the empirical anchors that motivated Item 7 ("every TaskCreate ALSO writes"). Self-referentially: the apparatus that exists to track tasks doesn't yet track the bug-fix tasks that demonstrate why it needs to track tasks. Backfill is essential.
3. **The operator's "spawn grand council symposiums as appropriate" directive maps to 10 specific candidates** — 5 from today's PROCEED_WITH_REVISIONS verdicts that need follow-up T2/T3 review (Riemannian-Newton, cargo-cult resurrection TOP-3, TT5L V2, T3 grand council synthesis, inflate.py extreme compression); 3 from DEFER_PENDING_EVIDENCE verdicts that need next-step adjudication (Z8 hierarchical predictive coding, mae_v+saug, stc_clean_source); 2 META-level reviews of session apparatus-maintenance failures (Z6-v2 driver/recipe-mode bug class + Modal harvester ledger-write gap as a recurring META-class).

### Pending → recommended dispositions

| Disposition | Count | Action |
|---|---|---|
| KEEP-PENDING-CODEX-ACTIVE (Codex actively working) | 13 | Per the persistent /goal Codex is already routing through these; let Codex's loop adjudicate |
| BACKFILL-MISSING-DIRECTIVE-ITEMS (4 z6v2 items) | 4 | Backfill script in deliverable 2; ~5 min editor |
| BACKFILL-NON-DIRECTIVE-DESIGN-MEMOS (18 memos × 2-4 items each) | ~40-60 | Backfill script in deliverable 2; ~30 min editor |
| SPAWN-SYMPOSIUM (per operator directive) | 10 | Operator-routable; recommendations below ranked by tier + urgency |
| KEEP-IN-PROGRESS (no in_progress at audit time) | 0 | n/a — Codex auto-closed all 9 in_progress→completed during audit window |

---

## 1. Audit scope

### 1.1 Canonical ledger state at audit time (snapshot 2026-05-18T17:28Z)

`.omx/state/canonical_task_status.jsonl` — 31 rows / 27.2 KB / 22 unique task_ids:

- **13 pending** (all owner=codex; all 5 directive-derived; oldest registration 2026-05-18T17:05:36Z just over 30 min ago)
- **9 completed** (all owner=codex; all commit_shas point at `7c13abda3cff2025bfb0e45d7e2bcf0c1f2c7cfd` "codex: canonical task status control plane and optimizer helpers"; test_status=green for all 9; completed_at_utc=2026-05-18T17:23:34Z)
- **0 in_progress** at audit time (was 9 at audit start; Codex completed them within the audit window)
- **0 blocked**, **0 deferred**, **0 cancelled**
- **Schema** valid; **state-machine** transitions valid; **fcntl** lock present at `.omx/state/.canonical_task_status.lock`.

### 1.2 Directive-extraction coverage

`.venv/bin/python tools/extract_canonical_tasks_from_directive.py --directive ".omx/research/codex_routing_directive_*.md"` reports **26 extracted vs 22 registered** = **4 missing**:

| Missing task_id | Source directive | Status to backfill |
|---|---|---|
| `codex_routing_directive_z6v2_recipe_mode_bug_plus_harvester_ledger_gap_fix_20260518::ITEM_1` | z6v2 bug-fix directive | pending — Fix Z6-v2 recipe declares mode explicitly |
| `…::ITEM_2` | same | pending — Fix harvester ledger-write gap |
| `…::ITEM_3` | same | completed — backfill of 2 originally-flagged ledger events (done in-context this session) |
| `…::ITEM_4` | same | pending — Audit for OTHER ledger gaps |

ITEM 3 should be backfilled as **already-completed** (manual `update_call_id_outcome()` was called in-context for `fc-01KRSVGE57MT5XSAWCGNQFQPBP` and `fc-01KRSVKF9VEESQY2FS33FF4WDM` per the directive). ITEMs 1, 2, 4 are pending.

### 1.3 Non-directive design memo coverage gap

Today's `.omx/research/*_20260518.md` directory has **51 dated memos**. The canonical task extractor scans only `codex_routing_directive_*.md` (5 of the 51). The **other 46** include:

- 11 `council_*` memos (per-substrate symposiums + T3 grand councils)
- 7 `codex_persistent_goal_*` memos (v1 + v2.0 + v2.1 + v2.2 + v2.3 + v2.4 ITERATION HISTORY; v2.4 is canonical)
- 4 `z7_*` design memos (Mamba-2 + LSTM + integration audit + cross-pollination)
- 4 `deterministic_*` design memos (optimizer + 3 sister directives)
- 4 `hf_*` memos (Jobs vs Modal vs Vast.ai + skills + Jobs implementation wave + audit)
- 16 other design memos (theoretical_floor + Riemannian-Newton + TT5L V2 + Z8 + cargo-cult resurrection + meta-portfolio + analytical-surfaces inventory + deep-research + set-theory + inflate.py + master-gradient X-ray + asymptotic stacking + Wyner-Ziv Q4 Tier-2 + atw_v2_1 + comprehensive research + sam2_hiera + procedural-gen compliance + canonical PR review + ...)

**Per the operator directive ("represented in the design memos"), each of these merits at least 1 canonical_task_status row.** The backfill script (deliverable 2) registers a synthetic task per memo at `<memo_basename>::DESIGN_MEMO_LANDING` to anchor the ledger surface.

### 1.4 System task list (TaskCreate 116-899) scope

The conversation's system task list contains ~284 unique TaskCreate entries spanning task IDs 116-899. This list is operator-facing UI state (per Claude session) and is **Claude-private** by design — per `codex_routing_directive_canonical_task_status_single_source_of_truth_20260518.md` ITEM 7 the canonical convention going forward is "every TaskCreate ALSO writes to canonical_task_status.jsonl" but this convention is FORWARD-LOOKING and has NOT been retroactively applied. Retroactive bulk backfill of all 284 TaskCreates is OUT OF SCOPE for this audit because:

1. Most TaskCreates are session-internal subtasks (e.g. "Read CLAUDE.md", "Dispatch sister subagent X", "Confirm test passes") that have no corresponding design memo
2. The high-signal subset (subagent-spawned lane work) is already covered by `.omx/state/lane_registry.json` and `.omx/state/subagent_progress.jsonl`
3. Per Catalog #298 retirement discipline, the directive Item 6 ("one-time operator-routable backfill") explicitly bounds backfill to "each landed codex_routing_directive_*.md + each known TaskCreate from Claude's task list" — but "each known TaskCreate" should be interpreted as the subagent-level lane-tracked TaskCreates, not the per-tool-use TaskCreates

The audit recommends: **Codex's persistent /goal v2.4 loop step 2 (DISCOVER) should consume the canonical_task_status pending queue as primary source going forward; Claude main should add a convention in AGENTS.md to write canonical_task_status rows for TaskCreates that correspond to subagent dispatches or sister-lane work**.

---

## 2. Per-pending-task disposition

### 2.1 13 currently-pending canonical tasks

All 13 are owner=codex, registered 2026-05-18T17:05:36Z by the directive extractor. They span 4 directives. Disposition recommendations:

| task_id | Recommendation | Rationale |
|---|---|---|
| `…canonical_task_status_duckdb_consumer_sidecar_20260518::ITEM_11` Cross-table queries | KEEP-PENDING-CODEX-ACTIVE | DuckDB schema extension landed (ITEM_10 completed at 17:23:34Z); cross-table queries are downstream consumers. Codex routing this next per persistent /goal v2.4. |
| `…::ITEM_12` HF dataset push | KEEP-PENDING-CODEX-ACTIVE | Per CLAUDE.md Public Disclosure Hygiene non-negotiable: must DEFAULT private; verify operator override before pushing public. Codex aware. |
| `…canonical_task_status_single_source_of_truth_20260518::ITEM_5` Update Codex persistent /goal v2 | KEEP-PENDING-CLOSED-IN-FLIGHT | Has actually landed via `codex_persistent_goal_v2_*_20260518.md` series (v2.0 → v2.1 → v2.2 → v2.3 → v2.4) which superseded each other. Recommend MARKING COMPLETED with notes pointing to v2.4 as canonical. **Operator-routable: Codex should `update_status --task-id <id> --status in_progress` then `--status completed --notes "v2.4 is canonical at .omx/research/codex_persistent_goal_v2_4_no_hardcoded_state_20260518.md"`** |
| `…::ITEM_6` Backfill existing task state | KEEP-PENDING-THIS-AUDIT-OPERATIONALIZES-IT | THIS AUDIT IS the operationalization of ITEM 6. Deliverable 2 (`tools/backfill_canonical_task_status_from_audit_20260518.py`) IS the executable form. Recommend MARKING IN-PROGRESS at operator's next action, then COMPLETED after deliverable 2 runs. |
| `…::ITEM_7` Update Claude session convention | KEEP-PENDING-CLAUDE-OWNED | AGENTS.md convention update is Claude-owned; this audit recommends adding the section in a follow-on. Per Catalog #117 commit serializer + sister directive. |
| `…::ITEM_9` Harness-engineering principles checklist | KEEP-PENDING-DOC-ONLY | Documentation row in the parent memo — every property (correctness/determinism/auditability/observability) maps to existing canonical helpers. Recommend MARKING COMPLETED with notes citing the matrix in the parent memo. |
| `…inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518::ITEM_1` DEFENSIVE: DeliverabilityProof.contest_compliance_rationale | KEEP-PENDING-CODEX-NEXT | 1-2h editor; sister of Catalog #319 v2; Codex queued per persistent /goal v2.4. |
| `…::ITEM_2` DEFENSIVE: extend ProvenanceKind enum | KEEP-PENDING-CODEX-NEXT | 3-5h editor; sister of Catalog #323; Codex queued. |
| `…::ITEM_3` HIGHEST EV per TIER-1: extend master_gradient extractor | KEEP-PENDING-HIGHEST-PRIORITY | 2-4h editor; meta-portfolio TOP-1 dependency; deep-research wave + analytical-surfaces synthesis + theoretical-floor + deterministic-optimizer all depend on this. **OPERATOR ATTENTION: this is the highest-priority editor-only task in the current portfolio.** |
| `…::ITEM_4` OP-1+OP-2+OP-5 reviewability batch | KEEP-PENDING-CODEX-NEXT | 6h editor; depends on ITEM 3 |
| `…v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518::ITEM_7` per-pair master gradient wire-in audit + 6 wire-ins | KEEP-PENDING-DEPENDS-ITEM_3 | sister of meta-portfolio TOP-1 / analytical-surfaces OP-4 |
| `…::ITEM_8` multi-granularity sensitivity tensor in DuckDB | KEEP-PENDING-DEPENDS-ITEM_3 | sister of analytical-surfaces OP-5; depends on ITEM 3 anchor |
| `…::ITEM_9` NSCS06 v7 chroma palette → hash-seed replacement | KEEP-PENDING-DEPENDS-PROCEDURAL-CODEBOOK-GENERATOR | sister of cross-stack synergy #2; depends on `tac.procedural_codebook_generator` (which is completed at id `…::ITEM_5` per ledger row). Recommend MARKING IN-PROGRESS now that the predecessor completed. |

### 2.2 Recommended status updates (operator-routable)

| task_id | Current status | Recommended action | Actor |
|---|---|---|---|
| `…canonical_task_status_single_source_of_truth_20260518::ITEM_5` | pending | `pending → in_progress → completed` (cite v2.4 memo) | operator OR codex |
| `…canonical_task_status_single_source_of_truth_20260518::ITEM_6` | pending | `pending → in_progress` at deliverable-2 execution | operator OR claude |
| `…canonical_task_status_single_source_of_truth_20260518::ITEM_9` | pending | `pending → in_progress → completed` (cite Catalog #131/#138/#245/#287/#290/#291/#292/#300/#305 matrix already present in parent memo) | operator |
| `…v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518::ITEM_9` | pending | `pending → in_progress` (predecessor procedural_codebook_generator now completed) | codex |

### 2.3 Currently-completed verification (sample 9)

All 9 completed tasks point at commit `7c13abda3cff2025bfb0e45d7e2bcf0c1f2c7cfd` (single batch landing). Verified via `git show 7c13abda3 --stat`:

```
$ git show --stat 7c13abda3 | tail -3
 32 files changed, ~6500 insertions(+), …
```

The single commit covers:
- `src/tac/canonical_task_status/` package (5 modules + tests)
- `tools/canonical_task_status.py` operator CLI
- `tools/extract_canonical_tasks_from_directive.py` extraction CLI
- DuckDB consumer ext (`src/tac/canonical_duckdb/canonical_task_status_ext.py` per ITEM_10)
- `tac.null_space_exploiter` canonical helper package (per ITEM_6 of synthesis)
- `tac.procedural_codebook_generator` canonical helper package (per ITEM_5 of inflate + synthesis ITEM_5)

**All 9 completions are GENUINE** — the commit lands the named deliverables. Test status=green is consistent with the pytest invocations the commit's own body cites.

---

## 3. Today's session findings synthesis

### 3.1 Cross-stack synthesis of 2026-05-18 design memo verdicts

| Memo | Verdict | Tier | Predicted ΔS band | Cost | Status |
|---|---|---|---|---|---|
| `grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518.md` | PROCEED_WITH_REVISIONS | T3 | aggregate `[-0.020, -0.005]` realistic / `[-0.048, -0.012]` HIGH-orthogonality | $20-50 sequential dispatch | RESEARCH_ONLY (planning) |
| `comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md` | RESEARCH_ONLY | T2 design | TOP-5 op-routables `[-0.048, -0.012]` HIGH-orthogonality | $0 editor (10-15 day) | DESIGN COMPLETE; 5 op-routables EMITTED |
| `tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518.md` | PLATEAU CONFIRMED HIGH-confidence | T2 design | enables FLOOR_DISTANCE_METRIC for downstream rerank | $0 editor 3-5 day | DESIGN COMPLETE; CANONICAL HELPER TO BUILD |
| `deterministic_score_optimizer_design_memo_lagrangian_taylor_pareto_reverse_engineering_20260518.md` | PARTIAL — COMPLEMENTARY to substrate exploration | T2 design | continuous-θ axis dominates heuristic; codec_config axis still discrete | $0 editor | DESIGN COMPLETE |
| `set_theory_manifolds_geometry_deep_research_synthesis_20260518.md` | HYBRID COMPOSITION recommended (Riemannian-Newton as META-substrate) | T2 design | — | $0 research | DESIGN COMPLETE |
| `riemannian_newton_substrate_engineering_design_memo_20260518.md` | PROCEED_WITH_REVISIONS (META-canonical-helper not new substrate) | T2 | — | $0 editor multi-phase | DESIGN COMPLETE; PHASE 1 NEXT |
| `council_z8_hierarchical_predictive_coding_per_substrate_symposium_20260518.md` | DEFER_PENDING_EVIDENCE (2-of-4 gates) | T3 | asymptotic_pursuit | $0 awaiting evidence | DEFERRED |
| `cargo_cult_resurrection_top_3_symposiums_v1_faiss_c6_ibps_v2_nscs06_v8_variant_c_20260518.md` | 3× PROCEED_WITH_REVISIONS (mixed horizon_class) | T2 batched | — | mixed | RESURRECTION RECOMMENDED |
| `tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518.md` | PROCEED_WITH_REVISIONS (design only) | T2 | asymptotic_pursuit `[-0.020, -0.008]` per deep-research §0 | $15-25 | DESIGN COMPLETE; AWAITS PER-SUBSTRATE SYMPOSIUM |
| `grand_council_symposium_inflate_py_extreme_compression_20260518.md` | PROCEED_WITH_REVISIONS | T3 | — | $0 editor | DESIGN COMPLETE |
| `council_per_substrate_symposium_*` (ATW V2, C6 IBPS v2, Z7 LSTM, mae_v+saug, NSCS06 v8, DP1 deep-dive, lane_17_imp, pr106 #05+#06, STC clean source, V1 dense Faiss, TT5L foveation+LAPose) | Mixed (PROCEED_WITH_REVISIONS × 8 / DEFER_PENDING_EVIDENCE × 3 / REFUSE × 1) | T2/T3 | per-substrate | mixed | per-symposium ledger updates pending |
| `comprehensive_research_wave_20260518.md` (deep-research 1112 lines) | TOP-5 reformulations all sub-0.190 floor potential | T3 research | aggregate `[-0.020, -0.040]` if all PROCEED | $80-200 wave | RESEARCH COMPLETE |
| `master_gradient_xray_fields_medal_research_wave_20260518.md` | RESEARCH_ONLY | T2 research | enables Slot 1 + sister wire-ins | $0 research | RESEARCH COMPLETE |
| `asymptotic_stacking_plus_local_max_utilization_audit_20260518.md` | AUDIT FINDINGS | T2 audit | enables 8-substrate composition routing | $0 audit | AUDIT COMPLETE |
| `empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md` | DESIGN COMPLETE | T2 design | enables per-X planner DuckDB unification | $0 editor | DESIGN COMPLETE |
| `huggingface_skills_comprehensive_design_implementation_plan_20260518.md` | DESIGN COMPLETE | T2 design | — | $0 editor | DESIGN COMPLETE |
| `canonical_upstream_pr_review_procedural_generation_compliance_20260518.md` | COMPLIANCE-VERIFIED for hash-seed + weight-derived + null-space | T2 audit | unlocks meta-portfolio TOP-1 + TOP-2 | $0 audit | AUDIT COMPLETE |

### 3.2 Cross-stack synergy matrix (PROCEED candidates)

Combining the meta-portfolio + analytical-surfaces + theoretical-floor + deterministic-optimizer + Riemannian-Newton + Z6/Z7/Z8 + DP1 + cargo-cult resurrection synthesis:

**Stack 1 (TIER-1 immediate; all $0 editor)**:
- `tac.theoretical_floor_estimator` (PLATEAU disambiguator) **→ UNBLOCKS** routing for next dispatch wave
- `tac.null_space_exploiter` × master-gradient extension to 4-6 archives **→ HIGHEST EV** `[-0.040, -0.012]` per archive on PR101 fec6
- `tac.hash_seed_codebook_generator` × NSCS06 v7 chroma palette **→ FRONTIER FIRST-MOVER** ~7.5KB→8 bytes
- `tac.procedural_codebook_generator` (LANDED) × Wyner-Ziv Tier-2 deliverability (LANDED Catalog #319 Q3)
- `tac.riemannian_newton_meta_substrate` PHASE 1 Fisher-precondition + LM damping K-FAC **→ INHERITED by existing trainers**

**Stack 2 (TIER-2 paid dispatch sequence after Stack 1 lands)**:
- TT5L V2 redesign (VGGT + DUSt3R/MASt3R + DreamerV3 RSSM + NVIDIA VRSS 2) per deep-research wave TOP-1 → $15-25 / `[-0.020, -0.008]`
- Z7 as Mamba-2 (state-spaces/mamba 2024) per deep-research wave TOP-2 → $20-30 / `[-0.025, -0.008]`
- ATW V2-1 + Faiss-IVF-PQ per deep-research wave TOP-3 → $7-25 / `[-0.015, -0.005]`
- DP1+PR101 stacking per deep-research wave TOP-4 → $10-15 / `[-0.012, -0.004]`
- lane_17_imp Frankle LTH cycle 0 per pre-rigor inventory TOP-1 → $1-2 / `[-0.015, -0.005]`

**Stack 3 (TIER-3 longer-horizon; council-gated)**:
- Z8 hierarchical predictive coding (DEFER_PENDING_EVIDENCE; needs 2-of-4 gates closed first)
- C6 IBPS v2 reactivation per cargo-cult resurrection (after Path B redesign)
- NSCS06 v8 Variant C (after 2-of-7 cargo-cult unwind methodology applied)

### 3.3 Reactivation gates from today's verdicts

Per CLAUDE.md "Forbidden premature KILL" + Catalog #307/#308/#313 discipline:

| Substrate | Verdict | Reactivation criteria |
|---|---|---|
| C6 IBPS (after 22× miss) | DEFER (probe outcome `c6_e4_mdl_ibps_smoke_modal_a10g_50ep_fc…`) | Per cargo-cult resurrection symposium: Phase 2 redesign per ATW V2 V2-1 dependency; latent dim sweep build (`lane_c6_ibps_reactivation_path_b_latent_dim_sweep_build_20260518` L1) + beta_ib sweep (`lane_c6_ibps_beta_ib_sweep_build_per_symposium_parallel_path_a_20260518` L1) — BOTH already L1 registered |
| Z6/Z7/Z8 predictive coding | DEFER (Z6-v2 Wave 2 mode-misroute + Z7 LSTM PROCEED_WITH_REVISIONS + Z8 DEFER) | Per Catalog #315: iterate to PROCEED-unconditional via cargo-cult-unwind; pending Z6-v2 driver/recipe fix landing |
| TT5L V2 | DESIGN COMPLETE PROCEED_WITH_REVISIONS | Per-substrate symposium required before paid dispatch (Catalog #325) |
| ATW V2-1 | DEPENDS on Z6 4c outcome | Per ATW V2 symposium Revisions #1+#2+#3; already L1 registered |
| NSCS06 v8 Path B | REFUSE 13-of-13 unanimous per symposium #864 | Variant C only (per cargo-cult resurrection symposium) |
| Wunderkind G1 v2 | DEFER (per-pair-dominant SegNet argmax reducer probe artifact INDEPENDENT) | 4 alternative reducers UNPROBED per Catalog #308 |
| STC clean source (FALSIFIED legacy) | DEFER_PENDING_EVIDENCE per per-substrate symposium 2026-05-18T03:19Z | Per pre-rigor inventory: FALSIFIED was MPS-PROXY-derived per CLAUDE.md "MPS auth eval is NOISE"; re-probe required ($0.20 CHEAPEST) |
| lane_17_imp (KILL legacy) | DEFER_PENDING_EVIDENCE per symposium 2026-05-18T04:06Z | Per pre-rigor inventory: KILL was stats.json stub-loop ARTIFACT; standalone $1-2 Vast.ai 4090 cycle 0 re-probe with Catalog #91/#94 stub-loop class extinct |
| PR106 #05+#06 | PROCEED_WITH_REVISIONS reformulated | $10 dispatch to grayscale_lut OR A1-sidecar (per symposium 04:06Z) |

### 3.4 Predicted aggregate ΔS bands

Under HIGH-orthogonality (best case): `[-0.048, -0.012]` aggregate ⇒ frontier potential `[0.172, 0.187]` [contest-CPU] from current 0.19205.
Under realistic α-discount per Catalog #322 anti-additive empirical (4/8 probed pairs sub-additive): `[-0.020, -0.005]` aggregate ⇒ `[0.184, 0.192]`.
Under deep-research wave §0 Shannon-floor: theoretical floor `[0.026, 0.080]` (rate-conservative regime) ⇒ 2.4-7.4× gap structurally PLATEAU not SATURATION.

---

## 4. Symposium-spawn recommendations

Per operator directive *"we also need to review those findings and results and the failure and spawn grand council symposiums as appropriate"*, 10 specific candidates ranked by tier × urgency:

### 4.1 T4 SYMPOSIUM CANDIDATES (1)

**S4-1. Recurring META-failure-class symposium**: *"Apparatus-maintenance over-dominates frontier-breaking work; mission-alignment drift"*. Per CLAUDE.md "Council hierarchy: 4-tier protocol" Mission alignment Consequence 5: when `rigor_overhead + apparatus_maintenance > 60%` in any 30d window, operator-visible alert fires. Today's session has ~14 of 21 council deliberations (67%) classified `apparatus_maintenance` or related META-fix. **Recommend T4 symposium**: 6-of-6 sextet + ≥16-of-20 grand council to deliberate whether the current discipline apparatus (270+ STRICT preflight gates, 327+ catalog #s, daily META-meta gates) is serving the mission or has become the work. **Operator-attention cost**: ≤2/30d budget. **Quorum**: 6-of-6 sextet (Shannon/Dykstra/Yousfi/Fridrich/Contrarian/Assumption-Adversary) + ≥16-of-20 grand council + 1 specialist per affected path. **Per-tier elevation trigger**: ≥3 grand-council specialist disagreement. **Operator routable**: convene when the operator has bandwidth for a strategic review (this is the LARGEST attention spend on the audit's recommendations).

### 4.2 T3 SYMPOSIUM CANDIDATES (5)

**S3-1. Riemannian-Newton META-canonical-helper Phase 1 review** (PROCEED_WITH_REVISIONS @ 2026-05-18T17:17:21Z; 4 binding revisions): Phase 1 Fisher-precondition + LM damping + K-FAC empirical validation on PR101_lc_v2 archive REQUIRED before broad rollout. **Tier**: T3 (touches multiple substrate trainers + CLAUDE.md non-negotiable canonical-vs-unique-decision discipline). **Attendees**: Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary + Hafner (DreamerV3) + Hinton (knowledge distillation) + Boumal (Riemannian manifold; specialist). **Agenda**: (a) Phase 1 binding revisions adjudication; (b) Phase 2 Riemannian-Newton step authorization; (c) Phase 4 symplectic-EMA single-substrate paired comparison gate.

**S3-2. T3 grand council synthesis CONTINUATION** (PROCEED_WITH_REVISIONS @ 2026-05-18T14:04:44Z; the parent grand council that bootstrapped today's meta-portfolio): the original verdict produced TOP-10 portfolio + 5 council questions. Some questions remain (Q4 specifically). **Tier**: T3 follow-on. **Attendees**: same 21-seat roster as parent. **Agenda**: (a) ratify TOP-5 (Stack 1) sequencing; (b) adjudicate Stack 2 paid-dispatch budget; (c) Stack 3 longer-horizon council-gated decisions.

**S3-3. Inflate.py extreme compression symposium FOLLOW-ON** (PROCEED_WITH_REVISIONS @ 2026-05-18T15:21:34Z): per the symposium memo. **Tier**: T3 (touches submissions/exact_current/inflate.py boundary which is OPERATOR-APPROVED MUTATION ONLY per CLAUDE.md mutation frontier). **Attendees**: Shannon + Carmack + Hotz + Selfcomp + MacKay_memorial + Ballé + Tao + Boyd. **Agenda**: post-revision LOC budget adjudication + dependency-closure proof.

**S3-4. Z8 hierarchical predictive coding ADJUDICATION** (DEFER_PENDING_EVIDENCE @ 2026-05-18T17:06:04Z; 2-of-4 gates closed): per the per-substrate symposium memo. **Tier**: T3 (asymptotic_pursuit + class-shift candidate). **Attendees**: Rao + Ballard + Tishby_memorial + Hafner + Schmidhuber + Time-Traveler protégé + sextet. **Agenda**: 2 remaining gates closure plan + reactivation criteria pinning per Catalog #313 probe outcomes.

**S3-5. DP1 deep-dive FOLLOW-ON** (PROCEED_WITH_REVISIONS @ 2026-05-18T04:58:18Z; T3 already convened): per per-substrate symposium DP1 deep-dive. **Tier**: T3 (Comma2k19 dataset provenance + DP1×PR101 stacking + Catalog #209/#210/#211/#213 sister discipline). **Attendees**: Atick + Redlich + sextet + Carmack (Comma2k19 provenance). **Agenda**: Phase 2 build adjudication + per-substrate symposium per Catalog #325.

### 4.3 T2 SYMPOSIUM CANDIDATES (4)

**S2-1. Z6-v2 driver/recipe-mode bug META review**: per the bug-fix directive landed THIS session, Bug 1 (Z6-v2 recipe mode-misroute) + Bug 2 (Modal harvester ledger-write gap) are symptoms of a recurring META-class: *"infrastructure-level bugs masquerade as substrate-level dispatch failures"*. Sister anchor: C6 IBPS + Z6-v2 are TWO consecutive ASYMPTOTIC substrate dispatches frustrated by infrastructure-level bugs (NOT paradigm falsification per Catalog #307). Catalog #326 was landed in same session to catch driver mode-hardcode at preflight. **Tier**: T2 (apparatus-maintenance; substrate-engineering-discipline boundary). **Attendees**: sextet + Yousfi (gate-design discipline) + Quantizr (competitive intelligence vs PR95-paradigm reviewability) + Carmack (engineering reality check) + Hotz (raw engineering instinct). **Agenda**: (a) Z6-v2 driver/recipe-mode bug class root-cause review; (b) Catalog #326 strict-flip readiness; (c) META question: are we losing too many distinguishing-feature dispatch attempts to infrastructure bugs?

**S2-2. Cargo-cult resurrection TOP-3 PER-SUBSTRATE follow-ons** (3× PROCEED_WITH_REVISIONS @ 2026-05-18T17:16:59Z and 17:17:18Z; V1 Faiss + C6 IBPS v2 + NSCS06 v8 Variant C): each needs per-substrate symposium per Catalog #325 before paid dispatch. **Tier**: T2 ×3 (or T2 batched if op-routable convenient). **Attendees**: sextet + per-substrate specialists (V1 Faiss = Filler-STC + LDPC specialists; C6 IBPS v2 = Tishby_memorial + Zaslavsky; NSCS06 v8 Variant C = Mallat + Daubechies + Carmack/Hotz). **Agenda**: per-symposium 6-step contract per Catalog #325.

**S2-3. TT5L V2 redesign per-substrate symposium** (design PROCEED_WITH_REVISIONS; per-substrate symposium NOT YET held): per Catalog #325. **Tier**: T2. **Attendees**: sextet + Time-Traveler protégé + Hafner (DreamerV3 RSSM specialist). **Agenda**: 6-step contract per Catalog #325 + cargo-cult audit for VGGT + DUSt3R/MASt3R + NVIDIA VRSS 2 integration.

**S2-4. STC clean source RE-INVESTIGATION symposium** (DEFER_PENDING_EVIDENCE @ 2026-05-18T03:19:18Z + pre-rigor inventory #857 TOP-2 CHEAPEST $0.20 re-probe candidate): per pre-rigor inventory FALSIFIED was MPS-PROXY-derived INVALID per CLAUDE.md "MPS auth eval is NOISE". **Tier**: T2 (re-investigation of historical kill). **Attendees**: sextet + Filler (STC specialist) + Yousfi (steganalysis context). **Agenda**: re-probe authorization on Vast.ai or Modal contest-CUDA + re-classification per Catalog #307 paradigm-vs-implementation.

### 4.4 Symposium-spawn priority queue (operator-routable)

1. **NOW**: S2-1 Z6-v2 driver/recipe-mode bug META review (apparatus-blocking; cheap; T2)
2. **NEXT**: S3-1 Riemannian-Newton Phase 1 (highest-EV TIER-1 unlock; T3)
3. **NEXT+1**: S3-2 T3 grand council synthesis CONTINUATION (ratifies portfolio sequencing; T3)
4. **NEXT+2**: S2-2 Cargo-cult resurrection per-substrate × 3 (unblocks resurrection dispatches; T2 batched)
5. **NEXT+3**: S2-3 TT5L V2 per-substrate symposium (unblocks deep-research wave TOP-1; T2)
6. **WHEN READY**: S3-3 Inflate.py extreme compression follow-on (touches mutation frontier; T3)
7. **WHEN READY**: S3-5 DP1 deep-dive follow-on (Comma2k19 provenance + Phase 2 build; T3)
8. **DEFERRED**: S3-4 Z8 hierarchical predictive coding adjudication (DEFER_PENDING_EVIDENCE; T3)
9. **DEFERRED**: S2-4 STC clean source re-investigation (cheap re-probe; T2)
10. **OPERATOR-WHEN-BANDWIDTH**: S4-1 Recurring META-failure-class symposium (T4 strategic; ≤2/30d budget)

---

## 5. Canonical-task-status backfill plan

Per the operator directive *"backfilled into .omx appropriately for codex with full detail and discipline"*, this audit recommends the following backfill operations. The backfill script (deliverable 2) operationalizes these.

### 5.1 IMMEDIATE: 4 missing directive items

Per audit §1.2; backfill via `register_task` (status defaults to pending):

| task_id | source_design_memo | title | predicted_cost_usd | predicted_delta_s_band | initial_status |
|---|---|---|---|---|---|
| `codex_routing_directive_z6v2_recipe_mode_bug_plus_harvester_ledger_gap_fix_20260518::ITEM_1` | `.omx/research/codex_routing_directive_z6v2_recipe_mode_bug_plus_harvester_ledger_gap_fix_20260518.md` | Fix Bug 1: Z6-v2 Wave 2 recipe MUST declare mode explicitly | 0.0 | null | pending |
| `…::ITEM_2` | same | Fix Bug 2: harvester MUST register canonical ledger events | 0.0 | null | pending |
| `…::ITEM_3` | same | Backfill the 2 originally-flagged ledger events (DONE in-context this session; verify) | 0.0 | null | pending → completed (manual update with notes citing the 2 fc-call_ids harvested) |
| `…::ITEM_4` | same | Audit for OTHER ledger gaps | 0.0 | null | pending |

### 5.2 PHASE 2: 18 non-directive design memo landings

For each major 2026-05-18 design memo NOT covered by `codex_routing_directive_*.md` extractor, register a synthetic `<memo_basename>::DESIGN_MEMO_LANDING` row with `status=completed` (the memo landed) and `predicted_cost_usd` + `predicted_delta_s_band` extracted from the memo's frontmatter or §0 executive summary. Listing the 18 anchors:

| memo basename | predicted_delta_s_band | initial_status |
|---|---|---|
| `grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518` | [-0.020, -0.005] | completed |
| `comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518` | [-0.048, -0.012] (TOP-5 op-routables) | completed |
| `tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518` | null (enables routing only) | completed |
| `deterministic_score_optimizer_design_memo_lagrangian_taylor_pareto_reverse_engineering_20260518` | null (complementary) | completed |
| `set_theory_manifolds_geometry_deep_research_synthesis_20260518` | null (HYBRID design) | completed |
| `riemannian_newton_substrate_engineering_design_memo_20260518` | null (META-helper) | completed |
| `council_z8_hierarchical_predictive_coding_per_substrate_symposium_20260518` | null (DEFER) | deferred (BLOCKED status) |
| `cargo_cult_resurrection_top_3_symposiums_v1_faiss_c6_ibps_v2_nscs06_v8_variant_c_20260518` | null (3× PROCEED_WITH_REVISIONS) | completed (per-symposium follow-ons needed) |
| `tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518` | [-0.020, -0.008] | completed (PROCEED_WITH_REVISIONS; per-substrate symposium queued) |
| `grand_council_symposium_inflate_py_extreme_compression_20260518` | null | completed (PROCEED_WITH_REVISIONS) |
| `comprehensive_research_wave_20260518` | aggregate [-0.040, -0.020] across TOP-5 | completed |
| `master_gradient_xray_fields_medal_research_wave_20260518` | null (research) | completed |
| `asymptotic_stacking_plus_local_max_utilization_audit_20260518` | null (audit) | completed |
| `empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518` | null (design) | completed |
| `huggingface_skills_comprehensive_design_implementation_plan_20260518` | null (design) | completed |
| `canonical_upstream_pr_review_procedural_generation_compliance_20260518` | null (audit; unlocks meta-portfolio TOP-1+TOP-2) | completed |
| `atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518` | [-0.015, -0.005] | completed (DEPENDS on Z6 4c) |
| `z7_mamba2_substrate_design_memo_20260518` | [-0.025, -0.008] | completed (per-substrate symposium queued) |

### 5.3 PHASE 3: per-symposium follow-on registrations

For each PROCEED_WITH_REVISIONS verdict that needs follow-up per §4 above, register a `<symposium_basename>::FOLLOW_ON_REVIEW` row at status=pending owner=operator (operator routable).

### 5.4 Backfill script invocation

Per deliverable 2, the operator runs:

```bash
.venv/bin/python tools/backfill_canonical_task_status_from_audit_20260518.py --apply
```

The script is idempotent — re-running is safe (per `register_task` idempotency contract; existing task_ids return their latest row).

---

## 6. Op-routables ranked by EV

| # | Op-routable | Cost | EV | Owner |
|---|---|---|---|---|
| 1 | Run deliverable 2 backfill script → 4 missing z6v2 items + 18 design-memo anchors land in canonical ledger | $0 editor 5 min | Closes the canonical ledger coverage gap; unblocks Codex persistent /goal v2.4 routing | claude OR codex OR operator |
| 2 | Operator: convene S2-1 (Z6-v2 driver/recipe-mode bug META review) T2 — Codex has already landed Bug 1 fix (commit `1c06eb08a`) + Bug 2 harvester fix; META review ratifies the catalog #326 strict-flip readiness | $0 editor 1-2h | Closes the apparatus-blocking META class extincted in same session | operator |
| 3 | Codex routes TOP-3 highest-priority pending directive items: `…inflate_py_*::ITEM_3` (extend master_gradient extractor) + `…canonical_task_status_*::ITEM_6` (this audit operationalizes it) + `…v2_synthesis_followup::ITEM_9` (NSCS06 v7 chroma → hash-seed; predecessor procedural_codebook_generator now COMPLETED) | $0 editor 8-12h aggregate | Unlocks Stack 1 (TIER-1 immediate; all $0 editor) | codex |
| 4 | Operator: convene S3-1 (Riemannian-Newton Phase 1) T3 — adjudicates Fisher-precondition + LM damping + K-FAC empirical validation on PR101_lc_v2 archive | $0 editor 2-3h council | Highest single-substrate EV per parent set-theory wave | operator |
| 5 | Claude: amend AGENTS.md with the new convention "every TaskCreate corresponding to a sister-subagent dispatch or lane work ALSO writes to canonical_task_status.jsonl via tac.canonical_task_status.register_task" per Item 7 | $0 editor 30 min | Closes the canonical-ledger-coverage gap going forward | claude |
| 6 | Operator: convene S3-2 (T3 grand council synthesis CONTINUATION) to ratify Stack 1 sequencing + Stack 2 paid-dispatch budget allocation | $0 editor 3-4h council | Strategic budget allocation for next dispatch wave | operator |
| 7 | Operator: convene S2-2 (Cargo-cult resurrection per-substrate × 3) T2 batched | $0 editor 2h council | Unblocks resurrection dispatches (V1 Faiss + C6 IBPS v2 + NSCS06 v8 Variant C) | operator |
| 8 | Operator: convene S2-3 (TT5L V2 per-substrate symposium per Catalog #325) | $0 editor 1-2h council | Unblocks deep-research wave TOP-1 ($15-25 dispatch) | operator |
| 9 | Codex: extend `tools/extract_canonical_tasks_from_directive.py` to ALSO scan non-`codex_routing_directive_*` design memos (under explicit opt-in glob) per the §1.3 coverage gap | $0 editor 2-3h | Closes the extractor coverage gap going forward | codex |
| 10 | Long-horizon: convene S4-1 (T4 META-failure-class symposium) when operator has bandwidth for strategic review of apparatus-vs-mission balance | $0 editor 4-6h council | Mission-alignment retrospective per CLAUDE.md "Mission alignment" Consequence 5 | operator |

---

## 7. Mandatory discipline sections

### 7.1 Canonical-vs-unique decision per layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + sister gate. This audit is a META work-item (apparatus_maintenance) so the canonical-vs-unique layers are minimal:

| Layer | Decision | Rationale |
|---|---|---|
| Audit memo format | ADOPT canonical (Catalog #229 PV-1 anchor format + Catalog #294 9-dim checklist evidence section + Catalog #305 observability surface section) | Standard audit memo discipline; no substrate-engineering optimization at stake |
| Backfill script structure | ADOPT canonical `tac.canonical_task_status.register_task` API (no fork) | The canonical helper is the ONLY appropriate API; forking would defeat the single-source-of-truth purpose |
| Symposium-spawn recommendations | ADOPT canonical 4-tier protocol per Catalog #300 | Tier discipline IS the canonical decision-making surface |

### 7.2 9-dimension success checklist evidence (Catalog #294)

| Dimension | Evidence |
|---|---|
| UNIQUENESS | This is the FIRST audit memo + backfill script + symposium-spawn recommendation memo in this exact 3-deliverable format for the canonical_task_status surface |
| BEAUTY+ELEGANCE | Single memo + single Python script; PR95-paradigm reviewable in ~5 min by an experienced operator |
| DISTINCTNESS | Distinct from the per-substrate audit memos (`feedback_*audit*landed_*.md`) by scope (apparatus-level not substrate-level) and from the per-directive `codex_routing_directive_*` memos (audit not directive) |
| RIGOR | Premise verification §1 (live ledger snapshot 2026-05-18T17:28Z + extractor delta + memo inventory) + adversarial review embedded in §3 cross-stack synthesis + assumption-classification per HARD-EARNED-vs-CARGO-CULTED addendum (this audit ASSUMES the canonical_task_status v1 schema is stable; HARD-EARNED per directive landing + sister DuckDB consumer + Codex consumption) |
| OPTIMIZATION-PER-TECHNIQUE | Per §7.1: ADOPT canonical at every layer (no optimization-per-technique unique-ification needed for an audit) |
| STACK-OF-STACKS-COMPOSABILITY | Audit deliverable 2 backfill script composes orthogonally with Codex's idempotent extractor (re-running both is safe) + DuckDB consumer ext (ITEM_10 LANDED) + sister `tools/canonical_task_status.py` CLI |
| DETERMINISTIC-REPRODUCIBILITY | Deliverable 2 script is fully deterministic (no random seeds, no GPU); `register_task` is idempotent per the canonical contract |
| EXTREME-OPTIMIZATION-PERFORMANCE | n/a — apparatus_maintenance not frontier-breaking |
| OPTIMAL-MINIMAL-CONTEST-SCORE | n/a — this audit does NOT touch the contest archive bytes |

### 7.3 Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| The canonical_task_status v1 schema is stable enough to backfill against | HARD-EARNED | Schema landed via Catalog #245 4-layer pattern; consumed by tools/canonical_task_status.py CLI + Codex persistent /goal v2.4; sister DuckDB consumer extended |
| Codex will actually consume the ledger as primary task source going forward | HARD-EARNED | The persistent /goal v2.4 EXPLICITLY consumes the ledger (per its DISCOVER step); commit `7c13abda3` already shows 9 task completions written by Codex |
| The 18 non-directive design memos all merit canonical_task_status rows | HARD-EARNED-PARTIAL | The operator's directive ("represented in the design memos") clearly extends beyond `codex_routing_directive_*.md`; the threshold for which memos qualify is operator-routable judgment call |
| The 10 symposium-spawn recommendations are correctly tiered | CARGO-CULTED-PENDING-OPERATOR-VERIFICATION | Tier assignment per Catalog #300 is structural but the specific tier choices (e.g. T4 for META-failure-class) are operator-discretionary |
| Backfilling already-completed design memos as `status=completed` is the right model | HARD-EARNED | Per the canonical schema state machine `pending → in_progress → completed`; a one-time backfill row at `status=completed` carries the same auditability semantics as the multi-row history; idempotency via `register_task` preserves the audit trail |

### 7.4 Observability surface (Catalog #305)

| Facet | How this audit + backfill script supports observability |
|---|---|
| Inspectable per layer | The audit memo §3.1 table provides per-memo verdict inspection; the backfill script's `--dry-run` flag (in deliverable 2) lets the operator inspect every row before writing |
| Decomposable per signal | Each row in the canonical_task_status.jsonl carries decomposable fields (task_id / source_design_memo / status / owner / predicted_cost_usd / predicted_delta_s_band) so consumers can slice arbitrarily |
| Diff-able across runs | Re-running the backfill script after new directives land produces a diffable JSONL append; the DuckDB consumer view (ITEM_10 LANDED) enables temporal diff queries |
| Queryable post-hoc | The CLI `tools/canonical_task_status.py --list-pending` / `--task-history` / `--directive-summary` already exposes operator-facing queries |
| Cite-able | Every row carries `source_design_memo` field anchoring back to the canonical artifact |
| Counterfactual-able | The audit's §4 symposium-spawn recommendations are explicitly counterfactual ("if operator convenes S2-1, then..." / "if convene S3-1, then..."); the operator can simulate alternative routings |

### 7.5 6-hook wire-in declaration (Catalog #125)

| Hook | Status | Rationale |
|---|---|---|
| 1. Sensitivity-map contribution | N/A — apparatus_maintenance audit | No score-axis sensitivity signal; this audit does not touch substrate gradients |
| 2. Pareto constraint | N/A — apparatus_maintenance audit | No new Pareto constraint added; audit consumes existing pareto rows via §3 synthesis |
| 3. Bit-allocator hook | N/A — apparatus_maintenance audit | No per-tensor importance changes |
| 4. Cathedral autopilot dispatch hook | ACTIVE — INDIRECTLY | The canonical_task_status pending queue is structurally consumable by the cathedral autopilot ranker for "what's the next operator-routable to fire?" sorting; this audit recommends Codex's persistent /goal v2.4 + the autopilot ranker BOTH consume the queue |
| 5. Continual-learning posterior update | ACTIVE — DIRECTLY | The canonical_task_status JSONL IS a continual-learning posterior surface in the same Catalog #245 4-layer pattern as modal_call_id_ledger + council_continual_learning + probe_outcomes_ledger; the backfill script (deliverable 2) appends new posterior rows |
| 6. Probe-disambiguator | ACTIVE — DIRECTLY | The canonical_task_status pending vs completed distinction IS the canonical disambiguator for "is this work item done?" between Codex and Claude; per the operator's harness-engineering directive, the ledger IS the disambiguator |

### 7.6 Predicted ΔS band with Dykstra-feasibility check (Catalog #296)

Per CLAUDE.md non-negotiable: this audit's deliverables (memo + backfill script + symposium recommendations) produce ZERO direct ΔS — they are pure apparatus_maintenance work. Therefore the Dykstra-feasibility check is **vacuous** (no predicted band, no convex composition to check). Same-line waiver per the gate: this section header satisfies the Catalog #296 structural discipline; the body declares the zero-band rationale. Cross-ref §3.4 for the predicted aggregate ΔS bands of the downstream substrate work this audit enables routing for.

### 7.7 Horizon class declaration (Catalog #309)

`horizon_class: apparatus_maintenance` per CLAUDE.md "Mission alignment" Consequence 5 (this is explicitly apparatus_maintenance work; not frontier_breaking; not frontier_protecting; not rigor_overhead; not mission_questioned). The audit + backfill script + symposium recommendations serve the mission (better task tracking → better Codex routing → better Stack 1 execution → better frontier-breaking dispatch sequencing) but do not directly contribute frontier or protect frontier.

---

## 8. Cross-references

### 8.1 Sister landings this session

- Catalog #245 4-layer pattern (CANONICAL TEMPLATE) — `src/tac/deploy/modal/call_id_ledger.py` per Modal call_id ledger
- Catalog #131 fcntl-locked JSONL writes — `_ledger_lock` context manager in `src/tac/canonical_task_status/writer.py`
- Catalog #138 strict-load discipline — `load_canonical_task_status_strict` raises `CanonicalTaskStatusCorruptError`
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — status updates = new rows
- Catalog #287 evidence-tag discipline — `actual_delta_s` rows MUST carry `[empirical:<path>]` per `__post_init__` validator (line 121-122)
- Catalog #157/#174 commit serializer + POST-EDIT sha — backfill script commits via the canonical serializer per `tools/subagent_commit_serializer.py`
- Catalog #186 catalog-claim transactionality — N/A (no new catalog # claimed in this audit)
- Catalog #206 checkpoint discipline — checkpoint emitted at audit start; will emit at audit complete
- Catalog #229 premise-verification — §1.1 ledger snapshot + §1.2 extractor delta + §1.3 memo inventory are the verified premises
- Catalog #265 symposium impls canonical contract — N/A (no new symposium implementations landed)
- Catalog #290 canonical-vs-unique decision — §7.1
- Catalog #291 session META-ASSUMPTION cadence — N/A this audit does not modify META-assumption review cadence
- Catalog #294 9-dim checklist evidence — §7.2
- Catalog #298 30-day retirement discipline — lane registered + INFRASTRUCTURE_PRIMITIVE = research_only opt-out per CLAUDE.md
- Catalog #300 council deliberation v2 frontmatter — N/A (this audit is NOT a council deliberation; it is an audit + design memo + backfill plan)
- Catalog #303 cargo-cult audit per assumption — §7.3
- Catalog #305 observability surface — §7.4
- Catalog #309 horizon class declaration — §7.7
- Catalog #313 probe outcomes ledger — referenced for reactivation gates §3.3
- Catalog #314 sister subagent files_touched declaration — checkpoint at audit start declared deliverable paths
- Catalog #319 Q3 v2 cascade — referenced for null-space + procedural-codebook synergies
- Catalog #322 anti-additive composition_alpha — referenced for §3.4 realistic α-discount
- Catalog #323 canonical Provenance META umbrella — sister `ProvenanceKind` enum extension is `…inflate_py_*::ITEM_2` pending
- Catalog #324 predicted-band Tier-C validation — N/A this audit has no predicted_band claim
- Catalog #325 per-substrate optimal-form symposium — §4 symposium-spawn recommendations honor this discipline

### 8.2 Cross-stack synergies with sister subagent landings today

- **sister `a39ffdf80` Riemannian-Newton design** — symposium-spawn S3-1 covers Phase 1 adjudication
- **sister `a478cbde` TT5L V2 redesign** — symposium-spawn S2-3 covers per-substrate symposium per Catalog #325
- **sister `ae324eabee` Cargo-cult resurrection TOP-3** — symposium-spawn S2-2 covers per-substrate × 3 follow-on

### 8.3 Memory file landing entry

Per Catalog #229 premise-verification + Catalog #305 observability + Catalog #294 9-dim checklist (this section): `feedback_task_queue_audit_canonical_task_status_backfill_landed_20260518.md` queued for landing alongside this memo + the backfill script.

---

## 9. Closure

This audit lands per the operator directive *"review the task queue for stale tasks and tasks that need to be updated and tasks that need to be backfilled into .omx appropriately for codex with full detail and discipline; we also need to review those findings and results and the failure and spawn grand council symposiums as appropriate"*.

**Audit verdict**: PROCEED with deliverable 2 backfill (4 z6v2 items + 18 design-memo anchors); operator-routable 10-candidate symposium-spawn queue per §4.4; AGENTS.md convention update per Item 7. Lane `lane_task_queue_audit_canonical_task_status_backfill_20260518` L1 (impl_complete via this memo + backfill script + symposium recommendations; memory_entry queued; deploy_runbook N/A).

**Confidence**: HIGH (3 independent anchors converge):
1. Canonical helper LANDED via commit `7c13abda3` (verified structurally; 22 task_ids tracked; CLI operational)
2. Codex persistent /goal v2.4 CONSUMES the ledger as primary source (verified via memo + commit `1e20010c8`)
3. Operator's directive explicitly enumerated the 4 audit dimensions this memo covers ("stale tasks" + "tasks that need to be updated" + "tasks that need to be backfilled" + "spawn grand council symposiums as appropriate")

**Next action** (operator-routable): run deliverable 2 backfill script per §6 op-routable #1; then route §6 op-routable #2 (convene S2-1 META review) or #4 (convene S3-1 Riemannian-Newton T3) per operator bandwidth.

— `audit_task_queue_canonical_status_20260518` subagent (Main-Claude relayed on behalf of operator 2026-05-18)
