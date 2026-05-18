---
memo_kind: closure_campaign_master
memo_date_utc: 2026-05-18
memo_subject: PURSUE-AND-CONFIRM 5 OP-AUDIT closure operations to COMPLETE + CORRECT verdicts
author: claude_main_session_9518b12a
lane_id: lane_closure_campaign_pursue_and_confirm_20260518
predecessor_audit: .omx/research/wiring_integration_orphan_audit_post_12_landings_20260518.md
predecessor_audit_commit: b1aae8536
sister_synthesis: .omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md
sister_synthesis_commit: 14c03c57a
operator_directive_verbatim: "must pursue and confirm all closure operations complete and correct"
horizon_class: apparatus_maintenance
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: "n/a"
predicted_band_validation_status: not_applicable
related_deliberation_ids:
  - wiring_integration_orphan_audit_post_12_landings_20260518
  - cross_stack_synthesis_9_design_landings_unified_framework_20260518
  - council_t3_grand_council_synthesis_all_research_eureka_engineering_meta_20260518
---

# Closure-Campaign Pursue-and-Confirm Master Memo

**Operator NON-NEGOTIABLE 2026-05-18 verbatim**: *"must pursue and confirm all closure operations complete and correct"*.

This memo is the **closure-campaign coordination surface** for the 5 OP-AUDIT operations identified in commit `b1aae8536` (`.omx/research/wiring_integration_orphan_audit_post_12_landings_20260518.md` §13). It does BOTH:

1. **PURSUE** — verifies routing-directive landings, identifies routing gaps, writes missing routing directives, registers canonical task status rows.
2. **CONFIRM** — designs the **closure-completion-verification framework** that validates each OP is COMPLETE *and* CORRECT (not just shipped) via canonical helper `tac.closure_completion_verifier`.

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #125 (6-hook wire-in declaration) + Catalog #245 (canonical 4-layer pattern) + Catalog #300 (council deliberation v2 frontmatter) + Catalog #313 (probe outcomes ledger) + Catalog #325 (per-substrate symposium evidence).

---

## 0. TL;DR — Executive summary

### 0.1 Per-OP current routing status (empirically verified)

| OP | Description | Cost | Routing status | Codex execution status | Verification status |
|---|---|---|---|---|---|
| **OP-AUDIT-1** | Codex executes OP-SYN-1 master-gradient extractor 6-archive extension | $0 + ~6-12h CPU | **ROUTED** (directive `op_syn_1_master_gradient_six_archive_extension_20260518.md` at commit `699fe19e6`) | **IN PROGRESS** (DP1 grammar fail-closed registered per commit `e49735449`; extract-all manifest runner per commit `04e1ea086`; PR106/PR107 projectors pending) | **VERIFIER PENDING** — `verify_closure_operation("OP-AUDIT-1")` returns IN_PROGRESS |
| **OP-AUDIT-2** | 3 NEW 4-layer canonical helpers (inbox + memory-export + hypergraph) | $0 + ~6 days editor | **ROUTED** × 3 (directives 10/13/14 landed) | **NOT STARTED** (zero canonical_task_status rows; Codex hasn't ingested) | **VERIFIER PENDING** — returns ROUTING_DIRECTIVE_LANDED |
| **OP-AUDIT-3** | 4 Tier-1 design memo canonical-helper packages (VENN + Fisher + Riemannian-Newton + Tropical d_seg) | $0 + ~12-15 days editor | **NOT ROUTED** | **NOT STARTED** | **NEEDS 4 NEW ROUTING DIRECTIVES** (this memo lands them in same commit batch) |
| **OP-AUDIT-4** | DP1+PR101 Path A canonical helper (full helper + Stage 2 Modal $5-15) | $0 editor + $5-15 Modal A100 | **PARTIALLY ROUTED** (`dp1_pr101_op1_op2_zero_cost_probes` covers OP-1+OP-2 $0 probes ONLY) | **NOT STARTED** | **NEEDS 1 NEW ROUTING DIRECTIVE** (full Path A helper) |
| **OP-AUDIT-5** | Multi-loop /goal v2.5.2 paste | $0 operator action | **LANDED** (v2.5.2 at commit `05c74c245`) | **N/A — operator action** | **VERIFIER PENDING** — checks `codex_persistent_session_state.jsonl` for `directive_executed=...inbox_integration...` row |

### 0.2 What this memo delivers

1. **Per-OP COMPLETE criteria** (§4) — explicit checklist per OP (file paths + test counts + state-ledger row counts + integration hooks)
2. **Per-OP CORRECT criteria** (§5) — explicit checklist per OP (tests green / preflight passing / 6 hooks declared / empirical anchor where applicable)
3. **Closure-completion-verification framework design** (§6) — `tac.closure_completion_verifier` API spec + `ClosureOperationVerdict` enum + nightly cron monitor + dashboard
4. **Routing-directive gap inventory** (§7) — explicit list of which directives are landed vs which need landing
5. **5 routing directives land in same commit batch** (separate files; this memo references them)
6. **TOP-5 op-routables ranked by EV** (§11) — execution-ordering for the closure campaign
7. **Sister-agent feedback loop** (§13) — Codex executes the closure work; this verifier monitors completion; Codex→Claude inbox channel surfaces blockers

### 0.3 Ranked op-routables (full table §11)

| Rank | Op-routable | Action | Estimated effort | EV (unblock_count / cost) |
|---|---|---|---|---|
| 1 | **OPR-CLOSE-1** | Codex execute OP-AUDIT-1 master-gradient 6-archive (DP1 + PR106_format0d + PR107_apogee projectors + extract-all batch runner) | $0 + ~6-12h CPU | ∞ (unblocks ALL 9 Tier-1 designs) |
| 2 | **OPR-CLOSE-2** | Land 4 NEW routing directives for OP-AUDIT-3 (this memo's deliverable batch 2) | ~2h editor | unblocks 4 Tier-1 canonical helpers (~1750 LOC) |
| 3 | **OPR-CLOSE-3** | Codex execute OP-AUDIT-2 (3 channels: inbox + memory-export + hypergraph) | $0 + ~6 days editor | unblocks observability surface + sister-agent inbox + memory hermetic export |
| 4 | **OPR-CLOSE-4** | Land 1 NEW routing directive for OP-AUDIT-4 (DP1+PR101 Path A full canonical helper) | ~1h editor | unblocks DP1+PR101 Modal A100 50ep smoke ($5-15) |
| 5 | **OPR-CLOSE-5** | Build `tac.closure_completion_verifier` canonical helper (3-layer mini-pattern) | $0 + ~1 day editor | makes closure campaign STRUCTURALLY queryable across sessions |

### 0.4 Council verdict matrix per Catalog #300

| Field | Value |
|---|---|
| Tier | T2 sextet pact |
| Quorum | 6/6 (no recusal — closure campaign topic does not overlap any member's authored work) |
| Verdict | **PROCEED_WITH_REVISIONS** |
| Dissent | Contrarian + Assumption-Adversary as recorded in §12 |
| Mission contribution | apparatus_maintenance (per Catalog #300 §"Mission alignment" enum) |
| Override invoked | false |
| Predicted band validation | not_applicable (apparatus-maintenance produces no score band) |

### 0.5 4 binding revisions per council dissent

1. **Per Contrarian** — closure-verification framework MUST distinguish COMPLETE from CORRECT separately so a half-shipped helper (tests pending) is NOT reported as fully closed. The 9-verdict taxonomy in §6.2 satisfies this.
2. **Per Assumption-Adversary** — the assumption "Codex's autonomous loop will close OP-AUDIT-2 + OP-AUDIT-3 without operator escalation" is CARGO-CULTED-PENDING-EMPIRICAL per sister audit §9.3. The verifier MUST emit operator-visible alerts when an OP stays in ROUTING_DIRECTIVE_LANDED state >7 days.
3. **Per Dykstra (CO-LEAD)** — Pareto-feasibility of "all 5 OPs close in parallel" is FALSE; they sequence per the dependency graph in §11.2 (OP-AUDIT-1 gates OP-AUDIT-3; OP-AUDIT-2 inbox channel gates OP-AUDIT-5). The TOP-5 op-routable ordering reflects this.
4. **Per Shannon (LEAD)** — the closure-completion verifier IS the canonical observability surface per Catalog #305 6-facet definition. This memo's §6 design spec satisfies all 6 facets (inspectable per layer / decomposable per signal / diff-able across runs / queryable post-hoc / cite-able / counterfactual-able).

---

## 1. Mission alignment per CLAUDE.md

Per CLAUDE.md "Mission alignment — non-negotiable" subsection: *"discipline serves the mission, NOT the reverse"*. This closure campaign is classified `apparatus_maintenance` per Catalog #300 §"Mission alignment" enum. The apparatus serves the frontier — without closure verification, the 5 OPs accumulate as silent integration debt: routing directives land, Codex executes asynchronously, and signal is lost when an OP gets stuck at IN_PROGRESS for weeks without operator-visible alert.

**Concrete connection to frontier 0.19205 [contest-CPU GHA Linux x86_64] (PR101 fec6 archive `6bae0201`)** per Catalog #316:

- **OP-AUDIT-1** unblocks per-archive master-gradient anchors for the 6 canonical archives (currently only `fec6_fp11_selector` + `pr101_lc_v2` + `a1_finetuned` emit real anchors; `dp1` + `pr106_format0d` + `pr106_packed` + `pr107_apogee` are fail-closed pending projector). Tier-1 designs (VENN/FISHER/RIEM/TROP) ALL depend on the 6-archive matrix for empirical anchoring.
- **OP-AUDIT-2** unblocks the Codex→Claude inbox channel which enables continual-learning feedback loop per AGENTS.md "Continual Learning Feedback Loop (canonical memo patterns)" section. Without the inbox, Codex's empirical findings (e.g. premise falsifications, bug-class discoveries) require operator-mediated relay.
- **OP-AUDIT-3** unblocks the 4 Tier-1 canonical helpers that compose the unified-Lagrangian action per Catalog #125 hook #1 migration target. Each helper adds a measurable ΔS per Tier-1 design's predicted band (FISHER `[-0.015, -0.005]`; RIEM `[-0.025, -0.008]`; VENN `-0.005`; TROP Phase 1 `[-0.010, -0.002]`).
- **OP-AUDIT-4** unblocks the DP1+PR101 Stage 2 Modal A100 smoke with predicted band `[0.180, 0.190]` per memo 7 §3.2 — a paid empirical anchor that resolves whether DP1's pretrained prior accelerates PR101 HNeRV training.
- **OP-AUDIT-5** activates the 5-loop coordination per multi-loop /goal design memo §1, structurally extincting the "Codex's continual-learning state is invisible to Claude" bug class.

**Race-mode rigor inversion trigger** per CLAUDE.md "Race-mode rigor inversion" non-negotiable: if the public leaderboard moves with a new sub-0.190 lower-bound during the closure window, this closure campaign DEFERS to parallel-dispatch first per the May 4 2026 postmortem template. The closure work resumes after the race-mode window closes. Concretely: OP-AUDIT-1 + OP-AUDIT-3 cheap-probe wave RESUMES race-mode-compatible because they are $0 + CPU-only; OP-AUDIT-4 paid Modal dispatch DEFERS.

---

## 2. Predecessor audit + synthesis cross-references

This memo is downstream of TWO authoritative landings (read in order):

1. **Cross-stack synthesis** (`.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` at commit `14c03c57a`; 1449 lines; 149.6 KB)
   - §3 per-landing summary table for the 9 designs
   - §4 9×9 cross-pollination matrix (CONSUMES / ADD / SUB / SAT / ORTHO / EXCL relationships)
   - §5 unified bilevel mathematical framework
   - §8 universal frontier anchor + 6-hook synthesis
   - §9 canonical task queue (10 EV-ranked op-routables)

2. **Wiring-integration orphan audit** (`.omx/research/wiring_integration_orphan_audit_post_12_landings_20260518.md` at commit `b1aae8536`; 841 lines; 68.6 KB)
   - §0 TL;DR with 5 OP-AUDIT op-routables (this memo's input)
   - §4 96-cell wire-in audit table (16 landings × 6 hooks)
   - §5 orphan classification (forward / inverse / 3 UPSTREAM_DEPENDENCY)
   - §6 integration debt inventory (18 PLANNED_BUT_UNROUTED gaps)
   - §13 TOP-5 closure op-routables (this memo's mandate)

This memo extends the audit's 5 OPs into:
- COMPLETE + CORRECT criteria per OP
- closure-completion-verification framework design spec
- 4 NEW + 1 NEW routing directives (5 total NEW directives land this batch)
- TOP-5 closure op-routables ranked by `unblock_count / cost`

---

## 3. Per-OP empirically-verified status table (pre-edit per Catalog #229)

Premise-verification before edit per Catalog #229 — each row's status verified via `ls` + `git log` + `tail` on canonical state surfaces:

### 3.1 OP-AUDIT-1 — master-gradient 6-archive extension

**Premise verified via**:
- `ls -la .omx/research/codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518.md` → exists (commit `699fe19e6`)
- `tail -10 .omx/state/codex_persistent_session_state.jsonl` → 2 most-recent rows show `directive_executed=op_syn_1_dp1_grammar_registry_slice` + `directive_executed=master_gradient_extractor_item_3_phase_b_ruff_scope_a1_pr101_fp64` → Codex IS actively executing this OP
- `grep -c master_gradient .omx/state/canonical_task_status.jsonl` → 3 task rows (OP_SYN_1 ITEM_3 + ITEM_7 etc.)

**Status**: ROUTED + IN_PROGRESS. ~50% complete (3-of-6 grammars have projectors; DP1 detection-only fail-closed; PR106_format0d + PR106_packed + PR107_apogee projectors pending).

**Remaining work per Codex's session state**:
- DP1 projector closure (currently detection-only, fail-closed)
- PR106 format0d projector
- PR106 packed projector
- PR107 Apogee projector
- 6-archive real anchor batch
- extract-all CLI ITEM_3 finalization

### 3.2 OP-AUDIT-2 — 3 NEW 4-layer canonical helpers

**Premise verified via**:
- `ls -la .omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` → exists (745fc2e19; 25.3 KB)
- `ls -la .omx/research/codex_routing_directive_claude_memory_hermetic_export_channel_20260518.md` → exists (a9330927a; 21.8 KB)
- `ls -la .omx/research/codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518.md` → exists (699fe19e6; 15.8 KB)
- `grep -c design_stack_hypergraph .omx/state/canonical_task_status.jsonl` → 0
- `grep -c claude_memory_hermetic .omx/state/canonical_task_status.jsonl` → 0
- `grep -c codex_to_claude_inbox .omx/state/canonical_task_status.jsonl` → 0

**Status**: ROUTED × 3 + NOT_STARTED. Codex has NOT yet ingested any of the 3 directives into canonical_task_status. This is the Assumption-Adversary's concern (§0.5 binding revision #2).

**Remaining work** per each directive's §3 4-layer pattern:
- Layer 1: canonical helper module (~600-700 LOC each)
- Layer 2: CLI tool
- Layer 3: STRICT preflight gate (Catalog #331 inbox / #333 memory / #334 hypergraph)
- Layer 4: operator_briefing.py wire-in + sister-agent consumer wire-ins

### 3.3 OP-AUDIT-3 — 4 Tier-1 design memo canonical helpers

**Premise verified via**:
- `ls -la .omx/research/n_set_venn_classification_design_memo_pair_region_class_frame_axis_20260518.md` → exists (71.9 KB)
- `ls -la .omx/research/phase_1_fisher_precondition_canonical_helper_design_memo_20260518.md` → exists (129.3 KB)
- `ls -la .omx/research/riemannian_newton_substrate_engineering_design_memo_20260518.md` → exists (119.0 KB)
- `ls -la .omx/research/tropical_d_seg_solver_design_memo_20260518.md` → exists (117.4 KB)
- `ls -la .omx/research/codex_routing_directive_canonical_n_set_venn_classification_package_*.md` → does NOT exist
- `ls -la .omx/research/codex_routing_directive_canonical_phase_1_fisher_precondition_package_*.md` → does NOT exist
- `ls -la .omx/research/codex_routing_directive_canonical_riemannian_newton_meta_substrate_package_*.md` → does NOT exist
- `ls -la .omx/research/codex_routing_directive_canonical_tropical_d_seg_solver_package_*.md` → does NOT exist

**Status**: NOT_ROUTED × 4. All 4 design memos exist with full canonical-helper specifications per Catalog #294 9-dim checklist + Catalog #303 cargo-cult audit + Catalog #305 observability + Catalog #296 Dykstra-feasibility predicted-band. THIS MEMO lands the 4 routing directives in the same commit batch (separate files).

**Remaining work** (each directive lands ~250-400 LOC routing spec):
- 3a `codex_routing_directive_canonical_n_set_venn_classification_package_20260518.md` (consumes 3-set Venn design)
- 3b `codex_routing_directive_canonical_phase_1_fisher_precondition_package_20260518.md`
- 3c `codex_routing_directive_canonical_riemannian_newton_meta_substrate_package_20260518.md`
- 3d `codex_routing_directive_canonical_tropical_d_seg_solver_package_20260518.md`

### 3.4 OP-AUDIT-4 — DP1+PR101 Path A canonical helper

**Premise verified via**:
- `ls -la .omx/research/dp1_pr101_composition_design_memo_20260518.md` → exists (116.4 KB)
- `ls -la .omx/research/codex_routing_directive_dp1_pr101_op1_op2_zero_cost_probes_20260518.md` → exists (8.3 KB)
- `head -50 .omx/research/codex_routing_directive_dp1_pr101_op1_op2_zero_cost_probes_20260518.md` → confirms scope is OP-1 + OP-2 $0 probes ONLY (OOD-similarity + architecture-compatibility); does NOT cover full Path A canonical helper

**Status**: PARTIALLY_ROUTED. The $0 probe directive lands the GATING checks before $5-15 Modal A100 smoke. The full Path A canonical helper (Stage 1 DP1 codebook weight init + Stage 2 PR101 HNeRV refinement scaffold) needs a NEW routing directive.

**Remaining work**:
- NEW directive `codex_routing_directive_dp1_pr101_path_a_canonical_helper_package_20260518.md` (lands in this memo's commit batch)
- Codex executes: substrate scaffold at `src/tac/substrates/dp1_pr101_path_a/`; trainer at `experiments/train_substrate_dp1_pr101_composition_path_a.py`; recipe at `.omx/operator_authorize_recipes/substrate_dp1_pr101_composition_path_a_modal_a100_dispatch.yaml`
- Sister T3 per-substrate symposium per Catalog #325 6-step contract
- After symposium PROCEED: Modal A100 50ep smoke ($5-15)

### 3.5 OP-AUDIT-5 — Multi-loop /goal v2.5.2 paste

**Premise verified via**:
- `git log --oneline | grep v2.5.2` → commit `05c74c245` "codex /goal: v2.5.2 aggressively compressed (2825 chars; supersedes v2.5/v2.5.1 which exceeded Codex CLI li...)"
- `ls -la .omx/research/multi_loop_codex_goal_design_memo_20260518.md` → exists (84.9 KB)
- `tail .omx/state/codex_persistent_session_state.jsonl | grep inbox_integration` → **NOT YET FIRED**

**Status**: v2.5.2 LANDED on disk; OPERATOR PASTE ACTION pending. No Codex routing directive needed — operator pastes into Codex CLI /goal context to activate.

**Remaining work**:
- Operator action: paste `codex_persistent_goal_v2_5_2_*.md` content into Codex CLI /goal context
- Validation: next Codex session writes `codex_persistent_session_state.jsonl` row with `directive_executed` field containing one of {`inbox_integration`, `memory_export_integration`, `hypergraph_integration`, `multi_loop_5_coordination`} tokens
- First Codex→Claude inbox channel question + answer roundtrip works

---

## 4. Per-OP COMPLETE criteria

COMPLETE = the OP's primary deliverables exist on disk; tests run; integration hooks land. Distinct from CORRECT (§5) which adds quality gates.

### 4.1 OP-AUDIT-1 COMPLETE criteria

- [ ] **5 NEW archive-grammar parsers** landed in `tools/extract_master_gradient.py` (currently 3 emit real anchors: `fec6_fp11_selector`, `pr101_lc_v2`, `a1_finetuned`; new: `dp1`, `pr106_format0d`, `pr106_packed`, `pr107_apogee`)
- [ ] **Test count ≥ 50** in `src/tac/tests/test_extract_master_gradient.py` (existing 32 + 5 × ~4 new = ~52; verified via `pytest --collect-only`)
- [ ] **CLI `list-grammars`** returns 8 entries (currently 8 grammar authority contracts; verified per Codex session state row `019e3ca6`)
- [ ] **`.omx/state/master_gradient_anchors.jsonl`** contains 6 rows (one per archive); each row carries canonical Provenance per Catalog #323
- [ ] **Memory entry** at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_op_syn_1_master_gradient_six_archive_complete_<utc>.md` lands per CLAUDE.md "Subagent coherence-by-default" mandatory wire-in declaration

### 4.2 OP-AUDIT-2 COMPLETE criteria (per channel)

**Inbox channel** (`tac.codex_to_claude_inbox`):
- [ ] `src/tac/codex_to_claude_inbox.py` (~600 LOC) — canonical helper with `Question`/`Answer` dataclasses + `submit_question`/`fetch_questions`/`submit_answer`/`fetch_answers` + fcntl-locked JSONL persistence at `.omx/state/codex_to_claude_inbox.jsonl`
- [ ] `tools/codex_to_claude_inbox.py` (~150 LOC) — CLI wrapper with `submit-question`/`list-questions`/`submit-answer` subcommands
- [ ] STRICT preflight gate `check_codex_to_claude_inbox_canonical_use` (Catalog #N TBD via `claim_catalog_number.py claim`)
- [ ] Operator briefing wire-in at `tools/operator_briefing.py::_inbox_section` showing pending questions/answers
- [ ] Tests at `src/tac/tests/test_codex_to_claude_inbox.py` (≥30 tests covering schema validation + persistence + fcntl-lock + 4-proc spawn-pool concurrent-append stress + CLI subprocess)
- [ ] Memory entry at `feedback_codex_to_claude_inbox_canonical_helper_landed_<utc>.md`

**Memory export channel** (`tac.claude_memory_hermetic_export`):
- [ ] `src/tac/claude_memory_hermetic_export.py` (~600 LOC) — canonical helper with `export_memory_to_dated_research_memo` / `import_memo_into_repo_local_state` + fcntl-locked write per Catalog #131
- [ ] `tools/claude_memory_hermetic_export.py` CLI
- [ ] STRICT preflight gate (Catalog #N TBD)
- [ ] Tests (≥25 tests)
- [ ] Memory entry

**Hypergraph channel** (`tac.design_stack_hypergraph`):
- [ ] `src/tac/design_stack_hypergraph.py` (~700 LOC) — canonical helper with 10 typed nodes + 7 typed edges + 3 hyperedge types per `design_stack_full_hypergraph_model_design_memo_20260518.md`
- [ ] `tools/design_stack_hypergraph.py` CLI with `query-node`/`query-edge`/`render-graph` subcommands
- [ ] STRICT preflight gate (Catalog #N TBD)
- [ ] Tests (≥35 tests covering schema + 10 node types + 7 edge types + 3 hyperedge types + GraphML/JSON export + cycle detection)
- [ ] Memory entry

### 4.3 OP-AUDIT-3 COMPLETE criteria (per Tier-1 helper)

Each helper follows the canonical 4-layer pattern per Catalog #245 sister directives. The 4 helpers + per-helper criteria:

**VENN (3-set Venn classification)** — `tac.canonical_n_set_venn_classification`:
- [ ] Layer 1: package at `src/tac/canonical_n_set_venn_classification/` with `Region` dataclass + 3-set venn primitive + `classify_pair_region_class` + cell density computation per design memo §5
- [ ] Layer 2: CLI `tools/canonical_n_set_venn_classification_cli.py`
- [ ] Layer 3: STRICT preflight gate (claim Catalog #N transactionally)
- [ ] Layer 4: operator_briefing wire-in + autopilot consumer at `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_n_set_venn_class_v3` (extends v2 cascade)
- [ ] Tests (≥40 tests per design memo §10 9-dim checklist)
- [ ] Memory entry

**FISHER (Phase 1 Fisher-precondition)** — `tac.phase_1_fisher_precondition`:
- [ ] Layer 1: package with `compute_fisher_diagonal` + `fisher_orthogonal_projection` + canonical verdict taxonomy `{validated_well_conditioned, validated_near_singular_requires_kfac, invalid_input}`
- [ ] Layer 2: CLI
- [ ] Layer 3: STRICT preflight gate
- [ ] Layer 4: autopilot consumer at `adjust_predicted_delta_for_fisher_conditioning_verdict` + sensitivity hook
- [ ] Posterior at `.omx/state/fisher_conditioning_anchors.jsonl` fcntl-locked per Catalog #131
- [ ] Tests (≥45 tests)
- [ ] Memory entry

**RIEM (Riemannian-Newton META)** — `tac.riemannian_newton_substrate_engineering`:
- [ ] Layer 1: package with Stiefel-manifold primitives + symplectic-EMA flow + Boumal-Absil-Mahony trust-region per design memo §6
- [ ] Layer 2: CLI
- [ ] Layer 3: STRICT preflight gate
- [ ] Layer 4: META-substrate inheritance pattern wired into `tac.substrates._shared.trainer_skeleton`
- [ ] Posterior at `.omx/state/riemannian_newton_convergence.jsonl`
- [ ] Tests (≥50 tests; geomstats integration)
- [ ] Memory entry

**TROP (Tropical d_seg solver)** — `tac.tropical_d_seg_solver`:
- [ ] Layer 1: package with tropical polynomial primitives + Mallat wavelet-multiscale boundary detector + Phase 1 (boundary) / Phase 2 (faithfulness probe) / Phase 3 (composition)
- [ ] Layer 2: CLI
- [ ] Layer 3: STRICT preflight gate
- [ ] Layer 4: autopilot consumer at `adjust_predicted_delta_for_tropical_phase_eligibility` + sensitivity hook
- [ ] Posterior at `.omx/state/tropical_d_seg_anchors.jsonl`
- [ ] Tests (≥40 tests)
- [ ] Memory entry

### 4.4 OP-AUDIT-4 COMPLETE criteria

- [ ] **Probe directive cleared** — `codex_routing_directive_dp1_pr101_op1_op2_zero_cost_probes_20260518.md` OP-1 + OP-2 results land in `.omx/state/probe_outcomes.jsonl` via `tac.probe_outcomes_ledger.register_probe_outcome` (per Catalog #313)
- [ ] **OP-1 OOD-similarity probe** runnable at `tools/probe_dp1_ood_similarity_vs_contest_video.py`; emits verdict {OOD_SIMILAR_ENOUGH, OOD_TOO_FAR, OOD_INDETERMINATE}
- [ ] **OP-2 architecture-compatibility probe** runnable at `tools/probe_dp1_pr101_architecture_compatibility.py`; emits verdict {LOADABLE, INCOMPATIBLE_SHAPE_MISMATCH, INCOMPATIBLE_DTYPE_MISMATCH}
- [ ] **Path A canonical helper** at `src/tac/substrates/dp1_pr101_path_a/__init__.py` + sister modules: `architecture.py` (DP1-warm-start HNeRV), `score_aware_loss.py` (canonical scorer routing per Catalog #164), `archive.py` (substrate archive grammar)
- [ ] **Trainer** at `experiments/train_substrate_dp1_pr101_composition_path_a.py` with canonical 9-dim checklist evidence in design memo header + Catalog #303 cargo-cult audit
- [ ] **Recipe** at `.omx/operator_authorize_recipes/substrate_dp1_pr101_composition_path_a_modal_a100_dispatch.yaml` with `dispatch_enabled: false` initially (research_only=true until per-substrate symposium PROCEED per Catalog #325)
- [ ] **Per-substrate symposium memo** at `.omx/research/council_t3_per_substrate_symposium_dp1_pr101_composition_path_a_20260518.md` per Catalog #325 6-step contract
- [ ] **Stage-2 Modal A100 50ep smoke** ready (operator-gated; `dispatch_enabled: true` flip after symposium PROCEED)
- [ ] Memory entry

### 4.5 OP-AUDIT-5 COMPLETE criteria

- [ ] **Operator action** — paste v2.5.2 content into Codex CLI /goal context (one-time; takes ~10 min)
- [ ] **Codex confirmation** — `tail -3 .omx/state/codex_persistent_session_state.jsonl` shows a row with `directive_executed` field containing one of: `inbox_integration`, `memory_export_integration`, `hypergraph_integration`, `multi_loop_5_coordination`, `goal_v2_5_2_activated`
- [ ] **First inbox question + answer roundtrip** works (verified via `tools/codex_to_claude_inbox.py list-questions` + `submit-answer` — gated on OP-AUDIT-2 inbox channel COMPLETE)

---

## 5. Per-OP CORRECT criteria

CORRECT = all tests green, all STRICT preflight gates pass, all 6 hooks declared per Catalog #125 (WIRED or N/A_WITH_RATIONALE; NO PLANNED_BUT_UNROUTED per audit's anti-pattern), empirical anchor lands as expected.

### 5.1 OP-AUDIT-1 CORRECT criteria

- [ ] **All 52+ tests in `test_extract_master_gradient.py` pass green** (no skipped tests)
- [ ] **Ruff F821 isolated CI gate passes** for `tools/extract_master_gradient.py` per Catalog `_isolated_f821_policy` (recent Codex landing)
- [ ] **Preflight Catalog #270 dispatch protocol** passes for the tool dispatch (per CLAUDE.md "Production-hardened dispatch optimization protocol" scope clarification — tool dispatches are categorically not subject to substrate-only Tier 1 primitives)
- [ ] **6-hook wire-in declaration in memory entry**: hook #1 sensitivity-map = ACTIVE (per-archive fp64 sensitivity tensor) / hook #2 Pareto = ACTIVE (per-archive Pareto-feasibility) / hook #3 bit-allocator = ACTIVE (per-byte sensitivity) / hook #4 cathedral autopilot = ACTIVE via `tac.master_gradient_consumers` / hook #5 continual-learning = ACTIVE (anchor ledger append-only) / hook #6 probe-disambiguator = ACTIVE (per-archive grammar fail-closed disambiguator)
- [ ] **Empirical anchor**: 6 archives × 1 anchor per archive = 6 anchors land in `.omx/state/master_gradient_anchors.jsonl` per CLAUDE.md "no signal loss" discipline; each row tagged `[macOS-CPU advisory]` for diagnostic anchors OR `[contest-CPU]` after paired Linux x86_64 anchor per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"

### 5.2 OP-AUDIT-2 CORRECT criteria (per channel)

- [ ] **All ≥90 tests across 3 channels green** (inbox ≥30 + memory ≥25 + hypergraph ≥35)
- [ ] **3 new STRICT preflight gates wired strict=True** in `preflight_all()` with live count = 0 at landing per CLAUDE.md "Strict-flip atomicity rule"
- [ ] **Catalog #176 META-meta gate clean** (each strict callsite has a CLAUDE.md catalog row)
- [ ] **Catalog #185 META-meta-meta gate clean** (each "Live count: 0" claim matches empirical gate state)
- [ ] **Per-channel 6-hook wire-in declarations** per Catalog #125 in 3 separate landing memos
- [ ] **Inbox channel roundtrip empirical**: 1 question + 1 answer roundtrip succeeds end-to-end via CLI (no manual file edits)
- [ ] **Memory export channel empirical**: 1 memory entry exports to dated research memo + imports back into repo-local state without provenance loss
- [ ] **Hypergraph empirical**: 9 design memos parse into hypergraph; all 10 node types + 7 edge types + 3 hyperedge types instantiate without schema violations

### 5.3 OP-AUDIT-3 CORRECT criteria (per Tier-1 helper)

For EACH of VENN / FISHER / RIEM / TROP:

- [ ] **All design-memo-specified tests green** (per Catalog #294 9-dim checklist test requirements)
- [ ] **Canonical-vs-unique decision per layer satisfied** per Catalog #290 — each layer's decision documented in landing memo
- [ ] **Cargo-cult audit section** per Catalog #303 — each assumption classified HARD-EARNED or CARGO-CULTED with unwind path
- [ ] **Observability surface declaration** per Catalog #305 — all 6 facets satisfied
- [ ] **Dykstra-feasibility predicted-band check** per Catalog #296 — predicted ΔS band cited with Dykstra feasibility OR first-principles citation
- [ ] **Per-design-memo predicted band** validated post-empirical per Catalog #324 (if recipe declares predicted_band)
- [ ] **STRICT preflight gate** wired strict=True with live count = 0 per Strict-flip atomicity rule
- [ ] **Operator-facing audit tool** runnable at `tools/audit_<helper>_compliance.py` per design memo's observability requirement
- [ ] **6-hook wire-in declaration** in landing memo per Catalog #125
- [ ] **Empirical anchor**: ΔS measured per OP-AUDIT-1's 6-archive matrix (paired-comparison smoke per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" canonical-vs-unique decision falling-rule)

### 5.4 OP-AUDIT-4 CORRECT criteria

- [ ] **OP-1 + OP-2 $0 probe verdicts** land in `.omx/state/probe_outcomes.jsonl` per Catalog #313 with `verdict in {LOADABLE, OOD_SIMILAR_ENOUGH}` (proceed condition) OR `verdict in {INCOMPATIBLE_*, OOD_TOO_FAR}` (block dispatch + emit research note)
- [ ] **Per-substrate symposium** lands with verdict `PROCEED` or `PROCEED_WITH_REVISIONS` per Catalog #325 6-step contract — NOT `DEFER_PENDING_EVIDENCE` (which blocks dispatch)
- [ ] **Catalog #240 recipe-vs-trainer-state** passes (recipe `dispatch_enabled: true` only after symposium PROCEED + trainer `_full_main` implements full Path A flow without `NotImplementedError`)
- [ ] **Catalog #272 distinguishing-feature integration contract** declares all 4 fields (`distinguishing_feature_name="dp1_warm_start_codebook"` + `distinguishing_bytes_path=...` + `inflate_consumer_function=...` + `byte_mutation_smoke_passes=True/False`)
- [ ] **Catalog #324 post-training Tier-C validation** — predicted band [0.180, 0.190] validated against landed archive post-Stage-2-smoke
- [ ] **Catalog #325 14-day symposium freshness** maintained — first symposium dated within 14d of first Stage-2 dispatch
- [ ] **Stage-2 Modal A100 50ep smoke** lands `[contest-CUDA T4]` anchor (paired with `[contest-CPU]` after promotion per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA")
- [ ] **6-hook wire-in declaration** per Catalog #125
- [ ] **Memory entry** per CLAUDE.md "Subagent coherence-by-default"

### 5.5 OP-AUDIT-5 CORRECT criteria

- [ ] **Codex session state row** lands within 24h of operator paste action containing `directive_executed` field with one of the canonical v2.5.2 tokens
- [ ] **First Codex→Claude inbox roundtrip** completes end-to-end (gated on OP-AUDIT-2 inbox channel CORRECT)
- [ ] **Multi-loop coordination** — Codex's 5 loops (read / plan / execute / verify / report) all fire at least once per 24h (verified via canonical_task_status row frequency)
- [ ] **No silent regressions** — operator-attention budget per Catalog #300 not exceeded (T1 unbounded / T2 ≤3/day / T3 ≤3/week / T4 ≤2/30d)

---

## 6. Closure-completion-verification framework design

The framework is canonical helper `tac.closure_completion_verifier`. This memo DESIGNS it at the API spec level. The 4-layer pattern build is routed to Codex via OPR-CLOSE-5.

### 6.1 4-layer pattern per Catalog #245 exemplar

**Layer 1** — canonical helper at `src/tac/closure_completion_verifier.py` (~500 LOC):
- `ClosureOperationVerdict` enum (9 values per §6.2)
- `verify_closure_operation(op_id: str) -> ClosureOperationVerdict`
- `verify_all_closure_operations() -> list[tuple[str, ClosureOperationVerdict]]`
- `closure_campaign_summary(*, format: str = "text") -> str`
- `register_closure_alert(op_id, verdict, blocker_reason) -> None` (fcntl-locked JSONL append to `.omx/state/closure_completion_anchors.jsonl` per Catalog #131)

**Layer 2** — CLI at `tools/closure_completion_verifier.py` (~150 LOC):
- `verify <op_id>` → prints verdict + blockers
- `verify-all` → prints table of all 5 OPs
- `summary` → human-readable summary
- `summary --json` → machine-readable for autopilot consumer
- `nightly-cron` → runs verify-all + emits alerts on INCORRECT_NEEDS_FIX / BLOCKED stuck >7 days

**Layer 3** — STRICT preflight gate `check_closure_completion_verifier_canonical_use` (claim Catalog #N transactionally; refuses bypass invocations outside the canonical CLI + module API)

**Layer 4** — wire-in:
- `tools/operator_briefing.py::_closure_campaign_section` shows per-OP verdict + last-checked timestamp + blockers
- `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_closure_state` — autopilot ranker can deprioritize dispatch candidates whose dependencies are BLOCKED in closure verifier
- nightly cron entry in multi-loop /goal v2.5.2 `nightly-cron` loop (per multi-loop design memo §3)

### 6.2 ClosureOperationVerdict enum (9 values)

```python
from enum import Enum

class ClosureOperationVerdict(Enum):
    """Per-OP closure-completion verdict per Catalog #N (TBD; this memo's design spec).

    Distinguishes COMPLETE from CORRECT separately per Contrarian binding revision (§0.5):
    a half-shipped helper (tests pending) is NOT reported as fully closed.
    """

    NOT_STARTED = "not_started"
    """No routing directive landed; OP exists only in this campaign memo's §3 table."""

    ROUTING_DIRECTIVE_LANDED = "routing_directive_landed"
    """Codex has the routing directive on disk; canonical_task_status rows
    NOT yet emitted; no execution started."""

    IN_PROGRESS = "in_progress"
    """Codex is actively executing per canonical_task_status.jsonl rows with
    status=in_progress; some sub-tasks complete; others pending."""

    COMPLETE_PENDING_TESTS = "complete_pending_tests"
    """Implementation landed (per §4 COMPLETE criteria); tests not yet green
    OR tests skipped OR test count below threshold."""

    COMPLETE_TESTS_GREEN = "complete_tests_green"
    """Implementation + tests green; integration hooks NOT yet wired
    (e.g. operator_briefing wire-in pending; autopilot consumer pending)."""

    COMPLETE_INTEGRATED = "complete_integrated"
    """Implementation + tests + ALL integration hooks wired per §5 CORRECT criteria.
    This is the canonical TERMINAL CORRECT verdict."""

    INCORRECT_NEEDS_FIX = "incorrect_needs_fix"
    """Implementation exists BUT tests red OR STRICT preflight gate failing
    OR integration broken. Operator-visible alert fires."""

    BLOCKED = "blocked"
    """Waiting on upstream dependency (per Catalog #313 probe outcomes ledger
    or sister OP). E.g. OP-AUDIT-3 BLOCKED by OP-AUDIT-1; OP-AUDIT-5 BLOCKED by OP-AUDIT-2 inbox channel."""

    OPERATOR_ACTION_REQUIRED = "operator_action_required"
    """Closure requires explicit operator action that cannot be auto-executed
    (e.g. OP-AUDIT-5 v2.5.2 paste). Distinct from BLOCKED (upstream dependency)
    because no further Codex/Claude work can advance this OP."""
```

### 6.3 Per-OP verification logic (concrete classifier functions)

Each `_verify_<op_id>()` function in `tac.closure_completion_verifier` follows this structure:

```python
def _verify_op_audit_1() -> ClosureOperationVerdict:
    """OP-AUDIT-1 master-gradient 6-archive extension."""
    # 1. Check routing directive landed
    directive = Path(".omx/research/codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518.md")
    if not directive.exists():
        return ClosureOperationVerdict.NOT_STARTED

    # 2. Check canonical_task_status has rows
    task_rows = _query_canonical_task_status(directive.stem)
    if not task_rows:
        return ClosureOperationVerdict.ROUTING_DIRECTIVE_LANDED

    # 3. Check anchor count (6 expected; currently 3)
    anchors = _count_master_gradient_anchors()
    if anchors < 6:
        # Check if Codex is actively working
        latest_codex_state = _latest_codex_persistent_session_state()
        if latest_codex_state and "master_gradient" in latest_codex_state.directive_executed:
            return ClosureOperationVerdict.IN_PROGRESS
        # Stalled
        return ClosureOperationVerdict.BLOCKED

    # 4. Check tests pass
    if not _pytest_passes("src/tac/tests/test_extract_master_gradient.py"):
        return ClosureOperationVerdict.COMPLETE_PENDING_TESTS

    # 5. Check 6-hook wire-in declaration
    memory_entry = _find_memory_entry_for_op("OP-AUDIT-1")
    if not memory_entry or not _has_6_hook_declaration(memory_entry):
        return ClosureOperationVerdict.COMPLETE_TESTS_GREEN

    # All criteria met
    return ClosureOperationVerdict.COMPLETE_INTEGRATED
```

Similar `_verify_op_audit_<N>()` for OP-AUDIT-2/3/4/5. Each verifier:
- Reads canonical state surfaces (filesystem + git + JSONL ledgers) — NO mocking
- Returns deterministic verdict per per-OP COMPLETE + CORRECT criteria from §4 + §5
- Emits structured advisory blockers when verdict is INCORRECT_NEEDS_FIX or BLOCKED

### 6.4 Nightly cron monitor (per multi-loop /goal v2.5.2)

The nightly-cron loop per `.omx/research/multi_loop_codex_goal_design_memo_20260518.md` §3 includes a closure-verifier invocation:

```bash
# multi-loop nightly cron entry
.venv/bin/python tools/closure_completion_verifier.py nightly-cron
```

The CLI:
1. Runs `verify_all_closure_operations()` returning list of (op_id, verdict) tuples
2. For each OP with verdict `INCORRECT_NEEDS_FIX` OR `BLOCKED` for >7 days OR `OPERATOR_ACTION_REQUIRED` for >7 days: emits operator-visible alert
3. Appends canonical posterior anchor to `.omx/state/closure_completion_anchors.jsonl` per Catalog #131 fcntl-locked
4. Writes daily summary to `.omx/research/closure_campaign_daily_summary_<YYYY-MM-DD>.md`
5. If inbox channel COMPLETE_INTEGRATED: emits inbox question via `tac.codex_to_claude_inbox.submit_question` for any INCORRECT_NEEDS_FIX OP routing closure question to Codex

### 6.5 Operator-facing dashboard

`tools/closure_campaign_dashboard.py` reads `.omx/state/closure_completion_anchors.jsonl` + latest filesystem state and emits human-readable dashboard:

```
=== Closure Campaign Dashboard (2026-05-18 20:45 UTC) ===

OP-AUDIT-1: IN_PROGRESS (last checked: 2026-05-18T20:45:00Z)
  Routing: ROUTED 2026-05-18T18:00:00Z (commit 699fe19e6)
  Codex: actively executing (4 task rows; ITEM_3 in_progress)
  Progress: 3-of-6 archives have projectors
  Blockers: PR106 format0d projector pending; PR106 packed projector pending; PR107 Apogee projector pending; extract-all batch CLI pending

OP-AUDIT-2 (3 channels): ROUTING_DIRECTIVE_LANDED (last checked: 2026-05-18T20:45:00Z)
  Inbox: directive 745fc2e19 landed; canonical_task_status rows = 0
  Memory: directive a9330927a landed; canonical_task_status rows = 0
  Hypergraph: directive 699fe19e6 landed; canonical_task_status rows = 0
  Alert: stuck in ROUTING_DIRECTIVE_LANDED for 0 days (threshold 7d)

OP-AUDIT-3 (4 Tier-1 helpers): NOT_STARTED (last checked: 2026-05-18T20:45:00Z)
  Status: 4 design memos landed; 4 routing directives PENDING (this memo's commit batch closes this)
  Alert: NO ALERT (this memo is the routing-directive landing event)

OP-AUDIT-4: ROUTING_DIRECTIVE_LANDED (partial: OP-1+OP-2 only; full Path A directive PENDING in this commit batch)
  Status: 1-of-2 directives landed
  Alert: NO ALERT (this memo closes the routing gap)

OP-AUDIT-5: OPERATOR_ACTION_REQUIRED (last checked: 2026-05-18T20:45:00Z)
  Status: v2.5.2 landed (commit 05c74c245); operator paste pending
  Alert: stuck in OPERATOR_ACTION_REQUIRED for 0 days

=== Summary ===

Total OPs: 5
COMPLETE_INTEGRATED: 0
IN_PROGRESS: 1
ROUTING_DIRECTIVE_LANDED: 4 (after this commit batch lands the 5 NEW directives)
INCORRECT_NEEDS_FIX: 0
BLOCKED: 0
OPERATOR_ACTION_REQUIRED: 1
```

### 6.6 Inbox channel auto-question on INCORRECT_NEEDS_FIX

When the closure verifier detects an OP transitioning to INCORRECT_NEEDS_FIX (e.g. OP-AUDIT-2 inbox channel tests fail), it auto-submits an inbox question to Codex per the canonical Codex→Claude inbox channel (gated on OP-AUDIT-2 inbox channel COMPLETE_INTEGRATED):

```python
from tac.codex_to_claude_inbox import submit_question

submit_question(
    question_id=f"closure_verifier_op_audit_2_inbox_tests_red_{utc_now()}",
    sender="closure_completion_verifier",
    recipient="codex",
    subject="OP-AUDIT-2 inbox channel: 2/30 tests red",
    body=(
        "The closure verifier detected 2 red tests in "
        "`src/tac/tests/test_codex_to_claude_inbox.py`. "
        "Specific failures: "
        "test_codex_to_claude_inbox_4_proc_spawn_pool_concurrent_append_safety, "
        "test_codex_to_claude_inbox_strict_load_corrupt_raises. "
        "Please fix per CLAUDE.md 'Bugs must be permanently fixed AND self-protected against'."
    ),
    severity="HIGH",
    blocking=True,
)
```

Codex picks up the question via `tools/codex_to_claude_inbox.py list-questions` in its next session per multi-loop /goal v2.5.2.

---

## 7. Routing-directive gap inventory

### 7.1 Landed routing directives (verified pre-edit)

| Directive | Path | Covers OP | Commit |
|---|---|---|---|
| 1 | `codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518.md` | OP-AUDIT-1 | `699fe19e6` |
| 2 | `codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` | OP-AUDIT-2 (inbox) | `745fc2e19` |
| 3 | `codex_routing_directive_claude_memory_hermetic_export_channel_20260518.md` | OP-AUDIT-2 (memory) | `a9330927a` |
| 4 | `codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518.md` | OP-AUDIT-2 (hypergraph) | `699fe19e6` |
| 5 | `codex_routing_directive_dp1_pr101_op1_op2_zero_cost_probes_20260518.md` | OP-AUDIT-4 (probes only) | (this is the cheap-probe wave directive) |

### 7.2 Routing directives landed in THIS commit batch (5 NEW)

| Directive | Path | Covers OP | Estimated LOC |
|---|---|---|---|
| 6 | `codex_routing_directive_canonical_n_set_venn_classification_package_20260518.md` | OP-AUDIT-3 VENN | ~300 |
| 7 | `codex_routing_directive_canonical_phase_1_fisher_precondition_package_20260518.md` | OP-AUDIT-3 FISHER | ~350 |
| 8 | `codex_routing_directive_canonical_riemannian_newton_meta_substrate_package_20260518.md` | OP-AUDIT-3 RIEM | ~400 |
| 9 | `codex_routing_directive_canonical_tropical_d_seg_solver_package_20260518.md` | OP-AUDIT-3 TROP | ~350 |
| 10 | `codex_routing_directive_dp1_pr101_path_a_canonical_helper_package_20260518.md` | OP-AUDIT-4 (full helper) | ~350 |
| MASTER | `closure_campaign_pursue_and_confirm_master_20260518.md` (THIS MEMO) | OP-AUDIT-1-5 coordination | ~2000 |

### 7.3 Operator action required (NOT routing — direct paste)

- OP-AUDIT-5 v2.5.2 paste — content at `.omx/research/multi_loop_codex_goal_design_memo_20260518.md` §V2.5.2 OR the standalone v2.5.2 source per commit `05c74c245`

---

## 8. Cargo-cult audit per Catalog #303

Per Catalog #303 each shared assumption surfaced + HARD-EARNED-vs-CARGO-CULTED classification per the hard-earned-vs-cargo-culted addendum.

### 8.1 Assumption — "Routing-directive landings close OPs without further verification"

**Classification**: CARGO-CULTED

**Source of cargo-cult**: routing directives are FIRE-AND-FORGET. Routing directive → Codex picks up in autonomous loop → canonical helper lands → end. This pattern WORKS for simple OPs (single helper module + tests). It FAILS for multi-layer OPs because there's no structural verification that ALL layers landed.

**Empirical evidence**: OP-SYN-1 in audit §9.3 showed canonical_task_status entries went through pending → in_progress → ITEM_9 completed. But NONE of the 7 routing directives' own helpers have explicit "landing event" rows. The audit had to use `ls -la src/tac/<package>/` to verify existence.

**Unwind plan**: this memo + §6 closure-completion-verifier IS the unwind. Verify each OP's COMPLETE + CORRECT criteria empirically via filesystem + ledger queries; emit nightly alerts when stuck >7 days; auto-submit inbox questions when INCORRECT_NEEDS_FIX.

### 8.2 Assumption — "PLANNED_BUT_UNROUTED gaps will close in canonical-task-status timeline"

**Classification**: CARGO-CULTED-PENDING-EMPIRICAL

**Source**: routing directives declare a 4-layer pattern with Layer 4 wire-in to operator_briefing + autopilot. The assumption is Codex's autonomous loop closes Layer 4 wire-in automatically. Empirical evidence (audit §9.3) shows this is partially true (some Layer 4 wire-ins land) and partially cargo-cult (some land Layer 1-3 but not Layer 4).

**Unwind plan**: per-OP CORRECT criteria (§5) require Layer 4 wire-in declaration in landing memo. Closure verifier checks for declaration; if missing, returns COMPLETE_TESTS_GREEN (not COMPLETE_INTEGRATED) — operator-visible distinction.

### 8.3 Assumption — "Codex's autonomous loop will execute routing directives in EV-order"

**Classification**: HARD-EARNED-PARTIAL

**Source**: Codex's persistent /goal v2.4 → v2.5.2 cycle reads routing directives by date suffix. EV-ordering is INSIDE each routing directive (the operator-routable list at the directive's bottom). Codex executes in directive-landing order; within a directive, EV-order may be respected.

**Empirical evidence**: today's Codex session executed OP-SYN-1 ITEM_3 (master gradient phase B) then ITEM_7 (pose-axis selector) — these are HIGH-EV within their directives.

**Unwind plan**: closure verifier's nightly alert + inbox auto-question MECHANISM lets us see when Codex's autonomous loop deviates from EV-order without operator escalation.

### 8.4 Assumption — "5 OPs are independent and can close in parallel"

**Classification**: CARGO-CULTED (refuted by audit §0.5 binding revision #3 + §12.2 dependency graph)

**Empirical evidence**: OP-AUDIT-1 unblocks ALL 9 Tier-1 designs → OP-AUDIT-3 SEQUENCES after OP-AUDIT-1; OP-AUDIT-5 requires OP-AUDIT-2 inbox channel COMPLETE_INTEGRATED → OP-AUDIT-5 SEQUENCES after OP-AUDIT-2 inbox layer 1-4 complete.

**Unwind plan**: §11.2 dependency graph + sequencing-aware closure verifier (BLOCKED verdict when upstream OP not yet COMPLETE_INTEGRATED).

---

## 9. 9-dimension success checklist evidence per Catalog #294

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS (class-shift not within-class) | First closure-campaign coordinator memo; NOT a re-execution of audit (which surfaced OPs); NOT a re-execution of synthesis (which produced 9-design cross-pollination). NEW dimensions: closure-completion-verifier framework + 9-verdict taxonomy + nightly cron monitor + inbox auto-question. |
| 2 | BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | TL;DR §0 single-page summary; verdict matrix §0.4 single table; 5-OP status §3 single table. Whole memo navigable via TOC. |
| 3 | DISTINCTNESS (explicitly different from sisters) | Distinct from audit (which surfaces OPs) — this memo CLOSES OPs. Distinct from synthesis (which integrates 9 designs) — this memo coordinates the 5 closure work-streams. Distinct from each individual routing directive (which routes 1 OP) — this memo routes 5 OPs + designs verification. |
| 4 | RIGOR (premise verification + adversarial review + assumption classification + empirical anchor) | Per Catalog #229: 5 premises verified pre-edit via filesystem + git + ledger queries (§3); Per Catalog #303: 4 assumptions classified in §8; Per Catalog #292: council deliberation per §0.4 verdict matrix |
| 5 | OPTIMIZATION PER TECHNIQUE | N/A (this is an apparatus_maintenance campaign coordinator; no per-technique optimization claim) |
| 6 | STACK-OF-STACKS-COMPOSABILITY | Closure verifier composable across future campaign coordinator memos (each new campaign batch can register OPs in same `.omx/state/closure_completion_anchors.jsonl`); 5 OPs themselves compose per §11.2 dependency graph |
| 7 | DETERMINISTIC REPRODUCIBILITY | Memo deterministically reproducible from filesystem state at landing time (2026-05-18T20:45:00Z initial checkpoint); routing directives + master memo land in single commit batch via canonical serializer with POST-EDIT shas per Catalog #157 |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | $0 GPU cost; ~3h editor time; canonical reusability is the closure-verifier framework spec at §6 which extends to future campaign coordinators with minimal modification |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | N/A (this is apparatus_maintenance; downstream contribution to score via OP-AUDIT-1-5 closure work feeds Tier-1 + Tier-2 + Tier-3 cascade per synthesis §9) |

---

## 10. Observability surface per Catalog #305

The closure verifier IS an observability surface; meta-observability per Catalog #305.

| Facet | Evidence |
|---|---|
| Inspectable per layer | §6.3 per-OP verifier logic is per-criterion cell-by-cell; each criterion independently inspectable |
| Decomposable per signal | §5 CORRECT criteria are per-OP × per-criterion matrix; each cell decomposable (e.g. test count / preflight gate / 6-hook declaration / empirical anchor) |
| Diff-able across runs | Nightly cron writes JSONL anchor per Catalog #131; future audits can diff yesterday's verdicts against today's to see which OPs flipped INCORRECT_NEEDS_FIX → COMPLETE_INTEGRATED |
| Queryable post-hoc | `tools/closure_completion_verifier.py summary --json` emits machine-readable verdict per OP; consumable by autopilot ranker + dashboard + operator briefing |
| Cite-able | Each verdict cites the canonical state surfaces (filesystem paths + JSONL row counts + git commit shas + canonical_task_status entries) per the verifier's `_verify_<op>()` implementation |
| Counterfactual-able | "what if OP-AUDIT-2 inbox channel landed?" → flips verdict from ROUTING_DIRECTIVE_LANDED to COMPLETE_INTEGRATED + unblocks OP-AUDIT-5 from OPERATOR_ACTION_REQUIRED to PROCEEDABLE |

---

## 11. TOP-5 op-routables ranked by EV

### 11.1 EV ranking methodology

Per Catalog #319 v2 cascade EV formula adapted for apparatus_maintenance: `EV = unblock_count / cost_envelope_upper_bound_usd_or_editor_days`. Tie-break by orchestration sequencing per §11.2.

### 11.2 TOP-5 ranked (op-routable closure list)

| Rank | OP-routable | Action | Concrete file paths | Cost envelope | Dependencies | Unblock count | EV |
|---|---|---|---|---|---|---|---|
| **1** | **OPR-CLOSE-1** | Codex execute OP-AUDIT-1 (extend `tools/extract_master_gradient.py` with 5 NEW projectors: DP1 + PR106_format0d + PR106_packed + PR107_apogee) + extract-all batch CLI | `tools/extract_master_gradient.py`; `src/tac/master_gradient.py`; `.omx/state/master_gradient_anchors.jsonl` (6 rows after) | $0 GPU + ~6-12h M5 Max CPU | NONE (directive landed; Codex actively in flight per session state row `019e3ca6`) | ALL 9 Tier-1 designs + OP-AUDIT-3 (FISHER/RIEM/VENN/TROP) | ∞ |
| **2** | **OPR-CLOSE-2** | Land 4 NEW routing directives for OP-AUDIT-3 in this memo's commit batch | `codex_routing_directive_canonical_n_set_venn_classification_package_20260518.md`; `codex_routing_directive_canonical_phase_1_fisher_precondition_package_20260518.md`; `codex_routing_directive_canonical_riemannian_newton_meta_substrate_package_20260518.md`; `codex_routing_directive_canonical_tropical_d_seg_solver_package_20260518.md` | $0 + ~2h editor | NONE | 4 Tier-1 canonical helpers (~1750 LOC across the 4 packages) | high (4 unblocks / 2h) |
| **3** | **OPR-CLOSE-3** | Codex execute OP-AUDIT-2 (3 channels: inbox + memory-export + hypergraph) per directives 10/13/14 | `src/tac/codex_to_claude_inbox.py`; `src/tac/claude_memory_hermetic_export.py`; `src/tac/design_stack_hypergraph.py`; 3 CLI tools + 3 STRICT gates | $0 + ~6 days editor | NONE (directives landed; Codex execution pending) | 18 PLANNED_BUT_UNROUTED gaps (3 channels + 15 downstream observability consumers) + unblocks OP-AUDIT-5 inbox roundtrip | `[3, 1]` per day |
| **4** | **OPR-CLOSE-4** | Land 1 NEW routing directive for OP-AUDIT-4 (DP1+PR101 Path A full canonical helper) in this memo's commit batch | `codex_routing_directive_dp1_pr101_path_a_canonical_helper_package_20260518.md` | $0 + ~1h editor | OP-AUDIT-4 OP-1+OP-2 $0 probe directive already landed | DP1+PR101 Stage 2 Modal A100 smoke ($5-15) | high (1 unblock + paid empirical anchor / 1h) |
| **5** | **OPR-CLOSE-5** | Build `tac.closure_completion_verifier` canonical helper (4-layer pattern) | `src/tac/closure_completion_verifier.py` (~500 LOC); `tools/closure_completion_verifier.py` (~150 LOC); STRICT Catalog #N gate; `tools/operator_briefing.py::_closure_campaign_section` wire-in; nightly cron entry in multi-loop /goal v2.5.2 | $0 + ~1 day editor | NONE (this memo's §6 design spec is sufficient input) | makes ALL 5 OPs structurally queryable across sessions; extincts "closure work invisible to operator" bug class | `[5, 1]` per day |

### 11.3 Operational sequencing

```
Week 1 — Routing directive landings + Codex execution
  OPR-CLOSE-2 (4 Tier-1 directives) → land same commit batch as master memo
  OPR-CLOSE-4 (1 Path A directive) → land same commit batch as master memo
  OPR-CLOSE-1 (OP-AUDIT-1) → Codex continues active execution

Week 1-2 — Codex execution + verifier build
  OPR-CLOSE-3 (OP-AUDIT-2 inbox / memory / hypergraph) → Codex picks up
  OPR-CLOSE-5 (closure verifier build) → routed via NEW directive in followon batch

Week 2-3 — OP-AUDIT-3 Tier-1 helpers cascade
  Phase 1 FISHER (depends on OP-AUDIT-1 anchor) → Codex builds
  3-set VENN (parallel) → Codex builds

Week 3-4 — OP-AUDIT-3 Tier-1 helpers cascade (cont)
  RIEMANNIAN-NEWTON META (depends on Phase 1 FISHER) → Codex builds
  TROPICAL Phase 1 (depends on VENN) → Codex builds

Week 4 — OP-AUDIT-4 + OP-AUDIT-5
  OP-AUDIT-4 OP-1 + OP-2 probes execute (depend on OP-1+OP-2 directive)
  OP-AUDIT-4 Path A canonical helper Codex builds
  OP-AUDIT-4 per-substrate symposium convenes (operator action)
  OP-AUDIT-4 Stage 2 Modal A100 smoke fires ($5-15 paid)
  OP-AUDIT-5 v2.5.2 paste (depends on OP-AUDIT-2 inbox channel COMPLETE_INTEGRATED)
```

### 11.4 Race-mode reordering if leaderboard moves

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" non-negotiable: if a sub-0.190 public archive lands during the closure window:

**Immediate action**: PAUSE OPR-CLOSE-3 + OPR-CLOSE-5 (long-burn apparatus maintenance); CONCENTRATE on OPR-CLOSE-1 (master-gradient 6-archive cheap-probe enabler).

**Defer**: OPR-CLOSE-2 + OPR-CLOSE-4 (routing directive landings can wait; they're 1-2h each).

**Frontier-priority**: cathedral autopilot consumes the 6-archive matrix for next-bolt-on candidate ranking; OPR-CLOSE-1 is the bolt-on enabler.

---

## 12. Council deliberation evidence per Catalog #292 + #300

### 12.1 Per-member operating-within assumption surfacing

**Shannon (LEAD)** — operating within: "information-theoretic decomposition of closure-completion verdict space justifies the 9-verdict taxonomy per §6.2. Each verdict carries distinct information content; collapsing them into PASS/FAIL loses signal."

Assumption-Adversary classification: HARD-EARNED (Shannon's per-OP MI decomposition argument is mathematically correct; 9 verdicts encode strict-superset-of-PASS/FAIL information).

**Dykstra (CO-LEAD)** — operating within: "Pareto-feasibility of 'all 5 OPs close in parallel' = FALSE per §11.2 dependency graph. Sequential closure is the only convex-feasible path; alternating-projections argument applies."

Assumption-Adversary classification: HARD-EARNED (dependency graph explicit; convex-feasibility computable).

**Yousfi** — operating within: "closure verification IS adversarial probing of routing-directive landings. The verifier's INCORRECT_NEEDS_FIX verdict surfaces what a steganalysis-style detector would surface: deviations from canonical contract."

Assumption-Adversary classification: HARD-EARNED-WITH-NUANCE (adversarial-probing framing is correct; the analogy to steganalysis is loose but operational).

**Fridrich** — operating within: "closure verification at apparatus_maintenance surface IS the inverse of substrate dispatch — substrate dispatch asks 'does this byte change the score?'; closure verification asks 'does this routing directive change the integration debt?'."

Assumption-Adversary classification: HARD-EARNED.

**Contrarian** — operating within: "fast-and-loose closure verification that conflates COMPLETE with CORRECT is a regression of the discipline this session has built. The 9-verdict taxonomy explicitly distinguishes COMPLETE_PENDING_TESTS from COMPLETE_TESTS_GREEN from COMPLETE_INTEGRATED to prevent this regression."

Assumption-Adversary classification: HARD-EARNED + Contrarian binding revision #1 (§0.5).

**Assumption-Adversary** — operating within: "the assumption 'Codex's autonomous loop will close OP-AUDIT-2 + OP-AUDIT-3 without operator escalation' is CARGO-CULTED-PENDING-EMPIRICAL per audit §9.3. Verifier MUST emit operator-visible alerts when an OP stays in ROUTING_DIRECTIVE_LANDED state >7 days."

Per Catalog #292: Assumption-Adversary's classification = CARGO-CULTED-PENDING-EMPIRICAL; unwind path = nightly cron monitor §6.4 + inbox auto-question §6.6.

### 12.2 Dissent recorded verbatim

**Contrarian VETO consideration** (withdrawn after binding revision #1):

> "I object to any closure verifier that returns a single boolean PASS/FAIL per OP. Such a verifier would re-introduce exactly the cargo-cult class this session has been extincting (CLAUDE.md FORBIDDEN_PATTERNS 'docstring overstatement' generalizes to 'verdict overstatement'). The 9-verdict taxonomy IS the unwind. If the 9 verdicts collapse to PASS/FAIL during implementation, I vote REFUSE."

Resolution: binding revision #1 in §0.5 — taxonomy preserved; VETO withdrawn.

**Assumption-Adversary VETO consideration** (withdrawn after binding revision #2):

> "The verifier without operator-visible alerts on ROUTING_DIRECTIVE_LANDED stuck >7 days re-introduces the assumption that Codex's autonomous loop closes everything. That assumption is empirically falsified per audit §9.3 — 3 routing directives have ZERO canonical_task_status rows after 24h. If the nightly cron alert is omitted from the verifier implementation, I vote REFUSE."

Resolution: binding revision #2 in §0.5 — nightly cron + 7-day alert preserved; VETO withdrawn.

### 12.3 Continual-learning anchor emission per Catalog #300

```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="closure_campaign_pursue_and_confirm_master_20260518",
    topic="5 OP-AUDIT closure operations COMPLETE + CORRECT verification framework",
    council_tier=CouncilTier.T2,
    council_attendees=("Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"),
    council_quorum_met=True,
    council_verdict="PROCEED_WITH_REVISIONS",
    council_dissent=(
        {"member": "Contrarian", "verbatim": "I object to any closure verifier that returns a single boolean PASS/FAIL per OP..."},
        {"member": "Assumption-Adversary", "verbatim": "The verifier without operator-visible alerts on ROUTING_DIRECTIVE_LANDED stuck >7 days..."},
    ),
    council_assumption_adversary_verdict=(
        {"assumption": "Routing-directive landings close OPs without further verification", "classification": "CARGO-CULTED", "rationale": "FIRE-AND-FORGET pattern verified empirically in audit §9.3"},
        {"assumption": "PLANNED_BUT_UNROUTED gaps close in canonical-task-status timeline", "classification": "CARGO-CULTED-PENDING-EMPIRICAL", "rationale": "partial evidence in canonical_task_status rows"},
        {"assumption": "Codex's autonomous loop executes routing directives in EV-order", "classification": "HARD-EARNED-PARTIAL", "rationale": "session state rows show HIGH-EV first"},
        {"assumption": "5 OPs are independent and can close in parallel", "classification": "CARGO-CULTED", "rationale": "§11.2 dependency graph refutes; sequential is the only convex-feasible path"},
    ),
    council_decisions_recorded=(
        "op-routable #1: Codex execute OP-AUDIT-1 master-gradient 6-archive ($0 + 6-12h CPU)",
        "op-routable #2: land 4 NEW Tier-1 routing directives this commit batch ($0 + 2h editor)",
        "op-routable #3: Codex execute OP-AUDIT-2 3 channels ($0 + 6 days editor)",
        "op-routable #4: land 1 NEW Path A routing directive this commit batch ($0 + 1h editor)",
        "op-routable #5: build tac.closure_completion_verifier canonical helper ($0 + 1 day editor)",
    ),
    predicted_mission_contribution="apparatus_maintenance",
    override_invoked=False,
    override_rationale="n/a",
    deferred_substrate_id=None,
    deferred_substrate_retrospective_due_utc=None,
)
append_council_anchor(record)
```

---

## 13. Sister-agent feedback loop per AGENTS.md

Per AGENTS.md "Continual Learning Feedback Loop (canonical memo patterns)":

### 13.1 Claude → Codex direction (this memo + 5 NEW routing directives)

This memo + the 5 NEW routing directives land in `.omx/research/` per Claude's design memo convention. Codex's persistent /goal v2.5.2 pickup at next session ingests them via the queue defined in the multi-loop /goal design memo §3 read-loop.

### 13.2 Codex → Claude direction (post-OP-AUDIT-2 inbox channel COMPLETE_INTEGRATED)

After OP-AUDIT-2 inbox channel lands COMPLETE_INTEGRATED, the closure verifier's nightly cron auto-submits questions per §6.6 when:
- OP transitions to INCORRECT_NEEDS_FIX
- OP stuck in BLOCKED for >7 days
- OP stuck in OPERATOR_ACTION_REQUIRED for >7 days

Codex picks up questions via `tools/codex_to_claude_inbox.py list-questions` in its next session and either:
- Fixes the underlying issue → submits answer via `submit-answer`
- Escalates to operator → writes findings memo per `codex_findings_<topic>_<utc>_codex.md` convention

### 13.3 Anti-overlap rules per AGENTS.md

- Claude OWNS this memo (closure campaign coordinator + design spec for closure verifier)
- Codex OWNS execution of OP-AUDIT-1 / OP-AUDIT-2 / OP-AUDIT-3 / OP-AUDIT-4 helper builds (per routing directives)
- Operator OWNS OP-AUDIT-5 paste action

No overlap: Claude does NOT bypass commit serializer; Codex does NOT spawn paradigm research; both honor CLAUDE.md non-negotiables.

---

## 14. Per-OP reactivation criteria per Catalog #325

Per CLAUDE.md "Forbidden premature KILL" non-negotiable: every closure operation has explicit reactivation criteria so if an OP stalls at BLOCKED state, the criteria document what unblocks it.

### 14.1 OP-AUDIT-1 reactivation

**Reactivation trigger**: PR106 format0d projector OR PR107 Apogee projector lands as canonical extractor function (verified via `grep -c "def parse_pr106_format0d_archive_bytes\|def parse_pr107_apogee_archive_bytes" tools/extract_master_gradient.py`).

**Predicted closure**: Week 1-2 per Codex's active execution velocity (3 projectors landed in 2 days; trajectory suggests 5 total in 3-5 days).

### 14.2 OP-AUDIT-2 reactivation (per channel)

**Reactivation trigger**: ANY of the 3 routing directives (inbox / memory / hypergraph) has its first canonical_task_status row emitted (`grep -c "codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518" .omx/state/canonical_task_status.jsonl > 0`).

**Predicted closure**: Week 1-2 (Codex auto-picks-up at next session per multi-loop /goal v2.5.2).

### 14.3 OP-AUDIT-3 reactivation (per Tier-1 helper)

**Reactivation trigger** (per helper): OP-AUDIT-1 6-archive matrix COMPLETE_INTEGRATED (per closure verifier verdict) AND helper's routing directive's first canonical_task_status row emitted.

**Predicted closure**: Week 2-4.

### 14.4 OP-AUDIT-4 reactivation

**Reactivation trigger**: OP-1 OOD-similarity probe verdict = `OOD_SIMILAR_ENOUGH` AND OP-2 architecture-compatibility verdict = `LOADABLE` AND per-substrate symposium PROCEED.

**Predicted closure**: Week 3-4 (probes are $0 + 10-15 min each; symposium ~1 day operator-attention).

### 14.5 OP-AUDIT-5 reactivation

**Reactivation trigger**: Operator action — paste v2.5.2 into Codex CLI /goal context.

**Predicted closure**: when operator next interacts with Codex CLI (any time within 7 days).

---

## 15. Cross-references

### 15.1 Primary upstream sources

- `.omx/research/wiring_integration_orphan_audit_post_12_landings_20260518.md` (commit `b1aae8536`; 841 lines) — THE AUTHORITY for 5 OP-AUDIT operations
- `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` (commit `14c03c57a`; 1449 lines) — 9 design memo synthesis
- `CLAUDE.md` (FULL non-negotiables; especially "Subagent coherence-by-default" + "Operator gates must be wired and used" + "Mission alignment")
- `AGENTS.md` (Claude × Codex role specialization + canonical memo patterns)

### 15.2 9 design memos (per cross-stack synthesis §3)

1. `.omx/research/n_set_venn_classification_design_memo_pair_region_class_frame_axis_20260518.md` (71.9 KB)
2. `.omx/research/phase_1_fisher_precondition_canonical_helper_design_memo_20260518.md` (129.3 KB)
3. `.omx/research/riemannian_newton_substrate_engineering_design_memo_20260518.md` (119.0 KB)
4. `.omx/research/tropical_d_seg_solver_design_memo_20260518.md` (117.4 KB)
5. `.omx/research/council_z8_hierarchical_predictive_coding_per_substrate_symposium_20260518.md` (sister T3 symposium)
6. `.omx/research/tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518.md` (111.7 KB)
7. `.omx/research/cargo_cult_resurrection_top_3_symposiums_v1_faiss_c6_ibps_v2_nscs06_v8_variant_c_20260518.md` (sister T3 symposium)
8. `.omx/research/grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518.md` (sister T3 symposium)
9. `.omx/research/tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518.md` (100.5 KB)
10. `.omx/research/dp1_pr101_composition_design_memo_20260518.md` (116.4 KB) — sister design (OP-AUDIT-4)

### 15.3 Sister routing directives (verified pre-edit)

- `.omx/research/codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518.md` (OP-AUDIT-1)
- `.omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` (OP-AUDIT-2 inbox)
- `.omx/research/codex_routing_directive_claude_memory_hermetic_export_channel_20260518.md` (OP-AUDIT-2 memory)
- `.omx/research/codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518.md` (OP-AUDIT-2 hypergraph)
- `.omx/research/codex_routing_directive_dp1_pr101_op1_op2_zero_cost_probes_20260518.md` (OP-AUDIT-4 probes only)

### 15.4 5 NEW routing directives landed in THIS commit batch

- `.omx/research/codex_routing_directive_canonical_n_set_venn_classification_package_20260518.md` (OP-AUDIT-3 VENN)
- `.omx/research/codex_routing_directive_canonical_phase_1_fisher_precondition_package_20260518.md` (OP-AUDIT-3 FISHER)
- `.omx/research/codex_routing_directive_canonical_riemannian_newton_meta_substrate_package_20260518.md` (OP-AUDIT-3 RIEM)
- `.omx/research/codex_routing_directive_canonical_tropical_d_seg_solver_package_20260518.md` (OP-AUDIT-3 TROP)
- `.omx/research/codex_routing_directive_dp1_pr101_path_a_canonical_helper_package_20260518.md` (OP-AUDIT-4 full helper)

### 15.5 Canonical helper specifications (referenced by §6 closure verifier design)

- `src/tac/closure_completion_verifier.py` (THIS memo's primary deliverable spec — Codex builds per OPR-CLOSE-5)
- `src/tac/council_continual_learning.py` (sister canonical helper pattern for council deliberation persistence)
- `src/tac/probe_outcomes_ledger.py` (sister canonical 4-layer helper for probe outcomes per Catalog #313)
- `src/tac/deploy/modal/call_id_ledger.py` (sister canonical 4-layer helper for Modal call_ids per Catalog #245)
- `tools/operator_briefing.py` (consumer for closure verifier dashboard surface)
- `tools/cathedral_autopilot_autonomous_loop.py` (consumer for closure-aware reward factor per hook #4)

### 15.6 CLAUDE.md non-negotiables this memo honors

- **Subagent coherence-by-default** — 6-hook wire-in declared (§13)
- **Operator gates must be wired and used** — closure verifier wired into operator_briefing
- **Mission alignment** — apparatus_maintenance classification per Catalog #300 §"Mission alignment"
- **Lane maturity registry** — lane `lane_closure_campaign_pursue_and_confirm_20260518` pre-registered per Catalog #126
- **META-ASSUMPTION ADVERSARIAL REVIEW** — Assumption-Adversary classifications per §12.1
- **Per-substrate symposium** — N/A (apparatus_maintenance, not substrate)
- **Predicted band validation** — N/A (no score band)
- **Forbidden premature KILL** — per-OP reactivation criteria per §14

---

## 16. Closing — operator-action checklist

### 16.1 IMMEDIATELY (this session)

- [x] Master memo lands at canonical path (THIS file)
- [x] Lane pre-registered: `lane_closure_campaign_pursue_and_confirm_20260518` (Catalog #126)
- [x] Checkpoint discipline trace per Catalog #206
- [ ] 5 NEW routing directives land via canonical serializer (sister commit batch immediately following this memo)
- [ ] Council deliberation anchor appended to `.omx/state/council_deliberation_posterior.jsonl` per Catalog #300 (sister to this memo's landing commit)

### 16.2 NEAR-TERM (Week 1-2)

- [ ] Operator pastes v2.5.2 into Codex CLI /goal context (OP-AUDIT-5)
- [ ] Codex executes OP-AUDIT-1 master-gradient 6-archive remaining projectors
- [ ] Codex executes OP-AUDIT-2 3 channels (inbox / memory / hypergraph)

### 16.3 MEDIUM-TERM (Week 2-4)

- [ ] Codex executes OP-AUDIT-3 4 Tier-1 helpers (cascading per dependency graph)
- [ ] Codex executes OP-AUDIT-4 Path A canonical helper
- [ ] Codex builds `tac.closure_completion_verifier` per OPR-CLOSE-5 (routed via followon directive)

### 16.4 ONGOING (post-completion)

- [ ] Nightly cron via multi-loop /goal v2.5.2 runs `tools/closure_completion_verifier.py nightly-cron`
- [ ] `.omx/state/closure_completion_anchors.jsonl` accumulates per-OP daily verdict rows
- [ ] Closure dashboard at `tools/closure_campaign_dashboard.py` operator-runnable any time
- [ ] Inbox auto-questions fire on INCORRECT_NEEDS_FIX (gated on OP-AUDIT-2 inbox channel COMPLETE_INTEGRATED)

---

**END OF CLOSURE CAMPAIGN PURSUE AND CONFIRM MASTER MEMO**

Operator verbatim from 2026-05-18: *"must pursue and confirm all closure operations complete and correct"* — this memo + 5 NEW routing directives + closure verifier design spec is the canonical pursue-and-confirm closure surface.

Per CLAUDE.md "Subagent coherence-by-default" sister-subagent ownership map: closure work is split across Claude (this memo + 5 routing directives + verifier design) and Codex (execution of routing directives + verifier build). Per AGENTS.md anti-overlap rules: Claude does NOT build the helpers; Codex does NOT design the campaign coordination.
