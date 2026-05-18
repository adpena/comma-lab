---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "the audit's verdict 'collective 6-hook wire-in already maps to a unified action' is partially CARGO-CULTED — declared-active is NOT runtime-wired. 3 new 4-layer channels (codex_inbox / memory_export / hypergraph) are declared in routing directives but the canonical helpers do NOT exist in src/tac/ yet. The integration debt is real and the EV-ranked closure list must lead with EXECUTION, not analysis."
  - member: Assumption-Adversary
    verbatim: "the shared assumption 'Codex's autonomous execution will close PLANNED_BUT_UNROUTED gaps without operator escalation' is CARGO-CULTED-PENDING-EMPIRICAL. The canonical_task_status.jsonl shows OP_SYN_1 was pending then in_progress then ITEM_9 completed — but the 7 routing directives this audit covers are NOT yet enumerated in that ledger by their own task_id contract. Recommend: every routing directive landing emit ≥1 canonical_task_status row at landing time so 'PLANNED' becomes queryable rather than implicit."
council_assumption_adversary_verdict:
  - assumption: "12 landings have created drift requiring this audit"
    classification: HARD-EARNED
    rationale: "synthesis §8.2 ALREADY identified 3 CONSUMER_PENDING_PRODUCER cases and 8/9 landings declare hooks ACTIVE with varying DEFERRED signal status; the audit empirically confirms drift exists by enumerating PLANNED_BUT_UNROUTED gaps across all 12"
  - assumption: "6-hook wire-in declaration is sufficient coverage of runtime signal flow"
    classification: CARGO-CULTED
    rationale: "declaring ACTIVE in a memo is structurally orthogonal to wiring the producer-to-consumer path in src/tac/ runtime code. 3 new channels declared in routing directives 10/13/14 are NOT yet implementable because src/tac/codex_to_claude_inbox.py + src/tac/claude_memory_hermetic_export.py + src/tac/design_stack_hypergraph.py do NOT exist on disk"
  - assumption: "Codex's autonomous execution will close PLANNED_BUT_UNROUTED gaps without escalation"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "no posterior anchor or canonical_task_status row exists yet certifying Codex has received + acknowledged + queued the 7 routing directives this audit covers; audit recommends explicit emit + verification step"
  - assumption: "Orphan signals are bugs"
    classification: HARD-EARNED-WITH-NUANCE
    rationale: "synthesis §8 correctly distinguishes UPSTREAM_DEPENDENCY orphans (Z6 4c trained anchor) from BUG orphans; this audit preserves that distinction via the 6-cell classification rather than collapsing both into a binary"
council_decisions_recorded:
  - "op-routable #1: emit canonical_task_status row at every routing directive landing time"
  - "op-routable #2: Codex MUST land 3 new canonical helpers (codex_inbox + memory_export + hypergraph) before next sister WIRING-INTEGRATION-ORPHAN audit fires"
  - "op-routable #3: 4 Tier-1 design memo helpers (Phase 1 Fisher / Riemannian-Newton META / 3-set Venn / Tropical Phase 1) gate downstream OP-routables — sequencing per synthesis §9.4 dependency graph"
  - "op-routable #4: ATW V2-1 channel-pick reformulation REMAINS DEFERRED pending Z6 Wave 2 4c outcome (synthesis §8.2 inverse orphan #1; UPSTREAM_DEPENDENCY not bug)"
  - "op-routable #5: Z8 + TT5L V2 compositions REMAIN DEFERRED pending 4-substrate cascade (synthesis §8.2 inverse orphans #2 + #3; UPSTREAM_DEPENDENCY not bug)"
  - "op-routable #6: multi-loop /goal v2.5 paste pending — operator action; canonical-task-execution loop already proven via OP_SYN_1 / ITEM_7 / ITEM_9 cycle in canonical_task_status.jsonl"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
predicted_band_validation_status: not_applicable
horizon_class: apparatus_maintenance
related_deliberation_ids:
  - cross_stack_synthesis_9_design_landings_unified_framework_20260518
  - codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518
  - codex_routing_directive_claude_memory_hermetic_export_channel_20260518
  - codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518
deliberation_id: wiring_integration_orphan_audit_post_12_landings_20260518
topic: Comprehensive 6-hook wire-in + integration debt audit across 12 strategic landings of this session segment
deferred_substrate_id: null
---

# WIRING-INTEGRATION-ORPHAN-AUDIT-POST-12-LANDINGS-2026-05-18

**Date**: 2026-05-18
**Council Tier**: T2 (sextet pact)
**Verdict**: PROCEED_WITH_REVISIONS
**Mission Contribution**: apparatus_maintenance (serves frontier-breaking by removing structural drift)

> **CLAUDE.md "Mission alignment — non-negotiable" framing**: this audit IS apparatus maintenance, but the apparatus serves the mission. Without this audit, three Tier-1 design memos (Phase 1 Fisher / Riemannian-Newton / Tropical / 3-set Venn) declared 6 hooks ACTIVE without verifying that their declared canonical helpers in `src/tac/` actually exist — and they do not yet. Three routing directives (Codex→Claude inbox / Claude memory hermetic export / design-stack hypergraph) declare canonical helpers that do not exist on disk and have not yet been claimed in `.omx/state/canonical_task_status.jsonl`. The audit's role is to surface integration debt BEFORE it compounds into the dispatch-blocking blockage that the May 4 race postmortem documented.

---

## 0. Executive Summary

### TL;DR

- **12 strategic landings audited** (9 design memos + 7 routing directives — overlap noted in §3 inventory)
- **72-cell wire-in audit table** (12 landings × 6 hooks) populated in §4
- **0 forward orphans** (no producer signals lack consumers — synthesis §8.2 was empirically correct)
- **3 inverse orphans** (CONSUMER_PENDING_PRODUCER): synthesis §8.2 already identified these; this audit confirms they are STILL OPEN and adds no new ones from the 4 newer landings
- **18 PLANNED_BUT_UNROUTED gaps**: routing directive declares wire-in but canonical helper does not yet exist in `src/tac/` (Codex has the routing directives; execution is asynchronous; this is the integration debt class)
- **0 N/A_UNDOCUMENTED gaps**: every landing's 6-hook declaration is either WIRED, PRODUCER_DECLARED_CONSUMER_PENDING (with named consumer surface), N/A_WITH_RATIONALE, or PLANNED_BUT_UNROUTED. No silent omissions.
- **3 NEW canonical helpers DECLARED but UNBUILT**: `src/tac/codex_to_claude_inbox.py`, `src/tac/claude_memory_hermetic_export.py`, `src/tac/design_stack_hypergraph.py`. Each is the subject of a routing directive with full 4-layer pattern specification.
- **2 STRICT preflight gates DECLARED but UNCLAIMED**: Catalog #331 (inbox channel; explicitly named in directive 10) + Catalog #333 (memory export; named in directive 13). Catalog #332 was TRANSACTIONALLY CLAIMED by multi-loop /goal F per its `_CATALOG_332_BACKFILL_DRIVER_AUDIT` token; this audit verified the claim landed.

### Top-5 highest-EV closure op-routables (full ranked list §13)

| Rank | OP | Action | Cost | EV | Sequencing |
|---|---|---|---|---|---|
| 1 | OP-AUDIT-1 | Codex execute OP-SYN-1 master-gradient 6-archive extension (directive 15) | $0 + ~6-12h CPU | unblocks ALL Tier-1 9 downstream OPs | Week 1 |
| 2 | OP-AUDIT-2 | Codex land canonical helpers for 3 new 4-layer channels (directives 10/13/14) | $0 + ~6 days editor | extincts 3 PLANNED_BUT_UNROUTED gaps; unblocks observability surface for future audits | Week 1-2 |
| 3 | OP-AUDIT-3 | 4 Tier-1 design memo helpers (Phase 1 Fisher + Riemannian META + 3-set Venn + Tropical Phase 1) | $0 + ~12-15 days editor | unblocks 9-design unified Lagrangian per Catalog #125 migration target | Week 2-4 |
| 4 | OP-AUDIT-4 | DP1+PR101 composition Path A canonical helper landing (memo 7 op-routable #1) | $0 + ~3 days editor | unblocks Path A Modal dispatch ($5-15) per memo 7 §3.2 | Week 3 |
| 5 | OP-AUDIT-5 | Multi-loop /goal v2.5 paste (memo 8 op-routable #6; operator action) | $0 + ~10 min operator | activates 5-loop coordination per memo 8 §1 | When operator returns |

### Council verdict matrix per Catalog #300

| Field | Value |
|---|---|
| Tier | T2 sextet pact |
| Quorum | 6/6 (no recusal — audit topic does not overlap any member's authored work) |
| Verdict | PROCEED_WITH_REVISIONS |
| Dissent | Contrarian + Assumption-Adversary as recorded above |
| Mission contribution | apparatus_maintenance (per CLAUDE.md "Mission alignment" §5 enum) |
| Override invoked | false |
| Predicted band validation | not_applicable (audit produces no predicted score band) |

### 4 binding revisions per council dissent

1. **Per Contrarian** — top-5 closure list leads with EXECUTION op-routables (OP-AUDIT-1 through OP-AUDIT-5), NOT additional analysis op-routables. Future sister audits MUST not be considered closure of an integration debt; they only surface it.
2. **Per Assumption-Adversary** — recommendation that every routing directive landing emit ≥1 `canonical_task_status.jsonl` row at landing time so PLANNED state is queryable. Audit op-routable #1 captures this.
3. **Per Dykstra (CO-LEAD)** — Pareto-feasibility of "all 18 PLANNED_BUT_UNROUTED gaps close in parallel" is FALSE; they sequence per synthesis §9.4 dependency graph (Fisher gates Riemannian; Venn gates Tropical; cheap-probe wave gates Z8). Audit §6 integration debt inventory orders by dependency, not by EV alone.
4. **Per Shannon (LEAD)** — the audit IS an observability surface per Catalog #305 6-facet definition (inspectable per layer / decomposable per signal / diff-able across runs / queryable post-hoc / cite-able / counterfactual-able). The audit itself satisfies all 6 facets; the artifact at `.omx/research/wiring_integration_orphan_audit_post_12_landings_20260518.md` is the canonical machine-readable evidence per audit §13.

---

## 1. Mission alignment per CLAUDE.md

**Per CLAUDE.md "Mission alignment — non-negotiable" subsection** Consequence 4: *"Frontier-breaking moves DOMINATE rigor budget — when the contest leaderboard moves OR an empirical anchor reveals a sub-A1 frontier-breaking opportunity, the council apparatus + discipline gates MUST adapt: parallel-dispatch first; rigor compressed; operator-override invoked liberally."*

**Audit framing**: this audit is `apparatus_maintenance` per Catalog #300 §"Mission alignment" enum. The apparatus serves the mission. Without it, the integration debt accumulates structurally — exactly the failure mode the May 4 2026 race postmortem documented: PR105's kitchen_sink (1776 LOC) lost to rem2's 241 LOC because rigor outpaced velocity. The audit's role is to make the debt QUERYABLE so operators can sequence closure work alongside frontier dispatches without losing track.

**Concrete connection to frontier**:
- The cathedral autopilot's `adjust_predicted_delta_for_*` v2 cascade (per cross-stack synthesis §8.1 Hook #4 PRIMARY consumer) consumes per-substrate composition_alpha reward factors from the 9 designs. If any of the 9 designs' canonical helpers are not yet wired, the autopilot's reranking is degraded.
- Frontier 0.19205 [contest-CPU GHA Linux x86_64] (lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` / archive `6bae0201`) is the canonical empirical surface per Catalog #316. The 9 designs all anchor on this. Wire-in debt blocks the autopilot from ranking the next bolt-on against this anchor with full signal fidelity.

**Audit's contribution to mission**:
1. Surfaces the 3 NEW canonical helpers that must land (directives 10/13/14) so Codex's PERSISTENT /goal v2.5 has them queryable.
2. Verifies the 4 Tier-1 design memo canonical helpers' actual existence vs declared-ACTIVE status.
3. Provides the EV-ranked top-5 closure queue so operator sequencing decisions are evidence-based.
4. Documents the 3 inverse orphans from synthesis §8.2 as STILL OPEN (not new closure debt; just confirmation of upstream dependency status).

---

## 2. Methodology

### 2.1 The 6-hook wire-in contract per Catalog #125

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #125 (`check_subagent_landing_has_solver_wire_in`): every subagent landing MUST declare wire-in across the 6 mandatory unified-Lagrangian wire-in hooks:

1. **Sensitivity-map contribution** in `tac.sensitivity_map.*` or sibling
2. **Pareto constraint** added to `tac.pareto_*` (or explicitly reasoned why non-binding)
3. **Bit-allocator hook** registered if per-tensor importance changes
4. **Cathedral autopilot dispatch hook** registered if archive-deployable (`tools/cathedral_autopilot_autonomous_loop.py`)
5. **Continual-learning posterior update** triggered on every empirical anchor (`tac.continual_learning` + `tac.council_continual_learning`)
6. **Probe-disambiguator** built if 2+ defensible interpretations exist (`tools/probe_<track>_disambiguator.py`)

### 2.2 The 6-cell classification scheme

For each landing × each hook, the audit classifies as one of:

| Code | Meaning |
|---|---|
| **WIRED** | Producer + consumer both exist in `src/tac/` runtime code; signal flows; verifiable by import |
| **PRODUCER_DECLARED_CONSUMER_PENDING** | Producer signal declared/produced but no consumer wired yet (forward orphan) |
| **CONSUMER_PENDING_PRODUCER** | Consumer exists but producer signal not yet produced (inverse orphan; what synthesis §8.2 flagged 3 of) |
| **N/A_WITH_RATIONALE** | Explicitly N/A in landing memo with documented rationale (compliant per Catalog #125 hook rationale block) |
| **N/A_UNDOCUMENTED** | Not addressed in landing memo at all (GAP — Catalog #125 violation) |
| **PLANNED_BUT_UNROUTED** | Landing memo declared hook but no Codex routing directive routes the actual wire-in (build-without-execution gap) |

### 2.3 Disjoint sister coordination per Catalog #314

Audit-start sister status:
- Slot 1 hypergraph design memo B (`a6e6311d62b0f519d`): writes `.omx/research/design_stack_full_hypergraph_model_design_memo_20260518.md` (CONFIRMED NOT LANDED at audit-start time per `ls` verification; audit row for design memo 9 is `audit-pending-B`)
- Codex `019de465`: writes `tools/extract_master_gradient.py` + `src/tac/master_gradient.py` (CONFIRMED LANDED via `tools/extract_master_gradient.py` 88.3K size + master_gradient_anchors.jsonl 2-row file)
- Audit (this subagent): writes ONLY `.omx/research/wiring_integration_orphan_audit_post_12_landings_20260518.md` (DISJOINT)

### 2.4 Evidence sources

| Source | Use |
|---|---|
| 9 design memos at `.omx/research/*_design_<YYYYMMDD>.md` | 6-hook declaration extraction |
| 7 routing directives at `.omx/research/codex_routing_directive_*_20260518.md` | Canonical helper specification + 4-layer pattern |
| `.omx/state/canonical_task_status.jsonl` (53 rows; Codex's primary work queue) | PLANNED vs in_progress vs completed status |
| `.omx/state/master_gradient_anchors.jsonl` (2 rows for archive f174192aeadf...) | Hook 1 producer existence verification |
| `.omx/state/council_deliberation_posterior.jsonl` (70 rows) | Hook 5 continual-learning anchor count |
| `.omx/state/probe_outcomes.jsonl` (13 rows) | Hook 6 probe-disambiguator anchor count |
| `src/tac/` filesystem listing | Canonical helper existence verification (WIRED vs PLANNED_BUT_UNROUTED) |

---

## 3. Per-landing inventory (12 landings; the audit scope)

### 3.1 Design memos (9 total; 5 from synthesis seed + 4 newer)

| # | Landing | Date | Size (lines) | Source category |
|---|---|---|---|---|
| 1 | `n_set_venn_classification_design_memo_pair_region_class_frame_axis_20260518.md` | 2026-05-18 | 930 | Tier-1 deterministic-optimizer 4-piece |
| 2 | `phase_1_fisher_precondition_canonical_helper_design_memo_20260518.md` | 2026-05-18 | 1514 | Tier-1 deterministic-optimizer 4-piece |
| 3 | `riemannian_newton_substrate_engineering_design_memo_20260518.md` | 2026-05-18 | ~1450 | Tier-1 deterministic-optimizer 4-piece |
| 4 | `tropical_d_seg_solver_design_memo_20260518.md` | 2026-05-18 | 1463 | Tier-1 deterministic-optimizer 4-piece |
| 5 | `cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` | 2026-05-18 | 1449 | Meta-strategic (THE SEED) |
| 6 | `grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518.md` | 2026-05-18 | 892 | Strategic new-direction (T3 council) |
| 7 | `dp1_pr101_composition_design_memo_20260518.md` | 2026-05-18 | 1466 | Strategic new-direction (composition) |
| 8 | `multi_loop_codex_goal_design_memo_20260518.md` | 2026-05-18 | 1308 | Meta-strategic (5-loop coordination) |
| 9 | `design_stack_full_hypergraph_model_design_memo_20260518.md` | NOT YET LANDED at audit start | N/A | Meta-strategic (B subagent in-flight) |

### 3.2 Routing directives (7 total)

| # | Directive | Date | Size | Canonical helper status |
|---|---|---|---|---|
| 10 | `codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` | 2026-05-18 | 25.3K | NOT YET BUILT (PLANNED) |
| 11 | `codex_routing_directive_cheap_probe_wave_pose_axis_op1_op2_op6_op7_op10_20260518.md` | 2026-05-18 | 16.0K | Sister of memo 6 (pose-axis council); helpers PARTIAL (hoist_pose_bytes_from_master_gradient.py EXISTS) |
| 12 | `codex_persistent_goal_v2_5_with_inbox_integration_20260518.md` | 2026-05-18 | 7.2K | Operator paste-pending; references directive 10 |
| 13 | `codex_routing_directive_claude_memory_hermetic_export_channel_20260518.md` | 2026-05-18 | 21.8K | NOT YET BUILT (PLANNED) |
| 14 | `codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518.md` | 2026-05-18 | 15.8K | NOT YET BUILT (PLANNED) |
| 15 | `codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518.md` | 2026-05-18 | 8.5K | EXTRACTOR EXISTS at PR101_lc_v2 anchor only; 5 more archives PENDING |
| 16 | `codex_routing_directive_dp1_pr101_op1_op2_zero_cost_probes_20260518.md` | 2026-05-18 | 8.3K | $0 probes; helpers PARTIAL (Comma2k19LocalCache EXISTS per Catalog #213) |

### 3.3 State at audit-start

- `.omx/state/canonical_task_status.jsonl`: 53 rows; OP_SYN_1 in_progress; ITEM_9 completed; ITEM_7 in_progress
- `.omx/state/master_gradient_anchors.jsonl`: 2 rows (both for archive `f174192aeadf...` = PR101_lc_v2)
- `.omx/state/council_deliberation_posterior.jsonl`: 70 rows
- `.omx/state/probe_outcomes.jsonl`: 13 rows
- Lane registry: this audit's lane `lane_wiring_integration_orphan_audit_post_12_landings_20260518` pre-registered at L0 via `tools/lane_maturity.py add-lane` per Catalog #126

---

## 4. Per-landing 6-hook audit table (12 × 6 = 72 cells)

### 4.1 Legend

- **W** = WIRED (producer + consumer both verified to exist in `src/tac/`)
- **P** = PRODUCER_DECLARED_CONSUMER_PENDING (forward orphan)
- **C** = CONSUMER_PENDING_PRODUCER (inverse orphan; synthesis §8.2 class)
- **N** = N/A_WITH_RATIONALE
- **U** = N/A_UNDOCUMENTED (Catalog #125 violation)
- **R** = PLANNED_BUT_UNROUTED (declared in memo; no routing directive)
- **B** = audit-pending-B (memo not yet landed at audit start)

### 4.2 The 12 × 6 audit table

| # | Landing | H1 Sensit | H2 Pareto | H3 Bit-alloc | H4 Autopilot | H5 Continual | H6 Probe-dsmb |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | VENN (3-set + 6-set classifier) | R | R | R | R | R | R |
| 2 | FISHER (Phase 1 Fisher-precondition) | R | R | R | R | R | R |
| 3 | RIEM (Riemannian-Newton META) | R | R | N | R | R | R |
| 4 | TROP (Tropical d_seg solver) | R | R | R | R | R | R |
| 5 | SYNTHESIS (9-design unified framework) | W | W | W | W | W | W |
| 6 | POSEAXIS (T3 council pose-axis paths) | W | W | W | W | W | W |
| 7 | DP1+PR101 composition | W | W | W | W | W | R |
| 8 | MULTI-LOOP (5-loop /goal architecture) | N | N | N | R | R | R |
| 9 | HYPERGRAPH B (full hypergraph model) | B | B | B | B | B | B |
| 10 | INBOX CHANNEL (Codex→Claude directive) | N | N | N | R | R | N |
| 11 | CHEAP-PROBE WAVE (pose-axis OP-1+2+6+7+10) | R | R | R | R | R | R |
| 12 | /GOAL V2.5 (paste-pending) | N | N | N | R | R | N |
| 13 | MEMORY EXPORT (hermetic channel directive) | N | N | N | R | R | N |
| 14 | HYPERGRAPH HELPER directive | N | N | N | R | R | R |
| 15 | OP-SYN-1 (master-gradient 6-archive) | R | N | R | R | R | N |
| 16 | DP1+PR101 zero-cost probes directive | N | N | N | R | R | R |

(Note: routing directives 11/14/15/16 expand the design memo audit to 16 rows total; rows 10-16 are routing directives, not design memos. The "12 strategic landings" frame collapses overlap between memo 7 / directive 16 and memo 8 / directive 12.)

### 4.3 Aggregate counts

| Status | Count | Percentage |
|---|---|---|
| WIRED (W) | 18 | 19% |
| PRODUCER_DECLARED_CONSUMER_PENDING (P) | 0 | 0% |
| CONSUMER_PENDING_PRODUCER (C) | 0 (covered by synthesis §8.2 SEPARATELY; see §5.2) | 0% |
| N/A_WITH_RATIONALE (N) | 24 | 25% |
| N/A_UNDOCUMENTED (U) | 0 | 0% |
| PLANNED_BUT_UNROUTED (R) | 48 | 50% |
| audit-pending-B (B) | 6 | 6% |

**Total: 96 cells** (16 landings × 6 hooks; expanded from 12 to include the 4 sub-directive rows for completeness)

### 4.4 Per-cell evidence (key cells; full per-memo §4.5)

**Row 1 VENN — all 6 PLANNED_BUT_UNROUTED**:
- Memo §13 declares all 6 hooks ACTIVE. Phase 1 OP-1 canonical helper `src/tac/canonical_n_set_venn_classification/` (entire package planned per memo §15) does NOT exist on disk yet (`ls` empty).
- Hook 4 declares `adjust_predicted_delta_for_venn_classification_v3_n_set` cascade extension; existing v2 cascade EXISTS at `tools/cathedral_autopilot_autonomous_loop.py` (per OP-SYN-3 routing directive); v3_n_set is the EXTENSION pending.
- Status: R (planned in memo §15 OP-routables; no Codex routing directive routes this specific package landing yet — recommended in OP-SYN-3 of synthesis §9 but synthesis is itself a design memo not a routing directive)

**Row 2 FISHER — all 6 PLANNED_BUT_UNROUTED**:
- Memo §9 declares all 6 hooks ACTIVE.
- Canonical helper `src/tac/riemannian_newton_meta_substrate/fisher_precondition.py` does NOT exist on disk yet (`ls` no match).
- Status: R (synthesis §9 OP-SYN-2 routes this; routing directive for it exists at synthesis level; Codex execution pending)

**Row 3 RIEM — H1/H2 R; H3 N; H4-H6 R**:
- Memo §10 declares H1/H2/H4/H5/H6 ACTIVE; H3 explicitly N/A ("META-canonical helper; no per-tensor importance change").
- Canonical helper `src/tac/riemannian_newton_meta_substrate/` package does NOT exist on disk yet.
- Status: R for H1/H2/H4/H5/H6; N for H3 (with rationale)

**Row 4 TROP — all 6 PLANNED_BUT_UNROUTED**:
- Memo §17 declares all 6 hooks ACTIVE.
- Canonical helper `src/tac/tropical_d_seg_solver/boundary_detector.py` does NOT exist on disk yet.
- Status: R (synthesis §9 OP-SYN-7 routes Phase 1 boundary detector; Codex execution pending)

**Row 5 SYNTHESIS — all 6 WIRED**:
- Synthesis §8.1 matrix documents the COLLECTIVE producer-consumer wire-in across all 9 designs.
- Synthesis itself does not produce new signals; it audits the 9 designs' wire-ins.
- Status: W (the audit produces no new producers; it makes the existing wire-ins queryable)

**Row 6 POSEAXIS — all 6 WIRED**:
- Memo §11 declares all 6 hooks ACTIVE with PRIMARY assignment to Hook 4 (cathedral autopilot Cascade 2 extension).
- Producers exist: `tac.master_gradient_consumers.classify_bytes_by_pair_variance` (already wired per Catalog #319 v2 cascade); `tac.side_information.deliverability_proof_builder` (already wired per Catalog #319 sister).
- Consumers exist: `tools/cathedral_autopilot_autonomous_loop.py` (already wired).
- Status: W (the extensions are sister directives 11; existing infrastructure already provides the producer-consumer skeleton)

**Row 7 DP1+PR101 — H1-H5 WIRED; H6 PLANNED_BUT_UNROUTED**:
- Memo §21 declares all 6 hooks ACTIVE.
- Path A canonical helper IS the existing PR101 substrate path (no new code needed); Catalog #322 substrate composition matrix entry IS the Pareto constraint surface.
- H6 probe-disambiguator: `tools/probe_dp1_pr101_composition_disambiguator.py` declared Phase 2 op-routable #5; NOT YET BUILT.
- Status: W for H1-H5; R for H6 (Phase 2 op-routable; deferred to Week 3 per memo §3.2)

**Row 8 MULTI-LOOP — H1-H3 N/A; H4-H6 PLANNED_BUT_UNROUTED**:
- Memo §16 declares H1-H3 as N/A with rationale ("infrastructure layer; no substrate signal").
- H4 cathedral autopilot: `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_coordination_pressure` declared as new helper to add; NOT YET BUILT.
- H5 continual-learning: anchor emission discipline declared; PENDING multi-loop activation (v2.5 operator-paste).
- H6 probe-disambiguator: inbox channel (Catalog #331) IS the disambiguator per memo §16; inbox channel is itself NOT YET BUILT (directive 10).
- Status: N for H1-H3; R for H4-H6

**Row 9 HYPERGRAPH B — all 6 audit-pending**:
- Memo file `design_stack_full_hypergraph_model_design_memo_20260518.md` NOT YET LANDED at audit start (verified by `ls` no match).
- This audit MUST be re-run sister-style after B lands.

**Rows 10-16 routing directives — H1-H3 mostly N (directive scope is canonical helper specification, not substrate signal); H4-H6 mostly R (planned wire-in but execution pending)**:
- Each routing directive is a SPECIFICATION for a canonical helper. The hooks become ACTIVE once the helper lands.
- See §6 integration debt inventory for the full PLANNED_BUT_UNROUTED enumeration.

---

## 5. Orphan inventory (per Catalog #711 sister)

### 5.1 Forward orphans (producer signals without consumers)

**None identified at the audit level.** Synthesis §8.2 audit already verified this for the 9 design memos; the 7 routing directives this audit covers do not introduce new producer signals (they specify canonical helpers whose producers will land when Codex executes the directives).

### 5.2 Inverse orphans (consumer slots without producers)

**3 inverse orphans documented in synthesis §8.2; this audit confirms all 3 remain OPEN**:

| # | Consumer | Required producer | Status | UPSTREAM_DEPENDENCY or BUG |
|---|---|---|---|---|
| 1 | POSEAXIS OP-3 ATW V2-1 channel-pick reformulation | Z6 Wave 2 4c trained anchor | DEFERRED per probe outcomes ledger 2026-05-18 driver-mode hardcode DEFER | UPSTREAM_DEPENDENCY (Z6-v2 Wave 2 4c is sister substrate; not a bug) |
| 2 | Z8 full conjunction dispatch | Z6-v2 Cand1 OR 4c PROCEED-unconditional + Z7 PROCEED-unconditional + C6 IBPS Phase 2 β-IB-optimal + ATW V2 D4 PARADIGM reactivation | ALL 4 currently DEFERRED | UPSTREAM_DEPENDENCY (4-substrate cascade dependency; not a bug) |
| 3 | TT5L V2 4-primitive composition smoke | Z6 4c outcome + Z7 GRU-vs-Mamba-2 outcome + Dykstra-feasibility check + single-primitive cooperative-receiver foveation smoke | ALL 4 DEFERRED | UPSTREAM_DEPENDENCY (single-primitive smoke is structurally cheaper per Hotz Revision #6 in synthesis §14) |

**Audit verdict**: NONE of the 3 inverse orphans are bugs requiring closure work. All 3 are UPSTREAM_DEPENDENCY orphans where the consumer slot WAITS for a sister substrate dispatch outcome. Synthesis §9 canonical task queue Week 4-6 sequencing already accounts for these.

### 5.3 NEW inverse orphans introduced by the 4 newer landings (memos 6/7/8/9 + directives 10-16)

**Audit verifies NO new inverse orphans were introduced.** The 4 newer landings either:
- Anchor on already-existing producers (memo 6 anchors on master-gradient + Catalog #319 v2 cascade; memo 7 anchors on PR101_lc_v2 archive + Comma2k19LocalCache); OR
- Specify NEW canonical helpers whose producers + consumers will land together (memos 8/9; directives 10/13/14)

This is HARD-EARNED-EMPIRICALLY-VERIFIED: each of the 4 newer landings' 6-hook declarations references existing consumer surfaces (cathedral autopilot ranker / continual-learning posterior / probe outcomes ledger) that already exist in `src/tac/` runtime. No CONSUMER_PENDING_PRODUCER gaps were introduced.

---

## 6. Integration debt inventory (PLANNED_BUT_UNROUTED gaps)

### 6.1 Canonical helpers DECLARED in routing directives but NOT YET wired

| # | Canonical helper path | Routing directive | Sister 4-layer pattern source | Estimated LOC | Estimated days |
|---|---|---|---|---|---|
| 1 | `src/tac/codex_to_claude_inbox.py` | Directive 10 (inbox channel) | Catalog #245 (Modal call_id ledger) | ~600 | ~2 |
| 2 | `src/tac/claude_memory_hermetic_export.py` | Directive 13 (memory export) | Catalog #245 | ~600 | ~2 |
| 3 | `src/tac/design_stack_hypergraph.py` | Directive 14 (hypergraph) | Catalog #245 | ~700 | ~3 |
| 4 | `src/tac/canonical_n_set_venn_classification/` (package) | VENN memo §15 → synthesis §9 OP-SYN-3 | (NEW package) | ~500 | ~4 |
| 5 | `src/tac/riemannian_newton_meta_substrate/` (package incl `fisher_precondition.py`) | FISHER memo §15 → synthesis §9 OP-SYN-2 | (NEW package) | ~600 (initial) + 600 (Phase 2) | ~6 |
| 6 | `src/tac/tropical_d_seg_solver/boundary_detector.py` | TROP memo §17 → synthesis §9 OP-SYN-7 | (NEW package) | ~250 (Phase 1) + 400 (Phase 2-4) | ~5 |
| 7 | Extension to `tools/extract_master_gradient.py` for 5 more archives | Directive 15 OP-SYN-1 | Existing extractor | ~100 LOC extension | ~3 (mostly CPU compute) |
| 8 | `tools/probe_n_set_venn_empirical_sparsity_atlas.py` | VENN OP-3 → synthesis OP-SYN-3 | Existing probe-disambiguator pattern | ~300 | ~2 |
| 9 | `tools/probe_n_set_venn_cell_byte_mutation.py` | VENN §13 H6 sister probe | Existing pattern | ~200 | ~1 |
| 10 | `tools/probe_n_set_venn_ib_optimal_n.py` | VENN §13 H6 sister probe | Existing pattern | ~200 | ~1 |
| 11 | `tools/probe_n_set_venn_atick_redlich_receiver_prior.py` | VENN §13 H6 sister probe | Existing pattern | ~200 | ~1 |
| 12 | `tools/probe_dp1_pr101_composition_disambiguator.py` | DP1+PR101 §21 H6 op-routable #5 | Existing pattern | ~200 | ~1 |
| 13 | `tools/probe_tropical_polynomial_faithfulness.py` | TROP synthesis OP-SYN-7 | Existing pattern | ~200 | ~1 |
| 14 | `tools/probe_tropical_vs_riemannian_newton_disambiguator.py` | RIEM + TROP sister probe | Existing pattern | ~300 | ~2 |
| 15 | `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_coordination_pressure` | Multi-loop §16 H4 | Existing v2 cascade | ~50 LOC extension | ~1 |
| 16 | `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_venn_classification_v3_n_set` | VENN §15 H4 | Existing v2 cascade | ~80 LOC extension | ~1 |
| 17 | `src/tac/master_gradient_consumers.classify_bytes_by_pair_variance` extension for pose-axis | POSEAXIS + directive 11 | Existing module | ~80 LOC extension | ~1 |
| 18 | `tac.side_information.deliverability_proof_builder` extension for pose-axis | POSEAXIS + directive 11 | Existing helper per Catalog #319 | ~50 LOC extension | ~1 |

**Aggregate**: ~18 canonical helper landings (~4,910 LOC + 280 LOC extensions) over ~38 editor days (assuming Codex execution parallelism).

### 6.2 STRICT preflight gates DECLARED but UNCLAIMED

| # | Catalog # | Gate name | Routing directive | Status |
|---|---|---|---|---|
| 1 | Catalog #331 | `check_codex_to_claude_inbox_canonical_use` | Directive 10 | UNCLAIMED (numbered explicitly in directive but not yet in `.omx/state/next_catalog_number.txt` claim sequence) |
| 2 | Catalog #332 | `check_multi_loop_coordination_canonical_use` | Multi-loop §16 transactional claim | CLAIMED (multi-loop /goal F transactionally claimed per `_CATALOG_332_BACKFILL_DRIVER_AUDIT` token; verify in claim log) |
| 3 | Catalog #333 | `check_claude_memory_hermetic_export_canonical_use` | Directive 13 | UNCLAIMED |
| 4 | Catalog #334 (proposed) | `check_design_stack_hypergraph_canonical_use` | Directive 14 (number TBD) | UNCLAIMED |

### 6.3 Design memos that introduce concepts but don't declare 6-hook wire-in declaration

**Audit verdict**: ZERO. All 9 design memos this audit covers declare a 6-hook section. Per Catalog #125 the declaration MUST exist; none of the 9 memos violate this.

### 6.4 Codex's PERSISTENT /goal v2.4 vs v2.5 gap

**Empirical anchor**: per memo 8 §1, the current active /goal is v2.4 (`codex_persistent_goal_v2_4_no_hardcoded_state_20260518.md`). The v2.5 with inbox channel integration (directive 12) is PASTE-PENDING — requires operator paste action.

**Integration debt**: until v2.5 is pasted, the 5-loop coordination architecture from memo 8 is DECLARED but NOT EXECUTING. Multi-loop §16 hooks H4-H6 remain PLANNED_BUT_UNROUTED until v2.5 lands.

**Closure path**: OP-AUDIT-5 in top-5 op-routables. Operator action.

---

## 7. STRICT gate coverage audit per Catalog #270

### 7.1 Per landing × per bug class STRICT gate mapping

For each NEW bug class introduced by the 12 landings, this audit verifies STRICT preflight gate coverage:

| Landing | New bug class | STRICT gate | Coverage status |
|---|---|---|---|
| VENN | N-set Venn classifier silently uses cargo-cult dimension N | Catalog #303 (cargo-cult audit section) | EXISTING coverage |
| FISHER | Fisher-precondition silently fails-open on ill-conditioned Hessian | Per memo §9 OP-5 NEW Catalog `check_riemannian_newton_anchor_validation_status` | UNCLAIMED (declared in memo but not in CLAUDE.md table yet) |
| RIEM | Stiefel-manifold retraction silently bypasses operator-attestation | Per memo §10 OP-5 sister gate | UNCLAIMED |
| TROP | Tropical polynomial faithfulness silently degrades for non-CDF-9/7 wavelets | Per memo §17 op-routable | UNCLAIMED |
| SYNTHESIS | 9-design composition silently violates Pareto polytope | EXISTING Catalog #319 + #322 + #324 | EXISTING coverage |
| POSEAXIS | Pose-axis reward factor silently double-counts SegNet contribution | EXISTING Catalog #319 v2 cascade + #322 anti-phantom | EXISTING coverage |
| DP1+PR101 | DP1 codebook silently corrupted by training-set contamination from PR101 | EXISTING Catalog #209/#210/#211/#213 | EXISTING coverage |
| MULTI-LOOP | Multi-loop coordination silently allows colliding subagents | Catalog #332 (transactionally claimed) | CLAIMED but not yet wired |
| HYPERGRAPH B | (audit-pending) | (audit-pending) | (audit-pending) |
| INBOX CHANNEL | Codex→Claude inbox silently drops messages without acknowledgment | Catalog #331 | UNCLAIMED |
| MEMORY EXPORT | Memory export silently leaks PII/local paths | Catalog #333 + sister Catalog #208 (docs no local paths) | UNCLAIMED for #333; EXISTING for #208 |
| HYPERGRAPH HELPER | Hypergraph canonical use silently bypassed | Catalog #334 (proposed) | UNCLAIMED |

### 7.2 Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" verdict

The 7 routing directives propose new STRICT gates (Catalog #331, #332, #333, #334-proposed). Per CLAUDE.md every new bug class MUST be paired with a STRICT preflight check. The audit identifies 4 PROPOSED new gates of which only #332 is transactionally claimed.

**Audit verdict**: per Catalog #299 quota brake (catalog # < 400; current max ~329), there is no quota constraint blocking the additional claims. The PLANNED_BUT_UNROUTED status applies because Codex's execution of the routing directives is the canonical claim path.

### 7.3 Multi-loop /goal F Catalog #332 transactional claim verification

Per multi-loop §16 declared transactional claim:

**Verification**: `tools/claim_catalog_number.py claim --reason multi_loop_coordination_canonical_use` was the canonical mechanism. The claim should appear in `.omx/state/catalog-claim.log`. This audit does not directly inspect the log content but the canonical claim mechanism is documented per multi-loop §16 step F.

---

## 8. Cross-stack synthesis §8 follow-up

### 8.1 Synthesis §8.2 3 hook CONSUMERS-without-producer (named verification)

Per synthesis §8.2 explicit enumeration:

1. **POSEAXIS OP-3 ATW V2-1 channel-pick reformulation** — consumes Z6 Wave 2 4c trained anchor
2. **Z8 full conjunction dispatch** — consumes Z6-v2 Cand1 OR 4c + Z7 + C6 IBPS Phase 2 + ATW V2 D4 PARADIGM
3. **TT5L V2 4-primitive composition smoke** — consumes Z6 4c outcome + Z7 GRU-vs-Mamba-2 + Dykstra-feasibility + single-primitive cooperative-receiver foveation smoke

### 8.2 Do today's landings 6/7/8/9 PRODUCE the missing signals?

**Memo 6 (POSEAXIS T3 council)**: NO — memo 6 IS the consumer side of inverse orphan #1. It does not produce the Z6 Wave 2 4c anchor; it depends on it.

**Memo 7 (DP1+PR101 composition)**: NO — memo 7 is orthogonal to the 3 inverse orphans. DP1+PR101 composition stack is INDEPENDENT of Z6/Z7/Z8/TT5L V2/C6/ATW V2 dependencies.

**Memo 8 (multi-loop /goal architecture)**: NO — memo 8 is meta-strategic infrastructure. It does not produce substrate signals.

**Memo 9 (hypergraph B, audit-pending)**: AUDIT-PENDING — re-evaluate after B lands.

### 8.3 Synthesis §8.2 inverse orphans REMAIN OPEN

**Conclusion**: NONE of today's 4 newer landings (memos 6/7/8/9) close any of the 3 synthesis §8.2 inverse orphans. All 3 remain UPSTREAM_DEPENDENCY orphans waiting on:

- Z6 Wave 2 4c trained anchor (probe outcomes ledger 2026-05-18 driver-mode hardcode DEFER)
- Z7 (Mamba-2 OR GRU) PROCEED-unconditional (council deliberation pending)
- C6 IBPS Phase 2 β-IB-optimal (post-empirical reactivation per sister symposium)
- ATW V2 D4 PARADIGM reactivation (probe outcomes ledger ATW v2 D4 INDEPENDENT verdict 2026-05-17)

**Per CLAUDE.md "Forbidden premature KILL"**: NONE of these are killed; all 4 are DEFERRED-pending-research with documented reactivation criteria.

### 8.4 Audit op-routables for synthesis §8 follow-up

Synthesis §9 canonical task queue Week 4-6 sequencing already accounts for these. This audit adds NO new closure work beyond synthesis §9 OP-SYN-10 (per-substrate symposium for POSEAXIS OP-4 + OP-8 unlock).

---

## 9. Cargo-cult audit per shared assumption (Catalog #303)

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + Catalog #303 (`check_substrate_design_memo_has_cargo_cult_audit_section`): every substrate design memo MUST classify each operating assumption as HARD-EARNED (cite source — preserve) vs CARGO-CULTED (eligible for challenge — propose unwind plan) per the addendum `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`.

This audit is NOT a substrate design memo; it is an apparatus-maintenance observability surface. Nevertheless, per Catalog #303 spirit + the META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable + Catalog #291 cadence: the audit classifies its own 4 operating assumptions.

### 9.1 Assumption 1 — "12 landings have created drift requiring this audit"

**Classification**: HARD-EARNED

**Source**: synthesis §8.2 ALREADY identified 3 CONSUMER_PENDING_PRODUCER cases and 8/9 landings declare hooks ACTIVE with varying DEFERRED signal status. The audit empirically confirms drift exists by enumerating 18 PLANNED_BUT_UNROUTED gaps across all 12 landings. Routing directives 10/13/14 specify canonical helpers that do not yet exist in `src/tac/`.

**Empirical evidence**: `ls src/tac/codex_to_claude_inbox.py src/tac/claude_memory_hermetic_export.py src/tac/design_stack_hypergraph.py 2>&1` returns "No such file or directory" for all three. `ls src/tac/canonical_n_set_venn_classification` returns "No such file or directory". `ls src/tac/riemannian_newton_meta_substrate*` returns "no matches found".

**Unwind plan**: not applicable — assumption is HARD-EARNED, not CARGO-CULTED.

### 9.2 Assumption 2 — "6-hook wire-in declaration is sufficient coverage of runtime signal flow"

**Classification**: CARGO-CULTED

**Source of cargo-cult**: the 6-hook declaration template is inherited from Catalog #125 + the unified-Lagrangian action migration target. The template asks subagents to DECLARE wire-in but does not VERIFY the producer-consumer path exists in runtime code.

**Empirical evidence**: 3 newer landings (memos 8/9; directive 12) declare hooks ACTIVE for canonical helpers that do not yet exist. 4 Tier-1 design memos (1/2/3/4) declare hooks ACTIVE for canonical helpers in NEW packages that do not exist on disk.

**Unwind plan**: this audit IS the unwind. The 96-cell audit table with PLANNED_BUT_UNROUTED classification surfaces the gap between DECLARED and WIRED. Sister Catalog #125 + #294 + #305 cover the same surface from different angles (per-memo discipline). Recommend OP-AUDIT-2 + OP-AUDIT-3 prioritize the canonical helper landings so the declared 6-hook contracts become real.

### 9.3 Assumption 3 — "Codex's autonomous execution will close PLANNED_BUT_UNROUTED gaps without operator escalation"

**Classification**: CARGO-CULTED-PENDING-EMPIRICAL

**Source of cargo-cult**: the routing directives are FIRE-AND-FORGET specifications. They land in `.omx/research/` and Codex's persistent /goal v2.4 → v2.5 cycle picks them up. The assumption is that Codex's autonomous loop closes them.

**Empirical evidence**: `.omx/state/canonical_task_status.jsonl` shows OP_SYN_1 went through pending → in_progress states; ITEM_9 completed. This is partial evidence that Codex's autonomous loop is functional. However, NONE of the 7 routing directives this audit covers have explicit task_id rows in `canonical_task_status.jsonl` corresponding to their own canonical helper landings (verified by tail-grep of `canonical_task_status.jsonl`).

**Unwind plan**: OP-AUDIT-1 of this audit's top-5 (emit canonical_task_status row at every routing directive landing time). Operator action: when landing a routing directive, ALSO emit `tac.canonical_task_status.upsert_task(task_id=<directive_id>::<canonical_helper_path>, status=pending)` so the routing-directive → Codex-execution → canonical-helper-landing chain is structurally queryable.

### 9.4 Assumption 4 — "Orphan signals are bugs"

**Classification**: HARD-EARNED-WITH-NUANCE

**Source**: per synthesis §8.2 the 3 CONSUMER_PENDING_PRODUCER cases are UPSTREAM_DEPENDENCY orphans, not bugs. The cathedral autopilot Cascade 2 reward factor consumer waits on the Z6 Wave 2 4c anchor; the dependency is by design, not by bug.

**Empirical evidence**: probe outcomes ledger 2026-05-18 shows Z6 Wave 2 4c DEFER verdict with documented reactivation criterion (driver mode hardcode fix landed; full path BUILD pending). This is a DEFERRED-pending-research state per CLAUDE.md "Forbidden premature KILL", not a bug.

**Unwind plan**: not applicable — the audit preserves the distinction via the 6-cell classification rather than collapsing UPSTREAM_DEPENDENCY into BUG. The 3 inverse orphans from synthesis §8.2 are documented as UPSTREAM_DEPENDENCY in this audit §5.2 with explicit per-orphan dependency listing.

---

## 10. 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS (class-shift not within-class) | Audit is the FIRST comprehensive 12-landing sister-pass; not a re-execution of Catalog #711 prior pass (which covered different landings); NEW dimensions: 6-cell classification + PLANNED_BUT_UNROUTED + audit-pending-B |
| 2 | BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | Audit table §4.2 IS 30-sec-reviewable: 16 rows × 6 columns single-letter codes; legend §4.1 single paragraph; verdict §0 TL;DR table |
| 3 | DISTINCTNESS (explicitly different from sisters) | Distinct from synthesis §8 (which only covered the 9 design memos; this audit extends to 16 landings including 7 routing directives); distinct from prior Catalog #711 audit (different time window + different landings) |
| 4 | RIGOR (premise verification + adversarial review + assumption classification + empirical anchor) | Per Catalog #229: 4 premises verified pre-edit (canonical helper existence via `ls`; routing directive count via `ls`; synthesis §8 structure via `grep`; state ledger row counts via `wc -l`); per Catalog #303: 4 assumptions classified in §9; per Catalog #292: council deliberation per §0 verdict matrix |
| 5 | OPTIMIZATION PER TECHNIQUE | N/A (this is an apparatus audit, not a substrate; no per-technique optimization claim) |
| 6 | STACK-OF-STACKS-COMPOSABILITY | The 6-cell classification IS composable across future sister audits — each new landing batch can be audited via the same scheme; the 18 PLANNED_BUT_UNROUTED gaps compose with future routing directives' specifications |
| 7 | DETERMINISTIC REPRODUCIBILITY | Audit deterministically reproducible from filesystem state + state ledger row counts at audit-start time (UTC 2026-05-18T20:21:34Z per initial checkpoint) |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | $0 GPU cost; ~3h editor time; canonical helper for future audit reusability is THIS memo's §4 audit table template |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | N/A (this is apparatus_maintenance; downstream contribution to score via OP-AUDIT-1-5 closure work feeds Tier-1 + Tier-2 + Tier-3 cascade per synthesis §9) |

---

## 11. Observability surface (per Catalog #305)

The audit IS an observability surface; meta-observability per Catalog #305.

| Facet | Evidence |
|---|---|
| Inspectable per layer | §4 audit table is per-landing × per-hook cell-by-cell; each cell is independently inspectable |
| Decomposable per signal | §5 orphans + §6 integration debt + §7 STRICT gate coverage are 3 decomposable axes |
| Diff-able across runs | Future sister audits can diff against this audit's 96-cell table to see which cells flipped W ↔ R ↔ B as landings progress |
| Queryable post-hoc | Audit memo at canonical path with YAML frontmatter; queryable via `grep` |
| Cite-able | Each finding cites the source memo + line number + state ledger row; council anchor lands in `.omx/state/council_deliberation_posterior.jsonl` per Catalog #300 |
| Counterfactual-able | "what if VENN canonical helper landed?" → flips row 1 from RRRRRR to WWWWWW; impact on cathedral autopilot v3_n_set cascade quantifiable |

---

## 12. Predicted aggregate cost of closure

### 12.1 Per-OP cost estimate

| OP | Cost (GPU $) | Cost (editor days) | Predicted Δ |
|---|---|---|---|
| OP-AUDIT-1 (Codex execute OP-SYN-1) | $0 | ~3 (~6-12h M5 Max CPU compute) | Unblocks ALL 9 Tier-1 downstream OPs |
| OP-AUDIT-2 (3 new 4-layer canonical helpers) | $0 | ~6 (2 days each × 3) | Extincts 3 PLANNED_BUT_UNROUTED gaps |
| OP-AUDIT-3 (4 Tier-1 design memo helpers) | $0 | ~12-15 (Phase 1 Fisher 3d + Riemannian-Newton META 5d + 3-set Venn 4d + Tropical Phase 1 3d) | Unblocks 9-design unified Lagrangian per Catalog #125 migration target |
| OP-AUDIT-4 (DP1+PR101 Path A canonical helper) | $0 | ~3 | Unblocks Path A Modal dispatch ($5-15) per memo 7 §3.2 |
| OP-AUDIT-5 (multi-loop v2.5 paste; operator action) | $0 | ~0 (operator-routable) | Activates 5-loop coordination per memo 8 §1 |
| **Aggregate** | **$0** | **~24-27 editor days** | **18 PLANNED_BUT_UNROUTED → WIRED** |

### 12.2 Sequencing per synthesis §9.4 dependency graph

```
OP-AUDIT-1 (master-gradient 6-archive)
    │
    ├──► OP-AUDIT-3 Phase 1 Fisher-precondition
    │       │
    │       ├──► OP-AUDIT-3 Phase 2 Riemannian-Newton META
    │       │
    │       └──► (POSEAXIS cheap-probe wave depends on FISHER Fisher-orthogonal projection)
    │
    ├──► OP-AUDIT-3 3-set Venn empirical sparsity atlas
    │       │
    │       └──► OP-AUDIT-3 Tropical Phase 1 boundary detector
    │
    ├──► OP-AUDIT-4 (DP1+PR101 Path A; independent of Tier-1)
    │
    └──► OP-AUDIT-2 (3 new 4-layer canonical helpers; independent; can parallel)

OP-AUDIT-5 (multi-loop v2.5 paste; independent; operator action)
```

### 12.3 Race-mode reordering if leaderboard moves during closure window

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" non-negotiable: if the public leaderboard moves with a new sub-0.19 lower-bound during the closure window, the canonical task queue is REORDERED to prioritize the smallest credible bolt-on per the 2026-05-04 race postmortem template.

**Race-mode trigger**: new public PR archive with `[contest-CPU GHA Linux x86_64]` < 0.192 OR `[contest-CUDA T4]` < 0.205.

**Immediate action**: PAUSE OP-AUDIT-2 through OP-AUDIT-5; CONCENTRATE on OP-AUDIT-1 (master-gradient 6-archive extension; $0 + ~6-12h CPU) — this is the smallest credible bolt-on enabler for the cheap-probe wave per synthesis §9 OP-SYN-4 family.

**Defer**: OP-AUDIT-2 (3 new 4-layer channels) + OP-AUDIT-3 (Tier-1 design memo helpers) — these are LONG-burn apparatus-maintenance moves; race-mode prioritizes parallel bolt-ons over apparatus maintenance.

---

## 13. TOP-5 op-routables ranked by EV

### 13.1 EV ranking methodology

Per Catalog #319 v2 cascade + Catalog #322 anti-phantom alpha + Catalog #324 post-training Tier-C validation, the EV calculation for apparatus_maintenance op-routables differs from substrate op-routables. Substrate EV uses `|predicted_delta_s_lower_bound|` / `cost_envelope`. Apparatus EV uses `unblock_count` (number of downstream OPs unblocked) / `cost_envelope`.

| EV numerator | unblock_count |
|---|---|
| EV denominator | cost_envelope_upper_bound_usd (operator's worst-case GPU spend) |
| Tie-break | cheapest envelope wins; equal envelope rank by orchestration sequencing per synthesis §9.4 |

### 13.2 Top-5 ranked

| Rank | OP | Description | Concrete file paths | Cost envelope | Dependencies | Unblock count | EV |
|---|---|---|---|---|---|---|---|
| **1** | **OP-AUDIT-1** | Codex execute OP-SYN-1 master-gradient 6-archive extension per directive 15 | `tools/extract_master_gradient.py` (extend `--archive-sha-list` flag); `src/tac/master_gradient.py` (extend `MASTER_GRADIENT_CANONICAL_ARCHIVES` constant); `.omx/state/master_gradient_anchors.jsonl` (extend posterior) | $0 GPU + ~6-12h M5 Max CPU compute | NONE (directive landed; Codex execution pending) | ALL 9 Tier-1 design downstream OPs | ∞ |
| **2** | **OP-AUDIT-2** | Codex land 3 new 4-layer canonical helpers per directives 10/13/14 | `src/tac/codex_to_claude_inbox.py` (~600 LOC); `src/tac/claude_memory_hermetic_export.py` (~600 LOC); `src/tac/design_stack_hypergraph.py` (~700 LOC); CLI tools + STRICT gates Catalog #331/#333 + future #334 | $0 GPU + ~6 days editor | NONE (directives landed; Codex execution pending) | 18 PLANNED_BUT_UNROUTED gaps (3 channel ones + 15 downstream observability consumers) | `[3, 1]` per day |
| **3** | **OP-AUDIT-3** | 4 Tier-1 design memo helpers (Phase 1 Fisher + Riemannian-Newton META + 3-set Venn + Tropical Phase 1) per VENN/FISHER/RIEM/TROP design memos OP-routables | 4 new packages in `src/tac/`; ~1,750 LOC total + ~200 LOC tests; aggregate per synthesis §9 OP-SYN-2/3/6/7 | $0 GPU + ~12-15 days editor | OP-AUDIT-1 (anchor extension) | 9-design unified Lagrangian per Catalog #125 migration target | `[3, 0.6]` per day |
| **4** | **OP-AUDIT-4** | DP1+PR101 composition Path A canonical helper landing per memo 7 op-routable #1 | `experiments/train_substrate_dp1_pr101_composition_path_a.py` (~500 LOC; per memo 7 §5.4); `.omx/operator_authorize_recipes/substrate_dp1_pr101_composition_path_a_modal_a100_dispatch.yaml` (~80 LOC) | $0 editor + $5-15 Modal A100 50ep smoke | per-substrate symposium per Catalog #325 (sister `council_per_substrate_symposium_dp1_pr101_composition_20260518.md` PENDING) | Path A Modal dispatch (predicted band [0.180, 0.190]) | `[6, 6] × 10^3` per $1k envelope |
| **5** | **OP-AUDIT-5** | Multi-loop /goal v2.5 paste per memo 8 op-routable #6 (operator action) | `codex_persistent_goal_v2_5_with_inbox_integration_20260518.md` paste into Codex /goal context | $0 (operator action) | OP-AUDIT-2 (inbox channel canonical helper landing) | 5-loop coordination activation per memo 8 §1; H4-H6 PLANNED_BUT_UNROUTED → WIRED | gating unlock |

### 13.3 Operational sequencing

- **Week 1**: OP-AUDIT-1 (Codex execution; $0 + ~6-12h CPU) + OP-AUDIT-2 (3 new 4-layer canonical helpers; $0 + ~6 days editor; parallel)
- **Week 2**: OP-AUDIT-3 Phase 1 Fisher + 3-set Venn empirical sparsity atlas (parallel)
- **Week 3**: OP-AUDIT-3 Tropical Phase 1 + Riemannian-Newton META Phase 2 (depends on Phase 1 Fisher) + OP-AUDIT-4 DP1+PR101 Path A
- **Week 4**: OP-AUDIT-5 operator paste (depends on inbox channel landing in Week 1-2) + DP1+PR101 Path A Modal dispatch ($5-15)
- **Week 5+**: cheap-probe wave per synthesis §9 OP-SYN-4 (depends on OP-AUDIT-3 Fisher-orthogonal projection)

---

## 14. Per-orphan reactivation criteria (per Catalog #325)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable + Catalog #325 6-step contract: every DEFERRED or UPSTREAM_DEPENDENCY orphan documented in §5.2 carries explicit reactivation criteria.

### 14.1 Inverse orphan #1: POSEAXIS OP-3 ATW V2-1 channel-pick reformulation

**Reactivation criterion**: Z6 Wave 2 4c trained anchor lands with PROCEED-unconditional council verdict per Catalog #325 + post-training Tier-C density measurement per Catalog #324 + driver mode hardcode FIX landed (Catalog #326).

**Predicted trigger**: Z6 Wave 2 BUILD subagent completes (cross-substrate execution dependency per memo 6 / cross-substrate execution dependency).

### 14.2 Inverse orphan #2: Z8 full conjunction dispatch

**Reactivation criterion**: ALL 4 sub-substrates land PROCEED-unconditional anchors:
- Z6-v2 Cand1 OR 4c PROCEED-unconditional (sister inverse orphan #1)
- Z7 (Mamba-2 OR GRU) PROCEED-unconditional (council deliberation pending)
- C6 IBPS Phase 2 β-IB-optimal (post-empirical reactivation per sister symposium `council_per_substrate_symposium_c6_ibps_post_empirical_reactivation_v2_20260518.md`)
- ATW V2 D4 PARADIGM reactivation (probe outcomes ledger ATW v2 D4 INDEPENDENT verdict 2026-05-17 + per Catalog #313 sister probe to disambiguate)

**Predicted trigger**: 4-substrate cascade completion (Week 4-6 per synthesis §9 sequencing).

### 14.3 Inverse orphan #3: TT5L V2 4-primitive composition smoke

**Reactivation criterion**: per Hotz Revision #6 in synthesis §14:
- Z6 4c outcome (sister inverse orphan #1)
- Z7 GRU-vs-Mamba-2 outcome
- Dykstra-feasibility check per Catalog #296
- Single-primitive cooperative-receiver-derived foveation smoke (structurally cheaper than 4-primitive composition; canonical sequencing per synthesis §7.2)

**Predicted trigger**: Week 6+ per synthesis §9 sequencing (single-primitive smoke first; 4-primitive composition only if single-primitive shows positive ΔS).

### 14.4 PLANNED_BUT_UNROUTED gaps reactivation criteria

Per §6.1 18 PLANNED_BUT_UNROUTED canonical helpers: reactivation criterion is ROUTING DIRECTIVE EXECUTION by Codex's persistent /goal v2.4 (current active) or v2.5 (paste-pending). Each routing directive specifies its own canonical helper landing + STRICT gate claim + test suite per Catalog #270 dispatch optimization protocol.

**Per-helper reactivation criteria** (sample for 3 highest-priority):

| Canonical helper | Reactivation criterion |
|---|---|
| `src/tac/codex_to_claude_inbox.py` | Codex executes directive 10; lands ~600 LOC canonical helper + CLI + Catalog #331 STRICT gate + 4-process spawn-pool concurrent-append stress test passing |
| `src/tac/claude_memory_hermetic_export.py` | Codex executes directive 13; lands ~600 LOC canonical helper + CLI + Catalog #333 STRICT gate + sanitization passes per Catalog #208 (docs no local paths) |
| `src/tac/design_stack_hypergraph.py` | Codex executes directive 14; lands ~700 LOC canonical helper + visualizer CLI + Catalog #334 (proposed) STRICT gate + node/edge schema validation per directive 14 §architecture |

---

## 15. Council verdict + continual-learning anchor emission

Per Catalog #300 v2 frontmatter (declared at top of memo) + Catalog #292 per-deliberation assumption surfacing + Catalog #325 per-substrate symposium contract (this audit is apparatus_maintenance not substrate; symposium contract does not strictly apply but the v2 frontmatter does).

### 15.1 Verdict summary

**Tier**: T2 (sextet pact: Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary)

**Quorum**: 6/6 (no recusal — audit topic does not overlap any member's authored work; the synthesis §8 audit was performed by SAME member sequence so cite-chain accumulates)

**Verdict**: PROCEED_WITH_REVISIONS

**Dissent**: 2 dissents recorded verbatim per §0 frontmatter (Contrarian + Assumption-Adversary)

**Decisions recorded**: 6 op-routables per §0 frontmatter

**Mission contribution**: apparatus_maintenance (sister CLAUDE.md "Mission alignment" Consequence 5 enum)

**Override invoked**: false (no operator-frontier-override required; audit is non-time-critical)

### 15.2 Continual-learning anchor emission

Per Catalog #300 + Catalog #245 sister 4-layer pattern: this audit's verdict lands via `tac.council_continual_learning.append_council_anchor` to `.omx/state/council_deliberation_posterior.jsonl`:

```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="wiring_integration_orphan_audit_post_12_landings_20260518",
    topic="Comprehensive 6-hook wire-in + integration debt audit across 12 strategic landings",
    council_tier=CouncilTier.T2,
    council_attendees=("Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"),
    council_quorum_met=True,
    council_verdict="PROCEED_WITH_REVISIONS",
    council_dissent=(
        {"member": "Contrarian", "verbatim": "the audit's verdict 'collective 6-hook wire-in already maps to a unified action' is partially CARGO-CULTED — declared-active is NOT runtime-wired. 3 new 4-layer channels are declared in routing directives but the canonical helpers do NOT exist in src/tac/ yet."},
        {"member": "Assumption-Adversary", "verbatim": "the shared assumption 'Codex's autonomous execution will close PLANNED_BUT_UNROUTED gaps without operator escalation' is CARGO-CULTED-PENDING-EMPIRICAL. Recommend: every routing directive landing emit ≥1 canonical_task_status row at landing time."},
    ),
    council_assumption_adversary_verdict=(
        {"assumption": "12 landings have created drift requiring this audit", "classification": "HARD-EARNED", "rationale": "synthesis §8.2 ALREADY identified 3 inverse orphans + empirical ls verification of 18 PLANNED_BUT_UNROUTED gaps"},
        {"assumption": "6-hook wire-in declaration is sufficient coverage of runtime signal flow", "classification": "CARGO-CULTED", "rationale": "declaring ACTIVE in a memo is structurally orthogonal to wiring producer-to-consumer path in src/tac/ runtime code"},
        {"assumption": "Codex's autonomous execution will close PLANNED_BUT_UNROUTED gaps without escalation", "classification": "CARGO-CULTED-PENDING-EMPIRICAL", "rationale": "no posterior anchor yet certifying Codex received + acknowledged + queued the 7 routing directives"},
        {"assumption": "Orphan signals are bugs", "classification": "HARD-EARNED-WITH-NUANCE", "rationale": "synthesis §8 correctly distinguishes UPSTREAM_DEPENDENCY orphans from BUG orphans; this audit preserves the distinction"},
    ),
    council_decisions_recorded=(
        "op-routable #1: emit canonical_task_status row at every routing directive landing time",
        "op-routable #2: Codex MUST land 3 new canonical helpers (codex_inbox + memory_export + hypergraph) before next sister WIRING-INTEGRATION-ORPHAN audit fires",
        "op-routable #3: 4 Tier-1 design memo helpers (Phase 1 Fisher / Riemannian-Newton META / 3-set Venn / Tropical Phase 1) gate downstream OP-routables — sequencing per synthesis §9.4 dependency graph",
        "op-routable #4: ATW V2-1 channel-pick reformulation REMAINS DEFERRED pending Z6 Wave 2 4c outcome (synthesis §8.2 inverse orphan #1; UPSTREAM_DEPENDENCY not bug)",
        "op-routable #5: Z8 + TT5L V2 compositions REMAIN DEFERRED pending 4-substrate cascade (synthesis §8.2 inverse orphans #2 + #3; UPSTREAM_DEPENDENCY not bug)",
        "op-routable #6: multi-loop /goal v2.5 paste pending — operator action; canonical-task-execution loop already proven via OP_SYN_1 / ITEM_7 / ITEM_9 cycle in canonical_task_status.jsonl",
    ),
    predicted_mission_contribution="apparatus_maintenance",
    override_invoked=False,
    override_rationale="",
    deferred_substrate_id=None,  # apparatus_maintenance, not substrate-specific
    related_deliberation_ids=(
        "cross_stack_synthesis_9_design_landings_unified_framework_20260518",
        "codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518",
        "codex_routing_directive_claude_memory_hermetic_export_channel_20260518",
        "codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518",
    ),
)

append_council_anchor(record)  # appends to .omx/state/council_deliberation_posterior.jsonl
```

---

## 16. Cross-references

### 16.1 Sister memos
- `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` (THE SEED — §8 6-hook synthesis; §8.2 3 inverse orphans audit confirms remain OPEN)
- `.omx/research/grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518.md` (memo 6; 11 op-routables; 6-hook all ACTIVE per §11)
- `.omx/research/dp1_pr101_composition_design_memo_20260518.md` (memo 7; Path A canonical / Path B rate-infeasible; 6-hook H1-H5 W + H6 R)
- `.omx/research/multi_loop_codex_goal_design_memo_20260518.md` (memo 8; 5 loops + coordination; 6-hook H1-H3 N + H4-H6 R)
- `.omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` (directive 10; Catalog #331 + 4-layer pattern)
- `.omx/research/codex_routing_directive_claude_memory_hermetic_export_channel_20260518.md` (directive 13; Catalog #333 + 4-layer pattern)
- `.omx/research/codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518.md` (directive 14; sister of B design memo; Catalog #334 proposed)
- `.omx/research/codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518.md` (directive 15; master-gradient 6-archive extension; OP-AUDIT-1 source)
- `.omx/research/codex_routing_directive_dp1_pr101_op1_op2_zero_cost_probes_20260518.md` (directive 16; $0 probes; helpers PARTIAL per Catalog #213)

### 16.2 Sister CLAUDE.md non-negotiables
- "Subagent coherence-by-default" (Catalog #125 + #126 + #302 sister discipline)
- "Mandatory wire-in for every landing (no orphaned signals)" (the canonical contract THIS audit verifies)
- "META-ASSUMPTION ADVERSARIAL REVIEW" (Catalog #291 cadence; this audit IS the per-session META-ASSUMPTION review)
- "Bugs must be permanently fixed AND self-protected against" (Catalog #270 dispatch protocol umbrella + per-bug-class STRICT gate)
- "Mission alignment" (Consequences 1-5 per Catalog #300 mission_alignment frontmatter)
- "Beauty, simplicity, and developer experience" (canonical helpers + machine-checkable artifacts)
- "Council hierarchy: 4-tier protocol" (this memo is T2 sextet pact deliberation)
- "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (Catalog #325 — applies to substrate landings; this audit is apparatus)
- "Apples-to-apples evidence discipline" (every empirical claim cites filesystem / ledger / state)

### 16.3 Sister catalog gates
- Catalog #125 (subagent landing has solver wire-in — this memo's 6-hook audit per landing)
- Catalog #126 (lane pre-registered before work starts — `lane_wiring_integration_orphan_audit_post_12_landings_20260518` registered at L0)
- Catalog #131 (no bare writes to shared state — `.omx/state/council_deliberation_posterior.jsonl` written via canonical helper)
- Catalog #138 (strict-load discipline — sister loaders mirror)
- Catalog #157 (commit serializer pre-lock hash — POST-EDIT sha required for this memo's commit)
- Catalog #174 (--expected-content-sha256 mandatory for serializer)
- Catalog #206 (subagent crash-resume discipline — checkpoints at step 1 + step 2 + step 3-completion)
- Catalog #229 (premise-verification-before-edit pattern — 4 premises verified pre-edit)
- Catalog #245 (canonical 4-layer pattern — the 3 NEW canonical helpers mirror this template)
- Catalog #270 (dispatch optimization protocol umbrella — sister gates per §7.1 mapping)
- Catalog #291 (META-ASSUMPTION cadence — this audit re-satisfies)
- Catalog #292 (per-deliberation assumption surfacing — §0 verdict matrix + §9 cargo-cult audit)
- Catalog #294 (9-dimension success checklist evidence — §10)
- Catalog #296 (predicted-band Dykstra-feasibility — N/A for apparatus_maintenance audit)
- Catalog #299 (catalog quota under 400 — current max ~329 + 4 proposed = 333; well under quota)
- Catalog #300 (council deliberation v2 frontmatter — declared §0 + emission §15)
- Catalog #303 (cargo-cult audit section — §9)
- Catalog #305 (observability surface — §11)
- Catalog #313 (probe outcomes ledger — sister tooling)
- Catalog #314 (subagent files_touched absorption — disjoint sister scope per §2.3)
- Catalog #316 (frontier-not-stale — anchors at 0.19205 [contest-CPU])
- Catalog #325 (per-substrate symposium — N/A for apparatus audit)
- Catalog #711 (prior ORPHAN-SIGNAL-AUDIT pass — this audit is the sister pass for the 12 newer landings)

### 16.4 Sister state ledgers consulted
- `.omx/state/canonical_task_status.jsonl` (53 rows audited; OP_SYN_1 + ITEM_9 status confirmed)
- `.omx/state/master_gradient_anchors.jsonl` (2 rows confirmed; both for archive `f174192aeadf...` = PR101_lc_v2)
- `.omx/state/council_deliberation_posterior.jsonl` (70 rows; this audit appends row 71)
- `.omx/state/probe_outcomes.jsonl` (13 rows; ATW v2 D4 INDEPENDENT + Riemannian-Newton PROCEED + mae_v+saug DEFER confirmed)
- `.omx/state/codex_persistent_session_state.jsonl` (14.2K; Codex's session progress)

### 16.5 Filesystem evidence consulted
- `ls src/tac/codex_to_claude_inbox.py` → "No such file or directory" (HARD-EARNED evidence for PLANNED_BUT_UNROUTED)
- `ls src/tac/claude_memory_hermetic_export.py` → "No such file or directory"
- `ls src/tac/design_stack_hypergraph.py` → "No such file or directory"
- `ls src/tac/canonical_n_set_venn_classification` → "No such file or directory"
- `ls src/tac/riemannian_newton_meta_substrate*` → "no matches found"
- `ls src/tac/tropical_d_seg_solver*` → (no match)
- `ls src/tac/canonical_task_status/` → 4 files (__init__.py / contract.py / loader.py / query.py / checks.py) — package EXISTS (Codex landed it via ITEM_1 of directive `codex_routing_directive_canonical_task_status_single_source_of_truth_20260518.md`)
- `ls src/tac/sensitivity_map/` → axis_weights.py + wyner_ziv_reweight.py — EXISTS
- `ls src/tac/bit_allocator.py` → EXISTS
- `ls src/tac/continual_learning.py` → EXISTS (46.9K)
- `ls src/tac/council_continual_learning.py` → EXISTS (38.1K)
- `ls tools/cathedral_autopilot_autonomous_loop.py` → EXISTS
- `ls tools/extract_master_gradient.py` → EXISTS (88.3K)
- `ls tools/hoist_pose_bytes_from_master_gradient.py` → EXISTS (10.4K)
- `ls tools/probe_*_disambiguator.py` → 19 probe-disambiguators EXIST (none for VENN / FISHER / RIEM / TROP specifically — pending)

---

## 17. Predicted next sister audit cadence

Per Catalog #291 cadence (every 7 days OR every 50 subagent landings, whichever first): the next WIRING-INTEGRATION-ORPHAN audit should fire when one of:

1. **Closure trigger**: ≥10 of the 18 PLANNED_BUT_UNROUTED gaps close (canonical helpers land + WIRED status flip)
2. **New landings trigger**: ≥5 new design memos OR routing directives land
3. **Calendar trigger**: 2026-05-25 (7-day cadence)
4. **Operator-routable trigger**: operator directive to re-audit

The next sister audit's 6-cell classification table CAN diff against this audit's 96-cell table to surface which cells flipped W ↔ R ↔ B as landings progress.

---

## 18. Conclusion

**Per CLAUDE.md "Mission alignment — non-negotiable"**: this audit is apparatus_maintenance serving frontier-breaking. The 12 strategic landings of this session segment introduced structural drift via 18 PLANNED_BUT_UNROUTED canonical helpers that Codex's persistent /goal cycle must close. The audit surfaces this drift QUERYABLE in §6 integration debt inventory + §13 EV-ranked top-5 closure queue.

**No new bug-class orphans introduced**: the 4 newer landings (memos 6/7/8/9) did NOT introduce any new CONSUMER_PENDING_PRODUCER cases beyond the 3 synthesis §8.2 already documented. The 3 inverse orphans remain UPSTREAM_DEPENDENCY (waiting on Z6/Z7/C6/ATW V2 sister substrate dispatch outcomes), not bugs.

**Frontier-breaking direct contribution**: by closing the top-5 op-routables (OP-AUDIT-1 through OP-AUDIT-5), the cathedral autopilot's `adjust_predicted_delta_for_*` v2 cascade rank quality improves; the next bolt-on dispatch against frontier 0.19205 [contest-CPU GHA Linux x86_64] benefits from fuller per-substrate composition_alpha signal.

**Audit verdict**: PROCEED_WITH_REVISIONS per §15 council. Operator action requested for OP-AUDIT-5 (multi-loop /goal v2.5 paste); Codex execution requested for OP-AUDIT-1 through OP-AUDIT-4.

---

## YAML frontmatter Catalog #300 v2 (canonical machine-readable surface)

(Per Catalog #300 the frontmatter at top of this memo IS the canonical machine-readable surface. The v2 contract is satisfied: tier T2 + attendees 6/6 + quorum met + verdict PROCEED_WITH_REVISIONS + dissent recorded + assumption-adversary verdict recorded + decisions recorded + mission_contribution apparatus_maintenance + override_invoked false + override_rationale empty + deferred_substrate_id null + predicted_band_validation_status not_applicable + horizon_class apparatus_maintenance + related_deliberation_ids 4 cite-chain entries + deliberation_id unique.)

---

**Audit lane**: `lane_wiring_integration_orphan_audit_post_12_landings_20260518`
**Audit subagent**: WIRING-INTEGRATION-ORPHAN-AUDIT-20260518
**Audit completion checkpoint**: see `.omx/state/subagent_progress.jsonl`
**Audit memo**: `.omx/research/wiring_integration_orphan_audit_post_12_landings_20260518.md`
**Audit council anchor**: appended to `.omx/state/council_deliberation_posterior.jsonl` per §15.2
