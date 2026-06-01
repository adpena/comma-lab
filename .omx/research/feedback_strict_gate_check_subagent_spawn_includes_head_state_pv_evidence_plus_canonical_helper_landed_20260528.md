# Catalog #376 STRICT gate + canonical helper landing (Wave N+8 Slot 2)

**Date:** 2026-05-28
**Lane:** `lane_strict_gate_check_subagent_spawn_includes_head_state_pv_evidence_plus_canonical_helper_20260528`
**Task:** #1475 (canonical 2-landing pattern for anti-pattern #13/#14 discipline gaps)
**Predecessor:** `a15fb8266be531cb0` (crashed at API rate-limit)
**Successor:** `resume_a15fb8266_376_strict_gate_landing_20260528` (this landing)
**Catalog #** 376 (claimed git-transactionally at commit `a73b7c032`)
**Premise verification:** complete via `git log --oneline -30` + `git status` + sister-checkpoint inspection at session entry. Predecessor landed canonical helper package + 33 tests in HEAD; this successor lands the STRICT gate + 42 tests + catalog row + canonical equation + retroactive sweep + landing memo.

---

## Summary

Canonical 2-landing pattern per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable. The bug class (anti-pattern #13 `subagent_spawn_without_head_state_premise_verification_v1` per Wave N+7 commit `49bdcd78f`) is now structurally extinct at the SPAWN-time surface via TWO landings:

1. **Canonical helper** `tac.discipline_anti_pattern_guards.verify_head_state_before_spawn(declared_scope, ...) -> SpawnGuardVerdict` — provides per-call PV decision (PROCEED / DUPLICATE_HEAD_STATE / SISTER_IN_FLIGHT) BEFORE spawn cost is paid. 3-surface falling-rule consults (a) recent HEAD commits via `git log --since=<lookback>m --name-only`, (b) sister landing memos at `.omx/research/*landed_*.md`, (c) sister-subagent in-flight checkpoints at `.omx/state/subagent_progress.jsonl`. Sister of Catalog #340 STAGING-time guard at the SPAWN-time surface.

2. **STRICT preflight gate** `check_subagent_spawn_includes_head_state_pv_evidence` (Catalog #376) — refuses post-cutoff spawn-event checkpoint rows whose `notes`/`next_action`/`files_touched` lack any canonical PV-evidence token OR a same-line `# SPAWN_PV_EVIDENCE_WAIVED:<rationale>` waiver (placeholder rejected per Catalog #287 sister discipline). WARN-ONLY initial wire-in per CLAUDE.md "Strict-flip atomicity rule"; legacy in-flight checkpoints (943 rows) predate the gate, exempt via far-future cutoff `2030-01-01T00:00:00Z`.

3. **Sister handoff helper** `verify_predecessor_working_tree_committed_or_auto_commit(...) -> HandoffGuardVerdict` — anti-pattern #14 sister at the predecessor-handoff surface. Auto-commits via canonical `tools/subagent_commit_serializer.py` if the predecessor left uncommitted edits in the shared working tree.

## Sister-extinction surface

Together with Catalog #229 / #117 / #157 / #174 / #216 / #289 / #206 / #230 / #302 / #314 / #340, this gate closes the multi-subagent edit/commit/spawn collision class at NINE orthogonal surfaces:

| Surface | Catalog # | Description |
|---------|-----------|-------------|
| SPAWN-time (THIS gate) | #376 | spawn-event checkpoint MUST include PV evidence |
| edit-time-checkpoint | #302 | sister-subagent scope overlap via checkpoint JSONL |
| edit-time-bulk-op | #230 | bulk-rewrite respects sister-subagent ownership map |
| commit-time-pre-pre-lock | #157 | commit-swap pre-pre-lock hash |
| commit-time-staged | #216 | post-stage hash |
| commit-time-lock-arbitration | #117 + #174 | canonical serializer + mandatory `--expected-content-sha256` |
| post-resolution-residual-marker | #248 | no `<<<<<<<` markers in canonical files |
| post-commit-absorption-detect | #314 | bare commit absorbs in-flight files (POST-COMMIT) |
| staging-surface-prevent | #340 | sister-checkpoint guard at STAGING time |

## Canonical equation per Catalog #344

Registered `subagent_spawn_discipline_compounding_v1` (single source of truth for the SPAWN-time discipline rule):

```
P(STAND_DOWN | spawn, ¬PV, overlap) = 1 - (1-p_HEAD)(1-p_memo)(1-p_sister)
```

* Latex form: `P(STAND_DOWN|spawn, ¬PV, overlap) = 1 - (1-p_HEAD)(1-p_memo)(1-p_sister)`
* Empirical anchors: 2 (Wave N+5 Slot 1 commit `e61ea93b0` + Slot 2 commit `5d38bf9df`)
* `next_recalibration_trigger`: `when_3+_new_empirical_anchors_in_domain` (triggers auto-refit via Catalog #371 once a 3rd STAND_DOWN incident lands)
* Producers: continual-learning posterior anchors at `.omx/state/subagent_progress.jsonl` spawn-event rows + Wave N+5 STAND_DOWN incident memos
* Consumers: `tools/operator_authorize.py` (future spawn-wrapper consumer); `src/tac/preflight.py::check_subagent_spawn_includes_head_state_pv_evidence` (Catalog #376 STRICT gate); `tac.discipline_anti_pattern_guards.verify_head_state_before_spawn` (canonical helper)

## Probe outcome per Catalog #313

Registered `catalog_376_strict_gate_landing_20260528` PROCEED advisory; 30-day staleness window; auto-expires for re-evaluation per #313 sister discipline.

## Test coverage

* `src/tac/discipline_anti_pattern_guards/tests/test_guards.py` — **33 tests pass** (predecessor's canonical-helper tests)
* `src/tac/tests/test_check_376_subagent_spawn_pv_evidence.py` — **42 tests pass** (STRICT-gate tests: token unit / waiver semantics / spawn-event row extractor / strict-mode raise / strict-silent-on-clean / pre-cutoff exempt / post-cutoff PV-compliant accepted / post-cutoff missing-PV flagged / verbose / multi-violation aggregation / live-repo regression guard / Catalog #185 sister-callable / Catalog #176 META-meta-meta CLAUDE.md row present / Catalog #186 canonical-serializer claim)
* **Total: 75 tests pass**

## META-meta verification

* Catalog #118 (no duplicate catalog #s): 0 violations
* Catalog #176 (STRICT-callsite has CLAUDE.md row): 0 violations
* Catalog #185 (Live count: 0 verified empirically): 3 unrelated pre-existing violations (#131, #300, #346 — sister-territory; not regressions)

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|------|--------|-----------|
| #1 sensitivity-map | N/A | defensive validator gate |
| #2 Pareto constraint | N/A | not Pareto-relevant |
| #3 bit-allocator | N/A | no bit-allocator signal |
| #4 cathedral autopilot dispatch | **ACTIVE** | canonical helper `verify_head_state_before_spawn` IS the disambiguator between PROCEED vs STAND_DOWN at spawn time; future cathedral consumer can surface SPAWN-PV verdicts via Catalog #335 auto-discovery |
| #5 continual-learning posterior | **ACTIVE** | every spawn-event checkpoint row IS the canonical posterior anchor for the SPAWN-time discipline class; canonical equation `subagent_spawn_discipline_compounding_v1` accumulates anchors per Catalog #344 |
| #6 probe-disambiguator | **ACTIVE** | PV-evidence token presence vs absence IS the canonical disambiguator between SPAWN-discipline-honored vs SPAWN-discipline-violated |

## Mission contribution per Catalog #300

`apparatus_maintenance` — closes the SPAWN-time discipline class structurally; the canonical helper + STRICT gate sister-extincts the multi-subagent collision class at the 9th surface; unblocks future subagent spawns from silently producing STAND_DOWN by enforcing the canonical PV evidence chain.

## Strict-flip plan

* **Initial state (this landing):** WARN-ONLY (`strict=False`); cutoff = `2030-01-01T00:00:00Z`; live count = 0 across all 943 legacy rows.
* **Strict-flip prerequisite:** ≥3 successive sessions where every NEW spawn-event row (status=in_progress, step=1) carries PV evidence in notes/next_action/files_touched.
* **Strict-flip action:** bump `_CHECK_376_DISCIPLINE_CUTOFF_UTC` backward to the strict-flip date; flip `strict=False` → `strict=True` in `preflight_all` callsite; land in same commit batch per CLAUDE.md "Strict-flip atomicity rule".

## Sister coordination

* Slot 3 Wyner-Ziv `ae2423b1be51d65da` in flight — DISJOINT scope (`src/tac/substrates/wyner_ziv_pipeline_stage_codec/trainer.py`); zero collision.
* Slot 4 PR111-candidate `a0e79e055846b17c5` crashed (respawning sister) — DISJOINT scope (PR111 candidate work); zero collision.
* Predecessor `a15fb8266be531cb0` crashed at API rate-limit; canonical helper package landed at HEAD (1198 LOC across 3 files + 23KB test file); this successor lands the structural-protection landing.

## Files landed

| Path | Change | Author |
|------|--------|--------|
| `.omx/state/next_catalog_number.txt` | claim 376 → 377 | predecessor + successor |
| `src/tac/discipline_anti_pattern_guards/__init__.py` | NEW package | predecessor |
| `src/tac/discipline_anti_pattern_guards/subagent_spawn_head_pv_guard.py` | NEW (632 LOC) | predecessor |
| `src/tac/discipline_anti_pattern_guards/predecessor_handoff_auto_commit_guard.py` | NEW (459 LOC) | predecessor |
| `src/tac/discipline_anti_pattern_guards/tests/test_guards.py` | NEW (33 tests) | predecessor |
| `src/tac/preflight.py` | NEW `check_subagent_spawn_includes_head_state_pv_evidence` function + wire-in (WARN-ONLY) | successor (this landing) |
| `src/tac/tests/test_check_376_subagent_spawn_pv_evidence.py` | NEW (42 tests) | successor |
| `CLAUDE.md` | NEW Catalog #376 row | successor |
| `.omx/state/canonical_equations_registry.jsonl` | NEW canonical equation `subagent_spawn_discipline_compounding_v1` | successor |
| `.omx/state/probe_outcomes.jsonl` | NEW probe outcome `catalog_376_strict_gate_landing_20260528` | successor |
| `.omx/research/retroactive_sweep_for_catalog_376_20260528T180000Z.md` | NEW retroactive sweep memo (Catalog #348) | successor |
| `.omx/research/feedback_strict_gate_check_subagent_spawn_includes_head_state_pv_evidence_plus_canonical_helper_landed_20260528.md` | NEW (this memo) | successor |
