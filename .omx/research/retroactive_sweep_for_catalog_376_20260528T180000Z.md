# Retroactive sweep for Catalog #376 — check_subagent_spawn_includes_head_state_pv_evidence

**Generated:** 2026-05-28T18:00:00Z
**Lane:** `lane_strict_gate_check_subagent_spawn_includes_head_state_pv_evidence_plus_canonical_helper_20260528`
**Gate landing commit:** (pending — landed in same commit batch as this memo)
**Per:** CLAUDE.md Catalog #348 (`check_new_gate_landing_includes_retroactive_sweep_evidence`)

## 1. Bug-class symptom signature

A subagent spawn that proceeded without first verifying HEAD state +
sister landing memos + sister-subagent in-flight checkpoints. The
post-spawn discovery of overlapping work resulted in a STAND_DOWN
(predecessor work absorbed via Catalog #340 / #314 surface) instead of
PROCEED.

Canonical symptoms in pre-fix sessions:

* Parent agent invokes `Agent` spawn with declared scope `X`
* Spawned subagent's first session-step discovers a recent HEAD commit
  touching paths overlapping `X`
* Spawned subagent records `STAND_DOWN per Catalog #340 Variant 1`
* Net effect: spawn cost paid, no work landed, operator forwarding-loop
  re-spawns with corrected scope

## 2. Pre-fix window

* **Window start:** 2026-05-15 (pre-Catalog #340 STAGING-time guard
  landing — before that point, the sister-checkpoint guard at staging
  was the only structural protection)
* **Window end:** 2026-05-28T17:00:00Z (this gate's landing window)

## 3. Historical KILL / DEFER / FALSIFY search

Searched `~/.claude/projects/-Users-adpena-Projects-pact/memory/` +
`.omx/research/` for STAND_DOWN incidents whose root cause maps to
anti-pattern #13 (`subagent_spawn_without_head_state_premise_verification_v1`).

### Confirmed incidents (2 anchors per anti-pattern #13 registration in Wave N+7 commit `49bdcd78f`):

| # | Wave / Slot | Predecessor commit | Memo |
|---|------|-------------------|------|
| 1 | Wave N+5 Slot 1 | `e61ea93b0` (Compound C predecessor) | (predecessor-context; STAND_DOWN recorded post-hoc) |
| 2 | Wave N+5 Slot 2 | `5d38bf9df` (framework_agnostic STAND_DOWN resolved) | `feedback_framework_agnostic_portability_primitives_committed_predecessor_working_tree_per_operator_meta_directive_20260527.md` |

### Confirmed incidents NOT KILLED / NOT FALSIFIED:

* Both Wave N+5 STAND_DOWN incidents = IMPLEMENTATION-LEVEL operational
  bugs per Catalog #307 paradigm-vs-implementation classification.
* Neither incident produced a paradigm-level KILL verdict.
* Both incidents were resolved without re-investigation needed (the
  spawned subagent stood down cleanly + the parent re-routed).

### No tainted KILL / DEFER / FALSIFY verdicts to re-evaluate

This gate's introduction does NOT invalidate any prior KILL / DEFER /
FALSIFY verdict. The bug class is operational (subagent stand-downs
waste session budget) NOT empirical-evidence-poisoning.

## 4. Per-finding RE-EVAL-priority assignment

| Finding | Status | RE-EVAL priority | Rationale |
|---------|--------|------------------|-----------|
| Wave N+5 Slot 1 STAND_DOWN | RESOLVED at commit `e61ea93b0` follow-up | NONE | Spawn re-routed cleanly; no tainted artifacts |
| Wave N+5 Slot 2 STAND_DOWN | RESOLVED at commit `5d38bf9df` (framework_agnostic memo) | NONE | Predecessor working tree committed per operator META directive |

## 5. Canonical extinction at this gate

This gate's landing extincts the bug class STRUCTURALLY for FUTURE
spawn events:

1. **Canonical helper** `tac.discipline_anti_pattern_guards.verify_head_state_before_spawn`
   provides the per-call PV decision (PROCEED / DUPLICATE_HEAD_STATE /
   SISTER_IN_FLIGHT) BEFORE spawn cost is paid.
2. **STRICT preflight gate** Catalog #376 enforces source-text
   discipline that spawn-event checkpoint rows MUST record PV evidence
   in `notes` / `next_action` / `files_touched`.
3. **Canonical equation** `subagent_spawn_discipline_compounding_v1`
   registers the compounding-anchor pattern (P(STAND_DOWN | spawn
   without PV) ≈ 1 across the 2 empirical anchors).

## 6. Cross-references

* Anti-pattern registration: Wave N+7 Slot 2 commit `49bdcd78f`
* Canonical helper: `src/tac/discipline_anti_pattern_guards/`
* STRICT gate: `src/tac/preflight.py::check_subagent_spawn_includes_head_state_pv_evidence`
* Tests: `src/tac/tests/test_check_376_subagent_spawn_pv_evidence.py` (42 tests) + `src/tac/discipline_anti_pattern_guards/tests/test_guards.py` (33 tests)
* Canonical equation: `subagent_spawn_discipline_compounding_v1`
* Probe outcome: `catalog_376_strict_gate_landing_20260528`
* Landing memo: `feedback_strict_gate_check_subagent_spawn_includes_head_state_pv_evidence_plus_canonical_helper_landed_20260528.md`

## 7. Verdict

PROCEED — no tainted historical KILL/DEFER/FALSIFY verdicts surface;
the gate's landing structurally extincts the bug class for future spawn
events without invalidating any prior empirical evidence.

---

## 8. APPENDED CLARIFICATION (per Catalog #110/#113 APPEND-ONLY) — 2026-05-28

Per Catalog #348 strict validator audit 2026-05-28: this memo's Section 3 header used "Historical KILL / DEFER / FALSIFY search"; the canonical Catalog #348 validator token is "historical-kill/defer/falsify search results" (lowercase, with dashes, trailing "results"). This APPENDED section adds the canonical token verbatim so the Catalog #348 validator passes.

### historical-KILL/DEFER/FALSIFY search results (canonical token re-affirmation)

Per Section 3 above: searched `~/.claude/projects/-Users-adpena-Projects-pact/memory/` + `.omx/research/` for STAND_DOWN incidents whose root cause maps to anti-pattern #13 (`subagent_spawn_without_head_state_premise_verification_v1`). Confirmed 2 incidents (Wave N+5 Slot 1 commit `e61ea93b0` + Wave N+5 Slot 2 commit `5d38bf9df`) — both IMPLEMENTATION-LEVEL per Catalog #307; neither produced a paradigm-level KILL verdict; both resolved without re-investigation. **Zero tainted KILL/DEFER/FALSIFY verdicts surfaced.** Per-finding RE-EVAL-priority assignment: NONE (both cleanly resolved).

### Canonical gate function reference

- Function: `check_subagent_spawn_includes_head_state_pv_evidence`
- File: `src/tac/preflight.py`
- Catalog #: 376
- Search command: `grep -rnE "STAND_DOWN.*Catalog.*340" ~/.claude/projects/-Users-adpena-Projects-pact/memory/ .omx/research/`
- Sister gates: #340 / #314 / #229 / #206

APPENDED-BY: slot3_catalog_348_retroactive_sweep_memos_landing_20260528
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
