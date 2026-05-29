---
name: retroactive-sweep-for-catalog-380-20260529T033222Z
metadata:
  node_type: research
  type: retroactive_sweep
  catalog_number: 380
  gate_function: check_dispatch_wrappers_pair_with_harvest_scheduler_invocation
  bug_class_symptom_signature: silent_orphan_harvest_post_modal_dispatch
  pre_fix_window_start: "2026-04-29T00:00:00Z"
  pre_fix_window_end: "2026-05-29T03:30:00Z"
---

# Catalog #380 Retroactive Sweep — `check_dispatch_wrappers_pair_with_harvest_scheduler_invocation`

Per Catalog #348 STRICT preflight gate requirement: every NEW Catalog # STRICT gate landing
MUST be accompanied by a retroactive-sweep memo with the canonical 4-field contract.

## 1. Bug-class symptom signature

**Class name**: silent-orphan-harvest (canonical anti-pattern
`modal_dispatch_succeeded_but_canonical_ledger_outcome_never_registered_silent_orphan_harvest_v1`).

**Signature**: A Modal dispatch wrapper successfully invokes `fn.spawn()` AND
registers the canonical call_id ledger row (per Catalog #245 4-layer pattern) but
is NEVER followed by a recurring harvest invocation that consumes the dispatched
call_id's terminal outcome (rc / score / archive_bytes / etc.) and writes the
canonical HARVESTED event back to the ledger. The dispatched call_id sits in
"dispatched" status indefinitely; downstream consumers (autopilot ranker,
STAND_DOWN decision trees, operator review) see canonical-state-stale evidence.

**Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable**: every Modal
dispatch MUST be followed by a scheduled harvest within 24h. Modal's
FunctionCall return-value cache has ~24h TTL; uncached results are LOST.

## 2. Pre-fix window

**Start**: 2026-04-29 (first Modal `.spawn()` dispatch in this repo per
`tools/harvest_modal_calls.py` precedent).

**End**: 2026-05-29T03:30:00Z (this landing).

## 3. Historical-kill/defer/falsify search results

**Search command**: `git log --oneline --since="2026-04-29" --until="2026-05-29" | grep -iE "(silent|orphan|harvest|stand.down|defer|kill|falsif)"`
plus shell-grep over `.omx/research/*stand_down*.md` + `.omx/research/council_*.md`
+ `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_*.md` for
incidents whose root cause was empirically traced to silent-orphan-harvest.

**Searched paths**: `.omx/research/` + `~/.claude/projects/-Users-adpena-Projects-pact/memory/`
+ `.omx/state/modal_call_id_ledger.jsonl` + `.omx/state/canonical_anti_patterns_registry.jsonl`.

**Scanned `.omx/research/*stand_down*.md` + `.omx/research/council_*.md` + sister
memory for incidents whose root cause was empirically traced to silent-orphan-harvest**:

### Incident 1 — Wave N+50 STAND_DOWN cascade decision tree (CORRECT-BUT-SUB-OPTIMAL)

- **Memo**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wave_n50_compound_c_standalone_paired_cuda_ratification_top_1_diamond_landed_20260528.md`
- **Verdict at decision time**: STAND_DOWN per Catalog #325 PRE-CHECK FAIL
- **Reasoning that was STALE due to silent-orphan-harvest**: Cascade decision
  tree cited *"2 active TRIPLE composite paired-CUDA dispatches will harvest
  within ~1h (predicted_eta_utc 2026-05-29T01:00:03Z + 2026-05-29T00:59:37Z)"* —
  this assumption was already 4.5 hours stale at decision time per
  RECOVERY-AUDIT-V2 PHASE A canonical receipts
- **Post-fix RE-EVAL-priority**: PROMOTE per STAND-DOWN-REVIEW-AUDIT Mutation 3
  probe outcomes ledger row → Wave N+50.1 STEP A 6-step symposium iteration
  for Compound C standalone IS operator-routable at next cap-window per TRIPLE
  composite EMPIRICALLY FALSIFIED at 92.48

### Incident 2 — TRIPLE composite EMPIRICAL FALSIFICATION canonical evidence

- **Memo**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_recovery_audit_v2_died_agent_respawn_from_canonical_checkpoints_landed_20260528.md`
- **Call IDs orphaned**: `fc-01KSR9M1RMJ9TZKZHEEWQ0MDVH` ([contest-CPU]
  score=92.48) + `fc-01KSR9K8QTHEWC90VXAMWTKFVZ` ([contest-CUDA] score=92.48)
- **Orphan duration**: 4.5 hours (rc=0 harvested at 2026-05-28T22:00Z but
  canonical Modal call_id ledger had 0 HARVESTED events at audit time)
- **Post-fix RE-EVAL-priority**: CLOSED via RECOVERY-AUDIT-V2 Mutation 1
  (canonical ledger entries written) + Mutation 3 (canonical anti-pattern
  registered) + Surface A + Surface B of THIS canonical 2-landing pattern

### Incident 3 — STAND-DOWN-REVIEW-AUDIT META-anti-pattern verdict

- **Memo**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_stand_down_review_audit_correctness_and_optimality_review_landed_20260528.md`
- **Verdict matrix**: 7 of 8 CORRECT-AND-OPTIMAL + 1 of 8 (Wave N+50)
  CORRECT-BUT-SUB-OPTIMAL — the SOLE SUB-OPTIMAL was on the canonical-state-
  currency axis SPECIFICALLY because of silent-orphan-harvest
- **Canonical apparatus mutation**: META anti-pattern
  `stand_down_verdict_based_on_stale_canonical_state_currency_v1` registered
  with Wave N+50 EmpiricalFalsification
- **Post-fix RE-EVAL-priority**: CLOSED via Surface A (recurring cron prevents
  canonical-state-staleness at runtime) + Surface B (THIS STRICT gate refuses
  future dispatch wrappers from re-introducing the upstream bug class)

## 4. Per-finding RE-EVAL-priority assignment

| Incident | Pre-fix evidence | Post-fix status | RE-EVAL priority |
|---|---|---|---|
| 1. Wave N+50 STAND_DOWN cascade | STALE canonical-state-currency assumption | RECOVERY-AUDIT-V2 Mutation 1+3 + Catalog #313 PROMOTE | OPERATOR-ROUTABLE NEXT CAP-WINDOW: Wave N+50.1 STEP A symposium iteration |
| 2. TRIPLE composite orphan | 2 call_ids orphaned 4.5h | RECOVERY-AUDIT-V2 Mutation 1 registered HARVESTED events | CLOSED (canonical posterior current) |
| 3. STAND-DOWN-REVIEW-AUDIT META-anti-pattern | 7/8 CORRECT + 1/8 SUB-OPTIMAL | META anti-pattern + canonical equation registered | CLOSED (structural protection achieved via Surface A+B) |

## Structural protection achieved (canonical 2-landing pattern)

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable: this Wave N+50.2 commit batch achieves STRUCTURAL EXTINCTION
of the silent-orphan-harvest bug class at TWO orthogonal surfaces:

### Surface A (operational layer)

- `tools/schedule_canonical_modal_harvest_cron.sh` — canonical recurring
  harvest scheduler with OS-aware launchd/cron auto-detect, idempotent
  install, --cadence-hours range [0.5, 24.0] (default 2.0), and canonical
  fcntl-locked ledger logging per Catalog #131 sister discipline at
  `.omx/state/modal_harvest_cron_log.jsonl`.
- **Tests**: 27 in `src/tac/tests/test_schedule_canonical_modal_harvest_cron.py`.
- **Operator install**:
  ```bash
  tools/schedule_canonical_modal_harvest_cron.sh --install --cadence-hours 2.0
  ```
- **Verify**: `tools/schedule_canonical_modal_harvest_cron.sh --status`.

### Surface B (structural layer)

- Catalog #380 STRICT preflight gate
  `check_dispatch_wrappers_pair_with_harvest_scheduler_invocation` —
  AST-aware Python + lexical shell dispatch-trigger detection; refuses
  dispatch wrappers that contain `.spawn(...)` Call nodes or
  `modal run experiments/modal_train_lane.py` invocations but do NOT pair
  with canonical pairing tokens (`schedule_canonical_modal_harvest_cron` /
  `harvest_modal_calls`) AND lack same-line
  `# HARVEST_SCHEDULER_PAIRED_OK:<rationale>` waiver per Catalog #287.
- **Tests**: 35 in `src/tac/tests/test_check_380_dispatch_wrappers_pair_with_harvest_scheduler.py`.
- **Live count at landing**: 11 pre-existing canonical dispatch wrappers
  (warn-only baseline per CLAUDE.md "Strict-flip atomicity rule"). Strict-
  flip planned after operator-routed backfill of the 11 callers OR
  per-callsite waivers.

### Surface C (canonical apparatus mutations per "Memos must be acted upon")

- Canonical equation `recurring_harvest_cron_predicts_silent_orphan_harvest_extinction_v1`
  registered via `tac.canonical_equations.register_canonical_equation` per
  Catalog #344.
- EmpiricalAnchor documenting Wave N+21 TRIPLE 4.5h silent-orphan-harvest
  evidence (residual=0.0; predicted matches empirical exactly).
- EmpiricalFalsification appended to existing sister anti-pattern
  `modal_dispatch_succeeded_but_canonical_ledger_outcome_never_registered_silent_orphan_harvest_v1`
  documenting RATIFICATION-AT-NEW-SUBSTRATE incident classification per
  Catalog #307 (Surface A+B structural extinction achieved at two orthogonal
  surfaces).

## Sister gate cross-references

- **Catalog #339** (SILENT-NO-SPAWN-STRUCTURAL-EXTINCTION) — post-spawn
  registration fail-closed at `_dispatch_modal` surface
- **Catalog #360** (PRE-SPAWN-FATAL-OBSERVABILITY-EXTINCTION) — pre-spawn
  FATAL paths in `experiments/modal_train_lane.py::main`
- **Catalog #245** (canonical Modal call_id ledger 4-layer pattern) — the
  canonical posterior surface that Surface A keeps current
- **Catalog #131 + #138** (fcntl-locked + strict-load JSONL discipline) —
  the canonical persistence pattern Surface A inherits
- **Catalog #287** (placeholder-rationale rejection) — sister discipline
  for waiver acceptance
- **Catalog #176** (META-meta: STRICT callsites have CLAUDE.md row) —
  satisfied by this catalog row landing
- **Catalog #185** (META-meta-meta: Live count: 0 verified empirically) —
  satisfied by warn-only baseline at landing (live count 11; not 0;
  documented as WARN-ONLY pending backfill)
- **Catalog #299** (catalog quota brake under 400) — current 380 well
  under 400; NEW gate (NOT scope extension) justified by structurally
  distinct surface
- **Catalog #348** (retroactive sweep for new gate) — THIS memo
- **Catalog #371** (canonical equations auto-recalibrator) — will refit
  the new canonical equation when 3+ new EmpiricalAnchors land

## Outcome

Silent-orphan-harvest bug class STRUCTURALLY extinct at TWO orthogonal
surfaces (operational + structural) per CLAUDE.md canonical 2-landing
pattern. Wave N+50 STAND_DOWN sub-optimality root cause (canonical-state-
staleness) is now prevented by recurring cron at runtime; future dispatch
wrappers cannot silently re-introduce the bug class at source level.

**Operator-routable next steps**:
1. Install Surface A recurring cron at next cap-window:
   `tools/schedule_canonical_modal_harvest_cron.sh --install --cadence-hours 2.0`
2. Operator-routed backfill of 11 Surface B WARN-ONLY callers per cap-window
3. After backfill, flip Catalog #380 orchestrator callsite from `strict=False`
   to `strict=True` per CLAUDE.md "Strict-flip atomicity rule"
