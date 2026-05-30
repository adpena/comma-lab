# Retroactive sweep: deferred items feeder audit 2026-05-30

Per Catalog #348 `check_new_gate_landing_includes_retroactive_sweep_evidence`:

## (1) Bug-class symptom signature
**Deferred items (probes / lanes / memos / tasks) accumulate in canonical state but never get picked up + fed into the canonical work queue or DAG even when reactivation criteria are satisfied.** Empirical anchor: 86 blocking probe outcomes + 159 deferred lanes + 9 deferred memos + 1 pending task + 7 canonical surfaces total, with ZERO canonical recurring sweep prior to this landing.

## (2) Pre-fix window
N/A — this is a META audit landing. The "pre-fix" window is the canonical pre-2026-05-30 state where no recurring feeder pass existed. The standing directive landed today at the same commit as this audit. Future feeder passes can ride on Catalog #371-class auto-recalibration if the `reactivation_criteria` field is consistently populated (see META Finding A in landing memo).

## (3) Historical KILL/DEFER/FALSIFY search
Searched `.omx/research/` for memos matching `*deferred*.md` (9 files; oldest 2026-05-09); `.omx/state/probe_outcomes.jsonl` for VERDICT in DEFER/KILL/INDEPENDENT (84 + 5 + 8 = 97 unique probes); `.omx/state/lane_registry.json` for notes matching `deferred.pending|research_only=true|reactivat` (159 lanes).

**HONEST FINDING per CLAUDE.md NO FAKE IMPLEMENTATIONS**: my initial Phase 2 token-overlap heuristic produced 4 FALSE POSITIVE "REACTIVATION_CRITERION_MET" verdicts (e.g., a UNIWARD 7th-order commit matched a z6/identity_predictor wire-in probe). These were REJECTED honestly per the non-negotiable; final verdict count is `REACTIVATION_CRITERION_MET_EMPIRICALLY = 0`. No historical KILL/DEFER/FALSIFY verdict is being INVALIDATED by this landing — the gate is a structural feeder audit, not a single-incident bug-class fix.

## (4) Per-finding RE-EVAL-priority assignment
- 11 `CRITERION_PAID_DISPATCH_REQUIRED` probes: PRIORITY 1 (operator-routable cap-window-gated paid dispatches)
- 6 DAG edges: PRIORITY 1 (canonical task status ledger insertion via `tac.canonical_task_status.transition_task_status`)
- 2 `CRITERION_WIRE_IN_REQUIRED` probes: PRIORITY 2 ($0 MLX-LOCAL canonical helper landings)
- 2 `CRITERION_SISTER_SUBAGENT_REQUIRED` probes: PRIORITY 2 (sister-subagent spawn requires Catalog #376 PV)
- 1 `CRITERION_SYMPOSIUM_REQUIRED` + 1 `CRITERION_MLX_LOCAL_FIRST_REQUIRED` + 2 `CRITERION_OPERATOR_DECISION_REQUIRED`: PRIORITY 3 (single-channel operator review)
- 4 `CRITERION_PATHS_ENUMERATED_IN_MEMO`: PRIORITY 3 (operator review of linked memos)
- 20 `CRITERION_NEEDS_MANUAL_PARSE`: PRIORITY 4 (audit subagent review next cap window)
- 70+/71 EMPTY `reactivation_criteria` probes: META PRIORITY 1 (canonical helper auto-derive from `next_action`)

## Recurring cadence
Per the standing directive, this audit should re-run every operator cap-window OR every N committed canonical state mutations (whichever first). Subsequent runs can compare DAG edge counts + queue insertion counts vs this baseline to detect drift in canonical apparatus health.
