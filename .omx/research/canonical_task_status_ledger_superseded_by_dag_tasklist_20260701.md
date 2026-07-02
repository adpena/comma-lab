# Canonical-task-status ledger SUPERSEDED — SoT is the DAG + live TaskList (2026-07-01)

**Drift-fix (operator directive 2026-07-01 "fix drift sources").** The
`.omx/state/canonical_task_status.jsonl` ledger's last event is dated
**2026-06-18T02:16:25Z** and it tracks NONE of the live level-set witness tasks
(#200-225). Task tracking migrated to the **DAG** (`FEED-*` blocks in
`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`) + the
live **TaskList** during the 2026-06-19 → 2026-07-01 witness campaign. The
JSONL ledger is stale, not the source of truth.

## Decision: DOCUMENT-ABANDONMENT (not full refresh)

Per the drift-fix task, the two options were (a) refresh the ledger to reflect
#200-225, or (b) document the abandonment + point to the DAG + TaskList as the
SoT. **Chosen: (b).**

Rationale (NO-FAKE): the ledger schema requires per-task
`predicted_cost_usd` + `predicted_delta_s_band` + `source_design_memo`.
Back-filling honest values for the ~25 live witness tasks (#200-225) is not
possible from the DAG alone without INVENTING cost / ΔS bands — that would be a
NO-FAKE class #4 (placeholder-in-canonical-data-field) violation. The DAG
FEED-* blocks + the live TaskList already carry the real, dated task state.
Duplicating a lossy, fabricated copy into the JSONL would create NEW drift, not
fix it.

## Source of truth for witness tasks (#200-225), effective 2026-07-01

1. **DAG** — `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`
   (`FEED-*` blocks; the trajectory/history leg of the triality).
2. **Live TaskList** — the session task list (`#200-225`), the actionable leg.
3. **DSL** — `tac.witness_dsl.*` (the executable program leg).
4. **Equations** — `tac.canonical_equations` (the confirmed-law leg; the
   2026-07-01 drift-fix registered the FEED-ly / FEED-dj / sig-proc measured
   findings here so the equations leg is consistent with the DAG leg).

## Ledger status going forward

- The historical rows (through 2026-06-18) are preserved append-only per
  Catalog #110/#113 HISTORICAL_PROVENANCE — NOT mutated.
- A single machine-readable marker row
  (`canonical_task_status_ledger_superseded_by_dag_tasklist_20260701`) is
  appended to the ledger itself so a future reader consulting the JSONL sees
  the SoT migration in-band.
- Stale pre-2026-06-18 `in_progress` rows (e.g.
  `pose_low_rank_radial_zoom_codec_build_20260617`) are LEFT AS-IS — their real
  resolution is unknown to this drift-fix pass, and asserting `completed` /
  `cancelled` without verification would itself be a NO-FAKE violation. The
  preflight gate `check_canonical_task_status_no_dangling_transitions` validates
  the transition state-machine (not staleness); leaving them pending/in_progress
  does not break it.
- If a future session wants the JSONL to be the live SoT again, it must
  re-adopt it deliberately (register #200-225 with REAL cost/ΔS metadata) and
  supersede this note.

## Cross-references

- Triality discipline: `docs/triality_dag_dsl_equations_deepmath.md` +
  CLAUDE.md "The Triality — DAG ↔ DSL ↔ equations".
- Drift-fix equations landing: `tac.canonical_equations.witness_measured_findings_20260701`
  (8 measured findings registered 2026-07-01).
