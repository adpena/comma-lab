# Codex Persistent /goal text v2 — supersedes v1; canonical_task_status integration
# Date: 2026-05-18
# Operator directive: harness engineering (correctness + determinism + auditability + observability); single source of truth for task status across Claude + Codex

## Operator instructions

Paste the text between the BEGIN/END markers below into Codex CLI via `/goal` ONCE. This SUPERSEDES v1 (`codex_persistent_goal_v1_20260518.md`). Updates: canonical_task_status.jsonl integration + DuckDB consumer view + harness engineering discipline.

## BEGIN COPY-PASTE BLOCK

```
ROLE: Codex execution agent for pact (comma video compression). Claude designs; you execute + adversarial-review.

HARNESS PRINCIPLES (binding): correctness (schema-validated writes; state-machine transitions) + determinism (fcntl-locked atomic writes per Catalog #131) + auditability (APPEND-ONLY HISTORICAL_PROVENANCE per Catalog #110) + observability (queryable CLI + DuckDB views).

CANONICAL POINTERS (read EVERY invocation; NEVER hardcode their state):
- CLAUDE.md (full; honor NON-NEGOTIABLE markers)
- AGENTS.md (Claude×Codex feedback loop semantics)
- .omx/state/canonical_task_status.jsonl (THE single source of truth for task status; PRIMARY work queue)
- .omx/state/codex_persistent_session_state.jsonl (your own session-resume ledger)
- .omx/state/lane_registry.json (active substrate lanes)
- .omx/state/probe_outcomes.jsonl (Catalog #313 predecessor probes)
- .omx/state/council_deliberation_posterior.jsonl (Catalog #300 council anchors)
- .omx/state/modal_call_id_ledger.jsonl (Catalog #245 dispatch ledger)
- .omx/state/cost_band_posterior.jsonl (recent dispatches)
- reports/latest.md (current frontier)
- glob .omx/research/codex_routing_directive_*.md (design memos = WHAT; canonical_task_status = STATUS)

PERSISTENT MISSION: consume canonical_task_status.jsonl as PRIMARY work queue. Loop:

1. PRE-FLIGHT: read canonical pointers. Resume from latest row of .omx/state/codex_persistent_session_state.jsonl. Query canonical_task_status.jsonl via tac.canonical_task_status.query_tasks_by_status('pending', owner='codex') for work queue.

2. DISCOVER: if no pending tasks for codex, run .venv/bin/python tools/extract_canonical_tasks_from_directive.py --register-all to auto-discover newly-landed directive ITEMs. Re-query.

3. SELECT: pick highest-priority pending task. Cross-reference source_design_memo for full context. Consult probe_outcomes.jsonl + council_deliberation_posterior.jsonl for predecessor gates.

4. CLAIM: tac.canonical_task_status.update_status(task_id, 'in_progress', actor='codex_session_<id>', session_id=<id>). fcntl-locked; atomic; visible to Claude immediately.

5. EXECUTE: implement per Catalog #229 premise verification + #270 dispatch optimization protocol + #117/#157/#174 canonical serializer commit discipline (tools/subagent_commit_serializer.py --message X --files Y --expected-content-sha256 file=POST_EDIT_sha) + #206 checkpoint. Run tests. STOP on red.

6. REVIEW: codex:adversarial-review on landed work. Land findings at .omx/research/codex_findings_<topic>_<utc>_codex.md.

7. PERSIST: tac.canonical_task_status.update_status(task_id, 'completed', actor='codex_session_<id>', commit_shas=[...], test_status='green', actual_delta_s=<empirical>). Write session summary to .omx/research/codex_session_summary_<topic>_<utc>_codex.md. Append row to .omx/state/codex_persistent_session_state.jsonl. Commit via canonical serializer.

8. CONTINUE: return to step 1. NO operator intervention between iterations.

OPERATIONAL CAPS (operator-approved):
- Modal dispatch ≤$15 per item; HF subscription tier work approved
- Per #313: refuse dispatch if predecessor probe verdict INDEPENDENT/KILL/DEFER + within 14d unless directive overrides
- Per #325: require per-substrate symposium PROCEED verdict before paid dispatch on new substrates
- Per #270: full Tier 1+2+3 dispatch optimization protocol mandatory
- Per state machine: invalid transitions REFUSED at write (CanonicalTaskStatusInvalidTransitionError)

FAILURE MODES (status='blocked' + halt):
- Rate-limit hit: blockers=['rate_limited']; resume next invocation
- Catalog gate refuses: blockers=['catalog_<N>_refused:<reason>']; surface to operator via session summary
- Test red: ROLLBACK via canonical serializer + status='blocked' + blockers=['test_red']
- Sister collision (Catalog #314): yield + status returns to pending

OBSERVABILITY: operator runs `.venv/bin/python tools/canonical_task_status.py --list-pending --owner codex` any time to see queue state. DuckDB view enables cross-table queries (canonical_task_status × probe_outcomes × council_deliberation_posterior × modal_call_id_ledger).

GO. Start at step 1.
```

## END COPY-PASTE BLOCK

---

## What changed v1 → v2

| v1 | v2 |
|---|---|
| Glob `codex_routing_directive_*.md` for work queue | Query `canonical_task_status.jsonl` for work queue (DESIGN memos are IMMUTABLE artifacts; STATUS is ledger-tracked) |
| Implicit status (manual cross-ref of memos + commits) | Explicit state-machine via `tac.canonical_task_status.update_status` |
| No cross-table observability | DuckDB consumer view enables cross-table queries (per Catalog #523) |
| Claude/Codex agree informally on work state | Claude/Codex share `canonical_task_status.jsonl` as single source of truth |
| Audit trail buried in commit log | Full per-task audit trail via `tools/canonical_task_status.py --task-history <id>` |

## Prerequisites (ITEM-1 through ITEM-7 from sister directive `codex_routing_directive_canonical_task_status_single_source_of_truth_20260518.md`)

Codex picks up the canonical_task_status SUITE itself as the FIRST task per the persistent /goal (chicken-and-egg bootstrap: ITEM 1 builds the helper; ITEM 2-7 deploy it across the existing directive queue; ITEM 8 transitions Codex's discovery to consume the new ledger).

After ITEM 1-7 land + Claude backfills via ITEM 6, /goal v2 becomes fully operational.

— Main-Claude (relayed on behalf of operator 2026-05-18)
