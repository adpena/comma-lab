# Codex routing directive: canonical task status single-source-of-truth (harness engineering)
# Date: 2026-05-18
# Per operator standing 2026-05-18: "harness engineering principles for correctness and determinism and auditability and observability; single source of truth representing the status of all tasks represented in the design memos"

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (full)
2. `/Users/adpena/Projects/pact/AGENTS.md` (full; Claude×Codex feedback loop)
3. `.omx/research/codex_persistent_goal_v1_20260518.md` → SUPERSEDED by v2 landed this directive
4. `.omx/state/codex_persistent_session_state.jsonl` (Codex's own loop ledger; remains separate)
5. `.omx/state/modal_call_id_ledger.jsonl` (Catalog #245 canonical 4-layer pattern — TEMPLATE for this work)
6. `src/tac/deploy/modal/call_id_ledger.py` (Catalog #245 canonical helper — TEMPLATE for `tac.canonical_task_status`)

## OPERATOR DIRECTIVE (verbatim 2026-05-18)

> *"we just need to make sure to use harness engineering principles for correctness and determinism and auditability and observability; we want codex and claude to be able to rely on a single source of truth representing the status of all tasks represented in the design memos"*

## THE PROBLEM

Current state has FRAGMENTED task status:
- Claude's `TaskCreate/TaskUpdate` (Claude-private; not visible to Codex)
- `codex_persistent_session_state.jsonl` (Codex-private loop ledger)
- Design memos at `.omx/research/codex_routing_directive_*.md` (immutable artifacts; status of their ITEMs not co-located)
- Lane registry `.omx/state/lane_registry.json` (substrate lanes only, not work-item tasks)

**No SINGLE canonical source covers task status across BOTH agents + ALL design-memo items.** This violates harness engineering correctness + auditability principles.

## THE DESIGN: `.omx/state/canonical_task_status.jsonl`

### Schema (canonical_task_status_v1_20260518)

```json
{
  "schema_version": "canonical_task_status_v1_20260518",
  "task_id": "<directive_basename>::<item_id>",
  "source_design_memo": "<path_to_memo>",
  "title": "<short title extracted from memo section header>",
  "status": "pending|in_progress|completed|blocked|deferred|cancelled",
  "owner": "claude|codex|operator|sister_subagent_<id>",
  "predicted_cost_usd": <float|null>,
  "predicted_delta_s_band": [<lower>, <upper>] | null,
  "actual_delta_s": <float|null>,
  "commit_shas": [<list of git shas>],
  "test_status": "green|red|n_a|pending",
  "blockers": [<list of blocker descriptors>],
  "started_at_utc": "<iso>|null",
  "completed_at_utc": "<iso>|null",
  "event_type": "registered|status_change|note|completion|blocked|cancelled",
  "event_timestamp_utc": "<iso>",
  "event_actor": "claude_main|codex_session_<id>|operator|sister_subagent_<id>",
  "event_notes": "<free-form>",
  "session_id": "<session>",
  "written_at_utc": "<iso>",
  "written_pid": <int>,
  "written_host": "<host>"
}
```

### Status state machine (validated at write)

```
pending → in_progress | blocked | cancelled
in_progress → completed | blocked | cancelled
blocked → in_progress | deferred | cancelled
deferred → pending | cancelled
completed → re-opened (becomes new pending row with predecessor link)
cancelled → re-opened (becomes new pending row)
```

Invalid transitions are REFUSED at write time (raises `CanonicalTaskStatusInvalidTransitionError`).

### Discipline references

- Per Catalog #131: fcntl-locked JSONL writes (sister to `.omx/state/modal_call_id_ledger.jsonl`)
- Per Catalog #110/#113: APPEND-ONLY HISTORICAL_PROVENANCE; status changes = new rows, NOT in-place mutation
- Per Catalog #138: `load_canonical_task_status_strict()` raises `CanonicalTaskStatusCorruptError` on JSON parse failure; quarantines corrupt files
- Per Catalog #245: 4-layer pattern (canonical helper module + CLI tool + STRICT preflight gate + operator-routable bypass)
- Per Catalog #287: every status update with `actual_delta_s` MUST carry `[empirical:<path>]` tag

### LIVE_STATE vs HISTORICAL_PROVENANCE classification

`.omx/state/canonical_task_status.jsonl` is **HYBRID**:
- **LIVE_STATE semantics** for current-status queries (latest-row-wins per task_id) — gitignored
- **HISTORICAL_PROVENANCE semantics** for audit trail (append-only; no in-place mutation) — preserved in git via daily snapshots to `.omx/state/archive/canonical_task_status_<YYYY-MM>.jsonl` per Catalog #154 archival policy

## ITEM 1 — Build canonical helper `src/tac/canonical_task_status/` package

```python
# src/tac/canonical_task_status/contract.py
@dataclass(frozen=True, slots=True)
class CanonicalTaskStatusRow:
    # All schema fields above as typed Pydantic-style frozen dataclass
    ...

# src/tac/canonical_task_status/writer.py
def register_task(
    task_id: str,
    source_design_memo: pathlib.Path,
    title: str,
    owner: str,
    predicted_cost_usd: float | None = None,
    predicted_delta_s_band: tuple[float, float] | None = None,
    *,
    actor: str,
    session_id: str,
) -> CanonicalTaskStatusRow: ...

def update_status(
    task_id: str,
    new_status: Literal["in_progress", "completed", "blocked", "deferred", "cancelled"],
    *,
    actor: str,
    session_id: str,
    notes: str = "",
    commit_shas: tuple[str, ...] = (),
    test_status: Literal["green", "red", "n_a", "pending"] = "pending",
    blockers: tuple[str, ...] = (),
    actual_delta_s: float | None = None,
) -> CanonicalTaskStatusRow: ...

def append_note(
    task_id: str,
    notes: str,
    *,
    actor: str,
    session_id: str,
) -> CanonicalTaskStatusRow: ...

# src/tac/canonical_task_status/loader.py
def load_canonical_task_status_strict(
    repo_root: pathlib.Path | None = None,
) -> list[CanonicalTaskStatusRow]: ...

def latest_status_by_task_id(
    task_id: str,
    repo_root: pathlib.Path | None = None,
) -> CanonicalTaskStatusRow | None: ...

# src/tac/canonical_task_status/query.py
def query_tasks_by_status(
    status: str,
    *,
    owner: str | None = None,
    source_design_memo: pathlib.Path | None = None,
) -> list[CanonicalTaskStatusRow]: ...

def query_tasks_by_directive(
    source_design_memo: pathlib.Path,
) -> list[CanonicalTaskStatusRow]: ...

def query_task_history(task_id: str) -> list[CanonicalTaskStatusRow]: ...
```

Per Catalog #131 sister to `tac.deploy.modal.call_id_ledger.append_event_locked` (fcntl.flock(LOCK_EX) on `.omx/state/.canonical_task_status.lock`; atomic `.tmp.<uuid12>` + `os.replace`).

## ITEM 2 — Extract tasks from design memos (one-time + ongoing)

Build `tools/extract_canonical_tasks_from_directive.py`:

```bash
# Operator-facing CLI:
.venv/bin/python tools/extract_canonical_tasks_from_directive.py \
    --directive .omx/research/codex_routing_directive_*.md \
    --register-all

# What it does:
# 1. Parse each codex_routing_directive_*.md for `### ITEM N` section headers
# 2. Extract title + predicted cost + predicted ΔS from each ITEM
# 3. Register a row in canonical_task_status.jsonl with status=pending, owner=codex
# 4. Idempotent: re-running doesn't double-register
```

Re-run periodically (or as Codex pre-flight step) to auto-discover newly-landed directive ITEMs.

## ITEM 3 — Operator-facing CLI tool `tools/canonical_task_status.py`

```bash
# Queries:
.venv/bin/python tools/canonical_task_status.py --list-pending
.venv/bin/python tools/canonical_task_status.py --list-blocked
.venv/bin/python tools/canonical_task_status.py --list-by-owner codex
.venv/bin/python tools/canonical_task_status.py --task-history codex_routing_directive_v2::ITEM_1
.venv/bin/python tools/canonical_task_status.py --directive-summary codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518.md
.venv/bin/python tools/canonical_task_status.py --json   # full machine-readable dump

# Operator-only updates (manual override):
.venv/bin/python tools/canonical_task_status.py update \
    --task-id codex_routing_directive_v2::ITEM_1 \
    --status blocked \
    --actor operator \
    --notes "blocked on Modal billing"
```

## ITEM 4 — STRICT preflight gate `check_canonical_task_status_no_dangling_transitions`

Refuses canonical_task_status.jsonl with invalid state-machine transitions OR missing required fields OR tasks orphaned from design memos. Same-line waiver `# CANONICAL_TASK_STATUS_OK:<reason>` honored. Claim Catalog # via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "canonical task status STRICT gate"`.

## ITEM 5 — Update Codex persistent /goal to v2 (consumes canonical_task_status.jsonl)

Land `.omx/research/codex_persistent_goal_v2_20260518.md` with the canonical-task-status integration replacing v1's directive-scanning logic. v1 → v2 migration is operator-pastes-once.

## ITEM 6 — Backfill existing task state into canonical_task_status.jsonl

One-time operator-routable backfill: for each landed codex_routing_directive_*.md + each known TaskCreate from Claude's task list, register canonical row. Claude can do this in-context (~30 min editor).

## ITEM 7 — Update Claude's session-level convention: every TaskCreate ALSO writes to canonical_task_status.jsonl

Claude's TaskCreate is Claude-private; the canonical ledger is dual-agent. Convention update: every Claude TaskCreate ALSO calls `tac.canonical_task_status.register_task(...)` so both surfaces stay in sync. Documented in AGENTS.md.

## ITEM 8 — Codex auto-discovers new directives via canonical_task_status.jsonl

Codex persistent loop step 2 (DISCOVER) becomes:
1. `query_tasks_by_status("pending", owner="codex")` → highest-priority unfinished work
2. Cross-reference source_design_memo path for context
3. Auto-update status to in_progress at step 3 execution start
4. Auto-update status to completed (+ commit_shas + test_status) at step 5 persist

## ITEM 9 — Harness-engineering principles checklist (per operator standing)

| Principle | How canonical_task_status.jsonl satisfies |
|---|---|
| **Correctness** | Schema validation at write + state-machine transition validation + Pydantic-style frozen dataclass; invalid writes REFUSED |
| **Determinism** | fcntl-locked atomic writes per Catalog #131; reproducible across CPU/CUDA; latest-row-wins query semantics deterministic |
| **Auditability** | APPEND-ONLY HISTORICAL_PROVENANCE per Catalog #110; every status change = new row with event_actor + event_timestamp; full per-task history queryable via CLI |
| **Observability** | CLI tool (operator-facing) + machine-readable JSON dump + canonical helper API for autopilot consumers + integration with cathedral autopilot v2 cascade reward factors |

## OPERATIONAL CONSEQUENCES

- Both Claude and Codex read `canonical_task_status.jsonl` to know "what's the current state of work?"
- Both Claude and Codex append rows to update status
- Operator runs `tools/canonical_task_status.py --list-pending` to see what's queued
- Codex's persistent /goal v2 consumes canonical_task_status.jsonl as PRIMARY source (not directive file glob)
- Design memos remain IMMUTABLE per Catalog #110 (the WHAT); ledger tracks STATUS (the WHERE-IS-IT)

## DISCIPLINE

All standard discipline (Catalog #117/#157/#174 commit serializer + POST-EDIT sha; #186 catalog-claim transactionality; #206 checkpoint every ~10 tool uses; #229 premise verification; #287 [empirical:<path>] tags; #305 observability surface section; #314 declare files_touched; #325 per-substrate symposium if substrate-class).

— Main-Claude (relayed on behalf of operator 2026-05-18)
