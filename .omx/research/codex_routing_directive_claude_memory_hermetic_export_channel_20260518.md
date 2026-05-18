# Codex routing directive: Claude memory hermetic export channel
# Date: 2026-05-18
# Operator: approved 2026-05-18 ("also write the directives for ... the memory hermetic export channel now")
# Sister of Codex→Claude inbox channel directive (commit 745fc2e19) — same Catalog #245 4-layer pattern
# Per CLAUDE.md "Subagent coherence-by-default" + "Beauty, simplicity, and developer experience" + OSS-hermetic discipline

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "Subagent coherence-by-default" + Catalog #290/#291/#292 OSS-hermetic policy + Catalog #245 canonical 4-layer pattern + Catalog #128 + #131 + #138 + #110 + #113 + #176 + #185)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `src/tac/deploy/modal/call_id_ledger.py` (THE CANONICAL TEMPLATE this channel mirrors)
4. `src/tac/codex_to_claude_inbox.py` (sister channel; built per directive 745fc2e19; mirror its structure)
5. `.omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` (sister directive; same template applied)
6. `tools/claim_catalog_number.py peek` returns next available after #331: **332** at directive time

## EMPIRICAL ANCHOR (the bug class this extincts)

Claude maintains ~504 KB of context in `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md` + per-topic memory files at `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_*.md`. These are machine-local + OSS-hermetic-policy-blocked from Codex per Catalog #290/#291/#292 (external `memory_dir=...` requires explicit opt-in; default-disabled to keep CI + clean clones hermetic).

The current workaround is Claude paraphrasing relevant memory entries into `codex_routing_directive_*.md` memos — ad-hoc, lossy, and burns Claude operator-attention every time the same topic recurs across multiple directives.

**Example signal loss**: today's Z6-v2 driver-mode recurrence — Claude has a memory entry `feedback_driver_fix_smoke_hardcode_plus_new_catalog_gate_cross_substrate_audit_landed_20260518.md` that explains the canonical fix pattern, but Codex never sees it. When the recurrence hit, Codex had to re-derive the canonical fix from CLAUDE.md fragments + the catalog entry.

The hermetic export channel makes relevant memory queryable + hermetic.

## CANONICAL 4-LAYER PATTERN (mirror Catalog #245 + sister channel exactly)

### LAYER 1 — Canonical helper at `src/tac/claude_memory_export.py`

Schema: `{schema_version, export_id, exported_at_utc, topic, source_memory_files, canonical_summary, citation_chain, shelf_life_days, agent, subagent_id, session_id, written_at_utc, written_pid, written_host, extra**}`

```python
# src/tac/claude_memory_export.py

EXPORT_SCHEMA_VERSION = "claude_memory_export_v1_20260518"
EXPORT_PATH = REPO_ROOT / ".omx" / "state" / "claude_memory_export.jsonl"
EXPORT_LOCK = EXPORT_PATH.with_suffix(".jsonl.lock")

VALID_EVENT_TYPES = frozenset({
    "export",            # Claude exports new memory summary for Codex consumption
    "supersede",         # Claude exports new version superseding prior export (citation: supersedes_export_id)
    "ack",               # Codex acks consumption of export (telemetry; informational)
    "retract",           # Claude retracts export (e.g., underlying memory revised; export no longer canonical)
})

VALID_STATUSES = frozenset({
    "active",            # current canonical version for topic
    "superseded",        # superseded by newer export (latest-row-wins)
    "retracted",         # explicitly retracted
    "expired",           # past shelf_life_days
})

class ExportRowValidationError(ValueError): pass
class ExportRowCorruptError(RuntimeError): pass

DEFAULT_SHELF_LIFE_DAYS = 30  # per CLAUDE.md "Substrate retirement discipline" 30-day window sister

def append_memory_export(*,
    topic: str,                              # canonical slug (e.g., "z6_v2_driver_mode_recurrence_canonical_fix")
    source_memory_files: tuple[str, ...],    # paths under memory dir (Claude-side; relative format)
    canonical_summary: str,                  # 100-2000 chars distilled summary
    citation_chain: tuple[str, ...],         # memory slugs cited (sister of memory file [[name]] cross-refs)
    shelf_life_days: int = DEFAULT_SHELF_LIFE_DAYS,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Claude appends a new memory export. Topic is the canonical key (one active row per topic via latest-row-wins).

    Validates:
    - topic is canonical slug (snake_case ASCII; matches r'^[a-z][a-z0-9_]*$')
    - source_memory_files non-empty (provenance required per Catalog #287)
    - canonical_summary len in [100, 2000]
    - citation_chain non-empty (forces traceability back to source memory files)
    - shelf_life_days in [1, 365]
    """

def append_memory_export_supersede(*,
    topic: str,
    supersedes_export_id: str,
    source_memory_files: tuple[str, ...],
    canonical_summary: str,
    citation_chain: tuple[str, ...],
    shelf_life_days: int = DEFAULT_SHELF_LIFE_DAYS,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Claude supersedes a prior export. Marks prior 'superseded' via latest-row-wins."""

def append_memory_export_ack(*,
    export_id: str,
    agent: str = "codex",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Codex acks consumption (telemetry; informational)."""

def append_memory_export_retract(*,
    export_id: str,
    retraction_reason: str,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Claude retracts an export (e.g., underlying memory revised)."""

def load_recent(topic: str, *, max_age_days: int = DEFAULT_SHELF_LIFE_DAYS, path: Path | None = None) -> dict[str, Any] | None:
    """Codex's primary read API: returns the current ACTIVE canonical export for a topic, or None.
    Respects max_age_days (returns None if export is older than threshold even if still 'active' by status).
    """

def query_by_topic(topic: str, *, path: Path | None = None) -> list[dict[str, Any]]:
    """All rows for a topic chronologically (export + supersede + ack + retract chain)."""

def query_active_topics(*, path: Path | None = None) -> list[str]:
    """All topics with at least one active export."""

def query_topics_needing_refresh(*, max_age_days: int = DEFAULT_SHELF_LIFE_DAYS, path: Path | None = None) -> list[str]:
    """Topics whose latest active export is past max_age_days (Claude polls + refreshes)."""

def latest_status_by_topic(*, path: Path | None = None) -> dict[str, str]:
    """{topic: status} — latest-row-wins per Catalog #245 sister pattern."""

def load_exports_strict(path: Path | None = None) -> list[dict[str, Any]]:
    """Catalog #138 fail-closed loader. Raises ExportRowCorruptError on JSON parse failure.
    Quarantines corrupt file to .corrupt.<utc> per Catalog #245 sister pattern.
    """
```

**Implementation invariants** (mirror Catalog #245 + sister channel):
- Append-only per Catalog #110 HISTORICAL_PROVENANCE
- fcntl-locked atomic writes per Catalog #131 — `_append_event_locked` does `.tmp.<uuid12>` + `os.replace`
- Strict-load per Catalog #138 — quarantines on parse failure
- JSON byte-stable per Catalog #245 — `sort_keys=True` + `ensure_ascii=False`
- 4-proc spawn-pool stress test in tests
- `topic` slug uniqueness enforced via latest-row-wins (one ACTIVE row per topic at any time)

### LAYER 2 — CLI tool at `tools/claude_memory_export.py`

```bash
# Claude exports memory summary on demand (typically after a significant memory landing):
.venv/bin/python tools/claude_memory_export.py export \
    --topic "z6_v2_driver_mode_recurrence_canonical_fix" \
    --source-memory-files "feedback_driver_fix_smoke_hardcode_plus_new_catalog_gate_cross_substrate_audit_landed_20260518.md" \
    --canonical-summary "$(cat <<'EOF'
Z6-v2 driver-mode bug class: recipe env_overrides must declare Z6_TRAINER_MODE explicitly.
Canonical fix lives at scripts/remote_lane_substrate_time_traveler_l5_z6.sh (Z6_TRAINER_MODE env consumption with explicit precedence over SMOKE_ONLY default).
Recipe-side fix: set Z6_TRAINER_MODE: "full" + SMOKE_ONLY: "0" in env_overrides block.
STRICT preflight gate: Catalog #326 check_substrate_driver_consumes_trainer_mode_env_var.
Cross-substrate audit tool: tools/audit_substrate_driver_mode_hardcode.py (7-verdict taxonomy).
EOF
)" \
    --citation-chain "feedback_driver_fix_smoke_hardcode_plus_new_catalog_gate_cross_substrate_audit_landed_20260518" \
    --shelf-life-days 30 \
    --agent claude --session-id $CLAUDE_SESSION_ID

# Codex reads at /goal LOOP pre-flight (programmatic via canonical helper):
python3 -c "from tac.claude_memory_export import load_recent; print(load_recent('z6_v2_driver_mode_recurrence_canonical_fix'))"

# Codex acks consumption (telemetry):
.venv/bin/python tools/claude_memory_export.py ack --export-id <event_id> --agent codex --session-id $CODEX_SESSION_ID

# Claude supersedes:
.venv/bin/python tools/claude_memory_export.py supersede --supersedes-export-id <prior_event_id> --topic <same_topic> --canonical-summary "..." --source-memory-files "..." --citation-chain "..."

# Operator audit:
.venv/bin/python tools/claude_memory_export.py summary --format=text
.venv/bin/python tools/claude_memory_export.py summary --format=json

# Claude polls topics needing refresh (in session start; updates stale topics):
.venv/bin/python tools/claude_memory_export.py topics-needing-refresh --max-age-days 30

# Claude retracts (e.g., underlying memory rewritten):
.venv/bin/python tools/claude_memory_export.py retract --export-id <event_id> --retraction-reason "underlying memory rewritten; new export pending"
```

CLI exit codes: 0 clean / 1 strict-mode failure / 2 CLI error.

### LAYER 3 — STRICT preflight gate Catalog #332 at `src/tac/preflight.py`

```python
# Catalog #332 (canonical name: check_claude_memory_export_citation_chain_traceable)
# Refuses export rows whose citation_chain references memory slugs that DON'T appear in source_memory_files
# (forces traceability back to actual memory files per Catalog #287 evidence-tag discipline)
# Same-line waiver: # MEMORY_EXPORT_CITATION_CHAIN_OK:<rationale>
# (placeholder <rationale> / <reason> literals rejected so the gate's docstring example cannot self-waive)

def check_claude_memory_export_citation_chain_traceable(
    *, strict: bool = False, verbose: bool = False
) -> list[str]: ...
```

**Wire into `preflight_all()` strict=False (warn-only at landing per CLAUDE.md "Strict-flip atomicity rule"; live count at landing: 0 since channel is brand-new).** Strict-flip after first 5 successful exports.

### LAYER 4 — Operator-facing summary tool wire-in

- `tools/operator_briefing.py` extends to include `memory_export_active_topics_count` + `memory_export_oldest_age_days` + `memory_export_topics_needing_refresh` in its briefing output

## INTEGRATION WITH PERSISTENT /goal LOOP

Codex's /goal v2.5 already lists `.omx/state/claude_memory_export.jsonl` in POINTERS (when v2.5 is paste-activated). Add explicit step:

```
LOOP step 1 PRE-FLIGHT (further extended after v2.5 lands):
- ... existing v2.5 steps ...
- query tac.claude_memory_export.query_active_topics for topics relevant to current pending tasks
  -> if relevant topic exists: load_recent(topic) + incorporate canonical_summary into task execution context
  -> emit ack via append_memory_export_ack for telemetry
```

Sister /goal extension is queued as v2.6 (separate; lands after this channel + 5 successful exports).

## CLAUDE-SIDE USAGE PATTERN

Main-Claude curates exports at high-value moments:
- After a significant memory landing that distills a recurring pattern (e.g., `Z6-v2 driver-mode recurrence canonical fix`)
- After a council deliberation produces a binding architectural decision (e.g., `pose-axis cheap-probe family cross-stack synergy with master-gradient extractor`)
- After a forensic investigation produces a canonical fix pattern (e.g., `commit-swap absorption-pattern Catalog #314 protection`)

Anti-pattern (FORBIDDEN): bulk-dumping all of MEMORY.md as exports. The hermetic export channel exists for HAND-CURATED, COUNCIL-GRADE knowledge transfer; not for transcript syncing.

## TESTS (mirror Catalog #245 + sister channel test structure)

`src/tac/tests/test_claude_memory_export.py` covering:

1. Schema invariants: VALID_EVENT_TYPES + VALID_STATUSES + schema_version pinned + DEFAULT_SHELF_LIFE_DAYS=30
2. `append_memory_export` happy path + 8 invalid-input rejections (empty topic, non-snake-case topic, empty source_memory_files, short canonical_summary < 100 chars, long > 2000 chars, empty citation_chain, shelf_life_days out of range, etc.)
3. `append_memory_export_supersede` validates supersedes_export_id exists + topic matches
4. `append_memory_export_ack` only valid for existing active exports
5. `append_memory_export_retract` only valid for active/superseded exports
6. `load_recent` returns None when no export exists for topic
7. `load_recent` returns None when latest export past max_age_days even if status='active'
8. `load_recent` returns active export within shelf life
9. `query_active_topics` returns only topics with at least one active export
10. `query_topics_needing_refresh` returns topics whose latest active export is past threshold
11. `latest_status_by_topic` per Catalog #245 latest-row-wins semantics
12. `load_exports_strict` raises ExportRowCorruptError on malformed JSON
13. `load_exports_strict` quarantines to `.corrupt.<utc>` on parse failure
14. Atomic write: no `.tmp.<uuid>` leakage on success
15. 4-proc spawn-pool concurrent-append stress (20 rows survive across 4 procs × 5 rows each)
16. JSONL byte-stable format: round-trip parse → json.dumps(sort_keys=True) preserves bytes
17. End-to-end lifecycle: export → supersede → ack (3 distinct events; latest-row-wins yields active version)
18. End-to-end retraction: export → retract (latest yields 'retracted')
19. Citation chain traceability: citation_chain entries match source_memory_files basenames (sister of Catalog #332 STRICT gate)
20. CLI subprocess smoke: each subcommand exits with expected code + writes expected row
21. STRICT preflight gate Catalog #332: synthetic export with untraceable citation_chain flagged; canonical accepted; waiver mechanism
22. Sister Catalog #131 path registered: `EXPORT_PATH` in `_SHARED_STATE_PATH_MARKERS`; `_export_lock` in `_BARE_WRITE_LOCK_TOKENS`; `append_memory_export` / `_append_event_locked` in `_BARE_WRITE_CANONICAL_HELPER_CALL_TOKENS`; helper file in `_BARE_WRITE_CANONICAL_HELPERS` exclusion list

Target: 30+ dedicated tests passing.

## CLAUDE.md catalog row (per Catalog #176 + #185)

After STRICT gate Catalog #332 lands:

```
332. `check_claude_memory_export_citation_chain_traceable` — Claude memory hermetic export channel self-protection 2026-05-18 (per operator directive "also write the directives for ... the memory hermetic export channel now"). Refuses memory export rows in `.omx/state/claude_memory_export.jsonl` whose `citation_chain` references memory slugs that don't appear in `source_memory_files` (forces traceability back to actual memory files per Catalog #287 evidence-tag discipline). Bug class anchor: Claude's machine-local ~504 KB of memory context is hermetic-blocked from Codex per Catalog #290/#291/#292; ad-hoc paraphrasing in routing directives is lossy. The 4-layer pattern (mirrors Catalog #245 + sister Catalog #331): canonical helper `src/tac/claude_memory_export.py` + CLI `tools/claude_memory_export.py` + STRICT gate (this catalog #) + operator-briefing wire-in via `tools/operator_briefing.py`. Acceptance: same-line `# MEMORY_EXPORT_CITATION_CHAIN_OK:<rationale>` waiver (placeholder rejected). Sister of Catalog #245 (Modal call_id ledger; same 4-layer canonical pattern) + Catalog #331 (Codex→Claude inbox bidirectional channel; same 4-layer pattern) + Catalog #131 (no bare writes to shared state) + Catalog #138 (strict-load) + Catalog #110/#113 (APPEND-ONLY HISTORICAL_PROVENANCE) + Catalog #290/#291/#292 (OSS-hermetic policy). Together they close the bidirectional Claude↔Codex knowledge-transfer surface hermetically. STRICT-from-byte-one per CLAUDE.md "Strict-flip atomicity rule" — live count at landing: 0 (channel is brand-new). 30+ dedicated tests in `src/tac/tests/test_claude_memory_export.py`. Memory: `feedback_claude_memory_hermetic_export_channel_landed_20260518.md`. Lane: `lane_claude_memory_hermetic_export_channel_20260518` L1.
```

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: N/A (infrastructure channel)
2. **Pareto constraint**: N/A
3. **Bit-allocator hook**: N/A
4. **Cathedral autopilot dispatch hook**: **ACTIVE** — `topics_needing_refresh` count is a freshness signal; autopilot ranker can deprioritize tasks whose canonical knowledge is stale
5. **Continual-learning posterior update**: **ACTIVE** — memory exports referencing council deliberations feed back into `tac.council_continual_learning` via citation_chain → council_anchor lookup
6. **Probe-disambiguator**: **ACTIVE** — when Codex's task has 2+ defensible interpretations AND a relevant memory export exists, Claude's export IS the disambiguator

## DISCIPLINE (Codex execution; mirror sister inbox channel directive)

All standard discipline applies:
- Catalog #229 premise verification BEFORE editing each file
- Catalog #117/#157/#174 commit via canonical serializer with POST-EDIT sha
- Catalog #186 catalog # claim transactional: `tools/claim_catalog_number.py claim --commit-via-serializer --reason "Claude memory hermetic export channel STRICT gate"`
- Catalog #206 checkpoint discipline every ~10 tool uses
- Catalog #131 the canonical helper file `src/tac/claude_memory_export.py` MUST be added to `_BARE_WRITE_CANONICAL_HELPERS` exclusion list AND `EXPORT_PATH` added to `_SHARED_STATE_PATH_MARKERS` AND `_export_lock` added to `_BARE_WRITE_LOCK_TOKENS` (in the same preflight.py edit)
- Catalog #126 lane pre-registered: `lane_claude_memory_hermetic_export_channel_20260518` via `tools/lane_maturity.py add-lane`
- Catalog #314 absorption avoidance: same scope as sister inbox channel (Codex owns the canonical helper + CLI + STRICT gate + tests + preflight.py edit; Claude owns ONLY the exports written via the CLI)

## EXIT CRITERIA (Codex done when ALL true)

- [ ] `src/tac/claude_memory_export.py` exists with all 10 public API functions
- [ ] `src/tac/tests/test_claude_memory_export.py` 30+ tests pass
- [ ] `tools/claude_memory_export.py` CLI runnable; all 7 subcommands implemented
- [ ] Catalog #332 wired into `preflight_all(strict=False)` warn-only initial wire-in
- [ ] CLAUDE.md row 332 appended per template
- [ ] `lane_claude_memory_hermetic_export_channel_20260518` at L1 in lane registry
- [ ] One smoke test: Claude exports a sample memory entry via CLI; `load_recent` returns it
- [ ] Codex appends `codex_persistent_session_state.jsonl` completion row + canonical_task_status 'completed' row + memory entry per Catalog #229+#287

## SISTER SUBAGENT COORDINATION

In-flight at directive-write time:
- Codex session `019de465` continuing ITEM_3 master-gradient extractor
- `a278dc871d4ce1461` TROPICAL d_seg solver design — owns `.omx/research/tropical_d_seg_solver_design_memo_20260518.md`

DISJOINT scope: this directive routes a brand-new file `src/tac/claude_memory_export.py` + sister artifacts; no overlap with in-flight subagents. Preflight.py is contended (Codex's sister edits + this directive's edits); MUST coordinate via POST-EDIT sha per Catalog #157 to catch absorption per Catalog #314.

## OPERATOR-FACING NOTE

After Codex lands this channel, the operator can review Claude's outstanding memory exports any time via:
```bash
.venv/bin/python tools/claude_memory_export.py summary --format=text
```

Claude curates new exports on demand at high-value moments (post-council, post-investigation, post-canonical-fix-landing). The hermetic export channel makes Claude's MEMORY.md knowledge queryable + APPEND-ONLY + auditable for Codex AND CI AND clean clones AND future Claude sessions — without ever requiring external `memory_dir=...` paths.

## CATALOG # CLARIFICATION (added post multi-loop /goal F landing 2026-05-18 commit 38db94424)

The literal Catalog # numbers cited in this directive (#332 for `check_claude_memory_export_citation_chain_traceable`) were INFORMATIONAL HINTS at directive-write time. **#332 has been CLAIMED transactionally** by the multi-loop Codex /goal design memo (commit 38db94424, lane `lane_multi_loop_codex_goal_design_20260518`) for `check_codex_loop_coordination_no_file_collision`. Codex MUST resolve the actual catalog # for THIS directive via:

```bash
.venv/bin/python tools/claim_catalog_number.py claim --commit-via-serializer --reason "Claude memory hermetic export channel STRICT gate"
```

The canonical helper returns the next available number (likely #333 or higher at execution time), committed transactionally per Catalog #186. Substitute the returned number into the gate function name + CLAUDE.md row template + test fixtures + sister inbox-channel directive cross-references. Catalog #118 no-duplicate-numbers protection is structurally extincted by this transactional pattern.

Sister directives with the same informational-hint pattern:
- `.omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` (proposed #331; resolves next-available)
- `.omx/research/codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518.md` (proposed #333; resolves next-available)

— Main-Claude 2026-05-18 (relayed on behalf of operator directive "also write the directives for ... the memory hermetic export channel now")
