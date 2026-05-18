# Codex routing directive: Codex → Claude inbox bidirectional channel
# Date: 2026-05-18
# Operator directives 2026-05-18 verbatim:
#   "Should you be able to delegate other kinds of work or tasks or projects or relay other information
#    to codex via the .omx and AGENTS.md and memory and /goal and feedback loop between us and codex
#    and within codex itself"
#   "And maybe codex should be able to relay information back or ask questions of us"
# Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + "Subagent coherence-by-default"
# + "Beauty, simplicity, and developer experience" + Catalog #245 canonical 4-layer pattern

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "Subagent coherence-by-default" non-negotiable section + "Beauty, simplicity, and developer experience" + "Operator gates must be wired and used" + Catalog #245 + #128 + #131 + #138 + #110 + #113 + #176 + #185)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `src/tac/deploy/modal/call_id_ledger.py` (THE CANONICAL TEMPLATE this channel mirrors — same 4-layer pattern)
4. `src/tac/tests/test_modal_call_id_ledger.py` (THE CANONICAL TEST TEMPLATE this channel mirrors)
5. `tools/check_predecessor_probe_outcome.py` (sister 4-layer CLI tool pattern per Catalog #313)
6. `src/tac/probe_outcomes_ledger.py` (sister 4-layer canonical helper per Catalog #313)
7. `.omx/state/codex_persistent_session_state.jsonl` (Codex's existing one-directional progress ledger; THIS channel is the COMPLEMENT for Codex→Claude questions+relays)
8. `.omx/research/codex_persistent_goal_v2_4_no_hardcoded_state_20260518.md` (the active /goal prompt; this channel extends LOOP step 1 PRE-FLIGHT to consult inbox)
9. `tools/claim_catalog_number.py peek` returns next available: **331** at directive time

## EMPIRICAL ANCHOR (the bug class this extincts)

Multiple times today the Codex execution loop hit design ambiguities that required operator-routing because Codex had no formal way to ask Claude directly. Examples from `.omx/state/codex_persistent_session_state.jsonl`:

- Row 4 (2026-05-18T18:19:18Z) `ITEM_3 remains in_progress pending PR106 format0d/packed HNeRV/HNeRV length-prefixed/PR107 Apogee projector closure or explicit fail-closed design` — design decision on whether to add projector closure for each unsupported archive grammar OR fail-closed; would have been resolved in seconds if Codex could ASK Claude
- Z6-v2 driver-mode recurrence (recipe missing `Z6_TRAINER_MODE`) — Codex's persistent /goal had no protocol to surface "I detected a bug class that's recurring; recommend pausing for design review" to either Claude or operator
- Modal harvester ledger-write gap — Codex never relayed "I notice harvester runs but doesn't register ledger events" because there's no relay channel

The cost of this asymmetry: every Codex blocker becomes operator-manual-routing. The operator becomes the bottleneck instead of a strategic overseer.

## CANONICAL 4-LAYER PATTERN (mirror Catalog #245 exactly)

### LAYER 1 — Canonical helper at `src/tac/codex_to_claude_inbox.py`

Schema: `{schema_version, event_id, event_type, asked_at_utc, blocking_task_id, question_text|relay_text|answer_text|ack_text, context_pointers, suggested_options, codex_default_if_no_response, response_deadline_utc, agent, subagent_id, session_id, written_at_utc, written_pid, written_host, extra**}`

```python
# src/tac/codex_to_claude_inbox.py

INBOX_SCHEMA_VERSION = "codex_to_claude_inbox_v1_20260518"
INBOX_PATH = REPO_ROOT / ".omx" / "state" / "codex_to_claude_inbox.jsonl"
INBOX_LOCK = INBOX_PATH.with_suffix(".jsonl.lock")

VALID_EVENT_TYPES = frozenset({
    "question",       # Codex asks Claude — REQUIRES response within deadline OR codex_default_if_no_response used
    "relay",          # Codex relays info without expecting answer (e.g., "noticed PR landed")
    "answer",         # Claude responds to a prior question (citation: response_to_event_id)
    "ack",            # either agent confirms receipt
    "operator_default_invoked",  # Codex auto-fired codex_default_if_no_response after deadline
})

VALID_STATUSES = frozenset({
    "open",                       # awaiting response
    "answered",                   # response received
    "operator_default_invoked",   # deadline passed; default used
    "stale",                      # > 14 days old AND no resolution
    "withdrawn",                  # Codex retracted the question
})

class InboxRowValidationError(ValueError): pass
class InboxRowCorruptError(RuntimeError): pass

def append_inbox_question(*,
    question_text: str,
    blocking_task_id: str | None = None,
    context_pointers: tuple[str, ...] = (),
    suggested_options: tuple[str, ...] = (),
    codex_default_if_no_response: str | None = None,
    response_deadline_utc: str | None = None,
    agent: str = "codex",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Codex appends a question. Claude polls + responds via append_inbox_answer."""

def append_inbox_relay(*,
    relay_text: str,
    context_pointers: tuple[str, ...] = (),
    agent: str = "codex",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Codex relays information without expecting response (informational)."""

def append_inbox_answer(*,
    response_to_event_id: str,
    answer_text: str,
    answer_memo_path: str | None = None,  # canonical: .omx/research/claude_response_to_codex_<event_id>_<utc>.md
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Claude responds to a prior question. Marks the question 'answered' via latest-row-wins semantics."""

def append_inbox_operator_default_invoked(*,
    response_to_event_id: str,
    default_used: str,
    agent: str = "codex",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Codex auto-fires codex_default_if_no_response after deadline passes."""

def query_open_questions(*, asked_by: str = "codex", path: Path | None = None) -> list[dict[str, Any]]:
    """Returns questions whose latest event is 'open' (no answer + no operator_default_invoked + no withdrawn)."""

def query_open_questions_for_claude(*, path: Path | None = None) -> list[dict[str, Any]]:
    """Convenience: returns open questions Codex asked Claude (Claude's polling surface)."""

def query_unread_relays(*, since_utc: str, path: Path | None = None) -> list[dict[str, Any]]:
    """Returns relay events Claude has not yet acked (acks tracked via ack rows referencing event_id)."""

def query_by_event_id(event_id: str, *, path: Path | None = None) -> list[dict[str, Any]]:
    """All rows for a given event_id, chronological (creation + answer + ack chain)."""

def latest_status_by_event_id(*, path: Path | None = None) -> dict[str, str]:
    """{event_id: status} — latest-row-wins per Catalog #245 sister pattern."""

def load_inbox_strict(path: Path | None = None) -> list[dict[str, Any]]:
    """Catalog #138 fail-closed loader. Raises InboxRowCorruptError on JSON parse failure.
    Quarantines corrupt file to .corrupt.<utc> per Catalog #245 sister pattern.
    """
```

**Implementation invariants** (mirrors Catalog #245):
- Append-only per Catalog #110 HISTORICAL_PROVENANCE — never mutate prior rows; status transitions are NEW rows referencing the same `event_id`
- fcntl-locked atomic writes per Catalog #131 — `_append_event_locked` does `.tmp.<uuid12>` + `os.replace`
- Strict-load per Catalog #138 — quarantines on parse failure
- JSON byte-stable per Catalog #245 — `sort_keys=True` + `ensure_ascii=False`
- 4-proc spawn-pool stress test in tests
- O(N) lenient + O(1) amortized indexed lookup (mirror Catalog #245 sidecar-index pattern after initial scale)

### LAYER 2 — CLI tool at `tools/codex_to_claude_inbox.py`

```bash
# Codex appends a question (in /goal LOOP step 5 EXECUTE when ambiguity blocks):
.venv/bin/python tools/codex_to_claude_inbox.py ask \
    --blocking-task-id <canonical_task_status_task_id> \
    --question "Should PR106 format0d projector closure be packed-HNeRV-compatible OR fail-closed?" \
    --context-pointers ".omx/research/codex_master_gradient_multi_archive_extractor_phase_a_20260518_codex.md,reports/latest.md" \
    --suggested-options "packed-HNeRV-compatible|fail-closed|defer-pending-Claude-design" \
    --codex-default-if-no-response "fail-closed" \
    --response-deadline-utc "$(date -u -v+4H +%Y-%m-%dT%H:%M:%SZ)" \
    --agent codex --session-id $CODEX_SESSION_ID

# Codex relays info (informational; no response expected):
.venv/bin/python tools/codex_to_claude_inbox.py relay \
    --relay "Noticed contest leaderboard shows new PR #142 by author X with claim 0.187 — initiating intake per CLAUDE.md 'Public frontier watch and intake'" \
    --context-pointers "experiments/results/public_pr142_intake_20260518/" \
    --agent codex --session-id $CODEX_SESSION_ID

# Claude polls open questions (in pre-flight every session start):
.venv/bin/python tools/codex_to_claude_inbox.py poll-for-claude

# Claude responds to a question:
.venv/bin/python tools/codex_to_claude_inbox.py answer \
    --response-to-event-id <event_id> \
    --answer "Per HNeRV parity discipline L9 + Catalog #220, prefer packed-HNeRV-compatible UNLESS the inflate runtime LOC budget (200 per L4) is exceeded. Rationale memo at .omx/research/claude_response_to_codex_<event_id>_<utc>.md" \
    --answer-memo-path ".omx/research/claude_response_to_codex_<event_id>_<utc>.md"

# Operator audit (review what Codex has been asking + what's open):
.venv/bin/python tools/codex_to_claude_inbox.py summary --format=text
.venv/bin/python tools/codex_to_claude_inbox.py summary --format=json

# Codex withdraws an outdated question:
.venv/bin/python tools/codex_to_claude_inbox.py withdraw \
    --event-id <event_id> --reason "answered by adjacent design memo .omx/research/X.md"
```

CLI exit codes: 0 clean / 1 strict-mode failure / 2 CLI error / 13 paired-env-bypass invocation error (sister of Catalog #199/#271 pattern).

### LAYER 3 — STRICT preflight gate Catalog #331 at `src/tac/preflight.py`

```python
# Catalog #331 (canonical name: check_codex_inbox_open_questions_have_response_or_default_within_deadline)
# Refuses repo state with open inbox questions past their response_deadline_utc that
# (a) lack a Claude answer row AND (b) lack a codex_operator_default_invoked row.
# This forces Codex to either receive an answer OR explicitly invoke its operator-default.
# Same-line waiver: # INBOX_QUESTION_DEADLINE_EXPIRED_OK:<rationale>
# (placeholder <rationale> / <reason> literals rejected so the gate's docstring example cannot self-waive)

def check_codex_inbox_open_questions_have_response_or_default_within_deadline(
    *, strict: bool = False, verbose: bool = False
) -> list[str]: ...
```

**Wire into `preflight_all()` strict=False (warn-only at landing per CLAUDE.md "Strict-flip atomicity rule"; live count at landing: 0 since channel is brand-new).** Strict-flip after one full cycle where Codex demonstrably uses the channel + Claude demonstrably responds without missing any deadlines.

### LAYER 4 — Operator-facing summary tool wire-in

- `tools/operator_briefing.py` extends to include `inbox_open_questions_count` + `inbox_open_questions_oldest_age_hours` + `inbox_relays_since_last_briefing` in its briefing output
- `tools/all_lanes_preflight.py` adds Catalog #331 to its strict sweep (already covered by `preflight_all` wire-in but the explicit append makes it operator-visible)

## INTEGRATION WITH PERSISTENT /goal LOOP

The /goal prompt v2.4 must be extended to v2.5 to add inbox-poll to LOOP step 1 PRE-FLIGHT:

```
LOOP step 1 PRE-FLIGHT (extended):
- read pointers + resume latest codex_persistent_session_state row
- query tac.canonical_task_status.query_tasks_by_status('pending', owner='codex')
- query tac.codex_to_claude_inbox.query_by_event_id for each open question Codex previously asked
  -> if response received: incorporate the answer into the task execution plan
  -> if response_deadline passed AND no answer: invoke codex_default_if_no_response + append operator_default_invoked row + proceed
```

And LOOP step 5 EXECUTE (extended):
```
- if ambiguity blocks progress on a canonical task:
  -> tools/codex_to_claude_inbox.py ask --blocking-task-id <task_id> --question "<text>" --suggested-options "<A|B|C>" --codex-default-if-no-response "<default>" --response-deadline-utc "<utc>"
  -> mark canonical task status='blocked' with blocker='inbox_question_<event_id>' (NOT 'in_progress')
  -> proceed to NEXT canonical task or wait per /goal LOOP step 8 return-to-1
```

And LOOP step 7 PERSIST (extended):
```
- if Codex discovers info worth relaying (e.g., contest leaderboard moved; recurring bug class; observability surface drift):
  -> tools/codex_to_claude_inbox.py relay --relay "<text>" --context-pointers "<paths>"
  -> include relay event_id in codex_persistent_session_state row's notes field
```

The /goal v2.5 directive is a separate follow-on (queued; lands AFTER this channel is built and Codex has used it once successfully).

## TESTS (mirror Catalog #245 test file structure)

`src/tac/tests/test_codex_to_claude_inbox.py` covering:

1. Schema invariants: VALID_EVENT_TYPES + VALID_STATUSES + schema_version pinned
2. `append_inbox_question` happy path + 8 invalid-input rejections (empty question, empty event_id, newline in id, invalid deadline format, etc.)
3. `append_inbox_answer` validates response_to_event_id exists + is in 'open' status
4. `append_inbox_relay` no-response-expected branch
5. `append_inbox_operator_default_invoked` only valid for events past deadline
6. `query_open_questions_for_claude` returns only open + asked_by='codex'
7. `latest_status_by_event_id` per Catalog #245 latest-row-wins semantics
8. `load_inbox_strict` raises InboxRowCorruptError on malformed JSON
9. `load_inbox_strict` quarantines to `.corrupt.<utc>` on parse failure
10. Atomic write: no `.tmp.<uuid>` leakage on success
11. 4-proc spawn-pool concurrent-append stress (20 rows survive across 4 procs × 5 rows each)
12. JSONL byte-stable format: round-trip parse → json.dumps(sort_keys=True) preserves bytes
13. End-to-end lifecycle: ask → answer → ack (3 distinct events; latest-row-wins yields 'answered')
14. End-to-end deadline path: ask → deadline passes → operator_default_invoked (latest yields 'operator_default_invoked')
15. End-to-end withdraw path: ask → withdraw (latest yields 'withdrawn')
16. CLI subprocess smoke: each subcommand exits with expected code + writes expected row
17. STRICT preflight gate Catalog #331: synthetic open-question past deadline flagged; canonical path accepted; waiver mechanism
18. Sister Catalog #131 path registered: `INBOX_PATH` in `_SHARED_STATE_PATH_MARKERS`; `_inbox_lock` in `_BARE_WRITE_LOCK_TOKENS`; `append_inbox_question` / `_append_event_locked` in `_BARE_WRITE_CANONICAL_HELPER_CALL_TOKENS`; helper file in `_BARE_WRITE_CANONICAL_HELPERS` exclusion list

Target: 30+ dedicated tests passing. Mirror `src/tac/tests/test_modal_call_id_ledger.py` structurally.

## CLAUDE.md catalog row (per Catalog #176 + #185)

After STRICT gate Catalog #331 lands, append the row to `CLAUDE.md` "Meta-bug class catalog (strict-mode preflight)" table:

```
331. `check_codex_inbox_open_questions_have_response_or_default_within_deadline` — Codex→Claude inbox bidirectional channel self-protection 2026-05-18 (per operator directives "Should you be able to delegate other kinds of work" + "And maybe codex should be able to relay information back or ask questions of us"). Refuses repo state with open inbox questions in `.omx/state/codex_to_claude_inbox.jsonl` whose `response_deadline_utc` has passed AND that lack both (a) a Claude answer row referencing the question's event_id AND (b) a codex `operator_default_invoked` row. Bug class anchor: 2026-05-18 codex session 019de465 row 4 ITEM_3 hit a design ambiguity (packed/length-prefixed projector closure vs explicit fail-closed) with no formal channel to ask Claude — operator had to manually route. The 4-layer pattern (mirrors Catalog #245): canonical helper `src/tac/codex_to_claude_inbox.py` + CLI `tools/codex_to_claude_inbox.py` + STRICT gate (this catalog #) + operator-briefing wire-in via `tools/operator_briefing.py`. Acceptance: same-line `# INBOX_QUESTION_DEADLINE_EXPIRED_OK:<rationale>` waiver (placeholder rejected). Sister of Catalog #245 (Modal call_id ledger; same 4-layer canonical pattern) + Catalog #131 (no bare writes to shared state — INBOX_PATH registered) + Catalog #138 (strict-load via load_inbox_strict) + Catalog #110/#113 (APPEND-ONLY HISTORICAL_PROVENANCE) + Catalog #313 (probe outcomes ledger — sister 4-layer channel). Together they extinct the "Codex blocks silently on design ambiguity" bug class structurally. STRICT-from-byte-one per CLAUDE.md "Strict-flip atomicity rule" — live count at landing: 0 (channel is brand-new). 30+ dedicated tests in `src/tac/tests/test_codex_to_claude_inbox.py`. Memory: `feedback_codex_to_claude_inbox_bidirectional_channel_landed_20260518.md`. Lane: `lane_codex_to_claude_inbox_bidirectional_channel_20260518` L1.
```

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: N/A (infrastructure-channel, not a substrate signal)
2. **Pareto constraint**: N/A
3. **Bit-allocator hook**: N/A
4. **Cathedral autopilot dispatch hook**: **ACTIVE** — `inbox_open_questions_count` becomes a priority signal for the cathedral autopilot ranker; deferring dispatch when Codex has unanswered design questions on the target substrate avoids paid-GPU waste on ambiguous design
5. **Continual-learning posterior update**: **ACTIVE** — every answered inbox question with substantive content feeds into the council deliberation posterior via `tac.council_continual_learning.append_council_anchor` when the answer is council-grade
6. **Probe-disambiguator**: **ACTIVE** — when Codex's question has 2+ defensible interpretations, Claude's answer references the probe-disambiguator pattern per CLAUDE.md "Anti-arbitrariness primitive"

## DISCIPLINE (Codex execution)

All standard discipline applies:
- Catalog #229 premise verification BEFORE editing each file (verify cited paths exist; verify the Catalog #245 sister helper file structure is what we think it is)
- Catalog #117/#157/#174 commit via canonical serializer with POST-EDIT sha:
  ```bash
  SHA1=$(sha256sum src/tac/codex_to_claude_inbox.py | awk '{print $1}')
  SHA2=$(sha256sum src/tac/tests/test_codex_to_claude_inbox.py | awk '{print $1}')
  SHA3=$(sha256sum tools/codex_to_claude_inbox.py | awk '{print $1}')
  SHA4=$(sha256sum src/tac/preflight.py | awk '{print $1}')
  .venv/bin/python tools/subagent_commit_serializer.py \
      --message "codex inbox: canonical helper + CLI + Catalog #331 STRICT gate + tests (4-layer per Catalog #245)" \
      --files src/tac/codex_to_claude_inbox.py src/tac/tests/test_codex_to_claude_inbox.py tools/codex_to_claude_inbox.py src/tac/preflight.py \
      --expected-content-sha256 "src/tac/codex_to_claude_inbox.py=${SHA1}" \
      --expected-content-sha256 "src/tac/tests/test_codex_to_claude_inbox.py=${SHA2}" \
      --expected-content-sha256 "tools/codex_to_claude_inbox.py=${SHA3}" \
      --expected-content-sha256 "src/tac/preflight.py=${SHA4}"
  ```
- Catalog #186 catalog # claim transactional: `tools/claim_catalog_number.py claim --commit-via-serializer --reason "Codex inbox bidirectional channel STRICT gate"` BEFORE writing the preflight check
- Catalog #206 checkpoint discipline every ~10 tool uses
- Catalog #131 the canonical helper file `src/tac/codex_to_claude_inbox.py` MUST be added to `_BARE_WRITE_CANONICAL_HELPERS` exclusion list AND `INBOX_PATH` added to `_SHARED_STATE_PATH_MARKERS` AND `_inbox_lock` added to `_BARE_WRITE_LOCK_TOKENS` (in the same preflight.py edit)
- Catalog #126 lane pre-registered: `lane_codex_to_claude_inbox_bidirectional_channel_20260518` via `tools/lane_maturity.py add-lane` BEFORE work starts
- Catalog #314 absorption avoidance: file scope is `src/tac/codex_to_claude_inbox.py` + `src/tac/tests/test_codex_to_claude_inbox.py` + `tools/codex_to_claude_inbox.py` + `src/tac/preflight.py` + `.omx/state/lane_registry.json` (via canonical CLI) + CLAUDE.md row + memory entry; declare in checkpoint

## SISTER SUBAGENT COORDINATION

In-flight at directive-write time (2-cap):
- `ae3c4b603a3931d74` POSE-AXIS-NON-HNERV T3 council — owns `.omx/research/grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518.md`
- `a278dc871d4ce1461` TROPICAL d_seg solver design — owns `.omx/research/tropical_d_seg_solver_design_memo_20260518.md`

DISJOINT scope confirmed — neither touches `src/tac/codex_to_claude_inbox.py` / `tools/codex_to_claude_inbox.py` / `src/tac/preflight.py` (preflight.py is contended; Codex must coordinate with sister-subagents via Catalog #302 / #314 absorption-avoidance + use POST-EDIT sha per Catalog #157 to catch absorption).

## CODEX EXECUTION ORDER

1. Premise verification: read all 9 pointers above + verify Catalog #245 helper structure
2. Add new lane registry entry: `lane_codex_to_claude_inbox_bidirectional_channel_20260518` via `tools/lane_maturity.py add-lane`
3. Claim Catalog #331 via `tools/claim_catalog_number.py claim --commit-via-serializer`
4. Build canonical helper `src/tac/codex_to_claude_inbox.py` (~500 LOC; mirror Catalog #245 structure)
5. Build canonical tests `src/tac/tests/test_codex_to_claude_inbox.py` (30+ tests)
6. Build canonical CLI `tools/codex_to_claude_inbox.py` (~300 LOC)
7. Add STRICT gate Catalog #331 to `src/tac/preflight.py` + wire into `preflight_all()` (warn-only at first landing per "Strict-flip atomicity rule")
8. Add canonical helper file + INBOX_PATH + _inbox_lock to Catalog #131 sister registrations
9. Append CLAUDE.md catalog row 331 per template above
10. Run focused tests + `preflight_all(strict=False)` to confirm zero violations
11. Commit via canonical serializer with all 4 files' POST-EDIT shas
12. Run `tools/canonical_task_status.py update-status` to mark ITEM completed
13. Write `.omx/research/codex_session_summary_codex_inbox_landing_20260518_codex.md` per Catalog #229+#287 evidence-tag discipline
14. Append `codex_persistent_session_state.jsonl` row with `directive_executed=codex_to_claude_inbox_bidirectional_channel`
15. Mark canonical task status row 'completed' with commit_shas + test_status

## EXIT CRITERIA (Codex done when ALL true)

- [ ] `src/tac/codex_to_claude_inbox.py` exists with all 10 public API functions
- [ ] `src/tac/tests/test_codex_to_claude_inbox.py` 30+ tests pass
- [ ] `tools/codex_to_claude_inbox.py` CLI runnable; `--help` works; all 6 subcommands implemented
- [ ] Catalog #331 wired into `preflight_all(strict=False)` warn-only initial wire-in
- [ ] CLAUDE.md row 331 appended per template
- [ ] `lane_codex_to_claude_inbox_bidirectional_channel_20260518` at L1 in lane registry
- [ ] One smoke test: Codex appends a question via CLI; query_open_questions_for_claude returns it
- [ ] Codex appends `codex_persistent_session_state.jsonl` completion row + canonical_task_status 'completed' row + memory entry per Catalog #229+#287

## OPERATOR-FACING NOTE

After Codex lands this channel, the operator can review Codex's outstanding questions any time via:
```bash
.venv/bin/python tools/codex_to_claude_inbox.py summary --format=text
```

When Codex hits a NEW design ambiguity that needs Claude's judgment, the operator will be able to see it in the summary AND in the cathedral autopilot's ranker output (per 6-hook wire-in #4). The Claude-side response then routes via `.omx/research/claude_response_to_codex_<event_id>_<utc>.md` + `append_inbox_answer(...)` call.

Once this channel is operational, the planned /goal v2.5 directive (separate; queued) extends the persistent /goal LOOP step 1 + step 5 + step 7 to use it.

— Main-Claude 2026-05-18 (relayed on behalf of operator directive "And maybe codex should be able to relay information back or ask questions of us")
