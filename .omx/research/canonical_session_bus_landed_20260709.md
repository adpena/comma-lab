# Canonicalization Unit 3 — the session-bus coordination pair (LANDED 2026-07-09)

**Task:** #388 (operator GO 2026-07-09 "Go on building all"). $0, no GPU, run dirs read-only.
**Pointer:** 0.19110 UNMOVED — this is coordination APPARATUS (a MEANS), not an exact-eval row.

## What & why

Two receipts from this campaign, both cross-agent coordination failures the apparatus could not
observe:

1. **STALENESS.** During the P6 seal, review round 1 found SYNTHESIS_v3's `b_c` section stale
   because a SIBLING agent landed the #386 gate MID-ROUND. Long-running agents had no channel to
   learn "a gate just RULED / a verdict LANDED / a spec was EDITED" short of re-grepping the repo.
2. **RECOVERY.** A session limit killed 3 agents mid-flight; recovery meant hand-reconstructing
   each one's context. Sister OPEN failure-ledger class:
   `sigurg_144_harness_kills_bg_bash_process_group`.

## What landed

`src/tac/session_bus/` package (fcntl-locked, LIVE_STATE, gitignored — not durable authority):

- **`bulletin.py`** — broadcast event feed over `.omx/state/session_events.jsonl`.
  - `post_event(kind, subject, payload, agent_label)` — closed 7-kind enum
    `{gate_ruled, verdict_landed, spec_edited, memo_landed, agent_spawned, agent_completed,
    agent_died}`; refuses unknown kinds / empty subject / empty agent_label / non-dict or
    non-serializable payload.
  - `events_since(marker, kinds)` — `marker` is `None` (all) | `int` offset (resume-exact) |
    ISO-8601 str (strictly-after). `event_count()` gives the current offset high-water mark.
  - `staleness_check(subjects, since, kinds) -> StalenessReport` — the seal-round primitive.
    Both-direction case-insensitive substring match against each event's `subject` and any
    `payload["subjects"]`; `.stale` / `.events` / `.matched_subjects` / `.describe()`.
  - Corruption model: `json.dumps` escapes embedded newlines so every row is exactly one physical
    line; the lenient reader skips malformed/torn/wrong-schema lines. A process killed mid-append
    corrupts only its own torn last line — no valid row is ever lost or corrupted.

- **`recovery_manifest.py`** — a THIN layer OVER the canonical `.omx/state/subagent_progress.jsonl`
  crash-resume store (Catalog #206). **P1 one-fact-one-store:** it writes through the tool's
  `append_checkpoint` and reads through its `read_checkpoints` (same file, same lock) — NOT a
  parallel store.
  - `register_inflight(subagent_id, respawn_context, ...)`, `heartbeat(...)`, `complete(...)`.
  - `recover_report(stale_after_seconds, now) -> [RecoveryEntry]` — for every agent whose latest
    checkpoint is not `complete` and whose last heartbeat is older than the window (or has an
    unparseable timestamp → surfaced, not hidden), sorted most-stale-first.
    `RecoveryEntry.render()` emits a ready-to-paste respawn block.
  - **Carry-forward (round-1 self-review catch):** fat context
    (`respawn_context`/`expected_outputs`/`lane_id`/`parent`) is taken from the most-recent
    NON-NULL value across the agent's history, so terse heartbeats never erase the context the
    initial `register_inflight` carried.

- **`tools/subagent_checkpoint.py`** (additive, legacy-compatible): `append_checkpoint` gained
  optional `respawn_context` + `expected_outputs`; CLI gained `--respawn-context` +
  `--expected-outputs`. Records written before these fields still load (readers `.get(...)`).

- **`tools/session_recover.py`** — CLI: `report | register | complete`.

- **Producer wired:** `tac.review_counter.record_round` now posts a `verdict_landed` bulletin
  event AFTER the durable ledger write, inside a fail-open `try/except`. The bulletin is
  score-neutral observability; a bulletin failure can never break the counter because the
  authority row already landed.

**Untouched (sealed):** the launch path (`tools/launch_witness_run.py`). **Not imported:**
`src/tac/through_r/`, `src/tac/verdicts/` (parallel siblings; #389 wires — `# TODO(#389)` N/A here,
no stub needed).

## Verification

- `tests/test_session_bus.py`: **32 tests** — locked 2-process concurrent appends (no lost rows) ·
  events_since offset/timestamp/kinds/negative/bool-guard · staleness hit/miss/payload-subjects/
  since-offset/kinds/empty-guard · partial-line + wrong-schema tolerance · register/heartbeat/
  complete round-trip · recovery on synthetic rows (stale/recent/complete-excluded/unparseable/
  latest-supersession/most-stale-first) · fail-open producer (counter survives a raising bulletin) ·
  CLI smoke. **ruff clean.**
- Regression: **54** `test_subagent_checkpoint*` + **14** `test_review_counter` GREEN.
- End-to-end: `tools/session_recover.py report` runs against the real store and surfaces 80
  genuinely-stale historical checkpoints (tool works live).

## Round-1 self-adversarial review (attacks + dispositions)

- **Lock contention** → 2 real processes × 30 appends, zero loss (fcntl LOCK_EX serializes).
- **Partial-line JSONL corruption** → one physical line per row + lenient reader; tested.
- **Fail-open swallowing real errors** → ACCEPTED risk: the swallowed path is a score-neutral
  notification AFTER the authoritative write; documented at the callsite (operating manual §8.9).
- **P1 duplicate-store risk** → recovery_manifest reuses the SAME store/lock as subagent_progress;
  bulletin is a DIFFERENT fact (broadcast vs. resume-checkpoint), boundaries documented in both
  module docstrings.
- **respawn_context lost on terse heartbeat** → FOUND and FIXED (carry-forward of last non-null).
- **Known tradeoff (documented, not a bug):** `staleness_check` both-direction substring is robust
  to id-vs-label drift but over-broad for 1-char query subjects; intended subjects are meaningful
  ids (`#386`, `b_c`).

## Producer / consumer list for #389 (the wiring pass)

**Producers of bulletin events (post_event) — currently:**
- `tac.review_counter.record_round` → `verdict_landed` (WIRED, fail-open).

**Producers to wire in #389 (candidates, additive + fail-open):**
- gate/verdict rulings from `src/tac/verdicts/` and `src/tac/through_r/` (the sealed-off siblings)
  → `gate_ruled` / `verdict_landed`.
- `tac.cathedral.verdict_ledger.append_consumer_invocation_batch` → `verdict_landed`.
- memo/DAG landings → `memo_landed`.
- the governed launcher / agent dispatch → `agent_spawned` / `agent_completed` / `agent_died`
  (NOTE: launch path is sealed — wire at the dispatcher, not inside launch).

**Consumers of the bulletin (read side):**
- seal-round / synthesis writers → `staleness_check(subjects, since=offset)` before declaring a
  section fresh (the P6-R1 failure this closes).
- the #247 costate controller SENSE layer → `events_since` for a live activity feed.

**Recovery-manifest consumers:**
- session-recovery on respawn → `recover_report()` (CLI `tools/session_recover.py report`).
- dispatchers → `register_inflight` at spawn, `heartbeat` per checkpoint, `complete` at finish.

## Files

- `src/tac/session_bus/__init__.py` (new)
- `src/tac/session_bus/bulletin.py` (new)
- `src/tac/session_bus/recovery_manifest.py` (new)
- `tools/session_recover.py` (new)
- `src/tac/tests/test_session_bus.py` (new, 32 tests)
- `tools/subagent_checkpoint.py` (additive fields + CLI)
- `src/tac/review_counter.py` (fail-open producer wire-in)
- `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (FEED-canon-u3)
