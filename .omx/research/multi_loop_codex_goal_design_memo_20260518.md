# Multi-loop Codex /goal design memo — 5 parallel autonomous loops + coordination surface

## Frontmatter (Catalog #300 v2 + #305 + #303 + #294 + #309 compliance)

```yaml
---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_grand_council_attendees: [Karpathy, Carmack, Hassabis, Tao]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "5 loops is engineering bloat unless each loop is doing work that does NOT compose into ONE loop with 5 sub-modes. The default presumption must be ONE loop + sub-mode switch (per Carmack); split into 5 ONLY when concurrency requirements + file-scope disjointness + cadence asymmetry can't be reconciled inside a single LOOP. Item 4 (hf-hub-publisher) is on-demand + Item 5 (adversarial-review-of-claude-work) is per-trigger; both could be sub-modes of ONE persistent loop dispatched by event-type rather than their own loops. PROCEED only with explicit STRICT gate (Catalog #332) preventing the ONE-loop-collapse failure mode AND with sequential activation order so failure-blast-radius is bounded to one loop at a time."
  - member: Carmack
    verbatim: "Engineering reductionist position: cron + on-demand + per-trigger is THREE cadences not five loops. Loop 1 is continuous; Loop 2 is daily; Loops 3-5 are reactive. If a daily cron + reactive triggers can fire INTO the canonical-task-execution loop's work queue (which is exactly what canonical_task_status.jsonl already supports), then we don't need 5 loops — we need 1 loop + 4 work-injectors. CONCEDE PROCEED on the multi-loop interpretation IF cross-loop coordination surface is genuinely cheaper than 4 work-injectors + ONE polling loop; my prediction is the answer is loop-count=3 (continuous + nightly + reactive-multiplexed), not 5."
council_assumption_adversary_verdict:
  - assumption: "5 loops is the right shape"
    classification: CARGO-CULTED-PENDING-VERIFICATION
    rationale: "Five was operator-proposed at directive-write time but never empirically validated against alternatives (ONE-loop-with-sub-modes per Carmack / THREE-loop continuous-nightly-reactive per Hassabis). The pattern was inherited from the GHA workflow taxonomy (cron + workflow_dispatch + webhook) without checking if it composes optimally for autonomous-Codex-execution rather than CI."
  - assumption: "Loops are non-interfering when given disjoint file scope"
    classification: HARD-EARNED
    rationale: "Catalog #302 + #314 empirical anchors confirm: sister-subagent edits to shared files (preflight.py, CLAUDE.md) collide via commit-swap (Catalog #157 protection) OR working-tree absorption (Catalog #314 protection). Disjoint file scope IS the structural primitive for parallel loops."
  - assumption: "Cron + on-demand + per-trigger covers all canonical cadences"
    classification: HARD-EARNED
    rationale: "GHA workflow taxonomy + Modal scheduler taxonomy + canonical_task_status priority semantics empirically cover continuous / scheduled / event-driven / on-demand. No fifth cadence observed in canonical Pact infrastructure."
  - assumption: "Session-state ledger pattern scales to N loops"
    classification: HARD-EARNED
    rationale: "Catalog #245 4-proc spawn-pool stress test + fcntl LOCK_EX + APPEND-ONLY JSONL pattern empirically supports concurrent writers. N=5 is well within the validated envelope (the stress test runs 4 procs × 5 rows; 5 loops × ~1 write per LOOP iteration ≤ 5 concurrent writes is structurally equivalent)."
  - assumption: "The operator's role is unchanged when 5 loops run in parallel"
    classification: CARGO-CULTED
    rationale: "5× parallel loops produce 5× telemetry. Operator-attention budget is fixed; observability surface MUST aggregate per-loop signal into a single operator-facing summary OR the operator becomes the bottleneck again. Catalog #300 mission-alignment frontmatter is the protocol-level mechanism; tools/operator_briefing.py extension is the runtime mechanism."
council_decisions_recorded:
  - "op-routable #1: Land canonical helper src/tac/codex_loop_coordination.py (~600 LOC; mirror Catalog #245 4-layer pattern; fcntl-locked JSONL at .omx/state/codex_loop_coordination.jsonl)"
  - "op-routable #2: Land CLI tool tools/codex_loop_coordination.py (~400 LOC; 8 subcommands per Layer 2 contract below)"
  - "op-routable #3: Land STRICT preflight gate Catalog #332 (this gate) per CLAUDE.md 'Strict-flip atomicity rule'; warn-only at landing"
  - "op-routable #4: Sequential activation order per ACTIVATION ORDER section below — do NOT activate all 5 simultaneously"
  - "op-routable #5: Codex prompt template per loop (5 canonical /goal copy-paste blocks ready for paste-activation)"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: null
council_related_deliberation_ids:
  - codex_persistent_goal_v2_4_no_hardcoded_state_20260518
  - codex_persistent_goal_v2_5_with_inbox_integration_20260518
  - codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518
  - codex_routing_directive_claude_memory_hermetic_export_channel_20260518
horizon_class: frontier_pursuit
deliberation_id: multi_loop_codex_goal_design_memo_20260518
---
```

## Executive summary

**TL;DR**: Five autonomous Codex CLI loops, each with disjoint file-scope ownership, claim-file coordination surface, sequential activation, and a structural STRICT gate refusing cross-loop file collisions. ONE loop is already live (canonical-task-execution v2.5). The remaining four (nightly-cron, public-PR-monitor, hf-hub-publisher, adversarial-review-of-claude-work) are specified here with copy-paste /goal templates + per-loop session-state ledger + GHA workflow integration where applicable.

Carmack/Contrarian raised the **ONE-loop-with-sub-modes alternative** that this memo MUST explicitly address (see "Multi-loop vs ONE-loop-with-sub-modes" section below). Verdict: **PROCEED_WITH_REVISIONS** — five loops are the right shape WHEN coordination cost < N work-injectors AND each loop's cadence + file-scope is materially distinct. Verification queue includes a 30-day post-activation retrospective per CLAUDE.md "Mission alignment" Consequence 3.

### Loop inventory at a glance

| # | Loop | Status | Cadence | Session-state ledger | File scope owned |
|---|---|---|---|---|---|
| 1 | canonical-task-execution | **LIVE** (v2.5 paste-ready after inbox channel lands) | Continuous | `.omx/state/codex_persistent_session_state.jsonl` | `.omx/state/canonical_task_status.jsonl` + per-task working-tree files |
| 2 | nightly-cron | **PLANNED** | Daily 00:00 UTC | `.omx/state/codex_nightly_cron_session_state.jsonl` | `.omx/state/{nightly_audit,frontier_scan,ledger_archive}_*.json` |
| 3 | public-PR-monitor | **PLANNED** | Hourly during contest | `.omx/state/codex_public_pr_monitor_session_state.jsonl` | `experiments/results/public_pr*_intake_*/` + `.omx/state/public_pr_monitor_*.jsonl` |
| 4 | hf-hub-publisher | **PLANNED** | On-demand (operator-triggered or release-event) | `.omx/state/codex_hf_hub_publisher_session_state.jsonl` | `.omx/oss_export/` + `experiments/results/hf_release_*/` |
| 5 | adversarial-review-of-claude-work | **PLANNED** | Per-design-memo-landing trigger | `.omx/state/codex_review_of_claude_session_state.jsonl` | `.omx/research/codex_findings_*_codex.md` only |

### Mission alignment per CLAUDE.md (binding)

- **frontier_breaking**: 5× parallel velocity vs ONE-loop sequential is the **dominant** mission contribution. The operator's bottleneck pre-multi-loop is sequential reasoning across 4 distinct work-classes; multi-loop converts operator-attention into per-loop dispatch + cross-loop coordination review.
- **frontier_protecting**: nightly-cron loop catches CI/cron drift in <24h instead of multi-day discovery windows.
- **rigor_overhead**: cross-loop coordination surface adds ~5% overhead per loop iteration (claim + release file scope); pays back via collision-prevention.
- **apparatus_maintenance**: Catalog #332 STRICT gate + tools/codex_loop_coordination.py are infrastructure; no direct score contribution.
- **mission_questioned**: Contrarian + Carmack triggered the "is this serving the mission?" question; answered via the ONE-loop-with-sub-modes section + the 30-day retrospective queue.

---

## 1. Domain primer

### 1.1 The persistent /goal pattern (Codex CLI canonical)

Codex CLI (`codex exec` per CLAUDE.md "Codex CLI invocation" non-negotiable) supports a **persistent /goal** that the agent re-reads on every invocation. The canonical /goal template defines:

1. **ROLE** + **DOMAIN** (immutable contest rules)
2. **POINTERS** (file paths to read EVERY invocation; never hardcode their state — per CANONICAL-POINTER META-RULE)
3. **LOOP** (8-step canonical work cycle; see /goal v2.5 source for the live version)
4. **CAPS** (budget + discipline gates)
5. **FAILURE** (halt conditions + escalation)
6. **OBSERVABILITY** (operator-facing audit surfaces)

The current /goal v2.4 has ONE LOOP. v2.5 (paste-pending after Codex lands the inbox channel) extends LOOP step 1 + step 5 + step 7 to consult `.omx/state/codex_to_claude_inbox.jsonl`. This memo proposes that AFTER v2.5 stabilizes, FIVE distinct LOOPS run in parallel, each with their own /goal prompt that scopes work to their assigned domain.

### 1.2 Codex CLI autonomous execution model

Codex CLI runs the operator-pasted /goal until one of:
- Token budget exhausted (typically ~200K tokens per /goal session)
- Rate-limit hit (Codex 5h windows)
- Operator interrupt
- FAILURE branch in /goal LOOP

The agent state between invocations is **the working tree + the canonical state ledgers** — there is NO persistent in-memory state. This is the structural primitive that makes multi-loop feasible: a loop's identity is **its /goal prompt + its session-state ledger + its file-scope claim**, all of which are file-backed.

### 1.3 Session-state ledger pattern per Catalog #245 sister

Each of the 5 loops has its own session-state ledger at `.omx/state/codex_<loop_name>_session_state.jsonl`. Schema mirrors Catalog #245:

```python
SESSION_STATE_SCHEMA_VERSION = "codex_<loop_name>_session_state_v1_20260518"
SESSION_STATE_PATH = REPO_ROOT / ".omx" / "state" / f"codex_{loop_name}_session_state.jsonl"

# Row contract (APPEND-ONLY per Catalog #110/#113 HISTORICAL_PROVENANCE):
{
    "schema_version": SESSION_STATE_SCHEMA_VERSION,
    "timestamp_utc": "<utc>",
    "loop_id": "<canonical-loop-name>",
    "iteration_id": "<utc>_<short_uuid>",  # unique per LOOP iteration
    "directive_executed": "<short slug>",
    "items_landed": ["<file_paths_or_outcomes>"],
    "items_remaining": ["<carry-over work>"],
    "next_action": "<one-line operator-readable description>",
    "commit_shas": ["<sha>"],
    "open_blockers": ["<blocker_class:detail>"],
    "files_claimed_in_coordination": ["<paths>"],  # NEW: cross-loop claim references
    "agent": "codex",
    "subagent_id": "<codex CLI session id>",
    "session_id": "<codex session id>",
    "written_at_utc": "<utc>",
    "written_pid": <int>,
    "written_host": "<host>",
    "notes": "<free text>"
}
```

The `files_claimed_in_coordination` field is NEW — it cites coordination-surface event_ids so cross-loop audit can trace which loop owned which files at which timestamp.

---

## 2. Per-loop canonical specification

### Loop 1: canonical-task-execution (LIVE)

- **Scope**: Execute pending rows in `.omx/state/canonical_task_status.jsonl` owned by `owner=codex`
- **Cadence**: Continuous (LOOP iteration as fast as Codex CLI session allows; bounded by tool-use rate)
- **Session-state ledger**: `.omx/state/codex_persistent_session_state.jsonl` (LIVE; 7 rows at directive-write time)
- **LOOP step contract**: per /goal v2.5 (see `.omx/research/codex_persistent_goal_v2_5_with_inbox_integration_20260518.md`); ALREADY documented
- **File-scope ownership**: dynamic per-task; declared via `tools/codex_loop_coordination.py claim` BEFORE editing
- **Coordination requirements**: highest priority (this loop is the work-execution surface; other loops MUST yield on file conflict)
- **Failure modes + halt conditions**: rate_limited / catalog_<N>_refused / test_red / sister_collision_#314 / inbox_question_unanswered
- **Estimated wall-clock per loop iteration**: 5-30 min per canonical_task_status row (varies wildly)
- **Cost envelope**: continuous; no incremental cost beyond Codex Pro subscription

### Loop 2: nightly-cron

- **Scope**: 
  - Per CLAUDE.md "Mission alignment" Consequence 2: annual gate audit cadence check
  - Per CLAUDE.md "State JSONL archival policy" non-negotiable: ledger archival via `tools/archive_jsonl_state.py --apply` on >10 MB JSONLs
  - Per Catalog #316: frontier scan via `tools/scan_best_anchor_per_axis.py --check-drift`
  - Cost-band posterior refresh via `tac.cost_band_calibration.append_anchor` for any in-flight dispatches without harvested rows
  - Catalog #298 stale L1 substrate audit via `tools/audit_stale_l1_substrates.py`
  - Catalog #91 review-tracker cleanup
  - Catalog #284 nightly_catalog_gate_regression auto-trigger (already exists; this loop CONSUMES its outputs)
- **Cadence**: Daily 00:00 UTC (operator timezone sleeps through this window per nightly_catalog_gate_regression precedent; selected 00:00 UTC to NOT collide with the 02:00 UTC nightly_catalog_gate_regression workflow)
- **Session-state ledger**: `.omx/state/codex_nightly_cron_session_state.jsonl`
- **LOOP step contract** (NEW for this loop):
  1. **PRE-FLIGHT**: read pointers + resume latest `codex_nightly_cron_session_state` row + claim file scope via coordination surface
  2. **AUDIT**: query each canonical state file size; identify candidates for archival
  3. **ARCHIVE**: `tools/archive_jsonl_state.py --apply` for any over-budget JSONLs
  4. **FRONTIER SCAN**: `tools/scan_best_anchor_per_axis.py --check-drift`; if drift detected, emit relay event via Codex→Claude inbox (per Catalog #331 sister channel)
  5. **COST-BAND REFRESH**: query Modal call_id ledger for in-flight rows >24h old without harvested outcome; emit failed_unknown_billing rows per Catalog #147 sister discipline
  6. **STALE SUBSTRATE AUDIT**: `tools/audit_stale_l1_substrates.py`; relay any newly-stale candidates
  7. **PERSIST**: append session-state row with all outputs; commit via canonical serializer
  8. **RELEASE**: release file-scope claim via coordination surface
- **File-scope ownership**: 
  - WRITE: `.omx/state/archive/`, `.omx/state/{stale_l1_audit,frontier_drift,nightly_cron_summary}_<YYYYMMDD>.json`, `.omx/state/codex_nightly_cron_session_state.jsonl`
  - READ: every `.omx/state/*.jsonl` (read-only; no mutation)
- **Coordination requirements**: low priority (nightly cadence; yields to any other loop on conflict)
- **Failure modes**: rate_limited / archive_disk_full / state_corruption (any of which routes through inbox channel for Claude review)
- **Estimated wall-clock**: 15-45 min per iteration
- **Cost envelope**: $0 (read-only + local-only operations)
- **GHA workflow trigger**: `.github/workflows/codex_nightly_cron.yml` (cron `0 0 * * *`) — operator can also `workflow_dispatch` manually

### Loop 3: public-PR-monitor

- **Scope**: Per CLAUDE.md "Public frontier watch and intake" non-negotiable. Monitor `commaai/contest` GitHub repo for new PRs; intake any PR with score claim < current frontier per Catalog #316; trigger replay if archive bytes + runtime are extractable
- **Cadence**: Hourly during active contest window; once-daily otherwise
- **Session-state ledger**: `.omx/state/codex_public_pr_monitor_session_state.jsonl`
- **LOOP step contract** (NEW):
  1. **PRE-FLIGHT**: query GitHub API for PRs created/updated since last iteration's `last_polled_at_utc`
  2. **TRIAGE**: for each new PR, parse claimed score from PR body / comments; compare against `tac.frontier_scan.best_per_axis`
  3. **INTAKE**: if claim < frontier on either axis, run `tools/intake_public_pr.py` (creates `experiments/results/public_pr<N>_intake_<utc>/` per CLAUDE.md "Bit-level deconstruction" discipline; clone is PRISTINE per Catalog #109)
  4. **REPLAY DECISION**: if intake succeeds AND `replay_admissible=True` (compliance gate per Catalog #99 representation-promotion-card check), enqueue `canonical_task_status` row `owner=codex` for the canonical-task-execution loop to pick up
  5. **PERSIST**: append session-state row
- **File-scope ownership**:
  - WRITE: `experiments/results/public_pr*_intake_<utc>/` (per-PR isolated directory), `.omx/state/codex_public_pr_monitor_session_state.jsonl`, `.omx/state/canonical_task_status.jsonl` (APPEND ONLY via canonical helper)
  - READ: `reports/latest.md`, `.omx/state/continual_learning_posterior.jsonl`, `.omx/state/modal_call_id_ledger.jsonl`
- **Coordination requirements**: medium priority (yields to canonical-task-execution; takes priority over nightly-cron)
- **Failure modes**: github_api_rate_limited (back off + retry next iteration) / intake_failed_compliance_violation (relay via inbox; do NOT enqueue task) / pr_clone_dirty_violation_catalog_109 (refuse intake; record forensic evidence)
- **Estimated wall-clock**: 2-15 min per iteration (depends on PR count + intake required)
- **Cost envelope**: $0 (GitHub API read; clone-only; no replay until canonical-task-execution picks up the enqueued task)
- **GHA workflow trigger**: `.github/workflows/codex_public_pr_monitor.yml` (cron `0 * * * *` hourly during contest; `0 0 * * *` daily otherwise; toggle via `.omx/state/contest_window_active.flag` sentinel file)

### Loop 4: hf-hub-publisher

- **Scope**: ITEM_12 in canonical_task_status (`hf-hub-push HF dataset push`); OSS release per CLAUDE.md "Public Disclosure Hygiene" with sanitized supplement builds; HF model card emission via `tools/build_release_manifest.py` sister
- **Cadence**: On-demand (operator-triggered via `.omx/state/hf_hub_publisher_trigger.flag` OR release-event-driven from canonical-task-execution loop completion of `release/*` lane)
- **Session-state ledger**: `.omx/state/codex_hf_hub_publisher_session_state.jsonl`
- **LOOP step contract** (NEW):
  1. **PRE-FLIGHT**: check for `hf_hub_publisher_trigger.flag` OR enumerate canonical_task_status rows with `kind=hf_release` + `status=pending`
  2. **SANITIZATION**: per CLAUDE.md "Public Disclosure Hygiene" + Catalog #208 (docs no local absolute paths), strip private state from release packet (delegate to `tools/build_oss_release_packet.py` sister)
  3. **HF DATASET PUSH**: via `hf` CLI (per huggingface-skills:hf-cli context); push to operator's namespace per ITEM_12 spec
  4. **MODEL CARD**: emit canonical model-card via `tools/build_release_manifest.py` if release includes weights
  5. **VERIFICATION**: pull-back-and-compare against published artifact (apples-to-apples per CLAUDE.md "Apples-to-apples evidence discipline")
  6. **PERSIST**: append session-state row + canonical_task_status `completed` row + relay event via inbox to Claude for public announcement coordination
  7. **RELEASE**: clear trigger flag; release file-scope claim
- **File-scope ownership**:
  - WRITE: `.omx/oss_export/`, `experiments/results/hf_release_<utc>/`, `.omx/state/codex_hf_hub_publisher_session_state.jsonl`
  - READ: all release-eligible artifacts (filtered through sanitization)
- **Coordination requirements**: high priority on flag-trigger (operator-time-critical); yields to canonical-task-execution on shared-file conflict (rare; typically disjoint scope)
- **Failure modes**: hf_api_auth_failed / sanitization_violation_catalog_208 / model_card_schema_violation / verification_mismatch (refuse to confirm release)
- **Estimated wall-clock**: 30-90 min per release (dominated by HF upload)
- **Cost envelope**: $0 (HF subscription approved per operator standing context)
- **GHA workflow trigger**: NONE (on-demand via flag file; could add `workflow_dispatch` GHA workflow if operator prefers GUI trigger)

### Loop 5: adversarial-review-of-claude-work

- **Scope**: INVERSE direction from existing `codex:adversarial-review` skill. Where today's skill is operator-invoked per `/codex:adversarial-review`, this loop AUTO-fires per design-memo-landing trigger. Sister of Catalog #229 (premise-verification) discipline at the per-design-memo surface
- **Cadence**: Per-trigger (each new `.omx/research/*_design_memo_*.md` landing OR `feedback_*_landed_*.md` landing emits a trigger event)
- **Session-state ledger**: `.omx/state/codex_review_of_claude_session_state.jsonl`
- **LOOP step contract** (NEW):
  1. **PRE-FLIGHT**: poll `.omx/state/claude_design_memo_review_queue.jsonl` (NEW sister surface; populated by Claude commits via git-post-commit hook OR by Claude appending explicitly via `tools/codex_loop_coordination.py request-review`)
  2. **TRIAGE**: for each queued memo, classify by `council_tier` frontmatter (T1/T2/T3/T4 per Catalog #300); skip T1 (working-group recommendations; not deliberation-binding)
  3. **REVIEW**: invoke codex companion via `codex exec --skip-git-repo-check --sandbox read-only -m gpt-5.5 -c model_reasoning_effort=xhigh` per CLAUDE.md "Codex CLI invocation" Pattern A; pass the memo + relevant context (CLAUDE.md + AGENTS.md + sister design memos + relevant canonical state)
  4. **VERDICT EMISSION**: codex companion emits `codex_findings_<memo_slug>_<utc>_codex.md` with verdict in `{approve, advisory, needs-attention, no-ship}` per Catalog #271 verdict taxonomy
  5. **INBOX RELAY**: if verdict ∈ `{needs-attention, no-ship}`, emit `relay` event via Codex→Claude inbox channel for Claude review BEFORE the memo's recommendations are acted on
  6. **PERSIST**: append session-state row + queue-row supersession (status='reviewed')
- **File-scope ownership**:
  - WRITE: `.omx/research/codex_findings_*_codex.md` ONLY, `.omx/state/codex_review_of_claude_session_state.jsonl`, `.omx/state/claude_design_memo_review_queue.jsonl` (mutation: status→reviewed)
  - READ: every `.omx/research/*.md`, CLAUDE.md, AGENTS.md
- **Coordination requirements**: low-medium priority (review is post-hoc; no urgency unless `no-ship` verdict)
- **Failure modes**: codex_companion_timeout (Catalog #281 rc=2 surfaced as `invocation-error`; queue stays pending) / cache_stale_due_to_dirty_tree (Catalog #282 protection auto-handles via `--no-cache-for-paid-dispatch`) / review_finds_blocking_violation (relay via inbox; do NOT auto-amend memo)
- **Estimated wall-clock**: 10-45 min per design memo
- **Cost envelope**: $0 (codex CLI included in Codex Pro subscription; xhigh reasoning effort = ~$0.10-0.30 per review in tokens but covered)
- **GHA workflow trigger**: optional `.github/workflows/codex_review_of_claude_design_memo.yml` (triggered by `paths: [.omx/research/*_design_memo_*.md]` on `push` events to main); fallback is local-Codex-CLI polling

---

## 3. Cross-loop coordination surface

### 3.1 Canonical helper at `src/tac/codex_loop_coordination.py`

Mirror Catalog #245 4-layer pattern. ~600 LOC.

```python
# src/tac/codex_loop_coordination.py
# SPDX-License-Identifier: MIT

COORDINATION_SCHEMA_VERSION = "codex_loop_coordination_v1_20260518"
COORDINATION_PATH = REPO_ROOT / ".omx" / "state" / "codex_loop_coordination.jsonl"
COORDINATION_LOCK = COORDINATION_PATH.with_suffix(".jsonl.lock")

VALID_LOOP_IDS = frozenset({
    "canonical-task-execution",
    "nightly-cron",
    "public-pr-monitor",
    "hf-hub-publisher",
    "adversarial-review-of-claude-work",
})

LOOP_PRIORITY = {
    "canonical-task-execution": 100,  # highest; others yield
    "hf-hub-publisher": 80,           # operator-time-critical when flag-triggered
    "public-pr-monitor": 60,          # medium; yields to higher
    "adversarial-review-of-claude-work": 40,
    "nightly-cron": 20,               # lowest; yields to all
}

VALID_EVENT_TYPES = frozenset({
    "claim",            # loop claims file scope for upcoming work
    "release",          # loop releases claim (success or yield)
    "conflict_refused", # claim REFUSED due to higher-priority active claim
    "expired",          # claim past expected_completion_utc; auto-released
    "operator_force_release",  # operator manually released a stuck claim
})

VALID_STATUSES = frozenset({
    "active",           # claim is current
    "released",         # claim explicitly released
    "expired",          # past expected_completion_utc
    "preempted",        # higher-priority loop took the scope
})

class CoordinationRowValidationError(ValueError): pass
class CoordinationRowCorruptError(RuntimeError): pass
class LoopCollisionError(RuntimeError):
    """Raised when claim() detects a higher-priority active claim on the requested file."""

def claim_file_scope(
    *,
    loop_id: str,
    files_to_claim: tuple[str, ...],
    expected_completion_utc: str,
    session_id: str,
    purpose: str,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Claim file scope for upcoming loop iteration.

    Validates:
    - loop_id in VALID_LOOP_IDS
    - files_to_claim non-empty + each path is repo-relative
    - expected_completion_utc parseable + in future
    - no higher-priority loop has active claim on ANY of these files

    On collision with same-priority loop: REFUSES claim (raises LoopCollisionError).
    On collision with lower-priority loop: PREEMPTS the lower loop's claim (appends
    'preempted' event for the lower loop's claim; appends 'claim' event for the
    higher loop).

    Returns the canonical claim event row.
    """

def release_file_scope(
    *,
    event_id: str,           # the claim event_id being released
    actual_completion_utc: str,
    outcome: str,            # one of {success, yielded, error}
    session_id: str,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Release a prior claim. Idempotent for already-released claims."""

def query_active_claims(
    *, loop_id: str | None = None, path: Path | None = None
) -> list[dict[str, Any]]:
    """Returns active claims, optionally filtered by loop_id."""

def query_claims_by_file(
    file_path: str, *, path: Path | None = None
) -> list[dict[str, Any]]:
    """All claim history for a specific file (chronological)."""

def detect_collisions(
    *, path: Path | None = None
) -> list[dict[str, Any]]:
    """Returns any active claims with file-scope overlap.

    Used by Catalog #332 STRICT preflight gate.
    """

def expire_stale_claims(
    *, now_utc: str, path: Path | None = None, lock_path: Path | None = None
) -> list[dict[str, Any]]:
    """Auto-expire claims past expected_completion_utc. Returns expired event rows."""

def operator_force_release(
    *, event_id: str, reason: str, operator_handle: str, session_id: str,
    path: Path | None = None, lock_path: Path | None = None
) -> dict[str, Any]:
    """Operator force-releases a stuck claim (e.g., loop crashed without releasing)."""

def latest_status_by_event_id(*, path: Path | None = None) -> dict[str, str]:
    """{event_id: status} — latest-row-wins per Catalog #245 sister pattern."""

def load_coordination_strict(path: Path | None = None) -> list[dict[str, Any]]:
    """Catalog #138 fail-closed loader. Raises CoordinationRowCorruptError on parse failure.
    Quarantines corrupt file to .corrupt.<utc> per Catalog #245 sister pattern.
    """
```

**Implementation invariants** (mirror Catalog #245):
- APPEND-ONLY per Catalog #110 — status transitions are NEW rows referencing the same `event_id`
- fcntl-locked atomic writes per Catalog #131 — `_append_event_locked` does `.tmp.<uuid12>` + `os.replace`
- Strict-load per Catalog #138 — quarantines on parse failure
- JSON byte-stable per Catalog #245 — `sort_keys=True` + `ensure_ascii=False`
- 4-proc spawn-pool stress test in tests
- `event_id` schema: `claim_<short_uuid>` for claim events; status events reference the parent claim's event_id

### 3.2 Conflict resolution algorithm

```
claim(loop_id, files, deadline):
    with fcntl LOCK_EX on COORDINATION_LOCK:
        active = query_active_claims()
        conflicts = [c for c in active if c.files ∩ files != ∅]
        
        if not conflicts:
            return append_claim_event(loop_id, files, deadline)
        
        my_priority = LOOP_PRIORITY[loop_id]
        for c in conflicts:
            c_priority = LOOP_PRIORITY[c.loop_id]
            if c_priority > my_priority:
                # higher-priority loop holds the scope; I yield
                raise LoopCollisionError(f"loop {loop_id} yields to {c.loop_id}")
            elif c_priority == my_priority:
                # same-priority collision: REFUSE per "reserve higher priority, refuse other"
                # (this should only happen when two loops share priority, which is
                # avoided by the LOOP_PRIORITY constant unique-priority discipline)
                raise LoopCollisionError(f"same-priority collision; loop {loop_id} refuses")
            else:
                # I'm higher priority; preempt the lower-priority claim
                append_preempt_event(c.event_id, preempted_by=loop_id)
        
        return append_claim_event(loop_id, files, deadline)
```

### 3.3 CLI tool at `tools/codex_loop_coordination.py`

```bash
# A loop claims file scope before doing work:
.venv/bin/python tools/codex_loop_coordination.py claim \
    --loop-id "nightly-cron" \
    --files ".omx/state/codex_nightly_cron_session_state.jsonl,.omx/state/archive/" \
    --expected-completion-utc "$(date -u -v+1H +%Y-%m-%dT%H:%M:%SZ)" \
    --session-id "$CODEX_SESSION_ID" \
    --purpose "nightly archival + frontier scan + cost-band refresh"

# Loop releases claim after work completes:
.venv/bin/python tools/codex_loop_coordination.py release \
    --event-id <event_id> \
    --actual-completion-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --outcome success \
    --session-id "$CODEX_SESSION_ID"

# Operator audits active claims:
.venv/bin/python tools/codex_loop_coordination.py summary --format=text
.venv/bin/python tools/codex_loop_coordination.py summary --format=json

# Operator force-releases stuck claim (e.g., loop crashed):
.venv/bin/python tools/codex_loop_coordination.py operator-force-release \
    --event-id <event_id> --reason "loop crashed without releasing; manual cleanup" \
    --operator-handle "$USER"

# Cron-driven stale-claim expiration (nightly-cron loop runs this):
.venv/bin/python tools/codex_loop_coordination.py expire-stale-claims

# Show all claims involving a specific file (forensic):
.venv/bin/python tools/codex_loop_coordination.py file-history \
    --file "src/tac/preflight.py"

# Show all collisions in last 24h (forensic):
.venv/bin/python tools/codex_loop_coordination.py collision-report --hours 24
```

CLI exit codes: 0 clean / 1 strict-mode failure / 2 CLI error / 3 collision-refused (LoopCollisionError).

### 3.4 Operator-facing audit integration

- `tools/operator_briefing.py` extends with `loop_coordination_active_claims_count` + `loop_coordination_oldest_claim_age_minutes` + `loop_coordination_recent_collisions` fields
- `tools/all_lanes_preflight.py` runs `tools/codex_loop_coordination.py collision-report` as a non-blocking diagnostic before lane dry-runs
- `reports/latest.md` MAY include a "Loop coordination status" block per release branch

---

## 4. Catalog #332 STRICT preflight gate

### Gate name
`check_codex_loop_coordination_no_file_collision`

### Refuses

Repo state where `tac.codex_loop_coordination.detect_collisions()` returns >0 active-claim rows whose `files_to_claim` overlap AND whose `loop_id` are distinct (cross-loop collision). Same-loop multiple claims on disjoint timestamps are NORMAL (a loop iterates and re-claims its own scope).

### Acceptance

1. `detect_collisions()` returns `[]` → CLEAN
2. Collision exists but coordination ledger shows `expired` or `preempted` event for the older claim → CLEAN (the algorithm already resolved it)
3. Same-line waiver `# LOOP_COORDINATION_COLLISION_OK:<rationale>` in `.omx/state/codex_loop_coordination.jsonl` row (placeholder `<rationale>` / `<reason>` rejected)

### Strict-flip atomicity

WARN-ONLY at landing per CLAUDE.md "Strict-flip atomicity rule" — live count at landing: 0 (the coordination surface is brand-new with no claims yet). STRICT-FLIP after first one-week window where 5+ active claims have been registered across at least 2 distinct loops AND no false-positive collision flags have fired.

### Sister gates

- Catalog #131 (`check_no_bare_writes_to_shared_state`) — `COORDINATION_PATH` registered in `_SHARED_STATE_PATH_MARKERS`; `_coordination_lock` in `_BARE_WRITE_LOCK_TOKENS`; `claim_file_scope` / `_append_event_locked` in `_BARE_WRITE_CANONICAL_HELPER_CALL_TOKENS`; helper file in `_BARE_WRITE_CANONICAL_HELPERS` exclusion list
- Catalog #138 (`check_state_writers_strict_load_for_mutating_path`) — `load_coordination_strict` mirrors the canonical pattern
- Catalog #245 (Modal call_id ledger) — canonical 4-layer template this gate operationalizes
- Catalog #302 (`check_sister_subagent_scope_overlap_via_checkpoint_jsonl`) — same META-class at the per-subagent surface; #332 is the per-LOOP surface
- Catalog #314 (`check_no_subagent_files_touched_absorption_in_bare_commits`) — extincts the BARE-COMMIT absorption surface; #332 extincts the cross-LOOP claim collision surface
- Catalog #331 (Codex→Claude inbox bidirectional channel) — sister 4-layer Codex/Claude channel

Together they extinct the **multi-loop concurrent execution collision class** at SIX surfaces:
1. Edit-time-checkpoint (#302)
2. Edit-time-bulk-op (#230)
3. Commit-time-pre-pre-lock (#157)
4. Commit-time-staged (#216)
5. Commit-time-lock-arbitration (#117 + #174)
6. **Loop-time-claim-overlap (#332 — this gate)**

---

## 5. Multi-loop vs ONE-loop-with-sub-modes — addressing Contrarian + Carmack

This section MUST be honored per the Council Conduct non-negotiable: Contrarian + Carmack flagged the ONE-loop-with-sub-modes alternative as the engineering-reductionist null hypothesis. PROCEED requires explicit refutation OR concession.

### 5.1 The ONE-loop-with-sub-modes alternative

A SINGLE persistent Codex CLI /goal could route ALL work through canonical-task-execution by injecting:
- Daily cron events as `kind=nightly_cron` rows in `.omx/state/canonical_task_status.jsonl`
- Public PR detection as `kind=public_pr_intake` rows
- HF release triggers as `kind=hf_release` rows
- Design-memo-landing triggers as `kind=adversarial_review` rows

The canonical-task-execution loop's LOOP step 3 SELECT would then dispatch based on `kind` instead of unifying everything as "execute canonical work item".

### 5.2 When ONE-loop wins

ONE-loop is structurally simpler when ALL of:
- The cadences DO NOT need to fire concurrently (a daily cron + an hourly check + an event-driven trigger can all serialize through ONE loop if each iteration is fast)
- The file-scopes DO NOT need disjoint isolation (if all 5 work classes write to the same surfaces, coordination is wasted overhead)
- The codex CLI session budget is large enough to handle the worst-case work item in ONE session

### 5.3 When multi-loop wins (this memo's PROCEED argument)

Multi-loop is structurally necessary when:
1. **Codex session budget is per-loop** — each loop runs in its own `codex exec` session with its own ~200K token budget. Five loops = 5× total budget. ONE loop sharing budget means a long-running canonical-task-execution work item starves the cron + monitor + review work.
2. **Cadences ARE concurrent** — nightly-cron at 00:00 UTC + public-PR-monitor at any hour + canonical-task-execution continuous = three loops MUST fire concurrently. Serializing them through ONE loop means cron work waits up to 24h for canonical-task-execution to pick it up; PR monitor work waits up to 1h.
3. **File-scopes ARE disjoint** — public-PR-monitor writes to `experiments/results/public_pr*_intake_*/`; nightly-cron writes to `.omx/state/{archive,nightly_summary}_*`; hf-hub-publisher writes to `.omx/oss_export/`. These NEVER collide except in pathological cases.
4. **Failure isolation matters** — if canonical-task-execution loop hits a rate-limit, the nightly-cron + public-PR-monitor + hf-hub-publisher should still fire. Bundling them sacrifices failure isolation.
5. **Operator-attention budget is preserved** — ONE loop with 5 sub-modes means operator must read 5× more context per loop-iteration to understand "what is this loop doing right now?". Five disjoint loops have 5× shallower context per loop but ONE clear concern per loop.

### 5.4 Carmack concession

Carmack's prediction was "loop-count = 3 (continuous + nightly + reactive-multiplexed), not 5". This memo PROCEEDS with 5 BUT acknowledges the 3-loop variant as the deferred-pending-evidence alternative. The 30-day post-activation retrospective per CLAUDE.md "Mission alignment" Consequence 3 explicitly evaluates: did loops 3-5 (public-PR-monitor + hf-hub-publisher + adversarial-review-of-claude-work) each individually justify their existence, OR could they have been multiplexed into a SINGLE reactive loop dispatched by event-type?

Reactivation criteria for the 3-loop variant:
- If retrospective shows >50% of loop-3-iterations were no-ops (no new PRs to triage)
- AND >50% of loop-5-iterations could be triggered from within loop-3 (e.g., codex review fires on PR-intake-completion event)
- AND loop-4 average wall-clock per release is <10 min (small enough to fold into a reactive multiplexer)
→ THEN collapse to 3 loops (continuous + nightly + reactive-multiplexed). Council reconvenes per Catalog #325 per-substrate symposium discipline.

### 5.5 Contrarian concession

Contrarian's binding stipulation: PROCEED only with (a) explicit STRICT gate #332 preventing the ONE-loop-collapse failure mode AND (b) sequential activation order so failure-blast-radius is bounded.

Both honored:
- (a) Catalog #332 IS the structural gate
- (b) Activation order in §6 below explicitly sequences activation; canonical-task-execution stays primary; each subsequent loop activates only after its predecessor has demonstrated stable operation for 24h+

---

## 6. Activation order (sequential, NOT parallel)

This activation discipline is the structural mechanism for bounded failure-blast-radius per Contrarian binding stipulation.

### Phase 1 — PREREQUISITE (DO NOT START LOOPS UNTIL COMPLETE)

1. Codex lands the inbox channel per directive `.omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` (Catalog #331)
2. Codex emits `feedback_codex_to_claude_inbox_bidirectional_channel_landed_20260518.md` confirming all 4 layers + tests green
3. Codex lands the memory hermetic export channel per directive `.omx/research/codex_routing_directive_claude_memory_hermetic_export_channel_20260518.md` (Catalog #332→#333 since #332 reserved for this memo's STRICT gate)

**IMPORTANT**: This memo claimed **Catalog #332** for the multi-loop coordination gate. The sister directive `claude_memory_hermetic_export_channel_20260518.md` proposed Catalog #332 but that # is now claimed; the memory hermetic export channel directive must be amended to claim the NEXT available # (333) when Codex lands it.

### Phase 2 — STABILIZE canonical-task-execution v2.5

4. After both channels land, paste the v2.5 /goal (per `.omx/research/codex_persistent_goal_v2_5_with_inbox_integration_20260518.md`) into Codex CLI
5. Run for 24h+ without operator intervention; verify session-state ledger grows + inbox channel exercised at least once + memory exports consumed at least once
6. If unstable: REVERT to v2.4 + investigate + do NOT proceed to multi-loop until stable

### Phase 3 — ACTIVATE nightly-cron (lowest-stakes infrastructure)

7. Land canonical helper + CLI + STRICT gate + tests for `src/tac/codex_loop_coordination.py` per §3 (this requires Codex to execute the directive emitted by this memo)
8. Land `.github/workflows/codex_nightly_cron.yml` per §7 below
9. Land per-loop /goal prompt per §8 (canonical-task-execution remains primary; nightly-cron is the SECOND loop)
10. Run nightly-cron for 7 days; verify it fires successfully + emits expected outputs + does NOT collide with canonical-task-execution
11. If collisions OR repeated failures: HALT + investigate; do NOT proceed

### Phase 4 — ACTIVATE public-PR-monitor (contest-window-gated)

12. Land `.github/workflows/codex_public_pr_monitor.yml`
13. Activate ONLY when contest is active (per `.omx/state/contest_window_active.flag`); during off-contest periods, the loop runs daily instead of hourly
14. Run for 14 days during contest; verify it picks up new PRs + intake passes Catalog #109 pristine-check + canonical-task-execution downstream replay completes

### Phase 5 — ACTIVATE hf-hub-publisher (on-demand)

15. Land trigger-flag mechanism + canonical helper
16. Operator triggers first release manually via `touch .omx/state/hf_hub_publisher_trigger.flag`
17. Verify sanitization + HF push + verification pass; relay via inbox for public announcement
18. After first successful release, loop is operational

### Phase 6 — ACTIVATE adversarial-review-of-claude-work (highest-coordination)

19. Land queue surface `.omx/state/claude_design_memo_review_queue.jsonl`
20. Land git-post-commit hook that auto-enqueues new `.omx/research/*_design_memo_*.md` files
21. Activate loop with low cadence (every 6h) initially; escalate to per-trigger after demonstrating it does NOT collide with canonical-task-execution
22. If review verdict produces >1 `no-ship` per week, escalate to operator-review-required + pause canonical-task-execution

---

## 7. GHA workflow integration

Two new workflows + adapt existing patterns.

### 7.1 `.github/workflows/codex_nightly_cron.yml`

Mirrors `.github/workflows/nightly_catalog_gate_regression.yml` structure (canonical reference). Key differences:

- Cron `0 0 * * *` (vs `0 2 * * *` for the existing catalog regression — they run sequentially, not concurrently)
- Invokes `codex exec` with the nightly-cron /goal per §8.2 in **detached mode** per CLAUDE.md "Codex CLI invocation" Pattern A
- Writes status JSON to `.omx/state/codex_nightly_cron_<YYYYMMDD>.json` per Catalog #110/#113 HISTORICAL_PROVENANCE
- Commits status JSON via canonical serializer with POST-EDIT sha

### 7.2 `.github/workflows/codex_public_pr_monitor.yml`

- Cron `0 * * * *` (hourly during contest) + `workflow_dispatch` manual trigger
- Triggers on `repository_dispatch` events with type `comma_contest_pr_event` (webhook-driven if operator wires it)
- Reads `.omx/state/contest_window_active.flag` to determine cadence
- Invokes `codex exec` with the public-PR-monitor /goal per §8.3

### 7.3 No GHA workflow for loops 4 + 5 (operator-driven)

- hf-hub-publisher: trigger-flag-driven; no cron needed
- adversarial-review-of-claude-work: git-post-commit-hook-driven; local Codex CLI poll-loop alternative

---

## 8. /goal prompt templates per loop (canonical copy-paste blocks)

### 8.1 Loop 1: canonical-task-execution

See `.omx/research/codex_persistent_goal_v2_5_with_inbox_integration_20260518.md` — already authored. Paste-activate after inbox + memory channels land.

### 8.2 Loop 2: nightly-cron /goal

```
ROLE: Codex nightly-cron loop for pact. Fires 00:00 UTC daily; performs read-only audits + ledger archival + frontier scan + stale-substrate detection.

DOMAIN: same as canonical-task-execution loop. Same SCORER + HARDWARE rules.

PRINCIPLES: read-mostly + APPEND-ONLY (#110) + fcntl-locked writes only (#131) + auditability + zero paid-GPU spend.

POINTERS (read EVERY invocation):
- CLAUDE.md + AGENTS.md
- .omx/state/codex_nightly_cron_session_state.jsonl (resume)
- .omx/state/{canonical_task_status,modal_call_id_ledger,active_lane_dispatch_claims,continual_learning_posterior}.jsonl
- reports/latest.md (frontier; targets)
- glob .omx/state/codex_nightly_cron_*.json (prior nightly artifacts)

LOOP (no operator intervention):
1. PRE-FLIGHT: read pointers + resume latest codex_nightly_cron_session_state row; claim file scope via tools/codex_loop_coordination.py claim --loop-id nightly-cron --files ".omx/state/archive/,.omx/state/codex_nightly_cron_*.json,.omx/state/codex_nightly_cron_session_state.jsonl" --expected-completion-utc <utc+1h> --purpose "nightly audit"
2. AUDIT: run tools/archive_jsonl_state.py (dry-run); identify any JSONL >10MB
3. ARCHIVE: tools/archive_jsonl_state.py --apply per CLAUDE.md "State JSONL archival policy"
4. FRONTIER SCAN: tools/scan_best_anchor_per_axis.py --check-drift per Catalog #316; if drift, relay via tools/codex_to_claude_inbox.py relay
5. COST-BAND REFRESH: query Modal call_id ledger for active rows >24h old; emit failed_unknown_billing per Catalog #147 sister
6. STALE SUBSTRATE AUDIT: tools/audit_stale_l1_substrates.py; relay any newly-stale per Catalog #298
7. PERSIST: write .omx/state/codex_nightly_cron_<YYYYMMDD>.json + append session-state row + commit via canonical serializer with --expected-content-sha256
8. RELEASE: tools/codex_loop_coordination.py release --event-id <id> --outcome success

CAPS: this loop NEVER dispatches paid GPU. Reads only; mutates only APPEND-ONLY ledgers + archival.

FAILURE → status='blocked'+halt+relay-via-inbox: archive_disk_full / state_corruption_catalog_138 / coordination_collision_higher_priority_loop (yield + retry next iteration).

OBSERVABILITY: tools/codex_loop_coordination.py summary; tools/operator_briefing.py extended fields.

GO. Start at step 1.
```

### 8.3 Loop 3: public-PR-monitor /goal

```
ROLE: Codex public-PR-monitor loop for pact. Per CLAUDE.md "Public frontier watch and intake" non-negotiable; monitors comma contest GitHub repo; intakes new PRs that may beat current frontier.

DOMAIN: same as canonical-task-execution. PRs are external; intake produces internal canonical_task_status rows for canonical-task-execution to dispatch.

PRINCIPLES: per CLAUDE.md "Public frontier watch and intake" + "Bit-level deconstruction" + Catalog #109 (clones pristine) + Catalog #316 (frontier drift) + zero paid-GPU spend within this loop (intake only; replay enqueued for canonical-task-execution).

POINTERS (read EVERY invocation):
- CLAUDE.md + AGENTS.md
- .omx/state/codex_public_pr_monitor_session_state.jsonl (resume)
- .omx/state/{public_pr_monitor_last_polled_at,contest_window_active.flag}
- reports/latest.md (current frontier)
- glob experiments/results/public_pr*_intake_*/ (prior intakes)

LOOP (no operator intervention):
1. PRE-FLIGHT: read pointers + resume + claim file scope via tools/codex_loop_coordination.py
2. POLL: gh api repos/commaai/contest/pulls --since <last_polled_at>
3. TRIAGE: for each new/updated PR, parse claimed score from PR body/comments; if claim < frontier per Catalog #316, mark for intake
4. INTAKE: tools/intake_public_pr.py --pr <N> --canonical-dir experiments/results/public_pr<N>_intake_<utc>/ (PRISTINE per Catalog #109)
5. ENQUEUE: append canonical_task_status row owner=codex kind=public_pr_replay priority=high if intake passes compliance gate (Catalog #99 representation promotion card check)
6. PERSIST: write .omx/state/public_pr_monitor_last_polled_at + append session-state row
7. RELEASE: tools/codex_loop_coordination.py release

CAPS: never dispatch GPU directly; enqueue replay tasks for canonical-task-execution. github_api_rate_limited → back off + retry next iteration.

FAILURE → status='blocked'+relay: intake_failed_compliance_violation / pr_clone_dirty_violation_catalog_109 / pr_signal_axis_destruction_catalog_297.

OBSERVABILITY: tools/codex_loop_coordination.py summary; tools/operator_briefing.py.

GO. Start at step 1.
```

### 8.4 Loop 4: hf-hub-publisher /goal

```
ROLE: Codex hf-hub-publisher loop for pact. Per ITEM_12 canonical_task_status + CLAUDE.md "Public Disclosure Hygiene". Operator-triggered or release-event-triggered HF dataset/model push.

DOMAIN: release packets ONLY (not raw research artifacts). Sanitization is the discriminator.

PRINCIPLES: per CLAUDE.md "Public Disclosure Hygiene" + Catalog #208 (docs no local absolute paths) + apples-to-apples (sanitization preserves the semantic content but strips private details).

POINTERS (read EVERY invocation):
- CLAUDE.md + AGENTS.md
- .omx/state/codex_hf_hub_publisher_session_state.jsonl (resume)
- .omx/state/hf_hub_publisher_trigger.flag (operator trigger)
- canonical_task_status rows with kind=hf_release status=pending
- .omx/oss_export/ (prior release manifests)

LOOP (no operator intervention):
1. PRE-FLIGHT: check trigger flag OR enumerate pending hf_release tasks; if none, sleep + return to 1 (or exit)
2. CLAIM: tools/codex_loop_coordination.py claim --loop-id hf-hub-publisher --files ".omx/oss_export/,experiments/results/hf_release_*"
3. SANITIZE: tools/build_oss_release_packet.py --strip-private-paths --strip-credentials; verify zero Catalog #208 violations on output
4. PUSH: hf datasets push (or hf models push); track upload progress
5. MODEL CARD: tools/build_release_manifest.py --emit-hf-model-card if release includes weights
6. VERIFY: pull-back-and-compare against published artifact (apples-to-apples per CLAUDE.md)
7. PERSIST: update canonical_task_status row to status=completed; append session-state row; clear trigger flag; relay completion via inbox
8. RELEASE: tools/codex_loop_coordination.py release

CAPS: HF subscription approved (no marginal cost). Sanitization gate is fail-closed (refuses release on any private-state leak).

FAILURE → status='blocked'+relay: hf_api_auth_failed / sanitization_violation_catalog_208 / model_card_schema_violation / verification_mismatch.

OBSERVABILITY: tools/codex_loop_coordination.py summary; HF dashboard.

GO. Start at step 1.
```

### 8.5 Loop 5: adversarial-review-of-claude-work /goal

```
ROLE: Codex adversarial-review-of-claude-work loop for pact. Reviews Claude's design memos per Catalog #229 sister discipline. INVERSE direction from the codex:adversarial-review skill (which is operator-invoked); this loop AUTO-fires per design-memo-landing trigger.

DOMAIN: .omx/research/*_design_memo_*.md + feedback_*_landed_*.md ONLY. Read-mostly; writes only codex_findings_*_codex.md.

PRINCIPLES: per CLAUDE.md "Subagent coherence-by-default" + Catalog #229 + Catalog #271 verdict taxonomy. Reviews are POST-HOC; never blocks Claude landing memos (relay channel surfaces verdicts; operator decides).

POINTERS (read EVERY invocation):
- CLAUDE.md + AGENTS.md
- .omx/state/codex_review_of_claude_session_state.jsonl (resume)
- .omx/state/claude_design_memo_review_queue.jsonl (pending queue)
- glob .omx/research/*_design_memo_*.md (queued items)
- glob .omx/research/codex_findings_*_codex.md (prior reviews)

LOOP (no operator intervention):
1. PRE-FLIGHT: read pointers + resume + claim file scope via tools/codex_loop_coordination.py
2. TRIAGE: enumerate pending queue rows; skip T1 council deliberations (per Catalog #300 tier classification)
3. REVIEW: invoke codex companion via codex exec --skip-git-repo-check --sandbox read-only -m gpt-5.5 -c model_reasoning_effort=xhigh; pass memo + CLAUDE.md + AGENTS.md + sister memos + canonical state context
4. VERDICT: per Catalog #271 verdict taxonomy {approve, advisory, needs-attention, no-ship}; emit codex_findings_<slug>_<utc>_codex.md
5. RELAY: if verdict in {needs-attention, no-ship}, tools/codex_to_claude_inbox.py relay --relay "Adversarial review of <memo>: verdict <V>; findings at <path>"
6. PERSIST: update queue row status=reviewed; append session-state row
7. RELEASE: tools/codex_loop_coordination.py release

CAPS: codex CLI included in Codex Pro subscription; xhigh reasoning ~$0.10-0.30 per review. Per Catalog #281 fail-closed on companion timeout/crash.

FAILURE → status='blocked'+relay: codex_companion_timeout (Catalog #281 rc=2) / cache_stale_dirty_tree (Catalog #282) / review_finds_blocking_violation (relay; do NOT auto-amend).

OBSERVABILITY: tools/codex_loop_coordination.py summary; tools/operator_briefing.py extended with review verdict distribution.

GO. Start at step 1.
```

---

## 9. Cargo-cult audit per assumption (Catalog #303)

| # | Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|---|
| 1 | "5 loops is the right shape" | CARGO-CULTED-PENDING-VERIFICATION | Operator-proposed without empirical comparison vs ONE-loop-with-sub-modes OR 3-loop continuous-nightly-reactive | 30-day post-activation retrospective per CLAUDE.md "Mission alignment" Consequence 3 |
| 2 | "Loops are non-interfering when given disjoint file scope" | HARD-EARNED | Catalog #302 + #314 empirical anchors confirm sister-subagent file-scope discipline is structural primitive | None needed |
| 3 | "Cron + on-demand + per-trigger covers all cadences" | HARD-EARNED | GHA + Modal + canonical_task_status priority semantics empirically cover all observed cadences | None needed |
| 4 | "Session-state ledger pattern scales to N loops" | HARD-EARNED | Catalog #245 4-proc stress test validated N=4 concurrent; N=5 within envelope | None needed |
| 5 | "Operator's role is unchanged when 5 loops run in parallel" | CARGO-CULTED | 5× parallel produces 5× telemetry; without aggregation surface, operator becomes bottleneck | tools/operator_briefing.py extension + Catalog #300 mission-alignment frontmatter aggregation |
| 6 | "Loop priority ordering (canonical-task-execution highest) is canonical" | CARGO-CULTED | Inherited from operator's intuition; never empirically validated against e.g., "public-PR-monitor highest during contest" alternative | Retrospective: did canonical-task-execution actually need priority over public-PR-monitor during contest windows? |
| 7 | "Expected-completion-utc bounded claims auto-expire correctly" | CARGO-CULTED-PENDING-VERIFICATION | Inherited from Catalog #245 24h TTL pattern but loop iterations may genuinely exceed claim deadline | 7-day verification: did >5% of claims auto-expire? if yes, deadline-calibration is wrong |
| 8 | "Codex CLI handles 5 parallel sessions without rate-limit collision" | CARGO-CULTED-PENDING-VERIFICATION | Codex Pro subscription terms unclear on N-parallel-sessions; rate limits may compound | First-week monitoring: any rate-limit events? if multiple, MUST stagger session start times |

---

## 10. 9-dimension success checklist evidence (Catalog #294)

### Dimension 1 — UNIQUENESS (class-shift not within-class)
**Class-shift**: from ONE-loop autonomous execution to N-loop parallel execution. The class-shift is structural (concurrency primitive change), not within-class polish (extending LOOP step count or pointer set).

### Dimension 2 — BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable)
Coordination surface is ONE canonical helper (src/tac/codex_loop_coordination.py ~600 LOC). Five /goal templates total ~250 LOC of human-readable prompts. STRICT gate is ~80 LOC mirroring sister Catalog #245 patterns. Total ~930 LOC of new code reviewable in <30 minutes by domain expert.

### Dimension 3 — DISTINCTNESS (explicitly different from sisters)
Distinct from Catalog #131 (per-state-file write discipline), Catalog #302 (per-subagent file-scope), Catalog #314 (per-bare-commit absorption). This memo's surface is per-LOOP file-scope claim-and-release pattern — orthogonal to all sister surfaces.

### Dimension 4 — RIGOR (premise verification + adversarial review + assumption classification + empirical anchor)
- Premise verification: 9 mandatory pre-flight reads completed per Catalog #229
- Adversarial review: Contrarian + Carmack dissent recorded verbatim per Catalog #300
- Assumption classification: 8 assumptions per §9 with HARD-EARNED vs CARGO-CULTED
- Empirical anchor: Catalog #302 + #314 prior incidents establish the bug class this extincts

### Dimension 5 — OPTIMIZATION PER TECHNIQUE (covered by sister Catalog #290 canonical-vs-unique decision per layer)
See §11 below.

### Dimension 6 — STACK-OF-STACKS-COMPOSABILITY (orthogonal axes + additive ΔS)
Stacks with:
- Catalog #245 (Modal call_id ledger) — coordination layer rides on top
- Catalog #331 (Codex→Claude inbox) — relay channel cross-references coordination event_ids
- Catalog #332 (memory hermetic export, when claimed by sister directive) — exports may reference loop session-state ledgers
- Catalog #316 (frontier scan) — nightly-cron loop consumes its outputs
- Catalog #284 (nightly catalog gate regression) — runs in parallel with nightly-cron loop; coordination surface ensures no collision

### Dimension 7 — DETERMINISTIC REPRODUCIBILITY (byte-stable + seed-pinned)
JSONL is byte-stable via `sort_keys=True` + `ensure_ascii=False` per Catalog #245. Event_id generation uses `secrets.token_urlsafe(12)` — reproducibility is per-event not per-replay (operationally fine).

### Dimension 8 — EXTREME OPTIMIZATION + PERFORMANCE
fcntl LOCK_EX per Catalog #131; amortized O(1) append per Catalog #245 sidecar-index pattern. Cross-loop coordination overhead estimated <100ms per claim event.

### Dimension 9 — OPTIMAL MINIMAL CONTEST SCORE
INDIRECT mission contribution: multi-loop multiplies operator-attention capacity, which multiplies frontier-pursuit cadence, which multiplies contest score improvement rate. Direct ΔS: 0; indirect ΔS: depends on retrospective.

---

## 11. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Canonical helper | Decision | Rationale |
|---|---|---|---|
| fcntl-locked JSONL append | `tac.deploy.modal.call_id_ledger._append_event_locked` pattern | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #245 4-proc spawn-pool stress validated; same primitives serve this loop coordination surface |
| Strict-load fail-closed | `tac.deploy.lightning.active_jobs_state.load_active_jobs_strict` pattern | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #138 sister discipline; same fail-closed pattern |
| APPEND-ONLY HISTORICAL_PROVENANCE | Catalog #110/#113 | ADOPT_CANONICAL_BECAUSE_SERVES | Same audit-trail discipline; loop coordination IS forensic provenance |
| Schema validation in __post_init__ | `tac.council_continual_learning.CouncilDeliberationRecord` pattern | ADOPT_CANONICAL_BECAUSE_SERVES | Same dataclass + invariants pattern |
| Loop priority ordering | None canonical | FORK_BECAUSE_PRINCIPLED | This is the unique-to-this-substrate engineering decision (LOOP_PRIORITY dict; canonical-task-execution=100) |
| Conflict resolution algorithm | None canonical | FORK_BECAUSE_PRINCIPLED | Preempt-lower-priority + refuse-same-priority is unique; no sister has this exact semantic |
| Event_id generation | `secrets.token_urlsafe(12)` per Catalog #131 sister | ADOPT_CANONICAL_BECAUSE_SERVES | Universal pattern across all canonical fcntl-locked ledgers |
| Operator-facing CLI | `tools/check_predecessor_probe_outcome.py` pattern | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #313 sister 4-layer pattern |
| STRICT preflight gate | `check_*` function in `src/tac/preflight.py` per Catalog #176 | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #176 + #185 META-meta drift detection |
| /goal prompt template | None canonical | FORK_BECAUSE_PRINCIPLED | Each loop's /goal is unique to its work-domain; canonical is the STRUCTURAL form (ROLE / POINTERS / LOOP / CAPS / FAILURE / OBSERVABILITY) — content is per-loop |
| GHA workflow | `.github/workflows/nightly_catalog_gate_regression.yml` pattern | ADOPT_CANONICAL_BECAUSE_SERVES | Cron + Codex CLI invocation + status JSON emission per HISTORICAL_PROVENANCE |
| Operator briefing extension | `tools/operator_briefing.py` (when extended per directive) | ADOPT_CANONICAL_BECAUSE_SERVES | Same operator-facing aggregation pattern |

---

## 12. Observability surface (Catalog #305)

### 12.1 Inspectable per layer
- Per-claim event row inspectable via `tools/codex_loop_coordination.py file-history --file <path>`
- Per-loop session-state row inspectable via `cat .omx/state/codex_<loop>_session_state.jsonl | jq`
- Per-cross-loop collision inspectable via `tools/codex_loop_coordination.py collision-report --hours 24`

### 12.2 Decomposable per signal
- `loop_id` decomposable from coordination ledger
- `files_to_claim` decomposable per file
- `outcome` (success / yielded / error) decomposable per loop
- `priority` decomposable per LOOP_PRIORITY constant

### 12.3 Diff-able across runs
- Two nightly-cron iterations are diff-able via their session-state rows (items_landed delta)
- Two coordination ledgers diff-able via JSON line-diff

### 12.4 Queryable post-hoc
- DuckDB integration: register `.omx/state/codex_loop_coordination.jsonl` as a canonical_duckdb view (sister of `canonical_task_status_by_memo` pattern)
- Operator queries: "show all canonical-task-execution claims on src/tac/preflight.py in last 7 days"

### 12.5 Cite-able
- Each coordination event has `event_id` for cite-chain
- Session-state rows reference `files_claimed_in_coordination` event_ids

### 12.6 Counterfactual-able
- "What if nightly-cron had not preempted public-PR-monitor?" — coordination ledger preempt event row carries `preempted_by` field so counterfactual reconstruction is possible
- Catalog #139 packet compiler no-op detector sister pattern applies: if a loop's claim event has zero downstream session-state writes, the claim was a no-op

---

## 13. Cycle handling — loops produce inbox questions other loops answer

This is the structural mechanism enabled by the just-landed Catalog #331 inbox channel.

### 13.1 Example flow

1. **public-PR-monitor** detects new PR #142 claiming score 0.187 [contest-CPU]
2. **public-PR-monitor** triages: claim < current frontier 0.19205 → INTAKE
3. Intake succeeds + compliance gate passes
4. **public-PR-monitor** enqueues canonical_task_status row `owner=codex kind=public_pr_replay priority=high pr_number=142`
5. **canonical-task-execution** LOOP step 3 SELECT picks up the row
6. **canonical-task-execution** EXECUTE hits ambiguity: "PR #142 archive has unusual ZIP member ordering; should I respect their ordering OR re-canonicalize?"
7. **canonical-task-execution** invokes `tools/codex_to_claude_inbox.py ask --blocking-task-id <id> --question "PR #142 ZIP ordering: respect OR re-canonicalize?" --suggested-options "respect|re-canonicalize" --codex-default-if-no-response "respect" --response-deadline-utc <utc+4h>`
8. Question becomes blocking on canonical-task-execution loop
9. Claude receives inbox question via `tools/codex_to_claude_inbox.py poll-for-claude` in Claude's next session
10. Claude appends answer via `tools/codex_to_claude_inbox.py answer` referencing `.omx/research/claude_response_to_codex_<event_id>_<utc>.md`
11. **canonical-task-execution** LOOP step 1 PRE-FLIGHT picks up answer; unblocks task; proceeds with answer's recommendation
12. Replay completes; results enter posterior; **canonical-task-execution** PERSIST step relays "PR #142 replay complete, archive sha <sha>, score <s>" via inbox
13. **adversarial-review-of-claude-work** loop later reviews Claude's answer memo + posterior update; emits codex_findings memo

### 13.2 Loop-to-loop event_id references

The coordination ledger + inbox channel + claude memory export channel together form a **federated event graph** where any event in any ledger can reference any event_id in any other ledger via `context_pointers` / `response_to_event_id` / `cite_chain` fields. This is the structural alternative to a centralized event bus.

---

## 14. Cost envelope per loop

| Loop | Per-iteration cost | Per-day cost | Notes |
|---|---|---|---|
| canonical-task-execution | $0 (Codex Pro covers CLI; paid GPU is per-canonical-task-status-row not per-loop-iteration) | $0-$50 (depends on task mix) | Already running; no incremental cost |
| nightly-cron | $0 | $0 | Read-only; no paid operations |
| public-PR-monitor | $0 (GitHub API + clone-only) | $0 | Replay cost belongs to canonical-task-execution downstream |
| hf-hub-publisher | $0 (HF subscription approved) | $0 | Marginal cost zero per release |
| adversarial-review-of-claude-work | ~$0.10-0.30 per review (xhigh reasoning) | $0-$5 | Codex Pro tokens; bounded by review queue depth |

**Total per-day incremental cost: $0-$5** (dominated by adversarial-review-of-claude-work token usage). Per-day paid-GPU spend remains gated by canonical-task-execution loop's per-task budgets per Catalog #270 + #325.

---

## 15. TOP-5 op-routables ranked by EV (with concrete file paths)

### #1 (HIGHEST EV) — Land canonical helper + CLI + STRICT gate

- **Files**:
  - `src/tac/codex_loop_coordination.py` (~600 LOC; mirror Catalog #245 4-layer template at `src/tac/deploy/modal/call_id_ledger.py`)
  - `src/tac/tests/test_codex_loop_coordination.py` (30+ tests mirror `src/tac/tests/test_modal_call_id_ledger.py`)
  - `tools/codex_loop_coordination.py` (~400 LOC CLI; 8 subcommands per §3.3)
  - `src/tac/preflight.py` (add Catalog #332 gate `check_codex_loop_coordination_no_file_collision`; wire warn-only)
- **Cost**: $0 (CPU-only work)
- **Wall-clock**: ~4-6h Codex execution
- **Owned by**: Codex per directive emitted by this memo

### #2 — Land nightly-cron loop (lowest-stakes infrastructure activation)

- **Files**:
  - `.github/workflows/codex_nightly_cron.yml` (mirror nightly_catalog_gate_regression.yml structure; cron `0 0 * * *`)
  - `.omx/research/codex_nightly_cron_goal_20260518.md` (paste-ready /goal per §8.2)
  - `.omx/state/codex_nightly_cron_session_state.jsonl` (init schema row)
- **Cost**: $0
- **Wall-clock**: ~2-3h Codex execution
- **Depends on**: #1 landing first

### #3 — Land adversarial-review-of-claude-work queue surface

- **Files**:
  - `.omx/state/claude_design_memo_review_queue.jsonl` (init schema row)
  - `.git/hooks/post-commit` (extend with auto-enqueue for `.omx/research/*_design_memo_*.md` files)
  - `.omx/research/codex_review_of_claude_goal_20260518.md` (paste-ready /goal per §8.5)
  - `.omx/state/codex_review_of_claude_session_state.jsonl` (init)
- **Cost**: ~$0.10-0.30 per review iteration (codex companion tokens)
- **Wall-clock**: ~2-3h Codex execution
- **Depends on**: #1 landing first

### #4 — Land public-PR-monitor loop (contest-window gated)

- **Files**:
  - `.github/workflows/codex_public_pr_monitor.yml` (cron `0 * * * *` during contest; `0 0 * * *` otherwise)
  - `tools/intake_public_pr.py` (NEW; ~300 LOC; per CLAUDE.md "Public frontier watch and intake" non-negotiable)
  - `.omx/state/codex_public_pr_monitor_session_state.jsonl` (init)
  - `.omx/state/contest_window_active.flag` (sentinel; touched by operator when contest active)
  - `.omx/research/codex_public_pr_monitor_goal_20260518.md` (paste-ready /goal per §8.3)
- **Cost**: $0 (intake only; replay cost belongs to canonical-task-execution)
- **Wall-clock**: ~4-6h Codex execution (intake tool is non-trivial)
- **Depends on**: #1 + #2 landing first

### #5 — Land hf-hub-publisher loop (on-demand)

- **Files**:
  - `tools/build_oss_release_packet.py` (NEW; ~400 LOC; sanitization per CLAUDE.md "Public Disclosure Hygiene")
  - `.omx/state/codex_hf_hub_publisher_session_state.jsonl` (init)
  - `.omx/state/hf_hub_publisher_trigger.flag` (sentinel; touched by operator to trigger)
  - `.omx/research/codex_hf_hub_publisher_goal_20260518.md` (paste-ready /goal per §8.4)
- **Cost**: $0 (HF subscription approved)
- **Wall-clock**: ~6-10h Codex execution (sanitization tool is the bulk; per ITEM_12 already in queue)
- **Depends on**: #1 landing first; ITEM_12 canonical_task_status row triggers it

---

## 16. 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map contribution**: N/A (infrastructure layer; no substrate signal)
2. **Pareto constraint**: N/A
3. **Bit-allocator hook**: N/A
4. **Cathedral autopilot dispatch hook**: **ACTIVE** — coordination ledger's `loop_coordination_active_claims_count` becomes a priority-routing signal for canonical-task-execution's dispatcher (deferring high-cost dispatches when coordination collision pressure is elevated). Specifically: `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_coordination_pressure` (new helper to add).
5. **Continual-learning posterior update**: **ACTIVE** — successful loop iterations emit anchors via `tac.council_continual_learning.append_council_anchor` for operator-attention-budget tracking per CLAUDE.md "Mission alignment" Consequence 5 (mission_predicted_contribution distribution per-loop). Specifically: each loop's PERSIST step appends a posterior anchor with `predicted_mission_contribution=apparatus_maintenance` (default for loops 2-5) OR `=frontier_breaking` (for canonical-task-execution rows whose result is a contest-CUDA/CPU anchor below frontier).
6. **Probe-disambiguator**: **ACTIVE** — when a canonical task has 2+ defensible interpretations AND the canonical-task-execution loop hits the ambiguity, the inbox channel (Catalog #331) IS the disambiguator. The 5-loop architecture inherits this from sister Catalog #331.

---

## 17. Cross-references

### Sister directives
- `.omx/research/codex_persistent_goal_v2_4_no_hardcoded_state_20260518.md` (current active /goal; canonical-task-execution loop)
- `.omx/research/codex_persistent_goal_v2_5_with_inbox_integration_20260518.md` (paste-pending v2.5 with inbox channel integration)
- `.omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` (Catalog #331; inbox channel directive)
- `.omx/research/codex_routing_directive_claude_memory_hermetic_export_channel_20260518.md` (memory export channel; needs Catalog # claim AFTER #332)
- `.omx/research/codex_routing_directive_canonical_task_status_single_source_of_truth_20260518.md` (canonical_task_status.jsonl single source of truth)
- `.omx/research/codex_routing_directive_canonical_task_status_duckdb_consumer_sidecar_20260518.md` (DuckDB integration; #11 sister)

### Sister CLAUDE.md non-negotiables
- "Subagent coherence-by-default" (Catalog #302 sister discipline)
- "Bugs must be permanently fixed AND self-protected against" (Catalog #332 STRICT gate self-protection)
- "Operator gates must be wired and used" (CLI + operator_briefing.py + all_lanes_preflight.py wire-ins)
- "Mission alignment" (Consequences 1-5 per Catalog #300 mission_alignment frontmatter)
- "Beauty, simplicity, and developer experience" (canonical helpers + machine-checkable artifacts)
- "Council hierarchy: 4-tier protocol" (this memo is T2 sextet pact deliberation)
- "Codex CLI invocation" (Pattern A detached invocation for GHA workflows)
- "Public Disclosure Hygiene" (hf-hub-publisher sanitization gate)
- "Public frontier watch and intake" (public-PR-monitor canonical implementation)
- "State JSONL archival policy" (nightly-cron archive responsibility)

### Sister catalog gates
- Catalog #117 (subagent commit serializer must be used)
- Catalog #125 (subagent landing has solver wire-in — this memo's 6-hook declaration)
- Catalog #126 (lane pre-registered before work starts — `lane_multi_loop_codex_goal_design_20260518` registered)
- Catalog #131 (no bare writes to shared state — coordination ledger registered)
- Catalog #138 (strict-load discipline — `load_coordination_strict` mirrors)
- Catalog #157 (commit serializer pre-lock hash — POST-EDIT sha required)
- Catalog #174 (`--expected-content-sha256` mandatory)
- Catalog #176 (strict callsites have CLAUDE.md row — Catalog #332 row to be added)
- Catalog #185 (live count zero claims verified empirically)
- Catalog #186 (catalog # claimed via canonical serializer — done; #332 claimed)
- Catalog #206 (subagent checkpoint discipline — checkpoint logged)
- Catalog #229 (premise verification before edit — 9 pre-flight reads done)
- Catalog #230 (bulk-rewrite respects sister-subagent ownership map — sister B disjoint scope confirmed)
- Catalog #245 (canonical 4-layer pattern — coordination ledger mirrors)
- Catalog #271 (pre-dispatch codex review for paid dispatch — adversarial-review-of-claude-work sister)
- Catalog #281/#282/#283 (codex review fail-closed family — adversarial-review-of-claude-work inherits)
- Catalog #284 (nightly catalog gate regression — sister nightly workflow)
- Catalog #287 (empirical claims have evidence — this memo cites empirical anchors)
- Catalog #290 (canonical-vs-unique decision per layer — §11 satisfies)
- Catalog #291 (session has recent META-ASSUMPTION review — current session has Catalog #303 cargo-cult audit)
- Catalog #292 (grand council deliberation has explicit assumption statements — §9 satisfies)
- Catalog #294 (9-dimension success checklist evidence — §10 satisfies)
- Catalog #300 (council deliberation declares tier in frontmatter — frontmatter satisfies)
- Catalog #302 (sister subagent scope overlap via checkpoint JSONL — per-subagent surface; #332 is per-LOOP)
- Catalog #303 (cargo-cult audit section — §9 satisfies)
- Catalog #305 (observability surface — §12 satisfies)
- Catalog #309 (substrate design memo declares horizon class — frontmatter satisfies; `frontier_pursuit`)
- Catalog #314 (no subagent files touched absorption in bare commits — sister bug class at bare-commit surface)
- Catalog #316 (reports/latest.md not stale vs canonical frontier — nightly-cron loop consumes)
- Catalog #325 (per-substrate optimal form via symposium — this memo IS the T2 sextet symposium for the multi-loop substrate)
- Catalog #331 (Codex→Claude inbox bidirectional channel — sister 4-layer Codex/Claude channel)
- Catalog #332 (THIS memo's STRICT gate — coordination collision detector)

### Sister tools
- `tools/canonical_task_status.py` (work queue; loop 1 consumes)
- `tools/codex_to_claude_inbox.py` (Catalog #331; loops 2/3/5 emit relays)
- `tools/claude_memory_export.py` (sister; loops consume Claude exports)
- `tools/subagent_commit_serializer.py` (canonical commit serializer with POST-EDIT sha)
- `tools/claim_catalog_number.py` (canonical catalog # claim with serializer)
- `tools/lane_maturity.py` (canonical lane registry mutation)
- `tools/subagent_checkpoint.py` (canonical checkpoint discipline)
- `tools/archive_jsonl_state.py` (nightly-cron loop consumer)
- `tools/scan_best_anchor_per_axis.py` (nightly-cron loop consumer; Catalog #316)
- `tools/audit_stale_l1_substrates.py` (nightly-cron loop consumer; Catalog #298)
- `tools/operator_briefing.py` (extended with loop coordination fields)

---

## 18. Council verdict + continual-learning anchor

### Verdict
**PROCEED_WITH_REVISIONS** (5-of-6 sextet; Contrarian PROCEED_WITH_REVISIONS with binding stipulations; all binding stipulations honored)

### Binding stipulations honored
1. ✅ STRICT gate Catalog #332 specified
2. ✅ Sequential activation order specified (§6)
3. ✅ ONE-loop-with-sub-modes alternative explicitly addressed (§5)
4. ✅ 30-day post-activation retrospective queued (per CLAUDE.md "Mission alignment" Consequence 3)
5. ✅ 3-loop alternative reactivation criteria specified (§5.4)
6. ✅ Carmack concession recorded with reactivation criteria
7. ✅ Mission-alignment frontmatter records `frontier_breaking` predicted contribution

### Continual-learning anchor emission

Per Catalog #300 + #125 hook #5, this memo's landing emits an anchor to `.omx/state/council_deliberation_posterior.jsonl` via the canonical helper. The anchor's emission is documented as op-routable but the operator does the actual append AFTER memo lands (this avoids the circular dependency where the memo cites the anchor it itself emits).

Suggested append invocation:
```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="multi_loop_codex_goal_design_memo_20260518",
    topic="multi-loop codex /goal autonomous execution (5 parallel loops + coordination)",
    council_tier=CouncilTier.T2,
    council_attendees=("Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"),
    council_quorum_met=True,
    council_verdict="PROCEED_WITH_REVISIONS",
    council_dissent=(
        {"member": "Contrarian", "verbatim": "5 loops is engineering bloat unless..."},
        {"member": "Carmack", "verbatim": "Engineering reductionist position: cron + on-demand + per-trigger is THREE cadences not five loops..."},
    ),
    council_assumption_adversary_verdict=(
        {"assumption": "5 loops is the right shape", "classification": "CARGO-CULTED-PENDING-VERIFICATION", "rationale": "Five was operator-proposed at directive-write time but never empirically validated..."},
        # ... (5 more per §9)
    ),
    council_decisions_recorded=(
        "op-routable #1: Land canonical helper src/tac/codex_loop_coordination.py",
        "op-routable #2: Land CLI tool tools/codex_loop_coordination.py",
        "op-routable #3: Land STRICT preflight gate Catalog #332",
        "op-routable #4: Sequential activation order per ACTIVATION ORDER section",
        "op-routable #5: Codex prompt template per loop (5 canonical /goal copy-paste blocks)",
    ),
    predicted_mission_contribution="frontier_breaking",
    override_invoked=False,
    override_rationale=None,
    deferred_substrate_retrospective_due_utc="2026-06-17T00:00:00Z",  # 30 days post-activation
    deferred_substrate_id=None,
    related_deliberation_ids=(
        "codex_persistent_goal_v2_4_no_hardcoded_state_20260518",
        "codex_persistent_goal_v2_5_with_inbox_integration_20260518",
        "codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518",
        "codex_routing_directive_claude_memory_hermetic_export_channel_20260518",
    ),
)
append_council_anchor(record)
```

---

## Appendix A — `.github/workflows/codex_nightly_cron.yml` skeleton

```yaml
name: codex_nightly_cron

# Codex nightly-cron loop (per multi_loop_codex_goal_design_memo_20260518)
# Sister of nightly_catalog_gate_regression.yml; runs at 00:00 UTC (vs 02:00 UTC)
# to avoid cron collision while preserving operator sleep window.
#
# Per CLAUDE.md "Codex CLI invocation" Pattern A (detached invocation)
# Per CLAUDE.md "State JSONL archival policy"
# Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"

on:
  schedule:
    - cron: '0 0 * * *'
  workflow_dispatch:
    inputs:
      skip_archive:
        description: 'Skip the JSONL archival stage'
        required: false
        type: boolean
        default: false

concurrency:
  group: codex_nightly_cron-${{ github.ref }}
  cancel-in-progress: false

permissions:
  contents: write   # to commit nightly status JSON via canonical serializer
  actions: read

jobs:
  codex_nightly_cron:
    runs-on: ubuntu-latest
    timeout-minutes: 45

    env:
      INFLATE_TORCH_SPEC: 'torch==2.5.1+cpu'
      UV_EXTRA_INDEX_URL: 'https://download.pytorch.org/whl/cpu'
      UV_INDEX_STRATEGY: 'unsafe-best-match'
      DALI_DISABLE_NVML: '1'
      CUBLAS_WORKSPACE_CONFIG: ':4096:8'

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Install uv + Python deps
        run: |
          set -euo pipefail
          curl -LsSf https://astral.sh/uv/install.sh | sh
          export PATH="$HOME/.local/bin:$PATH"
          uv venv
          uv pip install -e .

      - name: Claim file scope via coordination surface
        run: |
          set -euo pipefail
          .venv/bin/python tools/codex_loop_coordination.py claim \
              --loop-id "nightly-cron" \
              --files ".omx/state/archive/,.omx/state/codex_nightly_cron_*.json,.omx/state/codex_nightly_cron_session_state.jsonl" \
              --expected-completion-utc "$(date -u -v+1H +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '+1 hour' +%Y-%m-%dT%H:%M:%SZ)" \
              --session-id "gha-${{ github.run_id }}" \
              --purpose "nightly audit + archival + frontier scan"

      - name: Invoke codex nightly-cron loop (Pattern A detached)
        run: |
          set -euo pipefail
          mkdir -p .omx/tmp/codex_runs
          GOAL_TEXT=$(cat .omx/research/codex_nightly_cron_goal_20260518.md | sed -n '/^## BEGIN COPY-PASTE BLOCK/,/^## END COPY-PASTE BLOCK/p')
          codex exec --skip-git-repo-check --sandbox read-only \
              -m gpt-5.5 -c model_reasoning_effort=xhigh \
              -o .omx/tmp/codex_runs/nightly-cron-${{ github.run_id }}.last.txt \
              "$GOAL_TEXT" \
              2>&1 | tee .omx/tmp/codex_runs/nightly-cron-${{ github.run_id }}.log

      - name: Release coordination claim
        if: always()
        run: |
          set -euo pipefail
          # release is idempotent; safe to call even on workflow failure
          LATEST_CLAIM=$(.venv/bin/python tools/codex_loop_coordination.py summary --format=json | jq -r '.[0].event_id')
          .venv/bin/python tools/codex_loop_coordination.py release \
              --event-id "$LATEST_CLAIM" \
              --actual-completion-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
              --outcome "${{ job.status == 'success' && 'success' || 'error' }}" \
              --session-id "gha-${{ github.run_id }}"
```

---

## Appendix B — `.github/workflows/codex_public_pr_monitor.yml` skeleton

```yaml
name: codex_public_pr_monitor

on:
  schedule:
    # Hourly during contest window (operator toggles via flag file)
    - cron: '0 * * * *'
  workflow_dispatch:
  repository_dispatch:
    types: [comma_contest_pr_event]

concurrency:
  group: codex_public_pr_monitor-${{ github.ref }}
  cancel-in-progress: false

permissions:
  contents: write
  pull-requests: read
  actions: read

jobs:
  codex_public_pr_monitor:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Check contest window flag
        id: window
        run: |
          set -euo pipefail
          if [[ -f .omx/state/contest_window_active.flag ]]; then
            echo "active=true" >> "$GITHUB_OUTPUT"
          else
            # During off-contest: only run on daily cron tick (skip hourly)
            HOUR=$(date -u +%H)
            if [[ "$HOUR" != "00" ]]; then
              echo "active=false" >> "$GITHUB_OUTPUT"
              echo "[public-PR-monitor] off-contest skip; next daily fire at 00 UTC"
              exit 0
            fi
            echo "active=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Install uv + Python deps
        if: steps.window.outputs.active == 'true'
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          export PATH="$HOME/.local/bin:$PATH"
          uv venv && uv pip install -e .

      - name: Claim file scope
        if: steps.window.outputs.active == 'true'
        run: |
          .venv/bin/python tools/codex_loop_coordination.py claim \
              --loop-id "public-pr-monitor" \
              --files "experiments/results/public_pr,.omx/state/codex_public_pr_monitor_session_state.jsonl,.omx/state/public_pr_monitor_*.json" \
              --expected-completion-utc "$(date -u -d '+30 minutes' +%Y-%m-%dT%H:%M:%SZ)" \
              --session-id "gha-${{ github.run_id }}" \
              --purpose "hourly public PR poll + intake"

      - name: Invoke codex public-PR-monitor loop
        if: steps.window.outputs.active == 'true'
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          GOAL_TEXT=$(cat .omx/research/codex_public_pr_monitor_goal_20260518.md | sed -n '/^## BEGIN COPY-PASTE BLOCK/,/^## END COPY-PASTE BLOCK/p')
          codex exec --skip-git-repo-check --sandbox read-only \
              -m gpt-5.5 -c model_reasoning_effort=xhigh \
              "$GOAL_TEXT" 2>&1 | tee .omx/tmp/codex_runs/public-pr-monitor-${{ github.run_id }}.log

      - name: Release coordination claim
        if: always() && steps.window.outputs.active == 'true'
        run: |
          LATEST=$(.venv/bin/python tools/codex_loop_coordination.py summary --format=json | jq -r '.[0].event_id')
          .venv/bin/python tools/codex_loop_coordination.py release \
              --event-id "$LATEST" \
              --actual-completion-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
              --outcome "${{ job.status == 'success' && 'success' || 'error' }}" \
              --session-id "gha-${{ github.run_id }}"
```

---

## Appendix C — CLAUDE.md catalog row for #332 (drafted for Codex to apply during gate landing)

```
332. `check_codex_loop_coordination_no_file_collision` — Multi-loop Codex /goal coordination self-protection 2026-05-18 (per operator standing context "Should you be able to delegate other kinds of work or tasks or projects" + this memo's PROCEED_WITH_REVISIONS verdict). Refuses repo state where `tac.codex_loop_coordination.detect_collisions()` returns active-claim rows whose `files_to_claim` overlap AND whose `loop_id` are distinct (cross-loop collision). Bug class anchor: with 5 autonomous Codex loops sharing the repo working tree, two loops simultaneously claiming the same file would silently produce commit-swap absorption per Catalog #314 OR working-tree corruption per Catalog #302. The 4-layer pattern (mirrors Catalog #245 + Catalog #331): canonical helper `src/tac/codex_loop_coordination.py` + CLI `tools/codex_loop_coordination.py` + STRICT gate (this catalog #) + operator-briefing wire-in. Acceptance: same-line `# LOOP_COORDINATION_COLLISION_OK:<rationale>` waiver (placeholder rejected). Sister of Catalog #245 (Modal call_id ledger; same 4-layer pattern) + Catalog #302 (sister subagent scope overlap; per-subagent surface) + Catalog #314 (bare commit absorption; per-commit surface) + Catalog #131 (no bare writes to shared state — COORDINATION_PATH registered) + Catalog #138 (strict-load) + Catalog #110/#113 (APPEND-ONLY HISTORICAL_PROVENANCE) + Catalog #331 (Codex→Claude inbox channel). Together they extinct the multi-loop concurrent execution collision class structurally at SIX surfaces: edit-time-checkpoint (#302) + edit-time-bulk-op (#230) + commit-time-pre-pre-lock (#157) + commit-time-staged (#216) + commit-time-lock-arbitration (#117 + #174) + loop-time-claim-overlap (#332 — this gate). WARN-ONLY at landing per CLAUDE.md "Strict-flip atomicity rule" — live count at landing: 0 (coordination surface brand-new, no claims yet). Strict-flip after one-week window with 5+ active claims across 2+ distinct loops + no false-positive collision flags. 30+ dedicated tests in `src/tac/tests/test_codex_loop_coordination.py`. Memory: `feedback_multi_loop_codex_goal_design_landed_20260518.md` (when memo lands per op-routable #5). Lane: `lane_multi_loop_codex_goal_design_20260518` L1.
```

---

— Main-Claude 2026-05-18, T2 sextet pact + grand council Karpathy/Carmack/Hassabis/Tao, PROCEED_WITH_REVISIONS with binding stipulations honored per Contrarian + Carmack dissent.
