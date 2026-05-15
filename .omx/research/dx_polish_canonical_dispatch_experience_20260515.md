# DX polish: canonical dispatch experience (2026-05-15)

Cumulative DX guide for operator + future agents. Records the polished state
of the dispatch / harvest / council-review loop after the 2026-05-15
DX-WAVE landing. Per CLAUDE.md "Beauty, simplicity, and developer experience
— non-negotiable" this doc is the single read-once reference for the
operator-facing surfaces of these flows.

This file lives in `.omx/research/` (durable analysis ledger per CLAUDE.md
`tac` stays clean section). It is **NOT** an authority on policy — CLAUDE.md
non-negotiables remain the binding contract; this doc records the canonical
HOW of using them.

---

## 1. Modal `.spawn()` survives long-connection death

**Operator correction 2026-05-15** (verbatim): *"it's just your long
connection that died not the modal smoke itself, it probably ran"*.

`experiments/modal_train_lane.py` uses Modal's `.spawn()` exclusively. The
local-side dispatcher writes
`experiments/results/lane_<label>_modal/modal_metadata.json` (with `call_id`)
and exits. The Modal worker is launched server-side and persists for the
duration of the dispatch regardless of local-side process state.

**What this means for foreground SSH / shell sessions**:

- Local `subprocess.run([...modal_dispatch...])` may DIE (SIGURG, network
  drop, terminal close, broken pipe) during the ~30-90s mount/setup window.
- The Modal worker on the other side **keeps running**. Its result lands in
  the FunctionCall return-value cache (~24h TTL).
- Conclusion: **never assume "local process died" == "Modal smoke failed"**.
  Always run `tools/harvest_modal_calls.py --execute` (or the cron wrapper)
  to recover the real verdict.

The earlier `feedback_harness_sigurg_kills_subagent_modal_dispatch_permanent_20260514.md`
memo treated SIGURG as the failure mode; the operator's correction
SUPERSEDES that interpretation: the local-side death is a transport / shell
artifact, not a dispatch artifact.

**Always cross-check** with:

```bash
.venv/bin/python tools/check_modal_harvest_freshness.py
.venv/bin/python tools/harvest_modal_calls.py --execute
```

---

## 2. Modal harvest cadence (every 4h max for active dispatch periods)

Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE — NON-NEGOTIABLE, HIGHEST
EMPHASIS": every dispatch via `experiments/modal_train_lane.py` MUST be
followed by a scheduled harvest within 24h.

**Recommended cadence**: every 4 hours during active dispatch periods. The
24h cache TTL gives some buffer, but a 4h cadence catches OOM / rc=137 /
timeout dispatches early enough to re-route compute the same session.

### Manual freshness check

```bash
.venv/bin/python tools/check_modal_harvest_freshness.py
```

Exit code 0 = fresh; 1 = STALE (last harvest > 4h ago AND ≥1 unharvested
`modal_metadata.json` exists; printed banner includes the recovery command).

JSON variant for shell prompt / dashboard integration:

```bash
.venv/bin/python tools/check_modal_harvest_freshness.py --json
```

### Operator-installable cron

The optional template at `scripts/cron_harvest_modal.sh` wraps
`tools/harvest_modal_calls.py --execute` for crontab installation:

```cron
0 */4 * * * cd /Users/<you>/Projects/pact && ./scripts/cron_harvest_modal.sh >> /dev/null 2>&1
```

Logs to `.omx/state/_modal_harvest_cron.log` (start/end UTC + freshness
recheck after harvest). Dry-run with `./scripts/cron_harvest_modal.sh
--dry-run`.

**This is template-only**. The operator decides whether to install — there
is no auto-install. The 2026-05-15 baseline session showed an 8h gap with
5 unharvested dispatches at risk; the cron fixes the recurrence.

---

## 3. Pending council decisions viewer

**Source of truth**: `.omx/state/pending_council_design_decisions.jsonl`
(per the standing referral at
`.omx/research/all_design_decisions_through_grand_council_directive_20260514.md`).

Before the 2026-05-15 wave, operators had to grep raw JSONL to inspect the
queue. The new canonical viewer pretty-prints the queue grouped by
`council_priority`:

```bash
# Default: pending_council only, grouped HIGH/MEDIUM/LOW/UNKNOWN
.venv/bin/python tools/view_pending_council_decisions.py

# Show every row regardless of status
.venv/bin/python tools/view_pending_council_decisions.py --status all

# Show only resolved rows (audit trail)
.venv/bin/python tools/view_pending_council_decisions.py --status resolved

# Filter by decision_id substring
.venv/bin/python tools/view_pending_council_decisions.py --decision-id oss_

# Machine-readable JSON for downstream tooling
.venv/bin/python tools/view_pending_council_decisions.py --json
```

Per row, the viewer surfaces: decision_id, title, status, source lane /
subagent, cost, blocking dispatch (if any), queued UTC, options list,
resolution body + adjudicator (when resolved).

**This is a viewer, not an arbitrator**. Council deliberation still goes
through GRAND-COUNCIL-OMNIBUS-DESIGN-REVIEW or ad-hoc council per the
standing referral — the viewer just makes the queue visible.

---

## 4. Catalog #166 dirty-tree categorization heuristic

Catalog #166 (`check_modal_dispatch_verifies_worker_source_matches_head`)
records dispatch-time HEAD SHA + working-tree dirty summary into
`modal_metadata.json` and writes a worker-side
`modal_worker_head_ledger.json` so post-mortem can distinguish:

- **H1**: image cache stale (worker mounted older code)
- **H2**: eval-timing race (operator dispatched BEFORE fix landed)
- **H3**: deploy stale (Modal app redeploy missed)
- **H4**: manifest gap (mount missed required path)
- **H5**: PYTHONPATH ordering (canonical path shadowed)
- **HOK**: parity (worker ran exactly the dispatched HEAD)

**Operator categorization tip when triaging a failed Modal smoke**:

1. Read `modal_metadata.json::mounted_code_git_head`.
2. Read `working_tree_dirty=true` flag and `working_tree_dirty_summary`.
3. Compare against `git log --oneline` for that window — was the fix
   landed BEFORE or AFTER the dispatch UTC?
4. If AFTER: the failure is your local source state at dispatch time
   (H2). The worker did exactly what it was told.
5. If BEFORE: read `modal_worker_head_ledger.json::worker_sentinel_sha256`
   to compare against `sentinel_files_local_sha256` for H1/H3/H4/H5
   classification.

The runtime probe `tools/diagnose_modal_worker_source_staleness.py` is the
fastest one-shot diagnosis.

---

## 5. Operator-direct `!` prefix vs subagent fire — corrected understanding

When the operator types a message that includes `!<command>`, the harness
runs that command **synchronously in the operator's foreground shell** —
**not** in a subagent context, **not** in the harness Bash tool. Conclusion
from the 2026-05-14 SIGURG investigation revised in light of the
operator's 2026-05-15 correction:

- `!` is the operator-direct path. It DOES inherit the operator's terminal
  process group; if the terminal closes / SSH drops / SIGURG fires, the
  operator-direct `!` invocation is at risk same as any other foreground
  command.
- Subagent-fired commands run inside the harness Bash tool, which is
  isolated from the operator's terminal. Subagent dispatches survive the
  operator's terminal lifecycle by construction.

**Modal `.spawn()` decouples both from the dispatch lifecycle**. The
worker keeps running regardless of which path fired the dispatch. The
result lands in the FunctionCall cache. Harvest catches it.

---

## Cross-references

- CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable
- CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable
- CLAUDE.md "Subagent coherence-by-default" non-negotiable
- `.omx/research/all_design_decisions_through_grand_council_directive_20260514.md`
- `feedback_harness_sigurg_kills_subagent_modal_dispatch_permanent_20260514.md`
  (SUPERSEDED by the 2026-05-15 operator correction in section 1)
- Catalog #166 `check_modal_dispatch_verifies_worker_source_matches_head`
- Catalog #167 `check_substrate_dispatch_uses_smoke_before_full_pattern`
