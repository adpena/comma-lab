# Codex adversarial review rounds 7 + 8 — convergent findings + next-fix-subagent directive

<!-- generated_at: 2026-05-09T12:00:00Z, from_state_hash: codex_round7_bqr3j2pe4_round8_bf3vpwd94 -->

## Summary

Rounds 7 (`bqr3j2pe4`) and 8 (`bf3vpwd94`) both completed with verdict `needs-attention`. They converge on the SAME META class as rounds 1-6: **custody/concurrency/fail-open in dispatch + state-write paths**. The round-6 fix (in flight as `ae71ee10`) addresses round-6 findings; rounds 7+8 surface the NEXT layer.

This is the recursive "each round closes the next layer" pattern documented in `feedback_codex_round6_findings_fix_with_self_protection_landed_20260509.md`.

## Findings

### HIGH 1 (BOTH rounds) — Lightning ambiguous-submit-failure orphan window

**File**: `experiments/arch_shrink_x0.4_lightning_full.py:1026-1058` + `experiments/lossy_coarsening_lightning_cuda_test.py:1592-1639`

**Bug**: After writing the pending row, the dispatcher wraps the entire `submit_lightning_job(...)` call in `except BaseException` and unconditionally calls `cancel_pending_job_locked`. That assumes every exception happens BEFORE billing. But `submit_lightning_job` includes the actual `Job.run(...)` network call. A timeout, SDK exception after server-side creation, property-access failure, or KeyboardInterrupt during/after `Job.run` can leave a real paid Lightning job while this code deletes the pending row — making the job invisible to the harvester.

This is the round-6 #143 fix (`check_paid_job_register_before_submit`) recurring at the NEXT layer: the pending row is now created, but the cancel logic is too aggressive.

**Recommendation**:
- Do NOT cancel the pending row for exceptions that cross the `Job.run` boundary
- Split pre-network setup/import validation from the paid call
- Catch ONLY known pre-network failures for cancel (e.g., file-not-found, import error, schema validation)
- For ambiguous exceptions: leave the pending row OR mark it `failed_unknown_billing` / `submit_status_unknown` with recovery instructions
- Strict preflight check candidate: `check_lightning_submit_cancel_only_before_network` — refuses any `cancel_pending_job_locked` call inside an `except BaseException` block that wraps the `Job.run` window

### HIGH 2 (round 7) — Vast.ai register_instance silently wipes corrupt tracker

**File**: `scripts/launch_lane_on_vastai.py:812-821`

**Bug**: The changed path now calls `tac.vastai_tracker.register_instance` after `create_instance`. That helper's loader returns `[]` for malformed tracker JSON and then writes the new single record → corrupt `.omx/state/vastai_active_instances.json` no longer fails closed; it drops every previously tracked active instance.

This is `check_state_writers_strict_load_for_mutating_path` (#138) recurring on a NEW writer that wasn't covered by the strict-load enforcement.

**Recommendation**:
- Validate the Vast.ai tracker with strict-locked-load BEFORE creating the instance
- Make `register_instance` refuse / quarantine corrupt tracker state instead of treating it as empty
- Mirror the canonical pattern from `tac.deploy.lightning.active_jobs_state.load_active_jobs_strict`
- New canonical helper: `tac.vastai_tracker.load_active_instances_strict` raising `VastaiTrackerCorruptError` on JSON parse failure
- Apply Catalog #138 to `tac.vastai_tracker` (or extend its scope to cover the helper)

### HIGH 2 (round 8) — Phase B gate trusts mutable local memory state

**File**: `src/tac/lane_12_v2_nerv_as_renderer.py:903-1019`

**Bug**: `phase_b_preconditions_status()` now defaults to `consult_session_state=True` and derives MET/PENDING from files under a hard-coded `~/.claude` memory directory. That makes a reusable tac gate:
- Non-hermetic: depends on operator's local file system, not committed repo state
- Machine-dependent: same checkout produces different results on different machines
- Spoofable: matching memo filename OR `operator_phase_b_authorization=true` token in any `feedback_*.md` file passes the gate
- Already reports several formerly pending gates as MET on this checkout (because operator's local memos exist)

**Recommendation**:
- Default `consult_session_state=False`
- Make status deterministic from committed repo state (lane registry + dispatch claim ledger)
- If operator authorization is needed, require an explicit `--session-state-path <committed_path>` flag pointing to a committed dated operator directive
- Do NOT scan arbitrary local feedback memos for authorization tokens in the default dispatch gate
- Strict preflight check candidate: `check_no_local_memory_consultation_in_dispatch_gates` — refuses `Path.home() / ".claude"` resolution inside any function whose name matches `*_preconditions*` / `*_dispatch_gate*` / `*_status*`

### MEDIUM (round 8) — #131 false-green for lowercase shared-state bindings

**File**: `src/tac/preflight.py:29170-29305`

**Bug**: `check_no_bare_writes_to_shared_state` (Catalog #131) collects shared path variables matching an all-caps assignment regex. A lowercase binding like `state_path = Path('.omx/state/foo.json')` followed by `state_path.write_text(...)` is NOT recognized → false-green. Separately, ANY lock token in the previous 20 lines waives a direct `write_text/write_bytes` path, even without:
- Reload-inside-lock
- Unique tmp file
- fsync
- `os.replace`

That leaves a false-green path for the same lost-update / partial-read class the catalog is meant to close.

**Recommendation**:
- Use AST-based path binding for lower-case names, attributes (`self.foo_path`), and Path joins (`base / "active.json"`)
- Require either an approved canonical helper OR explicit transactional pattern (reload-inside-lock + unique tmp + fsync + os.replace)
- Add negative tests for lowercase shared-state variables AND for locked-direct-write_text-without-atomic-replace
- This is META-meta of #131 (which itself was META-meta of #128 — the recursion is structural, not accidental)

## Sequencing for the next-fix subagent

**Updated 2026-05-09 post-landing**: BOTH `ae71ee10` (round-6 fix) AND `a6535b1ed` (Check #125 backfill + 6 production-hardening surfaces) are now COMPLETE. Working tree on `src/tac/preflight.py` + `lane_12_v2_nerv_as_renderer.py` is settled.

**Remaining in flight**: `a3daf91` (5 beyond-Phase-4 modules — NEW files only, unlikely to conflict with the round 7+8 fix surfaces).

## NEW CONFLICT surfaced post-a6535b1ed (2026-05-09)

`a6535b1ed` intentionally LANDED `phase_b_preconditions_status(consult_session_state=True)` as default + 4 helper `_check_*` functions that consult `~/.claude` memory state. Their rationale: "consults real session state; 3 of 4 PENDING flags flip to MET (only `operator_phase_b_authorization` remains PENDING — correct)." This DIRECTLY CONTRADICTS round 8 codex HIGH 2:

> "Phase B gate now trusts mutable local memory state... non-hermetic, machine-dependent, operator authorization accepted by matching memo filename or `operator_phase_b_authorization=true` token in any feedback_*.md is stale/spoofable."

**Both views have merit**:
- a6535b1ed: consulting session state is more useful for operator workflow + makes the gate "real" (otherwise it's a permanently-PENDING placeholder)
- Round 8 codex: consulting session state makes the gate non-hermetic + spoofable; should be deterministic from committed repo state

**Recommendation for next-fix subagent**: SURFACE this as an operator design decision, do NOT silently override a6535b1ed's choice. Per CLAUDE.md "Design decisions — non-negotiable": this is a council-grade design tradeoff, not a clear bug. Possible reconciliations:
- Keep `consult_session_state=True` default + add hermetic fallback when run in non-interactive contexts (CI, fork, codex sandbox)
- Make `operator_phase_b_authorization` token require committed dated directive in `.omx/research/operator_authorizations/` (committed audit trail)
- Add explicit `--hermetic-only` CLI flag for fork-context callers

Once `a3daf91` lands (or is determined non-conflicting), spawn the next-fix subagent with this conflict surfaced + recommend AskUserQuestion before applying any change to a6535b1ed's intentional design.

## Original sequencing (kept for context):

1. Spawn `codex_round_7_8_findings_fix` subagent
2. Each finding lands its own STRICT preflight check per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
3. Catalog claim machinery (atomic via `tools/claim_catalog_number.py claim`) for #146/#147/#148/#149 (or whatever next-available is)
4. Commits via `tools/subagent_commit_serializer.py`
5. 3-clean-pass adversarial greenup
6. Strict-flip atomicity per CLAUDE.md (same commit-batch as fix)

## Per-finding catalog # plan

- HIGH 1 → `check_lightning_submit_cancel_only_before_network` — refuses `cancel_pending_job_locked` inside `except BaseException` block wrapping `Job.run` window
- HIGH 2 (round 7) → extend Catalog #138's scope to cover `tac.vastai_tracker` OR new `check_vastai_tracker_strict_load`
- HIGH 2 (round 8) → `check_no_local_memory_consultation_in_dispatch_gates` — refuses `Path.home()/.claude` resolution in `*_preconditions*` / `*_dispatch_gate*` functions
- MEDIUM → harden Catalog #131 in-place (no new catalog #) — AST-based path binding + transactional-pattern requirement

## Cross-references

- `feedback_codex_round5_findings_fix_with_self_protection_landed_20260509.md` (round 5 fix)
- `feedback_codex_round6_findings_fix_with_self_protection_landed_20260509.md` (round 6 landing)
- `project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md` (operator NOT YET items — none of these findings change the GPU-spend gates)
- META-meta convergence pattern: each round closes the layer the previous round opened
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable

## Status

- Round 7 verdict captured: `bqr3j2pe4.output`
- Round 8 verdict captured: `bf3vpwd94.output`
- ae71ee10 (round-6 fix) STILL IN FLIGHT — must complete before next-fix-subagent spawn
- No GPU spend triggered; no operator decision required at this layer (META-class fixes are $0 dev work)
