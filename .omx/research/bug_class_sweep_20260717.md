# Bug-class + meta-bug sweep — 2026-07-17

**Operator P0 (verbatim):** "Fix bug classes and meta bugs everywhere."
**Discipline:** two-landing per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
(fix + preflight gate) + the 6-7× spread rule (a class fixed at one surface is live at ~6 others).
**Constraints honored:** trainer FROZEN (not touched); control-plane files (system_memory_governor,
memory_guard, safe_run) NOT edited — only READ; serializer commits with post-edit shas; apparatus
commits tagged `[no-triality]`; 1-thread measurement; no launches. Pointer 0.19110 UNMOVED (apparatus).

## STORES CONSULTED
- Memories: `admission_gate_naive_counts_reclaimable_as_committed_20260716`,
  `tunnel_always_up_supervisor_canonical_20260717`, `launcher_buffered_log_not_hung_orphan_spawn...20260715`,
  `bench_lever_contaminates_measured_quantity_ckpt_every_confound_20260717` (POISON-5).
- Code: `src/tac/confound_gates.py` (L1/L2/L3 immune-system pattern), `tools/system_memory_governor.py`
  (canonical reclaimable-aware `read_system_memory_snapshot`), `tools/spawn_durable_daemon.py` (#406
  observer-flag exclusion), `tools/triality_drift_detector.py:847+` + `tools/dashboard_fm_events.py`
  (fmtools ADVISORY subprocess pattern).
- Ledger: `.omx/state/harness_failure_ledger.jsonl` (recurrence families).

## Commits
- `a21e7d2b59` — CLASS 1 + CLASS 2 (fixes + 2 gates + tests).
- `98b85c6063` — CLASS 3 + CLASS 4 (fixes + tests + digest wire-in).

---

## CLASS 1 — reclaimable-memory-as-committed (safety basis)
**Root:** on macOS `psutil.virtual_memory().available` = free+inactive counts DIRTY ANON in the
inactive queue as available; `.used` counts reclaimable file-cache as committed. MEASURED live
2026-07-17 (76-GiB trainer running): `.available` = 57.3 GiB but truly reclaimable-without-swap =
13.7 GiB → self-abort guards UNDER-protected the live trainer by ~43 GiB.
**Fix:** new canonical helper `tools/mem_basis.py` (`conservative_free_gib` / `true_committed_gib`)
reuses the governor's validated kernel-queue decomposition (`read_system_memory_snapshot`), graceful
fallback governor→psutil→default. All safety guards routed through it.
**Gate:** `check_no_raw_virtual_memory_safety_basis` (confound_gates, WARN-ONLY, live 0) — refuses raw
`virtual_memory().{available,used,free}` outside mem_basis/governor; `.total` (denominator) excluded;
same-line `# RAW_VM_BASIS_OK:<rationale>` waiver.

| file:line | usage | verdict |
|---|---|---|
| tools/measure_contour_string_flip_coding.py:558 | self-abort MEM-GUARD rc=7 | FIXED (routed) |
| tools/apply_perclass_bitalloc_witness.py:_mem_guard | abort rc=7 | FIXED (routed) |
| tools/apply_sensitivity_bitalloc_witness.py:_mem_guard | abort rc=7 | FIXED (routed) |
| tools/quadratic_basin_finisher_probe.py:ram_floor_ok | floor check | FIXED (routed) |
| tools/c2_witness_own_decomp.py:_free_gib | floor→REFUSE | FIXED (routed) |
| tools/d1_gpu_verdict_agreement_probe_n600.py:_free_gib | floor→abort | FIXED (routed) |
| tools/warp_vs_noise_flip_probe_n600.py:_free_gib | floor→REFUSE | FIXED (routed) |
| tools/dash_comb_probe_n600.py:_free_gib | floor→REFUSE | FIXED (routed) |
| tools/hard_frame_mechanism_atlas.py:_free_gib | floor→REFUSE | FIXED (routed) |
| tools/witness_applypass_batch.py:_free_gib | conservative MIN | FIXED (folded into MIN) |
| tools/watch_and_harvest_b1_checkpoint.py:free_memory_gb | min_free floor | FIXED (routed; psutil kept as waivered last-resort fallback) |
| src/tac/canonical_equations/verdict_parallel_workers_speedup...:_worker sizing | worker-count budget (DANGEROUS dir: over-trust→spawn too many→OOM) | FIXED (routed) |
| tools/run_annulus_live_monitor_guarded.sh | watchdog floor (shell inline python) | FIXED (routed) |
| tools/witness_memory_preflight.py:190 | `.total` denominator | ANNOTATED (gate excludes .total) |
| tools/witness_checkin.py:177 | status field | WAIVERED (telemetry) |
| tools/dashboard_flow_sequence.py:81 | dashboard display | WAIVERED (telemetry) |
| tools/dashboard_server.py:2306 | dashboard display | WAIVERED (telemetry) |
| src/tac/witness_control/perclass_verdict.py:87 | verdict telemetry field | WAIVERED (telemetry) |
| tools/spawn_durable_daemon.py (677) | admission | ALREADY-FIXED (uses governor `available_reclaimable_gib`) |
| tools/system_memory_governor.py | canonical accounting | CONTROL-PLANE (source of truth, not edited) |

**Surfaces:** 13 guards fixed · 5 telemetry waivered · 1 `.total` annotated · 2 already-correct
(spawn_durable_daemon, governor). Gate live-count 0.

## CLASS 2 — fail-closed guard false-positive on observers (token-in-joined-argv)
**Root:** guards classify a live process by trainer-token presence in its joined cmdline; observers
carry the trainer NAME as a flag VALUE (`--training-sig train_levelset_witness`) → misclassified.
**Fix:** new `tools/argv_role.py` (`strip_observer_flag_values` / `is_observer_stripped_launch`).
The p0_512 same-outdir spawn guard (`launch_witness_run.py`) now classifies on the observer-stripped
cmdline (mirrors the #406 fix), so a `--training-sig` monitor referencing the out_dir no longer
false-refuses a launch.
**Gate:** `check_process_guard_excludes_observer_flag_values` (confound_gates, WARN-ONLY, live 0) —
function-scoped: flags a guard that classifies by trainer-token + makes a kill/refuse decision without
an observer-flag exclusion; `# OBSERVER_ROLE_OK:<rationale>` waiver.

| surface | verdict |
|---|---|
| launch_witness_run.py p0_512 same-outdir spawn guard | FIXED (observer-stripped) |
| spawn_durable_daemon.py `_witness_dsl_compile_hash_gate` (#406) | ALREADY-FIXED (inline --training-sig exclusion) |
| dashboard_supervisor.py `select_kill_pids` | ALREADY-HARDENED (classifies by kind, excludes TRAINING_SIG + self group) |
| memory_guard.py kill-selector | CONTROL-PLANE, already custody-gated (positive allowlist + pgid-leader + denylist); NOT edited |
| system_memory_governor.py OUR_JOBS_PATTERN | CONTROL-PLANE; phantom count-only, charged-zero (benign, noted in admission memo); NOT edited |
| witness_checkin.py find_trainer_procs | TELEMETRY (no kill/refuse); anchored on python argv[0] + full module token; left |
| witness_chain_watchdog.py | ALREADY-ROBUST (pid-reuse cross-check vs registered argv tokens; verdict-only) |

**FM note:** structural discrimination is the authority; an fmtools role-classification would be a
tiebreaker/telemetry annotation only (not wired — structural check sufficed at every live surface).

## CLASS 3 — stop-hook matcher false-demands (already-registered P0s)
**Root:** `operator_p0_stop_hook.py` demanded re-registration of designations without checking
existing ledger rows' `verbatim_ask`. Fired `new_designations=1` at 16:01 today on P0s already in the
ledger.
**Fix:** `designation_already_covered()` — deterministic token-containment is the AUTHORITY
(≥0.60 containment ⇒ covered); fmtools advisory `_fm_covered` corroborates ONLY in the gray band
[0.40,0.60) with a nonzero deterministic overlap (NEVER sole gate); fail-closed (<3 meaningful
tokens or uncertain ⇒ still demand). Wired into check-B: covered designations are dropped before the
demand. **FM pattern per operator guidance** (subprocess under fmtools venv, graceful-degrade, advisory).
**Tests:** today's two real false positives (v10 capstone, bug-class-sweep) as fixtures + FM-never-sole.

## CLASS 4 — zero-work arm death detection (detect-only)
**Root:** a SPEC_v10 arm died at ~15 tokens/2 tool-uses having done NO work; only human-visible.
**Fix:** `tools/subagent_liveness.py::stale_zero_work_arms()` — flags arms that registered (step 0)
then went silent 20m–24h without advancing (still in_progress). Surfaced in `costate_digest` (a
digest line). Pure read; writes/kills nothing. Sister of witness_chain_watchdog at the subagent surface.

## CLASS 5 — bounded fresh-eyes hunt (≤25% effort) — HONEST NEGATIVE
Hunted `.omx/state/harness_failure_ledger.jsonl` recurrences + the `pgrep`/`ps|grep|head`
liveness universe (POISON-5 + `launcher_buffered_log...` memories: declare-dead ONLY via
`/usr/bin/grep` + abs paths + FILE mtimes + psutil, never buffered-log/`ps|grep|head`).
**Finding:** the phantom-death / silent-hang liveness family recurs (2× `codex_arm_silent_hang`,
1× `phantom_death_buffered_log_plus_misfired_grep_liveness`) BUT every LIVE surface is already
robust: `witness_chain_watchdog` uses file-mtime + psutil descendants + pid-reuse cross-check;
`codex_status`/`codex_delegate` use unique-token `pgrep -f {label}_{stamp}` (not the forbidden
`pgrep -fl | head`). The only residual forbidden `ps aux | grep … | head` instances are STALE legacy
fleet-monitor scripts (`training_watchdog.sh`, `status_updater.sh`, `FleetMonitor*.swift`) that
reference RETIRED trainers (`train_postfilter`/`segnet_boundary`) — zero live-path (`train_levelset`)
exposure. **Verdict:** no NEW live ≥2-surface class warranting a two-landing under this sweep.
**Related open item (not mine to close here):** ledger `owed_fix` "codex_status.py + delegate notifier:
flag RUNNING codex arms whose log mtime is stale" — the codex-arm analogue of CLASS 4; my
`subagent_liveness` generalizes the pattern (registered-then-silent) but the codex delegate surface is
a distinct liveness store; left as a scoped follow-on.

## Residuals / honest limits
- CLASS 1/2 gates are WARN-ONLY (live-count 0) per the Strict-flip atomicity rule; strict-flip is a
  follow-on once they've ridden a few sessions clean.
- Control-plane guards (memory_guard kill-selector, governor OUR_JOBS_PATTERN) were READ and judged
  already-hardened; NOT edited per the review discipline.
- Stale fleet-monitor scripts referencing retired trainers left as-is (low value; would need rewrite
  to the current run model, not a bug on the live path).
