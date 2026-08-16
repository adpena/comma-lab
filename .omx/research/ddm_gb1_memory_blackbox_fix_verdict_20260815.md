# VERDICT — ddm_gb1: memory_blackbox governor two-landing fix (#1073, P0 APPARATUS)

**Charter:** `.omx/research/ddm_gb1_memory_blackbox_two_landing_fix_charter_20260815.md`
**Baseline pin:** git `0002232fc2` (main at charter time). **Landed against:** the same tools.
**Daemon status: STILL OFF.** This arm did not start, restart, or signal it. MAIN adjudicates re-enable.
**Nothing live was signalled.** Every fixture is a synthetic process table / snapshot / tmp ledger;
the one non-synthetic test spawns its OWN child, the child stops ITSELF, and the test only ever
sends SIGCONT (plus a SIGKILL cleanup) to that child.

## Headline

All FIVE defects are fixed and each carries a MEASURED before/after. The second landing —
`check_throttle_rearms_and_admission_reconciles` — fires **5** violations against the pre-fix tree
and **0** against the fixed tree (detector zeroes on the cure), and ships with two EXECUTED positive
controls per the #831 ratchet. 48 new tests + 234 adjacent tests green.

This is APPARATUS, not goal progress. The exact pointer is unmoved by this landing and nothing here
claims otherwise. What it buys: three live measurements can no longer be frozen silently, and a
READY_TO_FIRE launch can no longer be refused by memory that is not being used.

## Per-defect evidence

### D1 — the cp measurement counted the wrong object

`measured_control_plane_rss_gib` returned `used_gib - tracked_current_gib` = the box's TOTAL used
memory including file cache. Receipt (`.omx/state/memory_blackbox.daemon.log`): *"SAFETY-FLOOR
CLAMP: measured_cp value 92.00 GiB clamped to 64.00 GiB"* at ~1.5 Hz on a machine with 40.5 GiB free.

**Fix.** Split into two honestly-named functions:

- `non_workload_used_gib(used_gib, tracked_current_gib)` — the OLD arithmetic, which is a real and
  useful quantity: the adaptive ceiling's BASELINE (OS + cache + control plane). Unchanged behaviour.
- `measured_control_plane_rss_gib(samples=None)` — sums RSS over the NAMED control-plane processes
  only, enumerated by the same identity gates the throttle uses to refuse touching them
  (`is_host_control_plane_process` + `_matches_extra_protected` + `is_protection_infra_cmd`).
  Returns `None` when unmeasurable (no `memory_guard`, or an empty LIVE scan = a failed scan), which
  makes `derive_safety_floor` fall back to its STATIC policy leg rather than to a fabricated zero.

**MEASURED, live on this box (read-only, 2026-08-15):**

| quantity | old object | new object |
|---|---|---|
| measured_cp | 35.92 GiB | **20.60 GiB** |
| derived floor | 42.32 GiB | **27.00 GiB** |

The 20.60 is honest, not small: one real `claude` process holds 18.81 GiB of it (26 named
control-plane processes total). The floor now FOLLOWS the control plane, which was always the
intent; it just was not measuring the control plane.

**Regression test** (`test_d1_wrong_object_measurement_is_what_clamped_the_floor`) reproduces the
receipt exactly: a synthetic table summing to 92.0 GiB, of which 86.8 is workload. Old object →
floor clamped to 64.0. New object → 5.2 + 6.4 = 11.6, unclamped.

### D2 — the throttle could not re-arm

`decide_governor_action` resumed on `level == "normal"` ALONE, and `classify_pressure` returns
"warn" whenever the macOS pressure level reads ≥ 2 — a STICKY signal. Receipt: *"GOVERNOR ALERT
avail=40.4GiB level=warn"* while five jobs sat in state `T` for 75+ minutes. Spike-guard
median-freeze genus (#304): a guard whose reference never re-arms.

**Fix — three rungs, all derived, none new constants:**

1. **Escape hatch (rung 0, outranks everything).** Any job SIGSTOPped ≥ `max_stop_duration_s`
   resumes unconditionally. `DEFAULT_MAX_STOP_DURATION_S = 10 x DEFAULT_BAND_INTERVAL_S = 300 s`:
   ten consecutive band evaluations with no recovery is not a transient spike, it is a throttle that
   is not working. 15x below the measured freeze. Env-tunable via `TAC_GOV_MAX_STOP_DURATION_S`;
   a `<= 0` value is LOUDLY refused — the hatch may be retuned, never switched off.
2. **Derived-free resume.** `derived_resume_free_gib(total) = derived_warn_free_gib + cp_headroom`
   → 21.2 GiB @128, 5.0 @8. Paused jobs resume when free ≥ that, even while the OS level reads warn.
   `level == "normal"` remains sufficient; it is no longer NECESSARY.
3. **Derived-free no-pause veto.** No pause fires while free ≥ the same threshold, so a sticky OS
   level cannot actuate the throttle in EITHER direction. Without this the two rungs alternate at
   the sample rate — a flapping SIGSTOP is not protection either. The `[warn, resume]` band is a
   one-`cp_headroom` hysteresis dead zone.

The actuation basis is `governing_free_gib(snapshot)` = reclaimable-aware available when the
accounting validated, else legacy. macOS `available` over-reports (dirty anon in the inactive
queue), so the conservative number makes the veto fire LESS and the resume fire LATER.

**Evidence.** `test_d2_sticky_os_warn_with_high_free_RESUMES_the_frozen_jobs` replays the incident
(level=warn at 40.4 GiB, two jobs paused) and asserts resume, highest-priority first.
`test_d2_genuine_pressure_still_pauses` and `test_d2_governing_free_uses_the_conservative_reclaimable_basis`
prove the guard is not disarmed. `test_d2_old_callsites_without_rearm_inputs_behave_exactly_as_before`
proves backward compatibility — the three re-arm inputs default to `None` and their rungs are skipped.

### D3 — daemon death stranded its victims

A SIGSTOPped process cannot resume itself, and the daemon had no exit path that resumed what it
stopped. MEASURED: the CONT sweep for five stranded jobs was reconstructed BY HAND from `ps`.

**Fix.** A persisted stopped-set ledger, `.omx/state/memory_blackbox_stopped_pids.json`
(fcntl-locked, tmp+rename atomic, schema `memory_blackbox_stopped_pids.v1`):

- pids are recorded **BEFORE** they are signalled, so a crash between the two leaves a sweepable
  ledger rather than a stranded pid;
- `atexit` + `SIGTERM`/`SIGINT` handlers SIGCONT the whole set, and `run_daemon`'s `finally` sweeps
  again (belt and braces);
- SIGKILL runs no handler **by definition** — which is exactly why the set is on DISK:
  `memory_blackbox.py --resume-stopped` is the recovery sweep, and daemon STARTUP also sweeps a
  predecessor's leftovers, so a restart is also a recovery;
- PID-recycling safe: a pid is only signalled when `ps` still reports state `T`. Every examined pid
  is dropped either way, so the sweep is idempotent;
- `_adopt_unrecorded_paused` starts the clock for victims of a PREVIOUS daemon, so the escape hatch
  bounds even holds it did not create. Late, but FINITE — silent-forever is the extincted mode.
- Ages are WALL clock (must survive a daemon restart; monotonic is per-process). A clock jump can
  only make a job look older and resume EARLIER — the safe direction, stated in the docstring.

**Evidence.** `test_d3_sweep_really_resumes_a_stopped_process` spawns its own child, waits for state
`T`, sweeps, and asserts the child left `T`. The rest are monkeypatched (`test_d3_sweep_conts_stopped_pids_and_skips_recycled_ones`
asserts every signal sent is SIGCONT and that a recycled pid in state `S` is skipped).

### D4 — safe_run's admission path did not reconcile

`witness_memory_preflight.system_aware_admission` (2026-07-09) and `spawn_durable_daemon._do_start`
(2026-07-11, under the registry lock) both converge the registry to ground truth before projecting.
`safe_run` did not. Every phantom `running` row charges `UNKNOWN_GROWTH_HEADROOM_GIB` = 25 GiB.
MEASURED: three dead rows (pids 7506, 8997 — the dead daemon itself — and 31881) summed to
"active-growth 100.0 GiB" and REFUSED a real relaunch twice; a manual reconcile admitted it at
projected 81.6 < ceiling 116.0.

**Fix.** `safe_run._system_admission_gate` calls `spawn_durable_daemon.reconcile_dead_daemons(verbose=False)`
fail-OPEN before the decision, mirroring `witness_memory_preflight.py:382-401` exactly. The decision
itself stays fail-CLOSED. Re-entrancy is safe: `_gate_and_reserve` already holds `_registry_lock()`
and that lock is re-entrant within a process (depth counter).

**Sister fix in the same batch** (CLAUDE.md sister-substrate audit): the governor's own
`--admit` CLI had the identical omission, so an operator diagnosing a refusal got the same phantom
numbers. It now reconciles too. That is why the gate's Leg B live count is 0 rather than 1.

**Evidence.** `test_d4_safe_run_reconciles_before_the_sum_over_ram_decision` asserts the CALL ORDER
`[reconcile_dead_daemons, live_admission_decision]`, not merely that both happen.
`test_d4_reconcile_leg_is_fail_open_and_the_decision_stays_fail_closed` raises inside the reconcile
and still gets rc=5 from a genuine refusal.

### D5 — INCIDENT #3: one launch tree charged three times (charter addendum, D1 family)

The wd3 W0_reset launch (READY_TO_FIRE, zero blockers) was refused with "active-growth 100.0 GiB" on
a box using 37.9 of 128 — and `reconcile_dead_daemons()` found ZERO stale rows. The growth was
charged to LIVE processes.

**I measured the live tree before the mp2 job exited** (read-only `ps`), and the diagnosis in the
charter is confirmed with one correction: the ppid chain was INTACT.

```
39740  ppid=1      pgid=39740  rss 0.01 GiB  launch_detached wrapper   proj 12.61  charged  6.56
39748  ppid=39740  pgid=39740  rss 0.07 GiB  contest_auth_eval         proj 31.04  charged 25.00
25923  ppid=39748  pgid=39740  rss 6.86 GiB  evaluate.py               proj 30.97  charged 25.00
group_rss_gb(39740) = 7.302 GiB   <- ONE tree, and it already contains all three
                                     SUM active growth = 81.56 GiB
```

**Root cause (measured, not inferred).** The `governed_descendant` mechanism already exists and
already does the right thing — but it recognised only ancestors in `owned_pids`, i.e. **registry**
rows. The mp2 tree's root (`launch_detached`) is ps-only, never registered, so `owned_pids = {25204}`
and no descendant could be recognised. The session-split conservatism then charged each session
leader +25 independently.

**D5a fix.** A candidate whose ancestor chain reaches ANOTHER CANDIDATE — registered or ps-only — is
a descendant of that launch tree. pid 0/1 excluded (init is every process's ancestor; if it counted,
one candidate under launchd would blind the gate to every other). The root keeps the charge, and its
`group_rss_gb` is descendant-inclusive, so nothing is under-counted. This also cures a ~3x
over-count in `tracked_current` (the ceiling BASELINE input), which the same triple-count corrupted.

**D5b fix — the #370 control-plane regression.** The dashboard row (pid 25204) carries no
`projected_peak_gib`, so it resolved to `0.22 + 25` = 25.22 → above `HEAVY_MIN_PROJECTED_GIB`, i.e.
a 0.22 GiB telemetry daemon counted as a HEAVY job reserving 25 GiB. That is precisely the case
`sum_active_growth_headroom_gib`'s own docstring says it excludes — the exclusion held only while
something recorded the 2.44 GiB projection the 2026-07-09 fix cites. **2.44 GiB IS `--rss-mb 2500`:
the number was always in the argv.** New `declared_rss_cap_gib(cmd)` reads the ENFORCED safe_run RSS
cap back as the projection when no explicit one was recorded. A cap is an upper bound the process
cannot exceed (safe_run kills at it), so this is a projection, not a guess. An explicit recorded
projection still outranks it.

**MEASURED, live on this box after the fix** (the mp2 tree had exited by then, so the tree leg is
proven on the captured fixture rather than live):

```
BEFORE:  dashboard proj=25.22 charged=25.00 | mp2 tree charged 6.56+25.00+25.00 | SUM = 81.56 GiB
AFTER :  dashboard proj= 2.44 charged= 0.00 | mp2 tree charged 25.00 (one root) | SUM = 25.00 GiB
```

`test_d5_incident3_refusal_arithmetic_is_reproduced_and_then_cured` replays the whole scene
(dashboard + tree) end to end and asserts the refused 20 GiB launch fits under the 116 GiB ceiling.
`test_d5a_an_independent_job_is_still_charged_independently` is the false-negative guard: aggregation
must not swallow a genuinely separate tree.

## Second landing — the gate, with an EXECUTED positive control

`check_throttle_rearms_and_admission_reconciles` in `src/tac/confound_gates.py`. ONE gate, TWO
anti-patterns (Catalog #299 consolidation discipline — not a pure-additive pair):

- **Leg A**: a function naming `resume_job`/`resume_targets` AND branching on a pressure class must
  also name `resume_free_gib` and `max_stop_duration`. Anchored on the throttle actuator, NOT on the
  bare word "resume" — checkpoint resume appears in 40+ files here, and a detector that matched it
  would be permanently red, which is how a gate ends up ignored (the #821 lesson). Comments AND the
  docstring are stripped before matching, so prose that NARRATES the cure cannot satisfy it.
- **Leg B**: a module calling `live_admission_decision` must also call `reconcile_dead_daemons`.
  Module granularity is correct because `spawn_durable_daemon` legitimately reconciles in the CALLER
  (`_do_start`, under the lock) rather than in its gate function.

**Executed before/after (the receipt):**

```
$ gate(repo_root=<git 0002232fc2 tools/>)          # PRE-FIX
  [throttle-rearm-and-admission-reconcile] WARN: 5 violation(s)
    tools/memory_blackbox.py:308           _govern_tick()            (Leg A)
    tools/system_memory_governor.py:1859   decide_governor_action()  (Leg A)
    tools/system_memory_governor.py:2392   main()                    (Leg A)
    tools/safe_run.py:273                  live_admission_decision   (Leg B)
    tools/system_memory_governor.py:2467   live_admission_decision   (Leg B)

$ gate(repo_root=<fixed repo>)                     # POST-FIX
  [throttle-rearm-and-admission-reconcile] OK (3 throttle-resume function(s) + 4 admission
  module(s) checked across 6474 in-scope source file(s))
```

**Cost, measured and fixed during review.** The first draft parsed all 6474 in-scope files and took
**18.86 s** — too slow to sit in `preflight_all` on every commit, and a gate people route around is
not protection. A cheap substring prefilter before `ast.parse` (the sister gates' pattern) took it
to **0.35 s** (54x) with the verdict unchanged in BOTH directions: still 0 on the fixed repo, still
5 on the pre-fix tree. `considered` (6474) is reported next to `scanned` (5) so the prefilter can
never hide a narrowed scan.

Registered in `CONFOUND_GATES` (28 → 29) and wired warn-only through `preflight_all`'s existing
`CONFOUND_GATES` loop. **Warn-only for one cycle** despite live count 0: both legs scan the
launcher/governor family that sibling arms are actively editing, and a strict gate that fires on
someone else's in-flight commit trains readers to bypass the suite. Strict-flip condition is named
in the docstring: one clean cycle at live count 0.

Two `PositiveControl` fixtures (one per leg) are registered and EXECUTED by the #831 class guard
`check_refusal_gates_have_live_positive_control`, which passes. Coverage 11 → 12 of 29;
`MAX_UNCOVERED_REFUSE_GATES` stays 17 (the gate lands covered, so the ceiling does not move).
`MIN_POSITIVE_CONTROL_COVERAGE` raised 9 → **12**, the MEASURED live value, not my own +1: the floor
had drifted three below actual, so three controls could have been deleted with the guard still
printing OK. A floor that lags the truth is not a ratchet.

## Tests

`src/tac/tests/test_ddm_gb1_memory_governor_rearm.py` — **49 tests, all green** (D1 ×5, D5 ×8,
D2 ×10, D3 ×11, D4 ×4, gate ×11). Adjacent suites re-run green: `test_memory_blackbox`,
`test_tier_scaled_safety_floor`, `test_system_memory_governor`, `test_safe_run_double_gate`,
`test_spawn_durable_daemon_memguard`, `test_witness_memory_preflight`, `test_memory_guard`,
`test_refusal_gate_positive_control_class_guard` — **337 passed**.

**Review pass 2 also caught and fixed** (each now tested): a duplicate source of truth for the 30 s
band cadence (the escape hatch derives from it — a second literal would let them decouple, so
`DEFAULT_BAND_INTERVAL_S` now lives in the governor and `memory_blackbox` binds the same object);
`resolve_max_stop_duration_s` being called per tick, which would have printed a malformed-override
warning at the sample rate (the exact log-spam shape the incident already had) — now resolved once
per daemon; two ledger reads per tick collapsed to one; `_adopt_unrecorded_paused` writing through
the LIVE lock path from tests; a malformed ledger key that could never be swept, making the file
grow without bound; and an unguarded `print` in `_log_action` reachable from the atexit sweep, where
a closed stdout would have aborted the SIGCONT sweep that is the whole point of the exit path.

Two existing tests asserted the OLD D1 semantics and were updated, not deleted:

- `test_tier_scaled_safety_floor.py::test_admission_growth_headroom_true_gib_matches_review_numbers`
  now calls `non_workload_used_gib` — same arithmetic, honest name.
- `test_memory_blackbox.py::test_sample_once_has_all_trust_and_ceiling_fields` now declares an
  8.0 GiB control plane as a process table instead of arriving at 8.0 via `used - tracked`. Every
  asserted number is unchanged; only the OBJECT the 8.0 describes is now the right one.

One adjacent guard was strengthened rather than worked around:
`test_tier_scaled_safety_floor.py::test_daemon_band_path_has_no_kill` scanned RAW source for
`SIGKILL`/`SIGTERM`/`kill(` and fired on a COMMENT that merely NAMED SIGKILL while explaining why
the ledger is persisted. It now strips comments and the docstring first. A comment cannot signal
anything — the same lesson the new gate implements.

## Honest state / what I did NOT do

- **The daemon is still OFF.** Not started, not restarted, not signalled. MAIN adjudicates re-enable.
  My recommendation: re-enable is now defensible (the freeze mode is structurally bounded at 300 s
  and the exit sweep exists), but the 2-`ps`-per-tick cost and the D5a aggregation deserve one
  supervised soak before it runs unattended.
- **`sample_once` now costs one extra `ps` per tick** (~40 ms per 2 s tick). `list_tracked_jobs`
  keeps its own live scan because that scan is also what records the RSS-growth history; reusing one
  table would have silently disabled the growth projections. Stated in the docstring, not hidden.
- **Five test failures on main are PRE-EXISTING and NOT mine.** A/B'd against the `0002232fc2` pin
  by running each gate at both roots: `check_no_raw_virtual_memory_safety_basis` 2 vs bound 0,
  `check_process_guard_excludes_observer_flag_values` 1 vs 0, `check_no_stub_lever_factories` 11 vs
  bound 10, `check_checkpoint_saves_do_not_silently_drop_optimizer_state` 1 vs 0 (also
  `test_ddm_op2_optimizer_state_persistence::test_gate_is_live_count_zero_against_the_real_repo`),
  and `check_levelset_hosc_requires_beta_end` 10 vs bound 9 (a sibling run dir accreted one more
  historical `launch.sh`). Identical counts at both roots ⇒ sibling-owned debt that grew past its
  recorded bound before this landing. I did **not** raise those bounds: doing so would launder other
  arms' debt under my landing. They are named here so the queue is tracked, not silently inherited.
- **No score claim.** Nothing here touches `d_seg`, `d_pose`, or archive bytes.

## Follow-ons (owned, not orphaned)

1. **MAIN:** adjudicate daemon re-enable. Suggested first step: `--no-govern` recorder for one
   session, then governed with the ledger CLI (`--stopped-ledger`) watched.
2. **This gate → STRICT** after one clean cycle at live count 0 (condition is in its docstring).
3. **Structural alternative to D4, deferred deliberately:** move the reconcile INSIDE
   `live_admission_decision` so the asymmetry becomes impossible rather than merely detected. Not
   done here because the charter's contract was explicit ("mirror witness_memory_preflight exactly")
   and because a registry mutation inside the shared decision function needs a hermeticity guard for
   tests. The gate makes the omission un-landable in the meantime.
4. **Pre-existing bound debt above** — owner: the arms that landed those gates. Blocker: none; each
   needs its own fix-or-rebaseline decision, which is not mine to make.
