# CHARTER — ddm_gb1: memory_blackbox governor two-landing fix (#1073, P0 APPARATUS)

**Operator bindings (2026-08-15):** "Must be hardened and polished and automated." + the
determinization program (#1047 "All of this should be automatic and not manual or ad hoc").
This charter cures the stuck-throttle incident CLASS, not the instance.

## The measured incident (all receipts on disk)

1. `tools/memory_blackbox.py --daemon` (pid 8997) SIGSTOPped every throttle-eligible python
   ~2026-08-15 16:53 local and never resumed them, on a box with 40.5 GiB free. Frozen 75+ min:
   mp2 eval (39748/40045/40055), wc1 run, wd3 W0 warm train (92077), dashboard, 3 safe_runs.
   Memory: `governor_stuck_throttle_froze_three_live_measurements_20260815.md` (frontmatter name
   `governor-stuck-throttle-froze-three-live-measurements`).
2. Daemon log receipt: `.omx/state/memory_blackbox.daemon.log` — "SAFETY-FLOOR CLAMP: measured_cp
   value 92.00 GiB clamped to 64.00 GiB" at ~1.5 Hz + "GOVERNOR ALERT avail=40.4GiB level=warn".
3. SECOND incident same day (~23:3x): the wd3 W0-warm v2 relaunch was REFUSED by the safe_run
   SUM-over-RAM admission gate with "active-growth 100.0 GiB" = 4 registry jobs × 25 GiB
   UNKNOWN_GROWTH_HEADROOM_GIB (tools/system_memory_governor.py:312), of which THREE were DEAD
   phantom rows (`.omx/state/durable_daemons.json`: pids 7506, 8997 — the dead daemon itself —
   and 31881, all status "running"). Manual `spawn_durable_daemon.reconcile_dead_daemons()`
   converged 3 rows → relaunch ADMITTED (projected 81.6 < ceiling 116.0). Refusal receipt:
   `/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/fire_rung2_w0warm_v2/launcher/refused_attempt_1_governor_phantom.log`.

## Root causes (three stacked defects + one asymmetry)

- **D1 wrong-object cp measurement:** `measured_control_plane_rss_gib` returns ~total used memory
  incl. file cache (~92 GiB), clamped to 64 — permanent WARN on a healthy machine.
- **D2 no throttle re-arm:** resume gates solely on the sticky macOS pressure level (spike-guard
  #304 genus — a guard whose reference never re-arms).
- **D3 no exit-resume:** daemon death leaves SIGSTOPped victims stopped forever.
- **D4 admission asymmetry (NEW, measured today):** `tools/witness_memory_preflight.py:389`
  auto-runs `reconcile_dead_daemons` before admission (the 2026-07-09 phantom-growth fix);
  `tools/safe_run.py`'s admission path does NOT — dead registry rows each charge 25 GiB phantom
  growth and can refuse a real launch (reproduced twice today).

## Fix contract (two-landing per CLAUDE.md; ALL FOUR)

1. **D1:** cp measurement counts ONLY named control-plane processes (enumerate by pid/cmd match),
   never vm-total or total-used. Add a unit test with a synthetic process table.
2. **D2:** throttle carries (a) a max-stop-duration escape hatch and (b) resume on the governor's
   OWN derived free-GiB thresholds — never solely the sticky OS pressure level. Positive control:
   simulate sticky-warn + high free → victims must resume.
3. **D3:** atexit + SIGTERM/SIGINT handler SIGCONTs every pid the daemon stopped (persist the
   stopped-set to disk so even SIGKILL leaves a resumable ledger a successor can sweep).
4. **D4:** safe_run's admission path calls `reconcile_dead_daemons(verbose=False)` fail-OPEN
   before the SUM-over-RAM decision (mirror witness_memory_preflight's pattern exactly; the
   admission decision itself stays fail-CLOSED).
5. **Second landing:** warn-only preflight gate refusing the D2/D4 anti-patterns (resume gated
   solely on OS pressure; admission path reading the registry without a reconcile), with an
   EXECUTED positive control per the #831 ratchet discipline.

## Constraints

- The daemon stays OFF until this lands; per-job safe_run envelopes carry OOM protection meanwhile.
- MAIN adjudicates daemon re-enable — the arm does NOT restart it.
- Control-plane kill-semantics gauntlet: never SIGSTOP/kill the control plane (#409/#172 lineage).
- Commits via tools/subagent_commit_serializer.py with post-edit working-tree shas; .py review ×2
  via tools/review_tracker.py; no REVIEW_GATE_OVERRIDE on .py.
- Detector-zeroes-on-the-cure test: with all fixes applied and nothing else wrong, every new gauge
  reads clean (structural_beats_procedural law).

## OPTIMAL FORM

- Reference form: the live `tools/memory_blackbox.py` + `tools/system_memory_governor.py` +
  `tools/safe_run.py` + `tools/spawn_durable_daemon.py` at git 0002232fc2 (main, 2026-08-15).
- Provenance pins: incident memory sha-on-disk
  `~/.claude/.../memory/governor_stuck_throttle_froze_three_live_measurements_20260815.md`;
  refusal receipt path above; `tools/system_memory_governor.py:312`
  (UNKNOWN_GROWTH_HEADROOM_GIB=25.0) + :1474 (refusal string) + :1540 (_load_registry_rows);
  `tools/witness_memory_preflight.py:382-401` (the auto-reconcile pattern to mirror); commit
  0002232fc2 = HEAD at charter time.
- Scope reductions: none (full class cure). Mechanism reductions: none — no TOY bracket.
- Every delta above is a build against the live tools; no raced family verdict is produced.

## INCIDENT #3 (appended 2026-08-16 ~00:0xZ — MEASURED, third same-day D1-family refusal)

The wd3 W0_reset governed launch (compiled config `ad6ab661c8e2f378`, READY_TO_FIRE, zero blockers)
was REFUSED by the safe_run SUM-over-RAM gate with "active-growth 100.0 GiB" on a box using 37.9 of
128 GiB. `reconcile_dead_daemons()` found ZERO stale rows this time — the growth was charged to LIVE
processes. Decomposition (from `live_admission_decision(projected_new_gib=20.0)`; refusal receipt at
`/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/fire_rung3_w0reset/launcher/run.log`):

- pid 25204 dashboard: rss 0.22 GiB → charged 25.22 (control-plane process; #370's exemption intent
  does not reach the registry-row charge — REGRESSION candidate).
- pid 39740 mp2 safe_run wrapper: declared 6.07 (correct — declared path works).
- pid 39748 mp2 contest_auth_eval (child of 39740's tree, own session): rss 4.39 → charged 29.39.
- pid 25923 mp2 evaluate.py (grandchild, own session): rss 4.32 → charged 29.32.

**ONE logical mp2 job charged ≈64.8 GiB against ≈9 GiB real** — the `:1831`-documented session-split
conservatism triple-counts a single safe_run tree. Fix-contract ADDITION (folds into D1): active-job
enumeration must AGGREGATE a launch tree (wrapper + descendants, cross-session via ppid/pgid walk or
the wrapper's child-pidfile) under the wrapper's DECLARED projection, never charge each session leader
the +25 default independently. Positive control: synthetic 3-process tree with a declared wrapper →
one charge, equal to the declaration.
