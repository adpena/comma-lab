# ddm_mb1 — memory_blackbox governor: the ACTUATOR is now default-OFF and durably armed

**Task #1073. Landed 2026-08-16. The daemon remains OFF — MAIN adjudicates the re-enable.**

## The finding, first

ddm_gb1 (`31acf05204`) already fixed the three defects the incident memo named. I verified all
three at source and did not redo them. What it did **not** fix is the actuator's **arming**: the
SIGSTOP throttle was still ON by a hardcoded default, and the auto-start path passed no opt-out.

So "the daemon is OFF until the fix lands" was true only because nobody had launched training yet.
The next `spawn_durable_daemon` launch would have silently restarted the un-adjudicated actuator
that froze three live measurements for 75+ minutes.

I found two more defects of the same family during review. All three are fixed here.

## What ddm_gb1 already fixed (verified at source, not redone)

| Incident defect | Cure | Verified at |
|---|---|---|
| D1 wrong-object cp measurement | cp leg counts only NAMED control-plane processes; throttle actuates on `governing_free_gib` | `system_memory_governor.py:1438-1448` (reclaimable-aware, reuses the cured admission basis) |
| D2 no re-arm | escape hatch (RUNG 0) + derived-free resume + no-pause veto | `system_memory_governor.py:2077-2115`; callsite `memory_blackbox.py:_govern_tick` |
| D3 no exit-resume | persisted stopped-set ledger + atexit + SIGTERM/SIGINT + `--resume-stopped` | `memory_blackbox.py:319-398` |

Baseline before my change: **60 tests passed** in the two gb1 suites.

## The three defects I fixed

**M1 — the actuator was armed by a silent default.** `run_daemon(govern: bool = True)`
(`memory_blackbox.py:639` pre-fix) and `ensure_blackbox_running` building
`[..., "memory_blackbox.py", "--daemon"]` with no opt-out (`memory_blackbox.py:766-772` pre-fix).
A score-affecting SIGSTOP actuator that re-arms itself on every launch, with no operator
adjudication and no recorded reason, is the inverse of CLAUDE.md's "'Off' is a tracked queue, never
a forgotten default". The incident's own verdict says off is the right default even post-fix: the
per-job `safe_run` RSS envelopes were the protection that actually worked.

**M2 — exit-resume was gated on `govern`.** The startup sweep of a predecessor's stranded set and
the `atexit`/signal install both sat behind `if govern:` (`memory_blackbox.py:665-670` pre-fix).
Once the actuator became default-OFF, that gating meant the **common** case — a recorder-only
daemon — would see a stranded ledger and walk past it. Both legs are SIGCONT-only, so gating them
bought nothing.

**M3 — the governor CLI stranded its own pauses.** `system_memory_governor.py --governor-tick
--apply` called `pause_job(action.target)` and recorded **nothing** (`gov.main`, pre-fix). A
CLI-initiated stop was invisible to the exit sweep, to `--resume-stopped`, and to the escape hatch
(which ages pids by their ledger timestamp). That is the incident's own D3 defect reached through
the second SIGSTOP entry point — and it strands a job exactly when no daemon is running to adopt
it. Found in review pass 2.

## Landing 1 — the fix

1. **One shared arming resolver.** `gov.resolve_arming(env_name, flag_path)` → `Arming(armed,
   source, detail)`. `admission_enforcing()` now calls it too, so there is no parallel twin that
   could diverge on truthy-parsing or precedence. Malformed or unreadable flag ⇒ NOT armed
   (fail-safe). New: `throttle_arming()` / `throttle_armed()`, env `TAC_GOV_THROTTLE_ARM`, durable
   flag `.omx/state/governor_throttle_arm.flag`.
2. **Tri-state `govern`.** `run_daemon(govern=None)` defers to the arming surface; explicit
   `True`/`False` is a per-invocation override. CLI: `--govern` forces on, `--no-govern` forces off,
   neither defers; a contradictory pair resolves to **off** (between two operator intents the safe
   one governs).
3. **The state is logged with its reason, every start.** `THROTTLE ACTUATOR ARMED|DISARMED
   [source=…] :: <detail>`. An unexplained "off" is how a disabled protection layer becomes a
   forgotten one. An explicit override reports `source=explicit`, never as a durable arming.
4. **Exit/startup sweeps are unconditional** (M2).
5. **The CLI records before it signals** (M3), mirroring the daemon's ledger discipline.
6. **Typed escape-hatch alarm.** An escape-hatch resume now emits
   `ALARM confound_alarm=throttle_escape_hatch` and carries `record["alarm"]`, instead of another
   indistinguishable RESUME line. A stop held past the ceiling means the re-arm references
   themselves failed — that is not routine. The token lives in ONE place
   (`gov.ESCAPE_HATCH_REASON_TOKEN`) so a reworded reason cannot silently kill the alarm.

### The coupling I had to repair, not paper over

`list_tracked_jobs` licenses the Layer-3 measured-growth admission relaxation on Layer 2 being able
to pause the job — the pre-existing comment says so explicitly (`system_memory_governor.py:1964-67`).
That precondition tested **structural eligibility** but not **actuator arming**. Disarming by
default would therefore have silently **widened** admission by removing the very backstop that
licensed the relaxation. The relaxation is now withdrawn wholesale while the throttle is disarmed
(`layer2_backstop_armed`, defaulting to the live arming), so disarming makes admission **stricter**,
never wider. One existing test caught this; it now states the precondition explicitly instead of
depending on host arming state.

**Known residual, stated not hidden:** "armed" means the actuator *may* run, not that a daemon is
running now. A liveness check would need `memory_blackbox`, which imports the governor — a cycle.
This is still strictly better than pre-ddm_mb1, which granted the relaxation with no throttle
condition at all. Closing it needs a registry-side liveness read; left as owed work.

## Landing 2 — self-protection

Per Catalog #299 consolidation (pure-additive gates are the slow death) this is a **third leg of the
existing gate**, not a new catalog number: `check_throttle_rearms_and_admission_reconciles`.

* **C1** — a function building the black-box daemon argv must not pass `--govern`.
* **C2** — a function taking the `govern` switch must not default it to a hardcoded `True`; a `None`
  default must actually resolve through `throttle_arming()` (an unresolved `None` is not a tracked
  default-OFF, it is an actuator nobody can arm).
* Waiver `# THROTTLE_ARM_OK:<rationale>`. WARN-ONLY per the strict-flip atomicity rule; **live count
  is 0** in this landing.
* The prefilter was widened so a file carrying *only* the Leg-C shape is still parsed — a prefilter
  that hides a leg is the "vacuity == pass" failure.
* **Two new positive controls** (`planted_actuator_spawn.py`, `planted_actuator_default.py`) — one
  per rule, so neither can be gutted silently. The class guard EXECUTES them.

## Test evidence (run, not asserted)

```
pytest test_ddm_mb1_throttle_arming.py test_ddm_gb1_memory_governor_rearm.py \
       test_memory_blackbox.py test_tier_scaled_safety_floor.py \
       test_system_memory_governor.py test_guard_bands.py \
       test_admission_reservation_toctou.py -q
→ 267 passed in 21.26s      rc=0
```

* New suite `src/tac/tests/test_ddm_mb1_throttle_arming.py`: **47 tests**.
* **POSITIVE CONTROL (real process, in-process loop):** a live child SIGSTOPs *itself*, the ledger
  records it exactly as an armed predecessor would, then a **disarmed** daemon starts and must
  SIGCONT it. Verified it leaves state `T` and the ledger drains. Nothing pre-existing is ever
  signalled; the child is killed in `finally`.
* Gate: live count **0**, non-vacuous denominator — `1 actuator-spawn path + 1 actuator-switch
  default` actually scanned in 414 of 6491 in-scope files.
* Positive-control harness: **16 controls fired, 0 violations**.
* `ruff`: **7 findings = the exact pre-existing baseline** (4 `memory_blackbox` + 1 governor + 2
  `confound_gates`, line numbers shifted only). Zero introduced.

Five `test_confound_gates.py::test_real_repo_live_count_bounded` failures are **pre-existing and not
mine** — confirmed by running those gates in a clean `HEAD` worktree, where they report the same
counts (2 / 1 / 1) and cite only files outside my changeset (other arms' in-flight work).

## What remains for MAIN

1. **The re-enable adjudication.** The daemon is **OFF**. It is not running, and
   `.omx/state/governor_throttle_arm.flag` does not exist. To arm after review:
   `echo 1 > .omx/state/governor_throttle_arm.flag` (or `TAC_GOV_THROTTLE_ARM=1` for one
   invocation). To keep it off: do nothing — that is now the code's default, not an accident of
   nobody having launched.
2. **Strict-flip of Leg C** after one clean cycle at live count 0.
3. **The known residual** above (armed ≠ running) if MAIN judges it worth the registry-side
   liveness read.
