# SETUP-stuck cost-leak — FIXED (R31 + Check 56)

**Date**: 2026-04-28
**Cost class**: silent GPU spend forever, no upper bound
**Class of bug**: heuristic-based health classifier with NO timeout for the in-flight SETUP state

## The bug class

`scripts/verify_vast_instances.py` classifies Vast.ai instances into
HEALTHY / IDLE / CRASHED / UNREACHABLE / SETUP / GONE. The original
auto-destroy path only fired on IDLE/CRASHED, time-boxed by
`--stale-minutes` (heartbeat freshness).

But a TRULY hung `setup_full.sh` (deadlocked, never writes a heartbeat)
is classified **SETUP**, not IDLE. The IDLE timer compares heartbeat
freshness — with no heartbeat, the comparison never fires, and the
instance accrues GPU cost silently forever.

The R31 reviewer correctly noted that excluding SETUP from auto-destroy
**is correct** for the common 5-15 min DALI install case (avoids
false-positive destroys). But there is no upper time bound, so the
exclusion becomes a cost leak for genuinely deadlocked setups.

## The fix (dual-threshold pattern)

Two independent timers in `verify_vast_instances.py`:

1. `--stale-minutes` (default 30) — IDLE/CRASHED heartbeat freshness.
2. `--setup-stale-minutes` (default 90) — SETUP first-seen age.

State tracking lives in `.omx/state/instance_setup_first_seen.json`
(map: instance_id → unix timestamp of first SETUP observation). Each
verify pass records the first-seen-as-SETUP timestamp; once SETUP age
> `--setup-stale-minutes`, auto-destroy fires. When an instance leaves
SETUP, its entry is dropped (so the file doesn't grow unbounded; it's
also pruned for GONE instances on every pass).

90 min is well past the max DALI install + Stage 0..N (~15 min on a
good host), so anything beyond is a true deadlock.

`InstanceHealth` gained a `setup_age_minutes: float | None` field so
the JSON output surfaces the SETUP age to the operator at debug time.

## The preflight guard (Check 56)

`check_verify_vast_setup_stuck_dual_threshold` — STRICT in
`preflight_all()` at land time; 0 live violations.

Scans `scripts/verify_vast_instances.py` for:
1. CLI flag definition: `--setup-stale-minutes`
2. CLI flag definition: `--stale-minutes`
3. Auto-destroy branch references `setup_age_minutes` or
   `setup_stale_minutes` (SETUP half present)
4. Auto-destroy branch references `"IDLE"` (IDLE half present)

Any future refactor that drops EITHER half fails preflight.

10 unit tests in `src/tac/tests/test_preflight_setup_stuck_dual_threshold.py`
+ 10 runtime tests in `src/tac/tests/test_verify_vast_setup_stuck.py`
covering state-file round-trip, JSON corruption tolerance, dataclass
shape, classify() invariants, and parser flag wiring.

## Catalog count

55 STRICT preflight checks → **56 STRICT** after this commit.
